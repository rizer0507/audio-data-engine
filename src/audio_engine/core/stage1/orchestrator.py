"""Stage-1 end-to-end orchestrator: clean → dual ASR → register → classify → export."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Callable

import yaml

from audio_engine.core.catalog import utc_now
from audio_engine.core.stage1.adapters import get_adapter
from audio_engine.core.stage1.cache_policy import FAMILY_RUN_ALIASES, REQUIRED_FAMILIES
from audio_engine.core.stage1.config_gen import (
    freeze_job_configs,
    write_dataset_with_runs,
    write_sensevoice_pipeline_override,
)
from audio_engine.core.stage1.digests import write_json
from audio_engine.core.stage1.gpu_inventory import bind_authorized_gpus
from audio_engine.core.stage1.gpu_lease import GpuLeaseError
from audio_engine.core.stage1.gpu_scheduler import DualGpuScheduler
from audio_engine.core.stage1.job import (
    ATTACH_PIPELINE,
    CLASSIFY_PIPELINE,
    CLEAN_PIPELINE,
    PREPARE_PIPELINE,
    SELECTION_RULE,
    Stage1JobRequest,
    Stage1JobState,
    StageState,
    append_event,
    load_job_state,
    save_job_state,
)
from audio_engine.core.stage1.locks import JobLock, JobLockError
from audio_engine.core.stage1.process import ServiceSession
from audio_engine.core.stage1.reconcile import reconcile_delivery
from audio_engine.core.stage1.retry_policy import (
    DEFAULT_MAX_ATTEMPTS,
    attempt_record,
    backoff_seconds,
    classify_error,
    should_trip_circuit,
)
from audio_engine.core.stage1.runtime_config import (
    MissingConfigError,
    Stage1RuntimeConfig,
    load_runtime_config,
)
from audio_engine.core.source_naming import STAGE1_ASR_DIR, STAGE1_CLEANED_DIR, STAGE1_DERIVED_DIR


CommandRunner = Callable[[list[str], dict[str, str], Path], int]


def _default_runner(cmd: list[str], env: dict[str, str], log_dir: Path) -> int:
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / "stdout.log"
    stderr_path = log_dir / "stderr.log"
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        completed = subprocess.run(cmd, env=env, stdout=stdout, stderr=stderr, check=False)
    return int(completed.returncode)


def audio_data_argv(*args: str) -> list[str]:
    return [sys.executable, "-m", "audio_engine.cli.main", *args]


_DOWNSTREAM_STAGES = (
    "write_dataset_config",
    "prepare",
    "attach",
    "classify",
    "export",
    "reconcile",
)


class Stage1Orchestrator:
    def __init__(
        self,
        job_root: Path,
        *,
        runner: CommandRunner | None = None,
        dry_run: bool = False,
        sleep_fn: Callable[[float], None] | None = None,
    ) -> None:
        self.job_root = Path(job_root)
        self.state = load_job_state(self.job_root)
        self.request = Stage1JobRequest(**self.state.request)
        self.runtime = load_runtime_config(self.request.runtime_config)
        self.runner = runner or _default_runner
        self.dry_run = dry_run
        self.sleep_fn = sleep_fn or time.sleep
        self.sessions: dict[str, ServiceSession] = {}
        self.paths = self._product_paths()
        self._lock: JobLock | None = None
        self._worker_token: str | None = None
        self._force_stages: set[str] = set()
        self._export_only: bool = False
        self._state_lock = threading.RLock()
        self._family_gpu: dict[str, str] = {}
        self._scheduler: DualGpuScheduler | None = None

    def _product_paths(self) -> dict[str, Path]:
        batch = self.request.batch
        asr = {
            alias: STAGE1_ASR_DIR / f"{alias}_asr_{batch}.parquet"
            for family in REQUIRED_FAMILIES
            for alias in FAMILY_RUN_ALIASES[family]
        }
        return {
            "cleaned": STAGE1_CLEANED_DIR / f"cleaned_{batch}.parquet",
            "prepared": STAGE1_DERIVED_DIR / f"prepared_v3_{batch}.parquet",
            "prepared_asr": STAGE1_DERIVED_DIR / f"prepared_asr_v3_{batch}.parquet",
            "classified": STAGE1_DERIVED_DIR
            / f"classified_five_class_v2_2_auto_noise_{batch}.parquet",
            "export": Path("data/exports")
            / f"summary_five_class_v2_2_auto_noise_{batch}.xlsx",
            "dataset": self.job_root / "dataset.yaml",
            "selection": self.job_root / "selection.yaml",
            **{f"asr_{k}": v for k, v in asr.items()},
            **{
                f"identity_{alias}": self.job_root / "run_identities" / f"{alias}_identity.yaml"
                for alias in asr
            },
            **{
                f"registered_{alias}": self.job_root
                / "run_identities"
                / f"{alias}_registered.yaml"
                for alias in asr
            },
        }

    def _env(self, overlays: dict[str, str] | None = None) -> dict[str, str]:
        env = dict(os.environ)
        if overlays:
            env.update(overlays)
        env["AUDIO_DATA_STAGE1_JOB_ID"] = self.state.job_id
        return env

    def _checkpoint_path(self, name: str) -> Path:
        return self.job_root / "checkpoints" / f"{name}.json"

    def _expected_outputs(self, name: str) -> list[Path]:
        if name == "clean":
            return [self.paths["cleaned"]]
        if name == "freeze_snapshot":
            return [self.job_root / "config_snapshot"]
        if name.startswith("asr_"):
            alias = name[len("asr_") :]
            return [self.paths[f"asr_{alias}"]]
        if name.startswith("register_"):
            alias = name[len("register_") :]
            return [self.paths[f"registered_{alias}"]]
        if name.startswith("serve_start_"):
            family = name[len("serve_start_") :]
            return [self.job_root / "serve" / family / "session.json"]
        if name == "write_dataset_config":
            return [self.paths["dataset"]]
        if name == "prepare":
            return [self.paths["prepared"]]
        if name == "attach":
            return [self.paths["prepared_asr"]]
        if name == "classify":
            return [self.paths["classified"]]
        if name == "export":
            return [self.paths["export"]]
        stage = self.state.stages.get(name)
        if stage and stage.outputs:
            return [Path(p) for p in stage.outputs]
        return []

    def _stage_outputs_valid(
        self, name: str, outputs: list[Path] | None = None
    ) -> bool:
        stage = self.state.stages.get(name)
        if stage is None or stage.status not in {"succeeded", "skipped"}:
            return False
        if self.dry_run and (stage.detail or {}).get("dry_run"):
            return True
        paths = outputs if outputs is not None else self._expected_outputs(name)
        if not paths:
            return True
        for path in paths:
            if path.is_dir():
                if not any(path.iterdir()):
                    return False
                continue
            if not path.is_file():
                # export may be multi-part: stem-part-001.xlsx
                if name == "export":
                    stem = path.with_suffix("")
                    parts = list(path.parent.glob(f"{stem.name}-part-*.xlsx"))
                    if parts:
                        continue
                return False
            if path.stat().st_size <= 0:
                return False
            if path.suffix.lower() in {".yaml", ".yml"}:
                try:
                    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
                    if raw is None:
                        return False
                except Exception:
                    return False
            if path.suffix.lower() == ".json":
                try:
                    import json

                    json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    return False
        return True

    def _write_checkpoint(self, name: str, **payload: Any) -> None:
        path = self._checkpoint_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "stage": name,
            "at": utc_now(),
            "worker_token": self._worker_token,
            "status": self.state.stages.get(name, StageState()).status,
            **payload,
        }
        write_json(path, data)
        stage = self.state.stages.setdefault(name, StageState())
        stage.checkpoint = {"path": str(path), **payload}
        save_job_state(self.job_root, self.state)

    def _heal_corrupt_successes(self) -> None:
        healed: list[str] = []
        for name, stage in list(self.state.stages.items()):
            if stage.status not in {"succeeded", "skipped"}:
                continue
            if self._stage_outputs_valid(name):
                continue
            stage.status = "pending"
            stage.error = "healed: missing or corrupt outputs"
            stage.finished_at = None
            stage.outputs = []
            stage.circuit_open = False
            ckpt = self._checkpoint_path(name)
            ckpt.unlink(missing_ok=True)
            healed.append(name)
        if healed:
            append_event(self.job_root, "heal_corrupt_successes", stages=healed)
            save_job_state(self.job_root, self.state)

    def _reset_stage(self, name: str) -> None:
        self.state.stages[name] = StageState(max_attempts=DEFAULT_MAX_ATTEMPTS)
        self._checkpoint_path(name).unlink(missing_ok=True)
        self._force_stages.add(name)

    def _acquire_lock(self, mode: str) -> None:
        lock = JobLock(self.job_root, job_id=self.state.job_id)
        info = lock.acquire(mode=mode, steal_stale=True)
        self._lock = lock
        self._worker_token = info.worker_token
        self.state.worker_token = info.worker_token
        self.state.pid = os.getpid()
        save_job_state(self.job_root, self.state)
        append_event(
            self.job_root,
            "lock_acquired",
            mode=mode,
            worker_token=info.worker_token,
            pid=info.pid,
        )

    def _release_lock(self) -> None:
        if self._lock is None:
            return
        token = self._worker_token
        try:
            self._lock.release(token=token)
            append_event(self.job_root, "lock_released", worker_token=token)
        except Exception:
            pass
        self._lock = None

    def _mark(
        self,
        name: str,
        *,
        status: str,
        error: str | None = None,
        outputs: list[str] | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        with self._state_lock:
            if self._lock is not None and self._worker_token is not None:
                self._lock.assert_owner(self._worker_token)
            stage = self.state.stages.setdefault(name, StageState())
            if status == "running":
                stage.started_at = utc_now()
                stage.error = None
            if status in {"succeeded", "failed", "skipped", "blocked"}:
                stage.finished_at = utc_now()
            stage.status = status
            stage.error = error
            if outputs is not None:
                stage.outputs = outputs
            if detail is not None:
                stage.detail = detail
            save_job_state(self.job_root, self.state)
            append_event(self.job_root, "stage", stage=name, status=status, error=error)

    def _consecutive_failures(self, stage: StageState) -> int:
        count = 0
        for record in reversed(stage.attempts):
            if record.get("status") == "failed":
                count += 1
            else:
                break
        return count

    def _call_with_retries(self, name: str, fn: Callable[[], None]) -> None:
        stage = self.state.stages.setdefault(name, StageState())
        if stage.max_attempts <= 0:
            stage.max_attempts = DEFAULT_MAX_ATTEMPTS

        if name not in self._force_stages and stage.status in {"succeeded", "skipped"}:
            if self._stage_outputs_valid(name):
                return

        if stage.circuit_open and name not in self._force_stages:
            raise RuntimeError(f"stage {name} circuit_open；请显式 retry 清除熔断后重跑")

        max_attempts = stage.max_attempts or DEFAULT_MAX_ATTEMPTS

        while True:
            stage.attempt_count += 1
            attempt_no = stage.attempt_count
            self._mark(name, status="running")
            try:
                fn()
                final_status = self.state.stages[name].status
                # Bodies may soft-fail (e.g. reconcile gate) without raising.
                if final_status == "failed":
                    stage.attempts.append(
                        attempt_record(
                            attempt=attempt_no,
                            status="failed",
                            error=self.state.stages[name].error,
                            error_kind=stage.error_kind,
                            worker_token=self._worker_token,
                        )
                    )
                    save_job_state(self.job_root, self.state)
                    self._force_stages.discard(name)
                    return
                if final_status not in {"succeeded", "skipped"}:
                    self._mark(name, status="succeeded")
                stage.attempts.append(
                    attempt_record(
                        attempt=attempt_no,
                        status="succeeded",
                        worker_token=self._worker_token,
                    )
                )
                stage.error_kind = None
                save_job_state(self.job_root, self.state)
                self._force_stages.discard(name)
                return
            except Exception as exc:  # noqa: BLE001
                ec = classify_error(exc)
                stage.error_kind = ec.kind
                stage.attempts.append(
                    attempt_record(
                        attempt=attempt_no,
                        status="failed",
                        error=str(exc),
                        error_kind=ec.kind,
                        worker_token=self._worker_token,
                    )
                )
                save_job_state(self.job_root, self.state)
                append_event(
                    self.job_root,
                    "stage_attempt_failed",
                    stage=name,
                    attempt=attempt_no,
                    error=str(exc),
                    error_kind=ec.kind,
                )

                # dry_run: do not sleep-retry; bodies are expected to succeed.
                can_retry = (
                    (not self.dry_run)
                    and ec.retryable
                    and attempt_no < max_attempts
                )
                if can_retry:
                    delay = backoff_seconds(attempt_no - 1)
                    append_event(
                        self.job_root,
                        "stage_backoff",
                        stage=name,
                        attempt=attempt_no,
                        sleep_s=delay,
                    )
                    self.sleep_fn(delay)
                    continue

                consecutive = self._consecutive_failures(stage)
                if should_trip_circuit(consecutive):
                    stage.circuit_open = True
                self._mark(
                    name,
                    status="failed",
                    error=str(exc),
                    detail={"error_kind": ec.kind, "circuit_open": stage.circuit_open},
                )
                raise

    def _run_cli(
        self, stage: str, args: list[str], *, env: dict[str, str] | None = None
    ) -> None:
        cmd = audio_data_argv(*args)
        attempt_no = self.state.stages.get(stage, StageState()).attempt_count or 1
        log_dir = self.job_root / "stages" / stage / f"attempt_{attempt_no}"
        write_json(
            log_dir / "command.json",
            {
                "argv": cmd,
                "env": env or {},
                "attempt": attempt_no,
                "worker_token": self._worker_token,
            },
        )
        if self.dry_run:
            # Bodies mark success themselves (including dry_run register synthesis).
            return
        code = self.runner(cmd, self._env(env), log_dir)
        if code != 0:
            raise RuntimeError(f"stage {stage} 失败 exit={code}; 见 {log_dir}")

    def _stop_owned_sessions(self) -> None:
        for family, session in list(self.sessions.items()):
            try:
                get_adapter(family).stop(session)
            except Exception:
                pass

    def _missing_asr_routes(self) -> list[str]:
        missing: list[str] = []
        for family in REQUIRED_FAMILIES:
            for alias in FAMILY_RUN_ALIASES[family]:
                asr_name = f"asr_{alias}"
                reg_name = f"register_{alias}"
                asr_ok = self._stage_outputs_valid(
                    asr_name, [self.paths[f"asr_{alias}"]]
                ) or (
                    self.dry_run
                    and self.state.stages.get(asr_name, StageState()).status
                    in {"succeeded", "skipped"}
                )
                reg_ok = self._stage_outputs_valid(
                    reg_name, [self.paths[f"registered_{alias}"]]
                ) or (
                    self.dry_run
                    and self.state.stages.get(reg_name, StageState()).status
                    in {"succeeded", "skipped"}
                )
                if not asr_ok or not reg_ok:
                    missing.append(alias)
        return missing

    def _block_downstream(self, *, reason: str) -> None:
        for name in _DOWNSTREAM_STAGES:
            stage = self.state.stages.get(name)
            if stage and stage.status in {"succeeded", "skipped"}:
                continue
            self._mark(name, status="blocked", error=reason)
        append_event(self.job_root, "downstream_blocked", reason=reason)

    def _all_asr_complete(self) -> bool:
        return not self._missing_asr_routes()

    def _only_export_failed(self) -> bool:
        if not self._all_asr_complete():
            return False
        for name in ("write_dataset_config", "prepare", "attach", "classify"):
            if not self._stage_outputs_valid(name) and not (
                self.dry_run
                and self.state.stages.get(name, StageState()).status
                in {"succeeded", "skipped"}
            ):
                return False
        export = self.state.stages.get("export", StageState())
        reconcile = self.state.stages.get("reconcile", StageState())
        export_bad = export.status in {"failed", "pending", "blocked"} or (
            export.status in {"succeeded", "skipped"}
            and not self._stage_outputs_valid("export")
            and not self.dry_run
        )
        reconcile_bad = reconcile.status in {"failed", "pending", "blocked"} or not (
            self.state.reconcile or {}
        ).get("ok")
        return export_bad or reconcile_bad

    # ------------------------------------------------------------------ public
    def run(self, mode: str = "run") -> Stage1JobState:
        self._acquire_lock(mode)
        self.state.status = "running"
        self.state.pid = os.getpid()
        self.state.error = None
        save_job_state(self.job_root, self.state)
        append_event(self.job_root, "job_started", pid=self.state.pid, mode=mode)
        try:
            self._heal_corrupt_successes()
            if self._export_only:
                return self._run_export_tail()

            self._call_with_retries("precheck", self._precheck_body)
            self._call_with_retries("freeze_snapshot", self._freeze_snapshot_body)
            self._call_with_retries("clean", self._clean_body)

            self._run_asr_phase()

            missing = self._missing_asr_routes()
            if missing:
                reason = f"必要 ASR 路次未齐: {missing}"
                self._block_downstream(reason=reason)
                self.state.status = "needs_attention"
                self.state.error = reason
                self.state.needs_attention.append(
                    {"kind": "missing_asr_routes", "routes": missing}
                )
                self.state.runtime_faults.append(
                    {"kind": "missing_asr_routes", "routes": missing, "at": utc_now()}
                )
                save_job_state(self.job_root, self.state)
                append_event(
                    self.job_root,
                    "job_needs_attention",
                    reason=reason,
                    routes=missing,
                )
                return self.state

            return self._run_downstream_and_finish()
        except JobLockError:
            raise
        except Exception as exc:  # noqa: BLE001
            if self.state.status != "needs_attention":
                self.state.status = "failed"
                self.state.error = str(exc)
            save_job_state(self.job_root, self.state)
            append_event(self.job_root, "job_failed", error=str(exc))
            self._stop_owned_sessions()
            raise
        finally:
            self._release_lock()

    def resume(self) -> Stage1JobState:
        self.state = load_job_state(self.job_root)
        self._force_stages.clear()
        self._heal_corrupt_successes()
        if self._only_export_failed():
            return self.retry_export_only()
        return self.run(mode="resume")

    def retry(
        self,
        *,
        family: str | None = None,
        run: int | str | None = None,
        failed_only: bool = True,
        export_only: bool = False,
    ) -> Stage1JobState:
        self.state = load_job_state(self.job_root)
        self._force_stages.clear()
        if export_only:
            return self.retry_export_only()

        targets = self._resolve_retry_targets(
            family=family, run=run, failed_only=failed_only
        )
        asr_touched = False
        for name in targets:
            self._reset_stage(name)
            if name.startswith("asr_") or name.startswith("register_"):
                asr_touched = True
            if name.startswith("serve_"):
                asr_touched = True

        if asr_touched:
            for name in _DOWNSTREAM_STAGES:
                self._reset_stage(name)

        save_job_state(self.job_root, self.state)
        append_event(
            self.job_root,
            "retry_requested",
            family=family,
            run=run,
            failed_only=failed_only,
            targets=sorted(targets),
        )
        return self.run(mode="retry")

    def retry_export_only(self) -> Stage1JobState:
        self.state = load_job_state(self.job_root)
        self._force_stages.clear()
        # Never re-run ASR on export-only recovery.
        for name in ("export", "reconcile"):
            self._reset_stage(name)
        save_job_state(self.job_root, self.state)
        append_event(self.job_root, "retry_export_only")
        self._export_only = True
        try:
            return self.run(mode="retry")
        finally:
            self._export_only = False

    def _stage_needs_retry(self, name: str) -> bool:
        stage = self.state.stages.get(name)
        if stage is None:
            return True
        if stage.status in {"failed", "needs_attention", "blocked", "pending"}:
            return True
        if stage.circuit_open:
            return True
        if stage.status in {"succeeded", "skipped"} and not self._stage_outputs_valid(name):
            return True
        return False

    def _resolve_retry_targets(
        self,
        *,
        family: str | None,
        run: int | str | None,
        failed_only: bool,
    ) -> list[str]:
        targets: list[str] = []
        if family is not None:
            family = family.strip().lower()
            if family not in REQUIRED_FAMILIES:
                raise ValueError(f"未知家族 {family}；支持 {REQUIRED_FAMILIES}")
            aliases = list(FAMILY_RUN_ALIASES[family])
            if run is not None:
                alias = f"{family}_{run}"
                if alias not in aliases:
                    raise ValueError(f"家族 {family} 无路次 run={run}（别名 {alias}）")
                aliases = [alias]
            route_stages = [
                f"asr_{alias}" for alias in aliases
            ] + [f"register_{alias}" for alias in aliases]
            serve_stages = [f"serve_start_{family}", f"serve_stop_{family}"]
            candidates = serve_stages + route_stages
            if failed_only:
                need_route = any(self._stage_needs_retry(name) for name in route_stages)
                for name in candidates:
                    if name in serve_stages:
                        # Restart serve only when a route for this family needs work.
                        if need_route or self._stage_needs_retry(name):
                            targets.append(name)
                    elif self._stage_needs_retry(name):
                        targets.append(name)
            else:
                targets.extend(candidates)
        elif failed_only:
            for name, stage in self.state.stages.items():
                if not self._stage_needs_retry(name):
                    continue
                if name.startswith("asr_") or name.startswith("register_"):
                    targets.append(name)
                elif name.startswith("serve_"):
                    targets.append(name)
                elif name in _DOWNSTREAM_STAGES or name in {
                    "precheck",
                    "freeze_snapshot",
                    "clean",
                }:
                    targets.append(name)
        else:
            targets = list(self.state.stages.keys())

        # Deduplicate preserving order.
        seen: set[str] = set()
        ordered: list[str] = []
        for name in targets:
            if name not in seen:
                seen.add(name)
                ordered.append(name)
        return ordered

    def _run_export_tail(self) -> Stage1JobState:
        self._call_with_retries("export", self._export_body)
        self._call_with_retries("reconcile", self._reconcile_body)
        return self._finalize_status()

    def _run_downstream_and_finish(self) -> Stage1JobState:
        self._call_with_retries("write_dataset_config", self._write_dataset_config_body)
        self._call_with_retries("prepare", self._prepare_body)
        self._call_with_retries("attach", self._attach_body)
        self._call_with_retries("classify", self._classify_body)
        self._call_with_retries("export", self._export_body)
        self._call_with_retries("reconcile", self._reconcile_body)
        return self._finalize_status()

    def _finalize_status(self) -> Stage1JobState:
        if self.state.reconcile.get("ok"):
            self.state.status = "succeeded"
            self.state.error = None
        else:
            self.state.status = "needs_attention"
            self.state.error = "对账门禁未通过；提交成功不等于批次成功"
            self.state.needs_attention.append(
                {
                    "kind": "reconcile_failed",
                    "errors": list(self.state.reconcile.get("errors") or []),
                }
            )
        save_job_state(self.job_root, self.state)
        append_event(self.job_root, "job_finished", status=self.state.status)
        return self.state

    def _run_asr_phase(self) -> None:
        """Run ASR families with dual-GPU scheduler when enabled; else serial."""
        remaining = {
            family
            for family in REQUIRED_FAMILIES
            if any(
                not self._stage_outputs_valid(f"asr_{alias}")
                or not self._stage_outputs_valid(f"register_{alias}")
                for alias in FAMILY_RUN_ALIASES[family]
            )
            or self.state.stages.get(f"serve_start_{family}", StageState()).status
            not in {"succeeded", "skipped"}
        }
        use_dual = bool(getattr(self.runtime, "dual_gpu_scheduler", True))
        if not use_dual or len(self.request.gpus) < 2 or self.dry_run:
            # dry_run stays serial for deterministic command recording; single GPU too.
            for family in REQUIRED_FAMILIES:
                self._family_gpu[family] = str(self._primary_gpu())
                self._run_family(family, gpu=self._family_gpu[family])
            return

        self._run_asr_dual_gpu(remaining or set(REQUIRED_FAMILIES))

    def _run_asr_dual_gpu(self, remaining: set[str]) -> None:
        token = self._worker_token or "no-token"
        scheduler = DualGpuScheduler.from_runtime(
            job_id=self.state.job_id,
            worker_token=token,
            gpus=list(self.request.gpus),
            runtime=self.runtime,
            allow_unknown_inventory=True,
        )
        self._scheduler = scheduler
        # Only enqueue families that still need work.
        need = {
            fam
            for fam in remaining
            if any(
                self._stage_needs_retry(f"asr_{alias}")
                or self._stage_needs_retry(f"register_{alias}")
                for alias in FAMILY_RUN_ALIASES[fam]
            )
            or self._stage_needs_retry(f"serve_start_{fam}")
        } or set(remaining)
        scheduler.enqueue_default_families(remaining=need)
        append_event(
            self.job_root,
            "dual_gpu_scheduler_start",
            gpus=list(self.request.gpus),
            pending=scheduler.pending_families(),
            binding_errors=list(scheduler.binding.errors),
            unknown=scheduler.binding.unknown,
        )
        write_json(
            self.job_root / "scheduler_plan.json",
            {
                "gpus": list(self.request.gpus),
                "binding": {
                    "index_to_uuid": dict(scheduler.binding.index_to_uuid),
                    "errors": list(scheduler.binding.errors),
                    "unknown": scheduler.binding.unknown,
                },
                "pending": scheduler.pending_families(),
            },
        )

        max_workers = min(2, len(self.request.gpus))
        futures: dict[Future, str] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            idle_rounds = 0
            while True:
                # Reap completed.
                done = [fut for fut in list(futures) if fut.done()]
                for fut in done:
                    gpu_key = futures.pop(fut)
                    try:
                        fut.result()
                    except Exception as exc:  # noqa: BLE001
                        append_event(
                            self.job_root,
                            "dual_gpu_worker_error",
                            gpu=gpu_key,
                            error=str(exc),
                        )
                    scheduler.release_gpu(gpu_key, clear_resident=True)

                decision = scheduler.try_claim(mode="run")
                if decision is not None:
                    idle_rounds = 0
                    self._family_gpu[decision.family] = decision.gpu_key
                    append_event(
                        self.job_root,
                        "gpu_claimed",
                        family=decision.family,
                        gpu=decision.gpu_key,
                        uuid=decision.gpu_uuid,
                        wait_s=decision.claim_wait_s,
                        resident_reuse=decision.resident_reuse,
                    )
                    fut = pool.submit(
                        self._run_family,
                        decision.family,
                        decision.gpu_key,
                    )
                    futures[fut] = decision.gpu_key
                    continue

                if not futures and not scheduler.pending_families():
                    break

                idle_rounds += 1
                if idle_rounds == 1 and scheduler.pending_families() and not futures:
                    # Nothing admissible yet — wait poll interval then retry.
                    append_event(
                        self.job_root,
                        "scheduler_idle",
                        pending=scheduler.pending_families(),
                        reasons=(scheduler.metrics.idle_reasons[-3:] or None),
                    )
                if futures:
                    wait(
                        list(futures),
                        timeout=scheduler.poll_interval_s,
                        return_when=FIRST_COMPLETED,
                    )
                else:
                    self.sleep_fn(scheduler.poll_interval_s)
                    # Avoid infinite spin if permanently blocked by foreign leases.
                    if idle_rounds > 60:
                        for family in list(scheduler.pending_families()):
                            fault = {
                                "kind": "scheduler_blocked",
                                "family": family,
                                "error": "长时间无法取得 GPU 租约/准入",
                                "at": utc_now(),
                            }
                            self.state.runtime_faults.append(fault)
                            self.state.needs_attention.append(fault)
                            append_event(
                                self.job_root,
                                "family_isolated_failure",
                                family=family,
                                error=fault["error"],
                            )
                        break

        metrics_path = self.job_root / "scheduler_metrics.json"
        write_json(metrics_path, scheduler.metrics.to_dict())
        append_event(
            self.job_root,
            "dual_gpu_scheduler_done",
            metrics=scheduler.metrics.to_dict(),
        )
        # Release any leftover leases for this worker.
        if self._worker_token:
            scheduler.lease_store.release_all_for_job(
                job_id=self.state.job_id, worker_token=self._worker_token
            )

    def _run_family(self, family: str, gpu: str | int | None = None) -> None:
        """Run one family with isolation: failures must not abort other families."""
        assigned = (
            str(int(gpu)) if gpu is not None and str(gpu).isdigit() else str(gpu)
            if gpu is not None
            else str(self._primary_gpu())
        )
        self._family_gpu[family] = assigned
        load_t0 = time.monotonic()
        try:
            self._call_with_retries(
                f"serve_start_{family}",
                lambda: self._serve_start_body(family, gpu=assigned),
            )
            if self._scheduler is not None:
                self._scheduler.record_load_time(family, time.monotonic() - load_t0)
            for alias in FAMILY_RUN_ALIASES[family]:
                self._call_with_retries(
                    f"asr_{alias}",
                    lambda a=alias: self._asr_body(family, a, gpu=assigned),
                )
                self._call_with_retries(
                    f"register_{alias}",
                    lambda a=alias: self._register_body(family, a),
                )
        except Exception as exc:  # noqa: BLE001
            fault = {
                "kind": "family_failure",
                "family": family,
                "gpu": assigned,
                "error": str(exc),
                "at": utc_now(),
            }
            with self._state_lock:
                self.state.runtime_faults.append(fault)
                self.state.needs_attention.append(fault)
                save_job_state(self.job_root, self.state)
            append_event(
                self.job_root,
                "family_isolated_failure",
                family=family,
                gpu=assigned,
                error=str(exc),
            )
        finally:
            try:
                self._call_with_retries(
                    f"serve_stop_{family}", lambda: self._serve_stop_body(family)
                )
            except Exception as stop_exc:  # noqa: BLE001
                append_event(
                    self.job_root,
                    "serve_stop_failed",
                    family=family,
                    error=str(stop_exc),
                )
            # Backoff / stop releases GPU resource for other families.
            if self._scheduler is not None:
                # lease release handled by caller after future completes
                pass

    # --------------------------------------------------------------- stage bodies
    def _precheck_body(self) -> None:
        name = "precheck"
        try:
            self.runtime.require_ready(deploy=True)
            for gpu in self.request.gpus:
                token: int | str = int(gpu) if str(gpu).isdigit() else gpu
                self.runtime.require_authorized_gpu(token)
            binding = bind_authorized_gpus(
                request_gpus=list(self.request.gpus),
                authorized_gpus=self.runtime.authorized_gpus,
                authorized_uuids=self.runtime.authorized_gpu_uuids,
            )
            if binding.errors and not binding.unknown:
                raise ValueError("GPU UUID 绑定失败:\n- " + "\n- ".join(binding.errors))
            write_json(
                self.job_root / "gpu_binding.json",
                {
                    "tokens": binding.tokens,
                    "index_to_uuid": dict(binding.index_to_uuid),
                    "errors": list(binding.errors),
                    "unknown": binding.unknown,
                    "note": "利用率低不能作为空闲判据",
                },
            )
            for family, model_path in self.request.models.items():
                if not Path(model_path).exists():
                    raise FileNotFoundError(f"模型路径不存在: {family}={model_path}")
            path_errors: list[str] = []
            for family in REQUIRED_FAMILIES:
                path_errors.extend(get_adapter(family).check_paths(self.runtime))
            if path_errors and not self.dry_run:
                critical = [
                    err
                    for err in path_errors
                    if "vllm_bin" in err or "engine_python" in err or "python_bin" in err
                ]
                if self.runtime.qwen.vllm_bin is None or (
                    self.runtime.qwen.vllm_bin and not self.runtime.qwen.vllm_bin.exists()
                ):
                    critical.append("families.qwen.vllm_bin 缺失或不可用")
                if (
                    not self.runtime.glm.python_bin.exists()
                    or not self.runtime.glm.vllm_bin.exists()
                ):
                    critical.append("GLM python/vllm 解释器路径不可用")
                if critical:
                    raise FileNotFoundError("启动器检查失败:\n- " + "\n- ".join(critical))
            self._mark(
                name,
                status="succeeded",
                detail={
                    "gpus": self.request.gpus,
                    "gpu_binding_unknown": binding.unknown,
                    "dual_gpu_scheduler": bool(
                        getattr(self.runtime, "dual_gpu_scheduler", True)
                    ),
                },
            )
            self._write_checkpoint(name, gpus=self.request.gpus)
        except (MissingConfigError, FileNotFoundError, ValueError):
            raise

    def _freeze_snapshot_body(self) -> None:
        name = "freeze_snapshot"
        if self._stage_outputs_valid(name, [self.job_root / "config_snapshot"]):
            self._mark(
                name,
                status="skipped",
                outputs=[str(self.job_root / "config_snapshot")],
            )
            return
        runtime = self._runtime_with_model_overrides()
        paths = freeze_job_configs(
            self.job_root,
            self.request,
            runtime,
            api_bases={
                "qwen": f"http://127.0.0.1:{runtime.qwen.port}",
                "glm": f"http://127.0.0.1:{runtime.glm.port}",
                "sensevoice": None,
            },
        )
        write_json(
            self.job_root / "config_snapshot" / "meta.json",
            {
                "selection_rule": SELECTION_RULE,
                "classify_pipeline": CLASSIFY_PIPELINE,
                "git_commit": self.state.git_commit,
                "models": self.request.models,
                "paths": {k: str(v) for k, v in paths.items()},
            },
        )
        outputs = [str(p) for p in paths.values()]
        self._mark(name, status="succeeded", outputs=outputs)
        self._write_checkpoint(name, outputs=outputs)

    def _runtime_with_model_overrides(self) -> Stage1RuntimeConfig:
        """Reload runtime but keep CLI model paths for digests via request.models."""
        return self.runtime

    def _clean_body(self) -> None:
        name = "clean"
        out = self.paths["cleaned"]
        if self._stage_outputs_valid(name, [out]):
            self._mark(name, status="skipped", outputs=[str(out)])
            return
        args = [
            "pipeline",
            "run",
            CLEAN_PIPELINE,
            "--source-name",
            self.request.batch,
        ]
        if self.request.source_kind == "directory":
            args.extend(["--source-dir", self.request.source])
        self._run_cli(name, args)
        if self.dry_run:
            self._mark(
                name,
                status="succeeded",
                outputs=[str(out)],
                detail={"dry_run": True},
            )
            self._write_checkpoint(name, outputs=[str(out)], dry_run=True)
            return
        if not out.is_file():
            raise FileNotFoundError(f"清洗产物缺失: {out}")
        self._mark(name, status="succeeded", outputs=[str(out)])
        self._write_checkpoint(name, outputs=[str(out)])

    def _primary_gpu(self) -> int | str:
        gpu = self.request.gpus[0]
        return int(gpu) if str(gpu).isdigit() else gpu

    def _gpu_for_family(self, family: str, gpu: str | int | None = None) -> int | str:
        if gpu is not None:
            return int(gpu) if str(gpu).isdigit() else gpu
        mapped = self._family_gpu.get(family)
        if mapped is not None:
            return int(mapped) if str(mapped).isdigit() else mapped
        return self._primary_gpu()

    def _serve_start_body(self, family: str, gpu: str | int | None = None) -> None:
        name = f"serve_start_{family}"
        assigned = self._gpu_for_family(family, gpu)
        session_path = self.job_root / "serve" / family / "session.json"
        if family == "sensevoice":
            if self._stage_outputs_valid(name, [session_path]) or (
                self.state.stages[name].status == "succeeded" and self.dry_run
            ):
                if session_path.is_file():
                    self.sessions[family] = ServiceSession.load(session_path)
                self._mark(name, status="skipped")
                return
            # Constrain ALL SenseVoice workers to the leased GPU only.
            write_sensevoice_pipeline_override(
                self.job_root, gpus=[str(assigned)]
            )
            session_dir = self.job_root / "serve" / family
            adapter = get_adapter(family)
            session = adapter.start(
                self.runtime,
                gpu=assigned,
                session_dir=session_dir,
                dry_run=self.dry_run,
            )
            session.client_env["SENSEVOICE_MODEL_PATH"] = self.request.models[family]
            session.client_env["CUDA_VISIBLE_DEVICES"] = str(assigned)
            session.gpu = assigned
            session.save(session_dir / "session.json")
            self.sessions[family] = session
            self._mark(
                name,
                status="succeeded",
                outputs=[str(session_path)],
                detail={
                    "client_env": session.client_env,
                    "kind": "local",
                    "gpu": str(assigned),
                    "dry_run": self.dry_run,
                },
            )
            self._write_checkpoint(name, kind="local", gpu=str(assigned))
            return

        if self._stage_outputs_valid(name, [session_path]):
            self.sessions[family] = ServiceSession.load(session_path)
            self._mark(name, status="skipped", outputs=[str(session_path)])
            return
        runtime = self._overlay_runtime_model_paths()
        session_dir = self.job_root / "serve" / family
        adapter = get_adapter(family)
        session = adapter.start(
            runtime,
            gpu=assigned,
            session_dir=session_dir,
            attach_existing=self.request.attach_existing_services,
            dry_run=self.dry_run,
        )
        self.sessions[family] = session
        self._mark(
            name,
            status="succeeded",
            outputs=[str(session_path)],
            detail={
                "api_base": session.api_base,
                "pid": session.pid,
                "owned": session.owned,
                "client_env": session.client_env,
                "gpu": str(assigned),
                "dry_run": self.dry_run,
            },
        )
        self._write_checkpoint(
            name,
            api_base=session.api_base,
            pid=session.pid,
            owned=session.owned,
            gpu=str(assigned),
        )

    def _overlay_runtime_model_paths(self) -> Stage1RuntimeConfig:
        """Write a job-local server.yaml with CLI model paths and reload."""
        raw = yaml.safe_load(Path(self.request.runtime_config).read_text(encoding="utf-8")) or {}
        families = raw.setdefault("families", {})
        for family, path in self.request.models.items():
            families.setdefault(family, {})["model_path"] = path
        if not raw.get("authorized_gpus"):
            raw["authorized_gpus"] = [
                int(g) if str(g).isdigit() else g for g in self.request.gpus
            ]
        overlay = self.job_root / "config_snapshot" / "server.job.yaml"
        overlay.parent.mkdir(parents=True, exist_ok=True)
        overlay.write_text(
            yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
        return load_runtime_config(overlay)

    def _serve_stop_body(self, family: str) -> None:
        name = f"serve_stop_{family}"
        if (
            name not in self._force_stages
            and self.state.stages[name].status == "succeeded"
        ):
            return
        session = self.sessions.get(family)
        if session is None:
            session_path = self.job_root / "serve" / family / "session.json"
            if session_path.is_file():
                session = ServiceSession.load(session_path)
        if session is not None and not self.dry_run:
            get_adapter(family).stop(session)
        self._mark(
            name,
            status="succeeded",
            detail={"owned": bool(session and session.owned), "dry_run": self.dry_run},
        )
        self._write_checkpoint(name, owned=bool(session and session.owned))

    def _asr_body(
        self, family: str, alias: str, gpu: str | int | None = None
    ) -> None:
        name = f"asr_{alias}"
        out = self.paths[f"asr_{alias}"]
        if self._stage_outputs_valid(name, [out]):
            self._mark(name, status="skipped", outputs=[str(out)])
            return
        session = self.sessions.get(family)
        env = dict(session.client_env if session else {})
        assigned = self._gpu_for_family(family, gpu)
        if family == "qwen":
            pipeline = self.runtime.qwen.pipeline
        elif family == "glm":
            pipeline = self.runtime.glm.pipeline
        else:
            pipeline = str(self.job_root / "pipelines" / "sensevoice_asr_batch.yaml")
            env["SENSEVOICE_MODEL_PATH"] = self.request.models[family]
            # All SenseVoice shard workers share the leased card only.
            env["CUDA_VISIBLE_DEVICES"] = str(assigned)
        args = [
            "pipeline",
            "run",
            pipeline,
            "--source-name",
            self.request.batch,
            "--asr-run",
            alias,
            "--runs-dir",
            str(self.job_root / "pipeline_runs" / alias),
        ]
        self._run_cli(name, args, env=env)
        if self.dry_run:
            self._mark(
                name,
                status="succeeded",
                outputs=[str(out)],
                detail={"alias": alias, "gpu": str(assigned), "dry_run": True},
            )
            self._write_checkpoint(name, alias=alias, gpu=str(assigned), dry_run=True)
            return
        if not out.is_file():
            raise FileNotFoundError(f"ASR 产物缺失: {out}")
        self._mark(
            name,
            status="succeeded",
            outputs=[str(out)],
            detail={"alias": alias, "gpu": str(assigned)},
        )
        self._write_checkpoint(name, alias=alias, gpu=str(assigned), outputs=[str(out)])

    def _register_body(self, family: str, alias: str) -> None:
        name = f"register_{alias}"
        asr = self.paths[f"asr_{alias}"]
        identity = self.paths[f"identity_{alias}"]
        registered = self.paths[f"registered_{alias}"]
        if self._stage_outputs_valid(name, [registered]):
            self._mark(name, status="skipped", outputs=[str(registered)])
            return
        args = [
            "artifact",
            "register-asr-run",
            str(asr),
            "--identity",
            str(identity),
            "--audio-base",
            str(self.paths["cleaned"]),
            "--output-identity",
            str(registered),
            "--catalog-dir",
            self.request.catalog_dir,
        ]
        if self.dry_run:
            raw = yaml.safe_load(identity.read_text(encoding="utf-8")) or {}
            raw["artifact_id"] = f"dryrun_{alias}"
            registered.parent.mkdir(parents=True, exist_ok=True)
            registered.write_text(
                yaml.safe_dump(raw, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
            self._mark(
                name,
                status="succeeded",
                outputs=[str(registered)],
                detail={"dry_run": True, "family": family},
            )
            self._write_checkpoint(name, alias=alias, dry_run=True)
            return
        self._run_cli(name, args)
        if not registered.is_file():
            raise FileNotFoundError(f"登记结果缺失: {registered}")
        self._mark(name, status="succeeded", outputs=[str(registered)])
        self._write_checkpoint(name, alias=alias, outputs=[str(registered)])

    def _write_dataset_config_body(self) -> None:
        name = "write_dataset_config"
        registered = [
            self.paths[f"registered_{alias}"]
            for family in REQUIRED_FAMILIES
            for alias in FAMILY_RUN_ALIASES[family]
        ]
        path = write_dataset_with_runs(self.job_root, registered)
        self._mark(name, status="succeeded", outputs=[str(path)])
        self._write_checkpoint(name, outputs=[str(path)])

    def _prepare_body(self) -> None:
        name = "prepare"
        out = self.paths["prepared"]
        if self._stage_outputs_valid(name, [out]):
            self._mark(name, status="skipped", outputs=[str(out)])
            return
        args = [
            "pipeline",
            "run",
            PREPARE_PIPELINE,
            "--source-name",
            self.request.batch,
            "--config",
            str(self.paths["dataset"]),
            "--force",
            "--runs-dir",
            str(self.job_root / "pipeline_runs" / "prepare"),
        ]
        self._run_cli(name, args)
        if self.dry_run:
            self._mark(
                name, status="succeeded", outputs=[str(out)], detail={"dry_run": True}
            )
            self._write_checkpoint(name, dry_run=True)
            return
        self._mark(name, status="succeeded", outputs=[str(out)])
        self._write_checkpoint(name, outputs=[str(out)])

    def _attach_body(self) -> None:
        name = "attach"
        out = self.paths["prepared_asr"]
        if self._stage_outputs_valid(name, [out]):
            self._mark(name, status="skipped", outputs=[str(out)])
            return
        args = [
            "pipeline",
            "run",
            ATTACH_PIPELINE,
            "--source-name",
            self.request.batch,
            "--config",
            str(self.paths["dataset"]),
            "--force",
            "--runs-dir",
            str(self.job_root / "pipeline_runs" / "attach"),
        ]
        self._run_cli(name, args)
        if self.dry_run:
            self._mark(
                name, status="succeeded", outputs=[str(out)], detail={"dry_run": True}
            )
            self._write_checkpoint(name, dry_run=True)
            return
        self._mark(name, status="succeeded", outputs=[str(out)])
        self._write_checkpoint(name, outputs=[str(out)])

    def _classify_body(self) -> None:
        name = "classify"
        out = self.paths["classified"]
        if self._stage_outputs_valid(name, [out]):
            self._mark(name, status="skipped", outputs=[str(out)])
            return
        args = [
            "pipeline",
            "run",
            CLASSIFY_PIPELINE,
            "--source-name",
            self.request.batch,
            "--config",
            str(self.paths["selection"]),
            "--force",
            "--runs-dir",
            str(self.job_root / "pipeline_runs" / "classify"),
        ]
        self._run_cli(name, args)
        if self.dry_run:
            self._mark(
                name, status="succeeded", outputs=[str(out)], detail={"dry_run": True}
            )
            self._write_checkpoint(name, dry_run=True)
            return
        self._mark(name, status="succeeded", outputs=[str(out)])
        self._write_checkpoint(name, outputs=[str(out)])

    def _export_body(self) -> None:
        name = "export"
        out = self.paths["export"]
        if self._stage_outputs_valid(name, [out]):
            self._mark(name, status="skipped", outputs=[str(out)])
            return
        args = [
            "review",
            "export-summary",
            str(self.paths["classified"]),
            "--output",
            str(out),
            "--max-rows",
            str(self.request.max_xlsx_rows),
            "--catalog-dir",
            self.request.catalog_dir,
        ]
        self._run_cli(name, args)
        if self.dry_run:
            self._mark(
                name, status="succeeded", outputs=[str(out)], detail={"dry_run": True}
            )
            self._write_checkpoint(name, dry_run=True)
            return
        self._mark(name, status="succeeded", outputs=[str(out)])
        self._write_checkpoint(name, outputs=[str(out)])

    def _reconcile_body(self) -> None:
        name = "reconcile"
        if self.dry_run:
            result = {
                "ok": False,
                "errors": ["dry-run 未执行真实流水线；不能标记正式成功"],
                "warnings": [],
                "stats": {"dry_run": True},
            }
            self.state.reconcile = result
            write_json(self.job_root / "reconcile.json", result)
            self._mark(name, status="succeeded", detail=result)
            self._write_checkpoint(name, dry_run=True, ok=False)
            return
        asr_paths = {
            alias: self.paths[f"asr_{alias}"]
            for family in REQUIRED_FAMILIES
            for alias in FAMILY_RUN_ALIASES[family]
        }
        registered = {
            alias: self.paths[f"registered_{alias}"]
            for family in REQUIRED_FAMILIES
            for alias in FAMILY_RUN_ALIASES[family]
        }
        report = reconcile_delivery(
            batch=self.request.batch,
            cleaned=self.paths["cleaned"],
            asr_paths=asr_paths,
            registered_identities=registered,
            classified=self.paths["classified"],
            export_xlsx=self.paths["export"],
            max_xlsx_rows=self.request.max_xlsx_rows,
        )
        self.state.reconcile = report.to_dict()
        self.state.outputs = {
            "cleaned": str(self.paths["cleaned"]),
            "classified": str(self.paths["classified"]),
            "export": str(self.paths["export"]),
            "job_dir": str(self.job_root),
            "dataset": str(self.paths["dataset"]),
            "selection": str(self.paths["selection"]),
        }
        write_json(self.job_root / "reconcile.json", report.to_dict())
        write_json(self.job_root / "outputs.json", self.state.outputs)
        if not report.ok:
            self._mark(
                name,
                status="failed",
                error="; ".join(report.errors),
                detail=report.to_dict(),
            )
            self._write_checkpoint(name, ok=False, errors=report.errors)
            return
        self._mark(name, status="succeeded", detail=report.to_dict())
        self._write_checkpoint(name, ok=True)


def plan_job_summary(request: Stage1JobRequest) -> dict[str, Any]:
    from audio_engine.core.stage1.job import default_stage_names

    request.validate()
    return {
        "job_id": request.job_id(),
        "config_digest": request.config_digest(),
        "batch": request.batch,
        "source": request.source,
        "source_kind": request.source_kind,
        "models": request.models,
        "gpus": request.gpus,
        "selection_rule": SELECTION_RULE,
        "classify_pipeline": CLASSIFY_PIPELINE,
        "stages": default_stage_names(),
        "note": "提交成功不等于批次执行成功；正式成功须对账门禁通过",
    }
