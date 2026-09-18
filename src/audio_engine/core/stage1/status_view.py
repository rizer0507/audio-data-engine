"""Unified stage-1 job status view (step 5): coverage, GPU, ETA, stall, gaps."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from audio_engine.core.stage1.cache_policy import FAMILY_RUN_ALIASES, REQUIRED_FAMILIES, all_run_aliases
from audio_engine.core.stage1.job import Stage1JobState, load_job_state
from audio_engine.core.stage1.process import pid_is_alive
from audio_engine.core.source_naming import STAGE1_ASR_DIR, STAGE1_CLEANED_DIR, STAGE1_DERIVED_DIR


# Capability flags — claim step 4 after dual-GPU lease scheduler lands.
IMPLEMENTATION_GAPS = {
    "step1_adapters": True,
    "step2_orchestrator": True,
    "step3_resume_retry": True,
    "step3_failed_only_requeue": True,
    "step4_dual_gpu_scheduler": True,
    "step4_gpu_lease": True,
    "server_e2e_accepted": False,
    "server_dual_gpu_benchmark": False,
}


@dataclass
class AggregateLogEntry:
    key: str
    count: int
    first_at: str | None
    last_at: str | None
    sample: str


@dataclass
class StatusView:
    job_id: str
    batch: str
    status: str
    display_status: str
    batch_succeeded: bool
    progress_pct: float | None
    stages_done: int
    stages_total: int
    active_stages: list[str]
    parallel_stages: list[str]
    family_coverage: dict[str, Any]
    asr_coverage_pct: float | None
    retry_pending: int | None
    needs_attention: int | None
    blocked_stages: list[str]
    throughput: dict[str, Any]
    eta: dict[str, Any]
    gpus: dict[str, Any]
    stall: dict[str, Any]
    worker: dict[str, Any]
    reconcile: dict[str, Any]
    outputs: dict[str, Any]
    aggregated_logs: list[dict[str, Any]]
    last_progress: str | None
    block_reason: str | None
    gaps: dict[str, Any]
    updated_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso(value: str | None) -> float | None:
    if not value:
        return None
    try:
        text = value.replace("Z", "+00:00")
        return datetime.fromisoformat(text).timestamp()
    except ValueError:
        return None


def _parquet_rows(path: Path) -> int | None:
    if not path.is_file():
        return None
    try:
        import pyarrow.parquet as pq

        return int(pq.ParquetFile(path).metadata.num_rows)
    except Exception:
        try:
            from audio_engine.core.manifest import Manifest

            return len(Manifest.load(path))
        except Exception:
            return None


def _read_progress_json(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "progress.json"
    if not path.is_file():
        # sharded layout may nest
        for candidate in run_dir.rglob("progress.json"):
            path = candidate
            break
        else:
            return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _latest_pipeline_progress(job_root: Path) -> dict[str, Any] | None:
    root = job_root / "pipeline_runs"
    if not root.is_dir():
        return None
    best: dict[str, Any] | None = None
    best_mtime = -1.0
    for progress_path in root.rglob("progress.json"):
        try:
            mtime = progress_path.stat().st_mtime
            data = json.loads(progress_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if mtime > best_mtime:
            best_mtime = mtime
            best = {**data, "_path": str(progress_path), "_mtime": mtime}
    return best


def _stage_success(state: Stage1JobState, name: str) -> bool:
    stage = state.stages.get(name)
    return bool(stage and stage.status in {"succeeded", "skipped"})


def _family_coverage(state: Stage1JobState, batch: str) -> dict[str, Any]:
    cleaned = STAGE1_CLEANED_DIR / f"cleaned_{batch}.parquet"
    cleaned_n = _parquet_rows(cleaned)
    out: dict[str, Any] = {}
    for family in REQUIRED_FAMILIES:
        runs = []
        success_runs = 0
        for alias in FAMILY_RUN_ALIASES[family]:
            asr_stage = f"asr_{alias}"
            reg_stage = f"register_{alias}"
            asr_path = STAGE1_ASR_DIR / f"{alias}_asr_{batch}.parquet"
            rows = _parquet_rows(asr_path)
            stage_ok = _stage_success(state, asr_stage) and _stage_success(state, reg_stage)
            # Real coverage: parquet exists with rows; do not count retries twice.
            ok = bool(asr_path.is_file() and (rows or 0) > 0 and stage_ok)
            if ok:
                success_runs += 1
            runs.append(
                {
                    "alias": alias,
                    "stage_status": (state.stages.get(asr_stage).status if state.stages.get(asr_stage) else "missing"),
                    "register_status": (
                        state.stages.get(reg_stage).status if state.stages.get(reg_stage) else "missing"
                    ),
                    "rows": rows,
                    "expected_rows": cleaned_n,
                    "ok": ok,
                    "coverage": (
                        None
                        if cleaned_n in (None, 0) or rows is None
                        else round(100.0 * min(rows, cleaned_n) / cleaned_n, 2)
                    ),
                }
            )
        out[family] = {
            "success_runs": success_runs,
            "expected_runs": 2,
            "runs": runs,
            "complete": success_runs == 2,
        }
    return out


def _asr_coverage_pct(family_coverage: dict[str, Any]) -> float | None:
    total = 0
    ok = 0
    for family in REQUIRED_FAMILIES:
        info = family_coverage.get(family) or {}
        total += int(info.get("expected_runs") or 2)
        ok += int(info.get("success_runs") or 0)
    if total <= 0:
        return None
    return round(100.0 * ok / total, 2)


def aggregate_events(
    events_path: Path,
    *,
    summary_window_s: float = 60.0,
) -> list[AggregateLogEntry]:
    """Deduplicate similar events; collapse repeats within the window."""
    if not events_path.is_file():
        return []
    buckets: dict[str, AggregateLogEntry] = {}
    order: list[str] = []
    for line in events_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            key = f"raw:{line[:80]}"
            raw = {"at": None, "event": "raw", "message": line[:200]}
        event = str(raw.get("event") or "event")
        stage = str(raw.get("stage") or "")
        status = str(raw.get("status") or "")
        error = str(raw.get("error") or raw.get("message") or "")
        # Normalize transient digits in errors for aggregation.
        norm_error = re.sub(r"\d+", "N", error)[:120]
        key = f"{event}|{stage}|{status}|{norm_error}"
        at = raw.get("at")
        if key not in buckets:
            buckets[key] = AggregateLogEntry(
                key=key,
                count=1,
                first_at=at if isinstance(at, str) else None,
                last_at=at if isinstance(at, str) else None,
                sample=error or f"{event} {stage} {status}".strip(),
            )
            order.append(key)
        else:
            entry = buckets[key]
            entry.count += 1
            if isinstance(at, str):
                entry.last_at = at
    # Keep recent / high-count entries; drop pure noise of count=1 stage running if many.
    result = [buckets[k] for k in order]
    return result[-50:]


def detect_stall(
    state: Stage1JobState,
    job_root: Path,
    *,
    stall_timeout_s: float = 300.0,
) -> dict[str, Any]:
    """Process alive but business progress stalled."""
    worker_alive = bool(state.pid and pid_is_alive(int(state.pid)))
    if state.status not in {"running", "pending"}:
        return {
            "stalled": False,
            "worker_alive": worker_alive,
            "reason": None,
            "idle_s": 0.0,
        }
    timestamps: list[float] = []
    for stage in state.stages.values():
        for value in (stage.started_at, stage.finished_at):
            ts = _parse_iso(value)
            if ts is not None:
                timestamps.append(ts)
    updated = _parse_iso(state.updated_at)
    if updated is not None:
        timestamps.append(updated)
    events = job_root / "events.jsonl"
    if events.is_file():
        timestamps.append(events.stat().st_mtime)
    progress = _latest_pipeline_progress(job_root)
    if progress and progress.get("_mtime"):
        timestamps.append(float(progress["_mtime"]))
    last = max(timestamps) if timestamps else updated
    now = time.time()
    idle = (now - last) if last is not None else None
    stalled = bool(
        worker_alive
        and idle is not None
        and idle >= stall_timeout_s
        and state.status == "running"
    )
    reason = None
    if stalled:
        reason = f"worker 存活但 {idle:.0f}s 无阶段/进度更新（阈值 {stall_timeout_s:.0f}s）"
    elif state.status == "running" and state.pid and not worker_alive:
        reason = f"状态为 running 但 worker pid={state.pid} 已退出"
    return {
        "stalled": stalled,
        "worker_alive": worker_alive,
        "reason": reason,
        "idle_s": None if idle is None else round(idle, 1),
        "stall_timeout_s": stall_timeout_s,
    }


def query_authorized_gpus(gpu_ids: list[str]) -> dict[str, Any]:
    """Best-effort nvidia-smi for authorized cards only. Never invent occupancy."""
    if not gpu_ids:
        return {
            "available": False,
            "scheduler": "not_step4",
            "note": "未配置授权 GPU",
            "devices": [],
        }
    if shutil.which("nvidia-smi") is None:
        return {
            "available": False,
            "scheduler": "not_step4",
            "note": "本机无 nvidia-smi；GPU 实况待服务器采集",
            "requested": gpu_ids,
            "devices": [],
        }
    try:
        query = (
            "index,uuid,memory.used,memory.total,utilization.gpu,utilization.memory"
        )
        completed = subprocess.run(
            [
                "nvidia-smi",
                f"--id={','.join(str(g) for g in gpu_ids)}",
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "available": False,
            "scheduler": "not_step4",
            "note": f"nvidia-smi 采集失败: {exc}",
            "requested": gpu_ids,
            "devices": [],
        }
    if completed.returncode != 0:
        return {
            "available": False,
            "scheduler": "not_step4",
            "note": (completed.stderr or completed.stdout or "nvidia-smi failed").strip(),
            "requested": gpu_ids,
            "devices": [],
        }
    devices = []
    for line in completed.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        mem_used = float(parts[2]) if parts[2].replace(".", "", 1).isdigit() else None
        mem_total = float(parts[3]) if parts[3].replace(".", "", 1).isdigit() else None
        util = float(parts[4]) if parts[4].replace(".", "", 1).isdigit() else None
        # Step 4 rule: util==0 must NOT imply idle/loadable.
        devices.append(
            {
                "index": parts[0],
                "uuid": parts[1],
                "memory_used_mib": mem_used,
                "memory_total_mib": mem_total,
                "utilization_gpu": util,
                "utilization_memory": float(parts[5])
                if parts[5].replace(".", "", 1).isdigit()
                else None,
                "idle_by_util_forbidden": True,
                "note": "利用率低不能作为空闲判据（第四步）",
            }
        )
    return {
        "available": True,
        "scheduler": "dual_gpu_lease",
        "note": "util≠空闲；准入看 UUID/显存/租约/进程归属。吞吐基准须服务器实测",
        "requested": gpu_ids,
        "devices": devices,
    }


def _throughput_and_eta(
    state: Stage1JobState,
    job_root: Path,
    *,
    stages_done: int,
    stages_total: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    progress = _latest_pipeline_progress(job_root)
    rate = None
    done = None
    total = None
    source = None
    if progress:
        rate = progress.get("rate_overall") or progress.get("rate_window")
        done = progress.get("done")
        total = progress.get("total")
        source = progress.get("_path")
    created = _parse_iso(state.created_at)
    elapsed = (time.time() - created) if created else None
    throughput = {
        "samples_per_s": rate,
        "pipeline_done": done,
        "pipeline_total": total,
        "elapsed_s": None if elapsed is None else round(elapsed, 1),
        "source": source,
    }
    eta: dict[str, Any] = {"seconds": None, "display": "未知", "basis": None}
    # Prefer sample-level ETA from active pipeline progress.
    if (
        isinstance(rate, (int, float))
        and rate > 0
        and isinstance(done, (int, float))
        and isinstance(total, (int, float))
        and total > done
    ):
        seconds = (total - done) / float(rate)
        eta = {
            "seconds": round(seconds, 1),
            "display": _format_duration(seconds),
            "basis": "pipeline_sample_rate",
        }
    elif (
        elapsed
        and elapsed > 30
        and stages_done > 0
        and stages_total > stages_done
        and state.status == "running"
    ):
        # Weak stage-level ETA — mark as low confidence.
        per_stage = elapsed / stages_done
        seconds = per_stage * (stages_total - stages_done)
        eta = {
            "seconds": round(seconds, 1),
            "display": _format_duration(seconds),
            "basis": "stage_average_low_confidence",
        }
    return throughput, eta


def _format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0 or seconds != seconds:
        return "未知"
    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def build_status_view(
    job_root: Path,
    *,
    stall_timeout_s: float = 300.0,
) -> StatusView:
    job_root = Path(job_root)
    state = load_job_state(job_root)
    request = state.request or {}
    batch = state.batch
    stages = state.stages or {}
    stages_total = len(stages) or 1
    stages_done = sum(1 for s in stages.values() if s.status in {"succeeded", "skipped"})
    active = [name for name, s in stages.items() if s.status == "running"]
    failed = [name for name, s in stages.items() if s.status == "failed"]
    # Parallel display: currently serial orchestrator; still list concurrent-looking actives.
    parallel = list(active)

    family_coverage = _family_coverage(state, batch)
    asr_pct = _asr_coverage_pct(family_coverage)
    reconcile_ok = bool(state.reconcile.get("ok")) if state.reconcile else False
    batch_succeeded = state.status == "succeeded" and reconcile_ok

    # Progress: never show 100% before reconcile ok.
    if batch_succeeded:
        progress_pct = 100.0
        display_status = "succeeded"
    elif state.status == "succeeded" and not reconcile_ok:
        display_status = "failed"
        progress_pct = min(99.0, round(100.0 * stages_done / stages_total, 2))
    elif state.status == "failed":
        display_status = "failed"
        progress_pct = min(99.0, round(100.0 * stages_done / stages_total, 2))
    else:
        display_status = state.status
        # Blend stage progress with ASR coverage; cap below 100.
        stage_pct = 100.0 * stages_done / stages_total
        blend = stage_pct if asr_pct is None else (0.5 * stage_pct + 0.5 * asr_pct)
        progress_pct = min(99.0, round(blend, 2))

    # Retry / attention queues (step 3).
    retry_pending = 0
    for name, stage in stages.items():
        if stage.status == "failed" and not stage.circuit_open:
            remaining = max(0, (stage.max_attempts or 3) - (stage.attempt_count or 0))
            if remaining > 0:
                retry_pending += 1
        elif stage.circuit_open:
            retry_pending += 0  # circuit-open waits explicit retry, not auto queue
    needs_attention = len(state.needs_attention or [])
    if failed and needs_attention == 0:
        needs_attention = len(failed)
    blocked = []
    for name, stage in stages.items():
        if stage.status == "blocked":
            blocked.append(name)
        elif stage.status == "pending":
            if failed and name not in failed:
                if name in {
                    "write_dataset_config",
                    "prepare",
                    "attach",
                    "classify",
                    "export",
                    "reconcile",
                } and asr_pct is not None and asr_pct < 100:
                    blocked.append(name)

    throughput, eta = _throughput_and_eta(
        state, job_root, stages_done=stages_done, stages_total=stages_total
    )
    gpus = query_authorized_gpus([str(g) for g in (request.get("gpus") or [])])
    stall = detect_stall(state, job_root, stall_timeout_s=stall_timeout_s)
    logs = [asdict(item) for item in aggregate_events(job_root / "events.jsonl")]

    block_reason = stall.get("reason")
    if not block_reason and state.error:
        block_reason = state.error
    if not block_reason and failed:
        block_reason = f"失败阶段: {', '.join(failed[:5])}"
    if not block_reason and blocked:
        block_reason = "下游被缺路阻断；可用 stage1 retry --failed-only 补齐"

    last_progress = None
    if active:
        last_progress = f"active={','.join(active)}"
    elif state.updated_at:
        last_progress = f"updated_at={state.updated_at}"

    classified = STAGE1_DERIVED_DIR / f"classified_five_class_v2_2_auto_noise_{batch}.parquet"
    gaps = {
        "implementation": IMPLEMENTATION_GAPS,
        "notes": [
            "第三步 resume/retry/failed-only 已实现；服务器重启现场恢复仍待验收",
            "第四步双卡租约调度已实现；串行/双卡吞吐基准须服务器实测，不得编造",
            "服务器整批一命令闭环 / 真实双跑 / 双卡接续性能仍待验收",
        ],
    }

    return StatusView(
        job_id=state.job_id,
        batch=batch,
        status=state.status,
        display_status=display_status,
        batch_succeeded=batch_succeeded,
        progress_pct=progress_pct,
        stages_done=stages_done,
        stages_total=stages_total,
        active_stages=active,
        parallel_stages=parallel,
        family_coverage=family_coverage,
        asr_coverage_pct=asr_pct,
        retry_pending=retry_pending,
        needs_attention=needs_attention,
        blocked_stages=blocked,
        throughput=throughput,
        eta=eta,
        gpus=gpus,
        stall=stall,
        worker={
            "pid": state.pid,
            "alive": bool(state.pid and pid_is_alive(int(state.pid))),
        },
        reconcile=dict(state.reconcile or {}),
        outputs=dict(state.outputs or {}),
        aggregated_logs=logs[-20:],
        last_progress=last_progress,
        block_reason=block_reason,
        gaps=gaps,
        updated_at=_utc_now_iso(),
    )


def format_status_text(view: StatusView) -> str:
    lines: list[str] = []
    lines.append(
        f"job {view.job_id}  display={view.display_status}  "
        f"raw={view.status}  batch_succeeded={view.batch_succeeded}"
    )
    pct = "n/a" if view.progress_pct is None else f"{view.progress_pct:.1f}%"
    lines.append(
        f"progress {pct}  stages {view.stages_done}/{view.stages_total}  "
        f"asr_coverage={view.asr_coverage_pct if view.asr_coverage_pct is not None else 'n/a'}%"
    )
    if view.active_stages or view.parallel_stages:
        lines.append(
            f"active={','.join(view.active_stages) or '-'}  "
            f"parallel={','.join(view.parallel_stages) or '-'}"
        )
    for family, info in view.family_coverage.items():
        runs = info.get("runs") or []
        detail = ", ".join(
            f"{r['alias']}:{'ok' if r['ok'] else r['stage_status']}"
            f"({r['rows'] if r['rows'] is not None else '?'}/{r['expected_rows'] if r['expected_rows'] is not None else '?'})"
            for r in runs
        )
        lines.append(
            f"family {family}  {info.get('success_runs')}/{info.get('expected_runs')}  {detail}"
        )
    retry = "n/a" if view.retry_pending is None else str(view.retry_pending)
    attention = "n/a" if view.needs_attention is None else str(view.needs_attention)
    lines.append(f"retry_pending={retry}  needs_attention={attention}")
    if view.blocked_stages:
        lines.append(f"blocked={','.join(view.blocked_stages)}")
    thr = view.throughput
    lines.append(
        f"throughput samples/s={thr.get('samples_per_s')}  "
        f"pipeline={thr.get('pipeline_done')}/{thr.get('pipeline_total')}  "
        f"elapsed={thr.get('elapsed_s')}s"
    )
    lines.append(f"eta={view.eta.get('display')}  basis={view.eta.get('basis')}")
    gpu_note = view.gpus.get("note") or ""
    lines.append(f"gpus scheduler={view.gpus.get('scheduler')}  {gpu_note}")
    for dev in view.gpus.get("devices") or []:
        lines.append(
            f"  gpu{dev.get('index')} mem={dev.get('memory_used_mib')}/"
            f"{dev.get('memory_total_mib')}MiB util={dev.get('utilization_gpu')}%"
        )
    if view.stall.get("stalled") or view.stall.get("reason"):
        lines.append(
            f"stall stalled={view.stall.get('stalled')}  "
            f"idle_s={view.stall.get('idle_s')}  reason={view.stall.get('reason')}"
        )
    if view.block_reason:
        lines.append(f"block: {view.block_reason}")
    if view.last_progress:
        lines.append(f"last: {view.last_progress}")
    # Aggregated log summary (low redundancy)
    hot = [e for e in view.aggregated_logs if e.get("count", 0) > 1][-5:]
    for entry in hot:
        lines.append(
            f"log×{entry['count']} {entry.get('sample')}"
        )
    impl = view.gaps.get("implementation") or {}
    lines.append(
        "gaps: "
        f"step3_resume_retry={impl.get('step3_resume_retry')} "
        f"step4_dual_gpu_scheduler={impl.get('step4_dual_gpu_scheduler')} "
        f"server_e2e_accepted={impl.get('server_e2e_accepted')}"
    )
    if not view.batch_succeeded:
        lines.append("note: 对账通过前不显示完成；提交成功≠批次成功")
    return "\n".join(lines)


def wait_exit_code(view: StatusView) -> int | None:
    """Return process exit code if terminal; None if still running.

    0 = formal success (reconcile ok)
    1 = failed / reconcile not ok
    3 = needs_attention (reserved for step3)
    """
    if view.batch_succeeded:
        return 0
    if view.display_status == "failed" or view.status == "failed":
        return 1
    if view.status == "succeeded" and not view.batch_succeeded:
        return 1
    if view.status == "needs_attention":
        return 3
    return None
