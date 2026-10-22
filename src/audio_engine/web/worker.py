"""Independent worker process: claim jobs, run attempts in subprocess, heartbeat, cancel tree."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from audio_engine.plugins.registry import PluginRegistry
from audio_engine.web.services.engine import EngineService
from audio_engine.web.settings import WebSettings, load_web_settings
from audio_engine.web.store import JobStore


def _kill_process_tree(pid: int) -> None:
    """Terminate only the task process tree. Never scans by model port/GPU."""

    if pid <= 0:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True,
            check=False,
        )
    else:
        try:
            os.killpg(pid, signal.SIGTERM)
        except Exception:
            try:
                os.kill(pid, signal.SIGTERM)
            except Exception:
                pass


class Worker:
    def __init__(self, settings: WebSettings, worker_id: str | None = None) -> None:
        self.settings = settings
        self.worker_id = worker_id or f"worker_{uuid.uuid4().hex[:8]}"
        self.store = JobStore(settings.sqlite_path)
        self.registry = PluginRegistry(project_root=settings.project_root)
        self.engine = EngineService(settings, self.registry)
        self._stop = threading.Event()

    def run_forever(self) -> None:
        self.engine.ensure_catalog()
        # Startup reconciliation: publication vs SQLite, then stale leases.
        self.store.reconcile_publications()
        self.store.reconcile_stale_attempts()
        while not self._stop.is_set():
            claimed = self.store.claim_next_job(
                worker_id=self.worker_id,
                lease_seconds=self.settings.worker.lease_seconds,
            )
            if not claimed:
                # Periodically re-check expired leases without reclaiming live PIDs.
                self.store.reconcile_stale_attempts()
                time.sleep(self.settings.worker.poll_interval_seconds)
                continue
            self._run_attempt(claimed)

    def stop(self) -> None:
        self._stop.set()

    def _run_attempt(self, job: dict[str, Any]) -> None:
        attempt_id = job["attempt_id"]
        lease_token = job["lease_token"]
        job_id = job["job_id"]
        attempt_no = job["attempt_no"]
        work_dir = (
            self.settings.runs_dir / job_id / f"attempt_{attempt_no}"
        ).resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        log_path = work_dir / "run.log"
        plan_row = self.store.get_plan(job["plan_id"])
        if not plan_row:
            self.store.finish_attempt(
                attempt_id,
                lease_token=lease_token,
                status="failed",
                job_status="failed",
                error={"code": "PLAN_MISSING", "message": "plan row missing"},
            )
            return

        plan = json.loads(plan_row["plan_json"])
        # Prefer snapshot plan.json if present (immutable).
        snap = plan_row.get("snapshot_path")
        if snap and (Path(snap) / "plan.json").is_file():
            plan = json.loads((Path(snap) / "plan.json").read_text(encoding="utf-8"))

        runner_script = work_dir / "execute_plan.py"
        runner_script.write_text(
            _EXECUTOR_TEMPLATE.format(
                project_root=json.dumps(str(self.settings.project_root)),
                plugins_config=json.dumps(str(self.settings.plugins_config)),
                plan_path=json.dumps(str(work_dir / "plan.json")),
                work_dir=json.dumps(str(work_dir)),
                job_id=json.dumps(job_id),
                attempt_id=json.dumps(attempt_id),
            ),
            encoding="utf-8",
        )
        (work_dir / "plan.json").write_text(
            json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # Task-private env for child only (never mutate API/worker global os.environ).
        # Clear inherited GLM/Qwen vars so parent shell cannot override frozen bindings.
        child_env = os.environ.copy()
        for key in list(child_env):
            if key.startswith(("GLM_ASR_", "QWEN_ASR_")):
                child_env.pop(key, None)
        # Persist route/model freeze for adapters; multi-route isolation uses
        # per-route env inside production adapter, not a single global base.
        bindings_path = work_dir / "model_bindings.json"
        bindings_path.write_text(
            json.dumps(
                {
                    "resolved_models": plan.get("resolved_models") or [],
                    "request": plan.get("request") or plan.get("task_request") or {},
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        child_env["AUDIO_ENGINE_WEB_MODEL_BINDINGS"] = str(bindings_path)
        # Convenience single-provider fallback for yaml_pipeline (one base each).
        by_provider: dict[str, list[dict[str, Any]]] = {}
        for model in plan.get("resolved_models") or []:
            by_provider.setdefault(str(model.get("provider") or ""), []).append(model)
        for provider, items in by_provider.items():
            bases = [str(m.get("base_url") or "") for m in items if m.get("base_url")]
            names = [str(m.get("model") or "") for m in items if m.get("model")]
            if provider == "glm" and bases:
                child_env["GLM_ASR_API_BASES"] = ",".join(bases)
                child_env["GLM_ASR_API_BASE"] = bases[0]
                if names:
                    child_env["GLM_ASR_MODEL"] = names[0]
            if provider == "qwen" and bases:
                child_env["QWEN_ASR_API_BASES"] = ",".join(bases)
                child_env["QWEN_ASR_API_BASE"] = bases[0]
                if names:
                    child_env["QWEN_ASR_MODEL"] = names[0]

        popen_kwargs: dict[str, Any] = {
            "cwd": str(self.settings.project_root),
            "env": child_env,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
            "text": True,
        }
        if os.name != "nt":
            popen_kwargs["start_new_session"] = True
        else:
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]

        proc = subprocess.Popen([sys.executable, str(runner_script)], **popen_kwargs)
        start_token = f"{proc.pid}:{time.time_ns()}"
        self.store.set_attempt_process(
            attempt_id,
            lease_token=lease_token,
            pid=proc.pid,
            process_start_token=start_token,
            work_dir=str(work_dir),
            log_path=str(log_path),
        )

        cancel_requested = False
        with log_path.open("a", encoding="utf-8") as log_fh:
            assert proc.stdout is not None
            last_beat = 0.0
            while True:
                line = proc.stdout.readline()
                if line:
                    log_fh.write(line)
                    log_fh.flush()
                elif proc.poll() is not None:
                    # drain remainder
                    rest = proc.stdout.read() or ""
                    if rest:
                        log_fh.write(rest)
                    break

                now = time.time()
                if now - last_beat >= self.settings.worker.heartbeat_interval_seconds:
                    self.store.heartbeat(
                        attempt_id,
                        lease_token=lease_token,
                        lease_seconds=self.settings.worker.lease_seconds,
                        pid=proc.pid,
                    )
                    last_beat = now

                job_row = self.store.get_job(job_id)
                if job_row and job_row["status"] == "cancelling" and not cancel_requested:
                    cancel_requested = True
                    log_fh.write("\n[worker] cancel requested; terminating task process tree\n")
                    log_fh.flush()
                    _kill_process_tree(proc.pid)
                    # wait grace
                    try:
                        proc.wait(timeout=self.settings.worker.cancel_grace_seconds)
                    except subprocess.TimeoutExpired:
                        _kill_process_tree(proc.pid)
                        proc.wait(timeout=5)

        rc = proc.returncode
        result_path = work_dir / "result.json"
        result = None
        if result_path.is_file():
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
            except Exception:
                result = None

        if cancel_requested or (job_row and job_row["status"] == "cancelling"):
            accepted = self.store.finish_attempt(
                attempt_id,
                lease_token=lease_token,
                status="cancelled",
                job_status="cancelled",
                error={"code": "CANCELLED", "message": "cancelled by user"},
            )
            if not accepted:
                # Lost lease: another reconcile/publish owns the job; do not override.
                pass
            return

        if rc != 0 or not result:
            self.store.finish_attempt(
                attempt_id,
                lease_token=lease_token,
                status="failed",
                job_status="failed",
                error={
                    "code": "ATTEMPT_FAILED",
                    "message": f"executor exit={rc}",
                    "result": result,
                },
            )
            return

        outcome = result.get("outcome") or "failed"
        if outcome == "succeeded":
            pub = self.store.try_publish(
                attempt_id,
                lease_token=lease_token,
                artifacts=result.get("artifacts") or [],
                progress=result.get("counts") or {},
                outcome="succeeded",
            )
            if not pub.get("accepted"):
                # Late publisher after lease loss / other attempt already published.
                with log_path.open("a", encoding="utf-8") as log_fh:
                    log_fh.write(
                        f"\n[worker] publish rejected: {pub.get('reason')}\n"
                    )
            return
        if outcome == "cancelled":
            self.store.finish_attempt(
                attempt_id,
                lease_token=lease_token,
                status="cancelled",
                job_status="cancelled",
                error={"code": "CANCELLED", "message": "plugin reported cancelled"},
            )
            return
        self.store.finish_attempt(
            attempt_id,
            lease_token=lease_token,
            status="failed",
            job_status="failed",
            error={
                "code": "PLUGIN_FAILED",
                "message": "plugin outcome failed",
                "result": result,
            },
            artifacts=result.get("artifacts") or [],
        )


_EXECUTOR_TEMPLATE = '''
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path({project_root}) / "src"))
from audio_engine.plugins.registry import PluginRegistry
from audio_engine.web.settings import load_web_settings
from audio_engine.web.services.engine import EngineService

project_root = {project_root}
plugins_config = {plugins_config}
plan = json.loads(Path({plan_path}).read_text(encoding="utf-8"))
work_dir = Path({work_dir})
job_id = {job_id}
attempt_id = {attempt_id}

# Minimal settings for engine
class S:
    pass
settings = S()
settings.project_root = Path(project_root)
settings.plugins_config = Path(plugins_config)
settings.models = []
settings.runs_dir = work_dir
settings.plan_ttl_seconds = 900

registry = PluginRegistry(project_root=Path(project_root))
registry.refresh(Path(plugins_config))
engine = EngineService(settings, registry)

def on_log(msg, details=None):
    print(msg, flush=True)

result = engine.execute_frozen_plan(
    plan,
    work_dir=work_dir,
    job_id=job_id,
    attempt_id=attempt_id,
    on_log=on_log,
)
Path(work_dir, "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print("OUTCOME", result.get("outcome"), flush=True)
sys.exit(0 if result.get("outcome") == "succeeded" else 1)
'''


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="audio-data-worker")
    parser.add_argument("--config", required=True)
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--worker-id", default=None)
    args = parser.parse_args(argv)
    settings = load_web_settings(args.config, project_root=args.project_root)
    worker = Worker(settings, worker_id=args.worker_id)
    try:
        worker.run_forever()
    except KeyboardInterrupt:
        worker.stop()


if __name__ == "__main__":
    main()
