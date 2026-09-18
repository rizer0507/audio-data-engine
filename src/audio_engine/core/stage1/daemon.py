"""Background worker launch for stage-1 jobs (survives SSH disconnect)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence


def spawn_job_worker(
    job_root: Path,
    *,
    mode: str = "run",
    family: str | None = None,
    run: str | int | None = None,
    failed_only: bool = True,
    export_only: bool = False,
) -> int:
    """Start a detached process that runs Stage1Orchestrator on job_root."""
    job_root = Path(job_root).resolve()
    log_path = job_root / "worker.log"
    cmd: list[str] = [
        sys.executable,
        "-m",
        "audio_engine.core.stage1.worker",
        str(job_root),
        "--mode",
        mode,
    ]
    if family:
        cmd.extend(["--family", family])
    if run is not None:
        cmd.extend(["--run", str(run)])
    if mode == "retry":
        if failed_only:
            cmd.append("--failed-only")
        else:
            cmd.append("--all")
        if export_only:
            cmd.append("--export-only")
    log_handle = log_path.open("a", encoding="utf-8")
    kwargs: dict = {
        "args": cmd,
        "stdout": log_handle,
        "stderr": subprocess.STDOUT,
        "cwd": str(Path.cwd()),
        "env": dict(os.environ),
    }
    if os.name == "nt":
        kwargs["creationflags"] = (
            getattr(subprocess, "DETACHED_PROCESS", 0)
            | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        )
        kwargs["close_fds"] = True
    else:
        kwargs["start_new_session"] = True
    proc = subprocess.Popen(**kwargs)
    log_handle.close()
    return int(proc.pid)


def worker_argv(
    job_root: Path,
    *,
    mode: str = "run",
    family: str | None = None,
    run: str | int | None = None,
    failed_only: bool = True,
    export_only: bool = False,
) -> Sequence[str]:
    """Build argv for tests / foreground helpers (without python -m)."""
    args = [str(job_root), "--mode", mode]
    if family:
        args.extend(["--family", family])
    if run is not None:
        args.extend(["--run", str(run)])
    if mode == "retry":
        args.append("--failed-only" if failed_only else "--all")
        if export_only:
            args.append("--export-only")
    return args
