"""Subprocess helpers: CUDA device scope, owned PID tree cleanup, session state."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass
class ManagedProcess:
    pid: int
    argv: list[str]
    log_path: Path
    gpu: int | str
    owned: bool = True
    pgid: int | None = None


@dataclass
class ServiceSession:
    family: str
    gpu: int | str
    port: int | None
    api_base: str | None
    served_model_name: str | None
    owned: bool
    client_env: dict[str, str]
    argv: list[str] = field(default_factory=list)
    pid: int | None = None
    pgid: int | None = None
    log_path: str | None = None
    session_dir: str | None = None
    attached: bool = False
    kind: str = "vllm"  # vllm | local
    extra: dict[str, Any] = field(default_factory=dict)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> ServiceSession:
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(**data)


def build_cuda_env(
    gpu: int | str,
    *,
    base: Mapping[str, str] | None = None,
    overlays: Mapping[str, str] | None = None,
    unset: tuple[str, ...] | list[str] = (),
) -> dict[str, str]:
    """Copy env and force CUDA_VISIBLE_DEVICES to the assigned GPU only."""
    env = dict(base if base is not None else os.environ)
    for key in unset:
        env.pop(str(key), None)
    if overlays:
        env.update({str(k): str(v) for k, v in overlays.items()})
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return env


def start_owned_process(
    argv: list[str],
    *,
    env: Mapping[str, str],
    log_path: Path,
    cwd: Path | None = None,
) -> ManagedProcess:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("ab")
    popen_kwargs: dict[str, Any] = {
        "args": argv,
        "env": dict(env),
        "stdout": log_handle,
        "stderr": subprocess.STDOUT,
        "cwd": str(cwd) if cwd else None,
    }
    if os.name != "nt":
        popen_kwargs["start_new_session"] = True
    proc = subprocess.Popen(**popen_kwargs)
    log_handle.close()
    pgid = None
    if os.name != "nt":
        try:
            pgid = os.getpgid(proc.pid)
        except ProcessLookupError:
            pgid = None
    return ManagedProcess(
        pid=proc.pid,
        argv=list(argv),
        log_path=log_path,
        gpu=env.get("CUDA_VISIBLE_DEVICES", ""),
        owned=True,
        pgid=pgid,
    )


def pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def terminate_owned_process(
    *,
    pid: int | None,
    pgid: int | None = None,
    timeout_s: float = 20.0,
) -> None:
    """Stop only the process (group) we started. No-op when pid is missing."""
    if pid is None or pid <= 0:
        return
    if not pid_is_alive(pid):
        return

    def _signal(sig: int) -> None:
        if os.name != "nt" and pgid is not None:
            try:
                os.killpg(pgid, sig)
                return
            except ProcessLookupError:
                return
            except PermissionError:
                pass
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            return

    _signal(signal.SIGTERM if os.name != "nt" else signal.SIGTERM)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if not pid_is_alive(pid):
            return
        time.sleep(0.2)
    _signal(signal.SIGKILL if os.name != "nt" else signal.SIGTERM)


def wait_for_http_ready(
    url: str,
    *,
    timeout_s: float,
    poll_s: float,
    predicate,
) -> Any:
    """Poll until predicate(url) succeeds or timeout."""
    deadline = time.time() + timeout_s
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            return predicate(url)
        except Exception as exc:  # noqa: BLE001 — readiness may fail until server is up
            last_error = exc
            time.sleep(poll_s)
    raise TimeoutError(f"服务在 {timeout_s:.0f}s 内未就绪: {url}; last_error={last_error}")
