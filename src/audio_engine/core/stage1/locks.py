"""Single-writer job lock and stale worker isolation."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from audio_engine.core.artifacts import atomic_write_json
from audio_engine.core.catalog import utc_now
from audio_engine.core.stage1.process import pid_is_alive


@dataclass
class LockInfo:
    pid: int
    worker_token: str
    acquired_at: str
    job_id: str
    mode: str  # run | resume | retry

    def to_dict(self) -> dict[str, Any]:
        return {
            "pid": self.pid,
            "worker_token": self.worker_token,
            "acquired_at": self.acquired_at,
            "job_id": self.job_id,
            "mode": self.mode,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LockInfo:
        return cls(
            pid=int(data["pid"]),
            worker_token=str(data["worker_token"]),
            acquired_at=str(data["acquired_at"]),
            job_id=str(data["job_id"]),
            mode=str(data.get("mode") or "run"),
        )


class JobLockError(RuntimeError):
    pass


class JobLock:
    """Exclusive lock file under job_root/worker.lock."""

    def __init__(self, job_root: Path, *, job_id: str):
        self.job_root = Path(job_root)
        self.job_id = job_id
        self.path = self.job_root / "worker.lock"
        self.token = uuid.uuid4().hex
        self._held = False

    def read(self) -> LockInfo | None:
        if not self.path.is_file():
            return None
        try:
            return LockInfo.from_dict(json.loads(self.path.read_text(encoding="utf-8")))
        except Exception:
            return None

    def acquire(self, *, mode: str = "run", steal_stale: bool = True) -> LockInfo:
        existing = self.read()
        if existing is not None:
            alive = pid_is_alive(existing.pid)
            if alive and existing.pid != os.getpid():
                raise JobLockError(
                    f"job 已被 worker 占用: pid={existing.pid} token={existing.worker_token[:8]}… "
                    f"mode={existing.mode}；拒绝双写"
                )
            if alive and existing.pid == os.getpid() and existing.worker_token:
                # Same process re-entrant (tests) — refresh token ownership.
                pass
            elif not alive and not steal_stale:
                raise JobLockError(
                    f"发现僵死锁 pid={existing.pid} 且未允许接管；请显式 resume"
                )
            # Stale lock: previous worker died — take over.
        info = LockInfo(
            pid=os.getpid(),
            worker_token=self.token,
            acquired_at=utc_now(),
            job_id=self.job_id,
            mode=mode,
        )
        self.job_root.mkdir(parents=True, exist_ok=True)
        atomic_write_json(self.path, info.to_dict())
        self._held = True
        return info

    def release(self, *, token: str | None = None) -> None:
        expected = token or self.token
        existing = self.read()
        if existing is None:
            self._held = False
            return
        if existing.worker_token != expected:
            # Old worker must not clear a newer owner's lock.
            return
        self.path.unlink(missing_ok=True)
        self._held = False

    def assert_owner(self, token: str) -> None:
        existing = self.read()
        if existing is None:
            raise JobLockError("worker.lock 丢失")
        if existing.worker_token != token:
            raise JobLockError(
                f"旧 worker token 已失效（当前 {existing.worker_token[:8]}…）；拒绝写入"
            )
        if existing.pid != os.getpid() and pid_is_alive(existing.pid):
            raise JobLockError("锁被其他存活进程持有")
