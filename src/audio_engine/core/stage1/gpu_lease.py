"""Cross-job GPU lease store: single writer per GPU, no foreign preemption."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from audio_engine.core.artifacts import atomic_write_json
from audio_engine.core.catalog import utc_now
from audio_engine.core.stage1.process import pid_is_alive


class GpuLeaseError(RuntimeError):
    pass


@dataclass
class GpuLease:
    gpu_key: str  # normalized index
    gpu_uuid: str | None
    job_id: str
    worker_token: str
    family: str
    pid: int
    acquired_at: str
    heartbeat_at: str
    mode: str = "run"  # run | resume | retry | hold

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GpuLease:
        return cls(
            gpu_key=str(data["gpu_key"]),
            gpu_uuid=data.get("gpu_uuid"),
            job_id=str(data["job_id"]),
            worker_token=str(data["worker_token"]),
            family=str(data["family"]),
            pid=int(data["pid"]),
            acquired_at=str(data["acquired_at"]),
            heartbeat_at=str(data.get("heartbeat_at") or data["acquired_at"]),
            mode=str(data.get("mode") or "run"),
        )


class GpuLeaseStore:
    """File-backed leases under ``runs/stage1/gpu_leases/<gpu_key>.json``."""

    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, gpu_key: int | str) -> Path:
        key = str(int(gpu_key)) if str(gpu_key).isdigit() else str(gpu_key).strip()
        safe = key.replace("/", "_").replace("\\", "_").replace(":", "_")
        return self.root / f"{safe}.json"

    def read(self, gpu_key: int | str) -> GpuLease | None:
        path = self.path_for(gpu_key)
        if not path.is_file():
            return None
        try:
            return GpuLease.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            return None

    def list_leases(self) -> list[GpuLease]:
        items: list[GpuLease] = []
        for path in sorted(self.root.glob("*.json")):
            try:
                items.append(
                    GpuLease.from_dict(json.loads(path.read_text(encoding="utf-8")))
                )
            except Exception:
                continue
        return items

    def acquire(
        self,
        gpu_key: int | str,
        *,
        job_id: str,
        worker_token: str,
        family: str,
        gpu_uuid: str | None = None,
        mode: str = "run",
        steal_stale_same_job: bool = True,
    ) -> GpuLease:
        existing = self.read(gpu_key)
        if existing is not None:
            alive = pid_is_alive(existing.pid)
            if alive and existing.job_id != job_id:
                raise GpuLeaseError(
                    f"GPU {gpu_key} 已被其他任务占用: job={existing.job_id} "
                    f"family={existing.family} pid={existing.pid}；拒绝抢占"
                )
            if alive and existing.job_id == job_id and existing.worker_token != worker_token:
                # Same job but different worker — only one holder.
                if existing.pid != os.getpid():
                    raise GpuLeaseError(
                        f"GPU {gpu_key} 已被本任务其他 worker 占用 "
                        f"token={existing.worker_token[:8]}…"
                    )
            if alive and existing.job_id == job_id and existing.pid == os.getpid():
                # Re-entrant refresh.
                pass
            elif not alive:
                if existing.job_id != job_id and not steal_stale_same_job:
                    raise GpuLeaseError(
                        f"GPU {gpu_key} 存在其他任务僵死租约 job={existing.job_id}；"
                        "不自动抢占，请人工清理"
                    )
                if existing.job_id != job_id:
                    # Never auto-steal foreign job leases — even if stale.
                    raise GpuLeaseError(
                        f"GPU {gpu_key} 存在其他任务僵死租约 job={existing.job_id}；"
                        "多批提交不能抢占同一租约，请清理后重试"
                    )
                # Same job stale: resume may take over.
            elif alive and existing.job_id == job_id:
                pass
            else:
                raise GpuLeaseError(f"GPU {gpu_key} 租约冲突")

        now = utc_now()
        lease = GpuLease(
            gpu_key=str(int(gpu_key)) if str(gpu_key).isdigit() else str(gpu_key),
            gpu_uuid=gpu_uuid,
            job_id=job_id,
            worker_token=worker_token,
            family=family,
            pid=os.getpid(),
            acquired_at=now,
            heartbeat_at=now,
            mode=mode,
        )
        atomic_write_json(self.path_for(gpu_key), lease.to_dict())
        return lease

    def heartbeat(self, gpu_key: int | str, *, worker_token: str) -> None:
        lease = self.read(gpu_key)
        if lease is None:
            return
        if lease.worker_token != worker_token:
            return
        lease.heartbeat_at = utc_now()
        atomic_write_json(self.path_for(gpu_key), lease.to_dict())

    def release(
        self,
        gpu_key: int | str,
        *,
        worker_token: str,
        job_id: str | None = None,
    ) -> None:
        lease = self.read(gpu_key)
        if lease is None:
            return
        if lease.worker_token != worker_token:
            # Old worker must not clear a newer owner's lease.
            return
        if job_id is not None and lease.job_id != job_id:
            return
        self.path_for(gpu_key).unlink(missing_ok=True)

    def release_all_for_job(self, *, job_id: str, worker_token: str) -> list[str]:
        released: list[str] = []
        for lease in self.list_leases():
            if lease.job_id != job_id:
                continue
            if lease.worker_token != worker_token:
                continue
            self.path_for(lease.gpu_key).unlink(missing_ok=True)
            released.append(lease.gpu_key)
        return released

    def held_by_other_job(self, gpu_key: int | str, *, job_id: str) -> GpuLease | None:
        lease = self.read(gpu_key)
        if lease is None:
            return None
        if lease.job_id != job_id and pid_is_alive(lease.pid):
            return lease
        return None
