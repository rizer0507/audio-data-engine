"""Checkpoint / resume / atomic publish for multi-tag runs (040 §5.3 / §12.3)."""

from __future__ import annotations

import json
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RunWorkspace:
    batch_id: str
    policy_version: str
    root: Path
    run_id: str
    snapshot_id: str

    @property
    def work_dir(self) -> Path:
        return self.root / "work" / self.run_id

    @property
    def published_dir(self) -> Path:
        return self.root / "published"

    @property
    def cache_dir(self) -> Path:
        return self.root / "cache"

    @property
    def state_path(self) -> Path:
        return self.root / "tag_state.json"

    @property
    def op_log_path(self) -> Path:
        return self.root / "manual_ops.jsonl"

    @property
    def pointer_path(self) -> Path:
        return self.published_dir / "CURRENT.json"

    def ensure(self) -> None:
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.published_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)


def resolve_work_root(
    *,
    batch_id: str,
    policy_version: str,
    template: str,
    override: str | Path | None = None,
) -> Path:
    if override:
        return Path(override)
    return Path(template.format(batch_id=batch_id, policy_version=policy_version))


def create_run_workspace(
    *,
    batch_id: str,
    policy_version: str,
    template: str,
    work_root: str | Path | None = None,
    resume_run_id: str | None = None,
) -> RunWorkspace:
    root = resolve_work_root(
        batch_id=batch_id,
        policy_version=policy_version,
        template=template,
        override=work_root,
    )
    root.mkdir(parents=True, exist_ok=True)
    run_id = resume_run_id or f"run_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    snapshot_id = f"snap_{uuid.uuid4().hex[:12]}"
    ws = RunWorkspace(
        batch_id=batch_id,
        policy_version=policy_version,
        root=root,
        run_id=run_id,
        snapshot_id=snapshot_id,
    )
    ws.ensure()
    return ws


def save_progress(ws: RunWorkspace, payload: dict[str, Any]) -> Path:
    path = ws.work_dir / "progress.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)
    return path


def load_progress(ws: RunWorkspace) -> dict[str, Any] | None:
    path = ws.work_dir / "progress.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_evidence_cache(ws: RunWorkspace, key: str, payload: dict[str, Any]) -> None:
    path = ws.cache_dir / f"{key}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def load_evidence_cache(ws: RunWorkspace, key: str) -> dict[str, Any] | None:
    path = ws.cache_dir / f"{key}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class PublishArtifacts:
    parquet_path: Path
    detail_path: Path
    summary_xlsx_path: Path
    meta_path: Path
    extras: dict[str, Path] = field(default_factory=dict)


def atomic_publish(
    ws: RunWorkspace,
    *,
    artifacts: dict[str, Path],
    meta: dict[str, Any],
) -> dict[str, Any]:
    """
    Copy completed work artifacts into published/<snapshot_id>/ then flip CURRENT pointer.
    Incomplete runs must not update CURRENT.
    """
    snap_dir = ws.published_dir / ws.snapshot_id
    staging = ws.published_dir / f".staging_{ws.snapshot_id}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)

    published_paths: dict[str, str] = {}
    for name, src in artifacts.items():
        src_path = Path(src)
        if not src_path.exists():
            raise FileNotFoundError(f"missing publish artifact {name}: {src_path}")
        dest = staging / src_path.name
        shutil.copy2(src_path, dest)
        published_paths[name] = str(dest.name)

    meta_out = dict(meta)
    meta_out.update(
        {
            "snapshot_id": ws.snapshot_id,
            "run_id": ws.run_id,
            "batch_id": ws.batch_id,
            "policy_version": ws.policy_version,
            "published_at": time.time(),
            "artifacts": published_paths,
        }
    )
    (staging / "meta.json").write_text(
        json.dumps(meta_out, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    if snap_dir.exists():
        shutil.rmtree(snap_dir)
    staging.replace(snap_dir)

    pointer = {
        "snapshot_id": ws.snapshot_id,
        "run_id": ws.run_id,
        "batch_id": ws.batch_id,
        "policy_version": ws.policy_version,
        "dir": str(snap_dir.as_posix()),
        "updated_at": time.time(),
    }
    tmp_ptr = ws.pointer_path.with_suffix(".tmp")
    tmp_ptr.write_text(json.dumps(pointer, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_ptr.replace(ws.pointer_path)
    return pointer


def read_current_pointer(ws: RunWorkspace) -> dict[str, Any] | None:
    if not ws.pointer_path.exists():
        return None
    return json.loads(ws.pointer_path.read_text(encoding="utf-8"))
