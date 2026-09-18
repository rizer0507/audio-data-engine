"""Digest helpers for stage-1 run identity and config snapshots."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from audio_engine.core.dataset_v3.audit_plan import digest_payload


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def path_content_digest(path: Path, *, max_files: int = 5000) -> str:
    """Stable content digest for a file or model directory.

    Regular files are fully hashed. Directories fingerprint every file's relative
    path and size; small text/config files (<=2MiB) are fully hashed, while large
    weight blobs contribute size+mtime only (still fails if files are missing).
    """
    root = Path(path).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"权重或配置路径不存在: {root}")
    if root.is_file():
        return sha256_file(root)

    text_suffixes = {
        ".json",
        ".jsonl",
        ".txt",
        ".md",
        ".yaml",
        ".yml",
        ".jinja",
        ".py",
        ".tiktoken",
        ".model",
        ".vocab",
        ".bpe",
    }
    entries: list[tuple[str, int, int, str]] = []
    files = sorted(p for p in root.rglob("*") if p.is_file())
    if len(files) > max_files:
        raise ValueError(
            f"模型目录文件数 {len(files)} 超过上限 {max_files}，拒绝摘要: {root}"
        )
    for file_path in files:
        rel = file_path.relative_to(root).as_posix()
        stat = file_path.stat()
        size = int(stat.st_size)
        mtime_ns = int(stat.st_mtime_ns)
        if file_path.suffix.lower() in text_suffixes and size <= 2 * 1024 * 1024:
            file_hash = sha256_file(file_path)
        else:
            file_hash = f"meta:{size}:{mtime_ns}"
        entries.append((rel, size, mtime_ns, file_hash))
    return digest_payload({"root": str(root), "files": entries})


def decode_config_digest(payload: dict[str, Any]) -> str:
    return digest_payload(payload)


def prompt_digest(payload: dict[str, Any] | str | None) -> str:
    if payload is None:
        return digest_payload({"prompt": ""})
    if isinstance(payload, str):
        return sha256_text(payload)
    return digest_payload(payload)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
