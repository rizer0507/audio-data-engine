"""Identity / digest helpers for batch warehouse binding."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from audio_engine.core.sample import Sample


def original_audio_sha(sample: Sample) -> str:
    labels = sample.labels or {}
    return str(
        labels.get("original_audio_sha256")
        or labels.get("source_audio_sha256")
        or sample.sha256
        or ""
    )


def classified_snapshot_digest(samples: Iterable[Sample]) -> str:
    """Stable digest of classified snapshot identity (id + audio + category + type)."""
    rows: list[dict[str, Any]] = []
    for sample in sorted(samples, key=lambda s: s.id):
        rows.append(
            {
                "id": sample.id,
                "original_audio_sha256": original_audio_sha(sample),
                "source_path": sample.source_path,
                "category": sample.labels.get("category"),
                "type": sample.labels.get("type") or sample.labels.get("classification_bucket"),
                "outcome": sample.labels.get("outcome"),
                "rule_version": sample.labels.get("rule_version"),
            }
        )
    payload = json.dumps(rows, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def preferred_audio_key(sample: Sample) -> str:
    labels = sample.labels or {}
    explicit = str(labels.get("warehouse_audio_key") or "").strip()
    if explicit:
        return explicit
    audio = sample.audio or {}
    for key in ("resampled_16k", "raw", "normalized_pcm"):
        if key in audio and audio[key]:
            return key
    if audio:
        return next(iter(audio))
    return "raw"


def resolve_audio_ref(sample: Sample) -> dict[str, Any]:
    """Resolve selected audio path + digests without copying bytes into the warehouse."""
    key = preferred_audio_key(sample)
    path_str = ""
    if sample.audio and key in sample.audio:
        path_str = str(sample.audio[key] or "")
    if not path_str:
        path_str = str(sample.source_path or "")
    path = Path(path_str) if path_str else None
    exists = bool(path and path.is_file())
    size = int(path.stat().st_size) if exists and path is not None else 0
    content_sha = ""
    if exists and path is not None and size > 0:
        from audio_engine.core.manifest import file_sha256

        content_sha = file_sha256(path)
    return {
        "audio_key": key,
        "audio_path": path_str,
        "audio_exists": exists,
        "audio_size_bytes": size,
        "audio_content_sha256": content_sha,
        "original_audio_sha256": original_audio_sha(sample),
        "sample_sha256": sample.sha256 or "",
    }
