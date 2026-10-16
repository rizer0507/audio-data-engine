"""Derived sample identity for tempo augmentation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from audio_engine.core.augmentation.params import OPERATOR_VERSION, TempoParams
from audio_engine.core.manifest import file_sha256


def short_digest(text: str, n: int = 12) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:n]


def content_sha256(path: Path) -> str:
    return file_sha256(Path(path))


def processing_fingerprint(
    *,
    source_batch: str,
    source_id: str,
    input_audio_sha256: str,
    params: TempoParams,
    ffmpeg_version: str,
    filter_graph: str,
) -> str:
    payload = {
        "source_batch": source_batch,
        "source_id": source_id,
        "input_audio_sha256": input_audio_sha256,
        "augmentation": "tempo",
        "speed_factor": params.speed_factor,
        "backend": params.backend,
        "ffmpeg_version": ffmpeg_version,
        "filter_graph": filter_graph,
        "output_format": params.output_format,
        "output_subtype": params.output_subtype,
        "channel_policy": params.channel_policy,
        "preserve_pitch": True,
        "operator_version": OPERATOR_VERSION,
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def derived_sample_id(fingerprint: str) -> str:
    return f"tempo_{fingerprint[:24]}"


def derived_batch_id(source_batch: str, factor: float, recipe_digest: str) -> str:
    from audio_engine.core.augmentation.params import factor_tag

    return f"{source_batch}__tempo_{factor_tag(factor)}_{recipe_digest[:12]}"


def audio_output_path(product_root: Path, derived_id: str) -> Path:
    """`<product_root>/audio/<hash_prefix>/<derived_id>.wav`."""
    prefix = derived_id.replace("tempo_", "")[:2] or "00"
    path = Path(product_root) / "audio" / prefix / f"{derived_id}.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def cache_key_for(
    *,
    source_batch: str,
    source_id: str,
    input_key: str,
    input_audio_sha256: str,
    params: TempoParams,
    ffmpeg_version: str,
    filter_graph: str,
) -> str:
    payload: dict[str, Any] = {
        "source_batch": source_batch,
        "source_id": source_id,
        "input_key": input_key,
        "input_audio_sha256": input_audio_sha256,
        **params.to_cache_dict(),
        "ffmpeg_version": ffmpeg_version,
        "filter_graph": filter_graph,
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
