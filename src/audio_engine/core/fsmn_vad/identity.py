"""Parent / segment identity helpers and Sample field mapping."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from audio_engine.core.fsmn_vad.types import VAD_PROCESSING_VERSION, VadKeptSegment
from audio_engine.core.sample import Sample


def stable_segment_id(
    *,
    source_audio_id: str,
    normalized_sha256: str,
    channel_index: int,
    start_sample: int,
    end_sample: int,
    vad_processing_version: str = VAD_PROCESSING_VERSION,
    config_sha256: str = "",
) -> str:
    """Deterministic segment id from parent identity + clip + processing version."""
    payload = {
        "source_audio_id": source_audio_id,
        "normalized_sha256": normalized_sha256,
        "channel_index": int(channel_index),
        "start_sample": int(start_sample),
        "end_sample": int(end_sample),
        "vad_processing_version": vad_processing_version,
        "config_sha256": config_sha256 or "",
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return f"seg_{digest[:28]}"


def segment_content_sha256(pcm16le: bytes) -> str:
    """Hash of raw PCM16 payload (unpadded segment body)."""
    return hashlib.sha256(pcm16le).hexdigest()


def wav_file_sha256(path_bytes: bytes) -> str:
    return hashlib.sha256(path_bytes).hexdigest()


def build_segment_labels(
    *,
    source_audio_id: str,
    source_pcm_sha256: str,
    normalized_sha256: str,
    normalized_audio_path: str,
    channel_index: int,
    segment: VadKeptSegment,
    segment_sha256: str,
    segment_index: int,
    vad_processing_version: str,
    config_sha256: str,
    model_fingerprint: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Field mapping for a published segment Sample.

    Critical: ``original_audio_sha256`` is the *segment* working identity.
    Parent PCM hash lives in ``source_pcm_sha256`` / ``parent_pcm_sha256`` only.

    ``vad_start_ms`` / ``vad_end_ms`` are model (or EOF-inferred) times used for the
    actual PCM crop. ``vad_reported_*`` are original-service return/dedup timestamps
    (split-end reported start is pending-100 ms and must not alter the crop).
    """
    labels: dict[str, Any] = {
        "is_vad_segment": True,
        "source_audio_id": source_audio_id,
        "source_pcm_sha256": source_pcm_sha256,
        "parent_pcm_sha256": source_pcm_sha256,
        "normalized_audio_sha256": normalized_sha256,
        "normalized_audio_path": normalized_audio_path,
        "channel_index": int(channel_index),
        "segment_id": None,  # filled by caller after id known
        "segment_index": int(segment_index),
        "vad_start_ms": segment.start_ms,
        "vad_end_ms": segment.end_ms,
        "vad_inferred_end_ms": segment.inferred_end_ms,
        "vad_reported_start_ms": segment.reported_start_ms,
        "vad_reported_end_ms": segment.reported_end_ms,
        "vad_end_reason": segment.end_reason,
        "vad_clip_branch": segment.clip_branch,
        "clip_start_sample": segment.start_sample,
        "clip_end_sample": segment.end_sample,
        "clip_duration_ms": segment.duration_ms,
        "buffer_end_sample": segment.buffer_end_sample,
        "boundary_clamped": segment.boundary_clamped,
        "extra_padding_ms": segment.extra_padding_ms,
        "vad_processing_version": vad_processing_version,
        "vad_config_sha256": config_sha256,
        # Working identity for cache / aggregate / warehouse / annotation.
        "original_audio_sha256": segment_sha256,
        "segment_pcm_sha256": segment_sha256,
        "warehouse_audio_key": "resampled_16k",
        "audio_pass": True,
    }
    if model_fingerprint:
        labels["vad_model_fingerprint"] = model_fingerprint
    return labels


def clear_inherited_downstream_state(sample: Sample) -> None:
    """Segments must not inherit parent ASR / classification / gold / completed states."""
    sample.transcripts = {}
    # Drop classification / gold / annotation keys; keep audio_pass lineage flags set later.
    drop_prefixes = (
        "category",
        "type",
        "outcome",
        "rule_version",
        "classification",
        "gold_",
        "annotator_",
        "reviewer_",
        "adjudicat",
        "label_tier",
        "label_grade",
        "candidate_text",
        "human_",
        "reviewed_",
        "semantic_",
        "tags",
        "multi_tag",
    )
    keep: dict[str, Any] = {}
    for key, value in sample.labels.items():
        if any(key == p or key.startswith(p) for p in drop_prefixes):
            continue
        keep[key] = value
    sample.labels = keep
    # Reset operator status except we rebuild after.
    sample.status = {}
    sample.errors = {}


def parent_source_audio_id(sample: Sample) -> str:
    existing = sample.labels.get("source_audio_id")
    if existing and not sample.labels.get("is_vad_segment"):
        return str(existing)
    return str(sample.id)


def is_segment_sample(sample: Sample) -> bool:
    return bool(sample.labels.get("is_vad_segment"))
