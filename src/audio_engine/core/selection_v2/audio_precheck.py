"""Audio-side precheck helpers for selection_v2.0.

Speech / VAD / overlap features may be absent in Phase A; callers must treat
missing evidence conservatively (prefer review over auto_empty).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from audio_engine.core.sample import Sample


@dataclass
class AudioFeatures:
    audio_valid: bool = True
    broken: bool = False
    duration_sec: float | None = None
    duration_ms: int | None = None
    speech_ratio: float | None = None
    vad_edge_risk: bool = False
    overlap_risk: bool = False
    noise_risk: bool = False
    has_speech_evidence: bool | None = None  # None = unknown
    duplicate_group_id: str | None = None


def _coerce_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def extract_audio_features(sample: Sample) -> AudioFeatures:
    labels = sample.labels or {}
    quality = sample.quality or {}

    broken = _coerce_bool(
        labels.get("label_broken") or labels.get("broken") or quality.get("broken")
    )
    duration = sample.duration
    if duration is None:
        duration = _coerce_float(quality.get("duration") or labels.get("duration"))
    duration_ms = None
    if duration is not None:
        duration_ms = int(round(float(duration) * 1000))
    raw_ms = labels.get("duration_ms") or quality.get("duration_ms")
    if raw_ms is not None:
        try:
            duration_ms = int(raw_ms)
            if duration is None:
                duration = duration_ms / 1000.0
        except (TypeError, ValueError):
            pass

    speech_ratio = _coerce_float(
        labels.get("speech_ratio") or quality.get("speech_ratio")
    )
    vad_edge_risk = _coerce_bool(
        labels.get("vad_edge_risk") or quality.get("vad_edge_risk")
    )
    overlap_risk = _coerce_bool(
        labels.get("overlap_risk") or quality.get("overlap_risk")
    )
    noise_risk = _coerce_bool(labels.get("noise_risk") or quality.get("noise_risk"))

    audio_valid = not broken and not (duration is not None and duration <= 0)

    # Explicit speech evidence if provided; otherwise leave unknown.
    has_speech: bool | None = None
    if speech_ratio is not None:
        has_speech = speech_ratio > 0.05
    explicit = labels.get("has_speech") or quality.get("has_speech")
    if explicit is not None and str(explicit).strip() != "":
        has_speech = _coerce_bool(explicit)
    if vad_edge_risk:
        has_speech = True

    dup = labels.get("duplicate_group_id") or quality.get("duplicate_group_id")
    duplicate_group_id = str(dup).strip() if dup else None

    return AudioFeatures(
        audio_valid=audio_valid,
        broken=broken,
        duration_sec=float(duration) if duration is not None else None,
        duration_ms=duration_ms,
        speech_ratio=speech_ratio,
        vad_edge_risk=vad_edge_risk,
        overlap_risk=overlap_risk,
        noise_risk=noise_risk,
        has_speech_evidence=has_speech,
        duplicate_group_id=duplicate_group_id,
    )


def is_true_silence(
    features: AudioFeatures,
    *,
    max_speech_ratio: float,
    vad_miss_auto_empty: bool,
) -> bool:
    """High-confidence silence only when speech evidence is explicitly weak."""
    if features.has_speech_evidence is True:
        return False
    if features.has_speech_evidence is False:
        return True
    if features.speech_ratio is not None:
        return features.speech_ratio <= max_speech_ratio and not features.vad_edge_risk
    # Missing features: never auto-empty unless config explicitly allows (default false).
    return bool(vad_miss_auto_empty)
