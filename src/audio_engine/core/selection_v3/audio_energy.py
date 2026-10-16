"""Versioned lightweight audio energy evidence for selection_five_class_v2.

DNSMOS remains optional supplemental evidence and never blocks energy rules.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from audio_engine.core.evidence.energy import (  # noqa: F401  (044 re-exports)
    AudioEnergyEvidence,
    _dbfs,
    _to_mono,
    classify_energy_state,
    compute_energy_from_array,
    compute_strict_channel_zero,
    read_energy_from_path,
)
from audio_engine.core.selection_v3.config import SelectionV3Config
from audio_engine.core.selection_v3.types import (
    ENERGY_STATE_AUDIBLE,
    ENERGY_STATE_BORDERLINE,
    ENERGY_STATE_FAILED,
    ENERGY_STATE_INAUDIBLE,
    ENERGY_STATE_TOO_SHORT,
)


def energy_evidence_from_quality(
    quality: dict[str, Any] | None,
    *,
    config: SelectionV3Config | None = None,
    duration_sec: float | None = None,
) -> AudioEnergyEvidence:
    """Prefer precomputed quality fields; fall back to duration-only too_short/failed."""
    q = quality if isinstance(quality, dict) else {}
    policy = (
        str(q.get("energy_policy_version") or "")
        or (config.audio_energy_policy_version if config else "audio_energy_v1")
    )
    min_duration = float(
        config.audio_energy_min_duration_ms if config else 300.0
    )
    min_rms = float(config.audio_energy_min_rms_dbfs if config else -50.0)
    min_peak = float(config.audio_energy_min_peak_dbfs if config else -40.0)
    min_ratio = float(config.audio_energy_min_non_silent_ratio if config else 0.02)
    margin = float(config.audio_energy_borderline_margin_db if config else 3.0)

    duration_ms = q.get("duration_ms")
    if duration_ms is None and duration_sec is not None:
        duration_ms = float(duration_sec) * 1000.0
    rms = q.get("rms_dbfs")
    peak = q.get("peak_dbfs")
    ratio = q.get("non_silent_ratio")
    state = str(q.get("energy_state") or "").strip()
    strict_zero = q.get("strict_all_channels_zero")
    any_nonzero = q.get("any_channel_nonzero")
    channel_count = q.get("channel_count")

    if state in {
        ENERGY_STATE_TOO_SHORT,
        ENERGY_STATE_INAUDIBLE,
        ENERGY_STATE_AUDIBLE,
        ENERGY_STATE_BORDERLINE,
        ENERGY_STATE_FAILED,
    } and (rms is not None or state in {ENERGY_STATE_TOO_SHORT, ENERGY_STATE_FAILED}):
        return AudioEnergyEvidence(
            duration_ms=float(duration_ms) if duration_ms is not None else None,
            rms_dbfs=float(rms) if rms is not None else None,
            peak_dbfs=float(peak) if peak is not None else None,
            non_silent_ratio=float(ratio) if ratio is not None else None,
            energy_state=state,
            energy_policy_version=policy,
            source="quality_fields",
            strict_all_channels_zero=(
                bool(strict_zero) if strict_zero is not None else None
            ),
            any_channel_nonzero=(
                bool(any_nonzero) if any_nonzero is not None else None
            ),
            channel_count=int(channel_count) if channel_count is not None else None,
        )

    if duration_ms is not None and rms is not None and peak is not None and ratio is not None:
        computed_state = classify_energy_state(
            duration_ms=float(duration_ms),
            rms_dbfs=float(rms),
            peak_dbfs=float(peak),
            non_silent_ratio=float(ratio),
            min_duration_ms=min_duration,
            min_rms_dbfs=min_rms,
            min_peak_dbfs=min_peak,
            min_non_silent_ratio=min_ratio,
            borderline_margin_db=margin,
        )
        return AudioEnergyEvidence(
            duration_ms=float(duration_ms),
            rms_dbfs=float(rms),
            peak_dbfs=float(peak),
            non_silent_ratio=float(ratio),
            energy_state=computed_state,
            energy_policy_version=policy,
            source="quality_metrics",
        )

    if duration_ms is not None and float(duration_ms) < min_duration:
        return AudioEnergyEvidence(
            duration_ms=float(duration_ms),
            rms_dbfs=float(rms) if rms is not None else None,
            peak_dbfs=float(peak) if peak is not None else None,
            non_silent_ratio=float(ratio) if ratio is not None else None,
            energy_state=ENERGY_STATE_TOO_SHORT,
            energy_policy_version=policy,
            source="duration_only",
        )

    return AudioEnergyEvidence(
        duration_ms=float(duration_ms) if duration_ms is not None else None,
        rms_dbfs=float(rms) if rms is not None else None,
        peak_dbfs=float(peak) if peak is not None else None,
        non_silent_ratio=float(ratio) if ratio is not None else None,
        energy_state=ENERGY_STATE_FAILED,
        energy_policy_version=policy,
        source="missing_metrics",
        error="audio_energy_metrics_missing",
    )


def compute_audio_energy_for_sample(
    *,
    audio_path: str | Path | None,
    config: SelectionV3Config,
    duration_sec: float | None = None,
    quality: dict[str, Any] | None = None,
) -> AudioEnergyEvidence:
    """Compute or reuse energy evidence. Never invent noise from DNSMOS alone."""
    existing = energy_evidence_from_quality(
        quality, config=config, duration_sec=duration_sec
    )
    if existing.source in {"quality_fields", "quality_metrics"} and existing.energy_state != ENERGY_STATE_FAILED:
        return existing
    if existing.energy_state == ENERGY_STATE_TOO_SHORT and existing.source == "duration_only":
        return existing

    if not audio_path:
        return AudioEnergyEvidence(
            duration_ms=(float(duration_sec) * 1000.0) if duration_sec is not None else None,
            rms_dbfs=None,
            peak_dbfs=None,
            non_silent_ratio=None,
            energy_state=ENERGY_STATE_FAILED,
            energy_policy_version=config.audio_energy_policy_version,
            source="no_audio_path",
            error="audio_path_missing",
        )

    path = Path(str(audio_path))
    if not path.is_file():
        return AudioEnergyEvidence(
            duration_ms=(float(duration_sec) * 1000.0) if duration_sec is not None else None,
            rms_dbfs=None,
            peak_dbfs=None,
            non_silent_ratio=None,
            energy_state=ENERGY_STATE_FAILED,
            energy_policy_version=config.audio_energy_policy_version,
            source="path_missing",
            error=f"audio_not_readable:{path}",
        )

    try:
        metrics = read_energy_from_path(
            path,
            frame_ms=int(config.audio_energy_frame_ms),
            silence_frame_dbfs=float(config.audio_energy_silence_frame_dbfs),
        )
        state = classify_energy_state(
            duration_ms=float(metrics["duration_ms"]),
            rms_dbfs=float(metrics["rms_dbfs"]),
            peak_dbfs=float(metrics["peak_dbfs"]),
            non_silent_ratio=float(metrics["non_silent_ratio"]),
            min_duration_ms=float(config.audio_energy_min_duration_ms),
            min_rms_dbfs=float(config.audio_energy_min_rms_dbfs),
            min_peak_dbfs=float(config.audio_energy_min_peak_dbfs),
            min_non_silent_ratio=float(config.audio_energy_min_non_silent_ratio),
            borderline_margin_db=float(config.audio_energy_borderline_margin_db),
        )
        return AudioEnergyEvidence(
            duration_ms=float(metrics["duration_ms"]),
            rms_dbfs=float(metrics["rms_dbfs"]),
            peak_dbfs=float(metrics["peak_dbfs"]),
            non_silent_ratio=float(metrics["non_silent_ratio"]),
            energy_state=state,
            energy_policy_version=config.audio_energy_policy_version,
            source="computed",
            strict_all_channels_zero=bool(metrics.get("strict_all_channels_zero")),
            any_channel_nonzero=bool(metrics.get("any_channel_nonzero")),
            channel_count=(
                int(metrics["channel_count"])
                if metrics.get("channel_count") is not None
                else None
            ),
        )
    except Exception as exc:  # noqa: BLE001 — isolate per-sample IO failures
        return AudioEnergyEvidence(
            duration_ms=(float(duration_sec) * 1000.0) if duration_sec is not None else None,
            rms_dbfs=None,
            peak_dbfs=None,
            non_silent_ratio=None,
            energy_state=ENERGY_STATE_FAILED,
            energy_policy_version=config.audio_energy_policy_version,
            source="read_failed",
            error=str(exc),
        )
