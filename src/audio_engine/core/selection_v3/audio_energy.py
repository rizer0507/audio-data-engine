"""Versioned lightweight audio energy evidence for selection_five_class_v2.

DNSMOS remains optional supplemental evidence and never blocks energy rules.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from audio_engine.core.selection_v3.config import SelectionV3Config
from audio_engine.core.selection_v3.types import (
    ENERGY_STATE_AUDIBLE,
    ENERGY_STATE_BORDERLINE,
    ENERGY_STATE_FAILED,
    ENERGY_STATE_INAUDIBLE,
    ENERGY_STATE_TOO_SHORT,
)


@dataclass(frozen=True)
class AudioEnergyEvidence:
    duration_ms: float | None
    rms_dbfs: float | None
    peak_dbfs: float | None
    non_silent_ratio: float | None
    energy_state: str
    energy_policy_version: str
    source: str = "computed"
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _to_mono(data: np.ndarray) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    return arr


def _dbfs(power: float) -> float:
    return float(10.0 * np.log10(max(power, 1e-12)))


def compute_energy_from_array(
    samples: np.ndarray,
    sample_rate: int,
    *,
    frame_ms: int = 20,
    silence_frame_dbfs: float = -45.0,
) -> dict[str, float]:
    mono = _to_mono(samples)
    n = int(mono.shape[0])
    sr = max(1, int(sample_rate))
    duration_ms = 1000.0 * n / sr
    if n == 0:
        return {
            "duration_ms": 0.0,
            "rms_dbfs": -120.0,
            "peak_dbfs": -120.0,
            "non_silent_ratio": 0.0,
        }

    rms_dbfs = _dbfs(float(np.mean(mono**2)))
    peak_dbfs = _dbfs(float(np.max(mono**2)))

    frame_len = max(1, int(sr * max(1, frame_ms) / 1000))
    n_frames = max(1, n // frame_len)
    frames = mono[: n_frames * frame_len].reshape(n_frames, frame_len)
    frame_db = 10.0 * np.log10(np.maximum(np.mean(frames**2, axis=1), 1e-12))
    non_silent = float(np.mean(frame_db >= float(silence_frame_dbfs)))
    return {
        "duration_ms": float(duration_ms),
        "rms_dbfs": float(rms_dbfs),
        "peak_dbfs": float(peak_dbfs),
        "non_silent_ratio": float(non_silent),
    }


def classify_energy_state(
    *,
    duration_ms: float | None,
    rms_dbfs: float | None,
    peak_dbfs: float | None,
    non_silent_ratio: float | None,
    min_duration_ms: float,
    min_rms_dbfs: float,
    min_peak_dbfs: float,
    min_non_silent_ratio: float,
    borderline_margin_db: float,
) -> str:
    if duration_ms is None or rms_dbfs is None or peak_dbfs is None or non_silent_ratio is None:
        return ENERGY_STATE_FAILED
    if float(duration_ms) < float(min_duration_ms):
        return ENERGY_STATE_TOO_SHORT

    margin = abs(float(borderline_margin_db))
    rms = float(rms_dbfs)
    peak = float(peak_dbfs)
    ratio = float(non_silent_ratio)

    clearly_above = (
        rms >= min_rms_dbfs + margin
        and peak >= min_peak_dbfs + margin
        and ratio >= min_non_silent_ratio
    )
    clearly_below = (
        rms < min_rms_dbfs - margin
        and peak < min_peak_dbfs - margin
    ) or (
        rms < min_rms_dbfs - margin
        and ratio < max(0.0, min_non_silent_ratio * 0.5)
    )
    if clearly_above:
        return ENERGY_STATE_AUDIBLE
    if clearly_below:
        return ENERGY_STATE_INAUDIBLE

    near_rms = abs(rms - min_rms_dbfs) <= margin
    near_peak = abs(peak - min_peak_dbfs) <= margin
    near_ratio = abs(ratio - min_non_silent_ratio) <= max(1e-9, min_non_silent_ratio * 0.5)
    above_floor = (
        rms >= min_rms_dbfs and peak >= min_peak_dbfs and ratio >= min_non_silent_ratio
    )
    if above_floor and not (near_rms or near_peak or near_ratio):
        return ENERGY_STATE_AUDIBLE
    if above_floor and (near_rms or near_peak or near_ratio):
        return ENERGY_STATE_BORDERLINE
    if near_rms or near_peak or near_ratio:
        return ENERGY_STATE_BORDERLINE
    return ENERGY_STATE_INAUDIBLE


def read_energy_from_path(
    path: str | Path,
    *,
    frame_ms: int = 20,
    silence_frame_dbfs: float = -45.0,
) -> dict[str, Any]:
    import soundfile as sf

    data, sr = sf.read(str(path), always_2d=False)
    metrics = compute_energy_from_array(
        np.asarray(data),
        int(sr),
        frame_ms=frame_ms,
        silence_frame_dbfs=silence_frame_dbfs,
    )
    metrics["sample_rate"] = int(sr)
    return metrics


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
