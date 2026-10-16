"""Neutral lightweight audio-energy computation (extracted from selection_v3, 044).

Pure signal metrics + threshold classification with explicit parameters. Policy
thresholds are *inputs*; this module never reads a selection config.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from audio_engine.core.evidence.types import (
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
    strict_all_channels_zero: bool | None = None
    any_channel_nonzero: bool | None = None
    channel_count: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_strict_channel_zero(samples: np.ndarray) -> dict[str, Any]:
    """Strict all-zero check *before* mono mix. Never use RMS thresholds here."""
    arr = np.asarray(samples)
    if arr.size == 0:
        return {
            "strict_all_channels_zero": True,
            "any_channel_nonzero": False,
            "channel_count": 0,
        }
    if arr.ndim == 1:
        channels = [arr]
    elif arr.ndim == 2:
        # soundfile always_2d → (frames, channels)
        if arr.shape[1] <= arr.shape[0] or arr.shape[0] < 8:
            channels = [arr[:, i] for i in range(arr.shape[1])]
        else:
            # Rare (channels, frames) layout.
            channels = [arr[i, :] for i in range(arr.shape[0])]
    else:
        flat = arr.reshape(-1)
        channels = [flat]
    all_zero = all(bool(np.all(ch == 0)) for ch in channels)
    any_nonzero = any(bool(np.any(ch != 0)) for ch in channels)
    return {
        "strict_all_channels_zero": all_zero,
        "any_channel_nonzero": any_nonzero,
        "channel_count": len(channels),
    }


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
    arr = np.asarray(data)
    metrics = compute_energy_from_array(
        arr,
        int(sr),
        frame_ms=frame_ms,
        silence_frame_dbfs=silence_frame_dbfs,
    )
    metrics["sample_rate"] = int(sr)
    metrics.update(compute_strict_channel_zero(arr))
    return metrics
