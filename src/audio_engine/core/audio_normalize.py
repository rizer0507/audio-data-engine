"""Shared audio normalize policy for 041: mono ch0 + 16 kHz target."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

RateSource = Literal["file_header", "source_config", "default_convention"]
ChannelPolicy = Literal["mono_passthrough", "take_channel_0"]

TARGET_SR = 16000
SELECTED_CHANNEL = 0


@dataclass(frozen=True)
class NormalizeMeta:
    source_sample_rate: int
    target_sample_rate: int
    source_channels: int
    selected_channel: int
    channel_policy: ChannelPolicy
    sample_rate_source: RateSource
    resampled: bool
    duration_source_sec: float
    duration_target_sec: float

    def as_quality(self) -> dict[str, Any]:
        return {
            "source_sample_rate": self.source_sample_rate,
            "target_sample_rate": self.target_sample_rate,
            "source_channels": self.source_channels,
            "selected_channel": self.selected_channel,
            "channel_policy": self.channel_policy,
            "sample_rate_source": self.sample_rate_source,
            "normalize_resampled": self.resampled,
            "duration_source_sec": self.duration_source_sec,
            "duration_target_sec": self.duration_target_sec,
        }

    def as_labels(self) -> dict[str, Any]:
        return {
            "source_sample_rate": self.source_sample_rate,
            "normalized_sample_rate": self.target_sample_rate,
            "source_channels": self.source_channels,
            "selected_channel": self.selected_channel,
            "channel_policy": self.channel_policy,
            "sample_rate_source": self.sample_rate_source,
        }


def take_channel_0(data: np.ndarray) -> tuple[np.ndarray, int, ChannelPolicy]:
    """Return mono float/int series from 1-D or 2-D array; always prefer channel 0."""
    arr = np.asanyarray(data)
    if arr.ndim == 1:
        return arr, 1, "mono_passthrough"
    if arr.ndim != 2:
        raise ValueError(f"unsupported audio ndim={arr.ndim}")
    n_ch = int(arr.shape[1])
    if n_ch < 1:
        raise ValueError("audio has zero channels")
    if n_ch == 1:
        return arr[:, 0], 1, "mono_passthrough"
    return arr[:, SELECTED_CHANNEL], n_ch, "take_channel_0"


def validate_headerless_pcm_bytes(
    n_values: int,
    *,
    channels: int,
    sample_width_bytes: int = 2,
) -> None:
    """Reject interleaved length that cannot form whole frames."""
    if channels < 1:
        raise ValueError(f"invalid channels={channels}")
    frame_values = channels
    if n_values % frame_values != 0:
        raise ValueError(
            f"headerless PCM length {n_values} values is not divisible by "
            f"declared channels={channels} (would mis-read interleaved multi-channel "
            f"as mono and corrupt duration/time)"
        )
    # sample_width kept for API clarity; int16 path already counts values not bytes.
    _ = sample_width_bytes


def resolve_headerless_rate(
    params: dict[str, Any],
    *,
    default_sr: int = TARGET_SR,
    labels: dict[str, Any] | None = None,
) -> tuple[int, RateSource]:
    """Resolve headerless PCM sample rate.

    Preference: per-sample ``labels.source_sample_rate`` (scene routing) →
    YAML ``params.source_sample_rate`` → default convention (16 kHz).
    """
    if labels and labels.get("source_sample_rate") is not None:
        return int(labels["source_sample_rate"]), "source_config"
    if "source_sample_rate" in params and params["source_sample_rate"] is not None:
        return int(params["source_sample_rate"]), "source_config"
    return int(default_sr), "default_convention"


def pcm_array_to_float32(data: np.ndarray) -> np.ndarray:
    """Map integer PCM into float32 amplitude domain before resampling / WAV write."""
    arr = np.asanyarray(data)
    if np.issubdtype(arr.dtype, np.floating):
        return arr.astype(np.float32, copy=False)
    if arr.dtype == np.int16:
        return (arr.astype(np.float32) / 32768.0).astype(np.float32)
    if arr.dtype == np.int32:
        return (arr.astype(np.float32) / 2147483648.0).astype(np.float32)
    if arr.dtype == np.uint8:
        return ((arr.astype(np.float32) - 128.0) / 128.0).astype(np.float32)
    return arr.astype(np.float32)


def write_pcm16_wav(path: str, data: np.ndarray, sample_rate: int) -> None:
    """Write mono/multi PCM16 WAV from float or integer arrays without int16-scale clipping.

    Integer inputs are converted to float amplitude first so ``soundfile`` does not
    treat ~[-32768, 32767] floats as normalized [-1, 1] and saturate.
    """
    import soundfile as sf

    arr = pcm_array_to_float32(np.asanyarray(data))
    sf.write(str(path), arr, int(sample_rate), subtype="PCM_16")


def is_non_target_rate(sr: int, target: int = TARGET_SR) -> bool:
    return int(sr) != int(target)


def format_rate_alert(
    *,
    sample_id: str,
    path: str,
    source_sr: int,
    target_sr: int,
    source_channels: int,
    selected_channel: int,
) -> str:
    tag = "8kHz" if source_sr == 8000 else f"{source_sr}Hz"
    return (
        f"[AUDIO-NORMALIZE] {tag} input → {target_sr}Hz mono ch{selected_channel}: "
        f"id={sample_id} path={path} source_channels={source_channels}"
    )
