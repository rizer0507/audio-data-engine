"""Speed-factor validation and tempo parameter resolution."""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

DEFAULT_SPEED_FACTOR = 1.2
OPERATOR_VERSION = "1.0.0"
BACKEND_NAME = "ffmpeg_atempo"


def normalize_speed_factor(raw: Any) -> float:
    """Accept finite numbers strictly greater than 1; reject bool/NaN/Inf/<=1."""
    if isinstance(raw, bool) or raw is None:
        raise ValueError(
            f"speed_factor must be a finite number > 1, got {raw!r}"
        )
    if isinstance(raw, str):
        text = raw.strip()
        if not text or text.lower() in {"nan", "inf", "+inf", "-inf", "true", "false"}:
            raise ValueError(f"speed_factor must be a finite number > 1, got {raw!r}")
        try:
            value = float(Decimal(text))
        except (InvalidOperation, ValueError) as exc:
            raise ValueError(f"speed_factor must be a finite number > 1, got {raw!r}") from exc
    elif isinstance(raw, (int, float)):
        value = float(raw)
    else:
        raise ValueError(f"speed_factor must be a finite number > 1, got {type(raw).__name__}")

    if not math.isfinite(value) or value <= 1.0:
        raise ValueError(f"speed_factor must be a finite number > 1, got {raw!r}")
    # Exact identity: round-trip via Decimal string of the float's shortest repr.
    return float(Decimal(str(value)))


def factor_tag(factor: float) -> str:
    """Readable path tag; precise identity comes from recipe_digest."""
    normalized = normalize_speed_factor(factor)
    text = format(Decimal(str(normalized)), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text.replace(".", "p")


@dataclass(frozen=True)
class TempoParams:
    speed_factor: float = DEFAULT_SPEED_FACTOR
    input_audio_key: str = "resampled_16k"
    output_audio_key: str = "aug_speed"
    backend: str = BACKEND_NAME
    preserve_pitch: bool = True
    output_format: str = "wav"
    output_subtype: str = "PCM_16"
    channel_policy: str = "preserve"
    backend_threads: int = 1
    timeout_seconds: float = 120.0
    max_attempts: int = 2
    workers: int = 4
    max_in_flight: int = 16
    ffmpeg_bin: str = "ffmpeg"
    allow_raw_pcm: bool = False
    pcm_sample_rate: int | None = None
    pcm_channels: int | None = None
    pcm_subtype: str | None = None

    def to_cache_dict(self) -> dict[str, Any]:
        return {
            "speed_factor": self.speed_factor,
            "input_audio_key": self.input_audio_key,
            "output_audio_key": self.output_audio_key,
            "backend": self.backend,
            "preserve_pitch": self.preserve_pitch,
            "output_format": self.output_format,
            "output_subtype": self.output_subtype,
            "channel_policy": self.channel_policy,
            "backend_threads": self.backend_threads,
            "operator_version": OPERATOR_VERSION,
        }


def resolve_tempo_params(raw: dict[str, Any] | None) -> TempoParams:
    data = dict(raw or {})
    factor = normalize_speed_factor(data.get("speed_factor", DEFAULT_SPEED_FACTOR))
    backend = str(data.get("backend") or BACKEND_NAME).strip()
    if backend != BACKEND_NAME:
        raise ValueError(
            f"unsupported tempo backend {backend!r}; only {BACKEND_NAME!r} is allowed "
            "(do not silent-switch to pitch-changing resample)"
        )
    if data.get("preserve_pitch") is False:
        raise ValueError("preserve_pitch must be true for augmentation.tempo")
    channel_policy = str(data.get("channel_policy") or "preserve").strip()
    if channel_policy != "preserve":
        raise ValueError(
            f"channel_policy={channel_policy!r} unsupported; first version only preserves channels"
        )
    threads = int(data.get("backend_threads") or 1)
    if threads < 1:
        raise ValueError("backend_threads must be >= 1")
    timeout = float(data.get("timeout_seconds") or 120.0)
    if timeout <= 0:
        raise ValueError("timeout_seconds must be > 0")
    attempts = int(data.get("max_attempts") or 2)
    if attempts < 1:
        raise ValueError("max_attempts must be >= 1")
    workers = max(1, int(data.get("workers") or 4))
    max_in_flight = max(workers, int(data.get("max_in_flight") or workers * 4))
    return TempoParams(
        speed_factor=factor,
        input_audio_key=str(data.get("input_audio_key") or "resampled_16k"),
        output_audio_key=str(data.get("output_audio_key") or "aug_speed"),
        backend=backend,
        preserve_pitch=True,
        output_format=str(data.get("output_format") or "wav").lower(),
        output_subtype=str(data.get("output_subtype") or "PCM_16"),
        channel_policy=channel_policy,
        backend_threads=threads,
        timeout_seconds=timeout,
        max_attempts=attempts,
        workers=workers,
        max_in_flight=max_in_flight,
        ffmpeg_bin=str(data.get("ffmpeg_bin") or "ffmpeg"),
        allow_raw_pcm=bool(data.get("allow_raw_pcm") or False),
        pcm_sample_rate=(
            int(data["pcm_sample_rate"]) if data.get("pcm_sample_rate") is not None else None
        ),
        pcm_channels=(
            int(data["pcm_channels"]) if data.get("pcm_channels") is not None else None
        ),
        pcm_subtype=(
            str(data["pcm_subtype"]) if data.get("pcm_subtype") is not None else None
        ),
    )
