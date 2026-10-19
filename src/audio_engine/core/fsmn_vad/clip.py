"""Sample-accurate PCM16 crop helpers (fenshen byte-offset rules)."""

from __future__ import annotations

import math
from dataclasses import dataclass

from audio_engine.core.fsmn_vad.types import BYTES_PER_MS, SR


@dataclass(frozen=True)
class ClipBounds:
    start_ms: float
    end_ms: float
    start_sample: int
    end_sample: int
    raw_start_sample: int
    raw_end_sample: int
    boundary_clamped: bool
    duration_ms: float


def ms_to_sample(ms: float, *, sample_rate: int = SR) -> int:
    """floor(ms * sr / 1000) — matches source integration doc."""
    return int(math.floor(float(ms) * sample_rate / 1000.0))


def ms_to_byte_offset(ms: float, *, bytes_per_ms: int = BYTES_PER_MS) -> int:
    """Original service: int(ms * SAMPLE_RATE * 2 / 1000)."""
    return int(float(ms) * bytes_per_ms)


def compute_clip_bounds(
    start_ms: float,
    end_ms: float,
    total_samples: int,
    *,
    sample_rate: int = SR,
    end_is_eof: bool = False,
) -> ClipBounds:
    raw_start = ms_to_sample(start_ms, sample_rate=sample_rate)
    if end_is_eof:
        raw_end = total_samples
    else:
        raw_end = ms_to_sample(end_ms, sample_rate=sample_rate)

    start = max(0, min(total_samples, raw_start))
    end = max(0, min(total_samples, raw_end))
    if end < start:
        end = start
    clamped = start != raw_start or end != raw_end
    duration_ms = (end - start) * 1000.0 / sample_rate if end > start else 0.0
    return ClipBounds(
        start_ms=start_ms,
        end_ms=end_ms,
        start_sample=start,
        end_sample=end,
        raw_start_sample=raw_start,
        raw_end_sample=raw_end,
        boundary_clamped=clamped,
        duration_ms=duration_ms,
    )


def clip_pcm16le(pcm16le: bytes, start_sample: int, end_sample: int) -> bytes:
    if len(pcm16le) % 2:
        raise ValueError("PCM16 byte length must be even")
    total = len(pcm16le) // 2
    a = max(0, min(total, int(start_sample)))
    b = max(0, min(total, int(end_sample)))
    if b < a:
        raise ValueError("Interval has inverted sample range")
    return pcm16le[a * 2 : b * 2]


def meets_min_keep(start_ms: float, end_ms: float, sample_count: int) -> tuple[bool, str]:
    """Legacy helper; unified 1s gate removed (049). Always keeps non-empty positive spans."""
    duration_ms = float(end_ms) - float(start_ms)
    if duration_ms <= 0 and sample_count <= 0:
        return False, f"nonpositive_duration_ms:{duration_ms}"
    return True, ""
