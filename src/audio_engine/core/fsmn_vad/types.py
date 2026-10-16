"""FSMN VAD segment types and constants."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

SR = 16000
CHUNK_MS = 200
CHUNK_SAMPLES = SR * CHUNK_MS // 1000  # 3200
MIN_KEEP_MS = 1000
MIN_KEEP_SAMPLES = SR * MIN_KEEP_MS // 1000  # 16000
# Post-endpoint clip extension (real audio only, clamped at EOF).
# Independent of model max_end_silence_time in fenshen_vad_config.yaml.
END_SILENCE_KEEP_MS = 675
VAD_FRAME_LENGTH_MS = 25
VAD_FRAME_SHIFT_MS = 10

# Processing fingerprint version — bump when pairing/clip/filter rules change.
VAD_PROCESSING_VERSION = "fsmn_vad_segment/v7"

END_REASON_VAD = "vad_endpoint"
END_REASON_EOF = "eof_fallback"

SOURCE_STATUS_COMPLETED = "completed"
SOURCE_STATUS_NO_SPEECH = "no_speech"
SOURCE_STATUS_ALL_SHORT = "all_short"
SOURCE_STATUS_EOF_WITHOUT_END = "eof_without_end"
SOURCE_STATUS_ENDPOINT_AT_EOF = "endpoint_at_eof"
SOURCE_STATUS_EMPTY = "empty_audio"
SOURCE_STATUS_DECODE_FAILED = "decode_failed"
SOURCE_STATUS_VAD_FAILED = "vad_failed"


@dataclass(frozen=True)
class VadRawInterval:
    """Model interval with retained clip bounds before short-utterance filter."""

    start_ms: float
    end_ms: float
    end_reason: str
    boundary_clamped: bool = False
    start_sample: int = 0
    end_sample: int = 0


@dataclass(frozen=True)
class VadDiscarded:
    start_ms: float
    end_ms: float
    end_reason: str
    reason: str
    start_sample: int = 0
    end_sample: int = 0
    duration_ms: float = 0.0


@dataclass(frozen=True)
class VadKeptSegment:
    """Accepted clip including end silence; end_ms preserves the model endpoint."""

    start_ms: float
    end_ms: float | None
    inferred_end_ms: float | None
    end_reason: str
    start_sample: int
    end_sample: int
    duration_ms: float
    boundary_clamped: bool
    pcm16le: bytes
    extra_padding_ms: float = 0.0


@dataclass
class VadStreamResult:
    """Per-source / per-channel VAD outcome."""

    candidates: list[VadRawInterval] = field(default_factory=list)
    kept: list[VadKeptSegment] = field(default_factory=list)
    discarded: list[VadDiscarded] = field(default_factory=list)
    eof_fallback: list[VadKeptSegment] = field(default_factory=list)
    event_log: list[dict[str, Any]] = field(default_factory=list)
    status: str = SOURCE_STATUS_COMPLETED
    error: str | None = None
