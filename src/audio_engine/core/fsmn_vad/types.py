"""FSMN VAD segment types and constants (fenshen / original-service aligned)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

SR = 16000
# Client packet size for deterministic file replay (matches compare script default).
PACKET_MS = 100
PACKET_SAMPLES = SR * PACKET_MS // 1000  # 1600
PACKET_BYTES = PACKET_SAMPLES * 2  # 3200
# Model generate() chunk_size argument (ms); original service always passes 200.
MODEL_CHUNK_MS = 200
# Legacy alias: inference merges two 100 ms packets → 200 ms of audio.
CHUNK_MS = MODEL_CHUNK_MS
CHUNK_SAMPLES = SR * CHUNK_MS // 1000  # 3200
BYTES_PER_MS = SR * 2 // 1000  # 32 for PCM16LE @ 16 kHz

# Branch short-utterance thresholds from vad_asr_core_new.process_vad_and_asr.
SPLIT_END_DROP_MAX_MS = 150  # drop when end - pending <= 150 (inclusive)
COMPLETE_PENDING_DROP_MAX_MS = 200  # drop when pending exists and end - pending <= 200
# Reported/dedup start for split-end is pending_start - this offset; PCM start is not shifted.
SPLIT_END_REPORT_LOOKBACK_MS = 100

# No fixed post-endpoint tail (049 replaces 675 ms keep). Kept for fingerprint clarity.
END_SILENCE_KEEP_MS = 0

# Processing fingerprint version — bump when pairing/clip/filter rules change.
VAD_PROCESSING_VERSION = "fsmn_vad_segment/v8"

END_REASON_VAD = "vad_endpoint"
END_REASON_EOF = "eof_fallback"

BRANCH_SPLIT_END = "split_end"
BRANCH_COMPLETE = "complete_endpoint"
BRANCH_EOF = "eof_fallback"

SOURCE_STATUS_COMPLETED = "completed"
SOURCE_STATUS_NO_SPEECH = "no_speech"
SOURCE_STATUS_ALL_SHORT = "all_short"
SOURCE_STATUS_EOF_WITHOUT_END = "eof_without_end"
SOURCE_STATUS_ENDPOINT_AT_EOF = "endpoint_at_eof"
SOURCE_STATUS_EMPTY = "empty_audio"
SOURCE_STATUS_DECODE_FAILED = "decode_failed"
SOURCE_STATUS_VAD_FAILED = "vad_failed"

# Deprecated unified 1s gate (041/045); retained only so imports do not break.
MIN_KEEP_MS = 0
MIN_KEEP_SAMPLES = 0
VAD_FRAME_LENGTH_MS = 25
VAD_FRAME_SHIFT_MS = 10


@dataclass(frozen=True)
class VadRawInterval:
    """Model interval with actual clip bounds before discard bookkeeping."""

    start_ms: float
    end_ms: float | None
    end_reason: str
    clip_branch: str = BRANCH_SPLIT_END
    boundary_clamped: bool = False
    start_sample: int = 0
    end_sample: int = 0
    reported_start_ms: float | None = None
    reported_end_ms: float | None = None


@dataclass(frozen=True)
class VadDiscarded:
    start_ms: float
    end_ms: float | None
    end_reason: str
    reason: str
    start_sample: int = 0
    end_sample: int = 0
    duration_ms: float = 0.0
    clip_branch: str = ""
    reported_start_ms: float | None = None
    reported_end_ms: float | None = None


@dataclass(frozen=True)
class VadKeptSegment:
    """Accepted clip; model timestamps and actual crop samples are recorded separately."""

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
    clip_branch: str = BRANCH_SPLIT_END
    reported_start_ms: float | None = None
    reported_end_ms: float | None = None
    buffer_end_sample: int = 0


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
