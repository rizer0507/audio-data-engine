"""FSMN VAD segmentation: streaming adapter, pairing, clip, identity."""

from audio_engine.core.fsmn_vad.adapter import (
    MockVadModel,
    ScheduleVadModel,
    load_funasr_vad,
    pcm16le_to_float32,
    run_streaming_vad,
)
from audio_engine.core.fsmn_vad.clip import compute_clip_bounds, meets_min_keep, ms_to_sample
from audio_engine.core.fsmn_vad.identity import (
    build_segment_labels,
    is_segment_sample,
    parent_source_audio_id,
    segment_content_sha256,
    stable_segment_id,
)
from audio_engine.core.fsmn_vad.pairing import pair_and_filter
from audio_engine.core.fsmn_vad.runtime import dependency_versions, prepare_runtime_model_dir
from audio_engine.core.fsmn_vad.types import (
    CHUNK_MS,
    CHUNK_SAMPLES,
    MIN_KEEP_MS,
    MIN_KEEP_SAMPLES,
    SR,
    VAD_PROCESSING_VERSION,
)

__all__ = [
    "CHUNK_MS",
    "CHUNK_SAMPLES",
    "MIN_KEEP_MS",
    "MIN_KEEP_SAMPLES",
    "MockVadModel",
    "SR",
    "ScheduleVadModel",
    "VAD_PROCESSING_VERSION",
    "build_segment_labels",
    "compute_clip_bounds",
    "dependency_versions",
    "is_segment_sample",
    "load_funasr_vad",
    "meets_min_keep",
    "ms_to_sample",
    "pair_and_filter",
    "parent_source_audio_id",
    "pcm16le_to_float32",
    "prepare_runtime_model_dir",
    "run_streaming_vad",
    "segment_content_sha256",
    "stable_segment_id",
]
