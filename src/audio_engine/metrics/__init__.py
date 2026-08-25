"""Model-agnostic text metric primitives and runner."""

from audio_engine.metrics.align import align_characters
from audio_engine.metrics.cer import calculate_cer
from audio_engine.metrics.normalization import normalize_text
from audio_engine.metrics.runner import MetricConfigError, run_text_metrics

__all__ = [
    "MetricConfigError",
    "align_characters",
    "calculate_cer",
    "normalize_text",
    "run_text_metrics",
]
