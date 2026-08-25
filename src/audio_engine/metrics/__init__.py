"""Model-agnostic text metric primitives and runner."""

from audio_engine.metrics.cer import calculate_cer
from audio_engine.metrics.runner import MetricConfigError, run_text_metrics

__all__ = ["MetricConfigError", "calculate_cer", "run_text_metrics"]
