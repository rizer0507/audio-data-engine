"""Model-agnostic text metric primitives and runner."""

from audio_engine.metrics.align import align_characters
from audio_engine.metrics.cer import calculate_cer
from audio_engine.metrics.normalization import normalize_text
from audio_engine.metrics.stat import MetricStat

# MetricConfigError lives with the sample-level runner; import path must stay light
# so transcript_reconcile → metrics.cer does not pull selection_v3.


class MetricConfigError(ValueError):
    pass


__all__ = [
    "MetricConfigError",
    "MetricStat",
    "align_characters",
    "calculate_cer",
    "normalize_text",
    "run_text_metrics",
]


def run_text_metrics(
    record: dict,
    comparison: dict,
    normalization: dict,
):
    from audio_engine.metrics.runner import run_text_metrics as _impl

    return _impl(record, comparison, normalization)


def __getattr__(name: str):
    if name in {"MetricRunner", "CORPUS_METRICS", "run_corpus_metrics"}:
        from audio_engine.metrics import runner as _runner

        return getattr(_runner, name)
    if name in {
        "BUSINESS_METRICS_VERSION",
        "aggregate_business_report",
        "compute_business_metrics_for_model",
    }:
        from audio_engine.metrics import business as _business

        return getattr(_business, name)
    if name in {
        "GATE_FAIL",
        "GATE_INCOMPLETE",
        "GATE_NEEDS_REVIEW",
        "GATE_PASS",
        "evaluate_release_gate",
    }:
        from audio_engine.metrics import gate as _gate

        return getattr(_gate, name)
    if name in {"JUDGE_VERSION", "load_semantic_judge"}:
        from audio_engine.metrics import semantic_judge as _judge

        return getattr(_judge, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
