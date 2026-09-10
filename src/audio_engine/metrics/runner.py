from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import yaml

from audio_engine.core.sample import Sample
from audio_engine.metrics import MetricConfigError
from audio_engine.metrics.cer import calculate_cer
from audio_engine.metrics.normalization import normalize_text
from audio_engine.metrics.stat import MetricStat

# Re-export for callers that import from metrics.runner
__all__ = [
    "CORPUS_METRICS",
    "MetricConfigError",
    "MetricRunner",
    "SAMPLE_METRICS",
    "load_business_risk_config",
    "metric_stat",
    "run_corpus_metrics",
    "run_text_metrics",
]

SAMPLE_METRICS = frozenset({"cer"})
CORPUS_METRICS = frozenset(
    {
        "cer",
        "nfr",
        "far",
        "nahr",
        "nahr_silence",
        "faer",
        "nshr",
        "positive_retention",
        "inference_failure_rate",
        "semantic_unk_rate",
        "far_unk_as_risk_upper",
    }
)


def run_text_metrics(
    record: dict[str, Any], comparison: dict[str, Any], normalization: dict[str, Any]
) -> dict[str, Any]:
    """Run one configured sample-level comparison against a flat record.

    Currently supports CER. Business risk rates are corpus-level and must go
    through :func:`run_corpus_metrics` / :class:`MetricRunner`.
    """
    ref_field = comparison.get("reference", {}).get("field")
    hyp_field = comparison.get("hypothesis", {}).get("field")
    for role, field in (("reference", ref_field), ("hypothesis", hyp_field)):
        if not field or field not in record:
            raise MetricConfigError(f"{role} field `{field}` not found in dataset")
    metrics = comparison.get("metrics", ["cer"])
    unsupported = set(metrics) - SAMPLE_METRICS
    if unsupported:
        raise MetricConfigError(
            f"unsupported sample-level metrics: {sorted(unsupported)}; "
            f"corpus metrics {sorted(CORPUS_METRICS - SAMPLE_METRICS)} "
            "require MetricRunner.run_corpus / run_corpus_metrics"
        )
    prefix = comparison.get("output", {}).get("prefix")
    if not prefix:
        raise MetricConfigError("comparison.output.prefix is required")
    output = calculate_cer(
        normalize_text(record[ref_field], normalization),
        normalize_text(record[hyp_field], normalization),
    )
    result = {f"{prefix}_cer": output.pop("cer")}
    result.update({f"{prefix}_{key}": value for key, value in output.items()})
    collisions = result.keys() & record.keys()
    if collisions and not comparison.get("overwrite", False):
        raise MetricConfigError(f"output fields already exist: {sorted(collisions)}")
    return result


def load_business_risk_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    file_path = Path(path)
    if not file_path.is_file():
        raise MetricConfigError(f"business metrics config not found: {path}")
    raw = yaml.safe_load(file_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise MetricConfigError(f"business metrics config must be a mapping: {path}")
    return raw


def run_corpus_metrics(
    samples: Sequence[Sample],
    models: Sequence[str],
    *,
    business_config: dict[str, Any] | Any = None,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Unified corpus-level MetricRunner entry (CER speech + business risk)."""
    from audio_engine.metrics.business import (
        BusinessMetricConfig,
        aggregate_business_report,
        load_business_metric_config,
    )

    if isinstance(business_config, BusinessMetricConfig):
        cfg = business_config
        raw: dict[str, Any] = {}
    else:
        raw = dict(business_config or {})
        if config_path is not None:
            loaded = load_business_risk_config(config_path)
            merged = dict(loaded)
            merged.update(raw)
            raw = merged
        cfg = load_business_metric_config(raw.get("business") or raw)
    return aggregate_business_report(samples, models, config=cfg)


class MetricRunner:
    """Single entry for sample CER and corpus business metrics / release gates."""

    def __init__(
        self,
        *,
        business_config_path: str | Path | None = None,
        business_config: dict[str, Any] | None = None,
    ) -> None:
        from audio_engine.metrics.business import load_business_metric_config
        from audio_engine.metrics.gate import load_gate_config
        from audio_engine.metrics.semantic_judge import load_semantic_judge

        raw = dict(business_config or {})
        if business_config_path is not None:
            loaded = load_business_risk_config(business_config_path)
            merged = dict(loaded)
            merged.update(raw)
            raw = merged
        self.raw_config = raw
        self.business = load_business_metric_config(raw.get("business") or raw)
        self.gate = load_gate_config(raw.get("gate") or {})
        self.judge = load_semantic_judge(
            lexicon_path=self.business.lexicon_path,
            judge_version=self.business.judge_version,
        )

    def score_sample(
        self,
        record: dict[str, Any],
        comparison: dict[str, Any],
        normalization: dict[str, Any],
    ) -> dict[str, Any]:
        return run_text_metrics(record, comparison, normalization)

    def score_corpus(
        self, samples: Sequence[Sample], models: Sequence[str]
    ) -> dict[str, Any]:
        from audio_engine.metrics.business import aggregate_business_report

        return aggregate_business_report(
            samples, models, config=self.business, judge=self.judge
        )

    def evaluate_gate(
        self,
        samples: list[Sample],
        *,
        baseline: str,
        candidate: str,
        gate: Any = None,
    ) -> dict[str, Any]:
        from audio_engine.metrics.gate import evaluate_release_gate

        result = evaluate_release_gate(
            list(samples),
            baseline=baseline,
            candidate=candidate,
            gate=gate or self.gate,
            business=self.business,
            judge=self.judge,
        )
        return result.to_dict()


def metric_stat(
    name: str,
    *,
    numerator: float | int,
    denominator: int,
    eligible_count: int | None = None,
    excluded_count: int = 0,
    extras: dict[str, Any] | None = None,
) -> MetricStat:
    """Helper used by tests / callers to build the unified metric shape."""
    return MetricStat.from_counts(
        name,
        numerator=numerator,
        denominator=denominator,
        eligible_count=eligible_count,
        excluded_count=excluded_count,
        extras=extras,
    )
