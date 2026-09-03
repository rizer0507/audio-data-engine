from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from audio_engine.core.operator import BaseOperator, OperatorConfig
from audio_engine.core.registry import register_operator
from audio_engine.core.sample import Sample
from audio_engine.metrics.runner import MetricConfigError, run_text_metrics


def _record(sample: Sample) -> dict[str, Any]:
    record = sample.to_flat_dict()
    record.update(sample.labels)
    record.update({f"{name}_text": sample.get_transcript_text(name) for name in sample.transcripts})
    return record


def _field_text(record: dict[str, Any], field: str | None) -> str:
    if not field:
        return ""
    value = record.get(field)
    if value is None:
        return ""
    try:
        if value != value:  # NaN
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _expand_vs_base_comparisons(
    vs_base: dict[str, Any],
    sample: Sample,
    *,
    agreement_base_override: str | None = None,
) -> tuple[list[dict[str, Any]], str, list[str]]:
    """Build per-hypothesis comparisons against a base transcript key."""
    base = str(agreement_base_override or vs_base.get("base") or "").strip()
    if not base:
        raise MetricConfigError("vs_base.base is required (or pass agreement_base param)")
    purpose = vs_base.get("purpose", "gold_generation_agreement")
    metrics = list(vs_base.get("metrics") or ["cer"])
    overwrite = bool(vs_base.get("overwrite", False))
    raw_hyps = vs_base.get("hypotheses")
    if raw_hyps is None:
        hypotheses = [key for key in sample.transcripts.keys() if key != base]
    else:
        hypotheses = [str(item).strip() for item in raw_hyps if str(item).strip()]
        hypotheses = [key for key in hypotheses if key != base]
    comparisons: list[dict[str, Any]] = []
    for hyp in hypotheses:
        comparisons.append(
            {
                "name": f"{hyp}_vs_{base}",
                "purpose": purpose,
                "reference": {"field": f"{base}_text"},
                "hypothesis": {"field": f"{hyp}_text"},
                "metrics": metrics,
                "overwrite": overwrite,
                "output": {"prefix": f"{hyp}_vs_{base}_agreement"},
            }
        )
    return comparisons, base, hypotheses


def _expand_vs_gold_comparisons(
    vs_gold: dict[str, Any],
    sample: Sample,
    *,
    hypotheses_override: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Build per-hypothesis comparisons against labels.gold_text (or gold transcript)."""
    reference_field = str(vs_gold.get("reference_field") or "gold_text").strip() or "gold_text"
    purpose = vs_gold.get("purpose", "model_evaluation")
    metrics = list(vs_gold.get("metrics") or ["cer"])
    overwrite = bool(vs_gold.get("overwrite", False))
    raw_hyps = hypotheses_override if hypotheses_override is not None else vs_gold.get("hypotheses")
    if raw_hyps is None:
        hypotheses = [key for key in sample.transcripts.keys() if key != "gold"]
    else:
        hypotheses = [str(item).strip() for item in raw_hyps if str(item).strip()]
        hypotheses = [key for key in hypotheses if key != "gold"]
    comparisons: list[dict[str, Any]] = []
    for hyp in hypotheses:
        comparisons.append(
            {
                "name": f"{hyp}_vs_gold",
                "purpose": purpose,
                "reference": {"field": reference_field},
                "hypothesis": {"field": f"{hyp}_text"},
                "metrics": metrics,
                "overwrite": overwrite,
                "output": {"prefix": hyp},
            }
        )
    return comparisons, hypotheses


@register_operator
class TextMetricOperator(BaseOperator):
    name = "text_metrics"
    version = "1.3.0"
    category = "quality"

    def _execute(self, sample: Sample, config: OperatorConfig) -> dict[str, Any]:
        metric_path = Path(config.params["config_path"])
        profile_path = Path(
            config.params.get("normalization_path", "configs/normalization/zh_asr_v1.yaml")
        )
        metric_config = yaml.safe_load(metric_path.read_text(encoding="utf-8")) or {}
        normalization = yaml.safe_load(profile_path.read_text(encoding="utf-8")) or {}
        skip_missing_reference = bool(config.params.get("skip_missing_reference", False))
        agreement_base_override = config.params.get("agreement_base")
        if agreement_base_override is not None:
            agreement_base_override = str(agreement_base_override).strip() or None
        eval_hypotheses = config.params.get("eval_hypotheses")
        if eval_hypotheses is not None:
            eval_hypotheses = [str(item).strip() for item in eval_hypotheses if str(item).strip()]

        quality: dict[str, Any] = {}
        record = _record(sample)
        skipped_no_gold = False
        if not _field_text(record, "gold_text"):
            fallback = _field_text(record, "label") or sample.get_transcript_text("gold")
            if fallback:
                record["gold_text"] = fallback

        comparisons: list[dict[str, Any]] = []
        vs_base_count = 0
        vs_base = metric_config.get("vs_base")
        if vs_base:
            expanded, base, hypotheses = _expand_vs_base_comparisons(
                vs_base,
                sample,
                agreement_base_override=agreement_base_override,
            )
            comparisons.extend(expanded)
            vs_base_count = len(expanded)
            quality["vs_base_hypothesis_count"] = len(hypotheses)
            quality["vs_base_reference"] = base
            if base not in sample.transcripts and f"{base}_text" not in record:
                record[f"{base}_text"] = ""
            for hyp in hypotheses:
                if f"{hyp}_text" not in record:
                    record[f"{hyp}_text"] = sample.get_transcript_text(hyp)
        vs_gold = metric_config.get("vs_gold")
        if vs_gold:
            gold_expanded, gold_hyps = _expand_vs_gold_comparisons(
                vs_gold,
                sample,
                hypotheses_override=eval_hypotheses,
            )
            comparisons.extend(gold_expanded)
            quality["vs_gold_hypothesis_count"] = len(gold_hyps)
            quality["vs_gold_hypotheses"] = gold_hyps
            for hyp in gold_hyps:
                if f"{hyp}_text" not in record:
                    record[f"{hyp}_text"] = sample.get_transcript_text(hyp)
        comparisons.extend(list(metric_config.get("comparisons") or []))

        cer_values: list[float] = []
        for index, comparison in enumerate(comparisons):
            ref_field = comparison.get("reference", {}).get("field")
            prefix = comparison.get("output", {}).get("prefix") or "metric"
            if skip_missing_reference and not _field_text(record, ref_field):
                skipped_no_gold = True
                quality[f"{prefix}_skipped_no_gold"] = True
                logger.info(
                    "[MetricRunner] skip missing reference sample={} field={} prefix={}",
                    sample.id,
                    ref_field,
                    prefix,
                )
                continue
            try:
                values = run_text_metrics({**record, **quality}, comparison, normalization)
            except MetricConfigError:
                if skip_missing_reference and ref_field and ref_field not in record:
                    skipped_no_gold = True
                    quality[f"{prefix}_skipped_no_gold"] = True
                    continue
                raise
            quality.update(values)
            cer_key = f"{prefix}_cer"
            if (
                index < vs_base_count
                and cer_key in values
                and isinstance(values[cer_key], (int, float))
            ):
                cer_values.append(float(values[cer_key]))
            logger.info(
                "[MetricRunner] metric=cer purpose={} reference={} hypothesis={} "
                "normalizer={} output={} records=1",
                comparison.get("purpose"),
                comparison["reference"]["field"],
                comparison["hypothesis"]["field"],
                normalization.get("name"),
                next(iter(values)),
            )

        if vs_base:
            if cer_values:
                quality["max_vs_base_agreement_cer"] = max(cer_values)
            else:
                quality["max_vs_base_agreement_cer"] = None
                quality.setdefault("vs_base_hypothesis_count", 0)

        if skipped_no_gold:
            quality["eval_skipped_no_gold"] = True
        return {
            "quality": quality,
            "lineage_entry": {
                "operator": self.full_name,
                "version": self.version,
                "params": dict(config.params),
            },
        }
