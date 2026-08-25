from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from audio_engine.core.operator import BaseOperator, OperatorConfig
from audio_engine.core.registry import register_operator
from audio_engine.core.sample import Sample
from audio_engine.metrics.runner import run_text_metrics


def _record(sample: Sample) -> dict[str, Any]:
    record = sample.to_flat_dict()
    record.update(sample.labels)
    record.update({f"{name}_text": sample.get_transcript_text(name) for name in sample.transcripts})
    return record


@register_operator
class TextMetricOperator(BaseOperator):
    name = "text_metrics"
    version = "1.0.0"
    category = "quality"

    def _execute(self, sample: Sample, config: OperatorConfig) -> dict[str, Any]:
        metric_path = Path(config.params["config_path"])
        profile_path = Path(
            config.params.get("normalization_path", "configs/normalization/zh_asr_v1.yaml")
        )
        metric_config = yaml.safe_load(metric_path.read_text(encoding="utf-8")) or {}
        normalization = yaml.safe_load(profile_path.read_text(encoding="utf-8")) or {}
        quality: dict[str, Any] = {}
        record = _record(sample)
        for comparison in metric_config.get("comparisons", []):
            values = run_text_metrics({**record, **quality}, comparison, normalization)
            quality.update(values)
            logger.info(
                "[MetricRunner] metric=cer purpose={} reference={} hypothesis={} "
                "normalizer={} output={} records=1",
                comparison.get("purpose"),
                comparison["reference"]["field"],
                comparison["hypothesis"]["field"],
                normalization.get("name"),
                next(iter(values)),
            )
        return {
            "quality": quality,
            "lineage_entry": {
                "operator": self.full_name,
                "version": self.version,
                "params": dict(config.params),
            },
        }
