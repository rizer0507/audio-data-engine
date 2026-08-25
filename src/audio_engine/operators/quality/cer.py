from __future__ import annotations

from typing import Any

from audio_engine.core.operator import BaseOperator, OperatorConfig
from audio_engine.core.registry import register_operator
from audio_engine.core.sample import Sample
from audio_engine.metrics.cer import calculate_cer


@register_operator
class CerOperator(BaseOperator):
    name = "cer"
    version = "1.0.0"
    category = "quality"

    def _execute(self, sample: Sample, config: OperatorConfig) -> dict[str, Any]:
        hyp_key = config.params.get("hypothesis", "qwen")
        ref_key = config.params.get("reference", "gold_text")

        hyp = sample.get_transcript_text(hyp_key)
        ref = sample.labels.get(ref_key, "")
        if not ref:
            ref = sample.get_transcript_text(ref_key)

        if not ref:
            return {"quality": {"cer": None, "cer_ref": ref_key, "cer_hyp": hyp_key}}

        cer = calculate_cer(ref, hyp)["cer"]
        return {
            "quality": {"cer": cer, "cer_ref": ref_key, "cer_hyp": hyp_key},
            "lineage_entry": {
                "operator": self.full_name,
                "version": self.version,
                "params": dict(config.params),
            },
        }
