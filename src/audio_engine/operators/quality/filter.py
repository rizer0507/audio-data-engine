from __future__ import annotations

from typing import Any

from audio_engine.core.operator import BaseOperator, OperatorConfig
from audio_engine.core.registry import register_operator
from audio_engine.core.sample import Sample


@register_operator
class FilterOperator(BaseOperator):
    """Apply a pandas query filter and mark samples as kept/dropped."""

    name = "filter"
    version = "1.0.0"
    category = "quality"

    def _execute(self, sample: Sample, config: OperatorConfig) -> dict[str, Any]:
        expr = config.params.get("expr", "")
        label_key = config.params.get("label_key", "filter_pass")

        if not expr:
            return {"labels": {label_key: True}}

        row = sample.to_flat_dict()
        try:
            import pandas as pd

            passed = bool(pd.DataFrame([row]).query(expr, engine="python").shape[0])
        except Exception as exc:
            return {"labels": {label_key: False, f"{label_key}_error": str(exc)}}

        return {
            "labels": {label_key: passed},
            "lineage_entry": {
                "operator": self.full_name,
                "version": self.version,
                "params": {"expr": expr},
            },
        }


@register_operator
class TranscriptDiffOperator(BaseOperator):
    name = "transcript_diff"
    version = "1.0.0"
    category = "quality"

    def _execute(self, sample: Sample, config: OperatorConfig) -> dict[str, Any]:
        model_a = config.params.get("model_a", "asr.qwen")
        model_b = config.params.get("model_b", "asr.sensevoice")
        key_a = model_a.split(".")[-1] if "." in model_a else model_a
        key_b = model_b.split(".")[-1] if "." in model_b else model_b

        text_a = sample.get_transcript_text(key_a)
        text_b = sample.get_transcript_text(key_b)
        match = text_a == text_b

        return {
            "quality": {
                "transcript_match": match,
                f"{key_a}_text": text_a,
                f"{key_b}_text": text_b,
            },
            "labels": {"transcript_mismatch": not match},
            "lineage_entry": {
                "operator": self.full_name,
                "version": self.version,
                "params": {"model_a": key_a, "model_b": key_b},
            },
        }
