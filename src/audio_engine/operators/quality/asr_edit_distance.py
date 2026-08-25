"""Compare non-baseline ASR transcripts against a configurable baseline model."""

from __future__ import annotations

import json
from typing import Any

from audio_engine.core.operator import BaseOperator, OperatorConfig
from audio_engine.core.registry import register_operator
from audio_engine.core.sample import Sample
from audio_engine.core.transcript_reconcile import levenshtein_ops


@register_operator
class AsrEditDistanceOperator(BaseOperator):
    """Compute edit distance (总/错字/少字/多字) vs a baseline transcript key.

    Params:
      baseline: transcript model key used as reference (default: qwen)
      compare_models: list of transcript keys to compare; default = all except baseline
      quality_key: prefix for stored quality fields (default: asr_edit)
    """

    name = "asr_edit_distance"
    version = "1.0.0"
    category = "quality"

    def _execute(self, sample: Sample, config: OperatorConfig) -> dict[str, Any]:
        baseline = str(config.params.get("baseline", "qwen"))
        quality_key = str(config.params.get("quality_key", "asr_edit"))
        compare = config.params.get("compare_models")
        if compare is None:
            compare_models = [key for key in sample.transcripts if key != baseline]
        else:
            compare_models = [str(item) for item in compare]

        baseline_text = sample.get_transcript_text(baseline)
        per_model: dict[str, Any] = {}
        flat: dict[str, Any] = {f"{quality_key}_baseline": baseline}

        for model in compare_models:
            if model == baseline:
                continue
            hyp = sample.get_transcript_text(model)
            prefix = f"{quality_key}_{model}"
            if not baseline_text and not hyp:
                ops = {
                    "total": None,
                    "错字": None,
                    "少字": None,
                    "多字": None,
                    "sub": None,
                    "delete": None,
                    "insert": None,
                    "cer": None,
                    "reason": "baseline_and_hypothesis_empty",
                }
            elif not baseline_text:
                ops = {
                    "total": None,
                    "错字": None,
                    "少字": None,
                    "多字": None,
                    "sub": None,
                    "delete": None,
                    "insert": None,
                    "cer": None,
                    "reason": "baseline_empty",
                }
            else:
                ops = levenshtein_ops(baseline_text, hyp)
            per_model[model] = ops
            flat[f"{prefix}_total"] = ops.get("total")
            flat[f"{prefix}_错字"] = ops.get("错字")
            flat[f"{prefix}_少字"] = ops.get("少字")
            flat[f"{prefix}_多字"] = ops.get("多字")
            flat[f"{prefix}_cer"] = ops.get("cer")
            if ops.get("reason"):
                flat[f"{prefix}_reason"] = ops["reason"]

        # JSON string keeps a nested payload without breaking parquet object columns.
        flat[f"{quality_key}_json"] = json.dumps(
            {"baseline": baseline, "models": per_model},
            ensure_ascii=False,
        )

        return {
            "quality": flat,
            "lineage_entry": {
                "operator": self.full_name,
                "version": self.version,
                "params": {
                    "baseline": baseline,
                    "compare_models": compare_models,
                    "quality_key": quality_key,
                },
            },
        }
