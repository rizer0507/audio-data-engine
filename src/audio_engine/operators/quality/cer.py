from __future__ import annotations

from typing import Any

from audio_engine.core.operator import BaseOperator, OperatorConfig
from audio_engine.core.registry import register_operator
from audio_engine.core.sample import Sample


def _levenshtein(a: str, b: str) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr.append(min(curr[-1] + 1, prev[j] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


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

        dist = _levenshtein(hyp, ref)
        cer = round(dist / max(len(ref), 1), 4)
        return {
            "quality": {"cer": cer, "cer_ref": ref_key, "cer_hyp": hyp_key},
            "lineage_entry": {
                "operator": self.full_name,
                "version": self.version,
                "params": dict(config.params),
            },
        }
