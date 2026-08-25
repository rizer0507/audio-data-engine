from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from audio_engine.core.operator import BaseOperator, OperatorConfig
from audio_engine.core.registry import register_operator
from audio_engine.core.sample import Sample
from audio_engine.metrics.cer import calculate_cer
from audio_engine.metrics.normalization import normalize_text

_DEFAULT_NORM = Path("configs/normalization/zh_asr_v1.yaml")


def _load_normalization(path: str | Path | None) -> dict[str, Any]:
    profile = Path(path) if path else _DEFAULT_NORM
    if profile.is_file():
        return yaml.safe_load(profile.read_text(encoding="utf-8")) or {}
    return {
        "unicode": {"normalize": True, "form": "NFKC"},
        "punctuation": {"remove": True},
        "whitespace": {"remove": True},
        "english": {"lowercase": True},
        "filler": {"remove": False},
    }


@register_operator
class CerOperator(BaseOperator):
    """Deprecated thin wrapper; prefer ``quality.text_metrics`` + MetricRunner."""

    name = "cer"
    version = "1.1.0"
    category = "quality"

    def _execute(self, sample: Sample, config: OperatorConfig) -> dict[str, Any]:
        hyp_key = config.params.get("hypothesis", "qwen")
        ref_key = config.params.get("reference", "gold_text")
        normalization = _load_normalization(config.params.get("normalization_path"))

        hyp = normalize_text(sample.get_transcript_text(hyp_key), normalization)
        ref = sample.labels.get(ref_key, "")
        if not ref:
            ref = sample.get_transcript_text(ref_key)
        ref = normalize_text(ref, normalization)

        metric = calculate_cer(ref, hyp)
        return {
            "quality": {
                "cer": metric["cer"],
                "cer_ref": ref_key,
                "cer_hyp": hyp_key,
                "cer_substitutions": metric["substitutions"],
                "cer_deletions": metric["deletions"],
                "cer_insertions": metric["insertions"],
            },
            "lineage_entry": {
                "operator": self.full_name,
                "version": self.version,
                "params": dict(config.params),
            },
        }
