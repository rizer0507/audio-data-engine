"""Rewrite ASR transcript texts to plain characters only."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from audio_engine.core.operator import BaseOperator, OperatorConfig
from audio_engine.core.registry import register_operator
from audio_engine.core.sample import Sample
from audio_engine.core.transcript_reconcile import (
    resolve_blank_exact_hotwords,
    rewrite_plain_transcript_entry,
)


@register_operator
class NormalizeTranscriptsOperator(BaseOperator):
    """Strip control tags / emotion markers / punctuation from selected transcripts.

    Params:
      models: transcript keys to clean (default: all present keys)
      keep_raw: if true, stash original text under ``extra.raw_text`` when missing
      blank_exact_hotwords: phrases / vocabulary that blank a model when the whole
        plain text equals one entry (same stage as punctuation stripping)
      blank_exact_hotwords_path: optional YAML containing ``blank_exact_hotwords``
    """

    name = "normalize_transcripts"
    version = "1.1.0"
    category = "quality"

    def _execute(self, sample: Sample, config: OperatorConfig) -> dict[str, Any]:
        params = dict(config.params)
        path = params.get("blank_exact_hotwords_path")
        if path:
            loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
            if "blank_exact_hotwords" in loaded:
                params.setdefault("blank_exact_hotwords", loaded["blank_exact_hotwords"])
            else:
                params.setdefault("blank_exact_hotwords", loaded)

        models = params.get("models")
        if models is None:
            model_keys = list(sample.transcripts.keys())
        else:
            model_keys = [str(item) for item in models]
        keep_raw = bool(params.get("keep_raw", True))
        blank_hotwords, blank_models = resolve_blank_exact_hotwords(
            params.get("blank_exact_hotwords")
        )

        updated: dict[str, Any] = {}
        for model in model_keys:
            entry = sample.transcripts.get(model)
            if entry is None:
                continue
            updated[model] = rewrite_plain_transcript_entry(
                entry,
                keep_raw=keep_raw,
                blank_hotwords=blank_hotwords,
                blank_model=model in blank_models,
            )

        return {
            "transcripts": updated,
            "lineage_entry": {
                "operator": self.full_name,
                "version": self.version,
                "params": {
                    "models": model_keys,
                    "keep_raw": keep_raw,
                    "blank_exact_hotwords": sorted(blank_hotwords),
                    "blank_models": sorted(blank_models),
                },
            },
        }
