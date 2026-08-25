"""Rewrite ASR transcript texts to plain characters only."""

from __future__ import annotations

from typing import Any

from audio_engine.core.operator import BaseOperator, OperatorConfig
from audio_engine.core.registry import register_operator
from audio_engine.core.sample import Sample
from audio_engine.core.transcript_reconcile import plain_transcript_text


@register_operator
class NormalizeTranscriptsOperator(BaseOperator):
    """Strip control tags / emotion markers / punctuation from selected transcripts.

    Params:
      models: transcript keys to clean (default: all present keys)
      keep_raw: if true, stash original text under ``extra.raw_text`` when missing
    """

    name = "normalize_transcripts"
    version = "1.0.0"
    category = "quality"

    def _execute(self, sample: Sample, config: OperatorConfig) -> dict[str, Any]:
        models = config.params.get("models")
        if models is None:
            model_keys = list(sample.transcripts.keys())
        else:
            model_keys = [str(item) for item in models]
        keep_raw = bool(config.params.get("keep_raw", True))

        updated: dict[str, Any] = {}
        for model in model_keys:
            entry = sample.transcripts.get(model)
            if entry is None:
                continue
            if isinstance(entry, dict):
                original = str(entry.get("text") or "")
                cleaned = plain_transcript_text(original)
                new_entry = dict(entry)
                if keep_raw:
                    extra = dict(new_entry.get("extra") or {})
                    extra.setdefault("raw_text", original)
                    new_entry["extra"] = extra
                new_entry["text"] = cleaned
                updated[model] = new_entry
            else:
                original = str(entry)
                cleaned = plain_transcript_text(original)
                payload: dict[str, Any] = {"text": cleaned}
                if keep_raw:
                    payload["extra"] = {"raw_text": original}
                updated[model] = payload

        return {
            "transcripts": updated,
            "lineage_entry": {
                "operator": self.full_name,
                "version": self.version,
                "params": {
                    "models": model_keys,
                    "keep_raw": keep_raw,
                },
            },
        }
