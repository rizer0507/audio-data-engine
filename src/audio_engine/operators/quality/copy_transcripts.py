"""Copy or alias transcript keys without re-running ASR."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from audio_engine.core.operator import BaseOperator, OperatorConfig
from audio_engine.core.registry import register_operator
from audio_engine.core.sample import Sample


@register_operator
class CopyTranscriptsOperator(BaseOperator):
    """Copy transcript entries between keys (e.g. ``qwen`` → ``old_model``).

    Params:
      mapping: ``{source_key: dest_key}`` (required)
      only_if_missing: if true (default), skip when dest already has non-empty text
    """

    name = "copy_transcripts"
    version = "1.0.0"
    category = "quality"

    def compute_cache_key(self, sample: Sample, config: OperatorConfig) -> str:
        mapping = {str(k): str(v) for k, v in (config.params.get("mapping") or {}).items()}
        payload = {
            "id": sample.id,
            "sha256": sample.sha256,
            "operator": self.full_name,
            "version": self.version,
            "mapping": mapping,
            "only_if_missing": bool(config.params.get("only_if_missing", True)),
            "sources": {src: sample.get_transcript_text(src) for src in mapping},
            "dests": {dst: sample.get_transcript_text(dst) for dst in mapping.values()},
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode()).hexdigest()

    def _execute(self, sample: Sample, config: OperatorConfig) -> dict[str, Any]:
        mapping = config.params.get("mapping") or {}
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError("copy_transcripts requires params.mapping as {src: dst}")
        only_if_missing = bool(config.params.get("only_if_missing", True))
        updated: dict[str, Any] = {}
        for src, dst in mapping.items():
            src_key = str(src)
            dst_key = str(dst)
            entry = sample.transcripts.get(src_key)
            if entry is None:
                continue
            if only_if_missing:
                existing = sample.transcripts.get(dst_key)
                if isinstance(existing, dict) and str(existing.get("text") or "").strip():
                    continue
                if existing is not None and not isinstance(existing, dict) and str(existing).strip():
                    continue
            if isinstance(entry, dict):
                updated[dst_key] = dict(entry)
            else:
                updated[dst_key] = {"text": str(entry)}
        return {
            "transcripts": updated,
            "lineage_entry": {
                "operator": self.full_name,
                "version": self.version,
                "params": {
                    "mapping": {str(k): str(v) for k, v in mapping.items()},
                    "only_if_missing": only_if_missing,
                },
            },
        }
