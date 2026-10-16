"""Rewrite ASR transcript texts to plain characters only."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from audio_engine.core.operator import BaseOperator, OperatorConfig
from audio_engine.core.registry import register_operator
from audio_engine.core.sample import Sample
from audio_engine.core.transcript_reconcile import (
    blank_applies_to_model,
    resolve_blank_exact_hotwords,
    rewrite_plain_transcript_entry,
)

_DEFAULT_NORMALIZATION_PATH = "configs/normalization/zh_asr_v1.yaml"


def _file_sha256(path: Path) -> str:
    if not path.is_file():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_blank_exact_into_params(params: dict[str, Any]) -> None:
    path = params.get("blank_exact_hotwords_path")
    if not path:
        return
    loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if "blank_exact_hotwords" in loaded:
        params.setdefault("blank_exact_hotwords", loaded["blank_exact_hotwords"])
    else:
        params.setdefault("blank_exact_hotwords", loaded)


@register_operator
class NormalizeTranscriptsOperator(BaseOperator):
    """Strip control tags / emotion markers / punctuation from selected transcripts.

    Params:
      models: transcript keys to clean (default: all present keys)
      keep_raw: if true, stash original text under ``extra.raw_text`` when missing
      normalization_path: zh_asr profile path (fingerprint; cleaning uses the same
        profile as ``plain_transcript_text`` / Metric Engine)
      blank_exact_hotwords: phrases / vocabulary that blank a model when the whole
        plain text equals one entry (same stage as punctuation stripping)
      blank_exact_hotwords_path: optional YAML containing ``blank_exact_hotwords``
    """

    name = "normalize_transcripts"
    version = "1.2.0"
    category = "quality"

    def dependency_fingerprint(self, config: OperatorConfig) -> dict[str, Any] | None:
        """Invalidate step checkpoints when normalization / blank YAML content changes."""
        params = dict(config.params)
        norm_path = Path(
            params.get("normalization_path") or _DEFAULT_NORMALIZATION_PATH
        )
        deps: dict[str, Any] = {
            "normalization_path": str(norm_path),
            "normalization_sha256": _file_sha256(norm_path),
            "keep_raw": bool(params.get("keep_raw", True)),
            "operator_version": self.version,
        }
        blank_path = params.get("blank_exact_hotwords_path")
        if blank_path:
            path = Path(blank_path)
            deps["blank_exact_hotwords_path"] = str(path)
            deps["blank_exact_hotwords_sha256"] = _file_sha256(path)
        return deps

    def compute_cache_key(self, sample: Sample, config: OperatorConfig) -> str:
        """Invalidate when transcript keys/texts or blank/normalization file content change."""
        params = dict(config.params)
        blank_path = params.get("blank_exact_hotwords_path")
        blank_fingerprint = ""
        if blank_path:
            blank_fingerprint = _file_sha256(Path(blank_path))
        norm_path = Path(
            params.get("normalization_path") or _DEFAULT_NORMALIZATION_PATH
        )
        norm_fingerprint = _file_sha256(norm_path)
        model_keys = params.get("models")
        if model_keys is None:
            model_keys = sorted(sample.transcripts.keys())
        else:
            model_keys = [str(item) for item in model_keys]
        texts = {
            key: sample.get_transcript_text(key)
            for key in model_keys
            if key in sample.transcripts
        }
        payload = {
            "sha256": sample.sha256,
            "operator": self.full_name,
            "version": self.version,
            "params": params,
            "blank_file_sha256": blank_fingerprint,
            "normalization_sha256": norm_fingerprint,
            "model_keys": model_keys,
            "texts": texts,
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode()).hexdigest()

    def _execute(self, sample: Sample, config: OperatorConfig) -> dict[str, Any]:
        params = dict(config.params)
        _load_blank_exact_into_params(params)

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
                blank_model=blank_applies_to_model(model, blank_models),
            )

        return {
            "transcripts": updated,
            "lineage_entry": {
                "operator": self.full_name,
                "version": self.version,
                "params": {
                    "models": model_keys,
                    "keep_raw": keep_raw,
                    "normalization_path": params.get(
                        "normalization_path", _DEFAULT_NORMALIZATION_PATH
                    ),
                    "blank_exact_hotwords": sorted(blank_hotwords),
                    "blank_models": sorted(blank_models),
                },
            },
        }
