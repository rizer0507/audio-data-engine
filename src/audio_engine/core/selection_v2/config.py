"""Configuration loader for selection_v2.0."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from audio_engine.core.selection_engine import DEFAULT_MODEL_FAMILIES
from audio_engine.core.selection_v2.types import RULE_VERSION


@dataclass
class SelectionV2Config:
    strict_threshold: float = 0.95
    consensus_threshold: float = 0.90
    dominant_ratio: float = 0.75
    empty_ratio_for_hallucination: float = 0.75
    min_family_count: int = 2
    short_audio_sec: float = 2.0
    short_text_chars: int = 6
    max_speech_ratio_silence: float = 0.05
    semantic_inversion_auto_accept: bool = False
    critical_token_conflict_auto_accept: bool = False
    vad_miss_auto_empty: bool = False
    model_families: dict[str, list[str]] = field(
        default_factory=lambda: dict(DEFAULT_MODEL_FAMILIES)
    )
    negative_phrases: list[str] = field(default_factory=list)
    positive_phrases: list[str] = field(default_factory=list)
    critical_tokens: list[str] = field(default_factory=list)
    profanity_or_reject: list[str] = field(default_factory=list)
    primary_family: str = "qwen"
    secondary_family: str = "sensevoice"
    rule_version: str = RULE_VERSION
    write_v1_subtype: bool = True

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> SelectionV2Config:
        similarity = params.get("similarity") or {}
        short = params.get("short_utterance") or {}
        silence = params.get("silence") or {}
        consensus = params.get("consensus") or {}
        risk = params.get("risk_gate") or {}
        compat = params.get("compat") or {}
        families = params.get("model_families") or DEFAULT_MODEL_FAMILIES

        negative: list[str] = []
        positive: list[str] = []
        critical: list[str] = []
        reject: list[str] = []

        # Inline semantic block (optional override)
        semantic = params.get("semantic") or {}
        if semantic:
            negative = [str(x) for x in (semantic.get("negative") or []) if str(x).strip()]
            positive = [str(x) for x in (semantic.get("positive") or []) if str(x).strip()]
            critical = [str(x) for x in (semantic.get("critical_tokens") or []) if str(x).strip()]
            reject = [
                str(x) for x in (semantic.get("profanity_or_reject") or []) if str(x).strip()
            ]

        lexicon_path = params.get("semantic_lexicon_path")
        if lexicon_path:
            loaded = _load_lexicon(lexicon_path)
            if not negative:
                negative = loaded.get("negative") or []
            if not positive:
                positive = loaded.get("positive") or []
            if not critical:
                critical = loaded.get("critical_tokens") or []
            if not reject:
                reject = loaded.get("profanity_or_reject") or []

        return cls(
            strict_threshold=float(
                similarity.get("strict_threshold", params.get("strict_threshold", 0.95))
            ),
            consensus_threshold=float(
                similarity.get(
                    "consensus_threshold", params.get("consensus_threshold", 0.90)
                )
            ),
            dominant_ratio=float(
                similarity.get(
                    "dominant_cluster_ratio",
                    params.get("dominant_ratio", 0.75),
                )
            ),
            empty_ratio_for_hallucination=float(
                params.get("empty_ratio_for_hallucination", 0.75)
            ),
            min_family_count=int(consensus.get("min_family_count", 2)),
            short_audio_sec=float(short.get("max_audio_sec", 2.0)),
            short_text_chars=int(short.get("max_text_chars", 6)),
            max_speech_ratio_silence=float(silence.get("max_speech_ratio", 0.05)),
            semantic_inversion_auto_accept=bool(
                risk.get("semantic_inversion_auto_accept", False)
            ),
            critical_token_conflict_auto_accept=bool(
                risk.get("critical_token_conflict_auto_accept", False)
            ),
            vad_miss_auto_empty=bool(risk.get("vad_miss_auto_empty", False)),
            model_families={str(k): [str(x) for x in (v or [])] for k, v in families.items()},
            negative_phrases=negative,
            positive_phrases=positive,
            critical_tokens=critical,
            profanity_or_reject=reject,
            primary_family=str(params.get("primary_family") or "qwen"),
            secondary_family=str(params.get("secondary_family") or "sensevoice"),
            rule_version=str(params.get("rule_version") or RULE_VERSION),
            write_v1_subtype=bool(compat.get("write_v1_subtype", True)),
        )


def _load_lexicon(path: str | Path) -> dict[str, list[str]]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    result: dict[str, list[str]] = {}
    for key in ("negative", "positive", "critical_tokens", "profanity_or_reject"):
        result[key] = [str(x) for x in (raw.get(key) or []) if str(x).strip()]
    return result
