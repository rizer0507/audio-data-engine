"""Versioned semantic polarity judge for business metrics (not selection engine)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from audio_engine.core.selection_v3.semantic_risk import (
    LexiconPatterns,
    compile_lexicon,
    polarity_of_text,
)
from audio_engine.core.selection_v3.types import (
    POLARITY_MIXED,
    POLARITY_NEGATIVE,
    POLARITY_NEUTRAL,
    POLARITY_POSITIVE,
    POLARITY_UNKNOWN,
)
from audio_engine.core.selection_v3.config import SelectionV3Config


JUDGE_VERSION = "business_semantic_judge_v1.0"


@dataclass(frozen=True)
class SemanticJudge:
    """Observe-only polarity; mixed/unknown never count as confirmed positive."""

    version: str
    lexicon_path: str
    patterns: LexiconPatterns

    def polarity(self, text: str) -> str:
        return polarity_of_text(str(text or ""), self.patterns)

    def is_confirmed_positive(self, text: str) -> bool:
        return self.polarity(text) == POLARITY_POSITIVE

    def is_pure_filler(self, text: str) -> bool:
        value = str(text or "").strip()
        if not value:
            return False
        if not (self.patterns.filler and self.patterns.filler.fullmatch(value)):
            return False
        # 「嗯，我不需要」等不得当纯语气词（非 fullmatch 已排除；再拦含否定）
        if self.patterns.negative and self.patterns.negative.search(value):
            return False
        if self.patterns.positive and self.patterns.positive.search(value):
            return False
        return True

    def to_meta(self) -> dict[str, Any]:
        return {
            "judge_version": self.version,
            "lexicon_path": self.lexicon_path,
        }


def load_semantic_judge(
    *,
    lexicon_path: str | Path | None = None,
    judge_version: str = JUDGE_VERSION,
) -> SemanticJudge:
    path = Path(
        lexicon_path or "configs/selection/semantic_lexicon_zh_v3.yaml"
    ).as_posix()
    # Reuse SelectionV3Config lexicon loading without requiring full selection YAML.
    raw: dict[str, Any] = {}
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"semantic lexicon missing: {path}")
    if file_path.is_file():
        loaded = yaml.safe_load(file_path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"semantic lexicon must be a mapping: {path}")
        raw = loaded
    cfg = SelectionV3Config(
        negative_phrases=[str(x) for x in (raw.get("negative") or []) if str(x).strip()],
        positive_phrases=[str(x) for x in (raw.get("positive") or []) if str(x).strip()],
        critical_tokens=[
            str(x) for x in (raw.get("critical_tokens") or []) if str(x).strip()
        ],
        profanity_or_reject=[
            str(x) for x in (raw.get("profanity_or_reject") or []) if str(x).strip()
        ],
        filler_phrases=[str(x) for x in (raw.get("filler") or []) if str(x).strip()],
        affirmation_phrases=[
            str(x) for x in (raw.get("affirmation") or []) if str(x).strip()
        ],
    )
    if not cfg.filler_phrases:
        cfg.filler_phrases = ["嗯嗯", "嗯", "啊", "哦", "呃", "额", "唔"]
    if not cfg.affirmation_phrases:
        cfg.affirmation_phrases = list(cfg.positive_phrases) or [
            "需要",
            "可以",
            "是",
            "有",
            "好的",
            "好",
        ]
    patterns = compile_lexicon(cfg)
    return SemanticJudge(version=judge_version, lexicon_path=path, patterns=patterns)


__all__ = [
    "JUDGE_VERSION",
    "POLARITY_MIXED",
    "POLARITY_NEGATIVE",
    "POLARITY_NEUTRAL",
    "POLARITY_POSITIVE",
    "POLARITY_UNKNOWN",
    "SemanticJudge",
    "load_semantic_judge",
]
