"""Semantic risk tags and polarity for selection_v3 (all configured routes)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

from audio_engine.core.evidence.semantic_polarity import (  # noqa: F401  (044 re-exports)
    LexiconPatterns,
    _compile_phrases,
    _token_presence,
    compile_lexicon_patterns,
    conflict_tags_for_texts,
    polarity_of_text,
    route_pair_has_semantic_conflict,
)
from audio_engine.core.selection_v3.config import SelectionV3Config
from audio_engine.core.selection_v3.types import (
    POLARITY_MIXED,
    POLARITY_NEGATIVE,
    POLARITY_NEUTRAL,
    POLARITY_POSITIVE,
    POLARITY_UNKNOWN,
    RISK_CRITICAL_TOKEN_CONFLICT,
    RISK_FALSE_AFFIRMATION,
    RISK_FILLER_AFFIRMATION,
    RISK_NEGATION_FLIP,
    RISK_PRESENCE_CONFLICT,
    RISK_REJECTION_SANITIZATION,
    RISK_SHORT_UTTERANCE,
)


def compile_lexicon(config: SelectionV3Config) -> LexiconPatterns:
    """Compile lexicon patterns from a selection config (delegates to evidence layer)."""
    return compile_lexicon_patterns(config)


@dataclass
class RiskAnalysis:
    risk_tags: list[str]
    polarity: str
    short_utterance: bool
    presence_conflict: bool
    semantic_risk: bool
    critical_content_risk: bool


def analyze_risks(
    *,
    comparison_texts: list[str],
    success_empty_count: int,
    success_text_count: int,
    short_utterance: bool,
    family_unstable: bool,
    noisy_audio: bool,
    quality_unknown: bool,
    crosstalk_suspected: bool,
    patterns: LexiconPatterns,
) -> RiskAnalysis:
    tags: set[str] = set()
    tags |= conflict_tags_for_texts(comparison_texts, patterns)

    presence = success_empty_count > 0 and success_text_count > 0
    if presence:
        tags.add(RISK_PRESENCE_CONFLICT)
    if short_utterance:
        tags.add(RISK_SHORT_UTTERANCE)
    from audio_engine.core.selection_v3.types import (
        RISK_CONTENT_COMPLEXITY,
        RISK_CROSSTALK_SUSPECTED,
        RISK_FAMILY_INSTABILITY,
        RISK_NOISY_AUDIO,
        RISK_QUALITY_UNKNOWN,
        SEMANTIC_RISK_TAGS,
    )

    if noisy_audio:
        tags.add(RISK_NOISY_AUDIO)
    if quality_unknown:
        tags.add(RISK_QUALITY_UNKNOWN)
    if crosstalk_suspected:
        tags.add(RISK_CROSSTALK_SUSPECTED)
    if family_unstable:
        tags.add(RISK_FAMILY_INSTABILITY)

    polarities = [polarity_of_text(t, patterns) for t in comparison_texts if t]
    distinct_texts = {t for t in comparison_texts if t}
    same_comparison_text = len(distinct_texts) <= 1
    if not polarities:
        polarity = POLARITY_UNKNOWN
    elif POLARITY_MIXED in polarities:
        polarity = POLARITY_MIXED
        if same_comparison_text:
            tags.add(RISK_CONTENT_COMPLEXITY)
    else:
        distinct = {p for p in polarities if p not in {POLARITY_NEUTRAL, POLARITY_UNKNOWN}}
        if len(distinct) > 1:
            polarity = POLARITY_MIXED
        elif len(distinct) == 1:
            polarity = next(iter(distinct))
        else:
            polarity = POLARITY_UNKNOWN if POLARITY_UNKNOWN in polarities else POLARITY_NEUTRAL

    semantic_risk = bool(tags & SEMANTIC_RISK_TAGS)
    # Mixed polarity on identical comparison texts is content complexity, not
    # inter-model critical_content_risk (020 / 0908 eighty-eight-sample class).
    critical_content = RISK_CRITICAL_TOKEN_CONFLICT in tags or (
        polarity == POLARITY_MIXED and not same_comparison_text
    )

    return RiskAnalysis(
        risk_tags=sorted(tags),
        polarity=polarity,
        short_utterance=short_utterance,
        presence_conflict=presence,
        semantic_risk=semantic_risk,
        critical_content_risk=critical_content,
    )
