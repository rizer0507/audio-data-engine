"""Neutral polarity / route-pair conflict helpers (extracted from selection_v3.semantic_risk).

No business routing here: this module only observes polarity of text and conflicts
between two route texts. Lexicon phrases are taken from explicit lists (or from any
object exposing the lexicon attributes), never from a selection business config type.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from audio_engine.core.evidence.types import (
    POLARITY_MIXED,
    POLARITY_NEGATIVE,
    POLARITY_NEUTRAL,
    POLARITY_POSITIVE,
    POLARITY_UNKNOWN,
    RISK_CONTENT_COMPLEXITY,
    RISK_CRITICAL_TOKEN_CONFLICT,
    RISK_FALSE_AFFIRMATION,
    RISK_FILLER_AFFIRMATION,
    RISK_NEGATION_FLIP,
    RISK_REJECTION_SANITIZATION,
)


@dataclass
class LexiconPatterns:
    negative: re.Pattern[str] | None
    positive: re.Pattern[str] | None
    critical: re.Pattern[str] | None
    reject: re.Pattern[str] | None
    filler: re.Pattern[str] | None
    affirmation: re.Pattern[str] | None
    critical_tokens: list[str] = field(default_factory=list)


def _compile_phrases(phrases: Iterable[str]) -> re.Pattern[str] | None:
    items = sorted({str(p).strip() for p in phrases if str(p).strip()}, key=len, reverse=True)
    if not items:
        return None
    return re.compile("|".join(f"(?:{re.escape(p)})" for p in items))


def compile_lexicon_phrases(
    *,
    negative: Iterable[str] = (),
    positive: Iterable[str] = (),
    critical_tokens: Iterable[str] = (),
    reject: Iterable[str] = (),
    filler: Iterable[str] = (),
    affirmation: Iterable[str] = (),
) -> LexiconPatterns:
    tokens = list(critical_tokens)
    return LexiconPatterns(
        negative=_compile_phrases(negative),
        positive=_compile_phrases(positive),
        critical=_compile_phrases(tokens),
        reject=_compile_phrases(reject),
        filler=_compile_phrases(filler),
        affirmation=_compile_phrases(affirmation),
        critical_tokens=tokens,
    )


def compile_lexicon_patterns(config: Any) -> LexiconPatterns:
    """Compile from any object exposing the lexicon phrase attributes (RouteConfig)."""
    return compile_lexicon_phrases(
        negative=config.negative_phrases,
        positive=config.positive_phrases,
        critical_tokens=config.critical_tokens,
        reject=config.profanity_or_reject,
        filler=config.filler_phrases,
        affirmation=config.affirmation_phrases,
    )


def _token_presence(text: str, tokens: Iterable[str]) -> frozenset[str]:
    value = str(text or "")
    hits: set[str] = set()
    ordered = sorted({str(t).strip() for t in tokens if str(t).strip()}, key=len, reverse=True)
    remaining = value
    for token in ordered:
        if token and token in remaining:
            hits.add(token)
            remaining = remaining.replace(token, "\0" * len(token))
    return frozenset(hits)


def polarity_of_text(text: str, patterns: LexiconPatterns) -> str:
    """Observe-only polarity for the current fragment; unknown when unclear."""
    value = str(text or "").strip()
    if not value:
        return POLARITY_UNKNOWN
    # Complex double-negation / ambiguous scope → mixed/unknown (送审)
    ambiguous_markers = ("不是不", "不是不要", "不是不需要", "不能不")
    if any(marker in value for marker in ambiguous_markers):
        return POLARITY_MIXED
    # Longest-phrase match: negative wins over positive when both fire.
    neg = bool(patterns.negative and patterns.negative.search(value))
    positive_scope = patterns.negative.sub("", value) if patterns.negative else value
    pos = bool(patterns.positive and patterns.positive.search(positive_scope))
    if neg and pos:
        # Positive material remains outside the matched negative span.
        return POLARITY_MIXED
    if neg:
        return POLARITY_NEGATIVE
    if pos:
        return POLARITY_POSITIVE
    if patterns.filler and patterns.filler.fullmatch(value):
        return POLARITY_NEUTRAL
    return POLARITY_UNKNOWN


def route_pair_has_semantic_conflict(
    left: str,
    right: str,
    patterns: LexiconPatterns,
) -> bool:
    """True when two comparison texts conflict on polarity / critical / filler / reject."""
    if not left or not right:
        return False
    # Same comparison text cannot be an inter-route semantic conflict (020).
    if left == right:
        return False
    tags = conflict_tags_for_texts([left, right], patterns)
    return bool(
        tags
        & {
            RISK_NEGATION_FLIP,
            RISK_FALSE_AFFIRMATION,
            RISK_FILLER_AFFIRMATION,
            RISK_REJECTION_SANITIZATION,
            RISK_CRITICAL_TOKEN_CONFLICT,
        }
    )


def conflict_tags_for_texts(
    texts: list[str],
    patterns: LexiconPatterns,
) -> set[str]:
    tags: set[str] = set()
    nonempty = [t for t in texts if t]
    if len(nonempty) < 2:
        return tags

    # Identical comparison texts are not model-vs-model conflicts (020 / 0908).
    # Single-utterance mixed polarity may still be content complexity for sampling.
    if len(set(nonempty)) == 1:
        if polarity_of_text(nonempty[0], patterns) == POLARITY_MIXED:
            tags.add(RISK_CONTENT_COMPLEXITY)
        return tags

    polarities = [polarity_of_text(t, patterns) for t in nonempty]
    has_pos = POLARITY_POSITIVE in polarities
    has_neg = POLARITY_NEGATIVE in polarities
    has_mixed = POLARITY_MIXED in polarities
    if has_pos and has_neg:
        tags.add(RISK_NEGATION_FLIP)
    if has_mixed:
        tags.add(RISK_CRITICAL_TOKEN_CONFLICT)

    # false_affirmation: non-affirmation vs affirmation
    has_affirm = False
    has_non_affirm = False
    for text, pol in zip(nonempty, polarities):
        if pol == POLARITY_POSITIVE:
            has_affirm = True
        elif pol in {POLARITY_NEGATIVE, POLARITY_NEUTRAL, POLARITY_UNKNOWN, POLARITY_MIXED}:
            # Neutral filler alone is handled by filler_affirmation; here non-positive content
            if pol in {POLARITY_NEGATIVE, POLARITY_UNKNOWN} or (
                pol == POLARITY_NEUTRAL
                and not (patterns.filler and patterns.filler.fullmatch(text))
            ):
                has_non_affirm = True
    if has_affirm and has_non_affirm and RISK_NEGATION_FLIP not in tags:
        # Distinct from pure negation_flip when one side is non-affirmative non-negative
        if any(pol in {POLARITY_NEUTRAL, POLARITY_UNKNOWN} for pol in polarities) and has_affirm:
            tags.add(RISK_FALSE_AFFIRMATION)

    # filler_affirmation: filler-only vs affirmation
    has_filler_only = False
    has_affirmation = False
    for text in nonempty:
        if patterns.filler and patterns.filler.fullmatch(text):
            has_filler_only = True
            continue
        pol = polarity_of_text(text, patterns)
        if pol == POLARITY_POSITIVE or (
            patterns.affirmation and patterns.affirmation.search(text)
        ):
            has_affirmation = True
    if has_filler_only and has_affirmation:
        tags.add(RISK_FILLER_AFFIRMATION)

    # rejection_sanitization
    has_reject = False
    has_positive_clean = False
    for text in nonempty:
        if patterns.reject and patterns.reject.search(text):
            has_reject = True
            continue
        if polarity_of_text(text, patterns) == POLARITY_POSITIVE:
            has_positive_clean = True
    if has_reject and has_positive_clean:
        tags.add(RISK_REJECTION_SANITIZATION)

    # critical token / entity presence conflict
    numbers = [set(re.findall(r"\d+(?:\.\d+)?", text)) for text in nonempty]
    if any(value != numbers[0] for value in numbers[1:]):
        tags.add(RISK_CRITICAL_TOKEN_CONFLICT)
    if patterns.critical_tokens:
        sets = [_token_presence(text, patterns.critical_tokens) for text in nonempty]
        if len(sets) >= 2 and any(item != sets[0] for item in sets[1:]):
            tags.add(RISK_CRITICAL_TOKEN_CONFLICT)

    return tags
