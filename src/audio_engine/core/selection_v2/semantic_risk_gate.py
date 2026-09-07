"""Semantic risk gates for selection_v2.0.

``similarity >= 0.95`` must never override ``semantic_risk=true``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from audio_engine.core.selection_engine import (
    TranscriptView,
    _compile_phrase_pattern,
    family_semantic,
    semantic_class,
)
from audio_engine.core.selection_v2.config import SelectionV2Config
from audio_engine.core.selection_v2.types import (
    SEMANTIC_CONFLICT,
    SEMANTIC_NEGATIVE,
    SEMANTIC_NONE,
    SEMANTIC_POSITIVE,
    SEMANTIC_UNKNOWN,
)


@dataclass
class SemanticSignals:
    semantic_class: str
    semantic_risk: bool
    critical_token_conflict: bool
    sanitization_risk: bool
    short_utterance: bool
    family_classes: dict[str, str]
    polarity_conflict: bool


def compile_lexicon(config: SelectionV2Config) -> dict[str, re.Pattern[str] | None]:
    return {
        "negative": _compile_phrase_pattern(config.negative_phrases),
        "positive": _compile_phrase_pattern(config.positive_phrases),
        "critical": _compile_phrase_pattern(config.critical_tokens),
        "reject": _compile_phrase_pattern(config.profanity_or_reject),
    }


def _token_presence(text: str, tokens: Iterable[str]) -> frozenset[str]:
    value = str(text or "")
    hits: set[str] = set()
    # Longer tokens first so「不需要」wins over「需要」/「不」.
    ordered = sorted({str(t).strip() for t in tokens if str(t).strip()}, key=len, reverse=True)
    remaining = value
    for token in ordered:
        if token and token in remaining:
            hits.add(token)
            remaining = remaining.replace(token, "\0" * len(token))
    return frozenset(hits)


def critical_token_sets_conflict(
    texts: list[str],
    tokens: list[str],
) -> bool:
    if len(texts) < 2 or not tokens:
        return False
    sets = [_token_presence(text, tokens) for text in texts if text]
    if len(sets) < 2:
        return False
    base = sets[0]
    return any(item != base for item in sets[1:])


def detect_sanitization(
    views: list[TranscriptView],
    *,
    reject_pattern: re.Pattern[str] | None,
    positive_pattern: re.Pattern[str] | None,
    negative_pattern: re.Pattern[str] | None,
) -> bool:
    if reject_pattern is None:
        return False
    has_reject = False
    has_positive_clean = False
    for item in views:
        if not item.text:
            continue
        if reject_pattern.search(item.text):
            has_reject = True
            continue
        cls = semantic_class(
            item.text,
            negative_pattern=negative_pattern,
            positive_pattern=positive_pattern,
        )
        if cls == SEMANTIC_POSITIVE:
            has_positive_clean = True
    return has_reject and has_positive_clean


def is_short_utterance(
    *,
    duration_sec: float | None,
    texts: list[str],
    max_audio_sec: float,
    max_text_chars: int,
) -> bool:
    if duration_sec is not None and duration_sec <= max_audio_sec:
        return True
    for text in texts:
        if text and len(text) <= max_text_chars:
            return True
    return False


def short_utterance_allows_pseudo_high(
    views: list[TranscriptView],
    *,
    negative_pattern: re.Pattern[str] | None,
    positive_pattern: re.Pattern[str] | None,
    critical_tokens: list[str],
) -> bool:
    """Require cross-family exact match OR critical-token + semantic agreement."""
    nonempty = [item for item in views if item.text]
    if len({item.family for item in nonempty}) < 2:
        return False
    texts = [item.text for item in nonempty]
    if len(set(texts)) == 1:
        return True
    # Per-family representative texts must match exactly across families.
    by_family: dict[str, str] = {}
    for item in sorted(nonempty, key=lambda x: x.model):
        by_family.setdefault(item.family, item.text)
    family_texts = list(by_family.values())
    if len(family_texts) >= 2 and len(set(family_texts)) == 1:
        return True
    classes = {
        semantic_class(
            text,
            negative_pattern=negative_pattern,
            positive_pattern=positive_pattern,
        )
        for text in family_texts
    }
    classes.discard(SEMANTIC_NONE)
    if len(classes) > 1:
        return False
    token_sets = [_token_presence(text, critical_tokens) for text in family_texts]
    # Empty critical-token sets on both sides is not evidence of agreement.
    if not any(token_sets):
        return False
    return len(token_sets) >= 2 and all(item == token_sets[0] for item in token_sets[1:])


def analyze_semantics(
    nonempty: list[TranscriptView],
    config: SelectionV2Config,
    *,
    duration_sec: float | None,
    patterns: dict[str, re.Pattern[str] | None],
) -> SemanticSignals:
    negative_pattern = patterns["negative"]
    positive_pattern = patterns["positive"]
    reject_pattern = patterns["reject"]

    family_classes: dict[str, str] = {}
    families = {item.family for item in nonempty}
    for family in families:
        family_classes[family] = family_semantic(
            nonempty,
            family,
            negative_pattern=negative_pattern,
            positive_pattern=positive_pattern,
        )

    polar = {
        cls
        for cls in family_classes.values()
        if cls in {SEMANTIC_POSITIVE, SEMANTIC_NEGATIVE}
    }
    polarity_conflict = SEMANTIC_POSITIVE in polar and SEMANTIC_NEGATIVE in polar

    texts = [item.text for item in nonempty]
    token_conflict = critical_token_sets_conflict(texts, config.critical_tokens)
    sanitization = detect_sanitization(
        nonempty,
        reject_pattern=reject_pattern,
        positive_pattern=positive_pattern,
        negative_pattern=negative_pattern,
    )
    short = is_short_utterance(
        duration_sec=duration_sec,
        texts=texts,
        max_audio_sec=config.short_audio_sec,
        max_text_chars=config.short_text_chars,
    )

    # Aggregate semantic class for the sample (best-effort).
    sample_classes = {
        semantic_class(
            text,
            negative_pattern=negative_pattern,
            positive_pattern=positive_pattern,
        )
        for text in texts
        if text
    }
    sample_classes.discard(SEMANTIC_NONE)
    if not sample_classes:
        agg = SEMANTIC_UNKNOWN
    elif sample_classes == {SEMANTIC_POSITIVE}:
        agg = SEMANTIC_POSITIVE
    elif sample_classes == {SEMANTIC_NEGATIVE}:
        agg = SEMANTIC_NEGATIVE
    elif SEMANTIC_CONFLICT in sample_classes or len(sample_classes) > 1:
        agg = SEMANTIC_CONFLICT if polarity_conflict else SEMANTIC_UNKNOWN
    else:
        agg = next(iter(sample_classes))

    semantic_risk = polarity_conflict or sanitization
    return SemanticSignals(
        semantic_class=agg,
        semantic_risk=semantic_risk,
        critical_token_conflict=token_conflict,
        sanitization_risk=sanitization,
        short_utterance=short,
        family_classes=family_classes,
        polarity_conflict=polarity_conflict,
    )
