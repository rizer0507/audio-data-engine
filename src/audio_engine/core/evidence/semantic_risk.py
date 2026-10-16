"""Neutral semantic evidence helpers (extracted from selection_v3.five_class_v2, 044).

``semantic_risk_v2`` observes multi-family stable-text polarity conflicts. It is an
*evidence observation*, not a business routing decision: callers (tagging or legacy
selection) decide what to do with the returned hit.
"""

from __future__ import annotations

from typing import Any

from audio_engine.core.evidence.family_states import FamilyStateBundle
from audio_engine.core.evidence.route_config import RouteConfig
from audio_engine.core.evidence.semantic_polarity import (
    compile_lexicon_patterns,
    polarity_of_text,
)
from audio_engine.core.evidence.semantic_strict import (
    FamilyPolarityView,
    StrictRiskHit,
    _objects,
    analyze_strict_risks,
    han_char_count,
)
from audio_engine.core.evidence.text_tolerance import apply_tolerance_key, to_simplified
from audio_engine.core.evidence.types import (
    FAMILY_STATE_STABLE_TEXT,
    RISK_TAG_CRITICAL_POLARITY_RESPONSE,
    RISK_TAG_POLARITY_CONFLICT,
    SUBTYPE_HALLUCINATED_ASSERTION,
    SUBTYPE_SEMANTIC_REVERSAL,
    SUBTYPE_SHORT_POLARITY_AMBIGUITY,
)

__all__ = [
    "han_char_count",
    "is_critical_short_response",
    "normalize_response",
    "risk_tag_for_subtype",
    "semantic_risk_v2",
]


def normalize_response(text: str) -> str:
    simplified, _ = to_simplified(text or "")
    return apply_tolerance_key(simplified).strip()


def is_critical_short_response(text: str, config: RouteConfig) -> bool:
    body = normalize_response(text)
    if not body:
        return False
    catalog = [
        normalize_response(x)
        for x in (config.critical_short_responses or [])
        if str(x).strip()
    ]
    if body in set(catalog):
        return True
    # Conservative: short polarity templates also block human_noise.
    if han_char_count(body) <= 4:
        pol = polarity_of_text(body, compile_lexicon_patterns(config))
        if pol in {"positive", "negative"}:
            return True
    return False


def risk_tag_for_subtype(subtype: str | None) -> str | None:
    """Map a strict-risk subtype to the public risk-tag vocabulary."""
    text = str(subtype or "").strip()
    if text in {
        RISK_TAG_POLARITY_CONFLICT,
        SUBTYPE_SEMANTIC_REVERSAL,
        "semantic_reversal",
    }:
        return RISK_TAG_POLARITY_CONFLICT
    if text in {
        RISK_TAG_CRITICAL_POLARITY_RESPONSE,
        SUBTYPE_SHORT_POLARITY_AMBIGUITY,
        "short_polarity_ambiguity",
    }:
        return RISK_TAG_CRITICAL_POLARITY_RESPONSE
    return text or None


def semantic_risk_v2(
    bundles: list[FamilyStateBundle],
    config: RouteConfig,
) -> Any:
    """Only multi stable_text family conflicts. Empty-vs-text is not semantic_risk."""
    patterns = compile_lexicon_patterns(config)
    views: list[FamilyPolarityView] = []
    for bundle in bundles:
        if bundle.state == FAMILY_STATE_STABLE_TEXT and bundle.representative_text:
            text = bundle.representative_text
            views.append(
                FamilyPolarityView(
                    family=bundle.family,
                    text=text,
                    polarity=polarity_of_text(text, patterns),
                    stable=True,
                    is_empty=False,
                    run_ids=[r.run_id for r in bundle.eligible],
                )
            )
    # Do not feed stable_empty into analyze_strict_risks — that would open
    # hallucinated_assertion, which 028 routes to environment_noise/hardcase.
    analysis = analyze_strict_risks(
        views,
        patterns=patterns,
        acoustic={},
        max_han_chars=int(getattr(config, "short_polarity_max_han_chars", 4) or 4),
        human_verified_no_target=False,
        human_unintelligible_polarity=False,
    )
    if analysis.hit and analysis.hit.formal:
        if analysis.hit.subtype == SUBTYPE_HALLUCINATED_ASSERTION:
            return None
        return analysis.hit
    # Explicit multi-family polarity conflict fallback.
    pos = [v for v in views if v.polarity == "positive"]
    neg = [v for v in views if v.polarity == "negative"]
    if pos and neg:
        # Require same-proposition-ish overlap via shared objects or short responses.
        for a in pos:
            for b in neg:
                oa, ob = _objects(a.text), _objects(b.text)
                if (oa and ob and not oa.isdisjoint(ob)) or (
                    han_char_count(a.text) <= 4 and han_char_count(b.text) <= 4
                ):
                    return StrictRiskHit(
                        subtype=SUBTYPE_SEMANTIC_REVERSAL,
                        evidence={
                            "family_a": a.family,
                            "family_b": b.family,
                            "text_a": a.text,
                            "text_b": b.text,
                        },
                        conflict_spans=[
                            {
                                "family_a": a.family,
                                "family_b": b.family,
                                "text_a": a.text,
                                "text_b": b.text,
                            }
                        ],
                    )
    return None
