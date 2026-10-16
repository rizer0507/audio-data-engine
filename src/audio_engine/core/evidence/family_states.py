"""Family dual-run state bundles (extracted from selection_v3.five_class_v2, 044).

Exactly one of ``stable_text | stable_empty | unstable | unavailable`` per family.
Behavior is identical to the historical v2 implementation; the only change is that
the config parameter is a ``RouteConfig`` instead of ``SelectionV3Config``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

from audio_engine.core.evidence.family_routes import RouteView
from audio_engine.core.evidence.route_config import RouteConfig
from audio_engine.core.evidence.semantic_polarity import (
    compile_lexicon_patterns,
    route_pair_has_semantic_conflict,
)
from audio_engine.core.evidence.text_tolerance import apply_tolerance_key, to_simplified
from audio_engine.core.evidence.types import (
    FAMILY_STATE_STABLE_EMPTY,
    FAMILY_STATE_STABLE_TEXT,
    FAMILY_STATE_UNAVAILABLE,
    FAMILY_STATE_UNSTABLE,
    ROUTE_ELIGIBLE,
    ROUTE_EXCLUDED,
    ROUTE_FAILED,
    ROUTE_MISSING,
    RUN_STATUS_FAILED,
    RUN_STATUS_MISSING,
    RUN_STATUS_SUCCESS_EMPTY,
    RUN_STATUS_SUCCESS_TEXT,
)


@dataclass
class FamilyStateBundle:
    family: str
    state: str
    routes: list[RouteView] = field(default_factory=list)
    eligible: list[RouteView] = field(default_factory=list)
    representative: RouteView | None = None
    representative_text: str | None = None
    opposing_text: str | None = None


def harmless_equivalent(left: str, right: str) -> bool:
    """Harmless text equivalence: simplified/tolerance key equal, 吗/吧 preserved."""
    if left == right:
        return True
    a, _ = to_simplified(left)
    b, _ = to_simplified(right)
    if a == b:
        return True
    ka = apply_tolerance_key(a)
    kb = apply_tolerance_key(b)
    if ka == kb:
        if ("吗" in a) != ("吗" in b) or ("嗎" in a) != ("嗎" in b):
            return False
        if ("吧" in a) != ("吧" in b):
            return False
        return True
    return False


def build_family_states(
    routes: list[RouteView], config: RouteConfig
) -> list[FamilyStateBundle]:
    """Build exactly one of stable_text|stable_empty|unstable|unavailable per family."""
    bundles: list[FamilyStateBundle] = []
    expected = int(config.expected_runs_per_family or 2)
    for family in config.ordered_families():
        members = [r for r in routes if r.family == family]
        excluded = [r for r in members if r.route_disposition == ROUTE_EXCLUDED]
        failed = [
            r
            for r in members
            if r.route_disposition in {ROUTE_FAILED, ROUTE_MISSING}
            or r.status in {RUN_STATUS_FAILED, RUN_STATUS_MISSING}
        ]
        # Eligible success only — exclusions/failures never become empty votes.
        eligible = [
            r
            for r in members
            if r.route_disposition == ROUTE_ELIGIBLE
            and r.status in {RUN_STATUS_SUCCESS_TEXT, RUN_STATUS_SUCCESS_EMPTY}
        ]
        text_routes = [
            r for r in eligible if r.status == RUN_STATUS_SUCCESS_TEXT and r.classify_text
        ]
        empty_routes = [
            r
            for r in eligible
            if r.status == RUN_STATUS_SUCCESS_EMPTY or not r.classify_text
        ]
        bundle = FamilyStateBundle(family=family, state=FAMILY_STATE_UNAVAILABLE, routes=members)
        bundle.eligible = eligible

        if len(text_routes) >= expected:
            texts = [r.classify_text for r in text_routes]
            pairwise_ok = all(
                harmless_equivalent(texts[i], texts[j])
                for i in range(len(texts))
                for j in range(i + 1, len(texts))
            )
            if pairwise_ok:
                patterns = compile_lexicon_patterns(config)
                conflict = any(
                    route_pair_has_semantic_conflict(a.classify_text, b.classify_text, patterns)
                    for a, b in combinations(text_routes, 2)
                )
                if conflict:
                    bundle.state = FAMILY_STATE_UNSTABLE
                    bundle.opposing_text = texts[0]
                else:
                    bundle.state = FAMILY_STATE_STABLE_TEXT
                    bundle.representative = sorted(text_routes, key=lambda r: r.run_id)[0]
                    bundle.representative_text = bundle.representative.classify_text
            else:
                bundle.state = FAMILY_STATE_UNSTABLE
                bundle.opposing_text = texts[0]
                bundle.representative = text_routes[0]
        elif text_routes and empty_routes:
            bundle.state = FAMILY_STATE_UNSTABLE
            bundle.opposing_text = text_routes[0].classify_text
            bundle.representative = text_routes[0]
        elif len(text_routes) == 1 and expected >= 2:
            # One success text + other failed/missing/excluded → unstable.
            bundle.state = FAMILY_STATE_UNSTABLE
            bundle.opposing_text = text_routes[0].classify_text
            bundle.representative = text_routes[0]
        elif not text_routes and len(empty_routes) >= expected:
            bundle.state = FAMILY_STATE_STABLE_EMPTY
        elif not text_routes and empty_routes and (failed or excluded):
            # One successful empty + incomplete peer → unstable (never fake stable_empty).
            bundle.state = FAMILY_STATE_UNSTABLE
        elif not eligible:
            bundle.state = FAMILY_STATE_UNAVAILABLE
        else:
            bundle.state = FAMILY_STATE_UNAVAILABLE

        bundles.append(bundle)
    return bundles
