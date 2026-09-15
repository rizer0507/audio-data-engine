"""Per-family dual-run evidence for selection_v3."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from audio_engine.core.sample import Sample
from audio_engine.core.selection_v3.config import SelectionV3Config
from audio_engine.core.selection_v3.input_contract import classify_run_status
from audio_engine.core.selection_v3.semantic_risk import (
    LexiconPatterns,
    polarity_of_text,
    route_pair_has_semantic_conflict,
)
from audio_engine.core.selection_v3.text import (
    comparison_text,
    raw_transcript_text,
    text_similarity,
    transcript_text,
)
from audio_engine.core.selection_v3.types import (
    FAMILY_INCOMPLETE,
    FAMILY_STABLE_EMPTY,
    FAMILY_STABLE_TEXT,
    FAMILY_UNSTABLE_PRESENCE,
    FAMILY_UNSTABLE_SEMANTIC,
    FAMILY_UNSTABLE_TEXT,
    ROUTE_EXCLUDED,
    ROUTE_FAILED,
    ROUTE_MISSING,
    RUN_STATUS_FAILED,
    RUN_STATUS_MISSING,
    RUN_STATUS_SUCCESS_EMPTY,
    RUN_STATUS_SUCCESS_TEXT,
)


@dataclass
class RouteView:
    run_id: str
    family: str
    status: str
    raw_text: str
    comparison_text: str
    transcript_text: str = ""
    classify_text: str = ""
    empty_reason_codes: tuple[str, ...] = ()
    pre_filter_language: str = ""
    body_text: str = ""
    route_disposition: str = "eligible"
    exclusion_reasons: tuple[str, ...] = ()


@dataclass
class FamilyEvidence:
    family: str
    routes: list[RouteView]
    status: str
    representative: RouteView | None = None
    retained_routes: list[RouteView] = field(default_factory=list)
    min_similarity: float | None = None
    provides_text_vote: bool = False


def _entry(sample: Sample, key: str) -> Any:
    return sample.transcripts.get(key)


def collect_route_views(
    sample: Sample,
    config: SelectionV3Config,
    *,
    quarantine_run_ids: set[str] | frozenset[str] | None = None,
) -> list[RouteView]:
    views: list[RouteView] = []
    punct = config.punctuation_to_strip
    blocked = set(quarantine_run_ids or ())
    chinese_only = config.uses_chinese_only_text()
    for family in config.ordered_families():
        for key in config.model_families.get(family, []):
            status = classify_run_status(sample, key, config)
            entry = _entry(sample, key)
            raw = raw_transcript_text(entry) if entry is not None else ""
            # Quarantined routes keep raw text for evidence but do not vote.
            if str(key) in blocked:
                views.append(
                    RouteView(
                        run_id=str(key),
                        family=family,
                        status=RUN_STATUS_FAILED,
                        raw_text=raw,
                        comparison_text="",
                    )
                )
                continue
            transcript = ""
            classify = ""
            reasons: tuple[str, ...] = ()
            pre_lang = ""
            body = ""
            disposition = "eligible"
            exclusions: tuple[str, ...] = ()
            if chinese_only:
                from audio_engine.core.selection_v3.classify_text import prepare_classify_text

                prepared = prepare_classify_text(
                    raw or (entry.get("text") if isinstance(entry, dict) else raw),
                    echo=config.echo_table_for(family),
                    keep_digits=config.classify_text_keep_digits,
                    policy=config.classify_text_policy,
                )
                transcript = prepared.transcript_text
                body = prepared.body_text or prepared.transcript_text
                classify = prepared.classify_text
                reasons = prepared.empty_reason_codes
                pre_lang = prepared.pre_filter_language
                disposition = prepared.route_disposition
                exclusions = prepared.exclusion_reasons
                if status in {RUN_STATUS_FAILED, RUN_STATUS_MISSING}:
                    cmp = ""
                    disposition = ROUTE_FAILED if status == RUN_STATUS_FAILED else ROUTE_MISSING
                elif prepared.is_excluded:
                    # Excluded routes keep audit text but never vote or count as empty.
                    status = RUN_STATUS_SUCCESS_EMPTY
                    cmp = ""
                    classify = ""
                elif prepared.status == RUN_STATUS_SUCCESS_EMPTY:
                    status = RUN_STATUS_SUCCESS_EMPTY
                    cmp = ""
                else:
                    status = RUN_STATUS_SUCCESS_TEXT
                    cmp = classify
            elif status == RUN_STATUS_SUCCESS_TEXT:
                cmp = comparison_text(
                    raw or (entry.get("text") if isinstance(entry, dict) else raw),
                    punctuation_to_strip=punct,
                )
                transcript = transcript_text(raw)
                body = transcript
            elif status == RUN_STATUS_SUCCESS_EMPTY:
                cmp = ""
                transcript = transcript_text(raw)
                body = transcript
            else:
                cmp = ""
                disposition = ROUTE_FAILED if status == RUN_STATUS_FAILED else (
                    ROUTE_MISSING if status == RUN_STATUS_MISSING else disposition
                )
            views.append(
                RouteView(
                    run_id=str(key),
                    family=family,
                    status=status,
                    raw_text=raw,
                    comparison_text=cmp,
                    transcript_text=transcript,
                    classify_text=classify,
                    empty_reason_codes=reasons,
                    pre_filter_language=pre_lang,
                    body_text=body,
                    route_disposition=disposition,
                    exclusion_reasons=exclusions,
                )
            )
    return views


def is_short_utterance(
    *,
    duration_sec: float | None,
    routes: list[RouteView],
    max_audio_sec: float,
    max_text_chars: int,
) -> bool:
    if duration_sec is not None and duration_sec <= max_audio_sec:
        return True
    for route in routes:
        if route.status == RUN_STATUS_SUCCESS_TEXT and route.comparison_text:
            if len(route.comparison_text) <= max_text_chars:
                return True
    return False


def _pick_family_medoid(routes: list[RouteView]) -> RouteView:
    """Family-balanced medoid among non-empty success routes; ties by run_id."""
    nonempty = [
        r for r in routes if r.status == RUN_STATUS_SUCCESS_TEXT and r.comparison_text
    ]
    if not nonempty:
        raise ValueError("medoid requires non-empty routes")
    ordered = sorted(nonempty, key=lambda r: r.run_id)
    if len(ordered) == 1:
        return ordered[0]
    best: RouteView | None = None
    best_score = -1.0
    for cand in ordered:
        scores = [
            text_similarity(cand.comparison_text, other.comparison_text)
            for other in ordered
            if other.run_id != cand.run_id
        ]
        mean = sum(scores) / len(scores)
        if mean > best_score or (
            mean == best_score and best is not None and cand.run_id < best.run_id
        ) or best is None:
            best_score = mean
            best = cand
    assert best is not None
    return best


def analyze_family(
    family: str,
    routes: list[RouteView],
    config: SelectionV3Config,
    patterns: LexiconPatterns,
    *,
    short: bool,
) -> FamilyEvidence:
    members = [r for r in routes if r.family == family]
    # 1. incomplete
    if any(r.status in {RUN_STATUS_FAILED, RUN_STATUS_MISSING} for r in members):
        return FamilyEvidence(
            family=family,
            routes=members,
            status=FAMILY_INCOMPLETE,
            retained_routes=list(members),
        )

    success = [
        r
        for r in members
        if r.status in {RUN_STATUS_SUCCESS_TEXT, RUN_STATUS_SUCCESS_EMPTY}
    ]
    nonempty = [r for r in success if r.status == RUN_STATUS_SUCCESS_TEXT and r.comparison_text]
    empty = [r for r in success if r.status == RUN_STATUS_SUCCESS_EMPTY or not r.comparison_text]

    # 2. stable_empty
    if success and not nonempty:
        return FamilyEvidence(
            family=family,
            routes=members,
            status=FAMILY_STABLE_EMPTY,
            retained_routes=list(members),
        )

    # 3. unstable_presence
    if nonempty and empty:
        return FamilyEvidence(
            family=family,
            routes=members,
            status=FAMILY_UNSTABLE_PRESENCE,
            retained_routes=list(members),
        )

    # 4. unstable_semantic
    if len(nonempty) >= 2 and route_pair_has_semantic_conflict(
        nonempty[0].comparison_text,
        nonempty[1].comparison_text,
        patterns,
    ):
        return FamilyEvidence(
            family=family,
            routes=members,
            status=FAMILY_UNSTABLE_SEMANTIC,
            retained_routes=list(members),
            min_similarity=text_similarity(
                nonempty[0].comparison_text, nonempty[1].comparison_text
            ),
        )

    # 5. stable_text
    if len(nonempty) >= 2:
        sim = text_similarity(nonempty[0].comparison_text, nonempty[1].comparison_text)
        if short:
            texts_equal = nonempty[0].comparison_text == nonempty[1].comparison_text
            stable = texts_equal
        else:
            stable = sim >= config.family_threshold
        if stable:
            medoid = _pick_family_medoid(nonempty)
            return FamilyEvidence(
                family=family,
                routes=members,
                status=FAMILY_STABLE_TEXT,
                representative=medoid,
                retained_routes=list(nonempty),
                min_similarity=sim,
                provides_text_vote=True,
            )
        return FamilyEvidence(
            family=family,
            routes=members,
            status=FAMILY_UNSTABLE_TEXT,
            retained_routes=list(members),
            min_similarity=sim,
        )

    # Single non-empty success (should be rare with expected_runs=2)
    if len(nonempty) == 1:
        return FamilyEvidence(
            family=family,
            routes=members,
            status=FAMILY_UNSTABLE_TEXT,
            retained_routes=list(members),
        )

    return FamilyEvidence(
        family=family,
        routes=members,
        status=FAMILY_UNSTABLE_TEXT,
        retained_routes=list(members),
    )


def analyze_families(
    sample: Sample,
    config: SelectionV3Config,
    patterns: LexiconPatterns,
    *,
    duration_sec: float | None,
    quarantine_run_ids: set[str] | frozenset[str] | None = None,
    routes: list[RouteView] | None = None,
) -> dict[str, FamilyEvidence]:
    route_views = routes if routes is not None else collect_route_views(
        sample, config, quarantine_run_ids=quarantine_run_ids
    )
    short = is_short_utterance(
        duration_sec=duration_sec,
        routes=route_views,
        max_audio_sec=config.short_audio_sec,
        max_text_chars=config.short_text_chars,
    )
    result: dict[str, FamilyEvidence] = {}
    for family in config.ordered_families():
        result[family] = analyze_family(
            family, route_views, config, patterns, short=short
        )
    return result


def active_family_count(families: dict[str, FamilyEvidence]) -> int:
    """Families still usable after quarantine / failures (not incomplete)."""
    return sum(1 for state in families.values() if state.status != FAMILY_INCOMPLETE)


def voting_representatives(
    families: dict[str, FamilyEvidence],
) -> list[RouteView]:
    """One vote per stable_text family only."""
    votes: list[RouteView] = []
    for state in families.values():
        if state.provides_text_vote and state.representative is not None:
            votes.append(state.representative)
    return votes


def all_success_routes(families: dict[str, FamilyEvidence]) -> list[RouteView]:
    routes: list[RouteView] = []
    for state in families.values():
        for route in state.routes:
            if route.status in {RUN_STATUS_SUCCESS_TEXT, RUN_STATUS_SUCCESS_EMPTY}:
                routes.append(route)
    return routes


def polarity_summary(
    routes: list[RouteView],
    patterns: LexiconPatterns,
) -> str:
    from audio_engine.core.selection_v3.types import (
        POLARITY_MIXED,
        POLARITY_NEGATIVE,
        POLARITY_NEUTRAL,
        POLARITY_POSITIVE,
        POLARITY_UNKNOWN,
    )

    classes = {
        polarity_of_text(r.comparison_text, patterns)
        for r in routes
        if r.status == RUN_STATUS_SUCCESS_TEXT and r.comparison_text
    }
    classes.discard(POLARITY_NEUTRAL)
    classes.discard(POLARITY_UNKNOWN)
    if not classes:
        return POLARITY_UNKNOWN
    if classes == {POLARITY_POSITIVE}:
        return POLARITY_POSITIVE
    if classes == {POLARITY_NEGATIVE}:
        return POLARITY_NEGATIVE
    if POLARITY_MIXED in classes or len(classes) > 1:
        return POLARITY_MIXED if len(classes) > 1 else POLARITY_MIXED
    return next(iter(classes))
