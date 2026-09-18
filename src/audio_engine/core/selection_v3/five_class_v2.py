"""028 five-class classifier: selection_five_class_v2_auto_noise.

Order (hit ends): voicemail → semantic_risk → environment_noise →
gold_candidate → hardcase.

Successful classification samples always receive exactly one of the five
categories. Manual review is only an adjunct on hardcase. Failed/missing/
foreign/echo routes never count as stable_empty.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

from audio_engine.core.sample import Sample
from audio_engine.core.selection_v3.audio_energy import (
    AudioEnergyEvidence,
    energy_evidence_from_quality,
)
from audio_engine.core.selection_v3.classify_text import apply_route_audit
from audio_engine.core.selection_v3.config import SelectionV3Config
from audio_engine.core.selection_v3.family_evidence import RouteView, collect_route_views
from audio_engine.core.selection_v3.gold_select import FamilyRep, select_weighted_family_text
from audio_engine.core.selection_v3.input_contract import is_physically_invalid
from audio_engine.core.selection_v3.result import ClassificationResultV3
from audio_engine.core.selection_v3.semantic_risk import compile_lexicon, polarity_of_text
from audio_engine.core.selection_v3.semantic_risk_strict import (
    FamilyPolarityView,
    analyze_strict_risks,
    han_char_count,
)
from audio_engine.core.selection_v3.text_tolerance import (
    apply_tolerance_key,
    to_simplified,
    tolerant_distance,
)
from audio_engine.core.selection_v3.types import (
    CATEGORY_ENVIRONMENT_NOISE,
    CATEGORY_GOLD_CANDIDATE,
    CATEGORY_HARDCASE,
    CATEGORY_SEMANTIC_RISK,
    CATEGORY_VOICEMAIL,
    DECISION_AUDIT_PENDING,
    DECISION_EXCLUDE,
    DECISION_MANUAL_REVIEW,
    ENERGY_STATE_AUDIBLE,
    ENERGY_STATE_BORDERLINE,
    ENERGY_STATE_FAILED,
    ENERGY_STATE_INAUDIBLE,
    ENERGY_STATE_TOO_SHORT,
    FAMILY_STATE_STABLE_EMPTY,
    FAMILY_STATE_STABLE_TEXT,
    FAMILY_STATE_UNAVAILABLE,
    FAMILY_STATE_UNSTABLE,
    LABEL_SOURCE_MODEL,
    LABEL_SOURCE_NONE,
    LABEL_TIER_NONE,
    LABEL_TIER_PSEUDO_HIGH,
    NOISE_KIND_AUDIO_TOO_SHORT,
    NOISE_KIND_BACKGROUND,
    NOISE_KIND_CROSSTALK,
    NOISE_KIND_HUMAN_NOISE,
    NOISE_KIND_SILENCE,
    OUTCOME_CLASSIFIED,
    OUTCOME_EXCLUDED,
    PRIORITY_P0,
    PRIORITY_P1,
    PRIORITY_P2,
    ROUTE_ELIGIBLE,
    ROUTE_EXCLUDED,
    ROUTE_FAILED,
    ROUTE_MISSING,
    RULE_VERSION_FIVE_CLASS_V2,
    RUN_STATUS_FAILED,
    RUN_STATUS_MISSING,
    RUN_STATUS_SUCCESS_EMPTY,
    RUN_STATUS_SUCCESS_TEXT,
    STATUS_CANDIDATE,
    STATUS_EXCLUDED,
    STATUS_MANUAL_REVIEW,
    SUBTYPE_HALLUCINATED_ASSERTION,
    TYPE_HARDCASE,
    TYPE_INVALID_AUDIO,
    TYPE_PSEUDO_HIGH,
    TYPE_SEMANTIC_RISK,
    TYPE_VOICEMAIL_CANDIDATE,
)

_SUBSTANTIVE_DIST = 0.30


@dataclass
class FamilyStateBundle:
    family: str
    state: str
    routes: list[RouteView] = field(default_factory=list)
    eligible: list[RouteView] = field(default_factory=list)
    representative: RouteView | None = None
    representative_text: str | None = None
    opposing_text: str | None = None


def _audio_refs(sample: Sample) -> dict[str, str]:
    refs: dict[str, str] = {}
    for key, path in (sample.audio or {}).items():
        if path:
            refs[str(key)] = str(path)
    if sample.source_path:
        refs.setdefault("source", str(sample.source_path))
    return refs


def _route_audit(routes: list[RouteView]) -> dict[str, Any]:
    return {
        r.run_id: {
            "family": r.family,
            "disposition": r.route_disposition,
            "status": r.status,
            "exclusion_reasons": list(r.exclusion_reasons),
            "empty_reason_codes": list(r.empty_reason_codes),
            "classify_text": r.classify_text,
            "body_text": r.body_text,
            "raw_text": r.raw_text,
        }
        for r in routes
    }


def _harmless_equivalent(left: str, right: str) -> bool:
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


def _normalize_response(text: str) -> str:
    simplified, _ = to_simplified(text or "")
    return apply_tolerance_key(simplified).strip()


def _is_critical_short_response(text: str, config: SelectionV3Config) -> bool:
    body = _normalize_response(text)
    if not body:
        return False
    catalog = [
        _normalize_response(x)
        for x in (config.critical_short_responses or [])
        if str(x).strip()
    ]
    if body in set(catalog):
        return True
    # Conservative: short polarity templates also block human_noise.
    if han_char_count(body) <= 4:
        pol = polarity_of_text(body, compile_lexicon(config))
        if pol in {"positive", "negative"}:
            return True
    return False


def build_family_states(
    routes: list[RouteView], config: SelectionV3Config
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
                _harmless_equivalent(texts[i], texts[j])
                for i in range(len(texts))
                for j in range(i + 1, len(texts))
            )
            if pairwise_ok:
                from audio_engine.core.selection_v3.semantic_risk import (
                    route_pair_has_semantic_conflict,
                )

                patterns = compile_lexicon(config)
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


def _family_counts(bundles: list[FamilyStateBundle]) -> dict[str, Any]:
    by_name = {b.family: b.state for b in bundles}
    reps = {
        b.family: b.representative_text
        for b in bundles
        if b.representative_text is not None
    }
    return {
        "stable_text_family_count": sum(
            1 for b in bundles if b.state == FAMILY_STATE_STABLE_TEXT
        ),
        "stable_empty_family_count": sum(
            1 for b in bundles if b.state == FAMILY_STATE_STABLE_EMPTY
        ),
        "unstable_family_count": sum(1 for b in bundles if b.state == FAMILY_STATE_UNSTABLE),
        "unavailable_family_count": sum(
            1 for b in bundles if b.state == FAMILY_STATE_UNAVAILABLE
        ),
        "family_state_by_name": by_name,
        "family_representative_text": reps,
    }


def _voicemail_hit(
    routes: list[RouteView],
    pattern: re.Pattern[str] | None,
) -> dict[str, Any] | None:
    if pattern is None:
        return None
    for route in routes:
        if route.route_disposition != ROUTE_ELIGIBLE:
            continue
        if route.status not in {RUN_STATUS_SUCCESS_TEXT, RUN_STATUS_SUCCESS_EMPTY}:
            continue
        for field_name, text in (
            ("body_text", route.body_text),
            ("classify_text", route.classify_text),
            ("transcript_text", route.transcript_text),
        ):
            if text and pattern.search(text):
                matched = pattern.search(text)
                return {
                    "run_id": route.run_id,
                    "family": route.family,
                    "field": field_name,
                    "snippet": text[:120],
                    "match": matched.group(0) if matched else "",
                }
    return None


def _semantic_risk_v2(
    bundles: list[FamilyStateBundle],
    config: SelectionV3Config,
) -> Any:
    """Only multi stable_text family conflicts. Empty-vs-text is not semantic_risk."""
    patterns = compile_lexicon(config)
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
        from audio_engine.core.selection_v3.semantic_risk_strict import _objects

        for a in pos:
            for b in neg:
                oa, ob = _objects(a.text), _objects(b.text)
                if (oa and ob and not oa.isdisjoint(ob)) or (
                    han_char_count(a.text) <= 4 and han_char_count(b.text) <= 4
                ):
                    from audio_engine.core.selection_v3.semantic_risk_strict import StrictRiskHit
                    from audio_engine.core.selection_v3.types import SUBTYPE_SEMANTIC_REVERSAL

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


def _environment_noise_decision(
    bundles: list[FamilyStateBundle],
    energy: AudioEnergyEvidence,
    config: SelectionV3Config,
) -> dict[str, Any] | None:
    counts = _family_counts(bundles)
    stable_text = [b for b in bundles if b.state == FAMILY_STATE_STABLE_TEXT]
    stable_empty = [b for b in bundles if b.state == FAMILY_STATE_STABLE_EMPTY]
    unstable = [b for b in bundles if b.state == FAMILY_STATE_UNSTABLE]
    unavailable = [b for b in bundles if b.state == FAMILY_STATE_UNAVAILABLE]
    state = energy.energy_state

    # D. too short — before other noise kinds when energy says so.
    if state == ENERGY_STATE_TOO_SHORT:
        return {
            "noise_kind": NOISE_KIND_AUDIO_TOO_SHORT,
            "classification_source": "auto_family_energy",
            "classification_confidence": "high",
            "reason": "environment_noise_audio_too_short",
        }

    # Borderline / failed energy cannot auto-enter noise.
    if state in {ENERGY_STATE_BORDERLINE, ENERGY_STATE_FAILED}:
        return None

    # Prohibit: ≥2 families support same text, semantic conflict handled upstream,
    # unique family unstable, failed counted as empty (already prevented), etc.
    if len(stable_text) >= 2:
        texts = [b.representative_text or "" for b in stable_text]
        if any(
            _harmless_equivalent(texts[i], texts[j])
            for i in range(len(texts))
            for j in range(i + 1, len(texts))
        ):
            return None

    # A. background
    if (
        stable_empty
        and not stable_text
        and not unstable
        and not unavailable
        and len(stable_empty) == len(bundles)
        and state == ENERGY_STATE_AUDIBLE
    ):
        return {
            "noise_kind": NOISE_KIND_BACKGROUND,
            "classification_source": "auto_family_energy",
            "classification_confidence": "high",
            "reason": "environment_noise_background",
        }

    # C. silence
    if (
        stable_empty
        and not stable_text
        and not unstable
        and not unavailable
        and len(stable_empty) == len(bundles)
        and state == ENERGY_STATE_INAUDIBLE
    ):
        return {
            "noise_kind": NOISE_KIND_SILENCE,
            "classification_source": "auto_family_energy",
            "classification_confidence": "high",
            "reason": "environment_noise_silence",
        }

    # B. human_noise
    if (
        len(stable_empty) >= 2
        and len(stable_text) == 1
        and not unstable
        and not unavailable
        and state == ENERGY_STATE_AUDIBLE
    ):
        only = stable_text[0]
        text = only.representative_text or ""
        if _is_critical_short_response(text, config):
            return None
        # Unique text unsupported by another family (already true by counts).
        noise_kind = NOISE_KIND_HUMAN_NOISE
        # Optional crosstalk refinement if trusted sidecar exists (not required).
        return {
            "noise_kind": noise_kind,
            "classification_source": "auto_family_energy",
            "classification_confidence": "medium",
            "reason": "environment_noise_human_noise",
            "unique_family": only.family,
            "unique_text": text,
            "family_counts": counts,
        }

    return None


def _gold_eligible(
    bundles: list[FamilyStateBundle],
    config: SelectionV3Config,
) -> tuple[bool, list[FamilyRep], str]:
    stable = [
        b for b in bundles if b.state == FAMILY_STATE_STABLE_TEXT and b.representative
    ]
    min_n = int(getattr(config, "gold_min_stable_families", 2) or 2)
    if len(stable) < min_n:
        return False, [], f"stable_families={len(stable)}<{min_n}"

    reps: list[FamilyRep] = []
    for bundle in stable:
        route = bundle.representative
        assert route is not None
        simplified, _ = to_simplified(route.classify_text)
        key = apply_tolerance_key(simplified)
        reps.append(
            FamilyRep(
                family=bundle.family,
                run_id=route.run_id,
                transcript_text=route.transcript_text or route.classify_text,
                raw_text=route.raw_text,
                tolerant_key=key,
                comparison_text=route.classify_text,
            )
        )
    for left, right in combinations(reps, 2):
        if not _harmless_equivalent(left.comparison_text, right.comparison_text):
            # Not all pairwise agree — may still have a agreeing majority pair.
            pass
    # Find a mutually agreeing subset of size >= min_n.
    agreeing: list[FamilyRep] = []
    for candidate in reps:
        if not agreeing:
            agreeing = [candidate]
            continue
        if all(
            _harmless_equivalent(candidate.comparison_text, other.comparison_text)
            for other in agreeing
        ):
            agreeing.append(candidate)
        # else skip opposing stable family for this cluster
    if len(agreeing) < min_n:
        # Try pairwise: any pair that agrees forms gold when min_n==2.
        for left, right in combinations(reps, 2):
            if _harmless_equivalent(left.comparison_text, right.comparison_text):
                cluster = [left, right]
                for extra in reps:
                    if extra in cluster:
                        continue
                    if all(
                        _harmless_equivalent(extra.comparison_text, x.comparison_text)
                        for x in cluster
                    ):
                        cluster.append(extra)
                if len(cluster) >= min_n:
                    agreeing = cluster
                    break
    if len(agreeing) < min_n:
        return False, [], "no_agreeing_stable_pair"

    # No other stable_text family may substantially oppose the agreeing cluster.
    agree_texts = {r.comparison_text for r in agreeing}
    agree_families = {r.family for r in agreeing}
    for bundle in bundles:
        if bundle.state != FAMILY_STATE_STABLE_TEXT:
            continue
        if bundle.family in agree_families:
            continue
        text = bundle.representative_text or ""
        if text and not any(_harmless_equivalent(text, t) for t in agree_texts):
            return False, [], f"opposing_stable_family:{bundle.family}"
        # Substantive opposition check
        for t in agree_texts:
            dist = tolerant_distance(apply_tolerance_key(text), apply_tolerance_key(t))
            if dist is not None and dist >= _SUBSTANTIVE_DIST:
                return False, [], f"opposing_stable_family:{bundle.family}"

    # Any unstable family blocks gold (028: dual-run must be stable).
    for bundle in bundles:
        if bundle.state == FAMILY_STATE_UNSTABLE:
            return False, [], f"family_unstable:{bundle.family}"

    # Stable empty third family is allowed (audit only).
    return True, agreeing, "ok"


def _hardcase_substantive(bundles: list[FamilyStateBundle]) -> tuple[bool, list[dict[str, Any]]]:
    texts: list[tuple[str, str]] = []
    for bundle in bundles:
        text = bundle.representative_text or bundle.opposing_text
        if text:
            texts.append((bundle.family, text))
    spans: list[dict[str, Any]] = []
    for (fa, ta), (fb, tb) in combinations(texts, 2):
        if _harmless_equivalent(ta, tb):
            continue
        dist = tolerant_distance(apply_tolerance_key(ta), apply_tolerance_key(tb))
        from audio_engine.core.selection_v3.semantic_risk_strict import _objects

        obj_a, obj_b = _objects(ta), _objects(tb)
        substantive = False
        if obj_a and obj_b and obj_a.isdisjoint(obj_b):
            # Different objects — substantive content divergence but NOT semantic_risk.
            substantive = True
        if dist is not None and dist >= _SUBSTANTIVE_DIST:
            substantive = True
        if not _harmless_equivalent(ta, tb) and (
            han_char_count(ta) >= 4 or han_char_count(tb) >= 4
        ):
            if dist is None or dist >= 0.15:
                substantive = True
        if substantive:
            spans.append(
                {
                    "family_a": fa,
                    "family_b": fb,
                    "text_a": ta,
                    "text_b": tb,
                    "distance": dist,
                }
            )
    return bool(spans), spans


def _base_result(**kwargs: Any) -> ClassificationResultV3:
    kwargs.setdefault("rule_version", RULE_VERSION_FIVE_CLASS_V2)
    kwargs.setdefault("outcome", OUTCOME_CLASSIFIED)
    kwargs.setdefault("annotation_state", "classified")
    return ClassificationResultV3(**kwargs)


def classify_five_class_v2(
    sample: Sample,
    config: SelectionV3Config,
    *,
    voicemail_pattern: re.Pattern[str] | None = None,
) -> ClassificationResultV3:
    routes = collect_route_views(sample, config)
    audit = _route_audit(routes)
    duration = float(sample.duration) if sample.duration is not None else None
    energy = energy_evidence_from_quality(
        sample.quality if isinstance(sample.quality, dict) else {},
        config=config,
        duration_sec=duration,
    )
    energy_dict = energy.as_dict()

    # Physical invalid → excluded (input contract failure, not a business class).
    if is_physically_invalid(sample):
        result = ClassificationResultV3(
            type=TYPE_INVALID_AUDIO,
            decision=DECISION_EXCLUDE,
            reason="broken_or_invalid_audio",
            category=None,
            status=STATUS_EXCLUDED,
            outcome=OUTCOME_EXCLUDED,
            rule_version=config.rule_version or RULE_VERSION_FIVE_CLASS_V2,
            configured_family_count=len(config.model_families),
            annotation_state="excluded",
            acoustic_evidence=energy_dict,
        )
        return apply_route_audit(result, config, routes)

    configured: list[str] = []
    for family in config.ordered_families():
        configured.extend(config.model_families.get(family, []))
    excluded_routes = [r for r in routes if r.route_disposition == ROUTE_EXCLUDED]
    if routes and all(r.route_disposition == ROUTE_EXCLUDED for r in routes):
        result = ClassificationResultV3(
            type="route_excluded",
            decision=DECISION_EXCLUDE,
            reason="all_routes_excluded",
            category=None,
            status=STATUS_EXCLUDED,
            outcome=OUTCOME_EXCLUDED,
            rule_version=config.rule_version or RULE_VERSION_FIVE_CLASS_V2,
            configured_family_count=len(config.model_families),
            reason_codes=["all_routes_excluded"],
            annotation_state="excluded",
            acoustic_evidence=energy_dict,
        )
        return apply_route_audit(result, config, routes)

    # 1. Voicemail
    vm = _voicemail_hit(routes, voicemail_pattern)
    if vm:
        result = _base_result(
            type=TYPE_VOICEMAIL_CANDIDATE,
            decision=DECISION_EXCLUDE,
            reason="voicemail_any_route",
            category=CATEGORY_VOICEMAIL,
            status=STATUS_EXCLUDED,
            reason_codes=["voicemail_any_route"],
            semantic_evidence=[vm],
            review_priority=PRIORITY_P2,
            review_queue="voicemail_isolation",
            candidate_text=vm.get("snippet"),
            configured_family_count=len(config.model_families),
            acoustic_evidence=energy_dict,
            classification_source="auto_rule",
            classification_confidence="high",
        )
        return apply_route_audit(result, config, routes)

    bundles = build_family_states(routes, config)
    counts = _family_counts(bundles)
    family_status = dict(counts["family_state_by_name"])

    def _attach_family_fields(result: ClassificationResultV3) -> ClassificationResultV3:
        result.family_status = family_status
        result.stable_text_family_count = counts["stable_text_family_count"]
        result.stable_empty_family_count = counts["stable_empty_family_count"]
        result.unstable_family_count = counts["unstable_family_count"]
        result.unavailable_family_count = counts["unavailable_family_count"]
        result.family_state_by_name = counts["family_state_by_name"]
        result.family_representative_text = counts["family_representative_text"]
        result.duration_ms = energy.duration_ms
        result.rms_dbfs = energy.rms_dbfs
        result.peak_dbfs = energy.peak_dbfs
        result.non_silent_ratio = energy.non_silent_ratio
        result.energy_state = energy.energy_state
        result.energy_policy_version = energy.energy_policy_version
        result.acoustic_evidence = {
            **(result.acoustic_evidence or {}),
            **energy_dict,
            "dnsmos_supplemental": {
                "dnsmos_ovrl": (sample.quality or {}).get("dnsmos_ovrl"),
                "noise_band": (sample.quality or {}).get("noise_band"),
                "calibrated": (sample.quality or {}).get("calibrated"),
            },
        }
        return result

    # 2. semantic_risk
    risk_hit = _semantic_risk_v2(bundles, config)
    if risk_hit is not None:
        result = _attach_family_fields(
            _base_result(
                type=TYPE_SEMANTIC_RISK,
                decision=DECISION_MANUAL_REVIEW,
                reason=risk_hit.subtype,
                category=CATEGORY_SEMANTIC_RISK,
                subtype=risk_hit.subtype,
                semantic_subtype=risk_hit.subtype,
                status=STATUS_MANUAL_REVIEW,
                reason_codes=[risk_hit.subtype],
                semantic_evidence=[risk_hit.evidence],
                review_priority=PRIORITY_P0,
                review_queue="manual_review",
                configured_family_count=len(config.model_families),
                classification_source="auto_rule",
                classification_confidence="high",
                needs_review=False,
            )
        )
        return apply_route_audit(result, config, routes)

    # 3. environment_noise (fully automatic)
    noise = _environment_noise_decision(bundles, energy, config)
    if noise is not None:
        # Optional crosstalk label from trusted sidecar only.
        q = sample.quality if isinstance(sample.quality, dict) else {}
        lab = sample.labels if isinstance(sample.labels, dict) else {}
        if noise["noise_kind"] == NOISE_KIND_HUMAN_NOISE and (
            q.get("human_crosstalk_confirmed") or lab.get("human_crosstalk_confirmed")
        ):
            noise = dict(noise)
            noise["noise_kind"] = NOISE_KIND_CROSSTALK
        result = _attach_family_fields(
            _base_result(
                type="environment_noise",
                decision=DECISION_EXCLUDE,
                reason=noise["reason"],
                category=CATEGORY_ENVIRONMENT_NOISE,
                status=STATUS_EXCLUDED,
                reason_codes=[noise["reason"]],
                noise_kind=noise["noise_kind"],
                review_priority=PRIORITY_P2,
                review_queue=None,
                configured_family_count=len(config.model_families),
                classification_source=noise["classification_source"],
                classification_confidence=noise["classification_confidence"],
                needs_review=False,
            )
        )
        return apply_route_audit(result, config, routes)

    # 4. gold_candidate
    ok, reps, gold_reason = _gold_eligible(bundles, config)
    if ok and reps:
        selection = select_weighted_family_text(
            reps,
            sample_id=sample.id,
            rule_version=config.rule_version or RULE_VERSION_FIVE_CLASS_V2,
            seed=str(
                getattr(config, "selection_seed", "") or RULE_VERSION_FIVE_CLASS_V2
            ),
            family_weights=getattr(config, "family_selection_weights", None) or {},
            family_order=config.ordered_families(),
        )
        assert selection is not None
        result = _attach_family_fields(
            _base_result(
                type=TYPE_PSEUDO_HIGH,
                decision=DECISION_AUDIT_PENDING,
                reason="gold_candidate",
                category=CATEGORY_GOLD_CANDIDATE,
                status=STATUS_CANDIDATE,
                reason_codes=["gold_candidate"],
                candidate_text=selection.transcript_text,
                selected_family=selection.family,
                selected_run_id=selection.run_id,
                selected_raw_text=selection.raw_text,
                label_source=LABEL_SOURCE_MODEL,
                label_tier=LABEL_TIER_PSEUDO_HIGH,
                is_human_verified=False,
                support_family_count=len(reps),
                selection_trace={
                    "policy": selection.selection_policy,
                    "weights": selection.selection_weights,
                    "draw": selection.selection_draw,
                    "seed": selection.selection_seed,
                    "version": selection.selection_version,
                    "representatives": selection.representatives,
                },
                selection_policy=selection.selection_policy,
                review_priority=PRIORITY_P2,
                review_queue="pseudo_audit",
                configured_family_count=len(config.model_families),
                classification_source="auto_rule",
                classification_confidence="high",
                needs_review=False,
            )
        )
        return apply_route_audit(result, config, routes)

    # 5. hardcase — sole fallback
    hardcase_reason = "hardcase_fallback"
    needs_review = True
    review_reason = ""
    spans: list[dict[str, Any]] = []

    if energy.energy_state == ENERGY_STATE_BORDERLINE:
        hardcase_reason = "energy_borderline"
        review_reason = "energy_state=borderline"
    elif energy.energy_state == ENERGY_STATE_FAILED:
        hardcase_reason = "energy_failed"
        review_reason = "energy_state=failed"
    elif any(b.state == FAMILY_STATE_UNSTABLE for b in bundles):
        hardcase_reason = "family_unstable"
        review_reason = "intra_family_dual_run_unstable"
    elif (
        counts["stable_empty_family_count"] >= 2
        and counts["stable_text_family_count"] == 1
    ):
        only = next(b for b in bundles if b.state == FAMILY_STATE_STABLE_TEXT)
        if _is_critical_short_response(only.representative_text or "", config):
            hardcase_reason = "critical_short_response_unsupported"
            review_reason = "unique_critical_short_response"
        else:
            hardcase_reason = "single_text_family_not_auto_noise"
            review_reason = "human_noise_conditions_incomplete"
    else:
        is_hard, spans = _hardcase_substantive(bundles)
        if is_hard:
            hardcase_reason = "substantive_divergence_non_strict_risk"
            review_reason = "multi_family_substantive_divergence"
        elif counts["unavailable_family_count"] or (
            counts["stable_text_family_count"] + counts["stable_empty_family_count"]
            < len(bundles)
        ):
            hardcase_reason = "insufficient_family_evidence"
            review_reason = "mixed_or_unavailable_families"
        elif excluded_routes and not any(
            r.route_disposition == ROUTE_ELIGIBLE for r in routes
        ):
            hardcase_reason = "insufficient_evidence_after_exclusion"
            review_reason = "exclusions_left_no_classify_evidence"
        else:
            hardcase_reason = f"unresolved:{gold_reason}" if gold_reason else "unresolved_rules"
            review_reason = hardcase_reason

    result = _attach_family_fields(
        _base_result(
            type=TYPE_HARDCASE,
            decision=DECISION_MANUAL_REVIEW,
            reason=hardcase_reason,
            category=CATEGORY_HARDCASE,
            status=STATUS_MANUAL_REVIEW,
            reason_codes=[hardcase_reason],
            semantic_evidence=spans,
            review_priority=PRIORITY_P0,
            review_queue="manual_review",
            configured_family_count=len(config.model_families),
            classification_source="auto_fallback",
            classification_confidence="low",
            needs_review=needs_review,
            review_reason=review_reason or hardcase_reason,
            hardcase_reason=hardcase_reason,
            label_source=LABEL_SOURCE_NONE,
            label_tier=LABEL_TIER_NONE,
        )
    )
    return apply_route_audit(result, config, routes)
