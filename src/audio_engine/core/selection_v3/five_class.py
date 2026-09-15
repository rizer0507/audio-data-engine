"""027 five-class classifier: selection_five_class_v1.

Order: per-route exclusion → all-excluded exit → any-route voicemail →
strict semantic_risk (②>③>①) → confirmed environment_noise → gold_candidate →
hardcase → direct manual_annotation. No pending_evidence exit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

from audio_engine.core.sample import Sample
from audio_engine.core.selection_v3.acoustic_evidence import collect_acoustic_evidence
from audio_engine.core.selection_v3.annotation_tasks import (
    AnnotationTask,
    build_annotation_task,
    merge_annotation_tasks,
)
from audio_engine.core.selection_v3.classify_text import apply_route_audit
from audio_engine.core.selection_v3.config import SelectionV3Config
from audio_engine.core.selection_v3.family_evidence import RouteView, collect_route_views
from audio_engine.core.selection_v3.gold_select import FamilyRep, select_weighted_family_text
from audio_engine.core.selection_v3.input_contract import is_physically_invalid
from audio_engine.core.selection_v3.noise_trigger import ensure_trigger_record, uses_asr_anomaly_noise
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
    LABEL_SOURCE_MODEL,
    LABEL_SOURCE_NONE,
    LABEL_TIER_NONE,
    LABEL_TIER_PSEUDO_HIGH,
    OUTCOME_CLASSIFIED,
    OUTCOME_EXCLUDED,
    OUTCOME_MANUAL_ANNOTATION,
    PRIORITY_P0,
    PRIORITY_P1,
    PRIORITY_P2,
    ROUTE_ELIGIBLE,
    ROUTE_EXCLUDED,
    ROUTE_FAILED,
    ROUTE_MISSING,
    RULE_VERSION_FIVE_CLASS,
    RUN_STATUS_FAILED,
    RUN_STATUS_MISSING,
    RUN_STATUS_SUCCESS_EMPTY,
    RUN_STATUS_SUCCESS_TEXT,
    STATUS_CANDIDATE,
    STATUS_EXCLUDED,
    STATUS_MANUAL_REVIEW,
    TASK_RESOLVE_SEMANTICS,
    TASK_TRANSCRIBE,
    TASK_VERIFY_TARGET_SPEECH,
    TYPE_HARDCASE,
    TYPE_INVALID_AUDIO,
    TYPE_PSEUDO_HIGH,
    TYPE_SEMANTIC_RISK,
    TYPE_VOICEMAIL_CANDIDATE,
)

_SUBSTANTIVE_DIST = 0.30


@dataclass
class FamilyBundle:
    family: str
    routes: list[RouteView]
    eligible: list[RouteView]
    excluded: list[RouteView]
    failed: list[RouteView]
    representative: RouteView | None = None
    stable_text: bool = False
    stable_empty: bool = False
    unstable: bool = False
    opposing_text: str | None = None


@dataclass
class Decision:
    outcome: str
    category: str | None
    reason: str
    subtype: str | None = None
    type_: str = TYPE_HARDCASE
    decision: str = DECISION_MANUAL_REVIEW
    status: str = STATUS_MANUAL_REVIEW
    candidate: str | None = None
    selected_family: str | None = None
    selected_run_id: str | None = None
    selection_trace: dict[str, Any] = field(default_factory=dict)
    risk_tags: list[str] = field(default_factory=list)
    reason_codes: list[str] = field(default_factory=list)
    tasks: list[AnnotationTask] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)
    review_priority: str | None = PRIORITY_P1
    review_queue: str | None = "manual_review"
    noise_kind: str | None = None
    skip_noise: bool = False


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
    """Pairwise harmless check. No transitive clustering."""
    if left == right:
        return True
    a, _ = to_simplified(left)
    b, _ = to_simplified(right)
    if a == b:
        return True
    ka = apply_tolerance_key(a)
    kb = apply_tolerance_key(b)
    if ka == kb:
        # Block question/particle flips that change request intent.
        # 需要吗 vs 需要啊: 吗 stays, 啊 drops → keys differ unless both lose particle.
        if ("吗" in a) != ("吗" in b) or ("嗎" in a) != ("嗎" in b):
            return False
        if ("吧" in a) != ("吧" in b):
            return False
        return True
    return False


def _build_families(routes: list[RouteView], config: SelectionV3Config) -> list[FamilyBundle]:
    bundles: list[FamilyBundle] = []
    for family in config.ordered_families():
        members = [r for r in routes if r.family == family]
        excluded = [r for r in members if r.route_disposition == ROUTE_EXCLUDED]
        failed = [
            r
            for r in members
            if r.route_disposition in {ROUTE_FAILED, ROUTE_MISSING}
            or r.status in {RUN_STATUS_FAILED, RUN_STATUS_MISSING}
        ]
        eligible = [
            r
            for r in members
            if r.route_disposition == ROUTE_ELIGIBLE
            and r.status in {RUN_STATUS_SUCCESS_TEXT, RUN_STATUS_SUCCESS_EMPTY}
            and r not in failed
        ]
        # Failed routes already filtered; keep eligible that are not excluded.
        eligible = [r for r in eligible if r.route_disposition != ROUTE_EXCLUDED]
        text_routes = [
            r for r in eligible if r.status == RUN_STATUS_SUCCESS_TEXT and r.classify_text
        ]
        empty_routes = [
            r
            for r in eligible
            if r.status == RUN_STATUS_SUCCESS_EMPTY or not r.classify_text
        ]
        bundle = FamilyBundle(
            family=family,
            routes=members,
            eligible=eligible,
            excluded=excluded,
            failed=failed,
        )
        expected = config.expected_runs_per_family
        if len(text_routes) >= expected:
            texts = [r.classify_text for r in text_routes]
            if len(set(texts)) == 1 or all(
                _harmless_equivalent(texts[0], t) for t in texts[1:]
            ):
                # Dual-run polarity conflict → unstable.
                if len(set(texts)) > 1 and not all(
                    _harmless_equivalent(texts[i], texts[j])
                    for i in range(len(texts))
                    for j in range(i + 1, len(texts))
                ):
                    bundle.unstable = True
                    bundle.opposing_text = texts[0]
                else:
                    # Check polarity conflict within family.
                    from audio_engine.core.selection_v3.semantic_risk import (
                        route_pair_has_semantic_conflict,
                    )

                    patterns = compile_lexicon(config)
                    conflict = False
                    for a, b in combinations(text_routes, 2):
                        if route_pair_has_semantic_conflict(
                            a.classify_text, b.classify_text, patterns
                        ):
                            conflict = True
                            break
                    if conflict:
                        bundle.unstable = True
                        bundle.opposing_text = text_routes[0].classify_text
                    else:
                        bundle.stable_text = True
                        bundle.representative = sorted(
                            text_routes, key=lambda r: r.run_id
                        )[0]
            else:
                bundle.unstable = True
                bundle.opposing_text = text_routes[0].classify_text
        elif len(text_routes) == 1 and expected >= 2:
            # Single remaining route: not a verified stable support, but opposition counts.
            bundle.opposing_text = text_routes[0].classify_text
            bundle.representative = text_routes[0]
            bundle.unstable = True
        elif eligible and not text_routes and len(empty_routes) >= expected:
            bundle.stable_empty = True
        elif text_routes and empty_routes:
            bundle.unstable = True
            bundle.opposing_text = text_routes[0].classify_text
            bundle.representative = text_routes[0]
        bundles.append(bundle)
    return bundles


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
        # Match body_text and classify_text; excluded routes already skipped.
        for field_name, text in (
            ("body_text", route.body_text),
            ("classify_text", route.classify_text),
            ("transcript_text", route.transcript_text),
        ):
            if text and pattern.search(text):
                return {
                    "run_id": route.run_id,
                    "family": route.family,
                    "field": field_name,
                    "snippet": text[:120],
                    "match": pattern.search(text).group(0),
                }
    return None


def _gold_eligible(
    bundles: list[FamilyBundle],
    config: SelectionV3Config,
) -> tuple[bool, list[FamilyRep], str]:
    stable = [b for b in bundles if b.stable_text and b.representative]
    min_n = int(getattr(config, "gold_min_stable_families", 3) or 3)
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
    # Pairwise equivalence — no transitive merge.
    for left, right in combinations(reps, 2):
        if not _harmless_equivalent(left.comparison_text, right.comparison_text):
            return False, [], f"pairwise_mismatch:{left.family}/{right.family}"

    # Other eligible Chinese routes must not substantially oppose.
    stable_texts = {r.comparison_text for r in reps}
    for bundle in bundles:
        if bundle.stable_text:
            continue
        if bundle.opposing_text:
            if not any(_harmless_equivalent(bundle.opposing_text, t) for t in stable_texts):
                return False, [], f"opposing_family:{bundle.family}"
        # Eligible non-stable text routes
        for route in bundle.eligible:
            if route.classify_text and not any(
                _harmless_equivalent(route.classify_text, t) for t in stable_texts
            ):
                # Failed/excluded already out of eligible.
                if route.status == RUN_STATUS_SUCCESS_TEXT:
                    return False, [], f"opposing_route:{route.run_id}"
        if bundle.unstable and bundle.opposing_text:
            # Intra-family polarity conflict is opposition.
            return False, [], f"family_unstable:{bundle.family}"
    return True, reps, "ok"


def _hardcase_substantive(bundles: list[FamilyBundle]) -> tuple[bool, list[dict[str, Any]]]:
    stable_text = [b for b in bundles if b.stable_text and b.representative]
    if len(stable_text) < 2:
        # Also consider unstable families with opposing text as disagreement signals.
        with_text = [
            b
            for b in bundles
            if (b.representative and b.representative.classify_text)
            or b.opposing_text
        ]
        if len(with_text) < 2:
            return False, []
    spans: list[dict[str, Any]] = []
    texts: list[tuple[str, str]] = []
    for bundle in bundles:
        text = None
        if bundle.representative and bundle.representative.classify_text:
            text = bundle.representative.classify_text
        elif bundle.opposing_text:
            text = bundle.opposing_text
        if text:
            texts.append((bundle.family, text))
    for (fa, ta), (fb, tb) in combinations(texts, 2):
        if _harmless_equivalent(ta, tb):
            continue
        dist = tolerant_distance(apply_tolerance_key(ta), apply_tolerance_key(tb))
        # Substantive if objects/actions differ or distance high.
        from audio_engine.core.selection_v3.semantic_risk_strict import _objects

        obj_a, obj_b = _objects(ta), _objects(tb)
        substantive = False
        if obj_a and obj_b and obj_a.isdisjoint(obj_b):
            substantive = True
        if dist is not None and dist >= _SUBSTANTIVE_DIST:
            substantive = True
        if not _harmless_equivalent(ta, tb) and (
            han_char_count(ta) >= 4 or han_char_count(tb) >= 4
        ):
            # Long non-equivalent bodies count as substantive divergence.
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


def _needs_noise_diagnosis(bundles: list[FamilyBundle], patterns) -> list[str]:
    """027 §9: never trigger on foreign/echo exclusions."""
    reasons: list[str] = []
    stable_empty = [b for b in bundles if b.stable_empty]
    stable_text = [b for b in bundles if b.stable_text and b.representative]
    if stable_empty and stable_text:
        for bundle in stable_text:
            pol = polarity_of_text(bundle.representative.classify_text, patterns)
            if pol in {"positive", "negative"}:
                reasons.append("empty_vs_polarity")
                break
    # Short polarity conflict among eligible families.
    short_pol = []
    for bundle in stable_text:
        text = bundle.representative.classify_text
        if han_char_count(text) <= 4:
            short_pol.append((bundle.family, text, polarity_of_text(text, patterns)))
    pols = {p for _, _, p in short_pol if p in {"positive", "negative"}}
    if len(pols) >= 2:
        reasons.append("short_polarity_conflict")
    return reasons


def classify_five_class(
    sample: Sample,
    config: SelectionV3Config,
    *,
    voicemail_pattern: re.Pattern[str] | None = None,
) -> ClassificationResultV3:
    patterns = compile_lexicon(config)
    routes = collect_route_views(sample, config)
    audit = _route_audit(routes)
    audio_refs = _audio_refs(sample)
    noise_calls = getattr(config, "noise_call_counter", None)

    def _count_noise() -> None:
        if isinstance(noise_calls, list):
            noise_calls.append(1)

    # Physical invalid → excluded (no manual task for broken input contract).
    if is_physically_invalid(sample):
        result = ClassificationResultV3(
            type=TYPE_INVALID_AUDIO,
            decision=DECISION_EXCLUDE,
            reason="broken_or_invalid_audio",
            category=None,
            status=STATUS_EXCLUDED,
            outcome=OUTCOME_EXCLUDED,
            rule_version=config.rule_version or RULE_VERSION_FIVE_CLASS,
            configured_family_count=len(config.model_families),
            annotation_state="excluded",
        )
        result.route_disposition_by_run = {
            r.run_id: r.route_disposition for r in routes
        }
        result.exclusion_reasons_by_run = {
            r.run_id: list(r.exclusion_reasons) for r in routes
        }
        return apply_route_audit(result, config, routes)

    configured = []
    for family in config.ordered_families():
        configured.extend(config.model_families.get(family, []))
    excluded_routes = [r for r in routes if r.route_disposition == ROUTE_EXCLUDED]
    non_excluded = [r for r in routes if r.route_disposition != ROUTE_EXCLUDED]
    all_configured_excluded = bool(configured) and all(
        r.route_disposition == ROUTE_EXCLUDED for r in routes if r.run_id in set(configured)
    ) and len(excluded_routes) == len(routes)

    # §5.5 / §11.4: all routes excluded → excluded exit, zero tasks, zero noise.
    if all_configured_excluded or (
        routes and all(r.route_disposition == ROUTE_EXCLUDED for r in routes)
    ):
        result = ClassificationResultV3(
            type="route_excluded",
            decision=DECISION_EXCLUDE,
            reason="all_routes_excluded",
            category=None,
            status=STATUS_EXCLUDED,
            outcome=OUTCOME_EXCLUDED,
            rule_version=config.rule_version or RULE_VERSION_FIVE_CLASS,
            configured_family_count=len(config.model_families),
            reason_codes=["all_routes_excluded"],
            annotation_state="excluded",
            review_priority=None,
            review_queue=None,
        )
        return apply_route_audit(result, config, routes)

    # Partial exclude + rest failed/missing → manual, not all-excluded.
    remaining_eligible = [
        r for r in non_excluded if r.route_disposition == ROUTE_ELIGIBLE
    ]
    remaining_failed = [
        r
        for r in non_excluded
        if r.route_disposition in {ROUTE_FAILED, ROUTE_MISSING}
        or r.status in {RUN_STATUS_FAILED, RUN_STATUS_MISSING}
    ]
    if not remaining_eligible and remaining_failed:
        task = build_annotation_task(
            sample_id=sample.id,
            task_type=TASK_TRANSCRIBE,
            questions=["部分路次被前置排除，其余失败/缺失，请听音转写。"],
            audio_refs=audio_refs,
            route_audit=audit,
            evidence={"excluded_count": len(excluded_routes), "failed_count": len(remaining_failed)},
            rule_version=config.rule_version or RULE_VERSION_FIVE_CLASS,
        )
        result = ClassificationResultV3(
            type=TYPE_HARDCASE,
            decision=DECISION_MANUAL_REVIEW,
            reason="partial_exclude_rest_failed",
            category=None,
            status=STATUS_MANUAL_REVIEW,
            outcome=OUTCOME_MANUAL_ANNOTATION,
            rule_version=config.rule_version or RULE_VERSION_FIVE_CLASS,
            configured_family_count=len(config.model_families),
            reason_codes=["partial_exclude_rest_failed"],
            annotation_tasks=[task.as_dict()],
            annotation_state="manual_annotation",
            review_priority=PRIORITY_P1,
            review_queue="manual_review",
        )
        return apply_route_audit(result, config, routes)

    # Voicemail: any eligible route hit → immediate end.
    vm = _voicemail_hit(routes, voicemail_pattern)
    if vm:
        result = ClassificationResultV3(
            type=TYPE_VOICEMAIL_CANDIDATE,
            decision=DECISION_EXCLUDE,
            reason="voicemail_any_route",
            category=CATEGORY_VOICEMAIL,
            status=STATUS_EXCLUDED,
            outcome=OUTCOME_CLASSIFIED,
            subtype=None,
            rule_version=config.rule_version or RULE_VERSION_FIVE_CLASS,
            configured_family_count=len(config.model_families),
            reason_codes=["voicemail_any_route"],
            semantic_evidence=[vm],
            annotation_state="classified",
            review_priority=PRIORITY_P2,
            review_queue="voicemail_isolation",
            candidate_text=vm.get("snippet"),
        )
        return apply_route_audit(result, config, routes)

    bundles = _build_families(routes, config)

    def _acoustic_payload() -> dict[str, Any]:
        payload = collect_acoustic_evidence(sample.quality, sample.labels).as_dict()
        q = sample.quality if isinstance(sample.quality, dict) else {}
        lab = sample.labels if isinstance(sample.labels, dict) else {}
        for key in (
            "short_response_missed",
            "target_speech_present",
            "polarity_syllable_unintelligible",
            "generally_unintelligible",
        ):
            if key in q:
                payload[key] = q.get(key)
            elif key in lab:
                payload[key] = lab.get(key)
        return payload

    acoustic = _acoustic_payload()

    # Noise diagnosis only when needed (after voicemail). Foreign never triggers.
    trigger_reasons = _needs_noise_diagnosis(bundles, patterns)
    diagnosis: dict[str, Any] = {}
    if trigger_reasons and uses_asr_anomaly_noise(config):
        _count_noise()
        diagnosis = ensure_trigger_record(sample, config) or {}
        acoustic = _acoustic_payload()
    elif trigger_reasons:
        diagnosis = {"status": "not_scored", "trigger_reasons": trigger_reasons}

    # Strict semantic risk.
    polarity_views: list[FamilyPolarityView] = []
    for bundle in bundles:
        if bundle.stable_text and bundle.representative:
            text = bundle.representative.classify_text
            polarity_views.append(
                FamilyPolarityView(
                    family=bundle.family,
                    text=text,
                    polarity=polarity_of_text(text, patterns),
                    stable=True,
                    is_empty=False,
                    run_ids=[r.run_id for r in bundle.eligible],
                )
            )
        elif bundle.stable_empty:
            polarity_views.append(
                FamilyPolarityView(
                    family=bundle.family,
                    text="",
                    polarity="unknown",
                    stable=True,
                    is_empty=True,
                    run_ids=[r.run_id for r in bundle.eligible],
                )
            )

    human_no_target = bool(
        sample.labels.get("human_no_target_speech")
        or sample.labels.get("no_target_speech_trusted")
    )
    human_unintelligible = bool(
        sample.labels.get("polarity_syllable_unintelligible")
        or (sample.quality or {}).get("polarity_syllable_unintelligible")
    )
    risk = analyze_strict_risks(
        polarity_views,
        patterns=patterns,
        acoustic=acoustic,
        max_han_chars=int(getattr(config, "short_polarity_max_han_chars", 4) or 4),
        human_verified_no_target=human_no_target,
        human_unintelligible_polarity=human_unintelligible,
    )
    if risk.hit and risk.hit.formal:
        task = build_annotation_task(
            sample_id=sample.id,
            task_type=TASK_RESOLVE_SEMANTICS,
            questions=[
                f"严格语义风险子类 {risk.hit.subtype}：请判定正确转写或目标语音情况。"
            ],
            audio_refs=audio_refs,
            route_audit=audit,
            evidence=risk.hit.evidence,
            conflict_spans=risk.hit.conflict_spans,
            rule_version=config.rule_version or RULE_VERSION_FIVE_CLASS,
        )
        result = ClassificationResultV3(
            type=TYPE_SEMANTIC_RISK,
            decision=DECISION_MANUAL_REVIEW,
            reason=risk.hit.subtype,
            category=CATEGORY_SEMANTIC_RISK,
            subtype=risk.hit.subtype,
            semantic_subtype=risk.hit.subtype,
            status=STATUS_MANUAL_REVIEW,
            outcome=OUTCOME_CLASSIFIED,
            rule_version=config.rule_version or RULE_VERSION_FIVE_CLASS,
            configured_family_count=len(config.model_families),
            reason_codes=[risk.hit.subtype],
            semantic_evidence=[risk.hit.evidence],
            acoustic_evidence=acoustic,
            annotation_tasks=[task.as_dict()],
            annotation_state="classified",
            review_priority=PRIORITY_P0,
            review_queue="manual_review",
            noise_diagnosis=dict(diagnosis or {}),
        )
        return apply_route_audit(result, config, routes)

    # Confirmed environment noise (not low DNSMOS alone).
    if acoustic.get("state") in {"environment_confirmed", "crosstalk_confirmed", "background_only"}:
        # If hallucination formal already handled; here noise is main class.
        if acoustic.get("state") == "background_only" and not (
            acoustic.get("no_target_speech") is True
            or human_no_target
        ):
            pass
        else:
            noise_kind = "crosstalk" if "crosstalk" in str(acoustic.get("state")) else "background"
            if acoustic.get("human_crosstalk_confirmed") and acoustic.get("background_only"):
                noise_kind = "mixed"
            task = build_annotation_task(
                sample_id=sample.id,
                task_type=TASK_VERIFY_TARGET_SPEECH,
                questions=["请确认是否存在可可靠转写的目标语音；标注噪声属性。"],
                audio_refs=audio_refs,
                route_audit=audit,
                evidence=acoustic,
                rule_version=config.rule_version or RULE_VERSION_FIVE_CLASS,
            )
            result = ClassificationResultV3(
                type="environment_noise",
                decision=DECISION_MANUAL_REVIEW,
                reason="environment_noise_confirmed",
                category=CATEGORY_ENVIRONMENT_NOISE,
                status=STATUS_MANUAL_REVIEW,
                outcome=OUTCOME_CLASSIFIED,
                rule_version=config.rule_version or RULE_VERSION_FIVE_CLASS,
                configured_family_count=len(config.model_families),
                reason_codes=["environment_noise_confirmed"],
                acoustic_evidence=acoustic,
                noise_kind=noise_kind,
                annotation_tasks=[task.as_dict()],
                annotation_state="classified",
                review_priority=PRIORITY_P1,
                review_queue="manual_review",
                noise_diagnosis=dict(diagnosis or {}),
            )
            return apply_route_audit(result, config, routes)

    # Gold candidate.
    ok, reps, gold_reason = _gold_eligible(bundles, config)
    if ok and reps:
        # Unresolved acoustic doubts block gold.
        if trigger_reasons and acoustic.get("state") == "unknown":
            task = build_annotation_task(
                sample_id=sample.id,
                task_type=TASK_VERIFY_TARGET_SPEECH,
                questions=["存在空/有字或短应答冲突但缺少可靠声学证据，请听音判定。"],
                audio_refs=audio_refs,
                route_audit=audit,
                evidence={"trigger_reasons": trigger_reasons, "risk_questions": risk.candidate_questions},
                rule_version=config.rule_version or RULE_VERSION_FIVE_CLASS,
            )
            result = ClassificationResultV3(
                type=TYPE_HARDCASE,
                decision=DECISION_MANUAL_REVIEW,
                reason="gold_blocked_missing_acoustic",
                category=None,
                status=STATUS_MANUAL_REVIEW,
                outcome=OUTCOME_MANUAL_ANNOTATION,
                rule_version=config.rule_version or RULE_VERSION_FIVE_CLASS,
                configured_family_count=len(config.model_families),
                reason_codes=["gold_blocked_missing_acoustic"],
                annotation_tasks=[task.as_dict()],
                annotation_state="manual_annotation",
                review_priority=PRIORITY_P1,
                review_queue="manual_review",
                noise_diagnosis=dict(diagnosis or {}),
            )
            return apply_route_audit(result, config, routes)

        selection = select_weighted_family_text(
            reps,
            sample_id=sample.id,
            rule_version=config.rule_version or RULE_VERSION_FIVE_CLASS,
            seed=str(getattr(config, "selection_seed", "") or RULE_VERSION_FIVE_CLASS),
            family_weights=getattr(config, "family_selection_weights", None) or {},
            family_order=config.ordered_families(),
        )
        assert selection is not None
        result = ClassificationResultV3(
            type=TYPE_PSEUDO_HIGH,
            decision=DECISION_AUDIT_PENDING,
            reason="gold_candidate",
            category=CATEGORY_GOLD_CANDIDATE,
            status=STATUS_CANDIDATE,
            outcome=OUTCOME_CLASSIFIED,
            rule_version=config.rule_version or RULE_VERSION_FIVE_CLASS,
            configured_family_count=len(config.model_families),
            support_family_count=len(reps),
            reason_codes=["gold_candidate"],
            candidate_text=selection.transcript_text,
            selected_family=selection.family,
            selected_run_id=selection.run_id,
            selected_raw_text=selection.raw_text,
            label_source=LABEL_SOURCE_MODEL,
            label_tier=LABEL_TIER_PSEUDO_HIGH,
            is_human_verified=False,
            selection_trace={
                "policy": selection.selection_policy,
                "weights": selection.selection_weights,
                "draw": selection.selection_draw,
                "seed": selection.selection_seed,
                "version": selection.selection_version,
                "representatives": selection.representatives,
            },
            selection_policy=selection.selection_policy,
            annotation_state="classified",
            review_priority=PRIORITY_P2,
            review_queue="pseudo_audit",
            noise_diagnosis=dict(diagnosis or {}),
        )
        return apply_route_audit(result, config, routes)

    # Hardcase: substantive multi-family divergence.
    is_hard, spans = _hardcase_substantive(bundles)
    if is_hard:
        questions = ["多家族实质分歧，请听音给出正确转写。"]
        if risk.candidate_questions:
            questions.extend(risk.candidate_questions)
        task = build_annotation_task(
            sample_id=sample.id,
            task_type=TASK_TRANSCRIBE if not risk.candidate_questions else TASK_RESOLVE_SEMANTICS,
            questions=questions,
            audio_refs=audio_refs,
            route_audit=audit,
            evidence={"gold_block_reason": gold_reason},
            conflict_spans=spans,
            rule_version=config.rule_version or RULE_VERSION_FIVE_CLASS,
        )
        result = ClassificationResultV3(
            type=TYPE_HARDCASE,
            decision=DECISION_MANUAL_REVIEW,
            reason="hardcase_substantive_divergence",
            category=CATEGORY_HARDCASE,
            status=STATUS_MANUAL_REVIEW,
            outcome=OUTCOME_CLASSIFIED,
            rule_version=config.rule_version or RULE_VERSION_FIVE_CLASS,
            configured_family_count=len(config.model_families),
            reason_codes=["hardcase_substantive_divergence"],
            semantic_evidence=spans,
            annotation_tasks=[task.as_dict()],
            annotation_state="classified",
            review_priority=PRIORITY_P0,
            review_queue="manual_review",
            noise_diagnosis=dict(diagnosis or {}),
        )
        return apply_route_audit(result, config, routes)

    # Remainder → direct manual annotation (not hardcase dump, not pending pool).
    questions = []
    task_type = TASK_TRANSCRIBE
    if risk.candidate_questions:
        questions.extend(risk.candidate_questions)
        task_type = TASK_RESOLVE_SEMANTICS
    if trigger_reasons:
        questions.append("请核验目标语音是否存在及噪声/串音情况。")
        if task_type == TASK_TRANSCRIBE:
            task_type = TASK_VERIFY_TARGET_SPEECH
    if gold_reason:
        questions.append(f"未达金标条件: {gold_reason}")
    if not questions:
        questions = ["自动证据不足，请听音转写。"]
    tasks = merge_annotation_tasks(
        [
            build_annotation_task(
                sample_id=sample.id,
                task_type=task_type,
                questions=questions,
                audio_refs=audio_refs,
                route_audit=audit,
                evidence={
                    "gold_block_reason": gold_reason,
                    "trigger_reasons": trigger_reasons,
                    "wide_triggers": risk.wide_triggers,
                },
                rule_version=config.rule_version or RULE_VERSION_FIVE_CLASS,
            )
        ]
    )
    result = ClassificationResultV3(
        type=TYPE_HARDCASE,
        decision=DECISION_MANUAL_REVIEW,
        reason="manual_annotation_insufficient_evidence",
        category=None,
        status=STATUS_MANUAL_REVIEW,
        outcome=OUTCOME_MANUAL_ANNOTATION,
        rule_version=config.rule_version or RULE_VERSION_FIVE_CLASS,
        configured_family_count=len(config.model_families),
        reason_codes=["manual_annotation_insufficient_evidence", gold_reason],
        annotation_tasks=[t.as_dict() for t in tasks],
        annotation_state="manual_annotation",
        review_priority=PRIORITY_P1,
        review_queue="manual_review",
        acoustic_evidence=acoustic,
        noise_diagnosis=dict(diagnosis or {}),
        label_source=LABEL_SOURCE_NONE,
        label_tier=LABEL_TIER_NONE,
    )
    return apply_route_audit(result, config, routes)
