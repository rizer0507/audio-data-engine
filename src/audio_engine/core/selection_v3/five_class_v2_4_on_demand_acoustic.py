"""037 five-class classifier: selection_five_class_v2_4_on_demand_acoustic.

Business formal name: five-class v3 (`selection_five_class_v3`); same engine.
Default production rule_version is V3; V2_4 remains the engineering-compatible ID.

ASR pre-route (queues A-E) -> on-demand Silero VAD -> on-demand strict-zero
energy -> on-demand DNSMOS(SIG). EfficientAT is not on the default path.

Classification conditions follow jianban-0922; compute order follows on-demand-acoustic-0922.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from audio_engine.core.sample import Sample
from audio_engine.core.selection_v3.audio_energy import (
    AudioEnergyEvidence,
    energy_evidence_from_quality,
)
from audio_engine.core.selection_v3.classify_text import apply_route_audit
from audio_engine.core.selection_v3.config import SelectionV3Config
from audio_engine.core.selection_v3.family_evidence import RouteView, collect_route_views
from audio_engine.core.selection_v3.five_class_v2 import (
    FamilyStateBundle,
    _family_counts,
    _gold_eligible,
    _is_critical_short_response,
    _semantic_risk_v2,
    build_family_states,
)
from audio_engine.core.selection_v3.gold_select import select_weighted_family_text
from audio_engine.core.selection_v3.input_contract import is_physically_invalid
from audio_engine.core.selection_v3.result import ClassificationResultV3
from audio_engine.core.selection_v3.types import (
    ACOUSTIC_STATUS_NOT_REQUIRED,
    CATEGORY_ENVIRONMENT_NOISE,
    CATEGORY_GOLD_CANDIDATE,
    CATEGORY_HARDCASE,
    CATEGORY_SEMANTIC_RISK,
    CATEGORY_VOICEMAIL,
    DECISION_AUDIT_PENDING,
    DECISION_EXCLUDE,
    DECISION_MANUAL_REVIEW,
    FAMILY_STATE_STABLE_EMPTY,
    FAMILY_STATE_STABLE_TEXT,
    FAMILY_STATE_UNAVAILABLE,
    FAMILY_STATE_UNSTABLE,
    LABEL_SOURCE_MODEL,
    LABEL_SOURCE_NONE,
    LABEL_TIER_NONE,
    LABEL_TIER_PSEUDO_HIGH,
    NOISE_KIND_BACKGROUND_SUSPECTED,
    NOISE_KIND_HUMAN_NOISE,
    NOISE_KIND_SILENCE,
    OUTCOME_CLASSIFIED,
    OUTCOME_EXCLUDED,
    PRIORITY_P0,
    PRIORITY_P2,
    RISK_TAG_CRITICAL_POLARITY_RESPONSE,
    RISK_TAG_POLARITY_CONFLICT,
    RISK_TAG_QWEN_HALLUCINATION_SUSPECTED,
    ROUTE_ELIGIBLE,
    ROUTE_EXCLUDED,
    RULE_VERSION_FIVE_CLASS_V3,
    RUN_STATUS_FAILED,
    RUN_STATUS_MISSING,
    RUN_STATUS_SUCCESS_EMPTY,
    RUN_STATUS_SUCCESS_TEXT,
    STATUS_CANDIDATE,
    STATUS_EXCLUDED,
    STATUS_MANUAL_REVIEW,
    SUBTYPE_SEMANTIC_REVERSAL,
    SUBTYPE_SHORT_POLARITY_AMBIGUITY,
    TYPE_HARDCASE,
    TYPE_INVALID_AUDIO,
    TYPE_PSEUDO_HIGH,
    TYPE_SEMANTIC_RISK,
    TYPE_VOICEMAIL_CANDIDATE,
)

QUEUE_A = "A"
QUEUE_B = "B"
QUEUE_C = "C"
QUEUE_D = "D"
QUEUE_E = "E"
QUEUE_NONE = "none"
QUEUES_NEED_VAD = frozenset({QUEUE_A, QUEUE_B, QUEUE_C, QUEUE_D, QUEUE_E})

VAD_PRESENT = "present"
VAD_ABSENT = "absent"


@dataclass
class AsrRouteDecision:
    """ASR-only pre-route (step 2). queue_id is a compute queue, not a category."""

    queue_id: str = QUEUE_NONE
    early_category: str | None = None
    early_reason: str | None = None
    configured_families: list[str] = field(default_factory=list)
    family_states: dict[str, str] = field(default_factory=dict)
    all_families_dual_success: bool = False
    voicemail_hit: dict[str, Any] | None = None
    risk_tag: str | None = None
    risk_evidence: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "queue_id": self.queue_id,
            "early_category": self.early_category,
            "early_reason": self.early_reason,
            "configured_families": list(self.configured_families),
            "family_states": dict(self.family_states),
            "all_families_dual_success": self.all_families_dual_success,
            "voicemail_hit": self.voicemail_hit,
            "risk_tag": self.risk_tag,
            "risk_evidence": self.risk_evidence,
        }


def _raw_nonempty(route: RouteView) -> bool:
    return bool(str(route.raw_text or "").strip())


def _family_routes(routes: list[RouteView], family: str) -> list[RouteView]:
    return [r for r in routes if r.family == family]


def _family_dual_run_success(routes: list[RouteView], family: str, expected: int) -> bool:
    """Both (expected) runs succeeded (text or empty); no failed/missing."""
    members = _family_routes(routes, family)
    if len(members) < expected:
        return False
    if any(r.status in {RUN_STATUS_FAILED, RUN_STATUS_MISSING} for r in members):
        return False
    success = [
        r
        for r in members
        if r.status in {RUN_STATUS_SUCCESS_TEXT, RUN_STATUS_SUCCESS_EMPTY}
    ]
    return len(success) >= expected


def _family_final_all_empty(bundle: FamilyStateBundle) -> bool:
    return bundle.state == FAMILY_STATE_STABLE_EMPTY


def _all_routes_raw_empty(routes: list[RouteView]) -> bool:
    return all(not _raw_nonempty(r) for r in routes)


def _voicemail_hit_raw(
    routes: list[RouteView],
    pattern: re.Pattern[str] | None,
) -> dict[str, Any] | None:
    """Match voicemail patterns against raw ASR text (pre-clean)."""
    if pattern is None:
        return None
    for route in routes:
        if route.status in {RUN_STATUS_FAILED, RUN_STATUS_MISSING}:
            continue
        text = str(route.raw_text or "")
        if not text.strip():
            continue
        matched = pattern.search(text)
        if matched:
            return {
                "run_id": route.run_id,
                "family": route.family,
                "field": "raw_text",
                "snippet": text[:120],
                "match": matched.group(0),
            }
    return None


def _map_risk_tag(subtype: str | None) -> str | None:
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


def route_asr_queue(
    routes: list[RouteView],
    bundles: list[FamilyStateBundle],
    config: SelectionV3Config,
    *,
    voicemail_pattern: re.Pattern[str] | None = None,
) -> AsrRouteDecision:
    """Steps 2+: voicemail then ASR queues A-E / early hardcase. No acoustics."""
    families = list(config.ordered_families())
    expected = int(config.expected_runs_per_family or 2)
    counts = _family_counts(bundles)
    family_status = dict(counts["family_state_by_name"])
    decision = AsrRouteDecision(
        configured_families=families,
        family_states=family_status,
    )

    vm = _voicemail_hit_raw(routes, voicemail_pattern)
    if vm:
        decision.early_category = CATEGORY_VOICEMAIL
        decision.early_reason = "voicemail_any_route_raw"
        decision.voicemail_hit = vm
        decision.queue_id = QUEUE_NONE
        return decision

    all_dual = all(_family_dual_run_success(routes, f, expected) for f in families)
    decision.all_families_dual_success = all_dual

    target = str(config.target_family or "qwen")
    other_families = [f for f in families if f != target]
    target_routes = _family_routes(routes, target)
    target_raw_once = any(_raw_nonempty(r) for r in target_routes)
    others_final_empty = bool(other_families) and all(
        _family_final_all_empty(b) for b in bundles if b.family in other_families
    )

    # A before unstable hardcase (Qwen may be unstable text).
    if all_dual and target_raw_once and others_final_empty:
        decision.queue_id = QUEUE_A
        return decision

    # B: all dual success + all raw empty.
    if all_dual and _all_routes_raw_empty(routes):
        decision.queue_id = QUEUE_B
        return decision

    # Unstable after A/B reserved.
    if any(b.state == FAMILY_STATE_UNSTABLE for b in bundles):
        decision.early_category = CATEGORY_HARDCASE
        decision.early_reason = "family_unstable"
        decision.queue_id = QUEUE_NONE
        return decision

    if any(b.state == FAMILY_STATE_UNAVAILABLE for b in bundles):
        decision.early_category = CATEGORY_HARDCASE
        decision.early_reason = "family_unavailable"
        decision.queue_id = QUEUE_NONE
        return decision

    stable_text = [b for b in bundles if b.state == FAMILY_STATE_STABLE_TEXT]
    stable_empty = [b for b in bundles if b.state == FAMILY_STATE_STABLE_EMPTY]
    has_negation = bool(stable_empty)

    # C: ?? stable text + polarity conflict + no negation votes.
    if len(stable_text) >= 2 and not has_negation:
        risk = _semantic_risk_v2(bundles, config)
        mapped = _map_risk_tag(getattr(risk, "subtype", None) if risk else None)
        if risk is not None and mapped == RISK_TAG_POLARITY_CONFLICT:
            decision.queue_id = QUEUE_C
            decision.risk_tag = RISK_TAG_POLARITY_CONFLICT
            decision.risk_evidence = getattr(risk, "evidence", None) or {
                "subtype": getattr(risk, "subtype", None)
            }
            return decision

        # D: critical short ambiguity, not C.
        if mapped == RISK_TAG_CRITICAL_POLARITY_RESPONSE:
            decision.queue_id = QUEUE_D
            decision.risk_tag = RISK_TAG_CRITICAL_POLARITY_RESPONSE
            decision.risk_evidence = getattr(risk, "evidence", None) or {
                "subtype": getattr(risk, "subtype", None)
            }
            return decision

        # Fallback D: ?? critical-short texts that are not harmless-equivalent
        # and not a clear polarity conflict (already handled).
        from audio_engine.core.selection_v3.five_class_v2 import _harmless_equivalent

        critical_reps = [
            b
            for b in stable_text
            if _is_critical_short_response(b.representative_text or "", config)
        ]
        if len(critical_reps) >= 2:
            texts = [b.representative_text or "" for b in critical_reps]
            all_same = all(
                _harmless_equivalent(texts[i], texts[j])
                for i in range(len(texts))
                for j in range(i + 1, len(texts))
            )
            if not all_same:
                decision.queue_id = QUEUE_D
                decision.risk_tag = RISK_TAG_CRITICAL_POLARITY_RESPONSE
                decision.risk_evidence = {
                    "families": [b.family for b in critical_reps],
                    "texts": texts,
                }
                return decision

    # E: gold shape (?? stable same + remaining nonempty same + no negation).
    if not has_negation:
        ok, _reps, _reason = _gold_eligible(bundles, config)
        if ok:
            decision.queue_id = QUEUE_E
            return decision

    decision.early_category = CATEGORY_HARDCASE
    decision.early_reason = "asr_route_no_queue"
    decision.queue_id = QUEUE_NONE
    return decision


def needs_silero_v2_4(route: AsrRouteDecision) -> dict[str, Any]:
    required = route.queue_id in QUEUES_NEED_VAD and route.early_category is None
    return {
        "required": required,
        "queue_id": route.queue_id,
        "reasons": [f"queue_{route.queue_id}"] if required else [],
        "trigger_version": "silero_vad_candidates_v2_4",
    }


def needs_energy_v2_4(
    route: AsrRouteDecision,
    *,
    vad_speech_state: str | None,
) -> dict[str, Any]:
    vad = str(vad_speech_state or "").strip().lower()
    required = (
        route.queue_id == QUEUE_B
        and route.early_category is None
        and vad == VAD_ABSENT
    )
    return {
        "required": required,
        "queue_id": route.queue_id,
        "vad_speech_state": vad or None,
        "reasons": ["queue_B_vad_absent"] if required else [],
        "trigger_version": "audio_energy_candidates_v2_4",
    }


def needs_dnsmos_v2_4(
    route: AsrRouteDecision,
    *,
    vad_speech_state: str | None,
) -> dict[str, Any]:
    vad = str(vad_speech_state or "").strip().lower()
    required = (
        route.queue_id in {QUEUE_A, QUEUE_B}
        and route.early_category is None
        and vad == VAD_PRESENT
    )
    return {
        "required": required,
        "queue_id": route.queue_id,
        "vad_speech_state": vad or None,
        "reasons": [f"queue_{route.queue_id}_vad_present"] if required else [],
        "trigger_version": "dnsmos_candidates_v2_4",
    }


def _sig_is_low(
    quality: dict[str, Any],
    config: SelectionV3Config,
) -> tuple[bool | None, str]:
    """Return (is_low, reason). None means unavailable ? must not guess."""
    status = str(quality.get("dnsmos_status") or "").strip().lower()
    if status == ACOUSTIC_STATUS_NOT_REQUIRED or status == "not_required":
        return None, "dnsmos_not_required"
    if status != "success":
        return None, f"dnsmos_status_{status or 'missing'}"
    sig = quality.get("dnsmos_sig")
    if sig is None:
        return None, "sig_missing"
    try:
        sig_f = float(sig)
    except (TypeError, ValueError):
        return None, "sig_invalid"
    # SIG ?????????????????
    weak = getattr(config, "dnsmos_weak_sig", None)
    if weak is None:
        return None, "weak_sig_unconfigured"
    try:
        weak_f = float(weak)
    except (TypeError, ValueError):
        return None, "weak_sig_invalid"
    return sig_f <= weak_f, "ok"


def _vad_state(quality: dict[str, Any]) -> str:
    if str(quality.get("vad_status") or "").strip().lower() == "failed":
        return "failed"
    if str(quality.get("vad_status") or "").strip().lower() == ACOUSTIC_STATUS_NOT_REQUIRED:
        return ACOUSTIC_STATUS_NOT_REQUIRED
    return str(quality.get("vad_speech_state") or "").strip().lower()


def _strict_zero_flag(quality: dict[str, Any]) -> tuple[bool | None, str]:
    """None = missing/failed check; True = all channels strict zero."""
    status = str(
        quality.get("energy_status") or quality.get("energy_compute_status") or ""
    ).strip().lower()
    if status == ACOUSTIC_STATUS_NOT_REQUIRED:
        return None, "energy_not_required"
    flag = quality.get("strict_all_channels_zero")
    if flag is None:
        energy_state = str(quality.get("energy_state") or "").strip().lower()
        if energy_state == "failed" or status == "failed":
            return None, "energy_failed"
        return None, "strict_zero_missing"
    return bool(flag), "ok"


def _base_result_v24(**kwargs: Any) -> ClassificationResultV3:
    """Legacy helper kept for imports; prefer nested helper inside classify."""
    kwargs.setdefault("rule_version", RULE_VERSION_FIVE_CLASS_V3)
    kwargs.setdefault("outcome", OUTCOME_CLASSIFIED)
    kwargs.setdefault("annotation_state", "classified")
    return ClassificationResultV3(**kwargs)


def classify_five_class_v2_4_on_demand_acoustic(
    sample: Sample,
    config: SelectionV3Config,
    *,
    voicemail_pattern: re.Pattern[str] | None = None,
) -> ClassificationResultV3:
    routes = collect_route_views(sample, config)
    duration = float(sample.duration) if sample.duration is not None else None
    quality = dict(sample.quality) if isinstance(sample.quality, dict) else {}
    energy = energy_evidence_from_quality(quality, config=config, duration_sec=duration)
    energy_dict = energy.as_dict()
    rule_ver = str(config.rule_version or RULE_VERSION_FIVE_CLASS_V3)

    def _base_result_v24(**kwargs: Any) -> ClassificationResultV3:
        kwargs.setdefault("rule_version", rule_ver)
        kwargs.setdefault("outcome", OUTCOME_CLASSIFIED)
        kwargs.setdefault("annotation_state", "classified")
        return ClassificationResultV3(**kwargs)

    if is_physically_invalid(sample):
        result = ClassificationResultV3(
            type=TYPE_INVALID_AUDIO,
            decision=DECISION_EXCLUDE,
            reason="broken_or_invalid_audio",
            category=None,
            status=STATUS_EXCLUDED,
            outcome=OUTCOME_EXCLUDED,
            rule_version=rule_ver,
            configured_family_count=len(config.model_families),
            annotation_state="excluded",
            acoustic_evidence=energy_dict,
            decision_evidence={
                "decision": "excluded",
                "reason": "broken_or_invalid_audio",
                "queue_id": QUEUE_NONE,
            },
        )
        return apply_route_audit(result, config, routes)

    if routes and all(r.route_disposition == ROUTE_EXCLUDED for r in routes):
        result = ClassificationResultV3(
            type="route_excluded",
            decision=DECISION_EXCLUDE,
            reason="all_routes_excluded",
            category=None,
            status=STATUS_EXCLUDED,
            outcome=OUTCOME_EXCLUDED,
            rule_version=config.rule_version or RULE_VERSION_FIVE_CLASS_V3,
            configured_family_count=len(config.model_families),
            reason_codes=["all_routes_excluded"],
            annotation_state="excluded",
            acoustic_evidence=energy_dict,
            decision_evidence={
                "decision": "excluded",
                "reason": "all_routes_excluded",
                "queue_id": QUEUE_NONE,
            },
        )
        return apply_route_audit(result, config, routes)

    bundles = build_family_states(routes, config)
    counts = _family_counts(bundles)
    family_status = dict(counts["family_state_by_name"])

    # Prefer precomputed route from labels (pipeline phase); else compute.
    labels = sample.labels if isinstance(sample.labels, dict) else {}
    precomputed = labels.get("asr_route_v2_4")
    if isinstance(precomputed, dict) and precomputed.get("queue_id") is not None:
        route = AsrRouteDecision(
            queue_id=str(precomputed.get("queue_id") or QUEUE_NONE),
            early_category=precomputed.get("early_category"),
            early_reason=precomputed.get("early_reason"),
            configured_families=list(
                precomputed.get("configured_families") or config.ordered_families()
            ),
            family_states=dict(precomputed.get("family_states") or family_status),
            all_families_dual_success=bool(precomputed.get("all_families_dual_success")),
            voicemail_hit=precomputed.get("voicemail_hit"),
            risk_tag=precomputed.get("risk_tag"),
            risk_evidence=precomputed.get("risk_evidence"),
        )
    else:
        route = route_asr_queue(
            routes, bundles, config, voicemail_pattern=voicemail_pattern
        )

    vad = _vad_state(quality)
    silero_req = needs_silero_v2_4(route)
    energy_req = needs_energy_v2_4(route, vad_speech_state=vad)
    dnsmos_req = needs_dnsmos_v2_4(route, vad_speech_state=vad)

    def _decision_evidence(
        *,
        decision: str,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "queue_id": route.queue_id,
            "early_category": route.early_category,
            "early_reason": route.early_reason,
            "configured_families": list(route.configured_families),
            "family_states": family_status,
            "silero_computed": bool(silero_req["required"]),
            "energy_computed": bool(energy_req["required"]),
            "dnsmos_computed": bool(dnsmos_req["required"]),
            "vad_speech_state": quality.get("vad_speech_state"),
            "vad_status": quality.get("vad_status"),
            "strict_all_channels_zero": quality.get("strict_all_channels_zero"),
            "dnsmos_sig": quality.get("dnsmos_sig"),
            "dnsmos_status": quality.get("dnsmos_status"),
            "energy_state": energy.energy_state,
            "decision": decision,
            "risk_tag": route.risk_tag,
        }
        if extra:
            payload.update(extra)
        return payload

    def _attach(result: ClassificationResultV3) -> ClassificationResultV3:
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
        result.vad_speech_state = quality.get("vad_speech_state")
        result.acoustic_evidence = {
            **(result.acoustic_evidence or {}),
            **energy_dict,
            "vad_speech_state": quality.get("vad_speech_state"),
            "vad_status": quality.get("vad_status"),
            "speech_ratio": quality.get("speech_ratio"),
            "strict_all_channels_zero": quality.get("strict_all_channels_zero"),
            "any_channel_nonzero": quality.get("any_channel_nonzero"),
            "dnsmos_sig": quality.get("dnsmos_sig"),
            "dnsmos_bak": quality.get("dnsmos_bak"),
            "dnsmos_ovrl": quality.get("dnsmos_ovrl"),
            "dnsmos_status": quality.get("dnsmos_status"),
            "queue_id": route.queue_id,
        }
        return result

    # --- Early exits (no acoustics) ---
    if route.early_category == CATEGORY_VOICEMAIL:
        result = _attach(
            _base_result_v24(
                type=TYPE_VOICEMAIL_CANDIDATE,
                decision=DECISION_EXCLUDE,
                reason="voicemail_any_route_raw",
                category=CATEGORY_VOICEMAIL,
                status=STATUS_EXCLUDED,
                reason_codes=["voicemail_any_route_raw"],
                semantic_evidence=[route.voicemail_hit] if route.voicemail_hit else [],
                review_priority=PRIORITY_P2,
                review_queue="voicemail_isolation",
                candidate_text=(route.voicemail_hit or {}).get("snippet"),
                configured_family_count=len(config.model_families),
                classification_source="auto_rule",
                classification_confidence="high",
                evidence_sources=["family_state_v2", "asr_route_v2_4"],
                decision_evidence=_decision_evidence(
                    decision=CATEGORY_VOICEMAIL,
                    extra={"rule_branch": "voicemail"},
                ),
            )
        )
        return apply_route_audit(result, config, routes)

    if route.early_category == CATEGORY_HARDCASE:
        reason = route.early_reason or "hardcase_fallback"
        result = _attach(
            _base_result_v24(
                type=TYPE_HARDCASE,
                decision=DECISION_MANUAL_REVIEW,
                reason=reason,
                category=CATEGORY_HARDCASE,
                status=STATUS_MANUAL_REVIEW,
                reason_codes=[reason],
                review_priority=PRIORITY_P0,
                review_queue="manual_review",
                configured_family_count=len(config.model_families),
                classification_source="auto_fallback",
                classification_confidence="low",
                needs_review=True,
                review_reason=reason,
                hardcase_reason=reason,
                label_source=LABEL_SOURCE_NONE,
                label_tier=LABEL_TIER_NONE,
                evidence_sources=["family_state_v2", "asr_route_v2_4"],
                decision_evidence=_decision_evidence(
                    decision=CATEGORY_HARDCASE,
                    extra={"rule_branch": reason},
                ),
            )
        )
        return apply_route_audit(result, config, routes)

    def _hard(reason: str, *, branch: str | None = None) -> ClassificationResultV3:
        return _attach(
            _base_result_v24(
                type=TYPE_HARDCASE,
                decision=DECISION_MANUAL_REVIEW,
                reason=reason,
                category=CATEGORY_HARDCASE,
                status=STATUS_MANUAL_REVIEW,
                reason_codes=[reason],
                review_priority=PRIORITY_P0,
                review_queue="manual_review",
                configured_family_count=len(config.model_families),
                classification_source="auto_fallback",
                classification_confidence="low",
                needs_review=True,
                review_reason=reason,
                hardcase_reason=reason,
                label_source=LABEL_SOURCE_NONE,
                label_tier=LABEL_TIER_NONE,
                evidence_sources=["family_state_v2", "asr_route_v2_4", "on_demand_acoustic"],
                decision_evidence=_decision_evidence(
                    decision=CATEGORY_HARDCASE,
                    extra={"rule_branch": branch or reason},
                ),
            )
        )

    def _env(*, kind: str, reason: str, branch: str) -> ClassificationResultV3:
        return _attach(
            _base_result_v24(
                type="environment_noise",
                decision=DECISION_EXCLUDE,
                reason=reason,
                category=CATEGORY_ENVIRONMENT_NOISE,
                status=STATUS_EXCLUDED,
                reason_codes=[reason],
                noise_kind=kind,
                review_priority=PRIORITY_P2,
                configured_family_count=len(config.model_families),
                classification_source="auto_family_acoustic",
                classification_confidence="high",
                evidence_sources=["family_state_v2", "asr_route_v2_4", "on_demand_acoustic"],
                decision_evidence=_decision_evidence(
                    decision=CATEGORY_ENVIRONMENT_NOISE,
                    extra={"rule_branch": branch, "noise_kind": kind},
                ),
            )
        )

    def _semantic(*, tag: str, reason: str, evidence: Any = None) -> ClassificationResultV3:
        return _attach(
            _base_result_v24(
                type=TYPE_SEMANTIC_RISK,
                decision=DECISION_MANUAL_REVIEW,
                reason=reason,
                category=CATEGORY_SEMANTIC_RISK,
                subtype=tag,
                semantic_subtype=tag,
                status=STATUS_MANUAL_REVIEW,
                reason_codes=[tag],
                semantic_evidence=[evidence] if evidence else [],
                review_priority=PRIORITY_P0,
                review_queue="manual_review",
                configured_family_count=len(config.model_families),
                classification_source="auto_rule",
                classification_confidence="high",
                evidence_sources=["family_state_v2", "asr_route_v2_4", "on_demand_acoustic"],
                decision_evidence=_decision_evidence(
                    decision=CATEGORY_SEMANTIC_RISK,
                    extra={"rule_branch": tag, "risk_tag": tag},
                ),
            )
        )

    def _gold(*, branch: str) -> ClassificationResultV3 | None:
        ok, gold_reps, _gold_reason = _gold_eligible(bundles, config)
        if not ok or not gold_reps:
            return None
        selection = select_weighted_family_text(
            gold_reps,
            sample_id=sample.id,
            rule_version=config.rule_version or RULE_VERSION_FIVE_CLASS_V3,
            seed=str(
                getattr(config, "selection_seed", "") or RULE_VERSION_FIVE_CLASS_V3
            ),
            family_weights=getattr(config, "family_selection_weights", None) or {},
            family_order=config.ordered_families(),
        )
        if selection is None:
            return None
        return _attach(
            _base_result_v24(
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
                support_family_count=len(gold_reps),
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
                evidence_sources=["family_state_v2", "asr_route_v2_4", "silero_vad_v1"],
                decision_evidence=_decision_evidence(
                    decision=CATEGORY_GOLD_CANDIDATE,
                    extra={"rule_branch": branch},
                ),
            )
        )

    # --- Step 4: VAD required for A?E ---
    if route.queue_id not in QUEUES_NEED_VAD:
        return apply_route_audit(_hard("asr_route_no_queue"), config, routes)

    if vad in {"failed", "", ACOUSTIC_STATUS_NOT_REQUIRED} or vad not in {
        VAD_PRESENT,
        VAD_ABSENT,
    }:
        reason = "vad_failed" if vad == "failed" else "vad_unavailable"
        return apply_route_audit(_hard(reason, branch=f"queue_{route.queue_id}_{reason}"), config, routes)

    # C / D / E finalize on VAD only.
    if route.queue_id == QUEUE_C:
        if vad == VAD_PRESENT:
            return apply_route_audit(
                _semantic(
                    tag=RISK_TAG_POLARITY_CONFLICT,
                    reason=RISK_TAG_POLARITY_CONFLICT,
                    evidence=route.risk_evidence,
                ),
                config,
                routes,
            )
        return apply_route_audit(
            _hard("polarity_conflict_no_speech", branch="queue_C_vad_absent"),
            config,
            routes,
        )

    if route.queue_id == QUEUE_D:
        if vad == VAD_PRESENT:
            return apply_route_audit(
                _semantic(
                    tag=RISK_TAG_CRITICAL_POLARITY_RESPONSE,
                    reason=RISK_TAG_CRITICAL_POLARITY_RESPONSE,
                    evidence=route.risk_evidence,
                ),
                config,
                routes,
            )
        return apply_route_audit(
            _hard("critical_polarity_no_speech", branch="queue_D_vad_absent"),
            config,
            routes,
        )

    if route.queue_id == QUEUE_E:
        if vad == VAD_PRESENT:
            gold = _gold(branch="queue_E_vad_present")
            if gold is not None:
                return apply_route_audit(gold, config, routes)
            return apply_route_audit(_hard("gold_ineligible"), config, routes)
        return apply_route_audit(
            _hard("gold_no_speech", branch="queue_E_vad_absent"),
            config,
            routes,
        )

    # A / B branches.
    if route.queue_id == QUEUE_B and vad == VAD_ABSENT:
        # Step 4.1: strict all-zero energy only.
        zero, zero_reason = _strict_zero_flag(quality)
        if zero is None:
            return apply_route_audit(
                _hard("strict_zero_unavailable", branch=zero_reason),
                config,
                routes,
            )
        if zero:
            return apply_route_audit(
                _env(
                    kind=NOISE_KIND_SILENCE,
                    reason="environment_noise_silence",
                    branch="queue_B_silence",
                ),
                config,
                routes,
            )
        return apply_route_audit(
            _env(
                kind=NOISE_KIND_BACKGROUND_SUSPECTED,
                reason="environment_noise_background_suspected",
                branch="queue_B_background_suspected",
            ),
            config,
            routes,
        )

    if route.queue_id in {QUEUE_A, QUEUE_B} and vad == VAD_PRESENT:
        # Step 5: DNSMOS SIG only.
        is_low, sig_reason = _sig_is_low(quality, config)
        if is_low is None:
            return apply_route_audit(
                _hard("sig_unavailable", branch=sig_reason),
                config,
                routes,
            )
        if not is_low:
            return apply_route_audit(
                _hard("sig_not_low", branch="sig_not_low"),
                config,
                routes,
            )
        if route.queue_id == QUEUE_A:
            return apply_route_audit(
                _semantic(
                    tag=RISK_TAG_QWEN_HALLUCINATION_SUSPECTED,
                    reason=RISK_TAG_QWEN_HALLUCINATION_SUSPECTED,
                ),
                config,
                routes,
            )
        return apply_route_audit(
            _env(
                kind=NOISE_KIND_HUMAN_NOISE,
                reason="environment_noise_human_noise",
                branch="queue_B_human_noise",
            ),
            config,
            routes,
        )

    if route.queue_id == QUEUE_A and vad == VAD_ABSENT:
        return apply_route_audit(
            _hard("qwen_hallucination_no_speech", branch="queue_A_vad_absent"),
            config,
            routes,
        )

    return apply_route_audit(_hard("hardcase_fallback"), config, routes)
