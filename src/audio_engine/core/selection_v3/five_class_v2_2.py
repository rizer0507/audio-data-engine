"""029 five-class classifier: selection_five_class_v2_2_auto_noise.

Incremental DNSMOS joint evidence on top of 028 / five_class_v2.
Order unchanged: voicemail → semantic_risk → environment_noise →
gold_candidate → hardcase.
"""

from __future__ import annotations

import re
from typing import Any

from audio_engine.core.sample import Sample
from audio_engine.core.selection_v3.audio_energy import (
    AudioEnergyEvidence,
    energy_evidence_from_quality,
)
from audio_engine.core.selection_v3.classify_text import apply_route_audit
from audio_engine.core.selection_v3.config import SelectionV3Config
from audio_engine.core.selection_v3.dnsmos_decision import (
    DNSMOS_NOISE_CLEAN,
    DNSMOS_NOISE_MODERATE,
    DNSMOS_NOISE_NOISY,
    DNSMOS_NOISE_UNAVAILABLE,
    DNSMOS_SPEECH_STRONG,
    DnsmosDecisionConfig,
    DnsmosDecisionEvidence,
    derive_dnsmos_decision,
)
from audio_engine.core.selection_v3.family_evidence import collect_route_views
from audio_engine.core.selection_v3.five_class_v2 import (
    FamilyStateBundle,
    _family_counts,
    _gold_eligible,
    _hardcase_substantive,
    _is_critical_short_response,
    _semantic_risk_v2,
    _voicemail_hit,
    build_family_states,
    classify_five_class_v2,
)
from audio_engine.core.selection_v3.gold_select import select_weighted_family_text
from audio_engine.core.selection_v3.input_contract import is_physically_invalid
from audio_engine.core.selection_v3.result import ClassificationResultV3
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
    PRIORITY_P2,
    ROUTE_ELIGIBLE,
    ROUTE_EXCLUDED,
    RULE_VERSION_FIVE_CLASS_V2,
    RULE_VERSION_FIVE_CLASS_V2_2,
    STATUS_CANDIDATE,
    STATUS_EXCLUDED,
    STATUS_MANUAL_REVIEW,
    TYPE_HARDCASE,
    TYPE_INVALID_AUDIO,
    TYPE_PSEUDO_HIGH,
    TYPE_SEMANTIC_RISK,
    TYPE_VOICEMAIL_CANDIDATE,
)


def _dnsmos_config_from_selection(config: SelectionV3Config) -> DnsmosDecisionConfig:
    return DnsmosDecisionConfig(
        policy_version=str(config.dnsmos_decision_policy_version or "dnsmos_decision_v2_2"),
        clean_bak=float(config.dnsmos_clean_bak),
        clean_ovrl=float(config.dnsmos_clean_ovrl),
        noisy_bak=float(config.dnsmos_noisy_bak),
        noisy_ovrl=float(config.dnsmos_noisy_ovrl),
        strong_sig=float(config.dnsmos_strong_sig),
        weak_sig=float(config.dnsmos_weak_sig),
        noisy_operator=str(config.dnsmos_noisy_operator or "or"),
        require_status_success=bool(config.dnsmos_require_status_success),
    )

def _base_result_v22(**kwargs: Any) -> ClassificationResultV3:
    kwargs.setdefault("rule_version", RULE_VERSION_FIVE_CLASS_V2_2)
    kwargs.setdefault("outcome", OUTCOME_CLASSIFIED)
    kwargs.setdefault("annotation_state", "classified")
    return ClassificationResultV3(**kwargs)

def _environment_noise_decision_v22(
    bundles: list[FamilyStateBundle],
    energy: AudioEnergyEvidence,
    config: SelectionV3Config,
    dnsmos: DnsmosDecisionEvidence,
) -> dict[str, Any] | None:
    """028 noise rules + 029 DNSMOS confidence / borderline / contradiction."""
    counts = _family_counts(bundles)
    stable_text = [b for b in bundles if b.state == FAMILY_STATE_STABLE_TEXT]
    stable_empty = [b for b in bundles if b.state == FAMILY_STATE_STABLE_EMPTY]
    unstable = [b for b in bundles if b.state == FAMILY_STATE_UNSTABLE]
    unavailable = [b for b in bundles if b.state == FAMILY_STATE_UNAVAILABLE]
    state = energy.energy_state
    all_empty = (
        bool(stable_empty)
        and not stable_text
        and not unstable
        and not unavailable
        and len(stable_empty) == len(bundles)
    )
    human_shape = (
        len(stable_empty) >= 2
        and len(stable_text) == 1
        and not unstable
        and not unavailable
    )

    # D. too short — DNSMOS not required.
    if state == ENERGY_STATE_TOO_SHORT:
        return {
            "noise_kind": NOISE_KIND_AUDIO_TOO_SHORT,
            "classification_source": "auto_family_energy",
            "classification_confidence": "high",
            "reason": "environment_noise_audio_too_short",
            "rule_branch": "audio_too_short",
            "v2_fallback": False,
            "evidence_sources": ["family_state_v2", "audio_energy_v1"],
            "dnsmos_consumed": False,
        }

    # Failed energy cannot auto-enter noise.
    if state == ENERGY_STATE_FAILED:
        return None

    # ≥2 agreeing stable texts → not environment noise (gold path).
    if len(stable_text) >= 2:
        from audio_engine.core.selection_v3.five_class_v2 import _harmless_equivalent

        texts = [b.representative_text or "" for b in stable_text]
        if any(
            _harmless_equivalent(texts[i], texts[j])
            for i in range(len(texts))
            for j in range(i + 1, len(texts))
        ):
            return None

    # C. silence — DNSMOS must not override inaudible → human_noise.
    if all_empty and state == ENERGY_STATE_INAUDIBLE:
        return {
            "noise_kind": NOISE_KIND_SILENCE,
            "classification_source": "auto_family_energy",
            "classification_confidence": "high",
            "reason": "environment_noise_silence",
            "rule_branch": "silence",
            "v2_fallback": False,
            "evidence_sources": ["family_state_v2", "audio_energy_v1"],
            "dnsmos_consumed": False,
        }

    # A. background (+ DNSMOS)
    if all_empty and state == ENERGY_STATE_AUDIBLE:
        if (
            dnsmos.noise_state == DNSMOS_NOISE_CLEAN
            and dnsmos.speech_state == DNSMOS_SPEECH_STRONG
        ):
            return {
                "hardcase": True,
                "hardcase_reason": "empty_asr_but_clean_strong_speech",
                "reason": "empty_asr_but_clean_strong_speech",
                "rule_branch": "background_dnsmos_contradiction",
                "v2_fallback": False,
                "evidence_sources": [
                    "family_state_v2",
                    "audio_energy_v1",
                    "dnsmos_decision_v2_2",
                ],
                "dnsmos_consumed": True,
                "classification_source": "auto_family_energy_dnsmos",
                "classification_confidence": "low",
            }
        if dnsmos.noise_state == DNSMOS_NOISE_NOISY:
            return {
                "noise_kind": NOISE_KIND_BACKGROUND,
                "classification_source": "auto_family_energy_dnsmos",
                "classification_confidence": "high",
                "reason": "environment_noise_background",
                "rule_branch": "background_dnsmos_noisy",
                "v2_fallback": False,
                "evidence_sources": [
                    "family_state_v2",
                    "audio_energy_v1",
                    "dnsmos_decision_v2_2",
                ],
                "dnsmos_consumed": True,
            }
        if dnsmos.noise_state == DNSMOS_NOISE_MODERATE:
            return {
                "noise_kind": NOISE_KIND_BACKGROUND,
                "classification_source": "auto_family_energy_dnsmos",
                "classification_confidence": "medium",
                "reason": "environment_noise_background",
                "rule_branch": "background_dnsmos_moderate",
                "v2_fallback": False,
                "evidence_sources": [
                    "family_state_v2",
                    "audio_energy_v1",
                    "dnsmos_decision_v2_2",
                ],
                "dnsmos_consumed": True,
            }
        if dnsmos.noise_state == DNSMOS_NOISE_CLEAN:
            # clean without strong speech: keep background, medium confidence
            return {
                "noise_kind": NOISE_KIND_BACKGROUND,
                "classification_source": "auto_family_energy_dnsmos",
                "classification_confidence": "medium",
                "reason": "environment_noise_background",
                "rule_branch": "background_dnsmos_clean_non_strong",
                "v2_fallback": False,
                "evidence_sources": [
                    "family_state_v2",
                    "audio_energy_v1",
                    "dnsmos_decision_v2_2",
                ],
                "dnsmos_consumed": True,
            }
        # unavailable → audited v2 fallback
        return {
            "noise_kind": NOISE_KIND_BACKGROUND,
            "classification_source": "auto_family_energy",
            "classification_confidence": "medium",
            "reason": "environment_noise_background",
            "rule_branch": "background_v2_fallback",
            "v2_fallback": True,
            "fallback_policy": RULE_VERSION_FIVE_CLASS_V2,
            "evidence_sources": ["family_state_v2", "audio_energy_v1"],
            "dnsmos_consumed": False,
        }

    # Borderline + all empty + DNSMOS noisy → background medium
    if (
        all_empty
        and state == ENERGY_STATE_BORDERLINE
        and config.dnsmos_decision_resolve_borderline_when_noisy
        and dnsmos.noise_state == DNSMOS_NOISE_NOISY
    ):
        return {
            "noise_kind": NOISE_KIND_BACKGROUND,
            "classification_source": "auto_family_energy_dnsmos",
            "classification_confidence": "medium",
            "reason": "environment_noise_background_borderline_dnsmos",
            "rule_branch": "borderline_background_dnsmos_noisy",
            "v2_fallback": False,
            "borderline_resolved": True,
            "evidence_sources": [
                "family_state_v2",
                "audio_energy_v1",
                "dnsmos_decision_v2_2",
            ],
            "dnsmos_consumed": True,
        }

    # B. human_noise (+ DNSMOS)
    if human_shape and state in {ENERGY_STATE_AUDIBLE, ENERGY_STATE_BORDERLINE}:
        only = stable_text[0]
        text = only.representative_text or ""
        if _is_critical_short_response(text, config):
            return None
        if state == ENERGY_STATE_BORDERLINE:
            if not (
                config.dnsmos_decision_resolve_borderline_when_noisy
                and dnsmos.noise_state == DNSMOS_NOISE_NOISY
            ):
                return None
            return {
                "noise_kind": NOISE_KIND_HUMAN_NOISE,
                "classification_source": "auto_family_energy_dnsmos",
                "classification_confidence": "medium",
                "reason": "environment_noise_human_noise_borderline_dnsmos",
                "rule_branch": "borderline_human_noise_dnsmos_noisy",
                "v2_fallback": False,
                "borderline_resolved": True,
                "unique_family": only.family,
                "unique_text": text,
                "family_counts": counts,
                "evidence_sources": [
                    "family_state_v2",
                    "audio_energy_v1",
                    "dnsmos_decision_v2_2",
                ],
                "dnsmos_consumed": True,
            }
        # audible human_noise
        if dnsmos.noise_state == DNSMOS_NOISE_NOISY:
            conf = "high"
            source = "auto_family_energy_dnsmos"
            branch = "human_noise_dnsmos_noisy"
            v2_fb = False
            evidence = [
                "family_state_v2",
                "audio_energy_v1",
                "dnsmos_decision_v2_2",
            ]
            consumed = True
        elif dnsmos.noise_state in {DNSMOS_NOISE_MODERATE, DNSMOS_NOISE_CLEAN}:
            conf = "medium"
            source = "auto_family_energy_dnsmos"
            branch = f"human_noise_dnsmos_{dnsmos.noise_state}"
            v2_fb = False
            evidence = [
                "family_state_v2",
                "audio_energy_v1",
                "dnsmos_decision_v2_2",
            ]
            consumed = True
        else:
            conf = "medium"
            source = "auto_family_energy"
            branch = "human_noise_v2_fallback"
            v2_fb = True
            evidence = ["family_state_v2", "audio_energy_v1"]
            consumed = False
        return {
            "noise_kind": NOISE_KIND_HUMAN_NOISE,
            "classification_source": source,
            "classification_confidence": conf,
            "reason": "environment_noise_human_noise",
            "rule_branch": branch,
            "v2_fallback": v2_fb,
            "fallback_policy": RULE_VERSION_FIVE_CLASS_V2 if v2_fb else None,
            "unique_family": only.family,
            "unique_text": text,
            "family_counts": counts,
            "evidence_sources": evidence,
            "dnsmos_consumed": consumed,
        }

    return None

def classify_five_class_v2_2(
    sample: Sample,
    config: SelectionV3Config,
    *,
    voicemail_pattern: re.Pattern[str] | None = None,
) -> ClassificationResultV3:
    # Explicit disable → audited pure v2 fallback (no silent v2.2 claim).
    if not config.dnsmos_decision_enabled:
        result = classify_five_class_v2(
            sample, config, voicemail_pattern=voicemail_pattern
        )
        result.rule_version = RULE_VERSION_FIVE_CLASS_V2_2
        result.decision_trace = {
            "dnsmos_decision": "disabled",
            "fallback_policy": RULE_VERSION_FIVE_CLASS_V2,
            "rule_branch": "disabled_fallback_v2",
        }
        result.evidence_sources = ["family_state_v2", "audio_energy_v1"]
        result.dnsmos_noise_state = "unavailable"
        result.dnsmos_speech_state = "unavailable"
        result.dnsmos_decision_policy_version = ""
        result.v2_fallback = True
        return result

    routes = collect_route_views(sample, config)
    duration = float(sample.duration) if sample.duration is not None else None
    energy = energy_evidence_from_quality(
        sample.quality if isinstance(sample.quality, dict) else {},
        config=config,
        duration_sec=duration,
    )
    energy_dict = energy.as_dict()
    dnsmos_cfg = _dnsmos_config_from_selection(config)
    dnsmos = derive_dnsmos_decision(
        sample.quality if isinstance(sample.quality, dict) else {},
        dnsmos_cfg,
    )

    if is_physically_invalid(sample):
        result = ClassificationResultV3(
            type=TYPE_INVALID_AUDIO,
            decision=DECISION_EXCLUDE,
            reason="broken_or_invalid_audio",
            category=None,
            status=STATUS_EXCLUDED,
            outcome=OUTCOME_EXCLUDED,
            rule_version=config.rule_version or RULE_VERSION_FIVE_CLASS_V2_2,
            configured_family_count=len(config.model_families),
            annotation_state="excluded",
            acoustic_evidence=energy_dict,
            dnsmos_noise_state=dnsmos.noise_state,
            dnsmos_speech_state=dnsmos.speech_state,
            dnsmos_decision_policy_version=dnsmos.policy_version,
            dnsmos_status=dnsmos.status,
        )
        return apply_route_audit(result, config, routes)

    excluded_routes = [r for r in routes if r.route_disposition == ROUTE_EXCLUDED]
    if routes and all(r.route_disposition == ROUTE_EXCLUDED for r in routes):
        result = ClassificationResultV3(
            type="route_excluded",
            decision=DECISION_EXCLUDE,
            reason="all_routes_excluded",
            category=None,
            status=STATUS_EXCLUDED,
            outcome=OUTCOME_EXCLUDED,
            rule_version=config.rule_version or RULE_VERSION_FIVE_CLASS_V2_2,
            configured_family_count=len(config.model_families),
            reason_codes=["all_routes_excluded"],
            annotation_state="excluded",
            acoustic_evidence=energy_dict,
            dnsmos_noise_state=dnsmos.noise_state,
            dnsmos_speech_state=dnsmos.speech_state,
            dnsmos_decision_policy_version=dnsmos.policy_version,
            dnsmos_status=dnsmos.status,
        )
        return apply_route_audit(result, config, routes)

    # 1. Voicemail — DNSMOS cannot change class.
    vm = _voicemail_hit(routes, voicemail_pattern)
    if vm:
        result = _base_result_v22(
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
            dnsmos_noise_state=dnsmos.noise_state,
            dnsmos_speech_state=dnsmos.speech_state,
            dnsmos_decision_policy_version=dnsmos.policy_version,
            dnsmos_status=dnsmos.status,
            evidence_sources=["family_state_v2"],
            decision_trace={
                "rule_branch": "voicemail",
                "dnsmos_consumed": False,
                "thresholds": dnsmos_cfg.as_dict(),
            },
        )
        return apply_route_audit(result, config, routes)

    bundles = build_family_states(routes, config)
    counts = _family_counts(bundles)
    family_status = dict(counts["family_state_by_name"])

    def _attach(result: ClassificationResultV3, *, noise_meta: dict[str, Any] | None = None) -> ClassificationResultV3:
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
        result.dnsmos_noise_state = dnsmos.noise_state
        result.dnsmos_speech_state = dnsmos.speech_state
        result.dnsmos_decision_policy_version = dnsmos.policy_version
        result.dnsmos_status = dnsmos.status
        meta = noise_meta or {}
        evidence = list(meta.get("evidence_sources") or ["family_state_v2", "audio_energy_v1"])
        if meta.get("dnsmos_consumed") and "dnsmos_decision_v2_2" not in evidence:
            evidence.append("dnsmos_decision_v2_2")
        result.evidence_sources = evidence
        result.v2_fallback = bool(meta.get("v2_fallback"))
        result.borderline_resolved_by_dnsmos = bool(meta.get("borderline_resolved"))
        result.decision_trace = {
            "rule_branch": meta.get("rule_branch") or result.reason,
            "dnsmos_consumed": bool(meta.get("dnsmos_consumed")),
            "v2_fallback": bool(meta.get("v2_fallback")),
            "fallback_policy": meta.get("fallback_policy"),
            "borderline_resolved": bool(meta.get("borderline_resolved")),
            "dnsmos_noise_state": dnsmos.noise_state,
            "dnsmos_speech_state": dnsmos.speech_state,
            "thresholds": dnsmos_cfg.as_dict(),
            "dnsmos_model_digest": dnsmos.model_digest,
            "dnsmos_preprocess_version": dnsmos.preprocess_version,
            "energy_policy_version": energy.energy_policy_version,
        }
        result.acoustic_evidence = {
            **(result.acoustic_evidence or {}),
            **energy_dict,
            **dnsmos.as_dict(),
        }
        return result

    # 2. semantic_risk — DNSMOS cannot change class.
    risk_hit = _semantic_risk_v2(bundles, config)
    if risk_hit is not None:
        result = _attach(
            _base_result_v22(
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
            ),
            noise_meta={
                "rule_branch": "semantic_risk",
                "dnsmos_consumed": False,
                "evidence_sources": ["family_state_v2"],
            },
        )
        return apply_route_audit(result, config, routes)

    # 3. environment_noise (DNSMOS joint)
    noise = _environment_noise_decision_v22(bundles, energy, config, dnsmos)
    if noise is not None and noise.get("hardcase"):
        result = _attach(
            _base_result_v22(
                type=TYPE_HARDCASE,
                decision=DECISION_MANUAL_REVIEW,
                reason=str(noise["hardcase_reason"]),
                category=CATEGORY_HARDCASE,
                status=STATUS_MANUAL_REVIEW,
                reason_codes=[str(noise["hardcase_reason"])],
                review_priority=PRIORITY_P0,
                review_queue="manual_review",
                configured_family_count=len(config.model_families),
                classification_source=noise.get(
                    "classification_source", "auto_family_energy_dnsmos"
                ),
                classification_confidence=noise.get("classification_confidence", "low"),
                needs_review=True,
                review_reason=str(noise["hardcase_reason"]),
                hardcase_reason=str(noise["hardcase_reason"]),
                label_source=LABEL_SOURCE_NONE,
                label_tier=LABEL_TIER_NONE,
            ),
            noise_meta=noise,
        )
        return apply_route_audit(result, config, routes)

    if noise is not None:
        q = sample.quality if isinstance(sample.quality, dict) else {}
        lab = sample.labels if isinstance(sample.labels, dict) else {}
        kind = noise["noise_kind"]
        if kind == NOISE_KIND_HUMAN_NOISE and (
            q.get("human_crosstalk_confirmed") or lab.get("human_crosstalk_confirmed")
        ):
            noise = dict(noise)
            noise["noise_kind"] = NOISE_KIND_CROSSTALK
            kind = NOISE_KIND_CROSSTALK
        result = _attach(
            _base_result_v22(
                type="environment_noise",
                decision=DECISION_EXCLUDE,
                reason=noise["reason"],
                category=CATEGORY_ENVIRONMENT_NOISE,
                status=STATUS_EXCLUDED,
                reason_codes=[noise["reason"]],
                noise_kind=kind,
                review_priority=PRIORITY_P2,
                review_queue=None,
                configured_family_count=len(config.model_families),
                classification_source=noise["classification_source"],
                classification_confidence=noise["classification_confidence"],
                needs_review=False,
            ),
            noise_meta=noise,
        )
        return apply_route_audit(result, config, routes)

    # 4. gold_candidate — DNSMOS may only attach quality tag.
    ok, reps, gold_reason = _gold_eligible(bundles, config)
    if ok and reps:
        selection = select_weighted_family_text(
            reps,
            sample_id=sample.id,
            rule_version=config.rule_version or RULE_VERSION_FIVE_CLASS_V2_2,
            seed=str(
                getattr(config, "selection_seed", "") or RULE_VERSION_FIVE_CLASS_V2_2
            ),
            family_weights=getattr(config, "family_selection_weights", None) or {},
            family_order=config.ordered_families(),
        )
        assert selection is not None
        quality_tag = None
        background_quality_risk = False
        gold_meta: dict[str, Any] = {
            "rule_branch": "gold_candidate",
            "dnsmos_consumed": False,
            "evidence_sources": ["family_state_v2", "audio_energy_v1"],
        }
        if (
            config.dnsmos_decision_attach_gold_quality_tag
            and dnsmos.noise_state == DNSMOS_NOISE_NOISY
        ):
            quality_tag = "background_noisy"
            background_quality_risk = True
            gold_meta = {
                "rule_branch": "gold_candidate_background_noisy",
                "dnsmos_consumed": True,
                "evidence_sources": [
                    "family_state_v2",
                    "audio_energy_v1",
                    "dnsmos_decision_v2_2",
                ],
            }
        result = _attach(
            _base_result_v22(
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
                quality_tag=quality_tag,
                background_quality_risk=background_quality_risk,
            ),
            noise_meta=gold_meta,
        )
        return apply_route_audit(result, config, routes)

    # 5. hardcase
    hardcase_reason = "hardcase_fallback"
    needs_review = True
    review_reason = ""
    spans: list[dict[str, Any]] = []
    hard_meta: dict[str, Any] = {
        "rule_branch": "hardcase",
        "dnsmos_consumed": dnsmos.noise_state != DNSMOS_NOISE_UNAVAILABLE,
        "evidence_sources": ["family_state_v2", "audio_energy_v1"],
    }
    if dnsmos.noise_state != DNSMOS_NOISE_UNAVAILABLE:
        hard_meta["evidence_sources"] = [
            "family_state_v2",
            "audio_energy_v1",
            "dnsmos_decision_v2_2",
        ]

    if energy.energy_state == ENERGY_STATE_BORDERLINE:
        hardcase_reason = "energy_borderline"
        review_reason = "energy_state=borderline"
        hard_meta["rule_branch"] = "hardcase_energy_borderline"
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

    result = _attach(
        _base_result_v22(
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
        ),
        noise_meta=hard_meta,
    )
    return apply_route_audit(result, config, routes)
