"""034 five-class classifier: selection_five_class_v2_3_min_acoustic.

Silero VAD + EfficientAT acoustic evidence replace DNSMOS as primary
speech/noise evidence. Order: voicemail → matrix(text × acoustic) with
strong veto / semantic_risk / gold / hardcase.
"""

from __future__ import annotations

import re
from typing import Any

from audio_engine.core.quality.acoustic_evidence import (
    AcousticEvidenceConfig,
    aggregate_acoustic_state,
    load_acoustic_evidence_config,
    optional_dnsmos_quality_tags,
    resolve_audio_event_state,
)
from audio_engine.core.sample import Sample
from audio_engine.core.selection_v3.audio_energy import (
    AudioEnergyEvidence,
    energy_evidence_from_quality,
)
from audio_engine.core.selection_v3.classify_text import apply_route_audit
from audio_engine.core.selection_v3.config import SelectionV3Config
from audio_engine.core.selection_v3.family_evidence import collect_route_views
from audio_engine.core.selection_v3.five_class_v2 import (
    FamilyStateBundle,
    _family_counts,
    _hardcase_substantive,
    _harmless_equivalent,
    _is_critical_short_response,
    _semantic_risk_v2,
    _voicemail_hit,
    build_family_states,
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
    ENERGY_STATE_TOO_SHORT,
    FAMILY_STATE_STABLE_EMPTY,
    FAMILY_STATE_STABLE_TEXT,
    FAMILY_STATE_UNAVAILABLE,
    FAMILY_STATE_UNSTABLE,
    LABEL_SOURCE_MODEL,
    LABEL_SOURCE_NONE,
    LABEL_TIER_NONE,
    LABEL_TIER_PSEUDO_HIGH,
    NOISE_KIND_BACKGROUND,
    NOISE_KIND_HUMAN_NOISE,
    NOISE_KIND_SILENCE,
    OUTCOME_CLASSIFIED,
    OUTCOME_EXCLUDED,
    PRIORITY_P0,
    PRIORITY_P2,
    ROUTE_ELIGIBLE,
    ROUTE_EXCLUDED,
    RULE_VERSION_FIVE_CLASS_V2_3,
    STATUS_CANDIDATE,
    STATUS_EXCLUDED,
    STATUS_MANUAL_REVIEW,
    TYPE_HARDCASE,
    TYPE_INVALID_AUDIO,
    TYPE_PSEUDO_HIGH,
    TYPE_SEMANTIC_RISK,
    TYPE_VOICEMAIL_CANDIDATE,
)

TEXT_ALL_EMPTY = "all_empty"
TEXT_SINGLE = "single_text"
TEXT_CONSENSUS = "consensus_text"
TEXT_STRICT = "strict_conflict"
TEXT_DIVERGENT = "divergent_text"
TEXT_INSUFFICIENT = "insufficient"

HALLUCINATION_NONE = "none"
HALLUCINATION_SINGLE = "single_family"
HALLUCINATION_MULTI_CONSENSUS = "multi_family_consensus"
HALLUCINATION_MULTI_DIVERGENT = "multi_family_divergent"

ACOUSTIC_SILENCE = "silence"
ACOUSTIC_NON_SPEECH = "non_speech_noise"
ACOUSTIC_SPEECH = "speech_supported"
ACOUSTIC_HUMAN = "human_noise_suspected"
ACOUSTIC_UNCERTAIN = "uncertain"
ACOUSTIC_FAILED = "failed"


def derive_text_evidence_state(
    bundles: list[FamilyStateBundle],
    config: SelectionV3Config,
) -> str:
    """Derive text_evidence_state from family four-states + strict risk."""
    stable_text = [b for b in bundles if b.state == FAMILY_STATE_STABLE_TEXT]
    stable_empty = [b for b in bundles if b.state == FAMILY_STATE_STABLE_EMPTY]
    unstable = [b for b in bundles if b.state == FAMILY_STATE_UNSTABLE]
    unavailable = [b for b in bundles if b.state == FAMILY_STATE_UNAVAILABLE]

    if unstable or unavailable:
        if not stable_text:
            return TEXT_INSUFFICIENT
        # Unstable/unavailable present: only allow conflict/divergent/consensus
        # when ≥2 stable_text; otherwise insufficient.
        if len(stable_text) < 2:
            return TEXT_INSUFFICIENT

    if not stable_text:
        if (
            stable_empty
            and not unstable
            and not unavailable
            and len(stable_empty) == len(bundles)
        ):
            return TEXT_ALL_EMPTY
        return TEXT_INSUFFICIENT

    if len(stable_text) == 1:
        if unstable or unavailable:
            return TEXT_INSUFFICIENT
        others = [b for b in bundles if b.family != stable_text[0].family]
        if others and all(b.state == FAMILY_STATE_STABLE_EMPTY for b in others):
            return TEXT_SINGLE
        if not others:
            return TEXT_SINGLE
        return TEXT_INSUFFICIENT

    # ≥2 stable_text
    risk = _semantic_risk_v2(bundles, config)
    if risk is not None:
        return TEXT_STRICT

    texts = [b.representative_text or "" for b in stable_text]
    all_equivalent = all(
        _harmless_equivalent(texts[i], texts[j])
        for i in range(len(texts))
        for j in range(i + 1, len(texts))
    )
    if all_equivalent:
        return TEXT_CONSENSUS
    return TEXT_DIVERGENT


def _hallucination_for(
    text_state: str,
    *,
    acoustic: str,
) -> str:
    if acoustic not in {ACOUSTIC_SILENCE, ACOUSTIC_NON_SPEECH}:
        return HALLUCINATION_NONE
    if text_state == TEXT_CONSENSUS:
        return HALLUCINATION_MULTI_CONSENSUS
    if text_state in {TEXT_STRICT, TEXT_DIVERGENT}:
        return HALLUCINATION_MULTI_DIVERGENT
    if text_state == TEXT_SINGLE:
        return HALLUCINATION_SINGLE
    return HALLUCINATION_NONE


def _quality_tags_for_sample(
    sample: Sample,
    config: SelectionV3Config,
    energy: AudioEnergyEvidence,
    acoustic_cfg: AcousticEvidenceConfig,
) -> list[str]:
    tags: list[str] = []
    if energy.energy_state == ENERGY_STATE_TOO_SHORT:
        tags.append("short_audio")
    q = sample.quality if isinstance(sample.quality, dict) else {}
    tags.extend(optional_dnsmos_quality_tags(q, cfg=acoustic_cfg))
    # Deduplicate preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for tag in tags:
        if tag and tag not in seen:
            seen.add(tag)
            out.append(tag)
    return out


def _base_result_v23(**kwargs: Any) -> ClassificationResultV3:
    kwargs.setdefault("rule_version", RULE_VERSION_FIVE_CLASS_V2_3)
    kwargs.setdefault("outcome", OUTCOME_CLASSIFIED)
    kwargs.setdefault("annotation_state", "classified")
    return ClassificationResultV3(**kwargs)


def classify_five_class_v2_3_min_acoustic(
    sample: Sample,
    config: SelectionV3Config,
    *,
    voicemail_pattern: re.Pattern[str] | None = None,
) -> ClassificationResultV3:
    routes = collect_route_views(sample, config)
    duration = float(sample.duration) if sample.duration is not None else None
    energy = energy_evidence_from_quality(
        sample.quality if isinstance(sample.quality, dict) else {},
        config=config,
        duration_sec=duration,
    )
    energy_dict = energy.as_dict()
    quality = dict(sample.quality) if isinstance(sample.quality, dict) else {}

    acoustic_cfg_path = str(getattr(config, "acoustic_evidence_config_path", "") or "")
    acoustic_cfg = (
        load_acoustic_evidence_config(acoustic_cfg_path)
        if acoustic_cfg_path
        else AcousticEvidenceConfig()
    )
    audio_event_state = resolve_audio_event_state(quality, cfg=acoustic_cfg)
    acoustic_state = str(quality.get("acoustic_state") or "").strip() or aggregate_acoustic_state(
        {**quality, "audio_event_state": audio_event_state, "energy_state": energy.energy_state},
        cfg=acoustic_cfg,
    )
    quality_tags = _quality_tags_for_sample(sample, config, energy, acoustic_cfg)

    if is_physically_invalid(sample):
        result = ClassificationResultV3(
            type=TYPE_INVALID_AUDIO,
            decision=DECISION_EXCLUDE,
            reason="broken_or_invalid_audio",
            category=None,
            status=STATUS_EXCLUDED,
            outcome=OUTCOME_EXCLUDED,
            rule_version=config.rule_version or RULE_VERSION_FIVE_CLASS_V2_3,
            configured_family_count=len(config.model_families),
            annotation_state="excluded",
            acoustic_evidence=energy_dict,
            decision_evidence={
                "decision": "excluded",
                "reason": "broken_or_invalid_audio",
                "acoustic_state": acoustic_state,
            },
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
            rule_version=config.rule_version or RULE_VERSION_FIVE_CLASS_V2_3,
            configured_family_count=len(config.model_families),
            reason_codes=["all_routes_excluded"],
            annotation_state="excluded",
            acoustic_evidence=energy_dict,
            decision_evidence={
                "decision": "excluded",
                "reason": "all_routes_excluded",
                "acoustic_state": acoustic_state,
            },
        )
        return apply_route_audit(result, config, routes)

    bundles = build_family_states(routes, config)
    counts = _family_counts(bundles)
    family_status = dict(counts["family_state_by_name"])
    text_state = derive_text_evidence_state(bundles, config)
    has_unstable = any(b.state == FAMILY_STATE_UNSTABLE for b in bundles)

    def _decision_evidence(
        *,
        decision: str,
        hallucination: str = HALLUCINATION_NONE,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "text_evidence_state": text_state,
            "family_states": family_status,
            "vad_speech_state": quality.get("vad_speech_state"),
            "speech_ratio": quality.get("speech_ratio"),
            "audio_event_state": audio_event_state,
            "environment_noise_prob": quality.get("environment_noise_prob"),
            "acoustic_state": acoustic_state,
            "energy_state": energy.energy_state,
            "decision": decision,
            "hallucination_state": hallucination,
            "quality_tags": list(quality_tags),
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
        result.text_evidence_state = text_state
        result.vad_speech_state = quality.get("vad_speech_state")
        result.audio_event_state = audio_event_state
        result.acoustic_state = acoustic_state
        result.quality_tags = list(quality_tags)
        if quality_tags and not result.quality_tag:
            result.quality_tag = quality_tags[0]
        result.acoustic_evidence = {
            **(result.acoustic_evidence or {}),
            **energy_dict,
            "vad_speech_state": quality.get("vad_speech_state"),
            "speech_ratio": quality.get("speech_ratio"),
            "audio_event_state": audio_event_state,
            "acoustic_state": acoustic_state,
            "efficientat_status": quality.get("efficientat_status"),
        }
        return result

    # 1. Voicemail short-circuit (acoustic contradiction → review only).
    vm = _voicemail_hit(routes, voicemail_pattern)
    if vm:
        needs_review = acoustic_state in {ACOUSTIC_SILENCE, ACOUSTIC_NON_SPEECH}
        result = _attach(
            _base_result_v23(
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
                classification_source="auto_rule",
                classification_confidence="high",
                needs_review=needs_review,
                review_reason=(
                    "voicemail_acoustic_contradiction" if needs_review else None
                ),
                evidence_sources=["family_state_v2", "silero_vad_v1"],
                hallucination_state=HALLUCINATION_NONE,
                decision_evidence=_decision_evidence(
                    decision=CATEGORY_VOICEMAIL,
                    extra={
                        "voicemail_acoustic_state": acoustic_state,
                        "rule_branch": "voicemail",
                    },
                ),
                decision_trace={
                    "rule_branch": "voicemail",
                    "voicemail_acoustic_state": acoustic_state,
                },
            )
        )
        return apply_route_audit(result, config, routes)

    # Acoustic operator hard failures → hardcase (do not forge silence/noise).
    if acoustic_state == ACOUSTIC_FAILED or str(quality.get("vad_status") or "") == "failed":
        hardcase_reason = "vad_failed"
        if str(quality.get("efficientat_status") or "") == "failed":
            hardcase_reason = "efficientat_failed"
        if energy.energy_state == "failed":
            hardcase_reason = "energy_failed"
        result = _attach(
            _base_result_v23(
                type=TYPE_HARDCASE,
                decision=DECISION_MANUAL_REVIEW,
                reason=hardcase_reason,
                category=CATEGORY_HARDCASE,
                status=STATUS_MANUAL_REVIEW,
                reason_codes=[hardcase_reason],
                review_priority=PRIORITY_P0,
                review_queue="manual_review",
                configured_family_count=len(config.model_families),
                classification_source="auto_fallback",
                classification_confidence="low",
                needs_review=True,
                review_reason=hardcase_reason,
                hardcase_reason=hardcase_reason,
                label_source=LABEL_SOURCE_NONE,
                label_tier=LABEL_TIER_NONE,
                hallucination_state=HALLUCINATION_NONE,
                evidence_sources=["family_state_v2", "silero_vad_v1", "acoustic_evidence_v1"],
                decision_evidence=_decision_evidence(
                    decision=CATEGORY_HARDCASE,
                    extra={"rule_branch": hardcase_reason},
                ),
            )
        )
        return apply_route_audit(result, config, routes)

    # family_unstable: veto gold; strong non-speech → env+review else hardcase.
    if has_unstable and text_state not in {TEXT_STRICT, TEXT_DIVERGENT, TEXT_CONSENSUS}:
        if acoustic_state in {ACOUSTIC_SILENCE, ACOUSTIC_NON_SPEECH}:
            hall = _hallucination_for(TEXT_INSUFFICIENT, acoustic=acoustic_state)
            kind = (
                NOISE_KIND_SILENCE
                if acoustic_state == ACOUSTIC_SILENCE
                else NOISE_KIND_BACKGROUND
            )
            result = _attach(
                _base_result_v23(
                    type="environment_noise",
                    decision=DECISION_EXCLUDE,
                    reason="environment_noise_family_unstable_non_speech",
                    category=CATEGORY_ENVIRONMENT_NOISE,
                    status=STATUS_EXCLUDED,
                    reason_codes=["family_unstable", "strong_acoustic_veto"],
                    noise_kind=kind,
                    review_priority=PRIORITY_P2,
                    configured_family_count=len(config.model_families),
                    classification_source="auto_family_acoustic",
                    classification_confidence="medium",
                    needs_review=True,
                    review_reason="family_unstable",
                    hallucination_state=hall,
                    evidence_sources=[
                        "family_state_v2",
                        "silero_vad_v1",
                        "acoustic_evidence_v1",
                    ],
                    decision_evidence=_decision_evidence(
                        decision=CATEGORY_ENVIRONMENT_NOISE,
                        hallucination=hall,
                        extra={"rule_branch": "family_unstable_strong_veto"},
                    ),
                )
            )
            return apply_route_audit(result, config, routes)
        result = _attach(
            _base_result_v23(
                type=TYPE_HARDCASE,
                decision=DECISION_MANUAL_REVIEW,
                reason="family_unstable",
                category=CATEGORY_HARDCASE,
                status=STATUS_MANUAL_REVIEW,
                reason_codes=["family_unstable"],
                review_priority=PRIORITY_P0,
                review_queue="manual_review",
                configured_family_count=len(config.model_families),
                classification_source="auto_fallback",
                classification_confidence="low",
                needs_review=True,
                review_reason="intra_family_dual_run_unstable",
                hardcase_reason="family_unstable",
                label_source=LABEL_SOURCE_NONE,
                label_tier=LABEL_TIER_NONE,
                hallucination_state=HALLUCINATION_NONE,
                evidence_sources=["family_state_v2"],
                decision_evidence=_decision_evidence(
                    decision=CATEGORY_HARDCASE,
                    extra={"rule_branch": "family_unstable"},
                ),
            )
        )
        return apply_route_audit(result, config, routes)

    # --- Decision matrix: text_evidence_state × acoustic_state ---
    hall = _hallucination_for(text_state, acoustic=acoustic_state)
    strong_veto = acoustic_state in {ACOUSTIC_SILENCE, ACOUSTIC_NON_SPEECH}

    def _env(
        *,
        kind: str,
        reason: str,
        confidence: str = "high",
        needs_review: bool = False,
        review_reason: str | None = None,
        hallucination: str | None = None,
        branch: str,
    ) -> ClassificationResultV3:
        h = hallucination if hallucination is not None else hall
        return _attach(
            _base_result_v23(
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
                classification_confidence=confidence,
                needs_review=needs_review,
                review_reason=review_reason,
                hallucination_state=h,
                evidence_sources=[
                    "family_state_v2",
                    "silero_vad_v1",
                    "acoustic_evidence_v1",
                ],
                decision_evidence=_decision_evidence(
                    decision=CATEGORY_ENVIRONMENT_NOISE,
                    hallucination=h,
                    extra={"rule_branch": branch, "noise_kind": kind},
                ),
                decision_trace={"rule_branch": branch, "noise_kind": kind},
            )
        )

    def _hard(reason: str, *, branch: str | None = None) -> ClassificationResultV3:
        return _attach(
            _base_result_v23(
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
                hallucination_state=HALLUCINATION_NONE,
                evidence_sources=[
                    "family_state_v2",
                    "silero_vad_v1",
                    "acoustic_evidence_v1",
                ],
                decision_evidence=_decision_evidence(
                    decision=CATEGORY_HARDCASE,
                    extra={"rule_branch": branch or reason},
                ),
                decision_trace={"rule_branch": branch or reason},
            )
        )

    def _semantic(risk_hit: Any) -> ClassificationResultV3:
        return _attach(
            _base_result_v23(
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
                hallucination_state=HALLUCINATION_NONE,
                evidence_sources=["family_state_v2"],
                decision_evidence=_decision_evidence(
                    decision=CATEGORY_SEMANTIC_RISK,
                    extra={"rule_branch": "semantic_risk", "subtype": risk_hit.subtype},
                ),
            )
        )

    def _gold(*, confidence: str, needs_review: bool, tag: str | None, branch: str) -> ClassificationResultV3 | None:
        if has_unstable:
            return None
        from audio_engine.core.selection_v3.five_class_v2 import _gold_eligible

        ok, gold_reps, _gold_reason = _gold_eligible(bundles, config)
        if not ok or not gold_reps:
            return None
        selection = select_weighted_family_text(
            gold_reps,
            sample_id=sample.id,
            rule_version=config.rule_version or RULE_VERSION_FIVE_CLASS_V2_3,
            seed=str(
                getattr(config, "selection_seed", "") or RULE_VERSION_FIVE_CLASS_V2_3
            ),
            family_weights=getattr(config, "family_selection_weights", None) or {},
            family_order=config.ordered_families(),
        )
        if selection is None:
            return None
        tags = list(quality_tags)
        if tag and tag not in tags:
            tags.append(tag)
        return _attach(
            _base_result_v23(
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
                classification_confidence=confidence,
                needs_review=needs_review,
                review_reason="gold_acoustic_review" if needs_review else None,
                quality_tag=tag or (tags[0] if tags else None),
                quality_tags=tags,
                hallucination_state=HALLUCINATION_NONE,
                evidence_sources=[
                    "family_state_v2",
                    "silero_vad_v1",
                    "acoustic_evidence_v1",
                ],
                decision_evidence=_decision_evidence(
                    decision=CATEGORY_GOLD_CANDIDATE,
                    extra={"rule_branch": branch, "confidence": confidence},
                ),
                decision_trace={"rule_branch": branch},
            )
        )

    # Matrix application.
    if text_state == TEXT_ALL_EMPTY:
        if acoustic_state == ACOUSTIC_SILENCE:
            return apply_route_audit(
                _env(
                    kind=NOISE_KIND_SILENCE,
                    reason="environment_noise_silence",
                    branch="all_empty_silence",
                ),
                config,
                routes,
            )
        if acoustic_state == ACOUSTIC_NON_SPEECH:
            return apply_route_audit(
                _env(
                    kind=NOISE_KIND_BACKGROUND,
                    reason="environment_noise_background",
                    branch="all_empty_non_speech",
                ),
                config,
                routes,
            )
        if acoustic_state == ACOUSTIC_SPEECH:
            return apply_route_audit(
                _hard("empty_asr_but_speech_present"), config, routes
            )
        if acoustic_state == ACOUSTIC_HUMAN:
            return apply_route_audit(
                _env(
                    kind=NOISE_KIND_HUMAN_NOISE,
                    reason="environment_noise_human_noise",
                    confidence="medium",
                    needs_review=True,
                    review_reason="human_noise_unconfirmed",
                    branch="all_empty_human_noise",
                ),
                config,
                routes,
            )
        return apply_route_audit(
            _hard("empty_asr_acoustic_uncertain"), config, routes
        )

    if text_state == TEXT_SINGLE:
        only = next(b for b in bundles if b.state == FAMILY_STATE_STABLE_TEXT)
        critical = _is_critical_short_response(only.representative_text or "", config)
        if strong_veto:
            return apply_route_audit(
                _env(
                    kind=(
                        NOISE_KIND_SILENCE
                        if acoustic_state == ACOUSTIC_SILENCE
                        else NOISE_KIND_BACKGROUND
                    ),
                    reason="environment_noise_hallucination_single",
                    hallucination=HALLUCINATION_SINGLE,
                    branch="single_text_strong_veto",
                ),
                config,
                routes,
            )
        if acoustic_state == ACOUSTIC_SPEECH:
            return apply_route_audit(
                _hard("single_text_family_supported_speech"), config, routes
            )
        if acoustic_state == ACOUSTIC_HUMAN:
            if critical:
                return apply_route_audit(
                    _hard("critical_short_response_unsupported"), config, routes
                )
            return apply_route_audit(
                _env(
                    kind=NOISE_KIND_HUMAN_NOISE,
                    reason="environment_noise_human_noise",
                    confidence="medium",
                    needs_review=True,
                    review_reason="human_noise_unconfirmed",
                    hallucination=HALLUCINATION_NONE,
                    branch="single_text_human_noise",
                ),
                config,
                routes,
            )
        return apply_route_audit(
            _hard("single_text_acoustic_uncertain"), config, routes
        )

    if text_state == TEXT_CONSENSUS:
        if strong_veto:
            return apply_route_audit(
                _env(
                    kind=(
                        NOISE_KIND_SILENCE
                        if acoustic_state == ACOUSTIC_SILENCE
                        else NOISE_KIND_BACKGROUND
                    ),
                    reason="environment_noise_multi_family_hallucination",
                    hallucination=HALLUCINATION_MULTI_CONSENSUS,
                    branch="consensus_strong_veto",
                ),
                config,
                routes,
            )
        if acoustic_state == ACOUSTIC_SPEECH:
            gold = _gold(
                confidence="high",
                needs_review=False,
                tag=None,
                branch="consensus_speech_supported",
            )
            if gold is not None:
                return apply_route_audit(gold, config, routes)
        if acoustic_state == ACOUSTIC_HUMAN:
            gold = _gold(
                confidence="medium",
                needs_review=True,
                tag="human_background_suspected",
                branch="consensus_human_noise_suspected",
            )
            if gold is not None:
                return apply_route_audit(gold, config, routes)
        if acoustic_state == ACOUSTIC_UNCERTAIN:
            gold = _gold(
                confidence="medium",
                needs_review=True,
                tag=None,
                branch="consensus_acoustic_uncertain",
            )
            if gold is not None:
                return apply_route_audit(gold, config, routes)
        return apply_route_audit(_hard("hardcase_fallback"), config, routes)

    if text_state == TEXT_STRICT:
        if strong_veto:
            return apply_route_audit(
                _env(
                    kind=(
                        NOISE_KIND_SILENCE
                        if acoustic_state == ACOUSTIC_SILENCE
                        else NOISE_KIND_BACKGROUND
                    ),
                    reason="environment_noise_strict_conflict_hallucination",
                    hallucination=HALLUCINATION_MULTI_DIVERGENT,
                    branch="strict_conflict_strong_veto",
                ),
                config,
                routes,
            )
        risk = _semantic_risk_v2(bundles, config)
        if risk is not None:
            return apply_route_audit(_semantic(risk), config, routes)
        return apply_route_audit(
            _hard("substantive_divergence_non_strict_risk"), config, routes
        )

    if text_state == TEXT_DIVERGENT:
        if strong_veto:
            return apply_route_audit(
                _env(
                    kind=(
                        NOISE_KIND_SILENCE
                        if acoustic_state == ACOUSTIC_SILENCE
                        else NOISE_KIND_BACKGROUND
                    ),
                    reason="environment_noise_divergent_hallucination",
                    hallucination=HALLUCINATION_MULTI_DIVERGENT,
                    branch="divergent_strong_veto",
                ),
                config,
                routes,
            )
        is_hard, _spans = _hardcase_substantive(bundles)
        if is_hard:
            return apply_route_audit(
                _hard("substantive_divergence_non_strict_risk"), config, routes
            )
        return apply_route_audit(_hard("hardcase_fallback"), config, routes)

    # insufficient / fallback
    if strong_veto and text_state == TEXT_INSUFFICIENT:
        # Prefer hardcase over forged noise when text structure is insufficient.
        return apply_route_audit(
            _hard("insufficient_family_evidence"), config, routes
        )
    if excluded_routes and not any(
        r.route_disposition == ROUTE_ELIGIBLE for r in routes
    ):
        return apply_route_audit(
            _hard("insufficient_evidence_after_exclusion"), config, routes
        )
    return apply_route_audit(_hard("hardcase_fallback"), config, routes)
