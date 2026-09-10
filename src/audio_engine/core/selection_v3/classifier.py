"""Main classify_sample entry for selection_v3.0 / consensus_v3."""

from __future__ import annotations

import re
from typing import Any

from audio_engine.core.sample import Sample
from audio_engine.core.selection_v3.config import SelectionV3Config
from audio_engine.core.selection_v3.consensus import (
    analyze_consensus,
    eight_route_full_agreement,
)
from audio_engine.core.selection_v3.family_evidence import (
    analyze_families,
    is_short_utterance,
    collect_route_views,
)
from audio_engine.core.selection_v3.input_contract import (
    is_physically_invalid,
)
from audio_engine.core.selection_v3.result import ClassificationResultV3
from audio_engine.core.selection_v3.review_router import route_review
from audio_engine.core.selection_v3.semantic_risk import (
    analyze_risks,
    compile_lexicon,
    polarity_of_text,
)
from audio_engine.core.selection_v3.text import text_similarity
from audio_engine.core.selection_v3.types import (
    DECISION_AUDIT_PENDING,
    DECISION_EXCLUDE,
    DECISION_MANUAL_REVIEW,
    DECISION_RETRY,
    FAMILY_INCOMPLETE,
    FAMILY_STABLE_TEXT,
    FAMILY_UNSTABLE_PRESENCE,
    FAMILY_UNSTABLE_SEMANTIC,
    FAMILY_UNSTABLE_TEXT,
    LABEL_SOURCE_MODEL,
    LABEL_TIER_PSEUDO_HIGH,
    LABEL_TIER_PSEUDO_MEDIUM,
    NOISE_BAND_CLEAN,
    NOISE_BAND_MODERATE,
    NOISE_BAND_NOISY,
    NOISE_BAND_UNKNOWN,
    POLARITY_MIXED,
    POLARITY_POSITIVE,
    RISK_CONSENSUS_AMBIGUOUS,
    RUN_STATUS_SUCCESS_EMPTY,
    RUN_STATUS_SUCCESS_TEXT,
    SEMANTIC_RISK_TAGS,
    TYPE_ALL_EMPTY_UNVERIFIED,
    TYPE_AUDIO_QUALITY_RISK,
    TYPE_CRITICAL_CONTENT_RISK,
    TYPE_FAMILY_UNSTABLE,
    TYPE_HARDCASE,
    TYPE_INFERENCE_INCOMPLETE,
    TYPE_INVALID_AUDIO,
    TYPE_PSEUDO_HIGH,
    TYPE_PSEUDO_MEDIUM,
    TYPE_QWEN_CORRECTION_CANDIDATE,
    TYPE_SEMANTIC_RISK,
    TYPE_SPEECH_PRESENCE_DISAGREEMENT,
    TYPE_VOICEMAIL_CANDIDATE,
)


def _duration_sec(sample: Sample) -> float | None:
    if sample.duration is None:
        return None
    try:
        return float(sample.duration)
    except (TypeError, ValueError):
        return None


def _quality_fields(sample: Sample) -> dict[str, Any]:
    q = sample.quality if isinstance(sample.quality, dict) else {}
    band = q.get("noise_band")
    risk = q.get("noise_risk")
    status = q.get("dnsmos_status")
    # Also accept nested quality from flat merge
    return {
        "noise_band": str(band) if band is not None else None,
        "noise_risk": risk if risk is None or isinstance(risk, bool) else None,
        "dnsmos_status": str(status) if status is not None else None,
        "crosstalk_suspected": bool(
            q.get("crosstalk_suspected")
            or sample.labels.get("crosstalk_suspected")
            or sample.labels.get("overlap_risk")
        )
        if (
            q.get("crosstalk_suspected") is not None
            or sample.labels.get("crosstalk_suspected") is not None
            or sample.labels.get("overlap_risk") is True
        )
        else False,
        # Only trust explicit human/external crosstalk feature
        "crosstalk_trusted": bool(
            q.get("crosstalk_suspected_trusted")
            or sample.labels.get("crosstalk_suspected_trusted")
        ),
    }


def _voicemail_hit(
    text: str,
    pattern: re.Pattern[str] | None,
) -> bool:
    if pattern is None or not text:
        return False
    return bool(pattern.search(text))


def _qwen_value_fields(
    families: dict,
    config: SelectionV3Config,
    teacher_cluster,
    patterns,
) -> dict[str, Any]:
    target = config.target_family
    qwen = families.get(target)
    qwen_status = qwen.status if qwen else None
    teacher_status = "none"
    sim1 = None
    sim2 = None
    qwen_risk: list[str] = []
    correction = False

    teacher_states = [families[f] for f in config.teacher_families if f in families]
    teachers_stable = all(s.status == FAMILY_STABLE_TEXT for s in teacher_states)

    if teacher_cluster is not None and teachers_stable:
        teacher_status = "stable_consensus"
        teacher_cmp = teacher_cluster.members[0].comparison_text
        if qwen is not None:
            q_routes = [
                r
                for r in qwen.routes
                if r.status == RUN_STATUS_SUCCESS_TEXT and r.comparison_text
            ]
            if len(q_routes) >= 1:
                sim1 = text_similarity(q_routes[0].comparison_text, teacher_cmp)
            if len(q_routes) >= 2:
                sim2 = text_similarity(q_routes[1].comparison_text, teacher_cmp)
            qwen_unstable = qwen.status != FAMILY_STABLE_TEXT
            below = False
            for sim in (sim1, sim2):
                if sim is not None and sim < config.teacher_consensus_threshold:
                    below = True
            if qwen_unstable or below:
                correction = True
            # Qwen-internal polarity vs teachers
            from audio_engine.core.selection_v3.semantic_risk import conflict_tags_for_texts

            texts = [teacher_cmp] + [r.comparison_text for r in q_routes]
            qwen_risk = sorted(conflict_tags_for_texts(texts, patterns))
    elif teachers_stable:
        teacher_status = "stable_no_unique_cluster"
    else:
        teacher_status = "unstable_or_incomplete"

    return {
        "qwen_status": qwen_status,
        "teacher_consensus_status": teacher_status,
        "qwen_vs_teacher_similarity_1": sim1,
        "qwen_vs_teacher_similarity_2": sim2,
        "qwen_risk_tags": qwen_risk,
        "qwen_correction_candidate": correction,
    }


def classify_sample(
    sample: Sample,
    config: SelectionV3Config,
    *,
    voicemail_pattern: re.Pattern[str] | None = None,
) -> ClassificationResultV3:
    patterns = compile_lexicon(config)
    quality = _quality_fields(sample)
    duration = _duration_sec(sample)
    routes = collect_route_views(sample, config)
    short = is_short_utterance(
        duration_sec=duration,
        routes=routes,
        max_audio_sec=config.short_audio_sec,
        max_text_chars=config.short_text_chars,
    )
    families = analyze_families(sample, config, patterns, duration_sec=duration)
    family_status = {name: state.status for name, state in families.items()}

    success_text_routes = [
        r for r in routes if r.status == RUN_STATUS_SUCCESS_TEXT and r.comparison_text
    ]
    success_empty_routes = [
        r
        for r in routes
        if r.status == RUN_STATUS_SUCCESS_EMPTY
        or (r.status == RUN_STATUS_SUCCESS_TEXT and not r.comparison_text)
    ]
    # Treat explicit success_empty only for presence
    success_empty_routes = [r for r in routes if r.status == RUN_STATUS_SUCCESS_EMPTY]
    incomplete = any(s.status == FAMILY_INCOMPLETE for s in families.values())
    family_unstable = any(
        s.status
        in {
            FAMILY_UNSTABLE_PRESENCE,
            FAMILY_UNSTABLE_SEMANTIC,
            FAMILY_UNSTABLE_TEXT,
        }
        for s in families.values()
    )

    noisy = quality["noise_band"] == NOISE_BAND_NOISY or quality["noise_risk"] is True
    quality_unknown = (
        quality["noise_band"] in {NOISE_BAND_UNKNOWN, None}
        or quality["dnsmos_status"] in {"failed", "unsupported", None}
        or quality["noise_risk"] is None
    )
    crosstalk = bool(quality["crosstalk_trusted"] and quality["crosstalk_suspected"])

    risks = analyze_risks(
        comparison_texts=[r.comparison_text for r in success_text_routes],
        success_empty_count=len(success_empty_routes),
        success_text_count=len(success_text_routes),
        short_utterance=short,
        family_unstable=family_unstable,
        noisy_audio=noisy,
        quality_unknown=quality_unknown and not noisy,
        crosstalk_suspected=crosstalk,
        patterns=patterns,
    )

    # Teacher consensus at 0.98 for Qwen correction detection
    teacher_consensus = analyze_consensus(
        {**{f: families[f] for f in config.teacher_families if f in families}},
        config,
        threshold=config.teacher_consensus_threshold,
        short=short,
    )
    # Only count teacher families that actually vote
    teacher_primary = None
    if (
        not teacher_consensus.consensus_ambiguous
        and teacher_consensus.primary is not None
        and teacher_consensus.primary.teacher_support_count
        >= len(config.teacher_families)
        and all(
            families[f].status == FAMILY_STABLE_TEXT for f in config.teacher_families
        )
    ):
        teacher_primary = teacher_consensus.primary

    qwen_fields = _qwen_value_fields(
        families, config, teacher_primary, patterns
    )

    def _finish(
        *,
        type_: str,
        decision: str,
        reason: str,
        candidate_text: str | None = None,
        label_source: str = "none",
        label_tier: str = "none",
        support_family_count: int = 0,
        support_ratio: float | None = None,
        teacher_support: int = 0,
        min_sim: float | None = None,
        ambiguous: bool = False,
        selected_run_id: str | None = None,
        support_run_ids: list[str] | None = None,
        extra_tags: list[str] | None = None,
        presence_has_affirmation: bool = False,
        review_reason: str | None = None,
    ) -> ClassificationResultV3:
        tags = list(dict.fromkeys(list(risks.risk_tags) + list(extra_tags or [])))
        priority, queue = route_review(
            type_=type_,
            risk_tags=tags,
            presence_has_affirmation=presence_has_affirmation,
        )
        if decision == DECISION_RETRY:
            priority, queue = None, "retry"
        if decision == DECISION_EXCLUDE:
            priority, queue = None, "exclude"
        return ClassificationResultV3(
            type=type_,
            decision=decision,
            reason=reason,
            review_priority=priority,
            review_queue=queue,
            candidate_text=candidate_text,
            label_source=label_source,
            label_tier=label_tier,
            is_human_verified=False,
            risk_tags=tags,
            polarity=risks.polarity,
            family_status=family_status,
            support_family_count=support_family_count,
            support_ratio_of_4=support_ratio,
            teacher_support_count=teacher_support,
            min_similarity=min_sim,
            consensus_ambiguous=ambiguous,
            selected_run_id=selected_run_id,
            support_run_ids=list(support_run_ids or []),
            noise_band=quality["noise_band"],
            noise_risk=quality["noise_risk"],
            dnsmos_status=quality["dnsmos_status"],
            short_utterance=short,
            rule_version=config.rule_version,
            review_reason=review_reason,
            **qwen_fields,
        )

    # 1. invalid_audio
    if is_physically_invalid(sample):
        return _finish(
            type_=TYPE_INVALID_AUDIO,
            decision=DECISION_EXCLUDE,
            reason="broken_or_invalid_audio",
            candidate_text="",
        )

    # 2. inference_incomplete
    if incomplete:
        return _finish(
            type_=TYPE_INFERENCE_INCOMPLETE,
            decision=DECISION_RETRY,
            reason="eight_route_incomplete",
            review_reason="retry_missing_or_failed_runs",
        )

    # 3. semantic_risk (P0)
    if risks.semantic_risk or (set(risks.risk_tags) & SEMANTIC_RISK_TAGS):
        return _finish(
            type_=TYPE_SEMANTIC_RISK,
            decision=DECISION_MANUAL_REVIEW,
            reason="semantic_risk_tags",
            review_reason="p0_semantic",
            # Preserve Qwen correction flag even under higher-priority bucket
        )

    # 4. critical_content_risk
    if risks.critical_content_risk or risks.polarity == POLARITY_MIXED:
        return _finish(
            type_=TYPE_CRITICAL_CONTENT_RISK,
            decision=DECISION_MANUAL_REVIEW,
            reason="critical_token_or_mixed_polarity",
            review_reason="p0_critical",
        )

    # 5. all_empty_unverified
    all_success = all(
        r.status in {RUN_STATUS_SUCCESS_TEXT, RUN_STATUS_SUCCESS_EMPTY} for r in routes
    )
    if all_success and not success_text_routes and success_empty_routes:
        return _finish(
            type_=TYPE_ALL_EMPTY_UNVERIFIED,
            decision=DECISION_MANUAL_REVIEW,
            reason="all_eight_success_empty",
            review_reason="forbid_auto_empty",
        )

    # 6. speech_presence_disagreement
    if risks.presence_conflict:
        has_affirm = any(
            polarity_of_text(r.comparison_text, patterns) == POLARITY_POSITIVE
            for r in success_text_routes
        )
        return _finish(
            type_=TYPE_SPEECH_PRESENCE_DISAGREEMENT,
            decision=DECISION_MANUAL_REVIEW,
            reason="success_empty_and_nonempty_mixed",
            presence_has_affirmation=has_affirm,
            review_reason="presence_conflict",
        )

    # 7. voicemail_candidate — ≥2 stable families, both routes hit
    if voicemail_pattern is not None:
        hit_families: list[str] = []
        for name, state in families.items():
            if state.status != FAMILY_STABLE_TEXT:
                continue
            route_hits = [
                _voicemail_hit(r.raw_text or r.comparison_text, voicemail_pattern)
                or _voicemail_hit(r.comparison_text, voicemail_pattern)
                for r in state.routes
            ]
            if len(route_hits) >= 2 and all(route_hits):
                hit_families.append(name)
        if len(hit_families) >= 2:
            return _finish(
                type_=TYPE_VOICEMAIL_CANDIDATE,
                decision=DECISION_MANUAL_REVIEW,
                reason="voicemail_multi_stable_family",
                support_family_count=len(hit_families),
                review_reason="voicemail_isolation",
            )

    # 8. qwen_correction_candidate
    if qwen_fields["qwen_correction_candidate"] and teacher_primary is not None:
        return _finish(
            type_=TYPE_QWEN_CORRECTION_CANDIDATE,
            decision=DECISION_MANUAL_REVIEW,
            reason="teachers_stable_consensus_qwen_differs",
            candidate_text=teacher_primary.candidate_text,
            support_family_count=teacher_primary.support_family_count,
            support_ratio=teacher_primary.support_ratio_of_4,
            teacher_support=teacher_primary.teacher_support_count,
            min_sim=teacher_primary.min_similarity,
            selected_run_id=teacher_primary.candidate_run_id,
            support_run_ids=[m.run_id for m in teacher_primary.members],
            review_reason="qwen_correction",
        )

    # 9. family_unstable
    if family_unstable:
        return _finish(
            type_=TYPE_FAMILY_UNSTABLE,
            decision=DECISION_MANUAL_REVIEW,
            reason="family_dual_run_unstable",
            review_reason="family_instability",
        )

    # 10. audio_quality_risk
    if (
        quality["noise_band"] in {NOISE_BAND_NOISY, NOISE_BAND_UNKNOWN}
        or quality_unknown
        or crosstalk
    ):
        return _finish(
            type_=TYPE_AUDIO_QUALITY_RISK,
            decision=DECISION_MANUAL_REVIEW,
            reason="dnsmos_noisy_or_unknown_or_crosstalk",
            review_reason="audio_quality",
        )

    # Full consensus for pseudo_high / medium
    full_consensus = analyze_consensus(
        families,
        config,
        threshold=config.pseudo_high_min_similarity,
        short=short,
    )
    if full_consensus.consensus_ambiguous:
        return _finish(
            type_=TYPE_HARDCASE,
            decision=DECISION_MANUAL_REVIEW,
            reason="consensus_ambiguous",
            ambiguous=True,
            extra_tags=[RISK_CONSENSUS_AMBIGUOUS],
            review_reason="ambiguous_max_clusters",
        )

    # 11. pseudo_high
    all_stable_text = all(s.status == FAMILY_STABLE_TEXT for s in families.values())
    eight_ok, eight_sim = eight_route_full_agreement(
        families,
        threshold=config.pseudo_high_min_similarity,
        short=short,
    )
    quality_ok = (
        quality["dnsmos_status"] == "success"
        and quality["noise_band"] in {NOISE_BAND_CLEAN, NOISE_BAND_MODERATE}
        and quality["noise_risk"] is False
        and not crosstalk
    )
    critical_ok = (
        not risks.critical_content_risk
        and not risks.semantic_risk
        and risks.polarity != POLARITY_MIXED
    )
    if (
        all_stable_text
        and eight_ok
        and quality_ok
        and critical_ok
        and full_consensus.primary is not None
        and full_consensus.primary.support_family_count == len(config.model_families)
    ):
        primary = full_consensus.primary
        return _finish(
            type_=TYPE_PSEUDO_HIGH,
            decision=DECISION_AUDIT_PENDING,
            reason="four_family_strict_consensus",
            candidate_text=primary.candidate_text,
            label_source=LABEL_SOURCE_MODEL,
            label_tier=LABEL_TIER_PSEUDO_HIGH,
            support_family_count=primary.support_family_count,
            support_ratio=primary.support_ratio_of_4,
            teacher_support=primary.teacher_support_count,
            min_sim=eight_sim if eight_sim is not None else primary.min_similarity,
            selected_run_id=primary.candidate_run_id,
            support_run_ids=[m.run_id for m in primary.members],
            review_reason="audit_pending_before_train",
        )

    # 12. pseudo_medium — ≥3 stable families, ≥0.95, remaining not higher-risk
    medium = analyze_consensus(
        families,
        config,
        threshold=config.pseudo_medium_min_similarity,
        short=short,
    )
    if (
        not medium.consensus_ambiguous
        and medium.primary is not None
        and medium.primary.support_family_count
        >= config.pseudo_medium_min_stable_families
    ):
        return _finish(
            type_=TYPE_PSEUDO_MEDIUM,
            decision=DECISION_MANUAL_REVIEW,
            reason="three_plus_family_consensus",
            candidate_text=medium.primary.candidate_text,
            label_source=LABEL_SOURCE_MODEL,
            label_tier=LABEL_TIER_PSEUDO_MEDIUM,
            support_family_count=medium.primary.support_family_count,
            support_ratio=medium.primary.support_ratio_of_4,
            teacher_support=medium.primary.teacher_support_count,
            min_sim=medium.primary.min_similarity,
            selected_run_id=medium.primary.candidate_run_id,
            support_run_ids=[m.run_id for m in medium.primary.members],
            review_reason="pseudo_medium_manual",
        )

    # 13. hardcase
    min_sim = None
    if success_text_routes:
        from audio_engine.core.selection_v3.text import pairwise_min_similarity

        min_sim = pairwise_min_similarity(
            [r.comparison_text for r in success_text_routes]
        )
    return _finish(
        type_=TYPE_HARDCASE,
        decision=DECISION_MANUAL_REVIEW,
        reason="no_unique_reliable_consensus",
        min_sim=min_sim,
        review_reason="hardcase",
    )
