"""Main classify_sample entry for selection_v2.0 / consensus_v2."""

from __future__ import annotations

import re
from typing import Any

from audio_engine.core.sample import Sample
from audio_engine.core.selection_engine import (
    TranscriptView,
    collect_transcripts,
    family_representative_text,
    family_semantic,
    pairwise_min_similarity,
    pick_medoid,
    semantic_class,
)
from audio_engine.core.selection_v2.audio_precheck import (
    extract_audio_features,
    is_true_silence,
)
from audio_engine.core.selection_v2.config import SelectionV2Config
from audio_engine.core.selection_v2.dataset_builder import assign_dataset_role
from audio_engine.core.selection_v2.family_consensus import (
    analyze_families,
    find_dominant_cluster,
    strict_cross_family_consensus,
    voting_views,
)
from audio_engine.core.selection_v2.result import ClassificationResultV2
from audio_engine.core.selection_v2.semantic_risk_gate import (
    analyze_semantics,
    compile_lexicon,
    short_utterance_allows_pseudo_high,
)
from audio_engine.core.selection_v2.types import (
    DATASET_ROLE_EXCLUDE,
    DECISION_AUTO_ACCEPT,
    DECISION_AUTO_EMPTY,
    DECISION_EXCLUDE,
    DECISION_MANUAL_REVIEW,
    DECISION_MODEL_REVIEW,
    LABEL_SOURCE_MODEL,
    LABEL_SOURCE_NONE,
    LABEL_TIER_NONE,
    LABEL_TIER_PSEUDO_HIGH,
    LABEL_TIER_PSEUDO_MEDIUM,
    RULE_VERSION,
    SEMANTIC_NONE,
    TYPE_CRITICAL_TOKEN_CONFLICT,
    TYPE_FAMILY_INTERNAL_CONFLICT,
    TYPE_HARDCASE,
    TYPE_HALLUCINATION,
    TYPE_INVALID_AUDIO,
    TYPE_MODEL_MISSING,
    TYPE_OVERLAP_CROSSTALK,
    TYPE_POSSIBLE_VAD_MISS,
    TYPE_PSEUDO_GOLD_HIGH,
    TYPE_PSEUDO_GOLD_MEDIUM,
    TYPE_SEMANTIC_INVERSION,
    TYPE_SEMANTIC_SANITIZATION,
    TYPE_SHORT_UTTERANCE_RISK,
    TYPE_TRUE_SILENCE,
    TYPE_VOICEMAIL,
)


def _raw_hit(sample: Sample, model: str, pattern: re.Pattern[str]) -> bool:
    entry = sample.transcripts.get(model)
    if not isinstance(entry, dict):
        return False
    extra = entry.get("extra") if isinstance(entry.get("extra"), dict) else {}
    raw = str(extra.get("raw_text") or "").strip()
    return bool(raw and pattern.search(raw))


def _voicemail_hit_families(
    sample: Sample,
    views: list[TranscriptView],
    pattern: re.Pattern[str] | None,
) -> set[str]:
    if pattern is None:
        return set()
    hit_families: set[str] = set()
    for view in views:
        texts = [view.text] if view.text else []
        entry = sample.transcripts.get(view.model)
        if isinstance(entry, dict):
            extra = entry.get("extra") if isinstance(entry.get("extra"), dict) else {}
            raw = str(extra.get("raw_text") or "").strip()
            if raw and raw not in texts:
                texts.append(raw)
        if any(pattern.search(text) for text in texts if text):
            hit_families.add(view.family)
    return hit_families


def _base_kwargs(
    *,
    audio,
    semantics=None,
    empty_families: list[str] | None = None,
    family_internal_conflict: bool = False,
    missing_family: str | None = None,
    diagnostics: dict[str, Any] | None = None,
    rule_version: str = RULE_VERSION,
) -> dict[str, Any]:
    diagnostics = diagnostics or {}
    semantics = semantics
    return {
        "qwen_family_text": diagnostics.get("qwen_family_text"),
        "sensevoice_family_text": diagnostics.get("sensevoice_family_text"),
        "semantic_qwen": diagnostics.get("semantic_qwen"),
        "semantic_sensevoice": diagnostics.get("semantic_sensevoice"),
        "audio_valid": audio.audio_valid,
        "duration_ms": audio.duration_ms,
        "speech_ratio": audio.speech_ratio,
        "vad_edge_risk": audio.vad_edge_risk,
        "overlap_risk": audio.overlap_risk,
        "noise_risk": audio.noise_risk,
        "duplicate_group_id": audio.duplicate_group_id,
        "short_utterance": bool(semantics.short_utterance) if semantics else False,
        "semantic_class": semantics.semantic_class if semantics else "unknown",
        "semantic_risk": bool(semantics.semantic_risk) if semantics else False,
        "critical_token_conflict": (
            bool(semantics.critical_token_conflict) if semantics else False
        ),
        "empty_model_families": list(empty_families or []),
        "family_internal_conflict": family_internal_conflict,
        "missing_family": missing_family,
        "rule_version": rule_version,
    }


def _finish(
    result: ClassificationResultV2,
) -> ClassificationResultV2:
    result.dataset_role = assign_dataset_role(result.type, result.decision)
    return result


def _result(
    *,
    type_: str,
    decision: str,
    reason: str,
    label: str | None = None,
    selected: TranscriptView | None = None,
    support: list[TranscriptView] | None = None,
    consensus_score: float | None = None,
    min_similarity: float | None = None,
    review_reason: str | None = None,
    subtype: str | None = None,
    label_source: str = LABEL_SOURCE_NONE,
    label_tier: str = LABEL_TIER_NONE,
    is_human_verified: bool = False,
    **extra: Any,
) -> ClassificationResultV2:
    support = support or ([] if selected is None else [selected])
    return _finish(
        ClassificationResultV2(
            type=type_,
            decision=decision,
            label=label,
            reason=reason,
            selected_model=selected.model if selected else None,
            support_models=[item.model for item in support],
            support_count=len(support),
            support_family_count=len({item.family for item in support}),
            consensus_score=consensus_score,
            min_similarity=min_similarity,
            review_reason=review_reason,
            subtype=subtype,
            label_source=label_source,
            label_tier=label_tier,
            is_human_verified=is_human_verified,
            **extra,
        )
    )


def classify_sample(
    sample: Sample,
    config: SelectionV2Config,
    *,
    voicemail_pattern: re.Pattern[str] | None = None,
) -> ClassificationResultV2:
    audio = extract_audio_features(sample)
    views = collect_transcripts(sample, config.model_families)
    nonempty = [item for item in views if item.text]
    empty = [item for item in views if not item.text]
    empty_families = sorted(
        {
            item.family
            for item in views
            if not any(v.text for v in views if v.family == item.family)
        }
    )

    patterns = compile_lexicon(config)
    diagnostics = {
        "qwen_family_text": family_representative_text(nonempty, config.primary_family),
        "sensevoice_family_text": family_representative_text(
            nonempty, config.secondary_family
        ),
        "semantic_qwen": family_semantic(
            nonempty,
            config.primary_family,
            negative_pattern=patterns["negative"],
            positive_pattern=patterns["positive"],
        ),
        "semantic_sensevoice": family_semantic(
            nonempty,
            config.secondary_family,
            negative_pattern=patterns["negative"],
            positive_pattern=patterns["positive"],
        ),
    }

    # 0. invalid audio
    if not audio.audio_valid:
        return _result(
            type_=TYPE_INVALID_AUDIO,
            decision=DECISION_EXCLUDE,
            reason="broken_or_invalid_audio",
            label="",
            **_base_kwargs(audio=audio, diagnostics=diagnostics, rule_version=config.rule_version),
        )

    family_states = analyze_families(views, config)
    any_family_conflict = any(s.internal_conflict for s in family_states.values())
    semantics = analyze_semantics(
        nonempty,
        config,
        duration_sec=audio.duration_sec,
        patterns=patterns,
    )
    base = _base_kwargs(
        audio=audio,
        semantics=semantics,
        empty_families=empty_families,
        family_internal_conflict=any_family_conflict,
        diagnostics=diagnostics,
        rule_version=config.rule_version,
    )

    # 2. overlap (if feature present)
    if audio.overlap_risk:
        return _result(
            type_=TYPE_OVERLAP_CROSSTALK,
            decision=DECISION_MODEL_REVIEW,
            reason="overlap_or_crosstalk_signal",
            label=None,
            support=nonempty,
            review_reason="overlap_risk",
            **base,
        )

    # 3. all empty → true_silence | possible_vad_miss
    if views and not nonempty:
        if is_true_silence(
            audio,
            max_speech_ratio=config.max_speech_ratio_silence,
            vad_miss_auto_empty=config.vad_miss_auto_empty,
        ):
            return _result(
                type_=TYPE_TRUE_SILENCE,
                decision=DECISION_AUTO_EMPTY,
                reason="all_empty_weak_speech_evidence",
                label="",
                support=[],
                **base,
            )
        return _result(
            type_=TYPE_POSSIBLE_VAD_MISS,
            decision=DECISION_MODEL_REVIEW,
            reason="all_empty_possible_speech_or_unknown",
            label=None,
            support=[],
            review_reason="prefer_review_over_auto_empty",
            **base,
        )

    # 4. voicemail
    voicemail_families = _voicemail_hit_families(sample, views, voicemail_pattern)
    if len(voicemail_families) >= 2:
        support = [
            item
            for item in nonempty
            if item.family in voicemail_families
            and voicemail_pattern is not None
            and (
                (item.text and voicemail_pattern.search(item.text))
                or _raw_hit(sample, item.model, voicemail_pattern)
            )
        ]
        if not support:
            support = [item for item in nonempty if item.family in voicemail_families]
        return _decide_voicemail(support, config, base)

    # 5. family internal conflict (when it blocks reliable voting and no higher risk)
    vote_pool = voting_views(family_states, exclude_conflict=True)
    if any_family_conflict and len(vote_pool) < config.min_family_count:
        # Continue — may still hit semantic risks below; if not, mark conflict.
        pass

    # 6. semantic risks — never auto-accept by majority
    if semantics.polarity_conflict:
        return _result(
            type_=TYPE_SEMANTIC_INVERSION,
            decision=DECISION_MODEL_REVIEW,
            reason="semantic_polarity_family_conflict",
            label=None,
            support=nonempty,
            review_reason="positive_negative_family_conflict",
            **base,
        )
    if semantics.sanitization_risk:
        return _result(
            type_=TYPE_SEMANTIC_SANITIZATION,
            decision=DECISION_MODEL_REVIEW,
            reason="reject_or_profanity_vs_positive",
            label=None,
            support=nonempty,
            review_reason="semantic_sanitization",
            **base,
        )
    if semantics.critical_token_conflict:
        return _result(
            type_=TYPE_CRITICAL_TOKEN_CONFLICT,
            decision=DECISION_MODEL_REVIEW,
            reason="critical_token_presence_conflict",
            label=None,
            support=nonempty,
            review_reason="critical_token_conflict",
            **base,
        )

    # 7. short utterance risk (before pseudo gold)
    if semantics.short_utterance:
        allows = short_utterance_allows_pseudo_high(
            nonempty,
            negative_pattern=patterns["negative"],
            positive_pattern=patterns["positive"],
            critical_tokens=config.critical_tokens,
        )
        if not allows:
            return _result(
                type_=TYPE_SHORT_UTTERANCE_RISK,
                decision=DECISION_MODEL_REVIEW,
                reason="short_utterance_without_strict_agreement",
                label=None,
                support=nonempty,
                review_reason="short_utterance_risk",
                **base,
            )

    # 8. majority empty + single family nonempty
    total_models = len(views)
    empty_ratio = (len(empty) / total_models) if total_models else 0.0
    nonempty_families = {item.family for item in nonempty}
    if (
        total_models > 0
        and empty_ratio >= config.empty_ratio_for_hallucination
        and len(nonempty_families) == 1
        and nonempty
    ):
        speech = audio.has_speech_evidence
        if speech is True or (speech is None and not config.vad_miss_auto_empty):
            # Prefer model_missing / disagreement over false empty
            missing = sorted(empty_families)
            primary_missing = config.primary_family in empty_families
            subtype = "qwen_missing" if primary_missing else None
            miss_family = (
                config.primary_family
                if primary_missing
                else (missing[0] if missing else None)
            )
            return _result(
                type_=TYPE_MODEL_MISSING,
                decision=DECISION_MODEL_REVIEW,
                reason="empty_majority_single_family_with_speech_or_unknown",
                label=None,
                support=nonempty,
                subtype=subtype,
                review_reason="avoid_false_empty",
                **{**base, "missing_family": miss_family},
            )
        # Explicit weak speech → hallucination
        decision = DECISION_AUTO_EMPTY if len({item.family for item in views}) >= 3 else DECISION_MODEL_REVIEW
        return _result(
            type_=TYPE_HALLUCINATION,
            decision=decision,
            reason="hallucination_isolated_family_weak_speech",
            label="" if decision == DECISION_AUTO_EMPTY else None,
            support=nonempty,
            min_similarity=pairwise_min_similarity([item.text for item in nonempty]),
            review_reason="isolated_family_others_empty",
            **base,
        )

    # Primary family empty, others nonempty → model_missing
    primary_views = [item for item in views if item.family == config.primary_family]
    primary_nonempty = [item for item in primary_views if item.text]
    other_nonempty = [item for item in nonempty if item.family != config.primary_family]
    if primary_views and not primary_nonempty and other_nonempty:
        return _result(
            type_=TYPE_MODEL_MISSING,
            decision=DECISION_MODEL_REVIEW,
            reason="primary_family_empty_conflict",
            label=None,
            support=other_nonempty,
            subtype="qwen_missing",
            review_reason="qwen_family_empty_other_nonempty",
            **{**base, "missing_family": config.primary_family},
        )

    if any_family_conflict and len(vote_pool) < config.min_family_count:
        return _result(
            type_=TYPE_FAMILY_INTERNAL_CONFLICT,
            decision=DECISION_MODEL_REVIEW,
            reason="family_internal_conflict_blocks_voting",
            label=None,
            support=nonempty,
            review_reason="family_internal_conflict",
            **base,
        )

    if len(nonempty) < 2 or len({item.family for item in nonempty}) < config.min_family_count:
        return _result(
            type_=TYPE_HARDCASE,
            decision=DECISION_MODEL_REVIEW,
            reason="insufficient_independent_families",
            label=None,
            support=nonempty,
            review_reason="need_at_least_two_families",
            **base,
        )

    # 9. pseudo_gold_high — strict cross-family consensus
    # Prefer family representatives when available; fall back to all nonempty.
    candidates = vote_pool if len({v.family for v in vote_pool}) >= config.min_family_count else nonempty
    # For high: require all nonempty models (not just reps) when all present,
    # matching v1 auto_gold spirit but with risk gates already passed.
    if len(nonempty) == len(views):
        strict = strict_cross_family_consensus(
            nonempty,
            threshold=config.strict_threshold,
            min_family_count=config.min_family_count,
        )
    else:
        strict = strict_cross_family_consensus(
            candidates,
            threshold=config.strict_threshold,
            min_family_count=config.min_family_count,
        )
    if strict is not None:
        support, min_sim = strict
        semantic_classes = {
            semantic_class(
                item.text,
                negative_pattern=patterns["negative"],
                positive_pattern=patterns["positive"],
            )
            for item in support
        }
        semantic_classes.discard(SEMANTIC_NONE)
        semantic_ok = len(semantic_classes) <= 1
        short_ok = (not semantics.short_utterance) or short_utterance_allows_pseudo_high(
            support,
            negative_pattern=patterns["negative"],
            positive_pattern=patterns["positive"],
            critical_tokens=config.critical_tokens,
        )
        if semantic_ok and short_ok and not semantics.semantic_risk and not semantics.critical_token_conflict:
            chosen = pick_medoid(support)
            return _result(
                type_=TYPE_PSEUDO_GOLD_HIGH,
                decision=DECISION_AUTO_ACCEPT,
                reason="strict_cross_family_consensus",
                label=chosen.text,
                selected=chosen,
                support=support,
                consensus_score=min_sim,
                min_similarity=min_sim,
                label_source=LABEL_SOURCE_MODEL,
                label_tier=LABEL_TIER_PSEUDO_HIGH,
                **base,
            )

    # 10. pseudo_gold_medium — dominant cluster
    cluster_hit = find_dominant_cluster(
        nonempty,
        threshold=config.consensus_threshold,
        dominant_ratio=config.dominant_ratio,
        total_models=max(len(views), 1),
        min_family_count=config.min_family_count,
    )
    if cluster_hit is not None:
        cluster, ratio, min_sim = cluster_hit
        semantic_classes = {
            semantic_class(
                item.text,
                negative_pattern=patterns["negative"],
                positive_pattern=patterns["positive"],
            )
            for item in cluster
        }
        semantic_classes.discard(SEMANTIC_NONE)
        if len(semantic_classes) <= 1 and not semantics.semantic_risk:
            chosen = pick_medoid(cluster)
            return _result(
                type_=TYPE_PSEUDO_GOLD_MEDIUM,
                decision=DECISION_MODEL_REVIEW,
                reason="dominant_cross_family_cluster",
                label=chosen.text,
                selected=chosen,
                support=cluster,
                consensus_score=ratio,
                min_similarity=min_sim,
                label_source=LABEL_SOURCE_MODEL,
                label_tier=LABEL_TIER_PSEUDO_MEDIUM,
                review_reason="pseudo_medium_needs_spot_check",
                **base,
            )

    # 11. hardcase
    min_sim = pairwise_min_similarity([item.text for item in nonempty])
    return _result(
        type_=TYPE_HARDCASE,
        decision=DECISION_MODEL_REVIEW,
        reason="no_reliable_consensus",
        label=None,
        support=nonempty,
        min_similarity=min_sim,
        review_reason="no_dominant_cluster",
        **base,
    )


def _decide_voicemail(
    support: list[TranscriptView],
    config: SelectionV2Config,
    base: dict[str, Any],
) -> ClassificationResultV2:
    if not support:
        return _result(
            type_=TYPE_VOICEMAIL,
            decision=DECISION_MANUAL_REVIEW,
            reason="voicemail_multi_family_no_text",
            label=None,
            review_reason="voicemail_hit_without_text",
            dataset_role=DATASET_ROLE_EXCLUDE,
            **base,
        )
    min_sim = pairwise_min_similarity([item.text for item in support])
    if min_sim is not None and min_sim >= config.strict_threshold:
        chosen = pick_medoid(support)
        return _result(
            type_=TYPE_VOICEMAIL,
            decision=DECISION_AUTO_ACCEPT,
            reason="voicemail_strict_consensus",
            label=chosen.text,
            selected=chosen,
            support=support,
            consensus_score=min_sim,
            min_similarity=min_sim,
            label_source=LABEL_SOURCE_MODEL,
            label_tier=LABEL_TIER_PSEUDO_HIGH,
            **base,
        )
    from audio_engine.core.selection_engine import _cluster_members

    cluster = _cluster_members(support, threshold=config.consensus_threshold)
    if cluster and len(cluster) / max(len(support), 1) >= config.dominant_ratio:
        min_cluster = pairwise_min_similarity([item.text for item in cluster])
        if min_cluster is not None and min_cluster >= config.consensus_threshold:
            chosen = pick_medoid(cluster)
            return _result(
                type_=TYPE_VOICEMAIL,
                decision=DECISION_AUTO_ACCEPT,
                reason="voicemail_dominant_cluster",
                label=chosen.text,
                selected=chosen,
                support=cluster,
                consensus_score=len(cluster) / max(len(support), 1),
                min_similarity=min_cluster,
                label_source=LABEL_SOURCE_MODEL,
                label_tier=LABEL_TIER_PSEUDO_MEDIUM,
                **base,
            )
    return _result(
        type_=TYPE_VOICEMAIL,
        decision=DECISION_MANUAL_REVIEW,
        reason="voicemail_low_agreement",
        label=None,
        support=support,
        min_similarity=min_sim,
        review_reason="voicemail_texts_disagree",
        **base,
    )
