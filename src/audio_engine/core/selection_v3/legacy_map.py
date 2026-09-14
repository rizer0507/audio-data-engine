"""Explicit map from 022 category/status to legacy type/decision/queue/tier.

Automatic gold and voicemail stay model candidates. They are not human gold
and must not use ``label_tier=gold`` or ``status=accepted``.
"""

from __future__ import annotations

from dataclasses import dataclass

from audio_engine.core.selection_v3.types import (
    CATEGORY_BUSINESS_CONSISTENT,
    CATEGORY_GOLD,
    CATEGORY_HARDCASE,
    CATEGORY_NOISE,
    CATEGORY_NON_SPEECH,
    CATEGORY_SEMANTIC_RISK,
    CATEGORY_VOICEMAIL,
    DECISION_AUDIT_PENDING,
    DECISION_EXCLUDE,
    DECISION_HOLD,
    DECISION_MANUAL_REVIEW,
    DECISION_RETRY,
    LABEL_SOURCE_MODEL,
    LABEL_SOURCE_NONE,
    LABEL_TIER_NONE,
    PRIORITY_P0,
    PRIORITY_P2,
    STATUS_AUTO_CLASSIFIED,
    STATUS_CANDIDATE,
    STATUS_EXCLUDED,
    STATUS_MANUAL_REVIEW,
    STATUS_RETRY,
    TYPE_HARDCASE,
    TYPE_INVALID_AUDIO,
    TYPE_ROUTE_QUARANTINE,
    TYPE_SEMANTIC_RISK,
)


@dataclass(frozen=True)
class LegacyFields:
    type: str
    decision: str
    review_priority: str | None
    review_queue: str
    label_tier: str
    label_source: str
    annotation_state: str


def map_legacy(
    *,
    category: str | None,
    status: str,
    reason: str = "",
) -> LegacyFields:
    if status == STATUS_EXCLUDED or reason == "broken_or_invalid_audio":
        return LegacyFields(
            type=TYPE_INVALID_AUDIO,
            decision=DECISION_EXCLUDE,
            review_priority=None,
            review_queue="exclude",
            label_tier=LABEL_TIER_NONE,
            label_source=LABEL_SOURCE_NONE,
            annotation_state="excluded",
        )
    if category == CATEGORY_SEMANTIC_RISK and status == STATUS_MANUAL_REVIEW:
        return LegacyFields(
            type=TYPE_SEMANTIC_RISK,
            decision=DECISION_MANUAL_REVIEW,
            review_priority=PRIORITY_P0,
            review_queue="manual_review",
            label_tier=LABEL_TIER_NONE,
            label_source=LABEL_SOURCE_NONE,
            annotation_state="manual_review",
        )
    if category == CATEGORY_HARDCASE and status == STATUS_MANUAL_REVIEW:
        return LegacyFields(
            type=TYPE_HARDCASE,
            decision=DECISION_MANUAL_REVIEW,
            review_priority=PRIORITY_P0,
            review_queue="manual_review",
            label_tier=LABEL_TIER_NONE,
            label_source=LABEL_SOURCE_NONE,
            annotation_state="manual_review",
        )
    if category == CATEGORY_NOISE and status == STATUS_CANDIDATE:
        return LegacyFields(
            type="noise",
            decision=DECISION_EXCLUDE,
            review_priority=None,
            review_queue="noise_archive",
            label_tier=LABEL_TIER_NONE,
            label_source="acoustic_evidence",
            annotation_state="noise_archive",
        )
    if category == CATEGORY_VOICEMAIL and status == STATUS_CANDIDATE:
        return LegacyFields(
            type="voicemail",
            decision=DECISION_AUDIT_PENDING,
            review_priority=PRIORITY_P2,
            review_queue="voicemail_spot_audit",
            label_tier="model_candidate",
            label_source=LABEL_SOURCE_MODEL,
            annotation_state="candidate",
        )
    if category == CATEGORY_GOLD and status == STATUS_CANDIDATE:
        return LegacyFields(
            type="gold",
            decision=DECISION_AUDIT_PENDING,
            review_priority=PRIORITY_P2,
            review_queue="spot_audit",
            label_tier="model_candidate",
            label_source=LABEL_SOURCE_MODEL,
            annotation_state="candidate",
        )
    if status == STATUS_RETRY:
        typ = TYPE_ROUTE_QUARANTINE if "quarantine" in reason or "speech_rate" in reason else "inference_incomplete"
        return LegacyFields(
            type=typ,
            decision=DECISION_RETRY,
            review_priority=None,
            review_queue="retry",
            label_tier=LABEL_TIER_NONE,
            label_source=LABEL_SOURCE_NONE,
            annotation_state="retry",
        )
    return LegacyFields(
        type="hold_unclassified",
        decision=DECISION_HOLD,
        review_priority=None,
        review_queue="hold",
        label_tier=LABEL_TIER_NONE,
        label_source=LABEL_SOURCE_NONE,
        annotation_state="hold",
    )


def map_business_v4(
    *,
    category: str | None,
    status: str,
    label_grade: str = "none",
    reason: str = "",
) -> LegacyFields:
    """Map 024 classes onto old consumers without calling them gold or accepted.

    ``business_consistent`` is not ``gold``. ``auto_classified`` is not ``accepted``.
    ``semantic_only`` / ``no_transcript`` are not verbatim training labels.
    """
    grade = label_grade or LABEL_TIER_NONE
    if grade in {"human_verbatim", "trusted_external_verbatim", "gold"}:
        grade = LABEL_TIER_NONE
    if status == STATUS_EXCLUDED or reason == "broken_or_invalid_audio":
        return LegacyFields(
            type=TYPE_INVALID_AUDIO,
            decision=DECISION_EXCLUDE,
            review_priority=None,
            review_queue="exclude",
            label_tier=LABEL_TIER_NONE,
            label_source=LABEL_SOURCE_NONE,
            annotation_state="excluded",
        )
    if category == CATEGORY_SEMANTIC_RISK and status == STATUS_MANUAL_REVIEW:
        return LegacyFields(
            type=TYPE_SEMANTIC_RISK,
            decision=DECISION_MANUAL_REVIEW,
            review_priority=PRIORITY_P0,
            review_queue="manual_review",
            label_tier=LABEL_TIER_NONE,
            label_source=LABEL_SOURCE_NONE,
            annotation_state="manual_review",
        )
    if category == CATEGORY_HARDCASE and status == STATUS_MANUAL_REVIEW:
        return LegacyFields(
            type=TYPE_HARDCASE,
            decision=DECISION_MANUAL_REVIEW,
            review_priority=PRIORITY_P0,
            review_queue="manual_review",
            label_tier=LABEL_TIER_NONE,
            label_source=LABEL_SOURCE_NONE,
            annotation_state="manual_review",
        )
    if category == CATEGORY_NON_SPEECH and status == STATUS_AUTO_CLASSIFIED:
        return LegacyFields(
            type="non_speech",
            decision=DECISION_AUDIT_PENDING,
            review_priority=PRIORITY_P2,
            review_queue="non_speech_spot_audit",
            label_tier="no_transcript",
            label_source="speech_presence",
            annotation_state="auto_classified",
        )
    if category == CATEGORY_VOICEMAIL and status == STATUS_AUTO_CLASSIFIED:
        return LegacyFields(
            type="voicemail",
            decision=DECISION_AUDIT_PENDING,
            review_priority=PRIORITY_P2,
            review_queue="voicemail_spot_audit",
            label_tier=grade if grade not in {LABEL_TIER_NONE, ""} else "semantic_only",
            label_source=LABEL_SOURCE_MODEL,
            annotation_state="auto_classified",
        )
    if category == CATEGORY_BUSINESS_CONSISTENT and status == STATUS_AUTO_CLASSIFIED:
        return LegacyFields(
            type="business_consistent",
            decision=DECISION_AUDIT_PENDING,
            review_priority=PRIORITY_P2,
            review_queue="spot_audit",
            label_tier=grade if grade not in {LABEL_TIER_NONE, ""} else "semantic_only",
            label_source=LABEL_SOURCE_MODEL,
            annotation_state="auto_classified",
        )
    if status == STATUS_RETRY:
        typ = TYPE_ROUTE_QUARANTINE if "quarantine" in reason or "speech_rate" in reason else "inference_incomplete"
        return LegacyFields(
            type=typ,
            decision=DECISION_RETRY,
            review_priority=None,
            review_queue="retry",
            label_tier=LABEL_TIER_NONE,
            label_source=LABEL_SOURCE_NONE,
            annotation_state="retry",
        )
    return LegacyFields(
        type="hold_unclassified",
        decision=DECISION_HOLD,
        review_priority=None,
        review_queue="hold",
        label_tier=LABEL_TIER_NONE,
        label_source=LABEL_SOURCE_NONE,
        annotation_state="hold",
    )
