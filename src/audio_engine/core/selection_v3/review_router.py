"""Review queue / priority routing for selection_v3 candidates."""

from __future__ import annotations

from audio_engine.core.selection_v3.types import (
    PRIORITY_P0,
    PRIORITY_P1,
    PRIORITY_P2,
    RISK_FILLER_AFFIRMATION,
    RISK_NEGATION_FLIP,
    RISK_PRESENCE_CONFLICT,
    RISK_REJECTION_SANITIZATION,
    TYPE_ALL_EMPTY_UNVERIFIED,
    TYPE_AUDIO_QUALITY_RISK,
    TYPE_CONTENT_COMPLEXITY,
    TYPE_CRITICAL_CONTENT_RISK,
    TYPE_FAMILY_UNSTABLE,
    TYPE_HARDCASE,
    TYPE_IMPLAUSIBLE_SPEECH_RATE,
    TYPE_PSEUDO_HIGH,
    TYPE_PSEUDO_MEDIUM,
    TYPE_QUALITY_UNCALIBRATED,
    TYPE_QWEN_CORRECTION_CANDIDATE,
    TYPE_ROUTE_QUARANTINE,
    TYPE_SEMANTIC_RISK,
    TYPE_SPEECH_PRESENCE_DISAGREEMENT,
    TYPE_VOICEMAIL_CANDIDATE,
)

# Dedicated isolation queue (not ordinary pseudo pool)
QUEUE_VOICEMAIL = "voicemail_isolation"
QUEUE_MANUAL = "manual_review"
QUEUE_AUDIT = "pseudo_audit"
QUEUE_RETRY = "retry"
QUEUE_EXCLUDE = "exclude"
QUEUE_CALIBRATION = "calibration_hold"


def route_review(
    *,
    type_: str,
    risk_tags: list[str] | set[str],
    presence_has_affirmation: bool = False,
) -> tuple[str | None, str | None]:
    """Return (priority, queue) for a classified candidate."""
    tags = set(risk_tags)

    if type_ in {TYPE_SEMANTIC_RISK, TYPE_CRITICAL_CONTENT_RISK}:
        return PRIORITY_P0, QUEUE_MANUAL

    if type_ == TYPE_SPEECH_PRESENCE_DISAGREEMENT:
        if presence_has_affirmation or RISK_PRESENCE_CONFLICT in tags:
            if presence_has_affirmation:
                return PRIORITY_P0, QUEUE_MANUAL
            return PRIORITY_P1, QUEUE_MANUAL
        return PRIORITY_P0, QUEUE_MANUAL

    if type_ == TYPE_ALL_EMPTY_UNVERIFIED:
        return PRIORITY_P1, QUEUE_MANUAL

    if type_ == TYPE_VOICEMAIL_CANDIDATE:
        return PRIORITY_P2, QUEUE_VOICEMAIL

    if type_ == TYPE_QWEN_CORRECTION_CANDIDATE:
        return PRIORITY_P1, QUEUE_MANUAL

    if type_ == TYPE_FAMILY_UNSTABLE:
        return PRIORITY_P1, QUEUE_MANUAL

    if type_ == TYPE_AUDIO_QUALITY_RISK:
        return PRIORITY_P1, QUEUE_MANUAL

    if type_ == TYPE_QUALITY_UNCALIBRATED:
        return None, QUEUE_CALIBRATION

    if type_ == TYPE_CONTENT_COMPLEXITY:
        return PRIORITY_P2, QUEUE_MANUAL

    if type_ == TYPE_ROUTE_QUARANTINE:
        return None, QUEUE_RETRY

    if type_ == TYPE_IMPLAUSIBLE_SPEECH_RATE:
        return None, QUEUE_EXCLUDE

    if type_ == TYPE_PSEUDO_HIGH:
        return None, QUEUE_AUDIT

    if type_ in {TYPE_PSEUDO_MEDIUM, TYPE_HARDCASE}:
        return PRIORITY_P2, QUEUE_MANUAL

    if tags & {RISK_NEGATION_FLIP, RISK_FILLER_AFFIRMATION, RISK_REJECTION_SANITIZATION}:
        return PRIORITY_P0, QUEUE_MANUAL

    return PRIORITY_P2, QUEUE_MANUAL
