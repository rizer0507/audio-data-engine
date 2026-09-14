"""Map classification evidence to a single mutual disposition (020)."""

from __future__ import annotations

from audio_engine.core.selection_v3.types import (
    DECISION_AUDIT_PENDING,
    DECISION_EXCLUDE,
    DECISION_HOLD,
    DECISION_RETRY,
    DISPOSITION_AUDIO_EXCLUDE,
    DISPOSITION_CALIBRATION_HOLD,
    DISPOSITION_CONTENT_COMPLEXITY,
    DISPOSITION_GOVERNANCE_HOLD,
    DISPOSITION_HUMAN_BLIND,
    DISPOSITION_MACHINE_RETRY,
    DISPOSITION_PRESENCE_CONFIRM,
    DISPOSITION_PSEUDO_PENDING_AUDIT,
    DISPOSITION_ROUTE_QUARANTINE,
    DISPOSITION_TRAIN_ASSISTED,
    DISPOSITION_VOICEMAIL_ISOLATION,
    QUALITY_STATE_FAILED,
    QUALITY_STATE_NOT_REQUIRED,
    QUALITY_STATE_UNCALIBRATED,
    QUALITY_STATE_UNSUPPORTED,
    RISK_CONTENT_COMPLEXITY,
    TYPE_ALL_EMPTY_UNVERIFIED,
    TYPE_AUDIO_QUALITY_RISK,
    TYPE_CONTENT_COMPLEXITY,
    TYPE_CRITICAL_CONTENT_RISK,
    TYPE_IMPLAUSIBLE_SPEECH_RATE,
    TYPE_INFERENCE_INCOMPLETE,
    TYPE_INVALID_AUDIO,
    TYPE_PSEUDO_HIGH,
    TYPE_PSEUDO_MEDIUM,
    TYPE_QUALITY_UNCALIBRATED,
    TYPE_QWEN_CORRECTION_CANDIDATE,
    TYPE_ROUTE_QUARANTINE,
    TYPE_SEMANTIC_RISK,
    TYPE_SPEECH_PRESENCE_DISAGREEMENT,
    TYPE_VOICEMAIL_CANDIDATE,
    RISK_IMPLAUSIBLE_SPEECH_RATE,
)


def decide_disposition(
    *,
    type_: str,
    decision: str,
    risk_tags: list[str] | set[str],
    quality_state: str,
    governance_hold: bool = False,
    review_queue: str | None = None,
) -> str:
    """Return exactly one disposition for reporting / shadow routing."""
    tags = set(risk_tags)
    queue = str(review_queue or "")

    if type_ == TYPE_INVALID_AUDIO:
        return DISPOSITION_AUDIO_EXCLUDE
    if type_ == TYPE_ROUTE_QUARANTINE:
        return DISPOSITION_ROUTE_QUARANTINE
    if type_ == TYPE_IMPLAUSIBLE_SPEECH_RATE or (
        decision == DECISION_EXCLUDE and type_ != TYPE_INFERENCE_INCOMPLETE
    ):
        # Legacy whole-sample exclude; route_quarantine path uses TYPE_ROUTE_QUARANTINE.
        return DISPOSITION_AUDIO_EXCLUDE
    if type_ == TYPE_INFERENCE_INCOMPLETE or decision == DECISION_RETRY:
        if RISK_IMPLAUSIBLE_SPEECH_RATE in tags or "implausible_speech_rate" in tags:
            return DISPOSITION_ROUTE_QUARANTINE
        return DISPOSITION_MACHINE_RETRY
    if quality_state == QUALITY_STATE_NOT_REQUIRED:
        pass
    elif type_ == TYPE_QUALITY_UNCALIBRATED or (
        type_ == TYPE_AUDIO_QUALITY_RISK
        and quality_state
        in {QUALITY_STATE_UNCALIBRATED, QUALITY_STATE_FAILED, QUALITY_STATE_UNSUPPORTED}
    ):
        if quality_state == QUALITY_STATE_FAILED:
            return DISPOSITION_MACHINE_RETRY
        return DISPOSITION_CALIBRATION_HOLD
    if type_ in {TYPE_ALL_EMPTY_UNVERIFIED, TYPE_SPEECH_PRESENCE_DISAGREEMENT}:
        return DISPOSITION_PRESENCE_CONFIRM
    if type_ == TYPE_VOICEMAIL_CANDIDATE or queue == "voicemail_isolation":
        return DISPOSITION_VOICEMAIL_ISOLATION
    if type_ == TYPE_PSEUDO_HIGH or decision == DECISION_AUDIT_PENDING:
        if governance_hold:
            return DISPOSITION_GOVERNANCE_HOLD
        return DISPOSITION_PSEUDO_PENDING_AUDIT
    if type_ == TYPE_PSEUDO_MEDIUM and governance_hold:
        return DISPOSITION_GOVERNANCE_HOLD
    if type_ == TYPE_QWEN_CORRECTION_CANDIDATE:
        return DISPOSITION_TRAIN_ASSISTED
    if type_ == TYPE_CONTENT_COMPLEXITY or (
        RISK_CONTENT_COMPLEXITY in tags
        and type_ not in {TYPE_SEMANTIC_RISK, TYPE_CRITICAL_CONTENT_RISK}
    ):
        return DISPOSITION_CONTENT_COMPLEXITY
    if decision == DECISION_HOLD:
        return DISPOSITION_CALIBRATION_HOLD
    if governance_hold and type_ not in {TYPE_SEMANTIC_RISK, TYPE_CRITICAL_CONTENT_RISK}:
        return DISPOSITION_GOVERNANCE_HOLD
    return DISPOSITION_HUMAN_BLIND
