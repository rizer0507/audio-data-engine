"""Classification result serialization for selection_v3."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from audio_engine.core.selection_v3.types import (
    DECISION_AUDIT_PENDING,
    DECISION_EXCLUDE,
    DECISION_MANUAL_REVIEW,
    DECISION_RETRY,
    LABEL_SOURCE_NONE,
    LABEL_TIER_NONE,
    RULE_VERSION,
)


@dataclass
class ClassificationResultV3:
    type: str
    decision: str
    reason: str
    review_priority: str | None = None
    review_queue: str | None = None
    candidate_text: str | None = None
    label_source: str = LABEL_SOURCE_NONE
    label_tier: str = LABEL_TIER_NONE
    is_human_verified: bool = False
    risk_tags: list[str] = field(default_factory=list)
    polarity: str = "unknown"
    family_status: dict[str, str] = field(default_factory=dict)
    support_family_count: int = 0
    support_ratio_of_4: float | None = None
    teacher_support_count: int = 0
    min_similarity: float | None = None
    consensus_ambiguous: bool = False
    selected_run_id: str | None = None
    support_run_ids: list[str] = field(default_factory=list)
    # Quality
    noise_band: str | None = None
    noise_risk: bool | None = None
    dnsmos_status: str | None = None
    # Qwen value (coexists with type)
    qwen_status: str | None = None
    teacher_consensus_status: str | None = None
    qwen_vs_teacher_similarity_1: float | None = None
    qwen_vs_teacher_similarity_2: float | None = None
    qwen_risk_tags: list[str] = field(default_factory=list)
    qwen_correction_candidate: bool = False
    short_utterance: bool = False
    rule_version: str = RULE_VERSION
    review_reason: str | None = None

    def to_labels(self, policy_version: str) -> dict[str, Any]:
        labels: dict[str, Any] = {
            "classification_bucket": self.type,
            "type": self.type,
            "decision": self.decision,
            "classification_reason_codes": [self.reason],
            "selection_policy_version": policy_version,
            "rule_version": self.rule_version,
            "review_priority": self.review_priority or "",
            "review_queue": self.review_queue or "",
            "review_reason": self.review_reason or "",
            "candidate_text": self.candidate_text if self.candidate_text is not None else "",
            "label_source": self.label_source,
            "label_tier": self.label_tier,
            "is_human_verified": False,  # stage B never claims human gold
            "risk_tags": list(self.risk_tags),
            "polarity": self.polarity,
            "family_status": dict(self.family_status),
            "support_family_count": self.support_family_count,
            "support_ratio_of_4": self.support_ratio_of_4,
            "teacher_support_count": self.teacher_support_count,
            "min_similarity": self.min_similarity,
            "consensus_ambiguous": self.consensus_ambiguous,
            "selected_run_id": self.selected_run_id or "",
            "support_run_ids": list(self.support_run_ids),
            "noise_band": self.noise_band,
            "noise_risk": self.noise_risk,
            "dnsmos_status": self.dnsmos_status,
            "qwen_status": self.qwen_status or "",
            "teacher_consensus_status": self.teacher_consensus_status or "",
            "qwen_vs_teacher_similarity_1": self.qwen_vs_teacher_similarity_1,
            "qwen_vs_teacher_similarity_2": self.qwen_vs_teacher_similarity_2,
            "qwen_risk_tags": list(self.qwen_risk_tags),
            "qwen_correction_candidate": self.qwen_correction_candidate,
            "short_utterance": self.short_utterance,
            "annotation_revision": policy_version,
            # Never promote candidate to gold in stage B
            "gold_text": None,
            "label": self.candidate_text if self.candidate_text is not None else "",
        }
        if self.decision == DECISION_EXCLUDE:
            labels["annotation_state"] = "excluded"
        elif self.decision == DECISION_RETRY:
            labels["annotation_state"] = "retry"
        elif self.decision == DECISION_AUDIT_PENDING:
            labels["annotation_state"] = "audit_pending"
        elif self.decision == DECISION_MANUAL_REVIEW:
            labels["annotation_state"] = "manual_review"
        else:
            labels["annotation_state"] = self.decision
        return labels


def result_as_dict(result: ClassificationResultV3) -> dict[str, Any]:
    return asdict(result)
