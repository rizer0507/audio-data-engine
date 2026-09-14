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
    is_business_semantic_rule,
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
    # Compat: semantics are support / configured_family_count (see support_ratio).
    support_ratio_of_4: float | None = None
    configured_family_count: int = 0
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
    # 018 speech-rate diagnostics
    max_chars_per_sec: float | None = None
    implausible_routes: list[str] = field(default_factory=list)
    rule_version: str = RULE_VERSION
    review_reason: str | None = None
    # 020 evidence / disposition (additive; shadow vs on controlled by config)
    quality_state: str | None = None
    disposition: str | None = None
    evidence_gap_reason: str | None = None
    # 022 structured fields. Null means the old rule did not set them.
    category: str | None = None
    subtype: str | None = None
    status: str | None = None
    reason_codes: list[str] = field(default_factory=list)
    chinese_available_family_count: int | None = None
    support_ratio_of_chinese: float | None = None
    char_comparable: bool | None = None
    selected_family: str | None = None
    selected_raw_text: str | None = None
    selection_distance: float | None = None
    transcript_support_count: int | None = None
    tie_break_reason: str | None = None
    tolerance_version: str | None = None
    auxiliary_tags: list[str] = field(default_factory=list)
    semantic_evidence: list[dict[str, Any]] = field(default_factory=list)
    acoustic_evidence: dict[str, Any] = field(default_factory=dict)
    language_by_run: dict[str, str] = field(default_factory=dict)
    abstain_reasons: dict[str, str] = field(default_factory=dict)
    selection_trace: dict[str, Any] = field(default_factory=dict)
    annotation_state: str | None = None
    voicemail_library_version: str | None = None
    # 023 diagnosis contract. Empty on the legacy full-quality path.
    noise_diagnosis: dict[str, Any] = field(default_factory=dict)
    # 024. Null on older rules. Automatic rows never become human accepted.
    coverage_bucket: str | None = None
    label_grade: str | None = None
    commitment: str | None = None
    speech_presence: dict[str, Any] = field(default_factory=dict)
    usage_blocks: list[str] = field(default_factory=list)
    # 025 classify-text pre-layer. Empty on legacy.
    classify_text_policy: str = "legacy"
    classify_text_version: str = ""
    classify_text_echo_fingerprint: str = ""
    empty_reason_by_run: dict[str, list[str]] = field(default_factory=dict)
    pre_filter_language_by_run: dict[str, str] = field(default_factory=dict)
    classify_text_by_run: dict[str, str] = field(default_factory=dict)

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
            # support_ratio is canonical; support_ratio_of_4 kept for downstream compat.
            "support_ratio": self.support_ratio_of_4,
            "support_ratio_of_4": self.support_ratio_of_4,
            "configured_family_count": self.configured_family_count,
            "teacher_support_count": self.teacher_support_count,
            "min_similarity": self.min_similarity,
            "consensus_ambiguous": self.consensus_ambiguous,
            "selected_run_id": self.selected_run_id or "",
            "support_run_ids": list(self.support_run_ids),
            "noise_band": self.noise_band,
            "noise_risk": self.noise_risk,
            "dnsmos_status": self.dnsmos_status,
            "quality_state": self.quality_state or "",
            "disposition": self.disposition or "",
            "evidence_gap_reason": self.evidence_gap_reason or "",
            "qwen_status": self.qwen_status or "",
            "teacher_consensus_status": self.teacher_consensus_status or "",
            "qwen_vs_teacher_similarity_1": self.qwen_vs_teacher_similarity_1,
            "qwen_vs_teacher_similarity_2": self.qwen_vs_teacher_similarity_2,
            "qwen_risk_tags": list(self.qwen_risk_tags),
            "qwen_correction_candidate": self.qwen_correction_candidate,
            "short_utterance": self.short_utterance,
            "max_chars_per_sec": self.max_chars_per_sec,
            "implausible_routes": list(self.implausible_routes),
            "annotation_revision": policy_version,
            # Never promote candidate to gold in stage B
            "gold_text": None,
            "label": self.candidate_text if self.candidate_text is not None else "",
            # 022. Old rows remain readable; missing category is not a fifth class.
            "category": self.category,
            "subtype": self.subtype,
            "status": self.status,
            "reason_codes": list(self.reason_codes or [self.reason]),
            "chinese_available_family_count": self.chinese_available_family_count,
            "support_ratio_of_chinese": self.support_ratio_of_chinese,
            "char_comparable": self.char_comparable,
            "selected_family": self.selected_family or "",
            "selected_raw_text": self.selected_raw_text if self.selected_raw_text is not None else "",
            "selection_distance": self.selection_distance,
            "transcript_support_count": self.transcript_support_count,
            "tie_break_reason": self.tie_break_reason or "",
            "tolerance_version": self.tolerance_version or "",
            "auxiliary_tags": list(self.auxiliary_tags),
            "semantic_evidence": list(self.semantic_evidence),
            "acoustic_evidence": dict(self.acoustic_evidence),
            "language_by_run": dict(self.language_by_run),
            "abstain_reasons": dict(self.abstain_reasons),
            "selection_trace": dict(self.selection_trace),
            "voicemail_library_version": self.voicemail_library_version or "",
            "noise_diagnosis": dict(self.noise_diagnosis),
            "noise_diagnosis_status": (self.noise_diagnosis or {}).get("status") or "",
            "noise_trigger_reasons": list((self.noise_diagnosis or {}).get("trigger_reasons") or []),
            "noise_policy": (self.noise_diagnosis or {}).get("policy") or "",
            "coverage_bucket": self.coverage_bucket or "",
            "label_grade": self.label_grade or "",
            "commitment": self.commitment or "",
            "speech_presence": dict(self.speech_presence or {}),
            "usage_blocks": list(self.usage_blocks or []),
            "classify_text_policy": self.classify_text_policy or "legacy",
            "classify_text_version": self.classify_text_version or "",
            "classify_text_echo_fingerprint": self.classify_text_echo_fingerprint or "",
            "empty_reason_by_run": dict(self.empty_reason_by_run or {}),
            "pre_filter_language_by_run": dict(self.pre_filter_language_by_run or {}),
            "classify_text_by_run": dict(self.classify_text_by_run or {}),
        }
        if is_business_semantic_rule(self.rule_version):
            # Do not wipe a previously accepted gold_text. The auto body stays
            # in candidate_text and cannot become the formal verbatim reference.
            labels.pop("gold_text", None)
            labels["classification_is_human_verified"] = False
            labels["is_human_verified"] = False
        if self.annotation_state:
            labels["annotation_state"] = self.annotation_state
        elif self.decision == DECISION_EXCLUDE:
            labels["annotation_state"] = "excluded"
        elif self.decision == DECISION_RETRY:
            labels["annotation_state"] = "retry"
        elif self.decision == DECISION_AUDIT_PENDING:
            labels["annotation_state"] = "audit_pending"
        elif self.decision == DECISION_MANUAL_REVIEW:
            labels["annotation_state"] = "manual_review"
        elif self.decision == "hold":
            labels["annotation_state"] = "calibration_hold"
        else:
            labels["annotation_state"] = self.decision
        return labels


def result_as_dict(result: ClassificationResultV3) -> dict[str, Any]:
    return asdict(result)
