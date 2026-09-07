"""Classification result + label serialization for selection_v2.0."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from audio_engine.core.selection_v2.types import (
    DECISION_AUTO_ACCEPT,
    DECISION_AUTO_EMPTY,
    DECISION_EXCLUDE,
    LABEL_SOURCE_NONE,
    LABEL_TIER_NONE,
    RULE_VERSION,
    V1_COMPAT_TYPE_MAP,
)


@dataclass
class ClassificationResultV2:
    type: str
    decision: str
    label: str | None
    reason: str
    selected_model: str | None = None
    support_models: list[str] = field(default_factory=list)
    support_count: int = 0
    support_family_count: int = 0
    consensus_score: float | None = None
    min_similarity: float | None = None
    qwen_family_text: str | None = None
    sensevoice_family_text: str | None = None
    semantic_qwen: str | None = None
    semantic_sensevoice: str | None = None
    review_reason: str | None = None
    rule_version: str = RULE_VERSION
    # v2 fields
    subtype: str | None = None
    label_source: str = LABEL_SOURCE_NONE
    label_tier: str = LABEL_TIER_NONE
    is_human_verified: bool = False
    audio_valid: bool = True
    duration_ms: int | None = None
    speech_ratio: float | None = None
    vad_edge_risk: bool = False
    overlap_risk: bool = False
    noise_risk: bool = False
    short_utterance: bool = False
    semantic_class: str = "unknown"
    semantic_risk: bool = False
    critical_token_conflict: bool = False
    empty_model_families: list[str] = field(default_factory=list)
    missing_family: str | None = None
    duplicate_group_id: str | None = None
    dataset_role: str = "exclude"
    family_internal_conflict: bool = False

    def to_labels(self, policy_version: str) -> dict[str, Any]:
        labels: dict[str, Any] = {
            "classification_bucket": self.type,
            "type": self.type,
            "subtype": self.subtype or "",
            "decision": self.decision,
            "classification_reason_codes": [self.reason],
            "selection_policy_version": policy_version,
            "rule_version": self.rule_version,
            "selected_model": self.selected_model or "",
            "support_models": list(self.support_models),
            "support_count": self.support_count,
            "support_family_count": self.support_family_count,
            "consensus_score": self.consensus_score,
            "min_similarity": self.min_similarity,
            "qwen_family_text": self.qwen_family_text or "",
            "sensevoice_family_text": self.sensevoice_family_text or "",
            "semantic_qwen": self.semantic_qwen or "",
            "semantic_sensevoice": self.semantic_sensevoice or "",
            "review_reason": self.review_reason or "",
            "annotation_revision": policy_version,
            "label_source": self.label_source,
            "label_tier": self.label_tier,
            "is_human_verified": self.is_human_verified,
            "audio_valid": self.audio_valid,
            "duration_ms": self.duration_ms,
            "speech_ratio": self.speech_ratio,
            "vad_edge_risk": self.vad_edge_risk,
            "overlap_risk": self.overlap_risk,
            "noise_risk": self.noise_risk,
            "short_utterance": self.short_utterance,
            "semantic_class": self.semantic_class,
            "semantic_risk": self.semantic_risk,
            "critical_token_conflict": self.critical_token_conflict,
            "empty_model_families": list(self.empty_model_families),
            "missing_family": self.missing_family or "",
            "duplicate_group_id": self.duplicate_group_id or "",
            "dataset_role": self.dataset_role,
            "family_internal_conflict": self.family_internal_conflict,
        }
        # Transition: also expose v1-compatible alias for tooling that still keys on old names.
        v1_alias = V1_COMPAT_TYPE_MAP.get(self.type)
        if v1_alias and not self.subtype:
            labels["v1_type_alias"] = v1_alias

        if self.decision == DECISION_AUTO_ACCEPT:
            text = str(self.label or "")
            labels.update(
                {
                    "annotation_state": "auto_accepted",
                    "gold_text": text,
                    "label": text,
                    "gold_source": self.selected_model or "",
                }
            )
        elif self.decision == DECISION_AUTO_EMPTY:
            labels.update(
                {
                    "annotation_state": "auto_accepted",
                    "gold_text": "",
                    "label": "",
                    "gold_source": self.selected_model or "",
                }
            )
        elif self.decision == DECISION_EXCLUDE:
            labels.update(
                {
                    "annotation_state": "excluded",
                    "gold_text": "",
                    "label": "",
                    "gold_source": "",
                }
            )
        else:
            labels.setdefault("gold_text", "")
            labels.setdefault("label", self.label if self.label is not None else "")
        return labels


def result_as_dict(result: ClassificationResultV2) -> dict[str, Any]:
    return asdict(result)
