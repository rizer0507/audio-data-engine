"""Shared types for composable multi-tag engine (040)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Ternary(str, Enum):
    MATCHED = "matched"
    NOT_MATCHED = "not_matched"
    UNKNOWN = "unknown"


class EvidenceStatus(str, Enum):
    SUCCESS = "success"
    NOT_COMPUTED = "not_computed"
    NOT_REQUIRED = "not_required"
    FAILED = "failed"
    MISSING = "missing"


class SampleEvalStatus(str, Enum):
    TAGGED = "tagged"
    UNTAGGED = "untagged"
    UNRESOLVED = "unresolved"
    NOT_EVALUATED = "not_evaluated"


@dataclass
class EvidenceResult:
    """One capability computation result; never mutates business tags."""

    capability: str
    value: Any = None
    status: EvidenceStatus = EvidenceStatus.SUCCESS
    input_fingerprint: str = ""
    capability_version: str = "1"
    config_digest: str = ""
    error: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "capability": self.capability,
            "value": self.value,
            "status": self.status.value if isinstance(self.status, EvidenceStatus) else self.status,
            "input_fingerprint": self.input_fingerprint,
            "capability_version": self.capability_version,
            "config_digest": self.config_digest,
            "error": self.error,
            "details": dict(self.details),
        }


@dataclass
class ConditionResult:
    ternary: Ternary
    evidence: list[EvidenceResult] = field(default_factory=list)
    explanation: dict[str, Any] = field(default_factory=dict)
    missing_deps: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ternary": self.ternary.value,
            "evidence": [e.as_dict() for e in self.evidence],
            "explanation": dict(self.explanation),
            "missing_deps": list(self.missing_deps),
        }


@dataclass
class RuleResult:
    rule_id: str
    rule_version: str
    tag_id: str
    ternary: Ternary
    condition: ConditionResult
    acoustic_plan: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "rule_version": self.rule_version,
            "tag_id": self.tag_id,
            "ternary": self.ternary.value,
            "condition": self.condition.as_dict(),
            "acoustic_plan": list(self.acoustic_plan),
        }


@dataclass
class TagSource:
    """One provenance for a tag on a sample."""

    tag_id: str
    source_type: str  # auto_rule | manual_add | manual_block
    rule_id: str | None = None
    rule_version: str | None = None
    snapshot_id: str | None = None
    stale: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "tag_id": self.tag_id,
            "source_type": self.source_type,
            "rule_id": self.rule_id,
            "rule_version": self.rule_version,
            "snapshot_id": self.snapshot_id,
            "stale": self.stale,
            "details": dict(self.details),
        }


POLICY_VERSION = "selection_multi_tag_v1"
ENGINE_NAME = "multi_tag_v1"
