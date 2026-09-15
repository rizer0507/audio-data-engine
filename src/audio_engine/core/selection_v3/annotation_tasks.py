"""Direct manual annotation tasks for selection_five_class_v1 (027 §8).

Tasks are executable work items, not a sixth business class and not a
pending_evidence pool. Idempotent ids prevent duplicate listen/review jobs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from audio_engine.core.selection_v3.types import (
    RULE_VERSION_FIVE_CLASS,
    TASK_RESOLVE_SEMANTICS,
    TASK_TRANSCRIBE,
    TASK_VERIFY_TARGET_SPEECH,
)


@dataclass
class AnnotationTask:
    annotation_task_id: str
    sample_id: str
    task_type: str
    questions: list[str] = field(default_factory=list)
    audio_refs: dict[str, str] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)
    route_audit: dict[str, Any] = field(default_factory=dict)
    conflict_spans: list[dict[str, Any]] = field(default_factory=list)
    rule_version: str = RULE_VERSION_FIVE_CLASS
    allow_invalid_audio: bool = True
    allow_unintelligible: bool = True

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _stable_id(sample_id: str, task_type: str, questions: Iterable[str], rule_version: str) -> str:
    payload = {
        "sample_id": sample_id,
        "task_type": task_type,
        "questions": list(questions),
        "rule_version": rule_version,
    }
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:20]
    return f"ann_{task_type}_{digest}"


def merge_annotation_tasks(tasks: list[AnnotationTask]) -> list[AnnotationTask]:
    """One listen per sample: merge multiple questions into a single task when possible."""
    if len(tasks) <= 1:
        return list(tasks)
    by_sample: dict[str, list[AnnotationTask]] = {}
    for task in tasks:
        by_sample.setdefault(task.sample_id, []).append(task)
    merged: list[AnnotationTask] = []
    for sample_id, group in by_sample.items():
        if len(group) == 1:
            merged.append(group[0])
            continue
        types = {t.task_type for t in group}
        if TASK_RESOLVE_SEMANTICS in types and TASK_TRANSCRIBE in types:
            primary = TASK_RESOLVE_SEMANTICS
        elif TASK_VERIFY_TARGET_SPEECH in types:
            primary = TASK_VERIFY_TARGET_SPEECH
        elif TASK_RESOLVE_SEMANTICS in types:
            primary = TASK_RESOLVE_SEMANTICS
        else:
            primary = TASK_TRANSCRIBE
        questions: list[str] = []
        evidence: dict[str, Any] = {}
        route_audit: dict[str, Any] = {}
        conflicts: list[dict[str, Any]] = []
        audio_refs: dict[str, str] = {}
        rule_version = group[0].rule_version
        for task in group:
            for q in task.questions:
                if q not in questions:
                    questions.append(q)
            evidence.update(task.evidence or {})
            route_audit.update(task.route_audit or {})
            conflicts.extend(task.conflict_spans or [])
            audio_refs.update(task.audio_refs or {})
            rule_version = task.rule_version or rule_version
        task_id = _stable_id(sample_id, primary, questions, rule_version)
        merged.append(
            AnnotationTask(
                annotation_task_id=task_id,
                sample_id=sample_id,
                task_type=primary,
                questions=questions,
                audio_refs=audio_refs,
                evidence=evidence,
                route_audit=route_audit,
                conflict_spans=conflicts,
                rule_version=rule_version,
            )
        )
    return merged


def build_annotation_task(
    *,
    sample_id: str,
    task_type: str,
    questions: list[str],
    audio_refs: dict[str, str] | None = None,
    evidence: dict[str, Any] | None = None,
    route_audit: dict[str, Any] | None = None,
    conflict_spans: list[dict[str, Any]] | None = None,
    rule_version: str = RULE_VERSION_FIVE_CLASS,
) -> AnnotationTask:
    q = [str(item).strip() for item in questions if str(item).strip()]
    if not q:
        q = ["请听音并完成标注"]
    return AnnotationTask(
        annotation_task_id=_stable_id(sample_id, task_type, q, rule_version),
        sample_id=sample_id,
        task_type=task_type,
        questions=q,
        audio_refs=dict(audio_refs or {}),
        evidence=dict(evidence or {}),
        route_audit=dict(route_audit or {}),
        conflict_spans=list(conflict_spans or []),
        rule_version=rule_version,
    )


def apply_human_fillback(
    existing_labels: dict[str, Any],
    fillback: dict[str, Any],
    *,
    overwrite_human: bool = False,
) -> dict[str, Any]:
    """Idempotent fillback. Never overwrite formal human gold unless explicitly allowed."""
    labels = dict(existing_labels or {})
    if labels.get("is_human_verified") and not overwrite_human:
        return labels
    if labels.get("gold_text") not in (None, "") and not overwrite_human:
        # Preserve existing human/external gold; still attach task completion meta.
        completed = dict(labels.get("annotation_fillback") or {})
        completed.update({k: v for k, v in fillback.items() if k.startswith("task_")})
        labels["annotation_fillback"] = completed
        return labels

    gold = fillback.get("gold_text")
    if gold is not None:
        labels["gold_text"] = gold
        labels["is_human_verified"] = True
        labels["label_source"] = "human"
    if fillback.get("invalid_audio") or fillback.get("unintelligible"):
        labels["annotation_resolution"] = (
            "invalid_audio" if fillback.get("invalid_audio") else "unintelligible"
        )
        labels["is_human_verified"] = True
    if fillback.get("semantic_subtype"):
        labels["semantic_subtype"] = fillback["semantic_subtype"]
        labels["category"] = fillback.get("category") or labels.get("category") or "semantic_risk"
    if fillback.get("noise_kind"):
        labels["noise_kind"] = fillback["noise_kind"]
        if fillback.get("category"):
            labels["category"] = fillback["category"]
    if fillback.get("target_speech_present") is not None:
        labels["target_speech_present"] = fillback["target_speech_present"]
    labels["annotation_fillback"] = dict(fillback)
    return labels
