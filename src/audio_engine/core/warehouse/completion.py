"""Whole-batch completion gates for warehouse freeze (Manifest-layer)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from audio_engine.core.annotation_v3.queue import requires_dual_review
from audio_engine.core.annotation_v3.config import AnnotationConfig
from audio_engine.core.annotation_v3.types import (
    GOLD_KIND_INVALID,
    GOLD_KIND_UNINTELLIGIBLE,
    GOLD_KIND_AMBIGUOUS_TARGET,
    STATE_ADJUDICATED,
    STATE_ANNOTATED,
    STATE_CONFLICT,
    STATE_PENDING,
    STATE_REJECTED,
    STATE_SECOND_REVIEW,
)
from audio_engine.core.sample import Sample
from audio_engine.core.warehouse.identity import original_audio_sha, resolve_audio_ref


class WarehouseGateError(ValueError):
    """Batch warehouse freeze refused."""


TERMINAL_OK_STATES = frozenset(
    {
        STATE_ANNOTATED,
        STATE_SECOND_REVIEW,
        STATE_ADJUDICATED,
        STATE_REJECTED,
    }
)

EXPLICIT_EXCLUDE_KINDS = frozenset(
    {
        GOLD_KIND_INVALID,
        GOLD_KIND_UNINTELLIGIBLE,
        GOLD_KIND_AMBIGUOUS_TARGET,
    }
)


@dataclass
class GateIssue:
    sample_id: str
    code: str
    message: str


@dataclass
class BatchCompletionReport:
    ok: bool
    issues: list[GateIssue] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "counts": dict(self.counts),
            "issues": [
                {"sample_id": i.sample_id, "code": i.code, "message": i.message}
                for i in self.issues
            ],
        }


def final_category(sample: Sample) -> str | None:
    labels = sample.labels or {}
    reviewed = labels.get("reviewed_category")
    if reviewed is not None and str(reviewed).strip():
        return str(reviewed).strip()
    cat = labels.get("category")
    if cat is None or str(cat).strip() == "":
        return None
    return str(cat).strip()


def is_explicitly_excluded(sample: Sample) -> bool:
    """Rejected / invalid / unintelligible / ambiguous kept in warehouse but not train-eligible."""
    labels = sample.labels or {}
    state = str(labels.get("annotation_state") or "")
    if state == STATE_REJECTED:
        return True
    kind = str(labels.get("gold_kind") or "")
    if kind in EXPLICIT_EXCLUDE_KINDS:
        return True
    decision = str(
        labels.get("adjudicator_decision")
        or labels.get("reviewer_decision")
        or labels.get("annotator_decision")
        or labels.get("decision")
        or ""
    ).lower()
    if decision in {"reject", "rejected"}:
        return True
    return False


def _has_definite_human_result(sample: Sample, config: AnnotationConfig) -> tuple[bool, str | None]:
    labels = sample.labels or {}
    state = str(labels.get("annotation_state") or "")
    if state in {STATE_PENDING, "", "manual_review"}:
        return False, "pending_annotation"
    if state == STATE_CONFLICT:
        return False, "unresolved_conflict"
    if state not in TERMINAL_OK_STATES:
        return False, f"incomplete_state:{state or 'missing'}"

    dual = requires_dual_review(sample, config)
    if dual and state == STATE_ANNOTATED:
        # First pass alone is not enough when dual review is required.
        return False, "dual_review_incomplete"
    if dual and state not in {STATE_SECOND_REVIEW, STATE_ADJUDICATED, STATE_REJECTED}:
        return False, "dual_review_incomplete"

    # Rejected / explicit exclude kinds count as definite human handling.
    if is_explicitly_excluded(sample):
        return True, None

    # Speech / non_speech must have gold_kind set after human pass.
    if not labels.get("gold_kind") and state != STATE_REJECTED:
        return False, "missing_gold_kind"
    return True, None


def assert_snapshot_alignment(
    classified: Sequence[Sample],
    reviewed: Sequence[Sample],
) -> list[GateIssue]:
    """1:1 id + audio identity between classified snapshot and freeze input."""
    issues: list[GateIssue] = []
    left = {s.id: s for s in classified}
    right = {s.id: s for s in reviewed}
    if len(left) != len(classified):
        issues.append(GateIssue("", "duplicate_classified_id", "classified snapshot has duplicate sample ids"))
    if len(right) != len(reviewed):
        issues.append(GateIssue("", "duplicate_reviewed_id", "reviewed manifest has duplicate sample ids"))
    missing = sorted(set(left) - set(right))
    extra = sorted(set(right) - set(left))
    for sid in missing:
        issues.append(GateIssue(sid, "missing_in_reviewed", "sample present in classified snapshot but missing after review"))
    for sid in extra:
        issues.append(GateIssue(sid, "unknown_in_classified", "sample not in classified snapshot"))
    if len(classified) != len(reviewed):
        issues.append(
            GateIssue(
                "",
                "count_mismatch",
                f"classified={len(classified)} reviewed={len(reviewed)}",
            )
        )
    for sid, src in left.items():
        if sid not in right:
            continue
        dst = right[sid]
        src_sha = original_audio_sha(src)
        dst_sha = original_audio_sha(dst)
        if not src_sha or not dst_sha or src_sha != dst_sha:
            issues.append(GateIssue(sid, "audio_identity_mismatch", "original_audio_sha256 mismatch vs classified"))
        # Original auto category must be preserved (reviewed_category is the override).
        src_cat = str(src.labels.get("category") or "")
        dst_cat = str(dst.labels.get("category") or "")
        if src_cat and dst_cat and src_cat != dst_cat:
            issues.append(
                GateIssue(
                    sid,
                    "category_mutated",
                    "labels.category must keep auto classification; use reviewed_category for overrides",
                )
            )
    return issues


def assert_batch_complete(
    reviewed: Sequence[Sample],
    *,
    config: AnnotationConfig,
    allowed_categories: Iterable[str] | None = None,
    require_audio: bool = True,
) -> BatchCompletionReport:
    """Manifest-layer completion gate (not per-sample publish)."""
    issues: list[GateIssue] = []
    allowed = {str(x) for x in allowed_categories} if allowed_categories is not None else None
    counts = {
        "total": len(reviewed),
        "pending": 0,
        "conflict": 0,
        "dual_incomplete": 0,
        "excluded_kept": 0,
        "audio_missing_unexcused": 0,
        "invalid_category": 0,
        "ok": 0,
    }

    seen_sha: dict[str, str] = {}
    for sample in reviewed:
        sha = original_audio_sha(sample)
        if sha:
            if sha in seen_sha and seen_sha[sha] != sample.id:
                issues.append(
                    GateIssue(
                        sample.id,
                        "duplicate_audio_identity",
                        f"same original_audio_sha256 as {seen_sha[sha]}",
                    )
                )
            else:
                seen_sha[sha] = sample.id

        ok, code = _has_definite_human_result(sample, config)
        if not ok:
            issues.append(GateIssue(sample.id, code or "incomplete", "sample lacks definite human handling"))
            if code == "pending_annotation":
                counts["pending"] += 1
            elif code == "unresolved_conflict":
                counts["conflict"] += 1
            elif code and code.startswith("dual"):
                counts["dual_incomplete"] += 1
            continue

        if is_explicitly_excluded(sample):
            counts["excluded_kept"] += 1
        else:
            counts["ok"] += 1

        cat = final_category(sample)
        if allowed is not None and cat is not None and cat not in allowed:
            # Excluded / outcome=excluded may have null category — only flag when set.
            issues.append(
                GateIssue(
                    sample.id,
                    "invalid_category",
                    f"final category {cat!r} not in allowed_categories",
                )
            )
            counts["invalid_category"] += 1

        if require_audio:
            ref = resolve_audio_ref(sample)
            has_bytes = bool(ref["audio_exists"] and ref["audio_size_bytes"] > 0)
            if not has_bytes:
                if is_explicitly_excluded(sample):
                    # Explicit invalid/reject may keep a record without readable audio.
                    # Do not mutate caller samples here; freeze enrichment records keep_reason.
                    pass
                else:
                    issues.append(
                        GateIssue(
                            sample.id,
                            "audio_missing",
                            "audio path missing or empty; only explicit invalid/reject may omit audio",
                        )
                    )
                    counts["audio_missing_unexcused"] += 1
            elif not ref["original_audio_sha256"] and not ref["audio_content_sha256"]:
                issues.append(
                    GateIssue(sample.id, "audio_digest_missing", "missing audio content digest")
                )

        # Import errors must never reach freeze.
        if sample.labels.get("review_validation_failed"):
            issues.append(GateIssue(sample.id, "import_validation_failed", "review_validation_failed flag set"))

    report = BatchCompletionReport(ok=not issues, issues=issues, counts=counts)
    return report


def raise_if_incomplete(report: BatchCompletionReport) -> None:
    if report.ok:
        return
    preview = "; ".join(f"{i.sample_id}:{i.code}" for i in report.issues[:12])
    raise WarehouseGateError(
        f"warehouse freeze refused: {len(report.issues)} issue(s); {preview}"
    )
