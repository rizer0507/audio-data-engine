"""Import validation and dual-review / adjudication state machine."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from audio_engine.core.annotation_v3.config import AnnotationConfig
from audio_engine.core.annotation_v3.contract import (
    AnnotationDraft,
    ContractViolation,
    decode_gold_text_from_tabular,
    drafts_conflict,
    may_pass_formal_gold,
    parse_boolish_crosstalk,
    validate_annotation_draft,
)
from audio_engine.core.annotation_v3.queue import requires_dual_review
from audio_engine.core.annotation_v3.types import (
    ANNOTATION_VERSION,
    GOLD_KIND_NON_SPEECH,
    GOLD_KIND_SPEECH,
    IMMUTABLE_EXPORT_COLUMNS,
    LABEL_SOURCE_HUMAN,
    LABEL_TIER_GOLD,
    PASS_ADJUDICATION,
    PASS_FIRST,
    PASS_SECOND,
    PASS_SPOT_CHECK,
    STATE_ADJUDICATED,
    STATE_ANNOTATED,
    STATE_CONFLICT,
    STATE_PENDING,
    STATE_REJECTED,
    STATE_SECOND_REVIEW,
)
from audio_engine.core.sample import Sample


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _split_tags(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    text = str(raw).strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except json.JSONDecodeError:
            pass
    return [x.strip() for x in text.replace(";", ",").split(",") if x.strip()]


def row_to_draft(row: dict[str, Any], *, actor_field: str = "annotator_id") -> AnnotationDraft:
    gold = decode_gold_text_from_tabular(row.get("gold_text"))
    decision = str(row.get("decision") or "").strip()
    kind = str(row.get("gold_kind") or "").strip() or None
    return AnnotationDraft(
        sample_id=str(row.get("sample_id") or ""),
        gold_kind=kind,
        gold_text=gold.value,
        speech_scope=str(row.get("speech_scope") or "").strip() or None,
        human_semantic=str(row.get("human_semantic") or "").strip() or None,
        human_noise=str(row.get("human_noise") or "").strip() or None,
        human_crosstalk=parse_boolish_crosstalk(row.get("human_crosstalk")),
        verified_error_tags=_split_tags(row.get("verified_error_tags")),
        audio_event_tags=_split_tags(row.get("audio_event_tags")),
        reason=str(row.get("reason") or "").strip(),
        decision=decision,
        annotator_id=str(row.get(actor_field) or row.get("annotator_id") or "").strip(),
        timestamp=str(row.get("timestamp") or "").strip() or _now_iso(),
    )


@dataclass
class ImportIssue:
    sample_id: str
    code: str
    message: str


@dataclass
class ImportResult:
    samples: list[Sample]
    applied: int = 0
    pending_left: int = 0
    conflicts: int = 0
    rejected: int = 0
    gold_promoted: int = 0
    issues: list[ImportIssue] = field(default_factory=list)
    artifact_payload: dict[str, Any] = field(default_factory=dict)


def _check_immutable(
    row: dict[str, Any],
    sample: Sample,
    *,
    expected_queue_id: str,
    expected_revision: str,
) -> list[ImportIssue]:
    issues: list[ImportIssue] = []
    sid = str(row.get("sample_id") or sample.id)
    if str(row.get("queue_id") or "") != expected_queue_id:
        issues.append(ImportIssue(sid, "queue_mismatch", "queue_id mismatch"))
    if str(row.get("queue_revision") or row.get("review_revision") or "") != expected_revision:
        issues.append(ImportIssue(sid, "stale_revision", "queue_revision mismatch"))
    orig = str(
        sample.labels.get("original_audio_sha256")
        or sample.labels.get("source_audio_sha256")
        or sample.sha256
        or ""
    )
    row_sha = str(row.get("original_audio_sha256") or row.get("sha256") or "")
    if not row_sha or not orig or row_sha != orig:
        issues.append(ImportIssue(sid, "hash_mismatch", "original_audio_sha256 mismatch"))
    # Reject edits to immutable classification identity columns when present in pack.
    for col in IMMUTABLE_EXPORT_COLUMNS:
        if col in {"sample_id", "queue_id", "queue_revision", "original_audio_sha256"}:
            continue
        if col not in row:
            continue
        # type / risk_tags / candidate_text are informational; reject only if changed from source.
        if col == "type":
            expected = str(sample.labels.get("type") or sample.labels.get("classification_bucket") or "")
            if str(row.get(col) or "") != expected and expected:
                issues.append(ImportIssue(sid, "immutable_edit", f"cannot edit column {col}"))
        if col == "candidate_text":
            expected = str(sample.labels.get("candidate_text") or "")
            got = str(row.get(col) or "")
            # Blind packs leave candidate_text blank — ignore blank.
            if got and expected and got != expected:
                issues.append(ImportIssue(sid, "immutable_edit", f"cannot edit column {col}"))
    return issues


def _draft_from_stored(sample: Sample, prefix: str) -> AnnotationDraft | None:
    kind = sample.labels.get(f"{prefix}_gold_kind")
    if kind is None and sample.labels.get(f"{prefix}_id") is None:
        return None
    gold_key = f"{prefix}_gold_text"
    gold_text = sample.labels.get(gold_key) if gold_key in sample.labels else None
    return AnnotationDraft(
        sample_id=sample.id,
        gold_kind=str(kind) if kind else None,
        gold_text=gold_text if gold_text is not None else None,
        speech_scope=sample.labels.get(f"{prefix}_speech_scope"),
        human_semantic=sample.labels.get(f"{prefix}_human_semantic"),
        human_noise=sample.labels.get(f"{prefix}_human_noise"),
        human_crosstalk=parse_boolish_crosstalk(sample.labels.get(f"{prefix}_human_crosstalk")),
        verified_error_tags=_split_tags(sample.labels.get(f"{prefix}_verified_error_tags")),
        audio_event_tags=_split_tags(sample.labels.get(f"{prefix}_audio_event_tags")),
        decision=str(sample.labels.get(f"{prefix}_decision") or "accepted"),
        annotator_id=str(sample.labels.get(f"{prefix}_id") or ""),
        timestamp=str(sample.labels.get(f"{prefix}_timestamp") or ""),
    )


def _store_pass(sample: Sample, draft: AnnotationDraft, prefix: str) -> None:
    sample.labels[f"{prefix}_id"] = draft.annotator_id
    sample.labels[f"{prefix}_timestamp"] = draft.timestamp or _now_iso()
    sample.labels[f"{prefix}_decision"] = draft.decision
    sample.labels[f"{prefix}_gold_kind"] = draft.gold_kind
    sample.labels[f"{prefix}_gold_text"] = draft.gold_text
    sample.labels[f"{prefix}_speech_scope"] = draft.speech_scope
    sample.labels[f"{prefix}_human_semantic"] = draft.human_semantic
    sample.labels[f"{prefix}_human_noise"] = draft.human_noise
    sample.labels[f"{prefix}_human_crosstalk"] = draft.human_crosstalk
    sample.labels[f"{prefix}_verified_error_tags"] = list(draft.verified_error_tags)
    sample.labels[f"{prefix}_audio_event_tags"] = list(draft.audio_event_tags)
    sample.labels[f"{prefix}_reason"] = draft.reason


def _promote_gold(sample: Sample, draft: AnnotationDraft, *, state: str) -> bool:
    if not may_pass_formal_gold(draft.gold_kind, draft.gold_text, state):
        # Still write human fields, but do not mark formal gold.
        sample.labels["gold_kind"] = draft.gold_kind
        sample.labels["gold_text"] = draft.gold_text
        sample.labels["speech_scope"] = draft.speech_scope
        sample.labels["human_semantic"] = draft.human_semantic
        sample.labels["human_noise"] = draft.human_noise
        sample.labels["human_crosstalk"] = draft.human_crosstalk
        sample.labels["verified_error_tags"] = list(draft.verified_error_tags)
        sample.labels["audio_event_tags"] = list(draft.audio_event_tags)
        sample.labels["is_human_verified"] = False
        return False

    sample.labels["gold_kind"] = draft.gold_kind
    sample.labels["gold_text"] = draft.gold_text
    if draft.gold_kind == GOLD_KIND_SPEECH:
        sample.labels["label"] = draft.gold_text
    elif draft.gold_kind == GOLD_KIND_NON_SPEECH:
        sample.labels["label"] = ""
    sample.labels["speech_scope"] = draft.speech_scope
    sample.labels["human_semantic"] = draft.human_semantic
    sample.labels["human_noise"] = draft.human_noise
    sample.labels["human_crosstalk"] = draft.human_crosstalk
    sample.labels["verified_error_tags"] = list(draft.verified_error_tags)
    sample.labels["audio_event_tags"] = list(draft.audio_event_tags)
    sample.labels["label_source"] = LABEL_SOURCE_HUMAN
    sample.labels["label_tier"] = LABEL_TIER_GOLD
    sample.labels["is_human_verified"] = True
    # Keep original type/risk_tags; do not rewrite to human_gold.
    return True


def apply_review_import_v3(
    samples: list[Sample],
    rows: list[dict[str, Any]],
    *,
    config: AnnotationConfig,
    expected_queue_id: str,
    expected_revision: str,
    review_pass: str,
    actor_id: str,
) -> ImportResult:
    """Apply a review package import. Partial rows stay pending.

    Hard rejects: unknown/duplicate IDs, hash/revision mismatch, same-person self-review,
    overwriting a finished gold with a stale revision, single accepted bypassing dual review.
    """
    indexed = {s.id: s.model_copy(deep=True) for s in samples}
    result = ImportResult(samples=[])
    seen_ids: set[str] = set()

    if review_pass not in {PASS_FIRST, PASS_SECOND, PASS_ADJUDICATION, PASS_SPOT_CHECK}:
        raise ValueError(f"invalid review_pass: {review_pass}")

    for row in rows:
        sid = str(row.get("sample_id") or "")
        if not sid:
            result.issues.append(ImportIssue("", "missing_id", "missing sample_id"))
            continue
        if sid in seen_ids:
            result.issues.append(ImportIssue(sid, "duplicate_id", "duplicate sample_id in import"))
            continue
        seen_ids.add(sid)
        if sid not in indexed:
            result.issues.append(ImportIssue(sid, "unknown_id", "unknown sample_id"))
            continue

        sample = indexed[sid]
        issues = _check_immutable(
            row,
            sample,
            expected_queue_id=expected_queue_id,
            expected_revision=expected_revision,
        )
        if issues:
            result.issues.extend(issues)
            continue

        existing_state = str(sample.labels.get("annotation_state") or "")
        if existing_state in {STATE_SECOND_REVIEW, STATE_ADJUDICATED, STATE_REJECTED}:
            result.issues.append(ImportIssue(sid, "refuse_overwrite", "finished annotation requires a new explicit revision workflow"))
            continue
        existing_rev = sample.labels.get("annotation_revision") or sample.labels.get(
            "queue_revision"
        )
        if existing_state in {
            STATE_SECOND_REVIEW,
            STATE_ADJUDICATED,
            "human_accepted",
        } and existing_rev not in {None, "", expected_revision}:
            result.issues.append(
                ImportIssue(
                    sid,
                    "refuse_overwrite",
                    f"refusing to overwrite finished annotation revision {existing_rev}",
                )
            )
            continue

        actor_field = {
            PASS_FIRST: "annotator_id",
            PASS_SECOND: "reviewer_id",
            PASS_ADJUDICATION: "adjudicator_id",
            PASS_SPOT_CHECK: "reviewer_id",
        }[review_pass]
        draft = row_to_draft(row, actor_field=actor_field)
        if actor_id:
            draft.annotator_id = actor_id
        if not draft.annotator_id:
            # Incomplete without actor is fine if decision empty; otherwise reject.
            if draft.decision:
                result.issues.append(
                    ImportIssue(sid, "missing_actor", f"{actor_field} required for completed row")
                )
                continue
            # leave pending
            sample.labels.setdefault("annotation_state", STATE_PENDING)
            sample.labels["queue_id"] = expected_queue_id
            sample.labels["annotation_revision"] = expected_revision
            sample.labels["annotation_version"] = config.annotation_version
            result.pending_left += 1
            continue

        if not draft.decision:
            sample.labels.setdefault("annotation_state", STATE_PENDING)
            sample.labels["queue_id"] = expected_queue_id
            sample.labels["annotation_revision"] = expected_revision
            result.pending_left += 1
            continue

        violations = validate_annotation_draft(
            draft,
            require_audio_event_for_non_speech=config.audio_event_tags_required_for_non_speech,
        )
        # Reject decision skips field contract except id.
        if (draft.decision or "").lower() not in {"reject", "rejected"} and violations:
            for v in violations:
                result.issues.append(ImportIssue(v.sample_id, v.code, v.message))
            continue

        dual = (draft.gold_kind == GOLD_KIND_NON_SPEECH) or requires_dual_review(sample, config) or str(
            row.get("requires_dual_review") or ""
        ).lower() in {"true", "1", "yes"}

        issue_count_before_pass = len(result.issues)
        if review_pass == PASS_FIRST:
            _apply_first(sample, draft, dual=dual, result=result, revision=expected_revision)
        elif review_pass == PASS_SECOND:
            _apply_second(sample, draft, dual=dual, result=result, revision=expected_revision)
        elif review_pass == PASS_SPOT_CHECK:
            _apply_spot_check(sample, draft, result=result, revision=expected_revision)
        else:
            _apply_adjudication(sample, draft, result=result, revision=expected_revision)

        if len(result.issues) != issue_count_before_pass:
            continue
        sample.labels["queue_id"] = expected_queue_id
        sample.labels["annotation_revision"] = expected_revision
        sample.labels["annotation_version"] = config.annotation_version or ANNOTATION_VERSION
        result.applied += 1

    # Preserve original order
    result.samples = [indexed[s.id] for s in samples]
    if config.spot_check.expand_on_critical_error:
        from audio_engine.core.annotation_v3.queue import spot_check_stratum
        blocked_layers = {spot_check_stratum(s) for s in result.samples if s.labels.get("spot_check_critical") or s.labels.get("spot_check_layer_blocked")}
        completed_layers = {layer for layer in blocked_layers if all(
            s.labels.get("spot_check_passed") or s.labels.get("annotation_state") == STATE_ADJUDICATED
            for s in result.samples if spot_check_stratum(s) == layer)}
        for sample in result.samples:
            if spot_check_stratum(sample) in blocked_layers:
                sample.labels["spot_check_layer_blocked"] = spot_check_stratum(sample) not in completed_layers
    result.artifact_payload = {
        "annotation_version": config.annotation_version,
        "queue_id": expected_queue_id,
        "queue_revision": expected_revision,
        "review_pass": review_pass,
        "actor_id": actor_id,
        "applied": result.applied,
        "pending_left": result.pending_left,
        "conflicts": result.conflicts,
        "rejected": result.rejected,
        "gold_promoted": result.gold_promoted,
        "issue_count": len(result.issues),
        "issues": [{"sample_id": i.sample_id, "code": i.code, "message": i.message} for i in result.issues],
    }
    return result


def _apply_first(
    sample: Sample,
    draft: AnnotationDraft,
    *,
    dual: bool,
    result: ImportResult,
    revision: str,
) -> None:
    if (draft.decision or "").lower() in {"reject", "rejected"}:
        _store_pass(sample, draft, "annotator")
        sample.labels["annotation_state"] = STATE_REJECTED
        sample.labels["annotator_id"] = draft.annotator_id
        sample.labels["is_human_verified"] = False
        result.rejected += 1
        return

    _store_pass(sample, draft, "annotator")
    sample.labels["annotator_id"] = draft.annotator_id
    sample.labels["annotation_state"] = STATE_ANNOTATED
    if dual:
        # Must not promote gold on first pass alone.
        sample.labels["gold_kind"] = draft.gold_kind
        sample.labels["gold_text"] = draft.gold_text
        sample.labels["is_human_verified"] = False
        sample.labels.pop("label_tier", None)
        if sample.labels.get("label_source") == LABEL_SOURCE_HUMAN:
            # Do not leave premature gold flags from a single accepted.
            sample.labels["label_source"] = "human_pending_second_review"
        return

    # Single-review path: promote gold after first complete annotation.
    if _promote_gold(sample, draft, state=STATE_ANNOTATED):
        result.gold_promoted += 1


def _apply_second(
    sample: Sample,
    draft: AnnotationDraft,
    *,
    dual: bool,
    result: ImportResult,
    revision: str,
) -> None:
    first = _draft_from_stored(sample, "annotator")
    if first is None:
        result.issues.append(
            ImportIssue(sample.id, "missing_first", "second review requires completed first pass")
        )
        return
    if not draft.annotator_id:
        result.issues.append(ImportIssue(sample.id, "missing_reviewer", "reviewer_id required"))
        return
    if draft.annotator_id == first.annotator_id:
        result.issues.append(
            ImportIssue(sample.id, "self_review", "same annotator cannot perform second review")
        )
        return
    _store_pass(sample, draft, "reviewer")
    sample.labels["reviewer_id"] = draft.annotator_id

    if drafts_conflict(first, draft):
        sample.labels["annotation_state"] = STATE_CONFLICT
        sample.labels["is_human_verified"] = False
        result.conflicts += 1
        return

    sample.labels["annotation_state"] = STATE_SECOND_REVIEW
    if _promote_gold(sample, draft, state=STATE_SECOND_REVIEW):
        result.gold_promoted += 1


def _apply_spot_check(
    sample: Sample,
    draft: AnnotationDraft,
    *,
    result: ImportResult,
    revision: str,
) -> None:
    first = _draft_from_stored(sample, "annotator")
    if first is None:
        result.issues.append(
            ImportIssue(sample.id, "missing_first", "spot check requires first annotation")
        )
        return
    if draft.annotator_id == first.annotator_id:
        result.issues.append(
            ImportIssue(sample.id, "self_review", "spot check cannot be same annotator")
        )
        return
    _store_pass(sample, draft, "spot_checker")
    sample.labels["spot_checker_id"] = draft.annotator_id
    if drafts_conflict(first, draft):
        sample.labels["annotation_state"] = STATE_CONFLICT
        sample.labels["is_human_verified"] = False
        sample.labels["label_tier"] = None
        result.conflicts += 1
        # Critical disagreement — caller may expand layer; flag it.
        sample.labels["spot_check_critical"] = True
        return
    # Agreement reinforces gold if already annotated on single path.
    if str(sample.labels.get("annotation_state") or "") == STATE_ANNOTATED:
        if _promote_gold(sample, first, state=STATE_ANNOTATED):
            result.gold_promoted += 1
    sample.labels["spot_check_passed"] = True


def _apply_adjudication(
    sample: Sample,
    draft: AnnotationDraft,
    *,
    result: ImportResult,
    revision: str,
) -> None:
    if str(sample.labels.get("annotation_state") or "") != STATE_CONFLICT:
        # Allow adjudication only from conflict (or explicit).
        if str(sample.labels.get("annotation_state") or "") != STATE_CONFLICT:
            result.issues.append(
                ImportIssue(
                    sample.id,
                    "not_in_conflict",
                    "adjudication requires conflict state",
                )
            )
            return
    first_id = str(sample.labels.get("annotator_id") or "")
    second_id = str(sample.labels.get("reviewer_id") or "")
    if draft.annotator_id and draft.annotator_id in {first_id, second_id}:
        result.issues.append(
            ImportIssue(
                sample.id,
                "self_adjudication",
                "adjudicator must be a third person",
            )
        )
        return
    if (draft.decision or "").lower() in {"reject", "rejected"}:
        _store_pass(sample, draft, "adjudicator")
        sample.labels["adjudicator_id"] = draft.annotator_id
        sample.labels["annotation_state"] = STATE_REJECTED
        sample.labels["is_human_verified"] = False
        result.rejected += 1
        return

    _store_pass(sample, draft, "adjudicator")
    sample.labels["adjudicator_id"] = draft.annotator_id
    sample.labels["annotation_state"] = STATE_ADJUDICATED
    if _promote_gold(sample, draft, state=STATE_ADJUDICATED):
        result.gold_promoted += 1


def load_review_rows(path: Path) -> list[dict[str, Any]]:
    """Load XLSX or JSONL review package rows."""
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
        return rows
    import pandas as pd

    # Keep sentinels; do not fillna("") which would destroy null encoding.
    frame = pd.read_excel(path, dtype=object)
    rows = []
    for record in frame.to_dict(orient="records"):
        cleaned = {}
        for k, v in record.items():
            if v is None:
                cleaned[k] = None
            elif isinstance(v, float) and v != v:
                cleaned[k] = None
            else:
                cleaned[k] = v
        rows.append(cleaned)
    return rows


def import_has_blocking_issues(result: ImportResult) -> bool:
    blocking = {
        "missing_human_field", "invalid_target_scope", "invalid_gold_semantic",
        "invalid_speech_scope", "invalid_human_semantic", "invalid_human_noise", "invalid_human_crosstalk",
        "speech_whitespace_only", "missing_id", "missing_reviewer",
        "queue_mismatch",
        "stale_revision",
        "hash_mismatch",
        "unknown_id",
        "duplicate_id",
        "refuse_overwrite",
        "self_review",
        "self_adjudication",
        "missing_actor",
        "missing_first",
        "not_in_conflict",
        "immutable_edit",
        "speech_null_gold",
        "speech_empty_gold",
        "non_speech_null",
        "non_speech_nonempty",
        "unintelligible_as_empty",
        "invalid_gold_kind",
        "invalid_decision",
        "non_speech_missing_event",
    }
    return any(i.code in blocking for i in result.issues)
