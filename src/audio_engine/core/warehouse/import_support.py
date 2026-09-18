"""Warehouse-aware review import extras (binding + reviewed_category)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

from audio_engine.core.annotation_v3.import_workflow import ImportIssue, ImportResult
from audio_engine.core.sample import Sample
from audio_engine.core.source_naming import validate_source_name
from audio_engine.core.warehouse.categories import validate_category_value
from audio_engine.core.warehouse.identity import classified_snapshot_digest, original_audio_sha


def validate_warehouse_binding(
    package_meta: dict[str, Any],
    *,
    batch: str | None = None,
    classified_samples: Sequence[Sample] | None = None,
    classified_path: str | Path | None = None,
) -> list[str]:
    """Validate pack meta warehouse_binding against expected batch / snapshot."""
    errors: list[str] = []
    binding = package_meta.get("warehouse_binding")
    if not isinstance(binding, dict):
        return errors  # Non-warehouse packs: no extra checks.

    meta_batch = str(binding.get("batch") or "").strip()
    if not meta_batch:
        errors.append("warehouse_binding.batch missing")
    else:
        try:
            validate_source_name(meta_batch)
        except ValueError as exc:
            errors.append(str(exc))
        if batch is not None and validate_source_name(batch) != meta_batch:
            errors.append(
                f"warehouse_binding.batch {meta_batch!r} != expected batch {batch!r}"
            )

    if classified_path is not None:
        expected_path = str(Path(classified_path))
        got_path = str(binding.get("classified_manifest") or "")
        if got_path and Path(got_path).resolve() != Path(expected_path).resolve():
            # Allow stem-equal relative/absolute differences when digests match.
            if Path(got_path).name != Path(expected_path).name:
                errors.append(
                    f"warehouse_binding.classified_manifest mismatch: {got_path!r} vs {expected_path!r}"
                )

    if classified_samples is not None:
        digest = classified_snapshot_digest(classified_samples)
        expected = str(binding.get("classified_digest") or "")
        if expected and expected != digest:
            errors.append(
                "warehouse_binding.classified_digest does not match import dataset snapshot"
            )
    return errors


def apply_reviewed_category_from_rows(
    result: ImportResult,
    rows: list[dict[str, Any]],
    *,
    allowed_categories: Iterable[str] | None = None,
) -> ImportResult:
    """Apply optional reviewed_category overrides after a successful v3 import pass.

    Blank reviewed_category keeps auto labels.category. Never mutates category itself.
    """
    by_id = {s.id: s for s in result.samples}
    for row in rows:
        sid = str(row.get("sample_id") or "")
        if not sid or sid not in by_id:
            continue
        sample = by_id[sid]
        # Protect auto category identity when pack carries it.
        pack_cat = str(row.get("category") or "").strip()
        src_cat = str(sample.labels.get("category") or "").strip()
        if pack_cat and src_cat and pack_cat != src_cat:
            result.issues.append(
                ImportIssue(
                    sid,
                    "category_mutated",
                    "labels.category / pack category mismatch; use reviewed_category",
                )
            )
            continue

        reviewed = str(row.get("reviewed_category") or "").strip()
        if not reviewed:
            continue
        err = validate_category_value(reviewed, allowed_categories)
        if err:
            result.issues.append(ImportIssue(sid, "invalid_reviewed_category", err))
            continue
        sample.labels["reviewed_category"] = reviewed
        sample.labels["reviewed_category_source"] = "human"
    return result


def assert_rows_belong_to_batch_samples(
    rows: list[dict[str, Any]],
    samples: Sequence[Sample],
) -> list[ImportIssue]:
    """Reject unknown IDs / audio identity edits before merge."""
    indexed = {s.id: s for s in samples}
    issues: list[ImportIssue] = []
    seen: set[str] = set()
    for row in rows:
        sid = str(row.get("sample_id") or "")
        if not sid:
            issues.append(ImportIssue("", "missing_id", "missing sample_id"))
            continue
        if sid in seen:
            issues.append(ImportIssue(sid, "duplicate_id", "duplicate sample_id in pack"))
            continue
        seen.add(sid)
        if sid not in indexed:
            issues.append(ImportIssue(sid, "unknown_id", "sample not in batch dataset"))
            continue
        sample = indexed[sid]
        row_sha = str(row.get("original_audio_sha256") or "")
        src_sha = original_audio_sha(sample)
        if row_sha and src_sha and row_sha != src_sha:
            issues.append(ImportIssue(sid, "hash_mismatch", "original_audio_sha256 mismatch"))
    return issues
