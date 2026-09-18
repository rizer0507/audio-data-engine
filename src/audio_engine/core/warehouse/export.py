"""Full-batch annotation pack export for warehouse (033).

Unlike default review export (priority∩queue), this covers every sample in the
classified Manifest. Batch binding is stored in package meta and checked on import.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from audio_engine.core.annotation_v3.config import AnnotationConfig
from audio_engine.core.annotation_v3.contract import encode_gold_text_for_tabular
from audio_engine.core.annotation_v3.package import write_review_package
from audio_engine.core.annotation_v3.queue import requires_dual_review
from audio_engine.core.annotation_v3.types import (
    ANNOTATION_VERSION,
    EMPTY_GOLD_SENTINEL,
    IMMUTABLE_EXPORT_COLUMNS,
    NULL_GOLD_SENTINEL,
    VIEW_ADJUDICATION,
    VIEW_BLIND,
    VIEW_CANDIDATE_CHECK,
    VIEW_SECOND,
    VIEW_SPOT_CHECK,
)
from audio_engine.core.sample import Sample
from audio_engine.core.source_naming import validate_source_name
from audio_engine.core.warehouse.identity import classified_snapshot_digest, original_audio_sha


def _risk_tags_cell(sample: Sample) -> str:
    raw = sample.labels.get("risk_tags") or []
    if isinstance(raw, str):
        return raw
    return ",".join(str(x) for x in raw)


def _shuffle_indices(n: int, seed: str) -> list[int]:
    keys = [(hashlib.sha256(f"{seed}\0{i}".encode()).hexdigest(), i) for i in range(n)]
    keys.sort()
    return [i for _, i in keys]


def _anonymous_candidates(sample: Sample, seed: str) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for key, entry in (sample.transcripts or {}).items():
        if isinstance(entry, dict):
            text = entry.get("text")
            text = "" if text is None else str(text)
        else:
            text = "" if entry is None else str(entry)
        items.append({"slot": key, "text": text})
    cand = sample.labels.get("candidate_text")
    if cand is not None and str(cand) not in {x["text"] for x in items}:
        items.append({"slot": "candidate", "text": str(cand)})
    order = _shuffle_indices(len(items), f"{seed}\0{sample.id}")
    return [{"anon_id": f"C{rank}", "text": items[idx]["text"]} for rank, idx in enumerate(order, start=1)]


def select_warehouse_export_samples(
    samples: Iterable[Sample],
    *,
    sample_ids: Sequence[str] | None = None,
) -> list[Sample]:
    """Select samples for a warehouse annotation pack.

    Default: entire classified Manifest (including environment_noise / excluded).
    Optional sample_ids enables sub-packs; freeze still requires full-batch coverage.
    """
    indexed = list(samples)
    if sample_ids is None:
        return list(indexed)
    wanted = {str(x) for x in sample_ids}
    selected = [s for s in indexed if s.id in wanted]
    missing = wanted - {s.id for s in selected}
    if missing:
        raise ValueError(f"warehouse export sample_ids not in classified snapshot: {sorted(missing)[:20]}")
    return selected


def build_warehouse_export_rows(
    samples: Iterable[Sample],
    *,
    config: AnnotationConfig,
    dataset_path: str,
    revision: str,
    batch: str,
    classified_digest: str,
    view: str = VIEW_BLIND,
    sample_ids: Sequence[str] | None = None,
    pack_index: int | None = None,
    pack_total: int | None = None,
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    """Build full-batch (or sub-pack) rows + meta with warehouse binding."""
    batch = validate_source_name(batch)
    selected = select_warehouse_export_samples(samples, sample_ids=sample_ids)
    if not selected:
        raise ValueError("warehouse export requires at least one sample")

    identity = [
        (
            s.id,
            original_audio_sha(s),
            s.labels.get("type"),
            s.labels.get("category"),
            s.labels.get("candidate_text"),
            s.labels.get("rule_version"),
        )
        for s in sorted(selected, key=lambda s: s.id)
    ]
    sample_set_digest = hashlib.sha256(
        json.dumps(identity, ensure_ascii=False, sort_keys=True, default=str).encode()
    ).hexdigest()
    scope = "full_batch" if sample_ids is None else "subset"
    q_parts = [
        "warehouse_v1",
        batch,
        str(dataset_path),
        revision,
        view,
        classified_digest,
        sample_set_digest,
        str(pack_index or 0),
    ]
    qid = "wh_pack_" + hashlib.sha256("\0".join(q_parts).encode()).hexdigest()[:16]

    rows: list[dict[str, Any]] = []
    for sample in selected:
        dual = requires_dual_review(sample, config)
        base: dict[str, Any] = {
            "sample_id": sample.id,
            "original_audio_sha256": original_audio_sha(sample),
            "queue_id": qid,
            "queue_revision": revision,
            "source_path": sample.source_path,
            "type": sample.labels.get("type") or sample.labels.get("classification_bucket") or "",
            "category": sample.labels.get("category") or "",
            # Editable: blank = keep auto category; set to override → reviewed_category on import.
            "reviewed_category": "",
            "status": sample.labels.get("status") or "",
            "outcome": sample.labels.get("outcome") or "",
            "label_tier": sample.labels.get("label_tier") or sample.labels.get("label_grade") or "",
            "label_grade": sample.labels.get("label_grade") or sample.labels.get("label_tier") or "",
            "coverage_bucket": sample.labels.get("coverage_bucket") or "",
            "commitment": sample.labels.get("commitment") or "",
            "usage_blocks": ",".join(str(item) for item in (sample.labels.get("usage_blocks") or [])),
            "risk_tags": _risk_tags_cell(sample),
            "review_priority": sample.labels.get("review_priority") or "",
            "review_queue": sample.labels.get("review_queue") or "",
            "candidate_text": sample.labels.get("candidate_text")
            if view in {VIEW_CANDIDATE_CHECK, VIEW_ADJUDICATION}
            else "",
            "requires_dual_review": "true" if dual else "false",
            "leakage_group_id": sample.labels.get("leakage_group_id") or "",
            "reservation_role": sample.labels.get("reservation_role")
            or sample.labels.get("dataset_role")
            or "",
            "decision": "",
            "gold_kind": "",
            "gold_text": NULL_GOLD_SENTINEL,
            "speech_scope": "",
            "human_semantic": "",
            "human_noise": "",
            "human_crosstalk": "",
            "verified_error_tags": "",
            "audio_event_tags": "",
            "reason": "",
            "annotator_id": "",
            "reviewer_id": "",
            "adjudicator_id": "",
            "annotation_revision": revision,
            "batch": batch,
        }
        if view == VIEW_CANDIDATE_CHECK:
            cands = _anonymous_candidates(sample, f"{config.candidate_seed_salt}|{revision}|{qid}")
            base["candidate_check_seed"] = hashlib.sha256(
                f"{config.candidate_seed_salt}|{revision}|{sample.id}".encode()
            ).hexdigest()[:12]
            for i, c in enumerate(cands, start=1):
                base[f"anon_candidate_{i}"] = c["text"]
                base[f"anon_candidate_{i}_id"] = c["anon_id"]
        elif view == VIEW_ADJUDICATION:
            base["first_gold_kind"] = sample.labels.get("annotator_gold_kind") or ""
            base["first_gold_text"] = encode_gold_text_for_tabular(
                sample.labels.get("annotator_gold_text")
                if "annotator_gold_text" in sample.labels
                else None
            )
            base["second_gold_kind"] = sample.labels.get("reviewer_gold_kind") or ""
            base["second_gold_text"] = encode_gold_text_for_tabular(
                sample.labels.get("reviewer_gold_text")
                if "reviewer_gold_text" in sample.labels
                else None
            )
        rows.append(base)

    meta = {
        "annotation_version": config.annotation_version or ANNOTATION_VERSION,
        "queue_id": qid,
        "queue_revision": revision,
        "view": view,
        "priorities": [],
        "queues": [],
        "sample_count": len(rows),
        "sample_set_digest": sample_set_digest,
        "immutable_rows": {
            r["sample_id"]: {k: r.get(k) for k in IMMUTABLE_EXPORT_COLUMNS} for r in rows
        },
        "dual_review_count": sum(1 for r in rows if r["requires_dual_review"] == "true"),
        "empty_gold_sentinel": EMPTY_GOLD_SENTINEL,
        "null_gold_sentinel": NULL_GOLD_SENTINEL,
        "warehouse_binding": {
            "batch": batch,
            "classified_manifest": str(dataset_path),
            "classified_digest": classified_digest,
            "scope": scope,
            "pack_index": pack_index,
            "pack_total": pack_total,
            "protocol": "warehouse_batch_v1",
        },
        "notes": (
            "Warehouse full-batch pack (033): covers classified Manifest samples, "
            "not only default review queues. "
            "reviewed_category: blank=keep auto category; set to override. "
            "gold_text: __NULL__=incomplete, __EMPTY__=confirmed non_speech. "
            "Do not treat auto candidate_text as human gold."
        ),
    }
    return qid, rows, meta


def export_warehouse_annotation_pack(
    samples: Sequence[Sample],
    *,
    config: AnnotationConfig,
    dataset_path: str | Path,
    output: str | Path,
    batch: str,
    revision: str,
    view: str = VIEW_BLIND,
    fmt: str = "both",
    sample_ids: Sequence[str] | None = None,
    pack_index: int | None = None,
    pack_total: int | None = None,
    classified_digest: str | None = None,
) -> dict[str, Any]:
    """Export annotation pack; never overwrite filled packs with different identity."""
    dataset_path = str(Path(dataset_path))
    digest = classified_digest or classified_snapshot_digest(samples)
    qid, rows, meta = build_warehouse_export_rows(
        samples,
        config=config,
        dataset_path=dataset_path,
        revision=revision,
        batch=batch,
        classified_digest=digest,
        view=view,
        sample_ids=sample_ids,
        pack_index=pack_index,
        pack_total=pack_total,
    )
    written = write_review_package(rows, meta, Path(output), fmt=fmt)
    return {
        "queue_id": qid,
        "sample_count": len(rows),
        "classified_digest": digest,
        "batch": validate_source_name(batch),
        "revision": revision,
        "view": view,
        "written": [str(p) for p in written],
        "warehouse_binding": meta["warehouse_binding"],
    }
