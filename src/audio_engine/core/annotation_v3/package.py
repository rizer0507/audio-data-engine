"""Blind / dual-review package builders for annotation_v3."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from audio_engine.core.annotation_v3.config import AnnotationConfig
from audio_engine.core.annotation_v3.contract import encode_gold_text_for_tabular
from audio_engine.core.annotation_v3.queue import queue_id_v3, requires_dual_review, select_review_batch
from audio_engine.core.annotation_v3.types import (
    ANNOTATION_VERSION,
    EMPTY_GOLD_SENTINEL,
    NULL_GOLD_SENTINEL,
    VIEW_ADJUDICATION,
    VIEW_BLIND,
    VIEW_CANDIDATE_CHECK,
    VIEW_SECOND,
    VIEW_SPOT_CHECK,
    IMMUTABLE_EXPORT_COLUMNS,
)
from audio_engine.core.artifacts import atomic_write_json
from audio_engine.core.sample import Sample


def _original_sha(sample: Sample) -> str:
    labels = sample.labels
    return str(
        labels.get("original_audio_sha256")
        or labels.get("source_audio_sha256")
        or sample.sha256
        or ""
    )


def _risk_tags_cell(sample: Sample) -> str:
    raw = sample.labels.get("risk_tags") or []
    if isinstance(raw, str):
        return raw
    return ",".join(str(x) for x in raw)


def _shuffle_indices(n: int, seed: str) -> list[int]:
    keys = [
        (hashlib.sha256(f"{seed}\0{i}".encode()).hexdigest(), i) for i in range(n)
    ]
    keys.sort()
    return [i for _, i in keys]


def _anonymous_candidates(sample: Sample, seed: str) -> list[dict[str, str]]:
    """Anonymous, randomly ordered candidate texts (no family / model names)."""
    items: list[dict[str, str]] = []
    for key, entry in (sample.transcripts or {}).items():
        if isinstance(entry, dict):
            text = entry.get("text")
            text = "" if text is None else str(text)
        else:
            text = "" if entry is None else str(entry)
        items.append({"slot": key, "text": text})
    # Also include candidate_text if present and not already in list
    cand = sample.labels.get("candidate_text")
    if cand is not None and str(cand) not in {x["text"] for x in items}:
        items.append({"slot": "candidate", "text": str(cand)})
    order = _shuffle_indices(len(items), f"{seed}\0{sample.id}")
    out = []
    for rank, idx in enumerate(order, start=1):
        out.append({"anon_id": f"C{rank}", "text": items[idx]["text"]})
    return out


def build_export_rows(
    samples: Iterable[Sample],
    *,
    config: AnnotationConfig,
    dataset_path: str,
    revision: str,
    view: str,
    priorities: list[str],
    queues: list[str] | None = None,
    limit: int | None = None,
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    """Build export rows + queue metadata for a review package."""
    selected = select_review_batch(
        samples,
        config,
        priorities=priorities,
        queues=queues,
        limit=limit,
        seed=f"{config.blind_seed_salt}|{revision}",
        view=view,
    )
    qid = queue_id_v3(
        dataset_path,
        revision=revision,
        view=view,
        priorities=priorities,
        queues=queues,
    )
    identity = [(s.id, _original_sha(s), s.labels.get("type"), s.labels.get("candidate_text"),
                 s.labels.get("rule_version"), s.labels.get("leakage_group_id")) for s in sorted(selected, key=lambda s: s.id)]
    qid = "review_v3_" + hashlib.sha256(json.dumps([qid, identity], ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]
    rows: list[dict[str, Any]] = []
    for sample in selected:
        dual = requires_dual_review(sample, config)
        base: dict[str, Any] = {
            "sample_id": sample.id,
            "original_audio_sha256": _original_sha(sample),
            "queue_id": qid,
            "queue_revision": revision,
            "source_path": sample.source_path,
            "type": sample.labels.get("type") or sample.labels.get("classification_bucket") or "",
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
            # Editable annotation columns
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
        }
        if view == VIEW_BLIND:
            # No model answers / family names — annotator must listen first.
            pass
        elif view == VIEW_CANDIDATE_CHECK:
            cands = _anonymous_candidates(
                sample, f"{config.candidate_seed_salt}|{revision}|{qid}"
            )
            base["candidate_check_seed"] = hashlib.sha256(
                f"{config.candidate_seed_salt}|{revision}|{sample.id}".encode()
            ).hexdigest()[:12]
            for i, c in enumerate(cands, start=1):
                base[f"anon_candidate_{i}"] = c["text"]
                base[f"anon_candidate_{i}_id"] = c["anon_id"]
        elif view in {VIEW_SECOND, VIEW_SPOT_CHECK}:
            # Second reviewer / spot check also blind to first answers and family names.
            pass
        elif view == VIEW_ADJUDICATION:
            # Adjudicator may see both prior annotations (filled by import merge tool),
            # but still not family-named model columns in the pack itself.
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
        "annotation_version": config.annotation_version,
        "queue_id": qid,
        "queue_revision": revision,
        "view": view,
        "priorities": list(priorities),
        "queues": list(queues or []),
        "sample_count": len(rows),
        "immutable_rows": {r["sample_id"]: {k: r.get(k) for k in IMMUTABLE_EXPORT_COLUMNS} for r in rows},
        "dual_review_count": sum(1 for r in rows if r["requires_dual_review"] == "true"),
        "empty_gold_sentinel": EMPTY_GOLD_SENTINEL,
        "null_gold_sentinel": NULL_GOLD_SENTINEL,
        "notes": (
            "Blind pack: listen to original unpadded audio before filling fields. "
            "gold_text: __NULL__=incomplete, __EMPTY__=confirmed non_speech empty. "
            "Candidate check is a separate view after independent transcription."
        ),
    }
    return qid, rows, meta


def write_review_package(
    rows: list[dict[str, Any]],
    meta: dict[str, Any],
    output: Path,
    *,
    fmt: str = "xlsx",
) -> list[Path]:
    """Write XLSX and/or JSONL package plus sidecar meta.json."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    meta_path = output.with_suffix(".meta.json")
    if meta_path.is_file():
        existing = json.loads(meta_path.read_text(encoding="utf-8"))
        if existing != meta:
            raise ValueError("review package already exists with different identity; choose a new revision/path")
        existing_files = [meta_path]
        for suffix in ((".xlsx", ".jsonl") if fmt == "both" else (f".{fmt}",)):
            path = output.with_suffix(suffix)
            if not path.is_file():
                raise ValueError("partial review package exists; recover explicitly without overwriting annotations")
            existing_files.append(path)
        return existing_files  # Preserve any manual edits already made to the pack.
    atomic_write_json(meta_path, meta)
    written.append(meta_path)

    if fmt in {"jsonl", "both"}:
        jsonl_path = output if output.suffix == ".jsonl" else output.with_suffix(".jsonl")
        with jsonl_path.open("w", encoding="utf-8") as fh:
            for row in rows:
                # JSONL uses native null for incomplete gold_text
                payload = dict(row)
                gt = payload.get("gold_text")
                if gt == NULL_GOLD_SENTINEL:
                    payload["gold_text"] = None
                elif gt == EMPTY_GOLD_SENTINEL:
                    payload["gold_text"] = ""
                fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
        written.append(jsonl_path)

    if fmt in {"xlsx", "both"}:
        import pandas as pd

        xlsx_path = output if output.suffix in {".xlsx", ".xls"} else output.with_suffix(".xlsx")
        pd.DataFrame(rows).to_excel(xlsx_path, index=False)
        written.append(xlsx_path)

    return written


def default_artifact_dir(queue_id: str, revision: str) -> Path:
    return Path("datasets/stage1/review") / queue_id / revision
