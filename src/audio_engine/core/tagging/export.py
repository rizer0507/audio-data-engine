"""Multi-tag export: parquet labels, rule details, XLSX summary, filter export."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from audio_engine.core.sample import Sample
from audio_engine.core.tagging.engine import SampleTaggingResult
from audio_engine.core.tagging.types import POLICY_VERSION


def apply_result_to_sample(
    sample: Sample,
    result: SampleTaggingResult,
    *,
    policy_version: str = POLICY_VERSION,
    snapshot_id: str = "",
) -> Sample:
    labels = dict(sample.labels or {})
    labels["tags"] = list(result.effective_tags)
    labels["tagging"] = {
        "policy_version": policy_version,
        "snapshot_id": snapshot_id,
        "eval_status": result.eval_status,
        "asr_fingerprint": result.asr_fingerprint,
        "untagged_reason": result.untagged_reason,
        "rule_coverage": {
            rr.rule_id: rr.ternary.value for rr in result.rule_results
        },
    }
    sample.labels = labels
    return sample


def write_rule_details(
    path: Path,
    results: list[SampleTaggingResult],
    *,
    batch_id: str,
    snapshot_id: str,
    policy_version: str,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for result in results:
        for rr in result.rule_results:
            rows.append(
                {
                    "batch_id": batch_id,
                    "snapshot_id": snapshot_id,
                    "policy_version": policy_version,
                    "sample_id": result.sample_id,
                    "rule_id": rr.rule_id,
                    "rule_version": rr.rule_version,
                    "tag_id": rr.tag_id,
                    "ternary": rr.ternary.value,
                    "acoustic_plan": json.dumps(rr.acoustic_plan, ensure_ascii=False),
                    "condition": json.dumps(rr.condition.as_dict(), ensure_ascii=False),
                }
            )
    frame = pd.DataFrame(rows)
    if path.suffix.lower() == ".parquet":
        frame.to_parquet(path, index=False)
    else:
        path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + ("\n" if rows else ""),
            encoding="utf-8",
        )
    return path


def write_summary_xlsx(
    path: Path,
    samples: list[Sample],
    results: list[SampleTaggingResult],
    *,
    batch_id: str,
    overlap_tags: list[str] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    by_id = {r.sample_id: r for r in results}

    sample_rows = []
    tag_counter: Counter[str] = Counter()
    status_counter: Counter[str] = Counter()
    unknown_rules = 0
    failed_deps: Counter[str] = Counter()

    for sample in samples:
        result = by_id.get(sample.id)
        tags = list(result.effective_tags) if result else list(
            (sample.labels or {}).get("tags") or []
        )
        status = (
            result.eval_status
            if result
            else str(((sample.labels or {}).get("tagging") or {}).get("eval_status") or "")
        )
        status_counter[status or "unknown"] += 1
        for t in tags:
            tag_counter[t] += 1
        if result:
            for rr in result.rule_results:
                if rr.ternary.value == "unknown":
                    unknown_rules += 1
                for dep in rr.condition.missing_deps:
                    failed_deps[dep] += 1
        sample_rows.append(
            {
                "sample_id": sample.id,
                "sha256": sample.sha256,
                "tags": ",".join(tags),
                "tag_count": len(tags),
                "eval_status": status,
                "untagged_reason": result.untagged_reason if result else None,
            }
        )

    tag_rows = [{"tag_id": k, "sample_count": v} for k, v in sorted(tag_counter.items())]
    status_rows = [{"eval_status": k, "count": v} for k, v in sorted(status_counter.items())]
    overlap_rows: list[dict[str, Any]] = []
    if overlap_tags:
        wanted = set(overlap_tags)
        for sample in samples:
            result = by_id.get(sample.id)
            tags = set(result.effective_tags if result else (sample.labels or {}).get("tags") or [])
            if wanted.issubset(tags):
                overlap_rows.append({"sample_id": sample.id, "tags": ",".join(sorted(tags))})

    dep_rows = [{"dependency": k, "count": v} for k, v in sorted(failed_deps.items())]
    overview = pd.DataFrame(
        [
            {"metric": "sample_count", "value": len(samples)},
            {"metric": "tag_assignment_count", "value": int(sum(tag_counter.values()))},
            {"metric": "unknown_rule_judgements", "value": unknown_rules},
            {
                "metric": "status_sum",
                "value": int(sum(status_counter.values())),
            },
        ]
    )

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame(sample_rows).to_excel(writer, sheet_name="samples", index=False)
        pd.DataFrame(tag_rows).to_excel(writer, sheet_name="tag_counts", index=False)
        pd.DataFrame(status_rows).to_excel(writer, sheet_name="eval_status", index=False)
        overview.to_excel(writer, sheet_name="overview", index=False)
        pd.DataFrame(dep_rows).to_excel(writer, sheet_name="failed_deps", index=False)
        if overlap_rows or overlap_tags:
            pd.DataFrame(overlap_rows).to_excel(writer, sheet_name="overlap", index=False)
    return path


def filter_samples(
    samples: list[Sample],
    *,
    include_all: list[str] | None = None,
    include_any: list[str] | None = None,
    exclude_tags: list[str] | None = None,
    require_unknown: bool = False,
    require_manual_block: bool = False,
) -> list[Sample]:
    out: list[Sample] = []
    for sample in samples:
        labels = sample.labels if isinstance(sample.labels, dict) else {}
        tags = set(labels.get("tags") or [])
        tagging = labels.get("tagging") if isinstance(labels.get("tagging"), dict) else {}
        if include_all and not set(include_all).issubset(tags):
            continue
        if include_any and not (set(include_any) & tags):
            continue
        if exclude_tags and (set(exclude_tags) & tags):
            continue
        if require_unknown:
            coverage = tagging.get("rule_coverage") or {}
            if not any(v == "unknown" for v in coverage.values()):
                continue
        if require_manual_block:
            # Best-effort: untagged_reason or sources not always on sample; check tagging note.
            if tagging.get("untagged_reason") != "matched_but_manually_blocked":
                # Also allow if tags empty but we cannot see blocks — skip unless reason set.
                continue
        out.append(sample)
    return out
