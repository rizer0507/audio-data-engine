"""Batch availability gate for selection_v3 (020 Phase 2).

Read-only over a classified (or prepared) manifest. Emits
`batch_availability.json` with blocking reasons. Does not mutate inputs or
flip `calibrated` / governance.

Usage:
  python scripts/batch_availability_v3.py \\
    --manifest datasets/stage1/derived/classified_v3_0908-30000.parquet \\
    --selection-config configs/selection/zh_asr_v3_0908_30000.yaml \\
    --output runs/local_batch_availability_0908-30000.json
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audio_engine.core.annotation_v3.config import AnnotationConfig
from audio_engine.core.annotation_v3.queue import requires_dual_review, select_review_batch
from audio_engine.core.manifest import Manifest
from audio_engine.core.selection_v3.config import SelectionV3Config
from audio_engine.core.selection_v3.disposition import decide_disposition
from audio_engine.core.selection_v3.quality_gate import (
    derive_quality_state,
    is_governance_hold,
    sample_quality_fields,
)


BLOCK_FULL_QUALITY_UNCALIBRATED = "full_quality_uncalibrated"
BLOCK_GOVERNANCE_HOLD = "full_governance_hold"
BLOCK_ZERO_PSEUDO = "zero_auto_pseudo_candidates"
BLOCK_SYSTEMIC_ROUTE_FAILURE = "systemic_route_failure"
BLOCK_HUMAN_BUDGET = "estimated_human_load_exceeds_budget"


def _estimate_review_jobs(samples, config: AnnotationConfig) -> dict:
    manual = [s for s in samples if str(s.labels.get("review_queue") or "") == "manual_review"]
    dual = sum(1 for s in manual if requires_dual_review(s, config))
    single = len(manual) - dual
    # Explicit assumption until timed pilot (020): 45s first / 30s second.
    seconds_first = 45.0
    seconds_second = 30.0
    jobs = len(manual) + dual
    minutes = (single * seconds_first + dual * (seconds_first + seconds_second)) / 60.0
    unique_audio = len(
        {
            str(
                s.labels.get("original_audio_sha256")
                or s.labels.get("source_audio_sha256")
                or s.sha256
                or s.id
            )
            for s in manual
        }
    )
    return {
        "manual_review_count": len(manual),
        "dual_review_count": dual,
        "single_review_count": single,
        "estimated_review_jobs": jobs,
        "estimated_reviewer_minutes": round(minutes, 1),
        "unique_audio_hashes": unique_audio,
        "assumption_seconds_first_pass": seconds_first,
        "assumption_seconds_second_pass": seconds_second,
    }


def analyze_batch(
    samples,
    *,
    selection: SelectionV3Config,
    annotation: AnnotationConfig,
    max_review_jobs: int | None = None,
    max_reviewer_minutes: float | None = None,
    systemic_failure_rate: float = 0.25,
) -> dict:
    n = len(samples)
    quality_states = collections.Counter()
    dispositions = collections.Counter()
    queues = collections.Counter()
    decisions = collections.Counter()
    types = collections.Counter()
    governance = 0
    pseudo = 0
    route_fail = collections.Counter()

    for sample in samples:
        q = sample_quality_fields(sample)
        qstate = str(sample.labels.get("quality_state") or "") or derive_quality_state(
            noise_band=q["noise_band"],
            noise_risk=q["noise_risk"],
            dnsmos_status=q["dnsmos_status"],
            quality_calibrated=selection.quality_calibrated,
        )
        quality_states[qstate] += 1
        typ = str(sample.labels.get("type") or "")
        decision = str(sample.labels.get("decision") or "")
        queue = str(sample.labels.get("review_queue") or "")
        tags = sample.labels.get("risk_tags") or []
        if isinstance(tags, str):
            tags = [x.strip() for x in tags.split(",") if x.strip()]
        gov = is_governance_hold(sample)
        if gov:
            governance += 1
        disp = str(sample.labels.get("disposition") or "") or decide_disposition(
            type_=typ,
            decision=decision,
            risk_tags=tags,
            quality_state=qstate,
            governance_hold=gov,
            review_queue=queue,
        )
        dispositions[disp] += 1
        queues[queue or "(empty)"] += 1
        decisions[decision or "(empty)"] += 1
        types[typ or "(empty)"] += 1
        if typ in {"pseudo_high", "pseudo_medium"} or queue == "pseudo_audit":
            pseudo += 1
        statuses = sample.labels.get("run_statuses") or {}
        if isinstance(statuses, dict):
            for key, st in statuses.items():
                if str(st) in {"failed", "missing"}:
                    route_fail[str(key)] += 1

    load = _estimate_review_jobs(samples, annotation)
    p0 = select_review_batch(
        samples, annotation, priorities=["P0"], queues=["manual_review"]
    )
    blocks: list[dict] = []

    uncal = quality_states.get("uncalibrated", 0)
    not_required = quality_states.get("not_required", 0)
    # 023: not_required is not an uncalibrated batch. Do not block publish for it.
    if n and uncal == n and not_required == 0:
        blocks.append(
            {
                "code": BLOCK_FULL_QUALITY_UNCALIBRATED,
                "severity": "error",
                "message": f"all {n} samples are quality uncalibrated; do not emit mass quality-failure transcription jobs",
                "count": uncal,
            }
        )
    if n and governance == n:
        blocks.append(
            {
                "code": BLOCK_GOVERNANCE_HOLD,
                "severity": "error",
                "message": f"all {n} samples are governance_hold; production publish blocked",
                "count": governance,
            }
        )
    if pseudo == 0:
        blocks.append(
            {
                "code": BLOCK_ZERO_PSEUDO,
                "severity": "warning",
                "message": "automatic pseudo candidates are zero",
                "count": 0,
            }
        )
    for route, fails in route_fail.items():
        rate = fails / n if n else 0.0
        if rate >= systemic_failure_rate:
            blocks.append(
                {
                    "code": BLOCK_SYSTEMIC_ROUTE_FAILURE,
                    "severity": "error",
                    "message": f"route {route} failure/missing rate {rate:.1%} exceeds {systemic_failure_rate:.0%}",
                    "route": route,
                    "count": fails,
                    "rate": round(rate, 4),
                }
            )
    if max_review_jobs is not None and load["estimated_review_jobs"] > max_review_jobs:
        blocks.append(
            {
                "code": BLOCK_HUMAN_BUDGET,
                "severity": "error",
                "message": "estimated review jobs exceed configured max_review_jobs",
                "estimated_review_jobs": load["estimated_review_jobs"],
                "max_review_jobs": max_review_jobs,
            }
        )
    if (
        max_reviewer_minutes is not None
        and load["estimated_reviewer_minutes"] > max_reviewer_minutes
    ):
        blocks.append(
            {
                "code": BLOCK_HUMAN_BUDGET,
                "severity": "error",
                "message": "estimated reviewer minutes exceed configured max_reviewer_minutes",
                "estimated_reviewer_minutes": load["estimated_reviewer_minutes"],
                "max_reviewer_minutes": max_reviewer_minutes,
            }
        )

    hard_blocks = [b for b in blocks if b["severity"] == "error"]
    permissions = {
        "allow_diagnostic_sorting": True,
        "allow_calibration_tasks": True,
        "allow_production_human_dispatch": not any(
            b["code"]
            in {
                BLOCK_FULL_QUALITY_UNCALIBRATED,
                BLOCK_GOVERNANCE_HOLD,
                BLOCK_HUMAN_BUDGET,
            }
            for b in hard_blocks
        ),
        "allow_publish": not any(
            b["code"] in {BLOCK_FULL_QUALITY_UNCALIBRATED, BLOCK_GOVERNANCE_HOLD}
            for b in hard_blocks
        )
        and pseudo > 0,
    }

    return {
        "sample_count": n,
        "quality_calibrated_config": selection.quality_calibrated,
        "refactor_020_mode": selection.refactor_020_mode,
        "counts": {
            "quality_state": dict(quality_states),
            "disposition": dict(dispositions),
            "review_queue": dict(queues),
            "decision": dict(decisions),
            "type": dict(types),
            "governance_hold": governance,
            "pseudo_candidates": pseudo,
            "route_failed_or_missing": dict(route_fail),
        },
        "export_filter_check": {
            "p0_manual_review": len(p0),
            "p0_manual_p1_leak": sum(
                1 for s in p0 if str(s.labels.get("review_priority") or "") == "P1"
            ),
        },
        "human_load_estimate": load,
        "blocks": blocks,
        "permissions": permissions,
        "notes": [
            "decision counts may exceed manual_review queue when voicemail also uses decision=manual_review",
            "disposition is derived when labels lack disposition (legacy classified)",
            "do not flip quality.calibrated or invent group IDs from this report",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--selection-config", required=True, type=Path)
    parser.add_argument("--annotation-config", type=Path, default=ROOT / "configs/annotation/zh_asr_v3.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-review-jobs", type=int, default=None)
    parser.add_argument("--max-reviewer-minutes", type=float, default=None)
    args = parser.parse_args()

    selection = SelectionV3Config.from_yaml(args.selection_config)
    annotation = (
        AnnotationConfig.load(args.annotation_config)
        if args.annotation_config.is_file()
        else AnnotationConfig.from_params({})
    )
    samples = list(Manifest.load(args.manifest))
    report = analyze_batch(
        samples,
        selection=selection,
        annotation=annotation,
        max_review_jobs=args.max_review_jobs,
        max_reviewer_minutes=args.max_reviewer_minutes,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"ok": True, "output": str(args.output), "blocks": len(report["blocks"]), "permissions": report["permissions"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
