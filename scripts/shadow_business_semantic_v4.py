"""Offline shadow for selection_business_semantic_v4.

Reads an existing prepared ASR manifest. Does not rerun ASR, call a remote
verifier, upload audio, or overwrite 022 classified / review / release files.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from audio_engine.core.manifest import Manifest
from audio_engine.core.selection_v3 import SelectionV3Config, classify_sample
from audio_engine.core.selection_v3.business_semantic import governance_release_status, partition_coverage
from audio_engine.core.selection_v3.types import is_business_semantic_rule


def main() -> None:
    parser = argparse.ArgumentParser(description="Shadow-run business-semantic v4 selection")
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--old-classified", default="")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-jsonl", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="overwrite existing v4 shadow files")
    args = parser.parse_args()

    prepared_path = Path(args.prepared)
    if not prepared_path.exists():
        raise SystemExit(f"prepared manifest not found: {prepared_path}")
    config = SelectionV3Config.from_yaml(args.config)
    if not is_business_semantic_rule(config.rule_version):
        raise SystemExit("config rule_version is not selection_business_semantic_v4")

    out = Path(args.output_json)
    if out.exists() and not args.force:
        raise SystemExit(f"refusing to overwrite existing report: {out}")
    if "business_semantic_v4" not in out.name:
        raise SystemExit("report filename must contain business_semantic_v4")
    jsonl_path = Path(args.output_jsonl) if args.output_jsonl else None
    if jsonl_path is not None:
        if jsonl_path.exists() and not args.force:
            raise SystemExit(f"refusing to overwrite existing jsonl: {jsonl_path}")
        if "business_semantic_v4" not in jsonl_path.name:
            raise SystemExit("jsonl filename must contain business_semantic_v4")

    samples = list(Manifest.load(prepared_path).samples)
    if args.limit and args.limit > 0:
        samples = samples[: args.limit]

    old = {}
    if args.old_classified:
        old_path = Path(args.old_classified)
        if not old_path.exists():
            raise SystemExit(f"old classified manifest not found: {old_path}")
        old = {sample.id: sample for sample in Manifest.load(old_path).samples}

    rows = []
    category_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    grade_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    migration: Counter[str] = Counter()
    governance_blocked = 0
    accepted = 0
    human_verified = 0
    ids = []
    for sample in samples:
        result = classify_sample(sample, config)
        labels = result.to_labels(config.policy_version)
        ids.append(sample.id)
        category_counts[result.category or f"unclassified:{result.status}"] += 1
        status_counts[str(result.status)] += 1
        grade_counts[result.label_grade or result.label_tier or "none"] += 1
        reason_counts[result.reason] += 1
        if result.status == "accepted" or labels.get("is_human_verified") is True:
            accepted += int(result.status == "accepted")
            human_verified += int(labels.get("is_human_verified") is True)
        gate = governance_release_status(
            {
                "missing_group_metadata": sample.labels.get("missing_group_metadata"),
                "dataset_role": sample.labels.get("dataset_role") or sample.labels.get("reservation_role"),
            }
        )
        if gate["train_eval_release"] == "blocked":
            governance_blocked += 1
        previous = old.get(sample.id)
        old_reason = ""
        old_category = ""
        old_status = ""
        if previous is not None:
            old_reason = ",".join(str(item) for item in (previous.labels.get("reason_codes") or []))
            old_category = str(previous.labels.get("category") or "")
            old_status = str(previous.labels.get("status") or "")
            migration[f"{old_status}:{old_reason or old_category}->{result.coverage_bucket}:{result.category or result.status}"] += 1
        rows.append(
            {
                "id": sample.id,
                "coverage_bucket": result.coverage_bucket,
                "category": result.category,
                "status": result.status,
                "label_tier": result.label_tier,
                "reason": result.reason,
                "old_status": old_status,
                "old_reason": old_reason,
            }
        )

    partition = partition_coverage(rows)
    report = {
        "rule_version": config.rule_version,
        "prepared": str(prepared_path),
        "sample_count": len(samples),
        "id_conserved": len(ids) == len(set(ids)) == len(samples),
        "prior_information_used": bool(getattr(config, "prior_information_used", False)),
        "semantic_verifier_mode": config.semantic_verifier_mode,
        "remote_verifier_configured": bool(str(config.semantic_verifier_endpoint or "").strip()),
        "audio_event_model_deployed": False,
        "coverage": partition,
        "category_or_unclassified": dict(category_counts),
        "status": dict(status_counts),
        "label_grade": dict(grade_counts),
        "reason": dict(reason_counts),
        "accepted": accepted,
        "human_verified": human_verified,
        "governance_release_blocked": governance_blocked,
        "train_eval_release": "blocked",
        "formal_eval_4000_claimed": False,
        "human_budget_3000_claimed": False,
        "note": (
            "Local business rules only. Empty output stays U until a calibrated speech-presence "
            "model is deployed. This report does not claim the 3,000 human budget or a 1% error bound."
        ),
        "migration_old_to_new": dict(migration.most_common(40)),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if jsonl_path is not None:
        with jsonl_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({k: report[k] for k in ("sample_count", "id_conserved", "coverage", "category_or_unclassified", "status", "label_grade", "accepted", "human_budget_3000_claimed")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
