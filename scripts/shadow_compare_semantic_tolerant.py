"""Offline shadow compare for selection_v3_semantic_tolerant_20260911.

Reads an existing prepared ASR manifest. Does not rerun ASR, upload audio,
or overwrite historical classified / review / release files.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from audio_engine.core.manifest import Manifest
from audio_engine.core.selection_v3 import SelectionV3Config, classify_sample
from audio_engine.core.selection_v3.types import is_semantic_tolerant_rule


def _load(path: Path):
    return Manifest.load(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Shadow-compare semantic-tolerant v3 selection")
    parser.add_argument("--prepared", required=True, help="Existing prepared ASR manifest")
    parser.add_argument("--config", required=True, help="Semantic-tolerant selection YAML")
    parser.add_argument("--old-classified", default="", help="Optional previous classified manifest")
    parser.add_argument("--output-json", required=True, help="Comparison report path; must be a new file")
    parser.add_argument("--limit", type=int, default=0, help="Optional cap; 0 means the whole file")
    args = parser.parse_args()

    prepared_path = Path(args.prepared)
    if not prepared_path.exists():
        raise SystemExit(f"prepared manifest not found: {prepared_path}")
    config = SelectionV3Config.from_yaml(args.config)
    if not is_semantic_tolerant_rule(config.rule_version):
        raise SystemExit("config rule_version is not the semantic-tolerant rule")

    prepared = _load(prepared_path)
    samples = list(prepared.samples)
    if args.limit and args.limit > 0:
        samples = samples[: args.limit]

    counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    usable_gold = 0
    hold = 0
    retry = 0
    mandatory = 0
    spot = 0
    results_by_id = {}
    for sample in samples:
        result = classify_sample(sample, config)
        results_by_id[sample.id] = result
        key = result.category or f"unclassified:{result.status}"
        counts[key] += 1
        status_counts[str(result.status)] += 1
        if result.status == "hold":
            hold += 1
        if result.status == "retry":
            retry += 1
        if result.category in {"semantic_risk", "hardcase"} and result.status == "manual_review":
            mandatory += 1
        if result.category in {"gold", "voicemail"} and result.status == "candidate":
            spot += 1
        # Automatic candidates are not usable published gold.
        if (
            result.category in {"gold", "voicemail"}
            and result.status == "accepted"
            and result.is_human_verified
        ):
            usable_gold += 1

    old_counts: Counter[str] = Counter()
    migration: Counter[str] = Counter()
    compared = 0
    if args.old_classified:
        old_path = Path(args.old_classified)
        if not old_path.exists():
            raise SystemExit(f"old classified manifest not found: {old_path}")
        old = {sample.id: sample for sample in _load(old_path).samples}
        for sample in samples:
            previous = old.get(sample.id)
            if previous is None:
                continue
            compared += 1
            old_type = str(previous.labels.get("type") or "missing")
            old_counts[old_type] += 1
            current = results_by_id[sample.id]
            new_key = current.category or f"unclassified:{current.status}"
            migration[f"{old_type}->{new_key}"] += 1

    report = {
        "rule_version": config.rule_version,
        "prepared": str(prepared_path),
        "sample_count": len(samples),
        "prepared_total": len(prepared.samples),
        "limited": bool(args.limit and args.limit > 0),
        "category_or_unclassified": dict(counts),
        "status": dict(status_counts),
        "mandatory_review": mandatory,
        "spot_audit_candidates": spot,
        "hold": hold,
        "retry": retry,
        "mandatory_review_S_H": mandatory,
        "spot_audit_candidates_G_V": spot,
        "accepted_human_gold": usable_gold,
        "usable_published_gold": usable_gold,
        "note": "hold/retry are not usable gold. Unit tests are not a business accuracy result.",
        "old_classified_compared": compared,
        "old_type_counts": dict(old_counts),
        "migration_old_type_to_new": dict(migration),
    }
    out = Path(args.output_json)
    if out.exists():
        raise SystemExit(f"refusing to overwrite existing report: {out}")
    forbidden = out.name.startswith("classified_v3_") and "semantic_tolerant" not in out.name
    if forbidden:
        raise SystemExit("refusing to write a historical classified filename")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
