"""Shadow run for asr_anomaly_noise_v1.

Reads an existing prepared ASR manifest. Does not rerun ASR, upload audio,
call paid services, or overwrite historical classified / review / release files.
Scoring is limited to the triggered audio subset. A missing model marks those
rows failed and still classifies the rest.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from audio_engine.core.manifest import Manifest
from audio_engine.core.selection_v3 import SelectionV3Config, classify_sample
from audio_engine.core.selection_v3.noise_trigger import (
    diagnose_samples,
    ensure_trigger_record,
    evaluate_noise_trigger,
)
from audio_engine.operators.quality.asr_anomaly_noise import _DnsMosSubsetScorer


def _human(labels: dict) -> bool:
    queue = str(labels.get("review_queue") or "")
    decision = str(labels.get("decision") or "")
    return queue in {"manual_review", "calibration_hold"} or decision in {"manual_review", "hold"}


def _candidate(labels: dict) -> bool:
    decision = str(labels.get("decision") or "")
    status = str(labels.get("status") or "")
    typ = str(labels.get("type") or "")
    if status == "candidate" and str(labels.get("category") or "") in {"gold", "voicemail"}:
        return True
    return decision == "audit_pending" or typ in {"pseudo_high", "pseudo_medium"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Shadow asr_anomaly_noise_v1 on existing ASR")
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--old-classified", default="")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--dnsmos-config", default="configs/quality/dnsmos_p835.yaml")
    parser.add_argument("--score", action="store_true", help="Score triggered audio if the model file exists")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    output = Path(args.output_json)
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    protected = {
        "classified_v3_0908-30000.parquet",
        "classified_v3_0908-30000.jsonl",
        "classified_v3_0908-30000_semantic_tolerant_20260911.parquet",
    }
    if output.name in protected:
        raise SystemExit(f"refusing protected artifact name: {output.name}")

    config = SelectionV3Config.from_yaml(args.config)
    config.noise_policy = "asr_anomaly_noise_v1"
    samples = list(Manifest.load(args.prepared).samples)
    if args.limit and args.limit > 0:
        samples = samples[: args.limit]

    reasons = Counter()
    both = 0
    for sample in samples:
        decision = evaluate_noise_trigger(sample, config)
        got = set(decision.reasons)
        if len(got) == 2:
            both += 1
        for reason in decision.reasons:
            reasons[reason] += 1

    model_note = "not_scored_pending"
    if args.score:
        import yaml

        dnsmos = yaml.safe_load(Path(args.dnsmos_config).read_text(encoding="utf-8")) or {}
        model = Path(str(dnsmos.get("model_path") or ""))
        scorer = _DnsMosSubsetScorer(dnsmos)
        model_note = "model_present" if model.is_file() else f"model_missing:{model}"
        report = diagnose_samples(samples, config, scorer, calibrated=False, threshold_version="uncalibrated")
    else:
        for sample in samples:
            ensure_trigger_record(sample, config)
        statuses = Counter(str((s.labels.get("noise_diagnosis") or {}).get("status") or "") for s in samples)
        report = {
            "status_counts": dict(statuses),
            "calls": 0,
            "scored_audio": 0,
            "cache_hits": 0,
            "not_required": statuses.get("not_required", 0),
        }

    new_human = 0
    new_candidate = 0
    new_types: Counter[str] = Counter()
    new_status: Counter[str] = Counter()
    for sample in samples:
        result = classify_sample(sample, config)
        labels = result.to_labels(config.policy_version)
        if _human(labels):
            new_human += 1
        if _candidate(labels):
            new_candidate += 1
        new_types[str(labels.get("type") or result.category or "")] += 1
        new_status[str(labels.get("status") or labels.get("decision") or "")] += 1

    old_human = None
    old_candidate = None
    if args.old_classified:
        old_samples = list(Manifest.load(args.old_classified).samples)
        old_human = sum(1 for s in old_samples if _human(s.labels))
        old_candidate = sum(1 for s in old_samples if _candidate(s.labels))

    payload = {
        "policy": "asr_anomaly_noise_v1",
        "rule_version": config.rule_version,
        "prepared": str(args.prepared),
        "sample_count": len(samples),
        "trigger_counts": {
            "all_families_no_valid_transcript": reasons.get("all_families_no_valid_transcript", 0),
            "non_chinese_transcript": reasons.get("non_chinese_transcript", 0),
            "intersection": both,
            "not_required": report["not_required"],
        },
        "diagnosis": {
            "status_counts": report["status_counts"],
            "calls": report["calls"],
            "scored_audio": report["scored_audio"],
            "cache_hits": report["cache_hits"],
            "model": model_note,
            "note": "call count is unique triggered audio scored in this run, not the input size",
        },
        "new_human": new_human,
        "new_candidate": new_candidate,
        "old_human": old_human,
        "old_candidate": old_candidate,
        "new_type_counts": dict(new_types),
        "new_status_counts": dict(new_status),
        "hold_or_retry_is_not_usable_gold": True,
        "noise_accuracy": "not_claimed_pending_listening_calibration",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("sample_count", "trigger_counts", "diagnosis", "new_human", "new_candidate", "old_human", "old_candidate")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
