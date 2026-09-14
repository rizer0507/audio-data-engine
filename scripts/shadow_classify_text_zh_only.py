"""Offline 0908 shadow report for classify_text_zh_only_v1.

Reads an existing prepared ASR manifest. Does not rerun ASR, open DNSMOS, or
overwrite classified_v3_* / 022 / 023 / 024 / review / release files.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from audio_engine.core.manifest import Manifest
from audio_engine.core.selection_v3 import SelectionV3Config, classify_sample
from audio_engine.core.selection_v3.classify_text import CLASSIFY_TEXT_VERSION, uses_chinese_only_text


def _load_old(path: str) -> dict[str, object]:
    if not path:
        return {}
    old_path = Path(path)
    if not old_path.exists():
        raise SystemExit(f"baseline classified not found: {old_path}")
    return {sample.id: sample for sample in Manifest.load(old_path).samples}


def main() -> None:
    parser = argparse.ArgumentParser(description="Shadow-run classify_text_zh_only_v1")
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--old-022", default="")
    parser.add_argument("--old-024", default="")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-jsonl", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    prepared_path = Path(args.prepared)
    if not prepared_path.exists():
        raise SystemExit(f"prepared manifest not found: {prepared_path}")
    config = SelectionV3Config.from_yaml(args.config)
    if not uses_chinese_only_text(config.classify_text_policy):
        raise SystemExit("config classify_text.policy is not chinese_only_v1")

    out = Path(args.output_json)
    if out.exists() and not args.force:
        raise SystemExit(f"refusing to overwrite existing report: {out}")
    if "zh_only" not in out.name:
        raise SystemExit("report filename must contain zh_only")
    jsonl_path = Path(args.output_jsonl) if args.output_jsonl else None
    if jsonl_path is not None:
        if jsonl_path.exists() and not args.force:
            raise SystemExit(f"refusing to overwrite existing jsonl: {jsonl_path}")
        if "zh_only" not in jsonl_path.name:
            raise SystemExit("jsonl filename must contain zh_only")

    samples = list(Manifest.load(prepared_path).samples)
    if args.limit and args.limit > 0:
        samples = samples[: args.limit]

    old_022 = _load_old(args.old_022)
    old_024 = _load_old(args.old_024)

    empty_reasons: Counter[str] = Counter()
    empty_routes = 0
    all_empty = 0
    category_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    speech_rate = 0
    language_hold = 0
    ids: list[str] = []
    rows = []
    migrate_022: Counter[str] = Counter()
    migrate_024: Counter[str] = Counter()

    for sample in samples:
        raw_by_run = {
            key: (
                (entry.get("extra") or {}).get("raw_text")
                if isinstance(entry, dict) and isinstance(entry.get("extra"), dict)
                else (entry.get("text") if isinstance(entry, dict) else entry)
            )
            for key, entry in sample.transcripts.items()
        }
        result = classify_sample(sample, config)
        ids.append(sample.id)
        reasons = result.empty_reason_by_run or {}
        emptied = 0
        for run_id, codes in reasons.items():
            if codes:
                emptied += 1
                empty_routes += 1
                for code in codes:
                    empty_reasons[str(code)] += 1
        if emptied and emptied == len(config.all_transcript_keys()):
            all_empty += 1
        if result.implausible_routes:
            speech_rate += 1
        if "language_unverified" in result.reason_codes or "language_unresolved" in result.reason_codes:
            language_hold += 1
        category_counts[result.category or f"unclassified:{result.status}"] += 1
        status_counts[str(result.status)] += 1

        def _migrate(old_map: dict, bucket: Counter[str]) -> None:
            previous = old_map.get(sample.id)
            if previous is None:
                return
            old_reason = ",".join(str(item) for item in (previous.labels.get("reason_codes") or []))
            old_cat = str(previous.labels.get("category") or previous.labels.get("type") or "")
            new = f"{result.coverage_bucket or result.type}:{result.category or result.status}"
            bucket[f"{old_cat or old_reason}->{new}"] += 1

        _migrate(old_022, migrate_022)
        _migrate(old_024, migrate_024)
        rows.append(
            {
                "id": sample.id,
                "coverage_bucket": result.coverage_bucket,
                "category": result.category,
                "status": result.status,
                "type": result.type,
                "reason": result.reason,
                "reason_codes": list(result.reason_codes),
                "empty_reason_by_run": dict(reasons),
                "pre_filter_language_by_run": dict(result.pre_filter_language_by_run or {}),
                "classify_text_by_run": dict(result.classify_text_by_run or {}),
                "noise_trigger_reasons": list(
                    (result.noise_diagnosis or {}).get("trigger_reasons") or []
                ),
                "implausible_routes": list(result.implausible_routes or []),
                "raw_text_unchanged": all(
                    str(raw_by_run.get(key) or "")
                    == str(
                        (
                            sample.transcripts.get(key) or {}
                        ).get("extra", {}).get("raw_text")
                        if isinstance(sample.transcripts.get(key), dict)
                        else sample.transcripts.get(key) or ""
                    )
                    for key in raw_by_run
                ),
            }
        )

    report = {
        "classify_text_version": CLASSIFY_TEXT_VERSION,
        "classify_text_policy": config.classify_text_policy,
        "classify_text_echo_fingerprint": config.classify_text_echo_fingerprint,
        "rule_version": config.rule_version,
        "prepared": str(prepared_path),
        "sample_count": len(samples),
        "id_conserved": len(ids) == len(set(ids)) == len(samples),
        "empty_routes": empty_routes,
        "empty_reason_counts": dict(empty_reasons),
        "all_configured_routes_empty": all_empty,
        "language_hold": language_hold,
        "speech_rate_quarantine_rows": speech_rate,
        "category_or_unclassified": dict(category_counts),
        "status": dict(status_counts),
        "migration_022": dict(migrate_022.most_common(40)),
        "migration_024": dict(migrate_024.most_common(40)),
        "note": (
            "Empty routes are not confirmed non_speech. This report does not claim "
            "review-budget savings or noise accuracy."
        ),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if jsonl_path is not None:
        with jsonl_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "sample_count",
                    "id_conserved",
                    "empty_routes",
                    "empty_reason_counts",
                    "all_configured_routes_empty",
                    "language_hold",
                    "category_or_unclassified",
                    "status",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
