#!/usr/bin/env python3
"""Export multi-ASR aggregate parquet/jsonl to Excel.

Rules:
  - <= 500_000 rows → one xlsx
  - > 500_000 rows → shard into part-001.xlsx, part-002.xlsx, ...

Example:
  python scripts/export_multi_asr_xlsx.py \\
    --manifest datasets/manifests/multi_asr_aggregate_source_A.parquet \\
    --output datasets/exports/multi_asr_aggregate_source_A.xlsx
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

DEFAULT_MAX_ROWS = 500_000


def _nonempty_text(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if value != value:  # NaN
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text or None


def _transcript_text(transcripts: Any, model: str) -> str:
    if not isinstance(transcripts, dict):
        return ""
    entry = transcripts.get(model, {})
    if isinstance(entry, dict):
        return str(entry.get("text") or "")
    return str(entry or "")


def _parse_quality(record: dict[str, Any]) -> dict[str, Any]:
    quality = record.get("quality") or {}
    if isinstance(quality, str):
        try:
            quality = json.loads(quality)
        except json.JSONDecodeError:
            quality = {}
    if not isinstance(quality, dict):
        quality = {}
    return quality


def _metric(record: dict[str, Any], quality: dict[str, Any], key: str) -> Any:
    for candidate in (f"quality_{key}", key):
        if candidate not in record:
            continue
        value = record[candidate]
        if value is None:
            continue
        try:
            if value != value:  # NaN
                continue
        except (TypeError, ValueError):
            pass
        return value
    return quality.get(key)


def flatten_row(record: dict[str, Any], models: list[str], baseline: str) -> dict[str, Any]:
    quality = _parse_quality(record)
    transcripts = record.get("transcripts") or {}
    if isinstance(transcripts, str):
        try:
            transcripts = json.loads(transcripts)
        except json.JSONDecodeError:
            transcripts = {}

    resolved_baseline = (
        _metric(record, quality, "asr_edit_baseline")
        or baseline
    )
    row: dict[str, Any] = {
        "id": record.get("id"),
        "source_path": record.get("source_path"),
        "duration": record.get("duration"),
        "sample_rate": record.get("sample_rate"),
        "channels": record.get("channels"),
        "baseline_model": resolved_baseline,
    }
    for model in models:
        text_col = f"{model}_text"
        # Prefer nested transcripts; flat parquet columns may be NaN even when nested text exists.
        nested = _nonempty_text(_transcript_text(transcripts, model))
        flat = _nonempty_text(record.get(text_col))
        row[text_col] = nested or flat or ""
        if model == resolved_baseline:
            continue
        for metric, label in (
            ("total", "total"),
            ("错字", "错字"),
            ("少字", "少字"),
            ("多字", "多字"),
            ("cer", "cer"),
        ):
            row[f"vs_{resolved_baseline}_{model}_{label}"] = _metric(
                record, quality, f"asr_edit_{model}_{metric}"
            )

    payload_raw = _metric(record, quality, "asr_edit_json")
    if isinstance(payload_raw, str) and payload_raw:
        try:
            payload = json.loads(payload_raw)
        except json.JSONDecodeError:
            payload = {}
        if isinstance(payload, dict):
            row["baseline_model"] = payload.get("baseline", resolved_baseline)
            for model, ops in (payload.get("models") or {}).items():
                if not isinstance(ops, dict):
                    continue
                for metric, label in (
                    ("total", "total"),
                    ("错字", "错字"),
                    ("少字", "少字"),
                    ("多字", "多字"),
                    ("cer", "cer"),
                ):
                    col = f"vs_{row['baseline_model']}_{model}_{label}"
                    if row.get(col) is None:
                        row[col] = ops.get(metric)

    status = record.get("status") or {}
    if isinstance(status, str):
        row["status_json"] = status
    elif isinstance(status, dict):
        row["status_json"] = json.dumps(status, ensure_ascii=False)
    errors = record.get("errors") or {}
    if isinstance(errors, str) and errors not in ("", "{}", "null"):
        row["errors_json"] = errors
    elif isinstance(errors, dict) and errors:
        row["errors_json"] = json.dumps(errors, ensure_ascii=False)
    return row


def load_records(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        import pandas as pd

        frame = pd.read_parquet(path)
        return frame.to_dict(orient="records")
    if suffix == ".jsonl":
        records: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no} invalid JSON: {exc}") from exc
        return records
    raise ValueError(f"Unsupported manifest format: {path.suffix} (use .parquet or .jsonl)")


def write_xlsx(rows: list[dict[str, Any]], path: Path) -> None:
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows).to_excel(tmp, index=False)
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export multi-ASR aggregate manifest to xlsx")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("datasets/manifests/multi_asr_aggregate_source_A.parquet"),
        help="Input parquet or jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/exports/multi_asr_aggregate_source_A.xlsx"),
        help="Output xlsx path (sharded outputs use stem-part-NNN.xlsx)",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=DEFAULT_MAX_ROWS,
        help=f"Shard threshold (default {DEFAULT_MAX_ROWS})",
    )
    parser.add_argument(
        "--models",
        default="qwen,sensevoice",
        help="Comma-separated transcript model keys to export",
    )
    parser.add_argument(
        "--baseline",
        default="qwen",
        help="Baseline model key used in column naming",
    )
    args = parser.parse_args()

    if not args.manifest.is_file():
        print(f"[ERROR] manifest not found: {args.manifest}", file=sys.stderr)
        return 1

    models = [item.strip() for item in str(args.models).split(",") if item.strip()]
    print(f"[INFO] loading {args.manifest}")
    raw = load_records(args.manifest)
    print(f"[INFO] rows={len(raw)} models={models} baseline={args.baseline}")
    flat = [flatten_row(record, models, args.baseline) for record in raw]

    max_rows = max(1, int(args.max_rows))
    if len(flat) <= max_rows:
        write_xlsx(flat, args.output)
        print(f"[OK] wrote {len(flat)} rows → {args.output}")
        return 0

    stem = args.output.with_suffix("")
    parts = 0
    for start in range(0, len(flat), max_rows):
        parts += 1
        chunk = flat[start : start + max_rows]
        out = Path(f"{stem}-part-{parts:03d}.xlsx")
        write_xlsx(chunk, out)
        print(f"[OK] part {parts}: {len(chunk)} rows → {out}")
    print(f"[OK] total {len(flat)} rows in {parts} xlsx shards (max_rows={max_rows})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
