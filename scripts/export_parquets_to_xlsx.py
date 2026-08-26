#!/usr/bin/env python3
"""合并多个 parquet，导出为 xlsx。

规则：
  - 默认合并全部输入行；单文件最多 ``--max-rows``（默认 80 万）行
  - 超过上限时拆成 ``stem-part-001.xlsx``、``stem-part-002.xlsx`` …
  - 可用 ``--truncate`` 只保留前 N 行并写一个 xlsx

Example:
  python scripts/export_parquets_to_xlsx.py \\
    --input datasets/manifests/qwen_asr_mt3000.parquet \\
    --input datasets/manifests/sensevoice_asr_mt3000.parquet \\
    --output datasets/exports/asr_mt3000.xlsx

  python scripts/export_parquets_to_xlsx.py \\
    datasets/manifests/qwen_asr_*.parquet \\
    -o datasets/exports/qwen_all.xlsx \\
    --models qwen \\
    --dedupe-id
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

DEFAULT_MAX_ROWS = 800_000
# Excel 理论上限约 1_048_576；默认 80 万留余量，避免打开卡死。
EXCEL_HARD_LIMIT = 1_048_576


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if value != value:  # NaN
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if text.lower() in {"nan", "none"}:
        return ""
    return text


def _transcript_text(transcripts: Any, model: str) -> str:
    if not isinstance(transcripts, dict):
        return ""
    entry = transcripts.get(model, {})
    if isinstance(entry, dict):
        return _as_text(entry.get("text"))
    return _as_text(entry)


def _jsonable(value: Any) -> Any:
    if value is None:
        return None
    try:
        if value != value:  # NaN
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        try:
            return json.dumps(value, ensure_ascii=False)
        except TypeError:
            return str(value)
    if isinstance(value, (list, tuple)):
        try:
            return json.dumps(list(value), ensure_ascii=False)
        except TypeError:
            return str(value)
    return str(value)


def flatten_asr_row(record: dict[str, Any], models: list[str]) -> dict[str, Any]:
    """Keep common columns + each model's text (from nested transcripts or flat cols)."""
    transcripts = record.get("transcripts") or {}
    if isinstance(transcripts, str):
        try:
            transcripts = json.loads(transcripts)
        except json.JSONDecodeError:
            transcripts = {}

    row: dict[str, Any] = {
        "id": record.get("id"),
        "source_path": record.get("source_path"),
        "duration": record.get("duration"),
        "sample_rate": record.get("sample_rate"),
        "channels": record.get("channels"),
    }
    for model in models:
        nested = _transcript_text(transcripts, model)
        flat = _as_text(record.get(f"{model}_text"))
        row[f"{model}_text"] = nested or flat

    quality = record.get("quality")
    if isinstance(quality, dict) and quality:
        for key, value in quality.items():
            row[f"quality_{key}"] = _jsonable(value)
    elif isinstance(quality, str) and quality.strip():
        row["quality_json"] = quality

    labels = record.get("labels")
    if isinstance(labels, dict) and labels:
        for key, value in labels.items():
            row[f"label_{key}"] = _jsonable(value)

    return row


def flatten_generic_row(record: dict[str, Any]) -> dict[str, Any]:
    """Serialize nested values so pandas can write Excel."""
    return {key: _jsonable(value) for key, value in record.items()}


def load_parquet(path: Path) -> list[dict[str, Any]]:
    import pandas as pd

    frame = pd.read_parquet(path)
    return frame.to_dict(orient="records")


def expand_inputs(raw_paths: list[str]) -> list[Path]:
    """Expand globs and keep order; drop duplicates."""
    seen: set[Path] = set()
    out: list[Path] = []
    for raw in raw_paths:
        path = Path(raw)
        matches = sorted(path.parent.glob(path.name)) if any(ch in raw for ch in "*?[]") else [path]
        if not matches:
            raise FileNotFoundError(f"no files matched: {raw}")
        for item in matches:
            resolved = item.resolve()
            if resolved in seen:
                continue
            if not item.is_file():
                raise FileNotFoundError(f"not a file: {item}")
            if item.suffix.lower() != ".parquet":
                raise ValueError(f"only .parquet supported: {item}")
            seen.add(resolved)
            out.append(item)
    return out


def write_xlsx(rows: list[dict[str, Any]], path: Path) -> None:
    import pandas as pd

    if len(rows) > EXCEL_HARD_LIMIT:
        raise ValueError(
            f"refusing to write {len(rows)} rows (Excel hard limit {EXCEL_HARD_LIMIT})"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows).to_excel(tmp, index=False, engine="openpyxl")
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="合并多个 parquet 导出 xlsx（单文件默认最多 80 万行）"
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help="输入 parquet（支持通配符）；也可用重复的 --input",
    )
    parser.add_argument(
        "--input",
        "-i",
        action="append",
        default=[],
        dest="input_flags",
        help="输入 parquet（可重复）",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="输出 xlsx；超限时写 stem-part-NNN.xlsx",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=DEFAULT_MAX_ROWS,
        help=f"单个 xlsx 最大行数（默认 {DEFAULT_MAX_ROWS}）",
    )
    parser.add_argument(
        "--truncate",
        action="store_true",
        help="超过 max-rows 时只保留前 N 行，仍写一个文件",
    )
    parser.add_argument(
        "--models",
        default="",
        help="逗号分隔模型名；指定后按 ASR manifest 展平为 id/source_path/{model}_text",
    )
    parser.add_argument(
        "--dedupe-id",
        action="store_true",
        help="按 id 去重（后出现的覆盖先出现的）",
    )
    parser.add_argument(
        "--source-col",
        default="_source_parquet",
        help="写入来源文件名的列名；空字符串表示不写",
    )
    args = parser.parse_args()

    raw_inputs = list(args.inputs) + list(args.input_flags)
    if not raw_inputs:
        print("[ERROR] 请至少指定一个 parquet（位置参数或 --input）", file=sys.stderr)
        return 1

    try:
        paths = expand_inputs(raw_inputs)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    max_rows = max(1, min(int(args.max_rows), EXCEL_HARD_LIMIT))
    models = [item.strip() for item in str(args.models).split(",") if item.strip()]

    print(f"[INFO] inputs={len(paths)} max_rows={max_rows} truncate={args.truncate}")
    rows: list[dict[str, Any]] = []
    for path in paths:
        print(f"[INFO] loading {path}")
        records = load_parquet(path)
        print(f"[INFO]   rows={len(records):,}")
        for record in records:
            if models:
                flat = flatten_asr_row(record, models)
            else:
                flat = flatten_generic_row(record)
            if args.source_col:
                flat[args.source_col] = path.name
            rows.append(flat)

    if args.dedupe_id:
        by_id: dict[str, dict[str, Any]] = {}
        missing = 0
        for row in rows:
            key = _as_text(row.get("id"))
            if not key:
                missing += 1
                continue
            by_id[key] = row
        print(
            f"[INFO] dedupe-id: {len(rows):,} → {len(by_id):,} "
            f"(dropped empty id={missing:,})"
        )
        rows = list(by_id.values())

    total = len(rows)
    print(f"[INFO] merged rows={total:,}")

    if total == 0:
        print("[ERROR] 无数据可写", file=sys.stderr)
        return 1

    if total <= max_rows:
        write_xlsx(rows, args.output)
        print(f"[OK] wrote {total:,} rows → {args.output}")
        return 0

    if args.truncate:
        kept = rows[:max_rows]
        write_xlsx(kept, args.output)
        print(
            f"[OK] truncated {total:,} → {len(kept):,} rows → {args.output} "
            f"(max_rows={max_rows})"
        )
        return 0

    stem = args.output.with_suffix("")
    parts = 0
    for start in range(0, total, max_rows):
        parts += 1
        chunk = rows[start : start + max_rows]
        out = Path(f"{stem}-part-{parts:03d}.xlsx")
        write_xlsx(chunk, out)
        print(f"[OK] part {parts}: {len(chunk):,} rows → {out}")
    print(f"[OK] total {total:,} rows in {parts} xlsx files (max_rows={max_rows})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
