#!/usr/bin/env python3
"""从 qwen 单字嗯证据 parquet 里去掉「接洽（身份核实）」。

输入是筛选脚本产出的 ``prepared_evidence_*_qwen_en.parquet``，外加一份 xlsx。
按父音频编号对齐：``parquet.label_source_audio_id == xlsx.file_id``。
证据行的 ``id`` 是 VAD 切段后的 ``seg_...``，不能拿来对表格的 ``file_id``。
只删除同时满足下面两点的行：

1. ``label_source_audio_id`` 能在 xlsx 的 ``file_id``（也认 ``file-id``）里找到
2. 该 ``file_id`` 至少有一行 ``AI话术分类标签`` 等于 ``DROP_LABEL``

同一个父音频切出的多条片段会一起删掉。xlsx 里没有的父音频、以及标签不是这一类的，都保留。
不改列、不改单元格。同一个 ``file_id`` 有多行时，只要其中一行是该标签，对应片段就删掉。
平铺列缺失时，回退读 ``labels.source_audio_id``。

改删除标签只改 ``DROP_LABEL``。括号全角、半角都算同一标签。

示例::

    python scripts/drop_qwen_en_identity_verify.py ^
        datasets/stage1/derived/prepared_evidence_batchA_batchB_qwen_en.parquet ^
        --xlsx path\\to\\labels.xlsx ^
        --output-dir tmp/1023
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from audio_engine.core.artifacts import atomic_path

# 要删掉的话术分类。比较前会把半角括号换成全角，并去掉首尾空白。
DROP_LABEL = "接洽（身份核实）"

LABEL_COLUMN = "AI话术分类标签"
FILE_ID_COLUMNS = ("file_id", "file-id", "file id")
# VAD 之后的父音频编号。切段 id 在 id 列，是 seg_...，不对 file_id。
MATCH_COLUMN = "label_source_audio_id"
NESTED_LABELS_COLUMN = "labels"
NESTED_SOURCE_KEY = "source_audio_id"


def normalize_header(name: Any) -> str:
    return str(name).strip().lower().replace("-", "_").replace(" ", "_")


def normalize_label(value: Any) -> str:
    if _is_missing(value):
        return ""
    text = str(value).strip()
    return text.replace("(", "（").replace(")", "）")


def normalize_id(value: Any) -> str:
    """把 parquet / Excel 的 id 收成可比较的字符串。纯数字的 ``123.0`` 收成 ``123``。"""
    if _is_missing(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    text = str(value).strip()
    if len(text) > 2 and text.endswith(".0") and text[:-2].lstrip("-").isdigit():
        return text[:-2]
    return text


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    try:
        if value != value:  # NaN
            return True
    except (TypeError, ValueError):
        return False
    return False


def _find_column(columns: list[Any], candidates: tuple[str, ...], label: str) -> str:
    by_norm = {normalize_header(column): column for column in columns}
    for candidate in candidates:
        found = by_norm.get(normalize_header(candidate))
        if found is not None:
            return str(found)
    shown = ", ".join(str(column) for column in columns)
    raise SystemExit(f"[ERROR] xlsx 缺少{label}列，需要其中之一: {candidates}。当前列: {shown}")


def load_sheet(path: Path, sheet: str | int | None) -> tuple[pd.DataFrame, str]:
    if sheet is not None:
        frame = pd.read_excel(path, sheet_name=sheet, dtype=object)
        return frame, str(sheet)
    book = pd.read_excel(path, sheet_name=None, dtype=object)
    if not book:
        raise SystemExit(f"[ERROR] xlsx 没有任何工作表: {path}")
    for name, frame in book.items():
        headers = {normalize_header(column) for column in frame.columns}
        has_id = any(normalize_header(column) in headers for column in FILE_ID_COLUMNS)
        has_label = normalize_header(LABEL_COLUMN) in headers
        if has_id and has_label:
            return frame, str(name)
    first_name = next(iter(book))
    return book[first_name], str(first_name)


def drop_file_ids(frame: pd.DataFrame, *, label: str = DROP_LABEL) -> tuple[set[str], dict[str, int]]:
    """返回应删除的 file_id，以及给日志用的计数。"""
    file_id_column = _find_column(list(frame.columns), FILE_ID_COLUMNS, "file_id")
    label_column = _find_column(list(frame.columns), (LABEL_COLUMN,), "话术分类")
    target = normalize_label(label)
    if not target:
        raise SystemExit("[ERROR] 删除标签不能为空")

    matched: set[str] = set()
    label_rows = 0
    blank_id_rows = 0
    for file_id, raw_label in zip(frame[file_id_column].tolist(), frame[label_column].tolist(), strict=True):
        if normalize_label(raw_label) != target:
            continue
        label_rows += 1
        text = normalize_id(file_id)
        if not text:
            blank_id_rows += 1
            continue
        matched.add(text)
    stats = {
        "xlsx_rows": int(len(frame)),
        "label_rows": label_rows,
        "blank_id_rows": blank_id_rows,
        "drop_ids": len(matched),
    }
    return matched, stats


def keep_mask(ids: list[Any], drop_ids: set[str]) -> list[bool]:
    return [normalize_id(item) not in drop_ids for item in ids]


def output_path_for(parquet_path: Path, output_dir: Path | None) -> Path:
    directory = output_dir if output_dir is not None else parquet_path.parent
    return directory / f"{parquet_path.stem}_drop_identity.parquet"


def _nested_source_audio_id(value: Any) -> Any:
    if isinstance(value, str):
        raw = value.strip()
        if raw[:1] in "{[":
            try:
                value = json.loads(raw)
            except json.JSONDecodeError:
                return None
    if isinstance(value, dict):
        return value.get(NESTED_SOURCE_KEY)
    return None


def parent_audio_ids(table: pa.Table) -> tuple[list[str], str]:
    """父音频编号。优先平铺列，空值再看 labels.source_audio_id。"""
    names = set(table.column_names)
    has_flat = MATCH_COLUMN in names
    has_nested = NESTED_LABELS_COLUMN in names
    if not has_flat and not has_nested:
        shown = ", ".join(table.column_names)
        raise SystemExit(
            f"[ERROR] parquet 缺少 {MATCH_COLUMN}，也没有 {NESTED_LABELS_COLUMN}。当前列: {shown}"
        )
    flat = table.column(MATCH_COLUMN).to_pylist() if has_flat else [None] * table.num_rows
    nested_raw = table.column(NESTED_LABELS_COLUMN).to_pylist() if has_nested else [None] * table.num_rows
    merged: list[str] = []
    used_nested = 0
    for flat_value, nested_value in zip(flat, nested_raw, strict=True):
        text = normalize_id(flat_value)
        if not text:
            text = normalize_id(_nested_source_audio_id(nested_value))
            if text:
                used_nested += 1
        merged.append(text)
    if has_flat and used_nested == 0:
        source = MATCH_COLUMN
    elif has_flat:
        source = f"{MATCH_COLUMN}+{NESTED_LABELS_COLUMN}.{NESTED_SOURCE_KEY}"
    else:
        source = f"{NESTED_LABELS_COLUMN}.{NESTED_SOURCE_KEY}"
    return merged, source


def filter_table(table: pa.Table, drop_ids: set[str]) -> tuple[pa.Table, dict[str, int]]:
    parent_ids, match_source = parent_audio_ids(table)
    mask = keep_mask(parent_ids, drop_ids)
    present = {item for item in parent_ids if item}
    dropped = len(mask) - sum(mask)
    blank_parent = parent_ids.count("")
    stats = {
        "match_source": match_source,
        "parquet_rows": table.num_rows,
        "dropped": dropped,
        "kept": sum(mask),
        "blank_parent": blank_parent,
        "drop_ids_hit": len(drop_ids & present),
        "drop_ids_miss": len(drop_ids - present),
        "parent_examples": sorted(present)[:3],
    }
    return table.filter(pa.array(mask, type=pa.bool_())), stats


def write_table(table: pa.Table, path: Path, *, force: bool) -> None:
    if path.exists() and not force:
        raise SystemExit(f"[ERROR] 输出已存在（加 --force 覆盖）: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(path) as tmp:
        pq.write_table(table, tmp)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="按 label_source_audio_id 对 file_id，去掉 qwen 单字嗯证据里的接洽（身份核实）",
    )
    parser.add_argument(
        "parquet",
        type=Path,
        help="prepared_evidence_*_qwen_en.parquet",
    )
    parser.add_argument(
        "--xlsx",
        type=Path,
        required=True,
        help="含 file_id 和 AI话术分类标签 的表格",
    )
    parser.add_argument(
        "--sheet",
        default=None,
        help="工作表名或从 0 开始的序号。默认自动找含这两列的表，找不到就用第一张",
    )
    parser.add_argument(
        "--label",
        default=DROP_LABEL,
        help=f"要删除的 AI话术分类标签，默认 {DROP_LABEL}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="输出目录。默认写到输入 parquet 所在目录",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="输出 parquet 路径。指定后忽略 --output-dir 的文件名规则",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="覆盖已存在的输出文件",
    )
    args = parser.parse_args(argv)

    parquet_path = Path(args.parquet)
    xlsx_path = Path(args.xlsx)
    if not parquet_path.is_file():
        raise SystemExit(f"[ERROR] 输入 parquet 不存在: {parquet_path}")
    if parquet_path.suffix.lower() != ".parquet":
        raise SystemExit(f"[ERROR] 输入必须是 .parquet: {parquet_path}")
    if not xlsx_path.is_file():
        raise SystemExit(f"[ERROR] 输入 xlsx 不存在: {xlsx_path}")
    if xlsx_path.suffix.lower() not in {".xlsx", ".xlsm"}:
        raise SystemExit(f"[ERROR] 输入必须是 .xlsx: {xlsx_path}")

    sheet: str | int | None = args.sheet
    if isinstance(sheet, str) and sheet.isdigit():
        sheet = int(sheet)

    frame, sheet_name = load_sheet(xlsx_path, sheet)
    drop_ids, xlsx_stats = drop_file_ids(frame, label=args.label)
    table = pq.read_table(parquet_path)
    filtered, parquet_stats = filter_table(table, drop_ids)

    if args.output is not None:
        out_path = Path(args.output)
    else:
        out_path = output_path_for(parquet_path, Path(args.output_dir) if args.output_dir else None)

    print(
        f"[INFO] sheet={sheet_name} label={normalize_label(args.label)!r} "
        f"xlsx_rows={xlsx_stats['xlsx_rows']} label_rows={xlsx_stats['label_rows']} "
        f"drop_ids={xlsx_stats['drop_ids']} blank_id_rows={xlsx_stats['blank_id_rows']}"
    )
    print(
        f"[INFO] match={parquet_stats['match_source']} "
        f"parquet_rows={parquet_stats['parquet_rows']} "
        f"dropped={parquet_stats['dropped']} kept={parquet_stats['kept']} "
        f"blank_parent={parquet_stats['blank_parent']} "
        f"drop_ids_hit={parquet_stats['drop_ids_hit']} "
        f"drop_ids_miss={parquet_stats['drop_ids_miss']}"
    )
    if parquet_stats["drop_ids_hit"] == 0 and xlsx_stats["drop_ids"]:
        xlsx_examples = sorted(drop_ids)[:3]
        print(
            f"[WARN] 父音频编号没有对上 file_id。"
            f"parquet 示例={parquet_stats['parent_examples']} xlsx 示例={xlsx_examples}"
        )
    print(f"[INFO] out={out_path}")
    write_table(filtered, out_path, force=args.force)
    print(f"[OK] {filtered.num_rows} rows, {filtered.num_columns} columns -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
