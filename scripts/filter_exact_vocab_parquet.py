#!/usr/bin/env python3
"""从两个（或多个）parquet 中筛出转写整段精准命中拒答词表的行，原样汇总成一个 parquet。

词表（来源列表里「不用」重复一次，匹配时只保留一次）::

    没有、暂时不用、不需要谢谢、不可以啊、不用、我不需要、不要、不需要

精准匹配：去掉首尾空白后，整段文本必须等于其中某一个词。
「我不用」「不用。」「暂时不用谢谢」都不会命中；「暂时不用」只命中自身，不会被更短的「不用」吃掉。
只筛选行，不改写、不删列。两个文件列不一致时按列名取并集，缺的列留空。

默认文本来源：
  - 列名含 qwen 且以 ``_text`` 结尾（如 ``qwen_text``、``qwen3-asr_text``）
  - 同时检查 ``transcripts`` 里键名含 qwen 的 ``text``
  - 没有上述列时，用 ``text``；再没有则用整个 ``transcripts``
  - 指定 ``--text-column`` 后只看这些列（``transcripts`` 表示嵌套转写）

示例::

    python scripts/filter_exact_vocab_parquet.py ^
        a.parquet b.parquet ^
        -o tmp/exact_vocab_hits.parquet
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from audio_engine.core.artifacts import atomic_path

# 用户给出的列表。第二个「不用」与前面重复，集合里只出现一次。
EXACT_PHRASES: tuple[str, ...] = (
    "没有",
    "暂时不用",
    "不需要谢谢",
    "不可以啊",
    "不用",
    "我不需要",
    "不要",
    "不需要",
)


def compile_exact_pattern(phrases: tuple[str, ...] = EXACT_PHRASES) -> re.Pattern[str]:
    """整段精准匹配。长词在前，避免未加锚点时短词抢先。"""
    ordered: list[str] = []
    seen: set[str] = set()
    for phrase in sorted(phrases, key=len, reverse=True):
        if phrase not in seen:
            seen.add(phrase)
            ordered.append(phrase)
    body = "|".join(re.escape(phrase) for phrase in ordered)
    return re.compile(rf"^(?:{body})$")


EXACT_RE = compile_exact_pattern()


def exact_phrase(value: Any) -> str | None:
    """首尾空白去掉后，整段等于词表中的某一个词则返回该词。"""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if not isinstance(value, str):
        return None
    text = value.strip()
    if text and EXACT_RE.fullmatch(text):
        return text
    return None


def _load_mapping(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return None
    raw = value.strip()
    if not raw or raw[0] not in "{[":
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def transcript_texts(value: Any, *, qwen_only: bool) -> list[str]:
    """从 transcripts 对象或 JSON 字符串取出待匹配文本。"""
    mapping = _load_mapping(value)
    if not mapping:
        return []
    texts: list[str] = []
    for key, entry in mapping.items():
        name = str(key)
        if qwen_only and "qwen" not in name.lower():
            continue
        if isinstance(entry, dict):
            text = entry.get("text")
        else:
            text = entry
        if isinstance(text, str):
            texts.append(text)
    return texts


def qwen_text_columns(names: list[str]) -> list[str]:
    return [
        name
        for name in names
        if "qwen" in name.lower() and name.lower().endswith("_text")
    ]


def resolve_sources(
    names: list[str],
    explicit: list[str] | None,
) -> tuple[list[str], bool, bool]:
    """返回 (扁平列, 是否扫描 transcripts, transcripts 是否只看 qwen 键)。"""
    if explicit:
        missing = [name for name in explicit if name not in names]
        if missing:
            raise SystemExit(
                f"[ERROR] 找不到列 {missing}。现有列: {names}"
            )
        flat = [name for name in explicit if name != "transcripts"]
        scan = "transcripts" in explicit
        qwen_only = False
        if scan:
            qwen_only = any("qwen" in name.lower() for name in explicit if name != "transcripts")
        return flat, scan, qwen_only

    qwen_cols = qwen_text_columns(names)
    if qwen_cols:
        return qwen_cols, "transcripts" in names, True
    if "text" in names:
        return ["text"], False, False
    for fallback in ("asr_text", "transcript", "hypothesis", "sentence"):
        if fallback in names:
            return [fallback], False, False
    if "transcripts" in names:
        return [], True, False
    raise SystemExit(
        "[ERROR] 无法判断文本列。请用 --text-column 指定。"
        f" 现有列: {names}"
    )


def match_row(
    values: dict[str, Any],
    flat_columns: list[str],
    *,
    scan_transcripts: bool,
    qwen_only: bool,
) -> str | None:
    for name in flat_columns:
        hit = exact_phrase(values.get(name))
        if hit:
            return hit
    if scan_transcripts:
        for text in transcript_texts(values.get("transcripts"), qwen_only=qwen_only):
            hit = exact_phrase(text)
            if hit:
                return hit
    return None


def filter_table(
    table: pa.Table,
    *,
    explicit_columns: list[str] | None,
) -> tuple[pa.Table, Counter[str], list[str], bool]:
    names = table.schema.names
    flat_columns, scan_transcripts, qwen_only = resolve_sources(names, explicit_columns)
    if scan_transcripts and qwen_only is False and not explicit_columns:
        # 没有 qwen 扁平列、只能看 transcripts 时：若存在 qwen 键则只匹配这些键。
        qwen_only = _transcripts_have_qwen(table)

    needed = list(flat_columns)
    if scan_transcripts and "transcripts" not in needed:
        needed.append("transcripts")
    columns = {name: table.column(name).to_pylist() for name in needed}
    hits: Counter[str] = Counter()
    mask: list[bool] = []
    row_count = table.num_rows
    for index in range(row_count):
        values = {name: columns[name][index] for name in needed}
        phrase = match_row(
            values,
            flat_columns,
            scan_transcripts=scan_transcripts,
            qwen_only=qwen_only,
        )
        mask.append(phrase is not None)
        if phrase:
            hits[phrase] += 1
    filtered = table.filter(pa.array(mask, type=pa.bool_()))
    described = list(flat_columns)
    if scan_transcripts:
        scope = "transcripts.qwen*.text" if qwen_only else "transcripts.*.text"
        described.append(scope)
    return filtered, hits, described, qwen_only


def _transcripts_have_qwen(table: pa.Table) -> bool:
    if "transcripts" not in table.schema.names:
        return False
    for value in table.column("transcripts").to_pylist():
        mapping = _load_mapping(value)
        if not mapping:
            continue
        if any("qwen" in str(key).lower() for key in mapping):
            return True
    return False


def concat_tables(tables: list[pa.Table]) -> pa.Table:
    if len(tables) == 1:
        return tables[0]
    try:
        return pa.concat_tables(tables, promote_options="permissive")
    except TypeError:
        return pa.concat_tables(tables, promote=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="按拒答词表整段精准匹配，从多个 parquet 筛行并原样汇总",
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="输入 parquet，至少两个",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        type=Path,
        help="汇总输出 parquet",
    )
    parser.add_argument(
        "--text-column",
        action="append",
        default=[],
        metavar="COLUMN",
        help="参与匹配的列，可重复。写 transcripts 表示扫描嵌套转写。默认自动选择",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="覆盖已存在的输出文件",
    )
    args = parser.parse_args(argv)

    paths = [Path(raw) for raw in args.inputs]
    if len(paths) < 2:
        raise SystemExit("[ERROR] 至少需要两个输入 parquet")
    output = Path(args.output)
    if output.suffix.lower() != ".parquet":
        raise SystemExit(f"[ERROR] --output 必须是 .parquet: {output}")
    if output.exists() and not args.force:
        raise SystemExit(f"[ERROR] 输出已存在（加 --force 覆盖）: {output}")

    explicit = list(args.text_column) or None
    filtered_tables: list[pa.Table] = []
    total_hits: Counter[str] = Counter()
    print(f"[INFO] exact_pattern={EXACT_RE.pattern}")
    print(f"[INFO] phrases={list(dict.fromkeys(EXACT_PHRASES))}")

    for path in paths:
        if not path.is_file():
            raise SystemExit(f"[ERROR] 输入 parquet 不存在: {path}")
        table = pq.read_table(path)
        filtered, hits, described, _qwen_only = filter_table(
            table,
            explicit_columns=explicit,
        )
        total_hits.update(hits)
        filtered_tables.append(filtered)
        detail = ", ".join(f"{phrase}={count}" for phrase, count in sorted(hits.items()))
        print(
            f"[INFO] {path} rows={table.num_rows} hits={filtered.num_rows} "
            f"columns={described}"
        )
        if detail:
            print(f"       {detail}")

    merged = concat_tables(filtered_tables)
    input_columns = []
    for table in filtered_tables:
        for name in table.schema.names:
            if name not in input_columns:
                input_columns.append(name)
    missing = [name for name in input_columns if name not in merged.schema.names]
    if missing:
        raise SystemExit(f"[ERROR] 汇总结果缺少原字段: {missing}")

    with atomic_path(output) as tmp:
        pq.write_table(merged, tmp)
    summary = ", ".join(
        f"{phrase}={total_hits.get(phrase, 0)}" for phrase in dict.fromkeys(EXACT_PHRASES)
    )
    print(f"[OK] wrote {merged.num_rows} rows, {merged.num_columns} columns → {output}")
    print(f"[OK] {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
