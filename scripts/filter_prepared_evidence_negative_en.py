#!/usr/bin/env python3
"""从两个 prepared_evidence parquet 里筛出两份证据。

1. 否定词：任意一路转写去掉标点、空白、控制标签后，整段等于下面词表中的某一条。
   输出 prepared_evidence_<batch1>_<batch2>_negative.parquet
2. qwen 单字嗯：任意一路 qwen 转写去掉标点、空白、控制标签后，整段等于「嗯」。
   输出 prepared_evidence_<batch1>_<batch2>_qwen_en.parquet

只筛行，不改写、不删列。两个文件列不一致时按列名取并集，缺的列留空。
同一行可以同时进入两份输出。金标列（列名含 gold，或以 label_ 开头）不参与匹配。

改否定词只改 ``configs/normalization/fenshen_1015_negative_phrases.yaml``。
改 qwen 单字只改 QWEN_SINGLE_CHAR。

示例::

    python scripts/filter_prepared_evidence_negative_en.py ^
        datasets/stage1/derived/prepared_evidence_batchA.parquet ^
        datasets/stage1/derived/prepared_evidence_batchB.parquet ^
        --output-dir tmp/1023
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from audio_engine.core.artifacts import atomic_path
from audio_engine.core.transcript_reconcile import plain_transcript_text

# 否定词。匹配规则是整段全等：归一化后的转写必须和表中某一条完全相同。
# 子串命中不算。重复词会自动去掉。多路同时命中时，统计记最长的那一条。
NEGATIVE_PHRASES_YAML = ROOT / "configs" / "normalization" / "fenshen_1015_negative_phrases.yaml"


def load_negative_phrases(path: Path | None = None) -> tuple[str, ...]:
    """读取否定词 yaml。``phrases`` 为非空字符串列表，重复项忽略。"""
    target = NEGATIVE_PHRASES_YAML if path is None else path
    if not target.is_file():
        raise SystemExit(f"[ERROR] 缺少否定词表: {target}")
    raw = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    phrases = raw.get("phrases") if isinstance(raw, dict) else None
    if not isinstance(phrases, list) or not phrases:
        raise SystemExit(f"[ERROR] 否定词表 phrases 为空: {target}")
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in phrases:
        text = str(item).strip()
        if not text:
            raise SystemExit(f"[ERROR] 否定词表含空词: {target}")
        if text in seen:
            continue
        seen.add(text)
        cleaned.append(text)
    return tuple(cleaned)


NEGATIVE_PHRASES: tuple[str, ...] = load_negative_phrases()

# 输出文件名里的 en 指这个单字。比较前会先做 plain 归一化。
QWEN_SINGLE_CHAR = "嗯"

_SKIP_TEXT_COLUMNS = {
    "gold_text",
    "human_gold_text",
    "candidate_text",
    "label_gold_text",
    "label_human_gold_text",
}


def prepare_phrases(phrases: tuple[str, ...] = NEGATIVE_PHRASES) -> frozenset[str]:
    """去重、去空白。比较用集合，因为准入条件是整段相等。"""
    seen: set[str] = set()
    for phrase in phrases:
        text = str(phrase).strip()
        if text:
            seen.add(text)
    return frozenset(seen)


PHRASE_SET = prepare_phrases()


def batch_name(path: Path) -> str:
    stem = path.stem
    prefix = "prepared_evidence_"
    if stem.startswith(prefix) and len(stem) > len(prefix):
        return stem[len(prefix) :]
    return stem


def is_route_text_column(name: str) -> bool:
    if not name.endswith("_text"):
        return False
    lower = name.lower()
    if lower.startswith("label_") or lower.startswith("quality_"):
        return False
    if "gold" in lower or lower in _SKIP_TEXT_COLUMNS:
        return False
    return True


def model_from_column(name: str) -> str:
    return name[: -len("_text")]


def is_qwen_route(name: str) -> bool:
    return "qwen" in name.lower()


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


def as_plain(value: Any) -> str:
    if _is_missing(value) or not isinstance(value, str):
        return ""
    return plain_transcript_text(value)


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


def route_plains(values: dict[str, Any], route_columns: list[str]) -> dict[str, str]:
    """一路一个纯文本。嵌套 transcripts 优先，扁平 *_text 只补空缺。"""
    routes: dict[str, str] = {}
    mapping = _load_mapping(values.get("transcripts"))
    if mapping:
        for key, entry in mapping.items():
            raw = entry.get("text") if isinstance(entry, dict) else entry
            text = as_plain(raw)
            if text:
                routes[str(key)] = text
    for name in route_columns:
        model = model_from_column(name)
        if routes.get(model):
            continue
        text = as_plain(values.get(name))
        if text:
            routes[model] = text
    return routes


def exact_negative(text: str, phrases: frozenset[str] = PHRASE_SET) -> str | None:
    if text in phrases:
        return text
    return None


def row_flags(routes: dict[str, str]) -> tuple[str | None, bool]:
    """返回 (命中的否定词, qwen 是否为单字嗯)。多路同时命中时记最长的一条。"""
    hit: str | None = None
    qwen_en = False
    for name, text in routes.items():
        if hit is None:
            hit = exact_negative(text)
        elif (found := exact_negative(text)) is not None and len(found) > len(hit):
            hit = found
        if is_qwen_route(name) and text == QWEN_SINGLE_CHAR:
            qwen_en = True
    return hit, qwen_en


def classify_table(
    table: pa.Table,
) -> tuple[pa.Table, pa.Table, Counter[str], int, list[str]]:
    names = table.schema.names
    route_columns = [name for name in names if is_route_text_column(name)]
    needed = list(route_columns)
    if "transcripts" in names and "transcripts" not in needed:
        needed.append("transcripts")
    if not needed:
        raise SystemExit(
            "[ERROR] 找不到转写列。需要 transcripts，或任意一路的 *_text 列。"
            f" 现有列: {names}"
        )
    columns = {name: table.column(name).to_pylist() for name in needed}
    negative_mask: list[bool] = []
    en_mask: list[bool] = []
    hits: Counter[str] = Counter()
    qwen_routes: set[str] = set()
    for index in range(table.num_rows):
        values = {name: columns[name][index] for name in needed}
        routes = route_plains(values, route_columns)
        for name in routes:
            if is_qwen_route(name):
                qwen_routes.add(name)
        phrase, qwen_en = row_flags(routes)
        negative_mask.append(phrase is not None)
        en_mask.append(qwen_en)
        if phrase:
            hits[phrase] += 1
    negative = table.filter(pa.array(negative_mask, type=pa.bool_()))
    qwen_single = table.filter(pa.array(en_mask, type=pa.bool_()))
    described = list(route_columns)
    if "transcripts" in names:
        described.append("transcripts.*.text")
    return negative, qwen_single, hits, len(qwen_routes), described


def concat_tables(tables: list[pa.Table]) -> pa.Table:
    if len(tables) == 1:
        return tables[0]
    try:
        return pa.concat_tables(tables, promote_options="permissive")
    except TypeError:
        return pa.concat_tables(tables, promote=True)


def write_table(table: pa.Table, path: Path, *, force: bool) -> None:
    if path.exists() and not force:
        raise SystemExit(f"[ERROR] 输出已存在（加 --force 覆盖）: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(path) as tmp:
        pq.write_table(table, tmp)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="从两个 prepared_evidence parquet 筛出否定词命中行，以及 qwen 单字嗯",
    )
    parser.add_argument(
        "inputs",
        nargs=2,
        type=Path,
        help="两个 prepared_evidence_<batch>.parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="输出目录。默认写到第一个输入文件所在目录",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="覆盖已存在的输出文件",
    )
    args = parser.parse_args(argv)

    paths = [Path(raw) for raw in args.inputs]
    for path in paths:
        if not path.is_file():
            raise SystemExit(f"[ERROR] 输入 parquet 不存在: {path}")
        if path.suffix.lower() != ".parquet":
            raise SystemExit(f"[ERROR] 输入必须是 .parquet: {path}")

    batches = [batch_name(path) for path in paths]
    output_dir = Path(args.output_dir) if args.output_dir is not None else paths[0].parent
    negative_path = output_dir / f"prepared_evidence_{batches[0]}_{batches[1]}_negative.parquet"
    en_path = output_dir / f"prepared_evidence_{batches[0]}_{batches[1]}_qwen_en.parquet"

    print(f"[INFO] negative_phrases={len(PHRASE_SET)} match=exact qwen_single={QWEN_SINGLE_CHAR!r}")
    print(f"[INFO] negative_out={negative_path}")
    print(f"[INFO] qwen_en_out={en_path}")

    negative_tables: list[pa.Table] = []
    en_tables: list[pa.Table] = []
    total_hits: Counter[str] = Counter()
    saw_qwen = False

    for path, batch in zip(paths, batches, strict=True):
        table = pq.read_table(path)
        negative, qwen_single, hits, qwen_route_count, described = classify_table(table)
        saw_qwen = saw_qwen or qwen_route_count > 0
        total_hits.update(hits)
        negative_tables.append(negative)
        en_tables.append(qwen_single)
        detail = ", ".join(f"{phrase}={count}" for phrase, count in hits.most_common())
        print(
            f"[INFO] {batch} rows={table.num_rows} "
            f"negative={negative.num_rows} qwen_en={qwen_single.num_rows} "
            f"qwen_routes={qwen_route_count} columns={described}"
        )
        if detail:
            print(f"       {detail}")

    if not saw_qwen:
        print("[WARN] 两个文件里都没有名字含 qwen 的转写路，qwen_en 结果会是 0 行")

    negative_merged = concat_tables(negative_tables)
    en_merged = concat_tables(en_tables)
    write_table(negative_merged, negative_path, force=args.force)
    write_table(en_merged, en_path, force=args.force)
    print(
        f"[OK] negative {negative_merged.num_rows} rows, "
        f"{negative_merged.num_columns} columns -> {negative_path}"
    )
    print(
        f"[OK] qwen_en {en_merged.num_rows} rows, "
        f"{en_merged.num_columns} columns -> {en_path}"
    )
    if total_hits:
        summary = ", ".join(f"{phrase}={count}" for phrase, count in total_hits.most_common())
        print(f"[OK] exact_hit {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
