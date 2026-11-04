#!/usr/bin/env python3
"""把三路转写和人工标注取交集，做字面替换后拆成 auto_gold / hardcase / non_speech。

匹配键是三路 parquet 的 ``id`` 和标注 parquet 的 ``sample_id``。只保留两边都有的 id。

字面替换作用在人工 ``gold_text`` 以及 qwen、glm、sensevoice 三路转写上，按子串全部替换。
``gold_kind=non_speech`` 的行不改写，分类保持 ``non_speech``。

其余行先替换，再用 ``plain_transcript_text`` 去掉标点、空白和控制标签后比较：

- ``auto_gold``：三路 plain 文本相同，并且等于人工标注
- ``hardcase``：plain(qwen) 与 plain(人工标注) 不同
- ``other``：qwen 与标注一致，但三路并不一致。单独落盘，避免交集里的行被丢掉

原文字段保留。替换后的文本写在 ``norm_gold_text``、``norm_qwen_1_text``、
``norm_glm_text``、``norm_sensevoice_text``。non_speech 的这四列等于原文。

改替换表只改 ``REPLACEMENTS``。更长的词先替换。

示例::

    python scripts/split_label_asr_buckets.py ^
        --transcripts datasets/stage1/asr/merged_qwen_sensevoice_glm_20261022-label.parquet ^
        --annotations tmp/1022/export_6e4cd687dec8480d8c017a46a18070b7 ^
        --output-dir tmp/1023/label_buckets
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from audio_engine.core.artifacts import atomic_path
from audio_engine.core.transcript_reconcile import plain_transcript_text

# 子串全部替换，包括人工标注。更长的词靠排序先替换，避免「事儿」被拆开。
REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("泥马", "你妈"),
    ("尼玛", "你妈"),
    ("事儿", "事"),
    ("啥", "什么"),
    ("哎", "喂"),
    ("唉", "喂"),
    ("啦", "了"),
    ("嘛", "吗"),
    ("那", "哪"),
    ("地", "的"),
    ("滴", "的"),
    ("呃", "啊"),
    ("昂", "啊"),
    ("哈", "啊"),
    ("噢", "哦"),
    ("喔", "哦"),
    ("恩", "嗯"),
    ("摁", "嗯"),
)

BUCKET_NON_SPEECH = "non_speech"
BUCKET_AUTO_GOLD = "auto_gold"
BUCKET_HARDCASE = "hardcase"
BUCKET_OTHER = "other"
BUCKETS = (BUCKET_AUTO_GOLD, BUCKET_HARDCASE, BUCKET_NON_SPEECH, BUCKET_OTHER)

_ID_CANDIDATES = ("id", "sample_id")
_ANN_ID_CANDIDATES = ("sample_id", "id")
_QWEN_CANDIDATES = ("qwen_1_text", "qwen_text")
_GLM_CANDIDATES = ("glm_text", "glm_1_text")
_SENSE_CANDIDATES = ("sensevoice_text", "sensevoice_1_text")
_GOLD_TEXT_CANDIDATES = ("gold_text", "label_gold_text", "human_gold_text")
_GOLD_KIND_CANDIDATES = ("gold_kind", "label_gold_kind", "label_agg_gold_kind")
_OPTIONAL_ANN_COLUMNS = (
    "status",
    "decision",
    "annotation_type",
    "noise_kind",
    "exclude_from_training",
    "uncertain",
)


def apply_colloquial(value: Any) -> str:
    """把口语近音字按 ``REPLACEMENTS`` 全部替换。空值收成空串。"""
    text = "" if value is None else str(value)
    for src, dst in sorted(REPLACEMENTS, key=lambda item: len(item[0]), reverse=True):
        if src:
            text = text.replace(src, dst)
    return text


def assign_bucket(gold_kind: Any, qwen: Any, glm: Any, sensevoice: Any, human: Any) -> str:
    """non_speech 优先。其余比较的是替换后的 plain 文本。"""
    if str(gold_kind or "").strip() == BUCKET_NON_SPEECH:
        return BUCKET_NON_SPEECH
    qwen_plain = plain_transcript_text(qwen)
    glm_plain = plain_transcript_text(glm)
    sense_plain = plain_transcript_text(sensevoice)
    human_plain = plain_transcript_text(human)
    if qwen_plain == glm_plain == sense_plain == human_plain:
        return BUCKET_AUTO_GOLD
    if qwen_plain != human_plain:
        return BUCKET_HARDCASE
    return BUCKET_OTHER


def find_annotation_parquet(path: Path) -> Path:
    """文件直接用。目录里取唯一的 ``annotations_*.parquet``。"""
    if path.is_file():
        return path
    if not path.is_dir():
        raise ValueError(f"找不到标注路径: {path}")
    matches = sorted(path.rglob("annotations_*.parquet"))
    if len(matches) != 1:
        names = [str(item) for item in matches]
        raise ValueError(f"标注目录里需要恰好一个 annotations_*.parquet，实际 {len(matches)} 个: {names}")
    return matches[0]


def split_buckets(
    transcripts: Path,
    annotations: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """取交集、替换、分类，并写出 classified 与四个 bucket parquet。"""
    asr = pq.read_table(transcripts)
    asr_names = asr.schema.names
    asr_id_col = _require_column(asr_names, _ID_CANDIDATES, f"{transcripts.name} id")
    qwen_col = _require_column(asr_names, _QWEN_CANDIDATES, "qwen 转写")
    glm_col = _require_column(asr_names, _GLM_CANDIDATES, "glm 转写")
    sense_col = _require_column(asr_names, _SENSE_CANDIDATES, "sensevoice 转写")

    ann_path = find_annotation_parquet(annotations)
    ann_schema = pq.read_schema(ann_path).names
    ann_id_col = _require_column(ann_schema, _ANN_ID_CANDIDATES, f"{ann_path.name} id")
    gold_text_col = _require_column(ann_schema, _GOLD_TEXT_CANDIDATES, f"{ann_path.name} 金标文本")
    gold_kind_col = _require_column(ann_schema, _GOLD_KIND_CANDIDATES, f"{ann_path.name} gold_kind")
    optional = [name for name in _OPTIONAL_ANN_COLUMNS if name in ann_schema]
    ann_columns = [ann_id_col, gold_text_col, gold_kind_col, *optional]
    ann_rows = pq.read_table(ann_path, columns=ann_columns).to_pylist()

    gold_by_id: dict[str, dict[str, Any]] = {}
    for row in ann_rows:
        sample_id = _text(row.get(ann_id_col))
        if not sample_id:
            raise ValueError(f"{ann_path.name} 有空 id")
        if sample_id in gold_by_id:
            raise ValueError(f"{ann_path.name} id 重复: {sample_id}")
        gold_by_id[sample_id] = row

    asr_rows = asr.to_pylist()
    seen_asr: set[str] = set()
    keep_index: list[int] = []
    for index, row in enumerate(asr_rows):
        sample_id = _text(row.get(asr_id_col))
        if not sample_id:
            raise ValueError(f"{transcripts.name} 有空 id")
        if sample_id in seen_asr:
            raise ValueError(f"{transcripts.name} id 重复: {sample_id}")
        seen_asr.add(sample_id)
        if sample_id in gold_by_id:
            keep_index.append(index)

    asr_only = len(seen_asr - gold_by_id.keys())
    ann_only = len(gold_by_id.keys() - seen_asr)
    selected = asr.take(pa.array(keep_index, type=pa.int64()))
    selected_rows = [asr_rows[index] for index in keep_index]

    gold_text: list[str] = []
    gold_kind: list[str] = []
    norm_gold: list[str] = []
    norm_qwen: list[str] = []
    norm_glm: list[str] = []
    norm_sense: list[str] = []
    buckets: list[str] = []
    rewritten = 0
    extras: dict[str, list[Any]] = {name: [] for name in optional}

    for row in selected_rows:
        sample_id = _text(row.get(asr_id_col))
        ann = gold_by_id[sample_id]
        human = _text(ann.get(gold_text_col))
        kind = _text(ann.get(gold_kind_col))
        qwen = _text(row.get(qwen_col))
        glm = _text(row.get(glm_col))
        sense = _text(row.get(sense_col))
        if kind == BUCKET_NON_SPEECH:
            qwen_norm, glm_norm, sense_norm, human_norm = qwen, glm, sense, human
        else:
            qwen_norm = apply_colloquial(qwen)
            glm_norm = apply_colloquial(glm)
            sense_norm = apply_colloquial(sense)
            human_norm = apply_colloquial(human)
            if (qwen_norm, glm_norm, sense_norm, human_norm) != (qwen, glm, sense, human):
                rewritten += 1
        gold_text.append(human)
        gold_kind.append(kind)
        norm_gold.append(human_norm)
        norm_qwen.append(qwen_norm)
        norm_glm.append(glm_norm)
        norm_sense.append(sense_norm)
        buckets.append(
            assign_bucket(kind, qwen_norm, glm_norm, sense_norm, human_norm)
        )
        for name in optional:
            extras[name].append(ann.get(name))

    output_columns: dict[str, list[Any]] = {
        "gold_text": gold_text,
        "gold_kind": gold_kind,
        "norm_gold_text": norm_gold,
        "norm_qwen_1_text": norm_qwen,
        "norm_glm_text": norm_glm,
        "norm_sensevoice_text": norm_sense,
        "bucket": buckets,
    }
    for name, values in extras.items():
        column_name = "annotation_status" if name == "status" else name
        output_columns[column_name] = values

    classified = _append_columns(selected, output_columns)
    output_dir.mkdir(parents=True, exist_ok=True)
    classified_path = output_dir / "classified.parquet"
    _write_table(classified, classified_path)

    written: dict[str, str] = {"classified": str(classified_path)}
    counts: dict[str, int] = {name: 0 for name in BUCKETS}
    for bucket in buckets:
        counts[bucket] += 1
    bucket_array = pa.array(buckets)
    for name in BUCKETS:
        subset = classified.filter(pa.compute.equal(bucket_array, name))
        path = output_dir / f"{name}.parquet"
        _write_table(subset, path)
        written[name] = str(path)

    summary = {
        "transcripts": str(transcripts),
        "annotations": str(ann_path),
        "asr_rows": asr.num_rows,
        "annotation_rows": len(ann_rows),
        "intersection": len(keep_index),
        "asr_only": asr_only,
        "annotation_only": ann_only,
        "rewritten_speech_rows": rewritten,
        "buckets": counts,
        "outputs": written,
    }
    summary_path = output_dir / "summary.json"
    with atomic_path(summary_path) as tmp:
        tmp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["summary"] = str(summary_path)
    return summary


def _append_columns(table: pa.Table, columns: dict[str, list[Any]]) -> pa.Table:
    for name, values in columns.items():
        if name in table.schema.names:
            index = table.schema.get_field_index(name)
            table = table.remove_column(index)
        table = table.append_column(name, pa.array(values))
    return table


def _write_table(table: pa.Table, path: Path) -> None:
    with atomic_path(path) as tmp:
        pq.write_table(table, tmp)


def _require_column(names: list[str], candidates: tuple[str, ...], label: str) -> str:
    for candidate in candidates:
        if candidate in names:
            return candidate
    raise ValueError(f"找不到{label}列，候选 {list(candidates)}，实际列 {names}")


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="三路转写与人工标注取交集后拆 auto_gold / hardcase / non_speech")
    parser.add_argument("--transcripts", type=Path, required=True, help="含 qwen、glm、sensevoice 的 parquet")
    parser.add_argument("--annotations", type=Path, required=True, help="标注 parquet，或含 annotations_*.parquet 的目录")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    if not args.transcripts.is_file():
        raise SystemExit(f"找不到三路转写: {args.transcripts}")
    summary = split_buckets(args.transcripts, args.annotations, args.output_dir)
    print(f"intersection {summary['intersection']} asr_only {summary['asr_only']} annotation_only {summary['annotation_only']}")
    print(f"rewritten_speech_rows {summary['rewritten_speech_rows']}")
    for name in BUCKETS:
        print(f"{name} {summary['buckets'][name]} {summary['outputs'][name]}")
    print(f"summary {summary['summary']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
