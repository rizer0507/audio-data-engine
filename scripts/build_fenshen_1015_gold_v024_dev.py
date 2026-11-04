#!/usr/bin/env python3
"""把 v024 验证集写成和 gold_fenshen_1015_dev_v030.parquet 相同的评测金标。

字准表不划分 auto_gold / hardcase / non_speech，只统计 overall。
空金标仍记 source_gold_kind=non_speech，计分参考是空串。
qwen_1_text 放预标注：人工包用 candidate_text，旧人工 auto_gold 用 qwen_1_text，
三路一致用补过标点的转写。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audio_engine.core.eval_ready import inspect_eval_manifest  # noqa: E402
from audio_engine.core.manifest import Manifest  # noqa: E402
from audio_engine.core.sample import LineageEntry, Sample  # noqa: E402

TRAIN_REPO = ROOT.parent / "audio-qwen3-asr-train"
SAMPLE_MAP = (
    TRAIN_REPO
    / "data"
    / "training_sets"
    / "exploratory"
    / "fenshen_1015"
    / "train_v024"
    / "sample_map.jsonl"
)
EXPORT_ROOT = (
    ROOT
    / "tmp"
    / "1102"
    / "export_be151575af424ec884a4a9cbbb054acf"
    / "be151575af424ec884a4a9cbbb054acf"
)
HUMAN_PARQUET = ROOT / "tmp" / "1028" / "人工标注" / "auto_gold_drop_identity.parquet"
DERIVED = ROOT / "datasets" / "stage1" / "derived"
OUT = DERIVED / "gold_fenshen_1015_dev_v024.parquet"
XLSX = ROOT / "tmp" / "1102" / "gold_fenshen_1015_dev_v024.xlsx"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _cell(value: Any) -> str:
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


def _load_xlsx_module():
    path = ROOT / "scripts" / "eval_fenshen_1015_xlsx.py"
    spec = importlib.util.spec_from_file_location("eval_fenshen_1015_xlsx", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_dev_rows() -> list[dict[str, Any]]:
    if not SAMPLE_MAP.is_file():
        raise SystemExit(f"[ERROR] 缺少 sample_map: {SAMPLE_MAP}")
    rows: list[dict[str, Any]] = []
    for line in SAMPLE_MAP.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if _cell(row.get("split")) == "dev":
            rows.append(row)
    if not rows:
        raise SystemExit("[ERROR] sample_map 没有 dev")
    return rows


def _candidate_by_id() -> dict[str, str]:
    found: dict[str, str] = {}
    for path in sorted(EXPORT_ROOT.rglob("annotations_*.parquet")):
        frame = pd.read_parquet(path, columns=["sample_id", "candidate_text"])
        for rec in frame.itertuples(index=False):
            sample_id = _cell(rec.sample_id)
            if sample_id:
                found[sample_id] = _cell(rec.candidate_text)
    if HUMAN_PARQUET.is_file():
        frame = pd.read_parquet(HUMAN_PARQUET, columns=["id", "qwen_1_text"])
        for rec in frame.itertuples(index=False):
            sample_id = _cell(rec.id)
            if sample_id and sample_id not in found:
                found[sample_id] = _cell(rec.qwen_1_text)
    return found


def build() -> None:
    dev_rows = _load_dev_rows()
    candidates = _candidate_by_id()
    xlsx_mod = _load_xlsx_module()
    samples: list[Sample] = []
    bucket_counts: dict[str, int] = {}
    for row in sorted(dev_rows, key=lambda item: _cell(item.get("sample_id"))):
        sample_id = _cell(row.get("sample_id"))
        audio_path = _cell(row.get("audio_path"))
        if not sample_id or not audio_path:
            raise SystemExit(f"[ERROR] dev 行缺少 id 或音频: {row}")
        bucket = _cell(row.get("selection_bucket"))
        component = _cell(row.get("source_component"))
        gold = _cell(row.get("gold_text"))
        empty_ref = bucket == "non_speech" or not gold
        if empty_ref and gold:
            raise SystemExit(f"[ERROR] non_speech 仍有金标: {sample_id}")
        if not empty_ref and not gold:
            raise SystemExit(f"[ERROR] speech 金标为空: {sample_id}")
        if component == "auto_gold_three_agree":
            candidate = gold
        elif sample_id in candidates:
            candidate = candidates[sample_id]
        elif bucket == "auto_gold":
            candidate = gold
        else:
            candidate = ""
        duration = row.get("duration")
        try:
            duration_f = float(duration) if duration not in (None, "") else None
        except (TypeError, ValueError):
            duration_f = None
        if duration_f is not None and duration_f != duration_f:
            duration_f = None
        labels: dict[str, Any] = {
            "gold_text": "" if empty_ref else gold,
            "label": "" if empty_ref else gold,
            "decision": "empty" if empty_ref else "accept",
            "annotation_state": "human_accepted",
            "is_human_verified": True,
            "label_tier": "gold",
            "label_source": "human",
            "gold_source": "labeling_platform",
            "source_gold_kind": "non_speech" if empty_ref else "speech",
            "type": "non_speech" if empty_ref else "speech",
            "eval_type_split": False,
            "exclude_from_training": False,
            "eval_split": "dev",
            "eval_origin": "train_v024_dev",
            "export_id": "train_v024",
        }
        samples.append(
            Sample(
                id=sample_id,
                source_path=audio_path,
                sha256=_cell(row.get("audio_sha256")),
                sample_rate=16000,
                channels=1,
                duration=duration_f,
                audio={"resampled_16k": audio_path},
                transcripts={
                    "qwen_1": {
                        "text": candidate,
                        "model": "qwen3-asr",
                        "version": "candidate_text",
                    }
                },
                labels=labels,
                lineage=[
                    LineageEntry(
                        operator="build_fenshen_1015_gold_v024_dev",
                        version="v1",
                        params={"dataset": "train_v024", "eval_type_split": False, "selection_bucket": bucket},
                    )
                ],
            )
        )
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

    if not samples:
        raise SystemExit("[ERROR] 没有可评测样本")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    manifest = Manifest(samples)
    manifest.save(OUT)
    manifest.save(OUT.with_suffix(".jsonl"))

    report = inspect_eval_manifest(OUT, require_formal_gold=True)
    if report.errors:
        raise SystemExit("[ERROR] 正式评测门禁未过:\n" + "\n".join(report.errors))

    gold_rows = xlsx_mod.load_gold_rows(OUT)
    if any(row["type"] for row in gold_rows):
        raise SystemExit("[ERROR] v024 评测金标不应写出分类 type")
    empty_refs = sum(not row["reference"] for row in gold_rows)
    if empty_refs != bucket_counts.get("non_speech", 0):
        raise SystemExit(f"[ERROR] 空参考 {empty_refs} 与 non_speech {bucket_counts.get('non_speech', 0)} 不一致")

    detail = pd.DataFrame(
        [
            {
                "id": row["id"],
                "gold_text": row["label"],
                "candidate_text": row["candidate_text"],
                "wav_path": row["wav_path"],
            }
            for row in gold_rows
        ]
    )
    summary_rows = [
        {"项": "评测条数", "值": len(samples)},
        {"项": "non_speech", "值": bucket_counts.get("non_speech", 0)},
        {"项": "hardcase", "值": bucket_counts.get("hardcase", 0)},
        {"项": "auto_gold", "值": bucket_counts.get("auto_gold", 0)},
        {"项": "分类", "值": "不做分类，字准表只统计 overall"},
    ]
    notes = pd.DataFrame(
        [
            ("金标", str(OUT)),
            ("来源", str(SAMPLE_MAP)),
            ("命令", "bash scripts/run_fenshen_1015_eval.sh lora v024 dev best"),
            ("范围", "train_v024 的 dev。每个类别 round(n*0.2)，id 不重复。"),
            ("分类", "label_eval_type_split=false。不划分 auto_gold / hardcase / non_speech。"),
            (
                "空参考",
                "source_gold_kind=non_speech 的计分参考是空串。这只用于计分，不单列 scope。",
            ),
            ("音频", "audio.resampled_16k，与训练集 sample_map 的 audio_path 相同。"),
            (
                "预标注",
                "1023/1027 标注包用 candidate_text。旧人工 auto_gold 用 qwen_1_text。三路一致用补过标点的转写。",
            ),
        ],
        columns=["字段/口径", "说明"],
    )
    XLSX.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(XLSX, engine="openpyxl") as writer:
        pd.DataFrame(summary_rows).to_excel(writer, index=False, sheet_name="汇总")
        detail.to_excel(writer, index=False, sheet_name="明细")
        notes.to_excel(writer, index=False, sheet_name="说明")

    meta = {
        "created_at": _now_iso(),
        "name": "fenshen_1015_dev_v024",
        "purpose": "eval_source",
        "sample_count": len(samples),
        "bucket_counts": bucket_counts,
        "empty_reference_count": empty_refs,
        "eval_type_split": False,
        "source": {"sample_map": str(SAMPLE_MAP), "split": "dev"},
        "type_policy": "不做分类。字准表只统计 overall。空金标按空串计分。",
        "next": "bash scripts/run_fenshen_1015_eval.sh lora v024 dev best",
    }
    OUT.with_suffix(".meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[OK] samples={len(samples)} buckets={bucket_counts} empty_ref={empty_refs} unclassified")
    print(f"[OK] gold {OUT}")
    print(f"[OK] xlsx {XLSX}")


if __name__ == "__main__":
    build()
