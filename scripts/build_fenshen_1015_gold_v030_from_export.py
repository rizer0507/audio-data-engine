#!/usr/bin/env python3
"""把标注导出写成 run_asr_eval.sh 要的 dev 金标。

这份评测集不做 auto_gold / hardcase / non_speech 分桶。
label_eval_type_split=false，字准表只统计 overall。
空金标仍记 source_gold_kind=non_speech，计分参考是空串。

没有金标、且 exclude_from_training 的 ambiguous_target 不能当正式评测参考，不写入金标。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audio_engine.core.eval_ready import inspect_eval_manifest  # noqa: E402
from audio_engine.core.manifest import Manifest  # noqa: E402
from audio_engine.core.sample import LineageEntry, Sample  # noqa: E402

EXPORT_ID = "106fd70d89754cbf898d1acb2c605745"
EXPORT_DIR = (
    ROOT
    / "tmp"
    / "1030"
    / f"export_{EXPORT_ID}"
    / EXPORT_ID
    / "7df2b7d9f30a4a5bbbee50d6c56a9a74"
)
DERIVED = ROOT / "datasets" / "stage1" / "derived"
OUT = DERIVED / "gold_fenshen_1015_dev_v030.parquet"
XLSX = ROOT / "tmp" / "1030" / "gold_fenshen_1015_dev_v030.xlsx"


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


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    text = _cell(value)
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _load_xlsx_module():
    path = ROOT / "scripts" / "eval_asr_xlsx.py"
    spec = importlib.util.spec_from_file_location("eval_asr_xlsx", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return _cell(value).lower() in {"1", "true", "yes", "y"}


def build() -> None:
    ann_path = EXPORT_DIR / "annotations_1021.parquet"
    ing_path = EXPORT_DIR / "ingested_20261021_with_gold.parquet"
    if not ann_path.is_file() or not ing_path.is_file():
        raise SystemExit(f"[ERROR] 缺少导出: {EXPORT_DIR}")

    annotations = { _cell(row.get("sample_id")): row for row in pq.read_table(ann_path).to_pylist() }
    ingested = { _cell(row.get("id")): row for row in pq.read_table(ing_path).to_pylist() }
    annotations.pop("", None)
    ingested.pop("", None)
    if set(annotations) != set(ingested):
        missing = sorted(set(annotations) ^ set(ingested))[:8]
        raise SystemExit(f"[ERROR] 标注和音频 id 不一致，例如 {missing}")

    xlsx_mod = _load_xlsx_module()
    samples: list[Sample] = []
    excluded: list[dict[str, Any]] = []
    for sample_id in sorted(annotations):
        ann = annotations[sample_id]
        ing = ingested[sample_id]
        gold = _cell(ann.get("gold_text"))
        candidate = _cell(ann.get("candidate_text"))
        gold_kind = _cell(ann.get("gold_kind"))
        decision = _cell(ann.get("decision"))
        excluded_flag = _bool(ann.get("exclude_from_training"))
        scoreable = (gold_kind == "speech" and decision == "accept" and bool(gold)) or (
            gold_kind == "non_speech" and decision == "empty"
        )
        if not scoreable or excluded_flag:
            excluded.append(
                {
                    "id": sample_id,
                    "candidate_text": candidate,
                    "gold_text": gold,
                    "reason": "无金标或已排除训练，不能当正式评测参考",
                }
            )
            continue

        audio = _as_dict(ing.get("audio"))
        wav = _cell(audio.get("resampled_16k") or ann.get("wav_path"))
        if not wav:
            raise SystemExit(f"[ERROR] 缺少 resampled_16k: {sample_id}")
        empty_ref = gold_kind == "non_speech" or decision == "empty"
        duration = ing.get("duration")
        try:
            duration_f = float(duration) if duration not in (None, "") else None
        except (TypeError, ValueError):
            duration_f = None
        if duration_f is not None and duration_f != duration_f:
            duration_f = None
        rate = ing.get("sample_rate")
        channels = ing.get("channels")
        labels: dict[str, Any] = {
            "gold_text": "" if empty_ref else gold,
            "label": "" if empty_ref else gold,
            "decision": "empty" if empty_ref else decision,
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
            "eval_origin": f"export_{EXPORT_ID}",
            "export_id": EXPORT_ID,
        }
        samples.append(
            Sample(
                id=sample_id,
                source_path=wav,
                sha256=_cell(ing.get("sha256")),
                sample_rate=int(rate) if isinstance(rate, int) and rate > 0 else 16000,
                channels=int(channels) if isinstance(channels, int) and channels > 0 else 1,
                duration=duration_f,
                audio={"resampled_16k": wav},
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
                        operator="build_fenshen_1015_gold_v030_from_export",
                        version="v1",
                        params={"export_id": EXPORT_ID, "eval_type_split": False},
                    )
                ],
            )
        )

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
        raise SystemExit("[ERROR] v030 不应写出分类 type")

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
        {"项": "未纳入", "值": len(excluded)},
        {"项": "导出总条数", "值": len(annotations)},
        {"项": "分类", "值": "不做分类，字准表只统计 overall"},
    ]
    notes = pd.DataFrame(
        [
            ("金标", str(OUT)),
            ("来源", str(EXPORT_DIR)),
            ("命令", "bash scripts/run_asr_eval.sh fenshen lora v030 dev best"),
            ("分类", "label_eval_type_split=false。不划分 auto_gold / hardcase / non_speech。"),
            (
                "空参考",
                "source_gold_kind=non_speech 的计分参考是空串。这只用于计分，不单列 scope。",
            ),
            (
                "未纳入",
                f"{len(excluded)} 条没有金标且已排除训练，正式评测注册会拒绝空金标，故不写入 parquet。",
            ),
            ("音频", "audio.resampled_16k，与导出 ingested 相同的服务器路径。"),
        ],
        columns=["字段/口径", "说明"],
    )
    XLSX.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(XLSX, engine="openpyxl") as writer:
        pd.DataFrame(summary_rows).to_excel(writer, index=False, sheet_name="汇总")
        detail.to_excel(writer, index=False, sheet_name="明细")
        pd.DataFrame(excluded).to_excel(writer, index=False, sheet_name="未纳入")
        notes.to_excel(writer, index=False, sheet_name="说明")

    meta = {
        "created_at": _now_iso(),
        "name": "fenshen_1015_dev_v030",
        "purpose": "eval_source",
        "sample_count": len(samples),
        "excluded_count": len(excluded),
        "eval_type_split": False,
        "source": {"export_id": EXPORT_ID, "export_dir": str(EXPORT_DIR)},
        "type_policy": "不做分类。字准表只统计 overall。空金标按空串计分。",
        "next": "bash scripts/run_asr_eval.sh fenshen lora v030 dev best",
    }
    OUT.with_suffix(".meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[OK] samples={len(samples)} excluded={len(excluded)} unclassified")
    print(f"[OK] gold {OUT}")
    print(f"[OK] xlsx {XLSX}")


if __name__ == "__main__":
    build()
