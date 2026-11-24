#!/usr/bin/env python3
"""把 1102 标注导出写成 dev 金标 v032。

只保留有文本的 speech，以及环境噪声里的 background_suspected。
human_noise 不进入评测集。
格式对齐 gold_fenshen_1015_dev_v031.parquet。
字准表不划分 auto_gold / hardcase / non_speech，只统计 overall。
background_suspected 记 source_gold_kind=non_speech，计分参考是空串。
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

EXPORT_ID = "658c83799a3149a3a2ae0eaf3bc8bef3"
EXPORT_DIR = (
    ROOT
    / "tmp"
    / "1102"
    / "2"
    / f"export_{EXPORT_ID}"
    / EXPORT_ID
    / "7df2b7d9f30a4a5bbbee50d6c56a9a74"
)
DERIVED = ROOT / "datasets" / "stage1" / "derived"
OUT = DERIVED / "gold_fenshen_1015_dev_v032.parquet"
XLSX_PATHS = (ROOT / "tmp" / "1102" / "gold_fenshen_1015_dev_v032.xlsx",)
DETAIL_COLUMNS = (
    "id",
    "gold_text",
    "candidate_text",
    "wav_path",
    "duration",
    "label_source_gold_kind",
    "label_noise_kind",
    "label_type",
    "label_decision",
    "label_annotation_state",
    "label_is_human_verified",
    "label_label_tier",
    "label_label_source",
    "label_gold_source",
    "label_eval_type_split",
    "label_exclude_from_training",
    "label_eval_split",
    "label_eval_origin",
    "label_export_id",
    "sample_rate",
    "channels",
    "sha256",
)


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


def _noise_kind(ann: dict[str, Any]) -> str:
    return _cell(ann.get("noise_kind")) or _cell(ann.get("annotation_type"))


def _keep(ann: dict[str, Any]) -> tuple[bool, str]:
    category = _cell(ann.get("category"))
    gold = _cell(ann.get("gold_text"))
    gold_kind = _cell(ann.get("gold_kind"))
    decision = _cell(ann.get("decision"))
    noise_kind = _noise_kind(ann)
    if category == "environment_noise" or noise_kind in {"human_noise", "background_suspected"}:
        if noise_kind == "human_noise":
            return False, "human_noise"
        if noise_kind != "background_suspected":
            return False, f"未保留的环境噪声: {noise_kind or '空'}"
        if gold_kind != "non_speech" or decision != "empty" or gold:
            return False, "background_suspected 金标形态异常"
        return True, "background_suspected"
    if gold and gold_kind == "speech" and decision == "accept":
        return True, "speech"
    if _bool(ann.get("exclude_from_training")):
        return False, _cell(ann.get("training_exclusion_reason")) or "已排除训练"
    return False, "没有文本，也不是 background_suspected"


def build() -> None:
    ann_path = EXPORT_DIR / "annotations_1021.parquet"
    ing_path = EXPORT_DIR / "ingested_20261021_with_gold.parquet"
    if not ann_path.is_file() or not ing_path.is_file():
        raise SystemExit(f"[ERROR] 缺少导出: {EXPORT_DIR}")

    annotations = {_cell(row.get("sample_id")): row for row in pq.read_table(ann_path).to_pylist()}
    ingested = {_cell(row.get("id")): row for row in pq.read_table(ing_path).to_pylist()}
    annotations.pop("", None)
    ingested.pop("", None)
    if set(annotations) != set(ingested):
        missing = sorted(set(annotations) ^ set(ingested))[:8]
        raise SystemExit(f"[ERROR] 标注和音频 id 不一致，例如 {missing}")

    xlsx_mod = _load_xlsx_module()
    samples: list[Sample] = []
    excluded: list[dict[str, Any]] = []
    bucket_counts: dict[str, int] = {}
    for sample_id in sorted(annotations):
        ann = annotations[sample_id]
        ing = ingested[sample_id]
        keep, reason = _keep(ann)
        gold = _cell(ann.get("gold_text"))
        candidate = _cell(ann.get("candidate_text"))
        if not keep:
            excluded.append(
                {
                    "id": sample_id,
                    "category": _cell(ann.get("category")),
                    "noise_kind": _noise_kind(ann),
                    "gold_kind": _cell(ann.get("gold_kind")),
                    "decision": _cell(ann.get("decision")),
                    "candidate_text": candidate,
                    "reason": reason,
                }
            )
            continue

        audio = _as_dict(ing.get("audio"))
        wav = _cell(audio.get("resampled_16k") or ann.get("wav_path"))
        if not wav:
            raise SystemExit(f"[ERROR] 缺少 resampled_16k: {sample_id}")
        empty_ref = reason == "background_suspected"
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
            "decision": "empty" if empty_ref else "accept",
            "annotation_state": "human_accepted",
            "is_human_verified": True,
            "label_tier": "gold",
            "label_source": "human",
            "gold_source": "labeling_platform",
            "source_gold_kind": "non_speech" if empty_ref else "speech",
            "type": "non_speech" if empty_ref else "speech",
            "noise_kind": "background_suspected" if empty_ref else "",
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
                        operator="build_fenshen_1015_gold_v032_from_export",
                        version="v1",
                        params={"export_id": EXPORT_ID, "eval_type_split": False, "bucket": reason},
                    )
                ],
            )
        )
        bucket_counts[reason] = bucket_counts.get(reason, 0) + 1

    if not samples:
        raise SystemExit("[ERROR] 没有可评测样本")
    speech_n = bucket_counts.get("speech", 0)
    noise_n = bucket_counts.get("background_suspected", 0)
    if speech_n + noise_n != len(samples):
        raise SystemExit(f"[ERROR] 分桶合计不一致: {bucket_counts}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    manifest = Manifest(samples)
    manifest.save(OUT)
    manifest.save(OUT.with_suffix(".jsonl"))

    report = inspect_eval_manifest(OUT, require_formal_gold=True)
    if report.errors:
        raise SystemExit("[ERROR] 正式评测门禁未过:\n" + "\n".join(report.errors))

    gold_rows = xlsx_mod.load_gold_rows(OUT)
    if any(row["type"] for row in gold_rows):
        raise SystemExit("[ERROR] v032 不应写出分类 type")
    empty_refs = sum(not row["reference"] for row in gold_rows)
    if empty_refs != noise_n:
        raise SystemExit(f"[ERROR] 空参考 {empty_refs} 与 background_suspected {noise_n} 不一致")
    if sum(bool(row["reference"]) for row in gold_rows) != speech_n:
        raise SystemExit("[ERROR] 有文本条数与 speech 不一致")

    saved = pq.read_table(OUT).to_pylist()
    detail = pd.DataFrame(
        [
            {
                "id": row["id"],
                "gold_text": row.get("label_gold_text") or "",
                "candidate_text": row.get("qwen_1_text") or "",
                "wav_path": row.get("source_path") or "",
                "duration": row.get("duration"),
                "label_source_gold_kind": row.get("label_source_gold_kind"),
                "label_noise_kind": row.get("label_noise_kind") or "",
                "label_type": row.get("label_type"),
                "label_decision": row.get("label_decision"),
                "label_annotation_state": row.get("label_annotation_state"),
                "label_is_human_verified": row.get("label_is_human_verified"),
                "label_label_tier": row.get("label_label_tier"),
                "label_label_source": row.get("label_label_source"),
                "label_gold_source": row.get("label_gold_source"),
                "label_eval_type_split": row.get("label_eval_type_split"),
                "label_exclude_from_training": row.get("label_exclude_from_training"),
                "label_eval_split": row.get("label_eval_split"),
                "label_eval_origin": row.get("label_eval_origin"),
                "label_export_id": row.get("label_export_id"),
                "sample_rate": row.get("sample_rate"),
                "channels": row.get("channels"),
                "sha256": row.get("sha256"),
            }
            for row in saved
        ],
        columns=list(DETAIL_COLUMNS),
    )
    summary_rows = [
        {"项": "评测条数", "值": len(samples)},
        {"项": "speech", "值": speech_n},
        {"项": "background_suspected", "值": noise_n},
        {"项": "空金标", "值": empty_refs},
        {"项": "去掉 human_noise", "值": sum(1 for row in excluded if row["reason"] == "human_noise")},
        {"项": "未纳入", "值": len(excluded)},
        {"项": "来源", "值": str(OUT)},
    ]
    notes = pd.DataFrame(
        [
            ("金标", str(OUT)),
            ("条数", str(len(samples))),
            ("分类", "label_eval_type_split=false。字准表只统计 overall。"),
            ("空参考", "background_suspected 的 source_gold_kind=non_speech，计分参考是空串。"),
            ("音频", "wav_path 即 source_path，对应 audio.resampled_16k。"),
            ("预标注", "candidate_text 即 transcripts.qwen_1.text。"),
            ("纳入", f"有文本标注的 speech {speech_n} 条，background_suspected {noise_n} 条。"),
            ("未纳入", "human_noise，以及多人说话、停顿、音频截断。"),
            ("命令", "bash scripts/run_asr_eval.sh fenshen lora v032 dev best"),
            ("导出", str(EXPORT_DIR)),
        ],
        columns=["字段/口径", "说明"],
    )
    for xlsx in XLSX_PATHS:
        xlsx.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
            pd.DataFrame(summary_rows).to_excel(writer, index=False, sheet_name="汇总")
            detail.to_excel(writer, index=False, sheet_name="明细")
            pd.DataFrame(excluded).to_excel(writer, index=False, sheet_name="未纳入")
            notes.to_excel(writer, index=False, sheet_name="说明")

    meta = {
        "created_at": _now_iso(),
        "name": "fenshen_1015_dev_v032",
        "purpose": "eval_source",
        "sample_count": len(samples),
        "excluded_count": len(excluded),
        "bucket_counts": bucket_counts,
        "empty_reference_count": empty_refs,
        "eval_type_split": False,
        "source": {"export_id": EXPORT_ID, "export_dir": str(EXPORT_DIR)},
        "type_policy": "不做分类。字准表只统计 overall。background_suspected 按空串计分。human_noise 不纳入。",
        "next": "bash scripts/run_asr_eval.sh fenshen lora v032 dev best",
    }
    OUT.with_suffix(".meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"[OK] samples={len(samples)} speech={speech_n} "
        f"background_suspected={noise_n} excluded={len(excluded)} unclassified"
    )
    print(f"[OK] gold {OUT}")
    for xlsx in XLSX_PATHS:
        print(f"[OK] xlsx {xlsx}")


if __name__ == "__main__":
    build()
