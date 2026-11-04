#!/usr/bin/env python3
"""从 fenshen_1015 的 sample_map 写出 train/dev 金标，供评测脚本注册。

v016 及以后有加速 wav 和三路一致机转，不在分身 cleaned 表里。
本脚本以 sample_map 的音频路径和训练文本为准：

- 人工 auto_gold、人工 hardcase：音频和 qwen_1 原文取 cleaned，金标取 sample_map.gold_text。
- 三路一致：音频取 sample_map，qwen_1_text 与金标相同。
- 加速 hardcase：音频取 1.4 倍 wav，金标取 sample_map.gold_text，qwen_1_text 取加速后转写。
  v016 / v017 的 gold_text 是加速前 qwen1。v018 / v019 / v020 的 gold_text 是三路命中的否定词。
- gold_text 为空才记 non_speech。v018 及以后 qwen1 为空的加速条已有否定词金标，不记空。
- 同一 id 只保留一行。v017 / v018 / v020 有 60 条人工样本同时在两个类别，文本和音频相同。
  v019 已去掉 auto_gold，没有这类重复。v020 只改了训练 prompt，金标与 v018 相同。

用法::

  python scripts/build_fenshen_1015_gold_from_sample_map.py v016 v017
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audio_engine.core.eval_ready import inspect_eval_manifest  # noqa: E402
from audio_engine.core.manifest import Manifest  # noqa: E402
from audio_engine.core.sample import LineageEntry, Sample  # noqa: E402

TRAIN_REPO = ROOT.parent / "audio-qwen3-asr-train"
CLEANED = (
    TRAIN_REPO
    / "data"
    / "batches"
    / "label_20261015_fenshen"
    / "v001"
    / "raw"
    / "cleaned_1012_tag_badcase1014_vad_merged_v2_with_gold.parquet"
)
DERIVED = ROOT / "datasets" / "stage1" / "derived"
HUMAN_COMPONENTS = frozenset({"auto_gold_human", "hardcase_drop_identity"})


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


def _load_builder():
    path = ROOT / "scripts" / "eval_fenshen_1015_xlsx.py"
    spec = importlib.util.spec_from_file_location("eval_fenshen_1015_xlsx", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_cleaned(path: Path) -> dict[str, dict[str, Any]]:
    table = pq.read_table(
        path,
        columns=["id", "sample_rate", "channels", "audio", "transcripts", "sha256"],
    )
    rows: dict[str, dict[str, Any]] = {}
    for row in table.to_pylist():
        sid = _cell(row.get("id"))
        if sid:
            rows[sid] = row
    return rows


def _qwen_text(row: dict[str, Any] | None) -> str:
    if not row:
        return ""
    transcripts = _as_dict(row.get("transcripts"))
    entry = transcripts.get("qwen_1")
    if isinstance(entry, dict):
        return _cell(entry.get("text"))
    return ""


def _load_split(sample_map: Path, split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with sample_map.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if _cell(row.get("split")) == split:
                rows.append(row)
    if not rows:
        raise SystemExit(f"[ERROR] sample_map 没有 split={split}: {sample_map}")
    return rows


def _sample_from_map(
    map_row: dict[str, Any],
    cleaned_row: dict[str, Any] | None,
    *,
    eval_origin: str,
) -> Sample:
    sample_id = _cell(map_row.get("sample_id"))
    component = _cell(map_row.get("source_component"))
    bucket = _cell(map_row.get("selection_bucket"))
    audio_path = _cell(map_row.get("audio_path"))
    if not sample_id or not audio_path:
        raise SystemExit(f"[ERROR] sample_map 缺少 id 或音频: {map_row}")

    gold = _cell(map_row.get("gold_text"))
    empty_target = not gold
    if empty_target:
        gold_kind = "non_speech"
        decision = "empty"
        row_type = "non_speech"
        qwen_text = _cell(map_row.get("qwen_after")) or _cell(map_row.get("qwen_plain"))
    elif component == "speed_1p4_qwen1":
        gold_kind = "speech"
        decision = _cell(map_row.get("decision")) or "accept"
        row_type = "hardcase"
        qwen_text = _cell(map_row.get("qwen_after")) or _cell(map_row.get("qwen_plain"))
    elif component == "auto_gold_three_agree":
        gold_kind = "speech"
        decision = "accept"
        row_type = "auto_gold"
        qwen_text = gold
    else:
        gold_kind = _cell(map_row.get("gold_kind")) or "speech"
        decision = _cell(map_row.get("decision")) or "accept"
        row_type = bucket or "human_transcript"
        qwen_text = _qwen_text(cleaned_row) or _cell(map_row.get("qwen_plain"))

    explicit_type = _cell(map_row.get("eval_type"))
    if explicit_type:
        row_type = explicit_type

    if component in HUMAN_COMPONENTS or component == "background_suspected":
        label_source = "human"
        gold_source = "labeling_platform"
    elif component == "speed_1p4_qwen1":
        label_source = "qwen1"
        gold_source = "speed_qwen_before"
    else:
        label_source = "three_agree"
        gold_source = "three_route_agree"

    human_category = _cell(map_row.get("human_category")) or bucket
    labels: dict[str, Any] = {
        "gold_text": "" if empty_target else gold,
        "label": "" if empty_target else gold,
        "decision": decision,
        "annotation_state": "human_accepted",
        "is_human_verified": True,
        "label_tier": "gold",
        "label_source": label_source,
        "gold_source": gold_source,
        "source_gold_kind": gold_kind,
        "type": row_type if empty_target else (bucket or row_type),
        "classification_bucket": "non_speech" if empty_target else bucket,
        "human_category": human_category,
        "source_component": component,
        "selection_bucket": bucket,
        "eval_type": explicit_type or row_type,
        "noise_kind": _cell(map_row.get("noise_kind")),
        "eval_split": _cell(map_row.get("split")),
        "eval_origin": eval_origin,
        "exclude_from_training": False,
    }
    if empty_target:
        labels["type"] = "non_speech"
        labels["source_gold_kind"] = "non_speech"

    sample_rate = 16000
    channels = 1
    if cleaned_row is not None:
        rate = cleaned_row.get("sample_rate")
        ch = cleaned_row.get("channels")
        if isinstance(rate, int) and rate > 0:
            sample_rate = rate
        if isinstance(ch, int) and ch > 0:
            channels = ch

    duration = map_row.get("duration")
    try:
        duration_f = float(duration) if duration not in (None, "") else None
    except (TypeError, ValueError):
        duration_f = None
    if duration_f is not None and duration_f != duration_f:
        duration_f = None

    return Sample(
        id=sample_id,
        source_path=audio_path,
        sha256=_cell(map_row.get("audio_sha256")),
        sample_rate=sample_rate,
        channels=channels,
        duration=duration_f,
        audio={"resampled_16k": audio_path},
        transcripts={
            "qwen_1": {
                "text": qwen_text,
                "model": "qwen3-asr",
                "version": "Qwen3-ASR-1.7B",
            }
        },
        labels=labels,
        lineage=[
            LineageEntry(
                operator="build_fenshen_1015_gold_from_sample_map",
                version="v1",
                params={
                    "source_component": component,
                    "selection_bucket": bucket,
                },
            )
        ],
    )


def build_one(version: str, split: str, *, cleaned: dict[str, dict[str, Any]]) -> Path:
    dataset_dir = (
        TRAIN_REPO / "data" / "training_sets" / "exploratory" / "fenshen_1015" / f"train_{version}"
    )
    sample_map = dataset_dir / "sample_map.jsonl"
    if not sample_map.is_file():
        raise SystemExit(f"[ERROR] 缺少 sample_map: {sample_map}")

    out = DERIVED / f"gold_fenshen_1015_{split}_{version}.parquet"
    rows = _load_split(sample_map, split)
    bucket_order = {"non_speech": 0, "hardcase": 1, "auto_gold": 2}
    rows.sort(
        key=lambda row: (
            bucket_order.get(_cell(row.get("selection_bucket")), 9),
            _cell(row.get("sample_id")),
        )
    )
    origin = f"qwen3_asr_train_exploratory_{split}"
    samples: list[Sample] = []
    seen: dict[str, dict[str, str]] = {}
    collapsed = 0
    kept_hardcase_over_auto_gold = 0
    missing_human = 0
    component_counts: dict[str, int] = {}
    for map_row in rows:
        sid = _cell(map_row.get("sample_id"))
        audio_path = _cell(map_row.get("audio_path"))
        gold = _cell(map_row.get("gold_text"))
        bucket = _cell(map_row.get("selection_bucket"))
        previous = seen.get(sid)
        if previous is not None:
            if previous["audio"] != audio_path:
                raise SystemExit(f"[ERROR] 重复 id 的音频不一致: {sid}")
            if previous["gold"] != gold:
                if previous["bucket"] in {"hardcase", "non_speech"} and bucket == "auto_gold":
                    collapsed += 1
                    kept_hardcase_over_auto_gold += 1
                    continue
                raise SystemExit(f"[ERROR] 重复 id 的金标不一致: {sid}")
            collapsed += 1
            continue
        seen[sid] = {"audio": audio_path, "gold": gold, "bucket": bucket}
        component = _cell(map_row.get("source_component"))
        cleaned_row = cleaned.get(sid)
        if component in HUMAN_COMPONENTS and cleaned_row is None:
            missing_human += 1
            continue
        if component in HUMAN_COMPONENTS and cleaned_row is not None:
            cleaned_audio = _as_dict(cleaned_row.get("audio")).get("resampled_16k")
            if _cell(cleaned_audio) and _cell(cleaned_audio) != audio_path:
                raise SystemExit(f"[ERROR] 人工条音频与 cleaned 不一致: {sid}")
        samples.append(_sample_from_map(map_row, cleaned_row, eval_origin=origin))
        component_counts[component] = component_counts.get(component, 0) + 1

    if missing_human:
        raise SystemExit(f"[ERROR] cleaned 缺少 {missing_human} 条人工样本")
    if not samples:
        raise SystemExit(f"[ERROR] {version} {split} 没有样本")

    out.parent.mkdir(parents=True, exist_ok=True)
    manifest = Manifest(samples)
    manifest.save(out)
    manifest.save(out.with_suffix(".jsonl"))

    report = inspect_eval_manifest(out, require_formal_gold=True)
    if report.errors:
        joined = "\n".join(report.errors)
        raise SystemExit(f"[ERROR] 正式评测门禁未过 {out}:\n{joined}")

    xlsx = _load_builder()
    gold_rows = xlsx.load_gold_rows(out)
    type_counts: dict[str, int] = {}
    for row in gold_rows:
        type_counts[row["type"]] = type_counts.get(row["type"], 0) + 1

    meta = {
        "created_at": _now_iso(),
        "name": f"fenshen_1015_{split}_{version}",
        "purpose": "eval_source",
        "sample_count": len(samples),
        "requested_rows": len(rows),
        "collapsed_duplicate_ids": collapsed,
        "kept_hardcase_over_auto_gold": kept_hardcase_over_auto_gold,
        "component_counts": component_counts,
        "type_counts": type_counts,
        "source": {
            "train_repo": str(TRAIN_REPO),
            "dataset_rel": f"training_sets/exploratory/fenshen_1015/train_{version}",
            "sample_map": str(sample_map),
            "cleaned_parquet": str(CLEANED),
            "split": split,
        },
        "text_policy": {
            "auto_gold_human": "gold=label_gold_text, qwen_1=cleaned qwen_1",
            "hardcase_drop_identity": "gold=v013 gold_text, qwen_1=cleaned qwen_1",
            "auto_gold_three_agree": "gold=qwen1_text, qwen_1=同一文本",
            "speed_1p4_qwen1": "gold=sample_map.gold_text, qwen_1=qwen_after, audio=tempo_1p4",
            "empty_gold": "gold_text 为空才记 non_speech。v018 加速条金标是命中否定词，不记空。",
        },
        "formal_markers": {
            "annotation_state": "human_accepted",
            "is_human_verified": True,
            "label_tier": "gold",
            "note": "覆盖评测用。加速条和三路一致条不是人工转写。非正式 Release eval_core。",
        },
        "next": [
            f'audio-data eval register "{out.as_posix()}" --name eval_fenshen_1015_{split}_{version} --force',
            f"bash scripts/run_fenshen_1015_eval.sh full {version}",
            f"python scripts/eval_fenshen_1015_xlsx.py full {version}",
        ],
    }
    meta_path = out.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"[OK] {version} {split} rows={len(rows)} unique={len(samples)} "
        f"collapsed={collapsed} types={type_counts} -> {out}"
    )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="从 sample_map 写 fenshen_1015 train/dev 金标")
    parser.add_argument("versions", nargs="+", help="例如 v016 v017")
    args = parser.parse_args(argv)
    if not CLEANED.is_file():
        raise SystemExit(f"[ERROR] 缺少 cleaned: {CLEANED}")
    cleaned = _load_cleaned(CLEANED)
    for version in args.versions:
        if not version.startswith("v") or not version[1:].isdigit():
            raise SystemExit(f"[ERROR] 版本号须为 v 加数字，当前: {version}")
        for split in ("train", "dev"):
            build_one(version, split, cleaned=cleaned)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
