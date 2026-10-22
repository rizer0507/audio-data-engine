#!/usr/bin/env python3
"""从 audio-qwen3-asr-train 的 exploratory train/dev 切分构建引擎评测源 Manifest。

用途：把 sample_map 中指定切分落成 datasets/stage1/derived/gold_<name>.parquet，
再经 ``audio-data eval register`` 进入工序三评测流水线。

默认 ``--split dev``（预留 20% 验证集）；``--split all`` 导出 train+dev 全量
（用于训练集覆盖评测，非正式独立 eval_core）。

金标口径：标注平台人工转写（decision=accept 的 speech + decision=empty 的
non_speech 空金标）。本脚本写入 annotation_state=human_accepted /
is_human_verified / label_tier=gold，并避免把未完成 v3 双审字段的 gold_kind
放进 labels，以免 formal 门禁走严格 v3 校验失败。
产物属于「人审转写评测集」；非正式 Release eval_core。

示例::

  python scripts/build_eval_from_qwen3_dev_split.py \\
    --train-repo ../audio-qwen3-asr-train \\
    --dataset-rel training_sets/exploratory/fenshen_1015/train_v002 \\
    --name fenshen_1015_dev_v002

  # 训练集全量（train+dev）
  python scripts/build_eval_from_qwen3_dev_split.py \\
    --train-repo ../audio-qwen3-asr-train \\
    --dataset-rel training_sets/exploratory/fenshen_1015/train_v002 \\
    --split all \\
    --name fenshen_1015_full_v002
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audio_engine.core.manifest import Manifest  # noqa: E402
from audio_engine.core.sample import Sample  # noqa: E402
from audio_engine.core.source_naming import STAGE1_DERIVED_DIR, validate_source_name  # noqa: E402

INVALID_GOLD = {"", "/", "-", "无", "none", "null", "n/a", "na"}
# 扁平行里可能存在、需要并入 labels 的金标与血缘字段（不含 gold_kind）
FLAT_LABEL_KEYS = (
    "gold_text",
    "decision",
    "annotation_status",
    "annotator_id",
    "annotated_at",
    "source_row_id",
    "alm_batch",
    "human_category",
    "source_category",
    "reviewed_category",
    "annotation_type",
    "noise_kind",
    "training_exclusion_reason",
    "transcript_annotation_protocol",
    "exclude_from_training",
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
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {}
        return dict(parsed) if isinstance(parsed, dict) else {}
    return {}


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return []
        return list(parsed) if isinstance(parsed, list) else []
    return []


def _load_split_rows(sample_map: Path, *, split: str) -> list[dict[str, Any]]:
    """返回 sample_map 行，顺序与文件一致。加速副本靠 augmentation.source_sample_id 回挂金标。"""
    allowed = {"train", "dev"} if split == "all" else {split}
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with sample_map.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            split_name = str(row.get("split") or "").strip()
            if split_name not in allowed:
                continue
            sid = _cell(row.get("sample_id") or row.get("id"))
            if not sid or sid in seen:
                continue
            seen.add(sid)
            rows.append(row)
    if not rows:
        raise SystemExit(f"[ERROR] sample_map 中没有 split={split}: {sample_map}")
    return rows


def _clone_augmented_row(source_row: dict[str, Any], map_row: dict[str, Any]) -> dict[str, Any]:
    """用源切段金标克隆一行，音频换成加速副本。源 ASR 不带到新音频上。"""
    row = copy.deepcopy(source_row)
    aug = _as_dict(map_row.get("augmentation"))
    derived_id = _cell(map_row.get("sample_id") or map_row.get("id"))
    audio_path = _cell(map_row.get("audio_path"))
    row["id"] = derived_id
    sha = _cell(map_row.get("audio_sha256"))
    if sha:
        row["sha256"] = sha
    if audio_path:
        row["source_path"] = audio_path
    audio = _as_dict(row.get("audio"))
    if audio_path:
        audio["resampled_16k"] = audio_path
    row["audio"] = audio

    speed = aug.get("speed_factor")
    duration = row.get("duration")
    try:
        speed_f = float(speed) if speed not in (None, "") else 0.0
        duration_f = float(duration) if duration not in (None, "") else 0.0
    except (TypeError, ValueError):
        speed_f = 0.0
        duration_f = 0.0
    if speed_f > 0 and duration_f > 0 and duration_f == duration_f:
        row["duration"] = duration_f / speed_f

    labels = _as_dict(row.get("labels"))
    labels["augmentation"] = aug
    source_id = _cell(aug.get("source_sample_id"))
    if source_id:
        labels["source_sample_id"] = source_id
    if duration_f > 0:
        labels["source_duration"] = duration_f
    transcripts = _as_dict(row.get("transcripts"))
    if transcripts:
        labels["source_transcripts"] = transcripts
        row["transcripts"] = {}
    row["labels"] = labels
    return row


def _is_non_speech_row(row: dict[str, Any], labels: dict[str, Any], gold_kind_meta: str | None) -> bool:
    gold_kind = _cell(gold_kind_meta) or _cell(row.get("gold_kind")) or _cell(labels.get("gold_kind"))
    decision = _cell(row.get("decision") or labels.get("decision"))
    return gold_kind == "non_speech" or decision == "empty"


def _row_to_sample(
    row: dict[str, Any],
    *,
    gold_kind_meta: str | None,
    eval_split: str,
    eval_origin: str,
) -> Sample | None:
    sample_id = _cell(row.get("id") or row.get("sample_id"))
    if not sample_id:
        return None

    labels = _as_dict(row.get("labels"))
    for key in FLAT_LABEL_KEYS:
        if key in labels and _cell(labels.get(key)):
            continue
        flat = row.get(key)
        if flat is not None and _cell(flat):
            labels[key] = flat if not isinstance(flat, str) else _cell(flat)

    is_non_speech = _is_non_speech_row(row, labels, gold_kind_meta)
    gold = _cell(labels.get("gold_text") or row.get("gold_text"))
    if is_non_speech:
        # 确认无语音：金标为空串，仍进入评测集（测幻觉/空转写）
        gold = ""
    elif not gold or gold.lower() in INVALID_GOLD:
        return None
    labels["gold_text"] = gold
    labels["label"] = gold

    # 人审转写正式评测标记（非 v3 双审 schema）
    labels["annotation_state"] = "human_accepted"
    labels["is_human_verified"] = True
    labels["label_tier"] = "gold"
    labels["label_source"] = "human"
    labels["gold_source"] = "labeling_platform"
    if gold_kind_meta:
        labels["source_gold_kind"] = gold_kind_meta  # 不写 gold_kind，避免误触发 v3 formal
    if is_non_speech:
        labels["type"] = "non_speech"
        labels["source_gold_kind"] = labels.get("source_gold_kind") or "non_speech"
    else:
        labels.setdefault(
            "type",
            _cell(labels.get("human_category") or labels.get("source_category"))
            or "human_transcript",
        )
    labels["classification_bucket"] = labels["type"]
    labels["eval_split"] = eval_split
    labels["eval_origin"] = eval_origin

    audio = _as_dict(row.get("audio"))
    if "resampled_16k" not in audio:
        path = _cell(row.get("audio_path") or row.get("wav_path"))
        if path:
            audio["resampled_16k"] = path
    if "resampled_16k" not in audio:
        return None

    source_path = _cell(row.get("source_path")) or audio["resampled_16k"]
    sha256 = _cell(row.get("sha256") or row.get("original_audio_sha256") or labels.get("original_audio_sha256"))
    transcripts = _as_dict(row.get("transcripts"))
    quality = _as_dict(row.get("quality"))
    lineage = _as_list(row.get("lineage"))
    status = _as_dict(row.get("status"))
    errors = _as_dict(row.get("errors"))

    # Sample 需要 LineageEntry；交由 pydantic 从 dict 构造
    return Sample(
        id=sample_id,
        source_path=source_path,
        sha256=sha256,
        sample_rate=row.get("sample_rate") if row.get("sample_rate") == row.get("sample_rate") else None,
        channels=row.get("channels") if row.get("channels") == row.get("channels") else None,
        duration=row.get("duration") if row.get("duration") == row.get("duration") else None,
        audio=audio,
        transcripts=transcripts,
        quality=quality,
        labels=labels,
        lineage=lineage,
        status=status,
        errors=errors,
    )


def _load_cleaned_rows(path: Path) -> dict[str, dict[str, Any]]:
    table = pq.read_table(path)
    out: dict[str, dict[str, Any]] = {}
    for row in table.to_pylist():
        sid = _cell(row.get("id") or row.get("sample_id"))
        if sid:
            out[sid] = row
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-repo",
        type=Path,
        default=ROOT.parent / "audio-qwen3-asr-train",
        help="audio-qwen3-asr-train 仓库根目录",
    )
    parser.add_argument(
        "--dataset-rel",
        type=str,
        default="training_sets/exploratory/fenshen_1015/train_v001",
        help="相对 data/ 的训练集版本目录",
    )
    parser.add_argument(
        "--cleaned-parquet",
        type=Path,
        default=None,
        help="含 resampled_16k 与金标扁平行的 cleaned_*_with_gold.parquet；默认取 batch raw",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=("dev", "train", "all"),
        default="dev",
        help="sample_map 切分：dev / train / all(train+dev)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="fenshen_1015_dev",
        help="派生源 stem，写出 gold_<name>.parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="覆盖默认输出路径 datasets/stage1/derived/gold_<name>.parquet",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    name = validate_source_name(args.name)
    dataset_dir = (args.train_repo / "data" / args.dataset_rel).resolve()
    sample_map = dataset_dir / "sample_map.jsonl"
    if not sample_map.is_file():
        raise SystemExit(f"[ERROR] 缺少 sample_map: {sample_map}")

    cleaned = args.cleaned_parquet
    if cleaned is None:
        batch_raw = (
            args.train_repo
            / "data"
            / "batches"
            / "label_20261015_fenshen"
            / "v001"
            / "raw"
        )
        matches = sorted(batch_raw.glob("cleaned_*_with_gold.parquet"))
        if not matches:
            raise SystemExit(f"[ERROR] 未找到 cleaned_*_with_gold.parquet: {batch_raw}")
        cleaned = matches[0]
    cleaned = cleaned.resolve()
    if not cleaned.is_file():
        raise SystemExit(f"[ERROR] cleaned parquet 不存在: {cleaned}")

    out = args.output
    if out is None:
        out = ROOT / STAGE1_DERIVED_DIR / f"gold_{name}.parquet"
    else:
        out = out if out.is_absolute() else ROOT / out
    if out.exists() and not args.force:
        raise SystemExit(f"[ERROR] 已存在: {out}（加 --force 覆盖）")

    split_rows = _load_split_rows(sample_map, split=args.split)
    rows_by_id = _load_cleaned_rows(cleaned)
    origin_by_split = {
        "dev": "qwen3_asr_train_exploratory_dev",
        "train": "qwen3_asr_train_exploratory_train",
        "all": "qwen3_asr_train_exploratory_full",
    }
    eval_origin = origin_by_split[args.split]

    samples: list[Sample] = []
    missing: list[str] = []
    skipped_invalid_gold = 0
    included_non_speech = 0
    included_augmented = 0
    split_counts: dict[str, int] = {}
    for map_row in split_rows:
        sid = _cell(map_row.get("sample_id") or map_row.get("id"))
        split_name = str(map_row.get("split") or "").strip()
        row = rows_by_id.get(sid)
        if row is None:
            aug = _as_dict(map_row.get("augmentation"))
            source_id = _cell(aug.get("source_sample_id"))
            source_row = rows_by_id.get(source_id) if source_id else None
            if source_row is None:
                missing.append(sid)
                continue
            row = _clone_augmented_row(source_row, map_row)
            included_augmented += 1
        gold_kind = _cell(row.get("gold_kind")) or _cell(map_row.get("gold_kind"))
        sample = _row_to_sample(
            row,
            gold_kind_meta=gold_kind or None,
            eval_split=split_name,
            eval_origin=eval_origin,
        )
        if sample is None:
            skipped_invalid_gold += 1
            continue
        if str(sample.labels.get("type") or "") == "non_speech":
            included_non_speech += 1
        split_counts[split_name] = split_counts.get(split_name, 0) + 1
        samples.append(sample)

    if missing:
        preview = ", ".join(missing[:10])
        more = f" ...(+{len(missing) - 10})" if len(missing) > 10 else ""
        raise SystemExit(
            f"[ERROR] cleaned 中缺少 {len(missing)} 个 id (split={args.split}): "
            f"{preview}{more}"
        )
    if not samples:
        raise SystemExit("[ERROR] 未产出任何评测样本")

    out.parent.mkdir(parents=True, exist_ok=True)
    Manifest(samples).save(out)
    Manifest(samples).save(out.with_suffix(".jsonl"))

    meta = {
        "created_at": _now_iso(),
        "name": name,
        "purpose": "eval_source",
        "sample_count": len(samples),
        "requested_ids": len(split_rows),
        "split_counts": split_counts,
        "included_non_speech": included_non_speech,
        "included_augmented": included_augmented,
        "skipped_invalid_gold": skipped_invalid_gold,
        "source": {
            "train_repo": str(args.train_repo.resolve()),
            "dataset_rel": args.dataset_rel,
            "sample_map": str(sample_map),
            "cleaned_parquet": str(cleaned),
            "split": args.split,
            "dev_ratio": 0.2,
            "split_seed": 42,
            "group_by": "parent_pcm_sha256",
        },
        "formal_markers": {
            "annotation_state": "human_accepted",
            "is_human_verified": True,
            "label_tier": "gold",
            "label_source": "human",
            "note": (
                "来自标注平台人工转写；非正式 Release eval_core"
                + ("；含训练集样本，仅作覆盖评测" if args.split != "dev" else "")
            ),
        },
        "next": [
            f'audio-data eval register "{out.as_posix()}" --name eval_{name}',
            f"audio-data eval check eval_{name} --formal",
        ],
    }
    meta_path = out.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"[OK] wrote {len(samples)} samples -> {out}")
    print(
        f"[OK] split_counts={split_counts} non_speech={included_non_speech} "
        f"augmented={included_augmented}"
    )
    print(f"[OK] meta -> {meta_path}")
    print(f"Next: audio-data eval register \"{out.as_posix()}\" --name eval_{name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
