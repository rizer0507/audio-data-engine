#!/usr/bin/env python3
"""从手工修订的 summary.xlsx + 已有推理 parquet 跑完解耦评测流程。

适用场景：
  - 未走完人审，直接有 summary.xlsx，并已改 type、补全 label
  - 已有各模型 ASR parquet（不必再起 vLLM）
  - 只要评测集与推理结果 id 对齐，即可相对 gold 算字准并按 type 汇总
  - type=noise：label/gold/全部转写写成 ``噪声``
  - type=review_queue 且 label 为空：同上（与 noise 相同占位口径）
  - 不用字面 ``/``：会被 zh_asr_v1 去标点清空，无法计字准

Example（本仓库当前本地产物）::

  python scripts/run_eval_from_summary.py \\
    --summary data/exports/summary_local_test.xlsx \\
    --model qwen-sft-e10=datasets/manifests/qwen-sft-e10_asr_sft-ep10.parquet \\
    --model qwen-sft-e100=datasets/manifests/qwen-sft-e100_asr_sft-ep100.parquet \\
    --eval-name eval_local_test \\
    --eval-model qwen1 \\
    --eval-model qwen-sft-e10 \\
    --eval-model qwen-sft-e100

流程：
  1) summary.xlsx → datasets/manifests/<eval-name>.parquet（label→gold_text，带 type）
  2) 用任一推理 parquet 补齐 resampled_16k / sha256（方便 eval check）
  3) eval_aggregate（评测集底表 + 按 id 左连接各模型）
  4) eval_metric_pipeline（vs gold + 按 type 导出 evaluation.xlsx）
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audio_engine.core.manifest import Manifest  # noqa: E402
from audio_engine.core.sample import Sample  # noqa: E402
from audio_engine.core.source_naming import validate_asr_run, validate_source_name  # noqa: E402

_SKIP_TEXT_COLS = {
    "gold_text",
    "label_text",
    "label_gold_text",
    "baseline_text",
    "text",
}
_META_COLS = {
    "sample_id",
    "id",
    "sha256",
    "source_path",
    "type",
    "classification_bucket",
    "classification_reason",
    "label",
    "gold_text",
    "gold_source",
    "annotation_state",
    "annotation_reason",
    "selection_policy_version",
    "label_gold_text",
}


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


def _parse_model_arg(raw: str) -> tuple[str, Path]:
    text = (raw or "").strip()
    if "=" not in text:
        raise SystemExit(
            f"[ERROR] --model 格式应为 alias=path.parquet，收到: {raw!r}\n"
            f"        例: --model qwen-sft-e10=datasets/manifests/qwen-sft-e10_asr_sft-ep10.parquet"
        )
    alias, _, path = text.partition("=")
    alias = validate_asr_run(alias.strip())
    path_obj = Path(path.strip())
    if not path_obj.is_absolute():
        path_obj = (ROOT / path_obj).resolve()
    if not path_obj.is_file():
        raise SystemExit(f"[ERROR] 模型 parquet 不存在: {path_obj}")
    return alias, path_obj


def _load_summary_xlsx(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise SystemExit(f"[ERROR] summary xlsx 不存在: {path}")
    frame = pd.read_excel(path, dtype=str).fillna("")
    if frame.empty:
        raise SystemExit(f"[ERROR] summary xlsx 为空: {path}")
    id_col = "sample_id" if "sample_id" in frame.columns else ("id" if "id" in frame.columns else None)
    if id_col is None:
        raise SystemExit(
            f"[ERROR] summary 缺少 sample_id/id 列；可用列: {list(frame.columns)}"
        )
    if "type" not in frame.columns and "classification_bucket" not in frame.columns:
        raise SystemExit("[ERROR] summary 缺少 type / classification_bucket 列")
    if "label" not in frame.columns and "gold_text" not in frame.columns:
        raise SystemExit("[ERROR] summary 缺少 label / gold_text 列（需要金标）")
    frame = frame.copy()
    frame["_id"] = frame[id_col].map(_cell)
    empty_ids = int((frame["_id"] == "").sum())
    if empty_ids:
        raise SystemExit(f"[ERROR] summary 有 {empty_ids} 行空 id")
    dup = frame["_id"][frame["_id"].duplicated()].tolist()
    if dup:
        raise SystemExit(f"[ERROR] summary 有重复 id: {dup[:10]}")
    return frame


def _transcripts_from_row(row: pd.Series) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for col, value in row.items():
        name = str(col)
        if name in _META_COLS or name.startswith("_"):
            continue
        if not name.endswith("_text"):
            continue
        key = name[: -len("_text")]
        if key in _SKIP_TEXT_COLS or not key:
            continue
        text = _cell(value)
        if text:
            out[key] = {"text": text}
    return out


_EMPTY_LABEL_MARKER = "噪声"  # 不用 "/"：zh_asr_v1 去标点后会变成空，reference_length=0 无法计字准
_NOISE_MARKER = _EMPTY_LABEL_MARKER  # 兼容旧名


def _bucket_of(sample: Sample) -> str:
    return str(sample.labels.get("type") or sample.labels.get("classification_bucket") or "")


def _set_all_transcripts(sample: Sample, text: str) -> None:
    for key, entry in list(sample.transcripts.items()):
        if isinstance(entry, dict):
            updated = dict(entry)
            updated["text"] = text
            sample.transcripts[key] = updated
        else:
            sample.transcripts[key] = {"text": text}


def apply_placeholder_markers(samples: list[Sample]) -> dict[str, int]:
    """占位规则（同 noise）：

    - type=noise：label/gold/全部转写 → 噪声
    - type=review_queue 且 label/gold 为空（或已是占位符）：同上
      （aggregate join 后仍会走「已是占位符」分支，覆盖 SFT 转写）
    """
    counts = {"noise": 0, "review_queue_empty_label": 0}
    for sample in samples:
        bucket = _bucket_of(sample)
        label = str(sample.labels.get("label") or "").strip()
        gold = str(sample.labels.get("gold_text") or "").strip()
        if bucket == "noise":
            sample.labels["type"] = "noise"
            sample.labels["classification_bucket"] = "noise"
            sample.labels["label"] = _EMPTY_LABEL_MARKER
            sample.labels["gold_text"] = _EMPTY_LABEL_MARKER
            _set_all_transcripts(sample, _EMPTY_LABEL_MARKER)
            counts["noise"] += 1
            continue
        empty_rq = bucket == "review_queue" and not label and not gold
        marked_rq = bucket == "review_queue" and (
            label == _EMPTY_LABEL_MARKER or gold == _EMPTY_LABEL_MARKER
        )
        if empty_rq or marked_rq:
            sample.labels["type"] = "review_queue"
            sample.labels["classification_bucket"] = "review_queue"
            sample.labels["label"] = _EMPTY_LABEL_MARKER
            sample.labels["gold_text"] = _EMPTY_LABEL_MARKER
            _set_all_transcripts(sample, _EMPTY_LABEL_MARKER)
            counts["review_queue_empty_label"] += 1
    return counts


def apply_noise_slash_marker(samples: list[Sample]) -> int:
    """兼容旧调用：仅统计 noise 条数。"""
    return int(apply_placeholder_markers(samples).get("noise", 0))


def summary_to_samples(frame: pd.DataFrame) -> list[Sample]:
    samples: list[Sample] = []
    for _, row in frame.iterrows():
        sample_id = _cell(row["_id"])
        bucket = _cell(row.get("type")) or _cell(row.get("classification_bucket")) or "unclassified"
        label = _cell(row.get("label"))
        gold = _cell(row.get("gold_text")) or label
        # 空 label，或上一轮已写成占位符的 review_queue
        empty_review = bucket == "review_queue" and (
            (not label and not gold)
            or label == _EMPTY_LABEL_MARKER
            or gold == _EMPTY_LABEL_MARKER
        )
        if bucket == "noise" or empty_review:
            label = _EMPTY_LABEL_MARKER
            gold = _EMPTY_LABEL_MARKER
        labels: dict[str, Any] = {
            "type": bucket,
            "classification_bucket": bucket,
            "label": label or gold,
            "gold_text": gold,
        }
        for key in (
            "gold_source",
            "annotation_state",
            "annotation_reason",
            "selection_policy_version",
            "classification_reason",
        ):
            value = _cell(row.get(key))
            if value:
                labels[key] = value
        transcripts = _transcripts_from_row(row)
        if bucket == "noise" or empty_review:
            for col in row.index:
                name = str(col)
                if not name.endswith("_text") or name in _META_COLS or name.startswith("_"):
                    continue
                key = name[: -len("_text")]
                if key in _SKIP_TEXT_COLS or not key:
                    continue
                transcripts[key] = {"text": _EMPTY_LABEL_MARKER}
            for key in list(transcripts.keys()):
                transcripts[key] = {"text": _EMPTY_LABEL_MARKER}
        samples.append(
            Sample(
                id=sample_id,
                source_path=_cell(row.get("source_path")) or f"{sample_id}.wav",
                sha256=_cell(row.get("sha256")),
                labels=labels,
                transcripts=transcripts,
            )
        )
    return samples

def enrich_audio_from_manifest(samples: list[Sample], path: Path) -> int:
    """Copy audio / sha256 from a same-id ASR parquet.

    Overwrites sha256 with the inference-side hash so eval_aggregate id+hash
    checks stay consistent when summary.xlsx hashes drifted from the ASR run.
    """
    incoming = {sample.id: sample for sample in Manifest.load(path)}
    filled = 0
    for sample in samples:
        other = incoming.get(sample.id)
        if other is None:
            continue
        changed = False
        if other.audio:
            sample.audio = dict(other.audio)
            changed = True
        if other.sha256:
            sample.sha256 = other.sha256
            changed = True
        if other.source_path and (
            not sample.source_path or sample.source_path == f"{sample.id}.wav"
        ):
            sample.source_path = other.source_path
            changed = True
        if changed:
            filled += 1
    return filled


def ensure_transcript_alias(path: Path, alias: str, scratch_dir: Path) -> Path:
    """Return a parquet whose transcript key == alias (rewrite to scratch if needed)."""
    manifest = Manifest.load(path)
    if not manifest.samples:
        raise SystemExit(f"[ERROR] 空 Manifest: {path}")
    keys = sorted({key for sample in manifest.samples for key in sample.transcripts})
    if alias in keys:
        # Prefer exact alias; if only that key exists, use original file.
        only_alias = keys == [alias]
        if only_alias:
            return path
        # Multiple keys including alias — still OK for aggregate (reads alias key).
        return path

    if len(keys) == 1:
        src_key = keys[0]
        print(f"[INFO] 将 {path.name} 的 transcript 键 {src_key!r} 重命名为 {alias!r}")
        rewritten: list[Sample] = []
        for sample in manifest.samples:
            copied = sample.model_copy(deep=True)
            entry = copied.transcripts.pop(src_key, None)
            if entry is None:
                rewritten.append(copied)
                continue
            if isinstance(entry, dict):
                copied.transcripts[alias] = dict(entry)
            else:
                copied.transcripts[alias] = {"text": str(entry)}
            rewritten.append(copied)
        out = scratch_dir / f"{alias}_asr_rewritten.parquet"
        Manifest(rewritten).save(out)
        return out

    raise SystemExit(
        f"[ERROR] {path} 的 transcript keys={keys} 不含别名 {alias!r}，"
        f"且无法自动重命名（多于 1 个键）。请先改 parquet 或换 --model 别名。"
    )


def run_cli(args: list[str]) -> None:
    cmd = [sys.executable, "-m", "audio_engine.cli.main", *args]
    print("[CMD]", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="summary.xlsx + 已有 ASR parquet → 解耦评测（aggregate + 字准）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--summary",
        type=Path,
        required=True,
        help="手工修订后的 summary.xlsx（需含 sample_id/type/label）",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        dest="models",
        help="推理结果 alias=path.parquet（可重复），如 qwen-sft-e10=.../xxx.parquet",
    )
    parser.add_argument(
        "--eval-name",
        default="eval_local_test",
        help="评测集 stem（默认 eval_local_test）",
    )
    parser.add_argument(
        "--eval-model",
        action="append",
        default=[],
        dest="eval_models",
        help="报告维度 transcript 键（可重复）。默认=summary 里的 qwen1 + 全部 --model 别名",
    )
    parser.add_argument(
        "--audio-from",
        type=Path,
        default=None,
        help="可选：用该 parquet 按 id 补齐 resampled_16k（默认用第一个 --model）",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="覆盖已有 eval 集 / 中间产物，并强制重跑 metric",
    )
    parser.add_argument(
        "--skip-metric",
        action="store_true",
        help="只做注册 + aggregate，不算字准",
    )
    parser.add_argument(
        "--min-gold-ratio",
        type=float,
        default=0.0,
        help="评测集金标覆盖下限（传给 eval check / register）",
    )
    args = parser.parse_args()

    if not args.models:
        raise SystemExit("[ERROR] 至少提供一个 --model alias=path.parquet")

    eval_name = validate_source_name(args.eval_name)
    summary_path = args.summary if args.summary.is_absolute() else (ROOT / args.summary)
    summary_path = summary_path.resolve()
    model_specs = [_parse_model_arg(item) for item in args.models]

    frame = _load_summary_xlsx(summary_path)
    samples = summary_to_samples(frame)
    marked = apply_placeholder_markers(samples)
    print(f"[INFO] summary → {len(samples)} samples from {summary_path}")
    if marked.get("noise"):
        print(f"[INFO] noise 转写/金标已写成 {_EMPTY_LABEL_MARKER!r}: {marked['noise']}")
    if marked.get("review_queue_empty_label"):
        print(
            f"[INFO] review_queue 空 label → {_EMPTY_LABEL_MARKER!r}: "
            f"{marked['review_queue_empty_label']}"
        )

    # 同步写回 summary.xlsx：noise 全量 + review_queue 空 label
    type_col = "type" if "type" in frame.columns else "classification_bucket"
    type_series = frame[type_col].map(_cell)
    label_series = frame["label"].map(_cell) if "label" in frame.columns else pd.Series([""] * len(frame))
    gold_series = (
        frame["gold_text"].map(_cell) if "gold_text" in frame.columns else pd.Series([""] * len(frame))
    )
    noise_mask = type_series.eq("noise")
    empty_rq_mask = type_series.eq("review_queue") & (
        (label_series.eq("") & gold_series.eq(""))
        | label_series.eq(_EMPTY_LABEL_MARKER)
        | gold_series.eq(_EMPTY_LABEL_MARKER)
    )
    write_mask = noise_mask | empty_rq_mask
    if write_mask.any():
        frame.loc[write_mask, "label"] = _EMPTY_LABEL_MARKER
        if "gold_text" in frame.columns:
            frame.loc[write_mask, "gold_text"] = _EMPTY_LABEL_MARKER
        for col in frame.columns:
            name = str(col)
            if not name.endswith("_text") or name in _META_COLS or name.startswith("_"):
                continue
            frame.loc[write_mask, col] = _EMPTY_LABEL_MARKER
        frame.drop(columns=["_id"], errors="ignore").to_excel(summary_path, index=False)
        print(
            f"[OK] 已更新 summary：noise={int(noise_mask.sum())}, "
            f"review_queue空label={int(empty_rq_mask.sum())} → {summary_path}"
        )
    audio_src = args.audio_from
    if audio_src is None:
        audio_src = model_specs[0][1]
    else:
        audio_src = audio_src if audio_src.is_absolute() else (ROOT / audio_src)
        audio_src = audio_src.resolve()
    filled = enrich_audio_from_manifest(samples, audio_src)
    print(f"[INFO] 从 {audio_src.name} 补齐音频元数据: {filled}/{len(samples)}")

    with_gold = sum(1 for s in samples if str(s.labels.get("gold_text") or "").strip())
    print(f"[INFO] 有金标: {with_gold}/{len(samples)} ({with_gold / max(len(samples), 1):.2%})")
    type_counts: dict[str, int] = {}
    for sample in samples:
        key = str(sample.labels.get("type") or "unclassified")
        type_counts[key] = type_counts.get(key, 0) + 1
    print(f"[INFO] type 分布: {dict(sorted(type_counts.items()))}")

    manifests_dir = ROOT / "datasets" / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    eval_path = manifests_dir / f"{eval_name}.parquet"
    if eval_path.exists() and not args.force:
        raise SystemExit(f"[ERROR] 评测集已存在: {eval_path}（加 --force 覆盖）")
    Manifest(samples).save(eval_path)
    Manifest(samples).save(eval_path.with_suffix(".jsonl"))
    print(f"[OK] 写出评测集 {eval_path}")

    scratch = ROOT / "runs" / f"_eval_from_summary_{eval_name}"
    if scratch.exists() and args.force:
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True, exist_ok=True)

    join_args: list[str] = []
    aliases: list[str] = []
    for alias, path in model_specs:
        ready = ensure_transcript_alias(path, alias, scratch)
        join_args += ["--join-manifest", f"{alias}={ready}"]
        aliases.append(alias)
        print(f"[INFO] join {alias} ← {ready}")

    # Default report dims: qwen1 (if present in summary) + all joined aliases.
    eval_models = [validate_asr_run(item) for item in args.eval_models]
    if not eval_models:
        present = sorted({key for sample in samples for key in sample.transcripts})
        if "qwen1" in present:
            eval_models.append("qwen1")
        elif "qwen" in present:
            eval_models.append("qwen")
        for alias in aliases:
            if alias not in eval_models:
                eval_models.append(alias)
    print(f"[INFO] --eval-model: {eval_models}")

    check_args = [
        "eval",
        "check",
        str(eval_path),
        "--min-gold-ratio",
        str(args.min_gold_ratio),
    ]
    if any("resampled_16k" in s.audio for s in samples):
        check_args += ["--require-audio-key", "resampled_16k"]
    else:
        check_args += ["--require-audio-key", ""]
    run_cli(check_args)

    aggregate_cmd = [
        "pipeline",
        "run",
        "pipelines/eval_aggregate.yaml",
        "--eval-name",
        eval_name,
        *join_args,
    ]
    if args.force:
        aggregate_cmd.append("--force")
    run_cli(aggregate_cmd)

    # join 后：noise +「空 label 已占位」的 review_queue，转写统一写成占位符
    agg_path = manifests_dir / f"eval_aggregate_{eval_name}.parquet"
    agg = Manifest.load(agg_path)
    marked_agg = apply_placeholder_markers(agg.samples)
    Manifest(agg.samples).save(agg_path)
    Manifest(agg.samples).save(agg_path.with_suffix(".jsonl"))
    print(
        f"[OK] aggregate 占位已写 {_EMPTY_LABEL_MARKER!r}: "
        f"noise={marked_agg.get('noise', 0)}, "
        f"review_queue空label={marked_agg.get('review_queue_empty_label', 0)} → {agg_path}"
    )

    if args.skip_metric:
        print("[OK] 已跳过 metric（--skip-metric）")
        return 0
    metric_cmd = [
        "pipeline",
        "run",
        "pipelines/eval_metric_pipeline.yaml",
        "--eval-name",
        eval_name,
    ]
    for model in eval_models:
        metric_cmd += ["--eval-model", model]
    if args.force:
        metric_cmd.append("--force")
    run_cli(metric_cmd)

    metrics_path = manifests_dir / f"eval_metrics_{eval_name}.parquet"
    print("\n[OK] 评测完成")
    print(f"  eval set:  {eval_path}")
    print(f"  aggregate: {manifests_dir / f'eval_aggregate_{eval_name}.parquet'}")
    print(f"  metrics:   {metrics_path}")
    print("  报告:      见上方 pipeline run 打印的 Run dir → reports/evaluation.xlsx")
    print(
        json.dumps(
            {
                "eval_name": eval_name,
                "samples": len(samples),
                "with_gold": with_gold,
                "models": aliases,
                "eval_models": eval_models,
                "type_counts": type_counts,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
