#!/usr/bin/env python3
"""【过渡旁路】从人工标注 xlsx + 多模型 ASR parquet 构建评测 summary 并跑字准流程。

正式主路径请用工序一拆分入口（产出 classified_* 再 eval register）::

  audio-data pipeline run pipelines/classify_external_gold.yaml \\
    --source-name "$BATCH" --aggregate-base qwen3-asr \\
    --external-gold path/to/金标.xlsx --label-col label_text_raw

详见 docs/04-改进需求/已完成/006-工序一清洗引擎拆分需求.md。

清洗口径：
  - 去【…】（含括号内文字）+ zh_asr_v1 去标点/空白
  - 整句精确命中热词则清空（与 blank_exact 一致）

type 规则（相对 base 模型清洗后文本 vs label）：
  - 命中语音信箱正则 → voicemail
  - 双方皆空 → noise（字准按占位符计为 1）
  - 完全一致 → auto_gold（项目原自动金标桶）
  - 否则 → hardcase

Example::

  python scripts/run_eval_from_label_xlsx.py \\
    --label-xlsx tmp/0904/AI面谈提示词原版测试字准率.xlsx \\
    --label-col label_text_raw \\
    --base-model qwen3-asr=datasets/manifests/qwen3-asr_asr_mt3000.parquet \\
    --model qwen3-asr-sft-e10=datasets/manifests/qwen3-asr-sft-e10_asr_mt3000.parquet \\
    --model qwen3-asr-sft-e100=datasets/manifests/qwen3-asr-sft-e100_asr_mt3000.parquet \\
    --eval-name eval_interview_mt3000 \\
    --force
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audio_engine.core.manifest import Manifest  # noqa: E402
from audio_engine.core.source_naming import validate_asr_run, validate_source_name  # noqa: E402
from audio_engine.core.transcript_reconcile import (  # noqa: E402
    parse_vocabulary_hotwords,
    plain_transcript_text,
    resolve_blank_exact_hotwords,
)

# Reuse helpers from the summary-based eval entrypoint.
import importlib.util  # noqa: E402

_SUMMARY_HELPERS = ROOT / "scripts" / "run_eval_from_summary.py"
_spec = importlib.util.spec_from_file_location("run_eval_from_summary", _SUMMARY_HELPERS)
assert _spec and _spec.loader
_summary_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_summary_mod)
_EMPTY_LABEL_MARKER = _summary_mod._EMPTY_LABEL_MARKER
apply_placeholder_markers = _summary_mod.apply_placeholder_markers
enrich_audio_from_manifest = _summary_mod.enrich_audio_from_manifest
ensure_transcript_alias = _summary_mod.ensure_transcript_alias
run_cli = _summary_mod.run_cli
summary_to_samples = _summary_mod.summary_to_samples

_SUMMARY_SKIP_IDS = {"总体统计", "汇总", "total", "summary"}


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
        raise SystemExit(f"[ERROR] --model/--base-model 格式应为 alias=path.parquet，收到: {raw!r}")
    alias, _, path = text.partition("=")
    alias = validate_asr_run(alias.strip())
    path_obj = Path(path.strip())
    if not path_obj.is_absolute():
        path_obj = (ROOT / path_obj).resolve()
    if not path_obj.is_file():
        raise SystemExit(f"[ERROR] parquet 不存在: {path_obj}")
    return alias, path_obj


def _load_hotwords(path: Path) -> frozenset[str]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cfg = loaded.get("blank_exact_hotwords", loaded)
    hotwords, _ = resolve_blank_exact_hotwords(cfg)
    return hotwords


def _load_voicemail_re(path: Path) -> re.Pattern[str] | None:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    patterns = [str(item).strip() for item in (raw.get("patterns") or []) if str(item).strip()]
    if not patterns:
        return None
    flags = re.IGNORECASE if "IGNORECASE" in str(raw.get("flags") or "IGNORECASE").upper() else 0
    return re.compile("|".join(f"(?:{p})" for p in patterns), flags)


def _blank_exact(text: str, hotwords: frozenset[str]) -> str:
    plain = plain_transcript_text(text)
    if plain and plain in hotwords:
        return ""
    return plain


def _transcript_map(path: Path, alias: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for sample in Manifest.load(path):
        text = sample.get_transcript_text(alias)
        if text is None and len(sample.transcripts) == 1:
            text = sample.get_transcript_text(next(iter(sample.transcripts)))
        out[sample.id] = _cell(text)
    return out


def build_summary_frame(
    *,
    label_xlsx: Path,
    label_col: str,
    base_alias: str,
    base_texts: dict[str, str],
    hotwords: frozenset[str],
    voicemail_re: re.Pattern[str] | None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    frame = pd.read_excel(label_xlsx, dtype=str).fillna("")
    if label_col not in frame.columns:
        raise SystemExit(f"[ERROR] 缺少列 {label_col!r}；可用列: {list(frame.columns)}")
    id_col = "sample_id" if "sample_id" in frame.columns else "id"
    if id_col not in frame.columns:
        raise SystemExit(f"[ERROR] 缺少 id/sample_id；可用列: {list(frame.columns)}")

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    counts = {
        "xlsx_rows": 0,
        "skipped_summary": 0,
        "dup_id": 0,
        "missing_in_parquet": 0,
        "aligned": 0,
        "auto_gold": 0,
        "hardcase": 0,
        "noise": 0,
        "review_queue": 0,
        "voicemail": 0,
    }
    for _, row in frame.iterrows():
        counts["xlsx_rows"] += 1
        sample_id = _cell(row.get(id_col))
        if not sample_id or sample_id in _SUMMARY_SKIP_IDS:
            counts["skipped_summary"] += 1
            continue
        if sample_id in seen:
            counts["dup_id"] += 1
            continue
        seen.add(sample_id)
        if sample_id not in base_texts:
            counts["missing_in_parquet"] += 1
            continue

        label_raw = _cell(row.get(label_col))
        qwen_raw = base_texts[sample_id]
        gold_plain = _blank_exact(label_raw, hotwords)
        qwen_plain = _blank_exact(qwen_raw, hotwords)

        match_texts = [label_raw, qwen_raw, gold_plain, qwen_plain]
        voicemail_hit = bool(
            voicemail_re and any(voicemail_re.search(t) for t in match_texts if t)
        )

        # 清洗后金标为空、模型有输出：标注侧视为无有效语音，模型幻觉 → review_queue
        # gold 保留原始【…】标注，便于过 skip 检查；CER 时 annotation_brackets 会清空参考。
        if voicemail_hit:
            bucket = "voicemail"
            label_out = gold_plain
            reason = "voicemail_or_phone_assistant"
        elif not gold_plain and not qwen_plain:
            bucket = "noise"
            label_out = _EMPTY_LABEL_MARKER
            reason = "empty_both_after_clean"
        elif not gold_plain and qwen_plain:
            bucket = "review_queue"
            label_out = label_raw or "【无声音输出】"
            reason = "empty_label_model_hallucination"
        elif gold_plain == qwen_plain:
            bucket = "auto_gold"
            label_out = gold_plain
            reason = "qwen_label_exact_match"
        else:
            bucket = "hardcase"
            label_out = gold_plain
            reason = "qwen_label_mismatch"
        counts[bucket] = counts.get(bucket, 0) + 1
        counts["aligned"] += 1

        rows.append(
            {
                "sample_id": sample_id,
                "id": sample_id,
                "type": bucket,
                "classification_bucket": bucket,
                "label": label_out,
                "gold_text": label_out,
                "label_text_raw": label_raw,
                "classification_reason": reason,
            }
        )

    if not rows:
        raise SystemExit("[ERROR] 对齐后无样本（检查 id 是否与 parquet 交集）")
    return pd.DataFrame(rows), counts


def main() -> int:
    print(
        "[WARN] run_eval_from_label_xlsx.py 为过渡旁路；"
        "正式路径请用: audio-data pipeline run pipelines/classify_external_gold.yaml "
        "→ eval register（见 docs/04-改进需求/已完成/006-工序一清洗引擎拆分需求.md）",
        file=sys.stderr,
    )
    parser = argparse.ArgumentParser(
        description="【过渡】人工标注 xlsx + ASR parquet → 清洗/分 type → 多模型字准评测"
    )
    parser.add_argument("--label-xlsx", type=Path, required=True)
    parser.add_argument("--label-col", default="label_text_raw")
    parser.add_argument(
        "--base-model",
        required=True,
        help="用于 type 判定的 base 模型 alias=path.parquet（通常为未 SFT）",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        dest="models",
        help="额外评测模型 alias=path.parquet（可重复）",
    )
    parser.add_argument("--eval-name", default="eval_interview_mt3000")
    parser.add_argument(
        "--hotwords-path",
        type=Path,
        default=ROOT / "configs/normalization/blank_exact_eval_interview_v1.yaml",
    )
    parser.add_argument(
        "--voicemail-patterns-path",
        type=Path,
        default=ROOT / "configs/selection/voicemail_patterns_v1.yaml",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-metric", action="store_true")
    parser.add_argument("--min-gold-ratio", type=float, default=0.0)
    args = parser.parse_args()

    eval_name = validate_source_name(args.eval_name)
    label_xlsx = args.label_xlsx if args.label_xlsx.is_absolute() else (ROOT / args.label_xlsx)
    label_xlsx = label_xlsx.resolve()
    hotwords_path = (
        args.hotwords_path if args.hotwords_path.is_absolute() else (ROOT / args.hotwords_path)
    )
    voicemail_path = (
        args.voicemail_patterns_path
        if args.voicemail_patterns_path.is_absolute()
        else (ROOT / args.voicemail_patterns_path)
    )

    base_alias, base_path = _parse_model_arg(args.base_model)
    extra_models = [_parse_model_arg(item) for item in args.models]
    # base 也参与字准评测
    all_models = [(base_alias, base_path)] + [
        item for item in extra_models if item[0] != base_alias
    ]

    hotwords = _load_hotwords(hotwords_path)
    if not hotwords:
        # 允许直接写 vocabulary 列表
        hotwords = parse_vocabulary_hotwords(
            ["贷款审批", "单位名称", "住址", "手机号", "联系人关系"]
        )
    voicemail_re = _load_voicemail_re(voicemail_path) if voicemail_path.is_file() else None
    base_texts = _transcript_map(base_path, base_alias)

    summary_df, counts = build_summary_frame(
        label_xlsx=label_xlsx,
        label_col=args.label_col,
        base_alias=base_alias,
        base_texts=base_texts,
        hotwords=hotwords,
        voicemail_re=voicemail_re,
    )
    print(
        json.dumps(
            {"hotwords": sorted(hotwords), "counts": counts},
            ensure_ascii=False,
            indent=2,
        )
    )

    export_dir = ROOT / "data" / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    summary_path = export_dir / f"summary_{eval_name}.xlsx"
    summary_df.to_excel(summary_path, index=False)
    print(f"[OK] summary → {summary_path} ({len(summary_df)} rows)")

    summary_df = summary_df.copy()
    summary_df["_id"] = summary_df["sample_id"].map(_cell)
    samples = summary_to_samples(summary_df)
    marked = apply_placeholder_markers(samples)
    print(f"[INFO] placeholder markers: {marked}")

    filled = enrich_audio_from_manifest(samples, base_path)
    print(f"[INFO] audio enrich: {filled}/{len(samples)}")

    manifests_dir = ROOT / "datasets" / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    eval_path = manifests_dir / f"{eval_name}.parquet"
    if eval_path.exists() and not args.force:
        raise SystemExit(f"[ERROR] 评测集已存在: {eval_path}（加 --force 覆盖）")
    Manifest(samples).save(eval_path)
    Manifest(samples).save(eval_path.with_suffix(".jsonl"))
    print(f"[OK] eval set → {eval_path}")

    scratch = ROOT / "runs" / f"_eval_from_label_{eval_name}"
    scratch.mkdir(parents=True, exist_ok=True)

    join_args: list[str] = []
    aliases: list[str] = []
    for alias, path in all_models:
        ready = ensure_transcript_alias(path, alias, scratch)
        join_args += ["--join-manifest", f"{alias}={ready}"]
        aliases.append(alias)
        print(f"[INFO] join {alias} ← {ready}")

    check_args = [
        "eval",
        "check",
        str(eval_path),
        "--min-gold-ratio",
        str(args.min_gold_ratio),
        "--require-audio-key",
        "",
    ]
    if any("resampled_16k" in s.audio for s in samples):
        check_args[-1] = "resampled_16k"
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

    agg_path = manifests_dir / f"eval_aggregate_{eval_name}.parquet"
    agg = Manifest.load(agg_path)
    marked_agg = apply_placeholder_markers(agg.samples)
    Manifest(agg.samples).save(agg_path)
    Manifest(agg.samples).save(agg_path.with_suffix(".jsonl"))
    print(f"[OK] aggregate placeholders: {marked_agg}")

    if args.skip_metric:
        print("[OK] skipped metric")
        return 0

    metric_cmd = [
        "pipeline",
        "run",
        "pipelines/eval_metric_pipeline.yaml",
        "--eval-name",
        eval_name,
    ]
    for alias in aliases:
        metric_cmd += ["--eval-model", alias]
    if args.force:
        metric_cmd.append("--force")
    run_cli(metric_cmd)

    print("\n[OK] 评测完成")
    print(f"  summary:   {summary_path}")
    print(f"  eval set:  {eval_path}")
    print(f"  aggregate: {agg_path}")
    print(f"  metrics:   {manifests_dir / f'eval_metrics_{eval_name}.parquet'}")
    print("  报告:      见上方 pipeline run 的 Run dir → reports/evaluation.xlsx")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
