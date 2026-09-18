#!/usr/bin/env python3
"""金标 xlsx 上按 type 统计「字准率=1」整句正确率 + 总字准率。

评测口径：
  - 测试集 = xlsx 中有金标的行（默认 label/gold_text 非空）
  - 字准率 = 1 - CER（zh_asr_v1 归一化后编辑距离 / max(ref_len, 1)）
  - 字准率 == 1 → 该条计为正确
  - 按原本 type/category 汇总：正确数 / 该类别条数
  - 汇总另附总字准率 = 1 - sum(edit) / sum(ref_len)；不含平均字准/错字分解
  - 汇总另附相对 qwen-base（可用 --base-model 改）的正确率 / 总字准率差值

默认评测 5 个模型（列名可自动匹配 fenshen / mt3000 命名）：
  - qwen全参数训练10epoch
  - qwen全参数训练100epoch
  - qwen-lora-audio-thinker-100epoch
  - qwen-lora-thinker-100epoch
  - qwen-base

Example（表内已有转写列）::

  python scripts/eval_gold_exact_by_type.py \\
    --xlsx datasets/stage3/reports/eval_fenshen_lora_ep100/evaluation.xlsx \\
    --output tmp/exact_acc_by_type.xlsx

Example（缺列时用 parquet 补齐）::

  python scripts/eval_gold_exact_by_type.py \\
    --xlsx tmp/0909/evaluation-12347.xlsx \\
    --model qwen-lora-thinker-100epoch=datasets/stage1/asr/qwen_asr_lora_thinker_ep100.parquet \\
    --model qwen-lora-audio-thinker-100epoch=datasets/stage1/asr/qwen_asr_lora_audio_thinker_ep100.parquet \\
    --output tmp/exact_acc_by_type.xlsx
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audio_engine.core.manifest import Manifest  # noqa: E402
from audio_engine.core.transcript_reconcile import levenshtein_ops  # noqa: E402

# display_name → 候选 xlsx 列名（按优先级；也可被 --model 覆盖）
DEFAULT_MODELS: list[tuple[str, tuple[str, ...]]] = [
    (
        "qwen全参数训练10epoch",
        (
            "qwen-sft-e10_text",
            "qwen-sft-e10",
            "qwen3-asr-sft-e10_text",
            "qwen3-asr-sft-e10",
            "qwen-sft-epoch10_text",
        ),
    ),
    (
        "qwen全参数训练100epoch",
        (
            "qwen-sft-e100_text",
            "qwen-sft-e100",
            "qwen3-asr-sft-e100_text",
            "qwen3-asr-sft-e100",
            "qwen-sft-epoch100_text",
        ),
    ),
    (
        "qwen-lora-audio-thinker-100epoch",
        (
            "qwen-lora-audio-thinker-ep100_text",
            "qwen-lora-audio-thinker-ep100",
            "qwen-lora-audio-ep100_text",
            "qwen-lora-audio-ep100",
            "qwen-lora-audio-thinker-100epoch_text",
        ),
    ),
    (
        "qwen-lora-thinker-100epoch",
        (
            "qwen-lora-thinker-ep100_text",
            "qwen-lora-thinker-ep100",
            "qwen-lora-thinker-100epoch_text",
        ),
    ),
    (
        "qwen-base",
        (
            "qwen1_text",
            "qwen1",
            "qwen3-asr_text",
            "qwen3-asr",
            "qwen-base_text",
            "qwen_text",
            "qwen",
        ),
    ),
]

_GOLD_COLS = ("label", "gold_text", "label_text_raw", "金标")
_TYPE_COLS = ("type", "category", "classification_bucket", "类型")
_ID_COLS = ("id", "sample_id", "音频id")
_SKIP_IDS = {"总体统计", "汇总", "total", "summary", ""}


@dataclass(frozen=True)
class ModelSource:
    display: str
    kind: str  # "column" | "parquet"
    value: str  # column name or parquet path


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


def _truthy(value: Any) -> bool | None:
    text = _cell(value).lower()
    if not text:
        return None
    if text in {"1", "true", "yes", "y", "是"}:
        return True
    if text in {"0", "false", "no", "n", "否"}:
        return False
    return None


def _pick_col(columns: list[str], candidates: tuple[str, ...], label: str) -> str:
    for name in candidates:
        if name in columns:
            return name
    raise SystemExit(f"[ERROR] 无法识别{label}列；候选={list(candidates)}；可用列={columns}")


def _row_metrics(ref: str, hyp: str) -> tuple[float, int, int, int]:
    """Return (字准率, exact_correct, edit_distance, ref_len)."""
    ops = levenshtein_ops(ref, hyp)
    cer = ops["cer"]
    acc = 0.0 if cer is None else round(max(0.0, 1.0 - float(cer)), 6)
    return acc, int(acc == 1.0), int(ops["total"] or 0), int(ops["ref_len"] or 0)


def _overall_acc(dis: int, ref_len: int, n: int) -> float | None:
    """Corpus 总字准率 = 1 - sum(edit) / sum(ref_len)."""
    if n <= 0:
        return None
    if ref_len <= 0:
        return 1.0 if dis == 0 else 0.0
    return round(max(0.0, 1.0 - dis / ref_len), 6)


def _delta(value: float | None, base: float | None) -> float | None:
    if value is None or base is None:
        return None
    return round(float(value) - float(base), 6)


def _resolve_base_display(models: list[ModelSource], requested: str | None) -> str | None:
    if requested:
        name = requested.strip()
        if any(m.display == name for m in models):
            return name
        raise SystemExit(f"[ERROR] --base-model {requested!r} 不在已解析模型中")
    for candidate in ("qwen-base", "qwen1", "qwen3-asr"):
        if any(m.display == candidate for m in models):
            return candidate
    return models[0].display if models else None


def _load_parquet_texts(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for sample in Manifest.load(path):
        text = None
        if len(sample.transcripts) == 1:
            text = sample.get_transcript_text(next(iter(sample.transcripts)))
        else:
            for key in sample.transcripts:
                text = sample.get_transcript_text(key)
                if _cell(text):
                    break
        out[sample.id] = _cell(text)
    return out


def _parse_model_arg(raw: str) -> tuple[str, str]:
    text = (raw or "").strip()
    if "=" not in text:
        raise SystemExit(
            f"[ERROR] --model 格式应为 显示名=列名或parquet路径，收到: {raw!r}\n"
            f"        例: --model qwen-base=qwen1_text\n"
            f"            --model qwen-lora-thinker-100epoch=datasets/stage1/asr/xxx.parquet"
        )
    display, _, source = text.partition("=")
    display = display.strip()
    source = source.strip()
    if not display or not source:
        raise SystemExit(f"[ERROR] --model 显示名与来源均不能为空: {raw!r}")
    return display, source


def _resolve_models(
    columns: list[str],
    overrides: list[str],
    *,
    only_defaults: bool,
) -> list[ModelSource]:
    """Resolve display → column or parquet.

    --model 同名显示名会覆盖默认候选；未覆盖的默认模型仍尝试自动匹配列。
    """
    override_map = dict(_parse_model_arg(item) for item in overrides)
    resolved: list[ModelSource] = []
    seen: set[str] = set()

    ordered_names = [name for name, _ in DEFAULT_MODELS]
    for name in override_map:
        if name not in ordered_names:
            ordered_names.append(name)

    for display in ordered_names:
        if only_defaults and display not in {n for n, _ in DEFAULT_MODELS}:
            continue
        if display in seen:
            continue
        source = override_map.get(display)
        if source:
            path = Path(source)
            if not path.is_absolute():
                candidate = (ROOT / path).resolve()
            else:
                candidate = path
            if candidate.suffix.lower() in {".parquet", ".jsonl"} or candidate.is_file():
                if not candidate.is_file():
                    raise SystemExit(f"[ERROR] parquet 不存在: {candidate}")
                resolved.append(ModelSource(display, "parquet", str(candidate)))
            else:
                col = source if source in columns else (f"{source}_text" if f"{source}_text" in columns else "")
                if not col:
                    raise SystemExit(
                        f"[ERROR] 模型 {display!r} 指定列 {source!r} 不在 xlsx；"
                        f"可用列含: {[c for c in columns if not str(c).startswith('vs_')]}"
                    )
                resolved.append(ModelSource(display, "column", col))
            seen.add(display)
            continue

        # auto from defaults
        candidates = next((c for n, c in DEFAULT_MODELS if n == display), ())
        col = next((c for c in candidates if c in columns), None)
        if col is None:
            print(f"[WARN] 跳过模型 {display!r}：xlsx 无匹配列，且未用 --model 指定", file=sys.stderr)
            continue
        resolved.append(ModelSource(display, "column", col))
        seen.add(display)

    if not resolved:
        raise SystemExit("[ERROR] 未解析到任何模型列/parquet；请检查 xlsx 列名或传入 --model")
    return resolved


def _has_gold_row(row: pd.Series, gold_col: str) -> bool:
    if "has_gold" in row.index:
        flag = _truthy(row.get("has_gold"))
        if flag is False:
            return False
        if flag is True:
            return True
    return bool(_cell(row.get(gold_col)))


def run(args: argparse.Namespace) -> Path:
    xlsx = args.xlsx if args.xlsx.is_absolute() else (ROOT / args.xlsx)
    xlsx = xlsx.resolve()
    if not xlsx.is_file():
        raise SystemExit(f"[ERROR] xlsx 不存在: {xlsx}")

    with pd.ExcelFile(xlsx) as xf:
        sheet = args.sheet if args.sheet is not None else 0
        frame = pd.read_excel(xf, sheet_name=sheet, dtype=str).fillna("")
    columns = [str(c) for c in frame.columns]
    id_col = _pick_col(columns, _ID_COLS if not args.id_col else (args.id_col,), "id")
    type_col = _pick_col(columns, _TYPE_COLS if not args.type_col else (args.type_col,), "type")
    gold_col = _pick_col(columns, _GOLD_COLS if not args.gold_col else (args.gold_col,), "金标")

    models = _resolve_models(columns, args.models, only_defaults=not args.models)
    parquet_cache: dict[str, dict[str, str]] = {}
    for model in models:
        if model.kind == "parquet" and model.value not in parquet_cache:
            print(f"[INFO] 加载 parquet: {model.display} ← {model.value}")
            parquet_cache[model.value] = _load_parquet_texts(Path(model.value))

    detail_rows: list[dict[str, Any]] = []
    # model -> type -> [correct, total, dis, ref_len]
    stats: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0, 0, 0]))
    skipped = {"empty_id": 0, "no_gold": 0, "dup_id": 0}
    seen_ids: set[str] = set()

    for _, row in frame.iterrows():
        sample_id = _cell(row.get(id_col))
        if not sample_id or sample_id in _SKIP_IDS:
            skipped["empty_id"] += 1
            continue
        if sample_id in seen_ids:
            skipped["dup_id"] += 1
            continue
        seen_ids.add(sample_id)
        if not _has_gold_row(row, gold_col):
            skipped["no_gold"] += 1
            continue

        gold = _cell(row.get(gold_col))
        bucket = _cell(row.get(type_col)) or "unclassified"
        detail: dict[str, Any] = {
            "id": sample_id,
            "type": bucket,
            "gold": gold,
        }

        for model in models:
            if model.kind == "column":
                hyp = _cell(row.get(model.value))
            else:
                hyp = parquet_cache[model.value].get(sample_id, "")
            _acc, correct, dis, ref_len = _row_metrics(gold, hyp)
            detail[f"{model.display}_text"] = hyp
            detail[f"{model.display}_正确"] = correct
            for key in (bucket, "__ALL__"):
                stats[model.display][key][0] += correct
                stats[model.display][key][1] += 1
                stats[model.display][key][2] += dis
                stats[model.display][key][3] += ref_len

        detail_rows.append(detail)

    if not detail_rows:
        raise SystemExit("[ERROR] 过滤后无样本（检查金标列 / has_gold）")

    types = sorted(
        {r["type"] for r in detail_rows},
        key=lambda x: (x == "unclassified", str(x)),
    )
    base_display = _resolve_base_display(models, args.base_model)
    base_by_bucket: dict[str, dict[str, float | None]] = {}
    if base_display:
        for bucket in ["__ALL__", *types]:
            correct, total, dis, ref_len = stats[base_display][bucket]
            rate = round(correct / total, 6) if total else None
            tot_acc = _overall_acc(dis, ref_len, total)
            base_by_bucket[bucket] = {"正确率": rate, "总字准率": tot_acc}

    summary_rows: list[dict[str, Any]] = []
    for model in models:
        for bucket in ["__ALL__", *types]:
            correct, total, dis, ref_len = stats[model.display][bucket]
            rate = round(correct / total, 6) if total else None
            tot_acc = _overall_acc(dis, ref_len, total)
            base = base_by_bucket.get(bucket, {})
            is_base = model.display == base_display
            d_rate = None if is_base else _delta(rate, base.get("正确率"))
            d_acc = None if is_base else _delta(tot_acc, base.get("总字准率"))
            summary_rows.append(
                {
                    "模型": model.display,
                    "来源": f"{model.kind}:{model.value}",
                    "type": "总计" if bucket == "__ALL__" else bucket,
                    "样本数": total,
                    "正确数(字准=1)": correct,
                    "正确率": rate,
                    "正确率%": None if rate is None else round(rate * 100, 2),
                    "总字准率": tot_acc,
                    "总字准率%": None if tot_acc is None else round(tot_acc * 100, 2),
                    "相对base正确率": d_rate,
                    "相对base正确率%点": None if d_rate is None else round(d_rate * 100, 2),
                    "相对base总字准率": d_acc,
                    "相对base总字准率%点": None if d_acc is None else round(d_acc * 100, 2),
                }
            )

    # type × model 透视：正确率% + 总字准率% + 相对 base 差值（不含平均字准/错字分解）
    pivot_rows: list[dict[str, Any]] = []
    for bucket in ["总计", *types]:
        key = "__ALL__" if bucket == "总计" else bucket
        row: dict[str, Any] = {
            "type": bucket,
            "样本数": stats[models[0].display][key][1],
            "base": base_display,
        }
        base = base_by_bucket.get(key, {})
        for model in models:
            correct, total, dis, ref_len = stats[model.display][key]
            rate = round(correct / total, 6) if total else None
            tot_acc = _overall_acc(dis, ref_len, total)
            is_base = model.display == base_display
            d_rate = None if is_base else _delta(rate, base.get("正确率"))
            d_acc = None if is_base else _delta(tot_acc, base.get("总字准率"))
            row[f"{model.display}_正确率%"] = (
                None if rate is None else round(rate * 100, 2)
            )
            row[f"{model.display}_总字准率%"] = (
                None if tot_acc is None else round(tot_acc * 100, 2)
            )
            row[f"{model.display}_Δ正确率%点"] = (
                None if d_rate is None else round(d_rate * 100, 2)
            )
            row[f"{model.display}_Δ总字准率%点"] = (
                None if d_acc is None else round(d_acc * 100, 2)
            )
        pivot_rows.append(row)

    output = args.output
    if output is None:
        output = xlsx.with_name(f"{xlsx.stem}_exact_by_type.xlsx")
    elif not output.is_absolute():
        output = (ROOT / output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        pd.DataFrame(detail_rows).to_excel(writer, index=False, sheet_name="明细")
        pd.DataFrame(summary_rows).to_excel(writer, index=False, sheet_name="按type正确率")
        pd.DataFrame(pivot_rows).to_excel(writer, index=False, sheet_name="透视对比")

    print(f"[OK] 有金标样本: {len(detail_rows):,}")
    print(f"[OK] 跳过: {skipped}")
    print(f"[OK] 模型: {[m.display for m in models]}")
    print(f"[OK] base: {base_display}")
    print("-" * 72)
    base_all = base_by_bucket.get("__ALL__", {})
    for model in models:
        correct, total, dis, ref_len = stats[model.display]["__ALL__"]
        pct = 100.0 * correct / total if total else 0.0
        tot = _overall_acc(dis, ref_len, total) or 0.0
        d_rate = _delta(round(correct / total, 6) if total else None, base_all.get("正确率"))
        d_acc = _delta(tot if total else None, base_all.get("总字准率"))
        extra = ""
        if model.display != base_display and d_rate is not None and d_acc is not None:
            extra = f" | Δ正确率 {d_rate * 100:+.2f}%点 | Δ总字准率 {d_acc * 100:+.2f}%点"
        print(
            f"  {model.display}: 正确率 {correct}/{total} = {pct:.2f}% | "
            f"总字准率 {tot * 100:.2f}%{extra}"
        )
        for bucket in types:
            c, t, d, r = stats[model.display][bucket]
            p = 100.0 * c / t if t else 0.0
            ta = _overall_acc(d, r, t) or 0.0
            print(f"    - {bucket}: 正确率 {p:.2f}% | 总字准率 {ta * 100:.2f}%")
    print("-" * 72)
    print(f"[OK] 已写入: {output}")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="金标 xlsx：按 type 统计字准率=1 的整句正确率 + 总字准率 + 相对base差值"
    )
    parser.add_argument("--xlsx", type=Path, required=True, help="带金标与 type 的评测表")
    parser.add_argument("--sheet", default=None, help="sheet 名或序号，默认第一张")
    parser.add_argument("--id-col", default=None)
    parser.add_argument("--type-col", default=None)
    parser.add_argument("--gold-col", default=None)
    parser.add_argument(
        "--base-model",
        default="qwen-base",
        help="相对差值基准模型显示名（默认 qwen-base）",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        dest="models",
        help="显示名=xlsx列名 或 显示名=parquet路径；可重复。同名覆盖默认列匹配",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="输出 xlsx（默认: 输入名_exact_by_type.xlsx）",
    )
    args = parser.parse_args(argv)
    if args.sheet is not None:
        try:
            args.sheet = int(args.sheet)
        except ValueError:
            pass
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
