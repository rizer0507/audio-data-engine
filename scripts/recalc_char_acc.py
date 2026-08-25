#!/usr/bin/env python3
"""按 label_text_raw 重算字准率，并单独比较 qwen_sft vs qwen。

基准：
  - label_text_raw → qwen_text / sensevoice_text / qwen_sft_text
  - qwen_text → qwen_sft_text（单独一对）

字准率 = 1 - cer，cer = 编辑距离 / max(len(ref), 1)
文本归一化与流水线一致（NFKC、去标点、去 SenseVoice 控制标签）。

Example:
  python scripts/recalc_char_acc.py \\
    --input 数据集/3000-比对.xlsx \\
    --output 数据集/3000-比对-字准率.xlsx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audio_engine.core.transcript_reconcile import (  # noqa: E402
    levenshtein_ops,
    normalize_transcript,
)

REF_COL = "label_text_raw"
HYP_COLS = (
    ("qwen_text", "vs_label_qwen"),
    ("sensevoice_text", "vs_label_sensevoice"),
    ("qwen_sft_text", "vs_label_qwen_sft"),
)
PAIR_COLS = ("qwen_text", "qwen_sft_text", "vs_qwen_qwen_sft")


def _as_text(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none"}:
        return ""
    return text


def _metrics(ref: str, hyp: str) -> dict[str, float | int | None]:
    ops = levenshtein_ops(ref, hyp)
    cer = ops["cer"]
    if cer is None:
        acc = None
    else:
        # 编辑距离可大于基准长度，字准率截断到 [0, 1]
        acc = round(max(0.0, 1.0 - float(cer)), 6)
    return {
        "total": ops["total"],
        "错字": ops["错字"],
        "少字": ops["少字"],
        "多字": ops["多字"],
        "cer": cer,
        "字准率": acc,
        "ref_len": ops["ref_len"],
        "hyp_len": ops["hyp_len"],
    }


def _empty_metrics() -> dict[str, float | int | None]:
    return {
        "total": None,
        "错字": None,
        "少字": None,
        "多字": None,
        "cer": None,
        "字准率": None,
        "ref_len": None,
        "hyp_len": None,
    }


def _attach_pair(
    df: pd.DataFrame,
    ref_col: str,
    hyp_col: str,
    prefix: str,
    *,
    require_ref: bool = True,
) -> dict[str, float | int]:
    """逐行写入 prefix_* 列，返回总体汇总（按字符加权）。

    require_ref=True 时：基准为空则该行指标留空，且不计入总体字准率。
    require_ref=False 时：两端皆空记字准率 1.0；仅用于模型间互比。
    """
    totals = {
        "dis": 0,
        "ref_len": 0,
        "n": 0,
        "n_skip": 0,
        "sum_acc": 0.0,
    }
    rows: list[dict[str, float | int | None]] = []

    for _, row in df.iterrows():
        ref_raw = _as_text(row.get(ref_col))
        hyp_raw = _as_text(row.get(hyp_col))
        ref = normalize_transcript(ref_raw)

        if require_ref and not ref:
            rows.append(_empty_metrics())
            totals["n_skip"] += 1
            continue

        if not ref_raw and not hyp_raw:
            rows.append(
                {
                    "total": 0,
                    "错字": 0,
                    "少字": 0,
                    "多字": 0,
                    "cer": 0.0,
                    "字准率": 1.0,
                    "ref_len": 0,
                    "hyp_len": 0,
                }
            )
            totals["n"] += 1
            totals["sum_acc"] += 1.0
            continue

        m = _metrics(ref_raw, hyp_raw)
        rows.append(m)
        totals["n"] += 1
        totals["dis"] += int(m["total"] or 0)
        totals["ref_len"] += int(m["ref_len"] or 0)
        totals["sum_acc"] += float(m["字准率"] or 0.0)

    for key in ("total", "错字", "少字", "多字", "cer", "字准率", "ref_len", "hyp_len"):
        df[f"{prefix}_{key}"] = [r[key] for r in rows]

    return totals


def _overall_acc(totals: dict[str, float | int]) -> float | None:
    """字符加权总体字准率，截断到 [0, 1]。"""
    ref_len = int(totals["ref_len"])
    if ref_len <= 0:
        return None
    return round(max(0.0, 1.0 - int(totals["dis"]) / ref_len), 6)


def _mean_acc(totals: dict[str, float | int]) -> float | None:
    n = int(totals["n"])
    if n <= 0:
        return None
    return round(float(totals["sum_acc"]) / n, 6)


def _print_summary(name: str, totals: dict[str, float | int]) -> None:
    overall = _overall_acc(totals)
    mean = _mean_acc(totals)

    def _fmt(v: float | None) -> str:
        return "N/A" if v is None else f"{v:.4f} ({v * 100:.2f}%)"

    print(f"  [{name}]")
    print(f"    参与行数:     {totals['n']:,}")
    print(f"    跳过行数:     {totals['n_skip']:,}  (基准为空)")
    print(f"    总编辑距离:   {totals['dis']:,}")
    print(f"    总基准字数:   {totals['ref_len']:,}")
    print(f"    总体字准率:   {_fmt(overall)}  (1 - sum(dis)/sum(ref_len), ≥0)")
    print(f"    平均字准率:   {_fmt(mean)}  (逐行均值)")


def main() -> int:
    parser = argparse.ArgumentParser(description="按 label_text_raw 重算字准率")
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=ROOT / "数据集" / "3000-比对.xlsx",
        help="输入 xlsx（默认: 数据集/3000-比对.xlsx）",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="输出 xlsx（默认: 输入文件名加 -字准率）",
    )
    args = parser.parse_args()

    input_path: Path = args.input
    if not input_path.exists():
        print(f"[ERROR] 输入不存在: {input_path}", file=sys.stderr)
        return 1

    output_path: Path = args.output or input_path.with_name(
        f"{input_path.stem}-字准率{input_path.suffix}"
    )

    print(f"[INFO] 读取: {input_path}")
    df = pd.read_excel(input_path)
    missing = [c for c, _ in HYP_COLS if c not in df.columns]
    if REF_COL not in df.columns:
        missing.append(REF_COL)
    if missing:
        print(f"[ERROR] 缺少列: {missing}", file=sys.stderr)
        print(f"        现有列: {list(df.columns)}", file=sys.stderr)
        return 1

    # 兼容旧列名写法 qwen-sft_text
    if "qwen-sft_text" in df.columns and "qwen_sft_text" not in df.columns:
        df = df.rename(columns={"qwen-sft_text": "qwen_sft_text"})

    print(f"[INFO] 行数: {len(df):,}")
    print("[INFO] 计算 vs label_text_raw ...")
    print("=" * 56)

    summaries: list[tuple[str, dict[str, float | int]]] = []
    for hyp_col, prefix in HYP_COLS:
        totals = _attach_pair(df, REF_COL, hyp_col, prefix, require_ref=True)
        summaries.append((f"{hyp_col} ← {REF_COL}", totals))
        _print_summary(f"{hyp_col} ← {REF_COL}", totals)

    print("-" * 56)
    print("[INFO] 单独计算 qwen_sft_text ← qwen_text ...")
    ref_col, hyp_col, prefix = PAIR_COLS
    pair_totals = _attach_pair(
        df, ref_col, hyp_col, prefix, require_ref=False
    )
    summaries.append((f"{hyp_col} ← {ref_col}", pair_totals))
    _print_summary(f"{hyp_col} ← {ref_col}", pair_totals)
    print("=" * 56)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="结果")

        summary_rows = []
        for name, totals in summaries:
            summary_rows.append(
                {
                    "对比": name,
                    "参与行数": totals["n"],
                    "跳过行数": totals["n_skip"],
                    "总编辑距离": totals["dis"],
                    "总基准字数": totals["ref_len"],
                    "总体字准率": _overall_acc(totals),
                    "平均字准率": _mean_acc(totals),
                }
            )
        pd.DataFrame(summary_rows).to_excel(writer, index=False, sheet_name="统计摘要")

    print(f"[INFO] 已写入: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
