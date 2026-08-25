#!/usr/bin/env python3
"""按 id 合并 ASR 表与标注表，重算字准率。

从 ASR 表提取: id, source_path, qwen_sft_text, qwen_text, sensevoice_text
从标注表提取: label_text_raw
按 id 内连接后写入一张表，并计算：
  - qwen_sft_text ← qwen_text 整体字准率
  - qwen_sft_text / qwen_text / sensevoice_text ← label_text_raw 字准率

Example:
  python scripts/merge_label_char_acc.py \\
    --asr 数据集/3000-比对.xlsx \\
    --label 数据集/AI面谈提示词原版测试字准率.xlsx \\
    --output 数据集/3000-合并字准率.xlsx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audio_engine.core.transcript_reconcile import levenshtein_ops, normalize_transcript  # noqa: E402

ASR_KEEP = ("id", "source_path", "qwen_sft_text", "qwen_text", "sensevoice_text")
LABEL_KEEP = ("id", "label_text_raw")
SKIP_IDS = {"总体统计"}

VS_LABEL = (
    ("qwen_sft_text", "vs_label_qwen_sft"),
    ("qwen_text", "vs_label_qwen"),
    ("sensevoice_text", "vs_label_sensevoice"),
)
VS_QWEN = ("qwen_text", "qwen_sft_text", "vs_qwen_qwen_sft")


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
    acc = None if cer is None else round(max(0.0, 1.0 - float(cer)), 6)
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
    totals = {"dis": 0, "ref_len": 0, "n": 0, "n_skip": 0, "sum_acc": 0.0}
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
            m = {
                "total": 0,
                "错字": 0,
                "少字": 0,
                "多字": 0,
                "cer": 0.0,
                "字准率": 1.0,
                "ref_len": 0,
                "hyp_len": 0,
            }
            rows.append(m)
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
    ref_len = int(totals["ref_len"])
    if ref_len <= 0:
        return None
    return round(max(0.0, 1.0 - int(totals["dis"]) / ref_len), 6)


def _mean_acc(totals: dict[str, float | int]) -> float | None:
    n = int(totals["n"])
    if n <= 0:
        return None
    return round(float(totals["sum_acc"]) / n, 6)


def _fmt_acc(v: float | None) -> str:
    return "N/A" if v is None else f"{v:.4f} ({v * 100:.2f}%)"


def _print_summary(name: str, totals: dict[str, float | int]) -> None:
    print(f"  [{name}]")
    print(f"    参与行数:     {totals['n']:,}")
    print(f"    跳过行数:     {totals['n_skip']:,}  (基准为空)")
    print(f"    总编辑距离:   {totals['dis']:,}")
    print(f"    总基准字数:   {totals['ref_len']:,}")
    print(f"    总体字准率:   {_fmt_acc(_overall_acc(totals))}  (1 - sum(dis)/sum(ref_len), ≥0)")
    print(f"    平均字准率:   {_fmt_acc(_mean_acc(totals))}  (逐行均值)")


def _load_asr(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    if "qwen-sft_text" in df.columns and "qwen_sft_text" not in df.columns:
        df = df.rename(columns={"qwen-sft_text": "qwen_sft_text"})
    missing = [c for c in ASR_KEEP if c not in df.columns]
    if missing:
        raise ValueError(f"ASR 表缺少列 {missing}，现有列: {list(df.columns)}")
    out = df.loc[:, list(ASR_KEEP)].copy()
    out["id"] = out["id"].astype(str).str.strip()
    out = out[~out["id"].isin(SKIP_IDS)].drop_duplicates(subset=["id"], keep="first")
    return out


def _load_label(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    missing = [c for c in LABEL_KEEP if c not in df.columns]
    if missing:
        raise ValueError(f"标注表缺少列 {missing}，现有列: {list(df.columns)}")
    out = df.loc[:, list(LABEL_KEEP)].copy()
    out["id"] = out["id"].astype(str).str.strip()
    out = out[~out["id"].isin(SKIP_IDS)].drop_duplicates(subset=["id"], keep="first")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="合并 ASR/标注表并计算字准率")
    parser.add_argument(
        "--asr",
        type=Path,
        default=ROOT / "数据集" / "3000-比对.xlsx",
        help="ASR 结果表（默认: 数据集/3000-比对.xlsx）",
    )
    parser.add_argument(
        "--label",
        type=Path,
        default=ROOT / "数据集" / "AI面谈提示词原版测试字准率.xlsx",
        help="标注表（默认: 数据集/AI面谈提示词原版测试字准率.xlsx）",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=ROOT / "数据集" / "3000-合并字准率.xlsx",
        help="输出 xlsx（默认: 数据集/3000-合并字准率.xlsx）",
    )
    args = parser.parse_args()

    for path in (args.asr, args.label):
        if not path.exists():
            print(f"[ERROR] 文件不存在: {path}", file=sys.stderr)
            return 1

    print(f"[INFO] ASR:   {args.asr}")
    print(f"[INFO] 标注:  {args.label}")
    asr = _load_asr(args.asr)
    label = _load_label(args.label)
    print(f"[INFO] ASR 行数: {len(asr):,} | 标注行数: {len(label):,}")

    merged = asr.merge(label, on="id", how="inner")
    only_asr = len(asr) - len(merged)
    only_label = len(set(label["id"]) - set(asr["id"]))
    print(
        f"[INFO] 内连接匹配: {len(merged):,} "
        f"(ASR 未匹配 {only_asr:,}, 标注未匹配 {only_label:,})"
    )
    if merged.empty:
        print("[ERROR] 无匹配 id，终止", file=sys.stderr)
        return 1

    # 输出列顺序：基础字段 + 指标列
    base_cols = list(ASR_KEEP) + ["label_text_raw"]
    df = merged.loc[:, base_cols].copy()

    print("=" * 56)
    print("[INFO] qwen_sft_text ← qwen_text")
    summaries: list[tuple[str, dict[str, float | int]]] = []
    ref_col, hyp_col, prefix = VS_QWEN
    pair_totals = _attach_pair(df, ref_col, hyp_col, prefix, require_ref=False)
    summaries.append((f"{hyp_col} ← {ref_col}", pair_totals))
    _print_summary(f"{hyp_col} ← {ref_col}", pair_totals)

    print("-" * 56)
    print("[INFO] 各模型 ← label_text_raw")
    for hyp_col, prefix in VS_LABEL:
        totals = _attach_pair(df, "label_text_raw", hyp_col, prefix, require_ref=True)
        summaries.append((f"{hyp_col} ← label_text_raw", totals))
        _print_summary(f"{hyp_col} ← label_text_raw", totals)
    print("=" * 56)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(args.output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="结果")
        summary_rows = [
            {
                "对比": name,
                "参与行数": totals["n"],
                "跳过行数": totals["n_skip"],
                "总编辑距离": totals["dis"],
                "总基准字数": totals["ref_len"],
                "总体字准率": _overall_acc(totals),
                "平均字准率": _mean_acc(totals),
            }
            for name, totals in summaries
        ]
        pd.DataFrame(summary_rows).to_excel(writer, index=False, sheet_name="统计摘要")

    print(f"[INFO] 已写入: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
