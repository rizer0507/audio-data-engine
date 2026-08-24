# -*- coding: utf-8 -*-
"""
gold_label_generator.py
=======================
Post-process badcases.xlsx files in qwen_vs_sensevoice directory:

1. Strip all <|...|> tags from sensevoice_text
2. Add a gold_label column:
   - If qwen and sensevoice (normalized) have similarity >= threshold -> gold label (qwen value)
   - If qwen is a pure interjection (e.g. xn/a/o/hng) and sensevoice is effectively empty -> "/"
   - Otherwise: blank (pending manual annotation)
3. Add character accuracy columns (ref=qwen, hyp=sensevoice):
   - 字准率 = 1 - 偏差字符数 / len(qwen)
   - 少了 / 多了 / 错了：编辑距离分解（删除 / 插入 / 替换）
   - 偏差字符数：少了 + 多了 + 错了
4. Output updated Excel files

Usage:
    python scripts/gold_label_generator.py [--threshold 0.9] [--input-dir qwen_vs_sensevoice] [--output-suffix _labeled]
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
from difflib import SequenceMatcher

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
DEFAULT_THRESHOLD = 0.9
INTERJECTIONS = {"\u5514", "\u554a", "\u54e6", "\u54fc"}  # {嗯, 啊, 哦, 哼}
SENSEVOICE_TAG_RE = re.compile(r"<\|[^|]+\|>")
PUNCT_STRIP_RE = re.compile(r"[^\w\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]")
# Global placeholders for voicemail detection (initialized in main)
VOICEMAIL_RE = None
VOICEMAIL_LABEL = None
# 新增语音信箱匹配正则（默认）
DEFAULT_VOICEMAIL_PATTERN = r"(?:语音信箱|语音留言|voicemail|voice\s*mail|请留下.*?姓名.*?来电原因|请您.*?留言|请您.*?录制.*?留言|已尝试联系.*?无法接听|提示音后.*?录制留言|录音完成后挂断|小助手|录音结束|天翼通信助理|我可以帮您转达)"
DEFAULT_VOICEMAIL_LABEL = "语音信箱"


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────


def remove_sv_tags(text) -> str:
    """Remove all <|...|> tags from sensevoice_text."""
    if pd.isna(text):
        return ""
    return SENSEVOICE_TAG_RE.sub("", str(text)).strip()


def is_voicemail(text: str) -> bool:
    """判断 sensevoice_text（已去标签）是否匹配语音信箱提示语句。"""
    if not text:
        return False
    return bool(VOICEMAIL_RE.search(text))


def normalize(text: str) -> str:
    """Normalize text: keep CJK chars and alphanumerics, strip punctuation/spaces."""
    return PUNCT_STRIP_RE.sub("", text)


def similarity(a: str, b: str) -> float:
    """Sequence similarity ratio between two normalized strings."""
    if a == "" and b == "":
        return 1.0
    if a == "" or b == "":
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def char_edit_stats(ref: str, hyp: str) -> dict:
    """以 ref 为标注、hyp 为识别结果，拆分字符级编辑错误。

    返回：
      - 字准率: 1 - 偏差字符数 / max(len(ref), 1)；两端皆空时为 1.0
      - 少了: 标注有、识别缺（delete）
      - 多了: 标注无、识别多（insert）
      - 错了: 同位替换（replace）
      - 偏差字符数: 少了 + 多了 + 错了
    """
    ref = ref or ""
    hyp = hyp or ""
    if ref == "" and hyp == "":
        return {"字准率": 1.0, "少了": 0, "多了": 0, "错了": 0, "偏差字符数": 0}
    if ref == "":
        return {"字准率": 0.0, "少了": 0, "多了": len(hyp), "错了": 0, "偏差字符数": len(hyp)}
    if hyp == "":
        return {"字准率": 0.0, "少了": len(ref), "多了": 0, "错了": 0, "偏差字符数": len(ref)}

    deleted = inserted = substituted = 0
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, ref, hyp).get_opcodes():
        if tag == "equal":
            continue
        if tag == "delete":
            deleted += i2 - i1
        elif tag == "insert":
            inserted += j2 - j1
        elif tag == "replace":
            ref_span = i2 - i1
            hyp_span = j2 - j1
            common = min(ref_span, hyp_span)
            substituted += common
            if ref_span > hyp_span:
                deleted += ref_span - hyp_span
            elif hyp_span > ref_span:
                inserted += hyp_span - ref_span

    edits = deleted + inserted + substituted
    acc = 1.0 - (edits / len(ref))
    return {
        "字准率": round(acc, 6),
        "少了": deleted,
        "多了": inserted,
        "错了": substituted,
        "偏差字符数": edits,
    }


def is_interjection_only(text: str) -> bool:
    """True if qwen_text (after stripping punctuation) is a single interjection character."""
    if not text:
        return False
    stripped = normalize(text)
    return stripped in INTERJECTIONS


def is_effectively_empty(text: str) -> bool:
    """True if sensevoice (after tag removal + punctuation strip) is empty."""
    return normalize(text) == ""


# ──────────────────────────────────────────────
# Core processing
# ──────────────────────────────────────────────


def process_dataframe(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    df = df.copy()

    # Step 1: strip sensevoice tags
    df["sensevoice_text"] = df["sensevoice_text"].apply(remove_sv_tags)

    # Normalized columns for comparison
    qwen_norm = df["qwen_text"].fillna("").apply(normalize)
    sv_norm = df["sensevoice_text"].fillna("").apply(normalize)

    # Overwrite original columns with punctuation‑removed text
    df["qwen_text"] = qwen_norm
    df["sensevoice_text"] = sv_norm

    gold_labels = []
    for idx in range(len(df)):
        q_raw = str(df["qwen_text"].iloc[idx]) if pd.notna(df["qwen_text"].iloc[idx]) else ""
        s_cleaned = (
            str(df["sensevoice_text"].iloc[idx])
            if pd.notna(df["sensevoice_text"].iloc[idx])
            else ""
        )
        q_norm = qwen_norm.iloc[idx]
        s_norm = sv_norm.iloc[idx]

        # Rule: both qwen and sensevoice empty after punctuation removal -> noise "/"
        if q_norm == "" and s_norm == "":
            gold_labels.append("/")
            continue

        # Rule: interjection + sensevoice effectively empty -> noise "/"
        if is_interjection_only(q_raw) and is_effectively_empty(s_cleaned):
            gold_labels.append("/")
            continue

        # Rule: voicemail detection -> label as voicemail
        if is_voicemail(s_cleaned):
            gold_labels.append(VOICEMAIL_LABEL)
            continue

        # Rule: similarity >= threshold -> gold label (use qwen value)
        sim = similarity(q_norm, s_norm)
        if sim >= threshold:
            gold_labels.append(q_raw if q_raw else s_cleaned)
            continue

        # Otherwise: pending
        gold_labels.append("")

    df["gold_label"] = gold_labels
    # 同一标记：如果 qwen_text 与 sensevoice_text 完全相同则为 "是", 否则为 "否"
    df["same_flag"] = (df["qwen_text"] == df["sensevoice_text"]).map({True: "是", False: "否"})

    # 字准率及错误分解：以 qwen 为参考、sensevoice 为假设
    edit_rows = [
        char_edit_stats(str(q) if pd.notna(q) else "", str(s) if pd.notna(s) else "")
        for q, s in zip(df["qwen_text"], df["sensevoice_text"])
    ]
    edit_df = pd.DataFrame(edit_rows)
    for col in ("字准率", "少了", "多了", "错了", "偏差字符数"):
        df[col] = edit_df[col]
    return df


def process_file(xlsx_path: Path, output_path: Path, threshold: float) -> dict:
    print(f"[Processing] {xlsx_path}")
    df = pd.read_excel(xlsx_path)

    total = len(df)
    df_out = process_dataframe(df, threshold)

    gold_mask = (
        (df_out["gold_label"] != "") & (df_out["gold_label"] != "/") & df_out["gold_label"].notna()
    )
    noise_mask = df_out["gold_label"] == "/"
    voicemail_mask = df_out["gold_label"] == VOICEMAIL_LABEL
    pending_mask = (df_out["gold_label"] == "") | df_out["gold_label"].isna()

    stats = {
        "file": str(xlsx_path),
        "total": total,
        "gold": int(gold_mask.sum()),
        "noise": int(noise_mask.sum()),
        "voicemail": int(voicemail_mask.sum()),
        "pending": int(pending_mask.sum()),
        "threshold": threshold,
    }

    # 使用临时文件写入后再替换，避免文件被占用导致的 PermissionError
    import tempfile

    tmp_path = output_path.with_suffix(".tmp.xlsx")
    df_out.to_excel(tmp_path, index=False)
    # 若目标文件已存在且可以删除，则删除
    if output_path.exists():
        try:
            output_path.unlink()
        except PermissionError:
            # 如果仍被占用，直接覆盖（pandas 默认会覆盖，但在 Windows 可能仍报错）
            pass
    tmp_path.replace(output_path)

    pct_gold = stats["gold"] / total * 100
    pct_noise = stats["noise"] / total * 100
    pct_pending = stats["pending"] / total * 100
    print(f"  -> Output: {output_path}")
    print(
        f"     Total: {total:,} | Gold: {stats['gold']:,} ({pct_gold:.1f}%) | Noise(/): {stats['noise']:,} ({pct_noise:.1f}%) | Pending: {stats['pending']:,} ({pct_pending:.1f}%)"
    )
    if total > 0:
        print(
            f"     字准率(均值): {df_out['字准率'].mean():.4f} | "
            f"少了: {int(df_out['少了'].sum()):,} | "
            f"多了: {int(df_out['多了'].sum()):,} | "
            f"错了: {int(df_out['错了'].sum()):,} | "
            f"偏差字符数: {int(df_out['偏差字符数'].sum()):,}"
        )
    return stats


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Generate gold labels for qwen vs sensevoice comparison"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Similarity threshold for gold label (default: {DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="qwen_vs_sensevoice",
        help="Root input directory containing batch subdirectories (default: qwen_vs_sensevoice)",
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="_labeled",
        help="Suffix appended to output filename (default: _labeled)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output root directory (default: same as input batch subdirectory)",
    )
    # 新增参数：语音信箱标签与正则模式
    parser.add_argument(
        "--voicemail-label",
        type=str,
        default=DEFAULT_VOICEMAIL_LABEL,
        help="Label to assign for voicemail rows (default: 语音信箱)",
    )
    parser.add_argument(
        "--voicemail-pattern",
        type=str,
        default=DEFAULT_VOICEMAIL_PATTERN,
        help="Regex pattern to detect voicemail text (default covers common Chinese/English prompts)",
    )
    args = parser.parse_args()

    # 初始化全局正则与标签
    global VOICEMAIL_RE, VOICEMAIL_LABEL
    VOICEMAIL_LABEL = args.voicemail_label
    VOICEMAIL_RE = re.compile(args.voicemail_pattern, flags=re.IGNORECASE)

    input_root = Path(args.input_dir)
    if not input_root.exists():
        print(f"[ERROR] Input directory not found: {input_root}", file=sys.stderr)
        sys.exit(1)

    xlsx_files = sorted(input_root.rglob("badcases.xlsx"))
    if not xlsx_files:
        print(f"[ERROR] No badcases.xlsx found under {input_root}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(xlsx_files)} file(s), similarity threshold: {args.threshold}")
    print("=" * 60)

    all_stats = []
    for xlsx_path in xlsx_files:
        if args.output_dir:
            rel = xlsx_path.relative_to(input_root)
            stem = xlsx_path.stem + args.output_suffix
            output_path = Path(args.output_dir) / rel.parent / (stem + ".xlsx")
        else:
            stem = xlsx_path.stem + args.output_suffix
            output_path = xlsx_path.parent / (stem + ".xlsx")

        stats = process_file(xlsx_path, output_path, args.threshold)
        all_stats.append(stats)

    print()
    print("=" * 60)
    print("Summary:")
    total_rows = sum(s["total"] for s in all_stats)
    total_gold = sum(s["gold"] for s in all_stats)
    total_noise = sum(s["noise"] for s in all_stats)
    total_voicemail = sum(s.get("voicemail", 0) for s in all_stats)
    total_pending = sum(s["pending"] for s in all_stats)
    print(f"  Total rows:   {total_rows:,}")
    print(f"  Gold labels:  {total_gold:,} ({total_gold / total_rows * 100:.1f}%)")
    print(f"  Noise (/):    {total_noise:,} ({total_noise / total_rows * 100:.1f}%)")
    print(f"  Voicemail:    {total_voicemail:,} ({total_voicemail / total_rows * 100:.1f}%)")
    print(f"  Pending:      {total_pending:,} ({total_pending / total_rows * 100:.1f}%)")
    print("Done.")


if __name__ == "__main__":
    main()
