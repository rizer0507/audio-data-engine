# -*- coding: utf-8 -*-
"""
merge_qwen_results.py
=====================
将 Qwen ASR pipeline 输出的 parquet manifest，与 test0923_new.xlsx 按 id
一一匹配，把识别结果回填到 xlsx 的 qwen_text 列，输出新的 xlsx 文件。

parquet 中 transcripts 列结构（JSON 字符串或 dict）：
    {"qwen": {"text": "识别文本", "model": "...", "version": "...", "extra": {}}}

xlsx 列顺序（不变）：
    id | 标注 | qwen_text | sensevoice_text | source

Usage
-----
# 最简：parquet 与 xlsx 同目录/路径均使用默认值
python scripts/merge_qwen_results.py \\
    --parquet  datasets/manifests/qwen_asr_source_A.parquet \\
    --xlsx     test0923_new.xlsx

# 自定义输出路径
python scripts/merge_qwen_results.py \\
    --parquet  datasets/manifests/qwen_asr_source_A.parquet \\
    --xlsx     test0923_new.xlsx \\
    --output   test0923_labeled.xlsx \\
    --transcript-key qwen          # parquet transcripts 里的 key（默认 qwen）
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

# ──────────────────────────────────────────────
# 字准率辅助函数（与 gold_label_generator.py 保持一致）
# ──────────────────────────────────────────────

_PUNCT_STRIP_RE = re.compile(r"[^\w\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]")


def _normalize(text: str) -> str:
    """去除标点、空格，只保留中文、日文、字母数字，用于字符级比较。"""
    if not isinstance(text, str):
        return ""
    return _PUNCT_STRIP_RE.sub("", text)


def _char_similarity(a: str, b: str) -> float:
    """归一化后的字符序列相似度（SequenceMatcher ratio），范围 [0, 1]。"""
    a, b = _normalize(a), _normalize(b)
    if a == "" and b == "":
        return 1.0
    if a == "" or b == "":
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def compute_char_stats(df, threshold: float) -> None:
    """
    对 DataFrame 中「标注」与「qwen_text」两列计算字符相似度（字准率），
    只统计两列均非空的行。

    输出：
      - 参与统计的行数
      - 平均字准率
      - 达到阈值（≥ threshold）的行数及占比
      - 相似度分段分布
    """
    try:
        import pandas as pd
    except ImportError:
        return

    if "标注" not in df.columns or "qwen_text" not in df.columns:
        print("[WARN] 缺少「标注」或「qwen_text」列，跳过字准率统计")
        return

    # 只取两列均非空的行
    mask = (
        df["标注"].notna() & (df["标注"].astype(str).str.strip() != "") &
        df["qwen_text"].notna() & (df["qwen_text"].astype(str).str.strip() != "")
    )
    sub = df[mask].copy()
    total = len(sub)

    if total == 0:
        print("[WARN] 无可统计的行（标注 或 qwen_text 列全部为空）")
        return

    sims = [
        _char_similarity(str(row["标注"]), str(row["qwen_text"]))
        for _, row in sub.iterrows()
    ]

    avg_sim    = sum(sims) / total
    above      = sum(1 for s in sims if s >= threshold)
    above_pct  = above / total * 100

    # 分段分布
    buckets = {"[1.0]": 0, "[0.9,1)": 0, "[0.7,0.9)": 0, "[0.5,0.7)": 0, "<0.5": 0}
    for s in sims:
        if s == 1.0:
            buckets["[1.0]"] += 1
        elif s >= 0.9:
            buckets["[0.9,1)"] += 1
        elif s >= 0.7:
            buckets["[0.7,0.9)"] += 1
        elif s >= 0.5:
            buckets["[0.5,0.7)"] += 1
        else:
            buckets["<0.5"] += 1

    print()
    print("=" * 55)
    print(f"字准率统计（标注 vs qwen_text，阈值={threshold:.2f}）")
    print(f"  参与统计行数: {total:,}")
    print(f"  平均字准率:   {avg_sim:.4f}  ({avg_sim*100:.2f}%)")
    print(f"  达到阈值行数: {above:,} / {total:,}  ({above_pct:.2f}%)")
    print(f"  相似度分布:")
    for label, cnt in buckets.items():
        bar = "█" * int(cnt / total * 40)
        print(f"    {label:<12s}  {cnt:>6,}  {cnt/total*100:5.1f}%  {bar}")
    print("=" * 55)


def _strip_sha_prefix(raw_id: str) -> str:
    """
    parquet 中的 id 格式为：<sha256前16位>_<原始sample.id>
    例：57200dca6f822af4_fenshen-cabee084-780d4ba8
    → 还原为：fenshen-cabee084-780d4ba8

    规则：去掉第一个下划线及其之前的所有内容。
    若不含下划线（历史数据或格式不同），原样返回。
    """
    if "_" in raw_id:
        return raw_id.split("_", 1)[1]
    return raw_id


def load_qwen_texts(parquet_path: Path, transcript_key: str) -> dict[str, str]:
    """
    读取 parquet，返回 {原始sample_id: qwen_text} 字典。
    parquet 中 id 带 sha256 前缀，自动剥离后再作为匹配键。
    transcripts 列可能是 JSON 字符串或 dict（取决于 manifest 保存方式）。
    """
    try:
        import pandas as pd
    except ImportError:
        print("[ERROR] 缺少 pandas，请安装：pip install pandas pyarrow", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] 读取 parquet: {parquet_path}")
    df = pd.read_parquet(parquet_path)

    if "id" not in df.columns:
        print("[ERROR] parquet 中缺少 'id' 列", file=sys.stderr)
        sys.exit(1)
    if "transcripts" not in df.columns:
        print("[ERROR] parquet 中缺少 'transcripts' 列，请确认 pipeline 已完成", file=sys.stderr)
        sys.exit(1)

    id_to_text: dict[str, str] = {}
    missing = 0
    dup_warn: list[str] = []

    for _, row in df.iterrows():
        raw_id = str(row["id"]).strip()
        # 剥离 sha256 前缀，得到与 xlsx 匹配的真实 id
        real_id = _strip_sha_prefix(raw_id)

        raw = row["transcripts"]

        # 解析 transcripts 字段（可能是 JSON 字符串 或 dict）
        if isinstance(raw, str):
            try:
                transcripts = json.loads(raw)
            except json.JSONDecodeError:
                missing += 1
                continue
        elif isinstance(raw, dict):
            transcripts = raw
        else:
            missing += 1
            continue

        # 取指定 key 下的 text
        entry = transcripts.get(transcript_key)
        if isinstance(entry, dict):
            text = str(entry.get("text", "")).strip()
        elif isinstance(entry, str):
            text = entry.strip()
        else:
            text = ""

        if real_id in id_to_text:
            dup_warn.append(real_id)
        id_to_text[real_id] = text

    print(
        f"[INFO] parquet 中共 {len(df)} 条，成功解析 {len(id_to_text)} 条"
        + (f"，{missing} 条缺少 transcripts 字段" if missing else "")
        + (f"，{len(dup_warn)} 条出现同名 id（已取最后一条）" if dup_warn else "")
    )
    if dup_warn:
        print(f"[WARN] 重复 id 示例（前5个）：{dup_warn[:5]}")
    return id_to_text



def merge_into_xlsx(
    id_to_text: dict[str, str],
    xlsx_path: Path,
    output_path: Path,
    transcript_key: str,
) -> None:
    try:
        import pandas as pd
    except ImportError:
        print("[ERROR] 缺少 pandas，请安装：pip install pandas openpyxl", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] 读取 xlsx: {xlsx_path}")
    df = pd.read_excel(xlsx_path, dtype=str)

    if "id" not in df.columns:
        print("[ERROR] xlsx 中缺少 'id' 列", file=sys.stderr)
        sys.exit(1)

    # 确保 qwen_text 列存在；若已有则覆盖，否则插入到第三列（标注 之后）
    before_total = len(df)
    df["id"] = df["id"].fillna("").str.strip()

    # 用 parquet 结果回填「标注」列（原本空白那列）
    qwen_col = df["id"].map(id_to_text)
    hit = qwen_col.notna().sum()

    if "标注" not in df.columns:
        print("[WARN] xlsx 中未找到「标注」列，将在第二列位置新建", file=sys.stderr)
        df.insert(1, "标注", "")

    # 只写入有识别结果且标注列当前为空的行，不覆盖已有内容
    mask_empty = df["标注"].isna() | (df["标注"].astype(str).str.strip() == "")
    df.loc[mask_empty, "标注"] = qwen_col[mask_empty]

    # 统计匹配情况
    no_match = before_total - hit
    print(f"[INFO] xlsx 共 {before_total} 行，命中 {hit} 行，无匹配 {no_match} 行")
    if no_match > 0:
        unmatched_ids = df.loc[qwen_col.isna(), "id"].tolist()
        print(f"[WARN] 前10个无匹配 id：{unmatched_ids[:10]}")

    # 逐行计算字准率（标注 vs qwen_text），写入新列
    print("[INFO] 计算字准率列...")
    if "qwen_text" in df.columns:
        def _row_sim(row) -> str:
            a = str(row["标注"]) if (row["标注"] is not None and str(row["标注"]) not in ("", "nan", "None")) else ""
            b = str(row["qwen_text"]) if (row["qwen_text"] is not None and str(row["qwen_text"]) not in ("", "nan", "None")) else ""
            if a.strip() == "" or b.strip() == "":
                return ""
            return f"{_char_similarity(a, b):.4f}"
        df["字准率"] = df.apply(_row_sim, axis=1)
        filled = (df["字准率"] != "").sum()
        print(f"[INFO] 字准率列：{filled} 行有效（其余两列之一为空，留空）")
    else:
        print("[WARN] 缺少 qwen_text 列，字准率列留空")
        df["字准率"] = ""

    # 写出（先写临时文件再原子替换，避免文件被占用时损坏）
    tmp_path = output_path.with_suffix(".tmp.xlsx")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(tmp_path, index=False)
    if output_path.exists():
        try:
            output_path.unlink()
        except PermissionError:
            print(f"[WARN] 无法删除旧文件 {output_path}（可能被 Excel 占用），尝试直接覆盖")
    tmp_path.replace(output_path)

    print(f"[INFO] 输出完成: {output_path}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="将 Qwen ASR parquet 识别结果回填到 test0923 xlsx"
    )
    parser.add_argument(
        "--parquet",
        required=True,
        help="Qwen ASR pipeline 输出的 parquet 路径",
    )
    parser.add_argument(
        "--xlsx",
        default="test0923_new.xlsx",
        help="目标 xlsx 文件（默认: test0923_new.xlsx）",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="输出路径（默认: 覆盖写入 --xlsx 文件）",
    )
    parser.add_argument(
        "--transcript-key",
        default="qwen",
        help="parquet transcripts 字段中的 key（默认: qwen）",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.9,
        help="字准率达标阈值，用于统计达标行数比例（默认: 0.9）",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    parquet_path = Path(args.parquet)
    if not parquet_path.is_absolute():
        parquet_path = project_root / parquet_path
    if not parquet_path.exists():
        print(f"[ERROR] parquet 文件不存在: {parquet_path}", file=sys.stderr)
        sys.exit(1)

    xlsx_path = Path(args.xlsx)
    if not xlsx_path.is_absolute():
        xlsx_path = project_root / xlsx_path
    if not xlsx_path.exists():
        print(f"[ERROR] xlsx 文件不存在: {xlsx_path}", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output) if args.output else xlsx_path
    if not output_path.is_absolute():
        output_path = project_root / output_path

    print("=" * 55)
    print(f"parquet:         {parquet_path}")
    print(f"xlsx:            {xlsx_path}")
    print(f"输出:            {output_path}")
    print(f"transcript key:  {args.transcript_key}")
    print("=" * 55)

    id_to_text = load_qwen_texts(parquet_path, args.transcript_key)
    df_result = merge_into_xlsx(id_to_text, xlsx_path, output_path, args.transcript_key)

    # 字准率统计（标注 vs qwen_text）
    if df_result is not None:
        compute_char_stats(df_result, threshold=args.threshold)

    print("\n完成。")


if __name__ == "__main__":
    main()
