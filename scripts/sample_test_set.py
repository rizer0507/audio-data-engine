# -*- coding: utf-8 -*-
"""
sample_test_set.py
==================
从 数据集/ 目录下的各 xlsx 文件中按规则随机抽样，合并输出为 test0923.xlsx。

抽样规则：
  - 标注一致-分身.xlsx      随机抽取 10000 条
  - 语音信箱.xlsx           随机抽取 200 条
  - 噪声数据集-非人声.xlsx  随机抽取 2000 条
  - hard-cases.xlsx         全部抽取
  - 噪声数据集-人声.xlsx    不做抽取（跳过）

输出列：
  A: id
  B: （空白，供人工填写）
  C: source（来源文件名，不含路径和扩展名）

Usage
-----
python scripts/sample_test_set.py

# 指定数据集目录、输出路径、随机种子
python scripts/sample_test_set.py \\
    --dataset-dir 数据集 \\
    --output test0923.xlsx \\
    --seed 42
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# ──────────────────────────────────────────────
# 抽样配置（文件名 stem → 抽样数量，None 表示全取）
# ──────────────────────────────────────────────

SAMPLING_RULES: dict[str, int | None] = {
    "标注一致-分身": 10000,
    "语音信箱": 200,
    "噪声数据集-非人声": 2000,
    "hard-cases": None,       # 全部抽取
    # "噪声数据集-人声": 跳过，不在此 dict 中
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="从各数据集 xlsx 按规则随机抽样，合并输出为 test0923.xlsx"
    )
    parser.add_argument(
        "--dataset-dir",
        default="数据集",
        help="数据集目录（默认: 数据集）",
    )
    parser.add_argument(
        "--output",
        default="test0923.xlsx",
        help="输出 xlsx 路径（默认: test0923.xlsx，位于项目根目录）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子，保证可复现（默认: 42）",
    )
    args = parser.parse_args()

    try:
        import pandas as pd
    except ImportError:
        print("[ERROR] 缺少 pandas，请先安装：pip install pandas openpyxl", file=sys.stderr)
        sys.exit(1)

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.is_absolute():
        dataset_dir = project_root / dataset_dir

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = project_root / output_path

    if not dataset_dir.is_dir():
        print(f"[ERROR] 数据集目录不存在: {dataset_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"数据集目录: {dataset_dir}")
    print(f"输出路径:   {output_path}")
    print(f"随机种子:   {args.seed}")
    print("=" * 55)

    frames: list[pd.DataFrame] = []

    for stem, n_sample in SAMPLING_RULES.items():
        # 尝试 .xlsx 和 .xls 两种扩展名
        xlsx_path = dataset_dir / f"{stem}.xlsx"
        xls_path  = dataset_dir / f"{stem}.xls"

        if xlsx_path.exists():
            file_path = xlsx_path
            engine = "openpyxl"
        elif xls_path.exists():
            file_path = xls_path
            engine = "xlrd"
        else:
            print(f"[WARN] 未找到文件: {stem}.xlsx / {stem}.xls，跳过")
            continue

        print(f"读取: {file_path.name} ...", end=" ", flush=True)
        try:
            df = pd.read_excel(file_path, engine=engine, dtype=str)
        except Exception as exc:
            print(f"\n[ERROR] 读取失败: {exc}", file=sys.stderr)
            continue

        total = len(df)

        # 确保有 id 列
        if "id" not in df.columns:
            # 尝试大小写不敏感匹配
            col_map = {c.lower(): c for c in df.columns}
            if "id" in col_map:
                df = df.rename(columns={col_map["id"]: "id"})
            else:
                print(f"[WARN] {file_path.name} 中未找到 'id' 列（可用列: {list(df.columns)}），跳过")
                continue

        # 随机抽样
        if n_sample is None or total <= n_sample:
            sampled = df.copy()
            actual = len(sampled)
            note = "全量" if n_sample is None else f"目标 {n_sample} 条，实际全量（仅 {total} 条）"
        else:
            sampled = df.sample(n=n_sample, random_state=args.seed)
            actual = len(sampled)
            note = f"从 {total} 条中随机抽 {actual} 条"

        print(f"{note}")

        # 构建输出行：id → 空白标注列 → qwen/sensevoice 识别结果 → 来源
        def _get_col(df_src: pd.DataFrame, col: str) -> pd.Series:
            """大小写不敏感取列，不存在则返回空串。"""
            col_map = {c.lower(): c for c in df_src.columns}
            real = col_map.get(col.lower())
            return df_src[real].fillna("") if real else pd.Series([""] * len(df_src))

        out = pd.DataFrame({
            "id":              sampled["id"].str.strip(),
            "标注":            "",                        # 第二列留空，供人工填写
            "qwen_text":       _get_col(sampled, "qwen_text").values,
            "sensevoice_text": _get_col(sampled, "sensevoice_text").values,
            "source":          stem,                      # 来源文件名（不含扩展名）
        })
        frames.append(out)

    if not frames:
        print("\n[ERROR] 没有任何数据可输出", file=sys.stderr)
        sys.exit(1)

    result = pd.concat(frames, ignore_index=True)

    # 去重（以防同一条 id 在多个来源中重复）
    before = len(result)
    result = result.drop_duplicates(subset=["id"])
    after = len(result)
    if before != after:
        print(f"\n[INFO] 跨文件去重：{before} → {after}（移除 {before - after} 条重复 id）")

    # 打乱最终顺序（让来源分布均匀，不按文件分块排列）
    result = result.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    # 写出
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_excel(output_path, index=False)

    print("=" * 55)
    print(f"汇总:")
    for src, grp in result.groupby("source", sort=False):
        print(f"  {src:<20s}  {len(grp):>6,} 条")
    print(f"  {'合计':<20s}  {len(result):>6,} 条")
    print(f"\n输出完成: {output_path}")


if __name__ == "__main__":
    main()
