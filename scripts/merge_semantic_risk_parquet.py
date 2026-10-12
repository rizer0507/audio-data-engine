#!/usr/bin/env python3
"""从两个（或多个）五类 v3 分类 parquet 中挑选 semantic_risk，原样汇总到一个 parquet。

只做挑选与拼接：不清洗、不改 category、不删列、不删 labels/transcripts/quality 等字段。
类别判定与 annotate_semantic_risk_transcripts / assemble_training_parquet 一致：
reviewed_category → labels.category → type。

示例::

  python scripts/merge_semantic_risk_parquet.py \\
    --input datasets/stage1/derived/classified_five_class_v3_BATCH_A.parquet \\
    --input datasets/stage1/derived/classified_five_class_v3_BATCH_B.parquet \\
    --output datasets/stage1/derived/semantic_risk_BATCH_A_BATCH_B.parquet

  python scripts/merge_semantic_risk_parquet.py \\
    --batch BATCH_A --batch BATCH_B \\
    --output datasets/stage1/derived/semantic_risk_BATCH_A_BATCH_B.parquet
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audio_engine.core.manifest import Manifest
from audio_engine.core.sample import Sample

DEFAULT_RULE_STEM = "classified_five_class_v3"
CATEGORY = "semantic_risk"
SCRIPT_VERSION = "1.0.0"


def resolve_path(raw: str | Path) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = ROOT / path
    return path


def default_classified_path(batch: str, *, rule_stem: str = DEFAULT_RULE_STEM) -> Path:
    return ROOT / "datasets" / "stage1" / "derived" / f"{rule_stem}_{batch}.parquet"


def category_of(sample: Sample) -> str:
    labels = sample.labels or {}
    for key in ("reviewed_category", "category", "type"):
        val = labels.get(key)
        if val is not None and str(val).strip():
            return str(val).strip()
    return ""


def collect_inputs(
    *,
    inputs: list[Path],
    batches: list[str],
    rule_stem: str,
) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()

    for raw in inputs:
        path = resolve_path(raw)
        key = path.resolve()
        if key in seen:
            continue
        seen.add(key)
        paths.append(path)

    for batch in batches:
        name = str(batch).strip()
        if not name:
            continue
        path = default_classified_path(name, rule_stem=rule_stem)
        key = path.resolve()
        if key in seen:
            continue
        seen.add(key)
        paths.append(path)

    return paths


def pick_semantic_risk(path: Path) -> tuple[list[Sample], Counter[str], int]:
    if not path.is_file():
        raise FileNotFoundError(f"输入 parquet 不存在: {path}")

    manifest = Manifest.load(path)
    selected: list[Sample] = []
    cat_counts: Counter[str] = Counter()
    missing_category = 0

    for sample in manifest.samples:
        cat = category_of(sample)
        if not cat:
            missing_category += 1
            cat_counts["<missing>"] += 1
            continue
        cat_counts[cat] += 1
        if cat == CATEGORY:
            # 深拷贝，避免后续操作意外改写源 Manifest 内存对象
            selected.append(sample.model_copy(deep=True))

    return selected, cat_counts, missing_category


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="从两个 V3 五类分类 parquet 中汇总全部 semantic_risk（不删字段）",
    )
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        metavar="PARQUET",
        help="分类结果 parquet（可重复；与 --batch 可混用）",
    )
    parser.add_argument(
        "--batch",
        action="append",
        default=[],
        metavar="BATCH",
        help=f"source-name，默认读 datasets/stage1/derived/{DEFAULT_RULE_STEM}_{{batch}}.parquet",
    )
    parser.add_argument(
        "--rule-stem",
        default=DEFAULT_RULE_STEM,
        help=f"配合 --batch 的文件名前缀（默认 {DEFAULT_RULE_STEM}）",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        type=Path,
        help="汇总输出 parquet 路径",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="允许覆盖已存在的 --output",
    )
    args = parser.parse_args(argv)

    paths = collect_inputs(
        inputs=[Path(p) for p in args.input],
        batches=list(args.batch),
        rule_stem=str(args.rule_stem).strip() or DEFAULT_RULE_STEM,
    )
    if len(paths) < 2:
        raise SystemExit(
            "[ERROR] 至少需要两个输入（--input 和/或 --batch 合计 ≥ 2）"
        )

    out_path = resolve_path(args.output)
    if out_path.suffix.lower() != ".parquet":
        raise SystemExit(f"[ERROR] --output 必须是 .parquet: {out_path}")
    if out_path.exists() and not args.force:
        raise SystemExit(f"[ERROR] 输出已存在（加 --force 覆盖）: {out_path}")

    merged: list[Sample] = []
    id_counter: Counter[str] = Counter()

    print(f"[INFO] script={SCRIPT_VERSION} category={CATEGORY}")
    for path in paths:
        selected, cat_counts, missing = pick_semantic_risk(path)
        total = sum(cat_counts.values())
        n_sr = len(selected)
        print(f"[INFO] {path}")
        print(f"       total={total} semantic_risk={n_sr} missing_category={missing}")
        if cat_counts:
            top = ", ".join(f"{k}={v}" for k, v in sorted(cat_counts.items()))
            print(f"       categories: {top}")
        for s in selected:
            id_counter[str(s.id)] += 1
        merged.extend(selected)

    dup_ids = sorted(sid for sid, n in id_counter.items() if n > 1)
    if dup_ids:
        print(
            f"[WARN] 跨输入重复 id {len(dup_ids)} 个（仍全部保留，不做去重），"
            f"例: {dup_ids[:5]}"
        )

    Manifest.from_samples(merged).save(out_path)
    print(f"[OK] wrote {len(merged)} semantic_risk rows → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
