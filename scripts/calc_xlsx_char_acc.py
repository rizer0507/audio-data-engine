#!/usr/bin/env python3
"""跨一个或多个 xlsx 指定文本列，相对 base 计算字准率 / 编辑距离。

列指定格式（避免 Windows 盘符冲突，用双冒号）::

    路径.xlsx::列名

示例::

  # 同一文件
  python scripts/calc_xlsx_char_acc.py \
    --base C:/Users/rizer/Desktop/0827-test/test-final.xlsx::label \
    --hyp  C:/Users/rizer/Desktop/0827-test/test-final.xlsx::qwen1_text \
    --hyp  C:/Users/rizer/Desktop/0827-test/test-final.xlsx::qwen-sft_text \
    --char-acc  --edit-distance \
    --output "C:/Users/rizer/Desktop/0827-test/test-final-字准率.xlsx"

  # 不同文件（按 id 对齐）
  python scripts/calc_xlsx_char_acc.py \\
    --base 数据集/a.xlsx::qwen_text \\
    --hyp  数据集/b.xlsx::sensevoice_text \\
    --join-key id \\
    --char-acc \\
    --output 数据集/对比.xlsx

  # 按 type 分组统计少字/错字/多字与字准率（默认检测 type 列）
  python scripts/calc_xlsx_char_acc.py \\
    --base a.xlsx::label --hyp a.xlsx::qwen1_text \\
    --char-acc --edit-distance --group-by type \\
    -o out.xlsx
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audio_engine.core.transcript_reconcile import (  # noqa: E402
    levenshtein_ops,
    normalize_transcript,
)

_SPEC_RE = re.compile(r"^(?P<path>.+)::(?P<col>.+)$")


@dataclass(frozen=True)
class ColSpec:
    path: Path
    column: str
    label: str  # unique internal name used in merged frame


def _parse_spec(raw: str, *, role: str, index: int = 0) -> ColSpec:
    text = raw.strip().strip('"').strip("'")
    match = _SPEC_RE.match(text)
    if not match:
        raise SystemExit(
            f"[ERROR] 列指定格式应为 路径.xlsx::列名，收到: {raw!r}\n"
            f"        例: 数据集/a.xlsx::qwen_text"
        )
    path = Path(match.group("path").strip())
    column = match.group("col").strip()
    if not column:
        raise SystemExit(f"[ERROR] 列名为空: {raw!r}")
    if role == "base":
        label = f"base__{column}"
    else:
        label = f"hyp{index}__{column}"
    return ColSpec(path=path, column=column, label=label)


def _as_text(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none"}:
        return ""
    return text


def _metric_row(
    ref_raw: str, hyp_raw: str, *, require_ref: bool
) -> dict[str, float | int | None]:
    empty = {
        "total": None,
        "错字": None,
        "少字": None,
        "多字": None,
        "cer": None,
        "字准率": None,
        "ref_len": None,
        "hyp_len": None,
    }
    ref = normalize_transcript(ref_raw)
    if require_ref and not ref:
        return empty
    if not ref_raw and not hyp_raw:
        return {
            "total": 0,
            "错字": 0,
            "少字": 0,
            "多字": 0,
            "cer": 0.0,
            "字准率": 1.0,
            "ref_len": 0,
            "hyp_len": 0,
        }
    ops = levenshtein_ops(ref_raw, hyp_raw)
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


def _validate_and_load(
    specs: list[ColSpec],
    join_key: str,
) -> dict[Path, pd.DataFrame]:
    """Load each unique xlsx; verify requested columns exist."""
    by_path: dict[Path, list[ColSpec]] = {}
    for spec in specs:
        by_path.setdefault(spec.path.resolve(), []).append(spec)

    loaded: dict[Path, pd.DataFrame] = {}
    for path, path_specs in by_path.items():
        if not path.exists():
            raise SystemExit(f"[ERROR] 文件不存在: {path}")
        print(f"[INFO] 读取: {path}")
        df = pd.read_excel(path, sheet_name=0)
        cols = list(df.columns)
        bad_cols = [s.column for s in path_specs if s.column not in df.columns]
        # de-dup
        bad_unique: list[str] = []
        seen: set[str] = set()
        for name in bad_cols:
            if name not in seen:
                seen.add(name)
                bad_unique.append(name)
        if bad_unique:
            raise SystemExit(
                f"[ERROR] {path.name} 缺少列: {bad_unique}\n"
                f"        现有列: {cols}"
            )
        df.attrs["missing_join_key"] = join_key not in df.columns
        loaded[path] = df
        print(f"[INFO]   ok 列={[s.column for s in path_specs]} rows={len(df):,}")
    return loaded


def _build_frame(
    base: ColSpec,
    hyps: list[ColSpec],
    loaded: dict[Path, pd.DataFrame],
    join_key: str,
    *,
    group_by: str | None = None,
) -> pd.DataFrame:
    paths = {base.path.resolve(), *(h.path.resolve() for h in hyps)}
    single_file = len(paths) == 1

    if single_file:
        path = next(iter(paths))
        src = loaded[path]
        out = pd.DataFrame()
        if join_key in src.columns:
            out[join_key] = src[join_key]
        if group_by and group_by in src.columns:
            out[group_by] = src[group_by]
        out[base.label] = src[base.column]
        out[base.column] = src[base.column]
        for hyp in hyps:
            out[hyp.label] = src[hyp.column]
            if hyp.column not in out.columns:
                out[hyp.column] = src[hyp.column]
        return out

    # 多文件：必须每张表都有 join_key
    for path, df in loaded.items():
        if df.attrs.get("missing_join_key"):
            raise SystemExit(
                f"[ERROR] 多文件对齐需要列 {join_key!r}，但缺失于: {path}"
            )

    def _slice(path: Path, columns: list[str], rename: dict[str, str]) -> pd.DataFrame:
        df = loaded[path][[join_key, *columns]].copy()
        dup = int(df[join_key].duplicated().sum())
        if dup:
            print(f"[WARN] {path.name} 的 {join_key} 有 {dup} 条重复，保留首行")
            df = df.drop_duplicates(subset=[join_key], keep="first")
        return df.rename(columns=rename)

    base_cols = [base.column]
    if group_by and group_by in loaded[base.path.resolve()].columns:
        base_cols.append(group_by)
    elif group_by:
        raise SystemExit(
            f"[ERROR] 分组列 {group_by!r} 不在 base 文件中: {base.path}\n"
            f"        现有列: {list(loaded[base.path.resolve()].columns)}"
        )

    base_df = _slice(
        base.path.resolve(),
        base_cols,
        {base.column: base.label},
    )
    base_df[base.column] = base_df[base.label]

    merged = base_df
    for hyp in hyps:
        part = _slice(
            hyp.path.resolve(),
            [hyp.column],
            {hyp.column: hyp.label},
        )
        merged = merged.merge(part, on=join_key, how="outer")
        if hyp.column not in merged.columns:
            merged[hyp.column] = merged[hyp.label]

    print(f"[INFO] 对齐后行数: {len(merged):,} (join_key={join_key})")
    return merged


def _prefix_for(base: ColSpec, hyp: ColSpec) -> str:
    """输出列前缀：vs_{base}_{hyp}，跨文件时带 hyp 文件名。"""
    base_name = base.column
    hyp_name = hyp.column
    if base.path.resolve() != hyp.path.resolve():
        hyp_name = f"{hyp.path.stem}__{hyp.column}"
    safe = re.sub(r"[^\w\u4e00-\u9fff\-]+", "_", f"vs_{base_name}_{hyp_name}")
    return safe[:80]


def _empty_bucket() -> dict[str, float | int]:
    return {
        "dis": 0,
        "ref_len": 0,
        "n": 0,
        "n_skip": 0,
        "sum_acc": 0.0,
        "错字": 0,
        "少字": 0,
        "多字": 0,
    }


def _add_metric(bucket: dict[str, float | int], m: dict[str, float | int | None]) -> None:
    bucket["n"] += 1
    bucket["dis"] += int(m["total"] or 0)
    bucket["ref_len"] += int(m["ref_len"] or 0)
    bucket["sum_acc"] += float(m["字准率"] or 0.0)
    bucket["错字"] += int(m["错字"] or 0)
    bucket["少字"] += int(m["少字"] or 0)
    bucket["多字"] += int(m["多字"] or 0)


def _group_label(value) -> str:
    text = _as_text(value)
    return text if text else "(空)"


def _attach(
    df: pd.DataFrame,
    ref_label: str,
    hyp_label: str,
    prefix: str,
    *,
    require_ref: bool,
    want_acc: bool,
    want_edit: bool,
    group_by: str | None = None,
) -> dict:
    totals: dict = _empty_bucket()
    totals["by_group"] = {}
    rows: list[dict[str, float | int | None]] = []

    for _, row in df.iterrows():
        ref_raw = _as_text(row.get(ref_label))
        hyp_raw = _as_text(row.get(hyp_label))
        gname = _group_label(row.get(group_by)) if group_by else None
        if require_ref and not normalize_transcript(ref_raw):
            rows.append(_metric_row("", "", require_ref=True))
            totals["n_skip"] += 1
            if gname is not None:
                bucket = totals["by_group"].setdefault(gname, _empty_bucket())
                bucket["n_skip"] += 1
            continue
        m = _metric_row(ref_raw, hyp_raw, require_ref=require_ref)
        rows.append(m)
        _add_metric(totals, m)
        if gname is not None:
            _add_metric(totals["by_group"].setdefault(gname, _empty_bucket()), m)

    keys: list[str] = []
    if want_edit:
        keys.extend(["total", "错字", "少字", "多字", "cer", "ref_len", "hyp_len"])
    if want_acc:
        keys.append("字准率")
    for key in keys:
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


def _print_summary(name: str, totals: dict) -> None:
    overall = _overall_acc(totals)
    mean = _mean_acc(totals)

    def fmt(v: float | None) -> str:
        return "N/A" if v is None else f"{v:.4f} ({v * 100:.2f}%)"

    print(f"  [{name}]")
    print(f"    参与行数:   {totals['n']:,}")
    print(f"    跳过行数:   {totals['n_skip']:,}  (基准为空且 --skip-empty-base)")
    print(f"    错字合计:   {totals.get('错字', 0):,}")
    print(f"    少字合计:   {totals.get('少字', 0):,}")
    print(f"    多字合计:   {totals.get('多字', 0):,}")
    print(f"    总编辑距离: {totals['dis']:,}")
    print(f"    总基准字数: {totals['ref_len']:,}")
    print(f"    总体字准率: {fmt(overall)}")
    print(f"    平均字准率: {fmt(mean)}")

    by_group = totals.get("by_group") or {}
    if by_group:
        print(f"    ---- 按分组 ----")
        for gname in sorted(by_group.keys(), key=lambda x: (x == "(空)", str(x))):
            g = by_group[gname]
            print(
                f"      [{gname}] n={g['n']:,} 错字={g['错字']:,} 少字={g['少字']:,} "
                f"多字={g['多字']:,} 字准率={fmt(_overall_acc(g))}"
            )


def _summary_rows(
    name: str,
    totals: dict,
    *,
    group_by: str | None,
) -> list[dict]:
    """Build flat summary rows: overall + per-group."""
    rows: list[dict] = [
        {
            "对比": name,
            group_by or "分组": "总计",
            "参与行数": totals["n"],
            "跳过行数": totals["n_skip"],
            "错字": totals.get("错字", 0),
            "少字": totals.get("少字", 0),
            "多字": totals.get("多字", 0),
            "总编辑距离": totals["dis"],
            "总基准字数": totals["ref_len"],
            "总体字准率": _overall_acc(totals),
            "平均字准率": _mean_acc(totals),
        }
    ]
    by_group = totals.get("by_group") or {}
    for gname in sorted(by_group.keys(), key=lambda x: (x == "(空)", str(x))):
        g = by_group[gname]
        rows.append(
            {
                "对比": name,
                group_by or "分组": gname,
                "参与行数": g["n"],
                "跳过行数": g["n_skip"],
                "错字": g["错字"],
                "少字": g["少字"],
                "多字": g["多字"],
                "总编辑距离": g["dis"],
                "总基准字数": g["ref_len"],
                "总体字准率": _overall_acc(g),
                "平均字准率": _mean_acc(g),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="跨 xlsx 指定列，相对 base 计算字准率 / 编辑距离",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "列格式: 路径.xlsx::列名\n"
            "例: python scripts/calc_xlsx_char_acc.py "
            "--base a.xlsx::qwen_text --hyp a.xlsx::qwen_sft_text "
            "--char-acc --edit-distance -o out.xlsx"
        ),
    )
    parser.add_argument("--base", required=True, help="基准列：路径.xlsx::列名")
    parser.add_argument(
        "--hyp",
        action="append",
        required=True,
        dest="hyps",
        help="假设列：路径.xlsx::列名（可重复）",
    )
    parser.add_argument(
        "--join-key",
        default="id",
        help="多文件对齐键（默认 id）；单文件可不需要该列",
    )
    parser.add_argument("--char-acc", action="store_true", help="计算字准率列")
    parser.add_argument(
        "--edit-distance",
        action="store_true",
        help="计算编辑距离列（total/错字/少字/多字/cer/ref_len/hyp_len）",
    )
    parser.add_argument(
        "--skip-empty-base",
        action="store_true",
        help="基准为空时跳过该行（默认：基准为空也继续计算）",
    )
    parser.add_argument(
        "--group-by",
        default="type",
        help="按该列枚举值分组统计错字/少字/多字与字准率（默认 type；"
        "设为空字符串则关闭分组）",
    )
    parser.add_argument(
        "--allow-empty-base",
        action="store_true",
        help=argparse.SUPPRESS,  # 兼容旧参数；现已是默认行为
    )
    parser.add_argument("--output", "-o", type=Path, required=True, help="输出 xlsx")
    parser.add_argument(
        "--keep-internal",
        action="store_true",
        help="保留内部对齐列名（base__/hypN__）",
    )
    args = parser.parse_args()

    if not args.char_acc and not args.edit_distance:
        raise SystemExit("[ERROR] 请至少指定 --char-acc 或 --edit-distance 之一")

    base = _parse_spec(args.base, role="base")
    hyps = [_parse_spec(h, role="hyp", index=i) for i, h in enumerate(args.hyps)]

    print("[INFO] 核验列 …")
    print(f"  base: {base.path} :: {base.column}")
    for hyp in hyps:
        print(f"  hyp:  {hyp.path} :: {hyp.column}")

    loaded = _validate_and_load([base, *hyps], args.join_key)

    base_df = loaded[base.path.resolve()]
    if base.column not in base_df.columns:
        raise SystemExit(
            f"[ERROR] base 列不存在: {base.column!r} @ {base.path}\n"
            f"        现有列: {list(base_df.columns)}"
        )
    print(f"[INFO] base 列核验通过: {base.column}")

    group_by = str(args.group_by or "").strip() or None
    if group_by and group_by not in base_df.columns:
        print(
            f"[WARN] 分组列 {group_by!r} 不存在，跳过分组统计；"
            f"现有列: {list(base_df.columns)}"
        )
        group_by = None
    if group_by:
        enums = sorted(
            {_group_label(v) for v in base_df[group_by].tolist()},
            key=lambda x: (x == "(空)", x),
        )
        print(f"[INFO] 分组列 {group_by!r} 枚举值 ({len(enums)}): {enums}")

    df = _build_frame(base, hyps, loaded, args.join_key, group_by=group_by)
    require_ref = bool(args.skip_empty_base)

    print("=" * 56)
    summaries: list[dict] = []
    for hyp in hyps:
        prefix = _prefix_for(base, hyp)
        totals = _attach(
            df,
            base.label,
            hyp.label,
            prefix,
            require_ref=require_ref,
            want_acc=args.char_acc,
            want_edit=args.edit_distance,
            group_by=group_by,
        )
        name = f"{hyp.column} ← {base.column}"
        if base.path.resolve() != hyp.path.resolve():
            name = (
                f"{hyp.path.name}:{hyp.column} ← {base.path.name}:{base.column}"
            )
        _print_summary(name, totals)
        summaries.extend(_summary_rows(name, totals, group_by=group_by))
    print("=" * 56)

    if not args.keep_internal:
        drop = [
            c
            for c in df.columns
            if (c.startswith("base__") or c.startswith("hyp"))
            and c not in {base.column, *(h.column for h in hyps)}
        ]
        df = df.drop(columns=drop, errors="ignore")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(args.output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="结果")
        summary_df = pd.DataFrame(summaries)
        summary_df.to_excel(writer, index=False, sheet_name="统计摘要")
        if group_by:
            # 单独一张：每个 hyp × 每个 type + 总计，便于阅读
            summary_df.to_excel(writer, index=False, sheet_name=f"按{group_by}统计")
    print(f"[INFO] 已写入: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
