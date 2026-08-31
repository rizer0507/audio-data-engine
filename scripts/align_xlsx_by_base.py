#!/usr/bin/env python3
"""按 base 表某一列的顺序，对齐并重排多个 xlsx。

以 ``--base 路径.xlsx::列名`` 的行顺序为基准；其余 ``--xlsx`` 按同名（或各自指定的）
键列匹配后重排整表。默认 left：只保留 base 中出现的键，顺序与 base 一致。

列指定格式（避免 Windows 盘符冲突，用双冒号）::

    路径.xlsx::列名

示例::

  # 以 a.xlsx 的 id 顺序为准，重排 b / c 整表，写入输出目录
  python scripts/align_xlsx_by_base.py \
    --base C:/Users/rizer/Desktop/0827-test/test-final-字准率.xlsx::id \
    --xlsx D:/Work/asr数据/数据集/0827/0827-test-all-sensevoice重跑.xlsx \
    --output-dir D:/Work/asr数据/数据集/0827/aligned

  # 合并为一张表（按 base 顺序左连接各表列）
  python scripts/align_xlsx_by_base.py \\
    --base a.xlsx::id \\
    --xlsx b.xlsx \\
    --xlsx c.xlsx \\
    --merge \\
    -o merged.xlsx

  # id 有的带 16 位 hex 前缀、有的已去掉时，匹配前统一剥前缀
  python scripts/align_xlsx_by_base.py \\
    --base a.xlsx::id --xlsx b.xlsx --strip-id-hash -o-dir out
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

_SPEC_RE = re.compile(r"^(?P<path>.+)::(?P<col>.+)$")
_ID_HASH_PREFIX_RE = re.compile(r"^[0-9a-fA-F]{16}_")


@dataclass(frozen=True)
class FileSpec:
    path: Path
    key: str
    role: str  # "base" | "xlsx"


def _parse_spec(raw: str, *, role: str, default_key: str | None = None) -> FileSpec:
    text = (raw or "").strip().strip('"').strip("'")
    match = _SPEC_RE.match(text)
    if match:
        path = Path(match.group("path").strip())
        key = match.group("col").strip()
        if not key:
            raise SystemExit(f"[ERROR] 列名为空: {raw!r}")
        return FileSpec(path=path, key=key, role=role)
    if role == "base":
        raise SystemExit(
            f"[ERROR] --base 格式应为 路径.xlsx::列名，收到: {raw!r}\n"
            f"        例: 数据集/a.xlsx::id"
        )
    if default_key is None:
        raise SystemExit(
            f"[ERROR] --xlsx 未写 ::列名 且无法继承 base 键: {raw!r}"
        )
    return FileSpec(path=Path(text), key=default_key, role=role)


def _as_key(value, *, strip_hash: bool) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return ""
    if strip_hash:
        text = _ID_HASH_PREFIX_RE.sub("", text)
    return text


def _load(spec: FileSpec) -> pd.DataFrame:
    if not spec.path.is_file():
        raise SystemExit(f"[ERROR] 文件不存在: {spec.path}")
    print(f"[INFO] 读取 [{spec.role}] {spec.path}")
    df = pd.read_excel(spec.path, sheet_name=0, dtype=object)
    if spec.key not in df.columns:
        raise SystemExit(
            f"[ERROR] {spec.path.name} 缺少键列 {spec.key!r}\n"
            f"        现有列: {list(df.columns)}"
        )
    print(f"[INFO]   rows={len(df):,} key={spec.key!r} cols={len(df.columns)}")
    return df


def _keyed_frame(
    df: pd.DataFrame,
    key: str,
    *,
    strip_hash: bool,
    keep_duplicates: bool,
) -> pd.DataFrame:
    out = df.copy()
    out["__key__"] = [_as_key(v, strip_hash=strip_hash) for v in out[key]]
    empty = int((out["__key__"] == "").sum())
    if empty:
        print(f"[WARN] {key!r} 有 {empty} 行空键，对齐时忽略这些行")
        out = out[out["__key__"] != ""].copy()
    dup = int(out["__key__"].duplicated().sum())
    if dup:
        if keep_duplicates:
            print(f"[WARN] {key!r} 有 {dup} 条重复键，全部保留（可能一对多）")
        else:
            print(f"[WARN] {key!r} 有 {dup} 条重复键，保留首行")
            out = out.drop_duplicates(subset=["__key__"], keep="first")
    return out


def _align_to_order(
    other: pd.DataFrame,
    order_keys: list[str],
    *,
    how: str,
) -> pd.DataFrame:
    """Reorder ``other`` rows to follow ``order_keys`` (left / inner / outer)."""
    by_key: dict[str, list[pd.Series]] = {}
    for _, row in other.iterrows():
        by_key.setdefault(str(row["__key__"]), []).append(row)

    used: set[str] = set()
    rows: list[pd.Series] = []
    missing = 0

    for key in order_keys:
        bucket = by_key.get(key)
        if not bucket:
            missing += 1
            if how == "inner":
                continue
            # left / outer：键在 base 有、other 没有 → 空行占位（仅保留 __key__）
            empty = pd.Series({col: pd.NA for col in other.columns})
            empty["__key__"] = key
            rows.append(empty)
            continue
        used.add(key)
        rows.extend(bucket)

    extra = 0
    if how == "outer":
        for key, bucket in by_key.items():
            if key in used:
                continue
            rows.extend(bucket)
            extra += 1

    if not rows:
        aligned = other.iloc[0:0].copy()
    else:
        aligned = pd.DataFrame(rows).reset_index(drop=True)

    print(
        f"[INFO]   align how={how}: base_keys={len(order_keys):,} "
        f"matched={len(order_keys) - missing:,} missing_in_other={missing:,} "
        f"extra_appended={extra:,} → rows={len(aligned):,}"
    )
    return aligned


def _drop_helper(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=["__key__"], errors="ignore")


def _unique_out_name(path: Path, used: set[str]) -> str:
    stem = path.stem
    name = f"{stem}_aligned.xlsx"
    if name not in used:
        used.add(name)
        return name
    i = 2
    while True:
        name = f"{stem}_aligned_{i}.xlsx"
        if name not in used:
            used.add(name)
            return name
        i += 1


def _prefix_columns(df: pd.DataFrame, prefix: str, key: str) -> pd.DataFrame:
    """Rename non-key columns with file stem prefix to avoid collisions on merge."""
    rename = {
        col: f"{prefix}__{col}"
        for col in df.columns
        if col not in {key, "__key__"}
    }
    return df.rename(columns=rename)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="按 base xlsx 某列顺序对齐并重排多个 xlsx",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "例:\n"
            "  python scripts/align_xlsx_by_base.py "
            "--base a.xlsx::id --xlsx b.xlsx --xlsx c.xlsx -o-dir out\n"
            "  python scripts/align_xlsx_by_base.py "
            "--base a.xlsx::id --xlsx b.xlsx --merge -o merged.xlsx"
        ),
    )
    parser.add_argument(
        "--base",
        required=True,
        help="基准表：路径.xlsx::键列（决定顺序与键集合）",
    )
    parser.add_argument(
        "--xlsx",
        action="append",
        default=[],
        dest="xlsx_list",
        help="待对齐表：路径.xlsx 或 路径.xlsx::键列（可重复；省略列名则用 base 的键列名）",
    )
    parser.add_argument(
        "--how",
        choices=("left", "inner", "outer"),
        default="left",
        help="left=只保留 base 键顺序；inner=只保留各方都有的键；outer=base 之后追加 other 独有键",
    )
    parser.add_argument(
        "--strip-id-hash",
        action="store_true",
        help="匹配前去掉 id 的 16 位 hex_ 前缀（对齐已清洗/未清洗 id）",
    )
    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="保留重复键的多行（默认每个键只留首行）",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="合并为一张表输出（非键列加文件名前缀）；需 --output",
    )
    parser.add_argument(
        "--output-dir",
        "-o-dir",
        type=Path,
        default=None,
        help="分别写出各表 *_aligned.xlsx 的目录",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="合并输出路径（配合 --merge）；或不配 --merge 时只写出 base 对齐结果",
    )
    parser.add_argument(
        "--include-base",
        action="store_true",
        help="分文件输出时也写出 base 的对齐副本（默认只写 --xlsx）",
    )
    args = parser.parse_args()

    if not args.xlsx_list and not args.merge:
        raise SystemExit("[ERROR] 请至少提供一个 --xlsx")
    if args.merge and args.output is None:
        raise SystemExit("[ERROR] --merge 需要同时指定 --output")
    if not args.merge and args.output_dir is None and args.output is None:
        raise SystemExit("[ERROR] 请指定 --output-dir 或 --output（或 --merge -o）")

    base_spec = _parse_spec(args.base, role="base")
    other_specs = [
        _parse_spec(raw, role="xlsx", default_key=base_spec.key)
        for raw in args.xlsx_list
    ]

    base_df = _keyed_frame(
        _load(base_spec),
        base_spec.key,
        strip_hash=args.strip_id_hash,
        keep_duplicates=args.keep_duplicates,
    )
    order_keys = [str(k) for k in base_df["__key__"].tolist()]
    print(f"[INFO] base 顺序键数: {len(order_keys):,}")

    others: list[tuple[FileSpec, pd.DataFrame]] = []
    for spec in other_specs:
        df = _keyed_frame(
            _load(spec),
            spec.key,
            strip_hash=args.strip_id_hash,
            keep_duplicates=args.keep_duplicates,
        )
        aligned = _align_to_order(df, order_keys, how=args.how)
        others.append((spec, aligned))

    if args.merge:
        if args.how == "inner" and others:
            key_sets = [set(base_df["__key__"])]
            for _, aligned in others:
                key_sets.append(set(aligned["__key__"].dropna().astype(str)))
            keep = set.intersection(*key_sets)
            order_keys = [k for k in order_keys if k in keep]
            base_df = base_df[base_df["__key__"].isin(keep)].copy()
            others = [
                (spec, aligned[aligned["__key__"].isin(keep)].copy())
                for spec, aligned in others
            ]
            print(f"[INFO] inner merge 保留键: {len(order_keys):,}")

        merged = base_df[["__key__"]].copy()
        base_body = _drop_helper(base_df)
        if args.strip_id_hash and base_spec.key in base_body.columns:
            base_body[base_spec.key] = [
                _as_key(v, strip_hash=True) for v in base_body[base_spec.key]
            ]
        merged = merged.merge(
            base_body.assign(__key__=base_df["__key__"].values),
            on="__key__",
            how="left",
        )
        for spec, aligned in others:
            body = aligned.drop(columns=["__key__"], errors="ignore").copy()
            if spec.key in body.columns and spec.key == base_spec.key:
                body = body.drop(columns=[spec.key])
            body = _prefix_columns(body, spec.path.stem, base_spec.key)
            body["__key__"] = aligned["__key__"].values
            # 一对多时 merge 会扩行；先按 __key__ 去重保留首行以免爆炸
            if body["__key__"].duplicated().any() and not args.keep_duplicates:
                body = body.drop_duplicates(subset=["__key__"], keep="first")
            merged = merged.merge(body, on="__key__", how="left")
        merged = _drop_helper(merged)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        merged.to_excel(args.output, index=False)
        print(f"[OK] merged {len(merged):,} rows → {args.output}")
        return 0

    # Per-file output
    out_dir = args.output_dir
    if out_dir is None and args.output is not None:
        # single --output without --merge: only write first other (or base)
        out_dir = args.output.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        target = others[0][1] if others else base_df
        spec = others[0][0] if others else base_spec
        frame = _drop_helper(target)
        if args.strip_id_hash and spec.key in frame.columns:
            frame[spec.key] = [_as_key(v, strip_hash=True) for v in frame[spec.key]]
        frame.to_excel(args.output, index=False)
        print(f"[OK] wrote {len(frame):,} rows → {args.output}")
        return 0

    assert out_dir is not None
    out_dir.mkdir(parents=True, exist_ok=True)
    used_names: set[str] = set()

    to_write: list[tuple[FileSpec, pd.DataFrame]] = list(others)
    if args.include_base:
        to_write.insert(0, (base_spec, base_df))

    for spec, frame in to_write:
        out = _drop_helper(frame)
        if args.strip_id_hash and spec.key in out.columns:
            out[spec.key] = [_as_key(v, strip_hash=True) for v in out[spec.key]]
        # left/outer 占位行：若原键列为空，填回对齐键
        if spec.key in out.columns:
            keys = frame["__key__"].tolist()
            filled = []
            for val, key in zip(out[spec.key].tolist(), keys):
                text = _as_key(val, strip_hash=False)
                filled.append(text if text else key)
            out[spec.key] = filled
        name = _unique_out_name(spec.path, used_names)
        path = out_dir / name
        out.to_excel(path, index=False)
        print(f"[OK] wrote {len(out):,} rows → {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
