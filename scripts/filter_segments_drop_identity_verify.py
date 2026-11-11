#!/usr/bin/env python3
"""VAD 切段后、ASR 前：按 xlsx 话术标签去掉「接洽（身份核实）」父源的全部切段。

匹配键（注意不是切段行顶层 ``id``）::

    parquet 父源 ``label_source_audio_id`` / ``labels.source_audio_id``
        ↔ xlsx ``file_id``（也认 ``file-id``）

切段后顶层 ``id`` 是 ``seg_...``，不能拿来对表格的 ``file_id``。
同一个父音频切出的多条片段会一起删掉。xlsx 里没有的父音频、以及标签不是
这一类的，都保留。同一 ``file_id`` 有多行时，只要其中一行是该标签就删。

执行顺序固定为两段：

1. 先核对 xlsx 列名、样本值，以及 parquet 父源键与 ``file_id`` 的命中统计
2. 确认匹配字段后，再写出过滤后的 segments parquet

默认：全量备份为同目录 ``segments_<batch>_vad_all.parquet``，过滤结果写回
``segments_<batch>.parquet``，后续 ``--source-name`` 的 ASR / DNSMOS 直接消费。

示例::

    python scripts/filter_segments_drop_identity_verify.py \\
        --source-name "$BATCH" \\
        --xlsx data/raw/分身13-14日志_合并清洗数据_已打标签.xlsx

只看匹配报告、不写盘::

    python scripts/filter_segments_drop_identity_verify.py \\
        --source-name "$BATCH" \\
        --xlsx path/to/labels.xlsx \\
        --dry-run
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from audio_engine.core.source_naming import (  # noqa: E402
    manifest_stem,
    resolve_existing_manifest,
    validate_source_name,
)

_DROP_SCRIPT = ROOT / "scripts" / "drop_qwen_en_identity_verify.py"


def _load_drop_helpers():
    spec = importlib.util.spec_from_file_location("drop_qwen_en_identity_verify", _DROP_SCRIPT)
    if spec is None or spec.loader is None:
        raise SystemExit(f"[ERROR] 无法加载共用脚本: {_DROP_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse_sheet(raw: str | None) -> str | int | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if text.isdigit():
        return int(text)
    return text


def _preview_ids(values: list[Any], *, limit: int = 5) -> list[str]:
    drop = _load_drop_helpers()
    out: list[str] = []
    seen: set[str] = set()
    for item in values:
        text = drop.normalize_id(item)
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
        if len(out) >= limit:
            break
    return out


def _resolve_input_parquet(*, source_name: str | None, parquet: Path | None) -> Path:
    if parquet is not None:
        path = Path(parquet)
        if not path.is_file():
            raise SystemExit(f"[ERROR] 输入 parquet 不存在: {path}")
        return path
    if source_name is None:
        raise SystemExit("[ERROR] 需要 --source-name 或显式 parquet 路径")
    name = validate_source_name(source_name)
    return Path(resolve_existing_manifest(manifest_stem("segments", name)))


def _backup_path(parquet_path: Path) -> Path:
    return parquet_path.with_name(f"{parquet_path.stem}_vad_all{parquet_path.suffix}")


def _print_match_report(
    *,
    xlsx_path: Path,
    sheet_name: str,
    file_id_column: str,
    label_column: str,
    label: str,
    frame,
    drop_ids: set[str],
    xlsx_stats: dict[str, int],
    parquet_path: Path,
    parent_ids: list[str],
    match_source: str,
    parquet_stats: dict[str, Any],
) -> None:
    drop = _load_drop_helpers()
    present = {item for item in parent_ids if item}
    xlsx_ids = {
        drop.normalize_id(value)
        for value in frame[file_id_column].tolist()
        if drop.normalize_id(value)
    }
    parents_in_xlsx = present & xlsx_ids
    parents_not_in_xlsx = present - xlsx_ids
    label_examples = []
    for raw_id, raw_label in zip(
        frame[file_id_column].tolist(), frame[label_column].tolist(), strict=True
    ):
        if drop.normalize_label(raw_label) != drop.normalize_label(label):
            continue
        text = drop.normalize_id(raw_id)
        if text:
            label_examples.append((text, drop.normalize_label(raw_label)))
        if len(label_examples) >= 3:
            break

    print("[MATCH] ========== 1) xlsx 字段核对 ==========")
    print(f"[MATCH] xlsx={xlsx_path}")
    print(f"[MATCH] sheet={sheet_name}")
    print(f"[MATCH] file_id_column={file_id_column!r}")
    print(f"[MATCH] label_column={label_column!r}")
    print(f"[MATCH] drop_label={drop.normalize_label(label)!r}")
    print(
        f"[MATCH] xlsx_rows={xlsx_stats['xlsx_rows']} "
        f"label_rows={xlsx_stats['label_rows']} "
        f"drop_ids={xlsx_stats['drop_ids']} "
        f"blank_id_rows={xlsx_stats['blank_id_rows']}"
    )
    print(f"[MATCH] xlsx_file_id_examples={_preview_ids(frame[file_id_column].tolist())}")
    print(f"[MATCH] drop_label_examples={label_examples}")

    print("[MATCH] ========== 2) parquet 父源键核对 ==========")
    print(f"[MATCH] parquet={parquet_path}")
    print(f"[MATCH] match_key={match_source}")
    print(
        "[MATCH] note=切段顶层 id 是 seg_...；这里用父源 source_audio_id 对 xlsx.file_id"
    )
    print(f"[MATCH] parquet_parent_examples={_preview_ids(parent_ids)}")
    print(
        f"[MATCH] parquet_rows={parquet_stats['parquet_rows']} "
        f"unique_parents={len(present)} blank_parent={parquet_stats['blank_parent']}"
    )

    print("[MATCH] ========== 3) 匹配命中统计 ==========")
    print(
        f"[MATCH] parents_in_xlsx={len(parents_in_xlsx)} "
        f"parents_not_in_xlsx={len(parents_not_in_xlsx)} "
        f"drop_ids_hit={parquet_stats['drop_ids_hit']} "
        f"drop_ids_miss={parquet_stats['drop_ids_miss']}"
    )
    print(
        f"[MATCH] would_drop_rows={parquet_stats['dropped']} "
        f"would_keep_rows={parquet_stats['kept']}"
    )
    if parquet_stats["drop_ids_hit"] == 0 and xlsx_stats["drop_ids"]:
        print(
            f"[MATCH] WARN 父源编号没有对上 file_id。"
            f"parquet 示例={parquet_stats['parent_examples']} "
            f"xlsx 示例={sorted(drop_ids)[:3]}"
        )


def main(argv: list[str] | None = None) -> int:
    drop = _load_drop_helpers()
    parser = argparse.ArgumentParser(
        description=(
            "VAD segments 按父源 source_audio_id 对 xlsx.file_id，"
            "去掉「接洽（身份核实）」后再交给 ASR/DNSMOS"
        ),
    )
    parser.add_argument(
        "parquet",
        nargs="?",
        type=Path,
        default=None,
        help="segments_*.parquet；与 --source-name 二选一",
    )
    parser.add_argument(
        "--source-name",
        default=None,
        help="批次名；解析 datasets/stage1/segments/.../segments_<batch>.parquet",
    )
    parser.add_argument(
        "--xlsx",
        type=Path,
        required=True,
        help="含 file_id 和 AI话术分类标签 的表格",
    )
    parser.add_argument(
        "--sheet",
        default=None,
        help="工作表名或从 0 开始的序号。默认自动找含这两列的表",
    )
    parser.add_argument(
        "--label",
        default=drop.DROP_LABEL,
        help=f"要删除的 AI话术分类标签，默认 {drop.DROP_LABEL}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="过滤结果路径。默认写回输入 segments parquet",
    )
    parser.add_argument(
        "--backup",
        type=Path,
        default=None,
        help="全量备份路径。默认 <stem>_vad_all.parquet；与 --no-backup 互斥",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="不备份全量 segments（仍建议先 --dry-run）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印匹配核对与命中统计，不写盘",
    )
    parser.add_argument(
        "--allow-zero-hits",
        action="store_true",
        help="xlsx 有待删 file_id 但 parquet 一个都没命中时仍继续写盘",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="覆盖已存在的输出 / 备份文件",
    )
    args = parser.parse_args(argv)

    if args.source_name and args.parquet is not None:
        raise SystemExit("[ERROR] --source-name 与显式 parquet 路径不要同时传")
    if args.no_backup and args.backup is not None:
        raise SystemExit("[ERROR] --no-backup 与 --backup 互斥")

    parquet_path = _resolve_input_parquet(source_name=args.source_name, parquet=args.parquet)
    xlsx_path = Path(args.xlsx)
    if not xlsx_path.is_file():
        raise SystemExit(f"[ERROR] 输入 xlsx 不存在: {xlsx_path}")
    if xlsx_path.suffix.lower() not in {".xlsx", ".xlsm"}:
        raise SystemExit(f"[ERROR] 输入必须是 .xlsx: {xlsx_path}")

    frame, sheet_name = drop.load_sheet(xlsx_path, _parse_sheet(args.sheet))
    file_id_column = drop._find_column(list(frame.columns), drop.FILE_ID_COLUMNS, "file_id")
    label_column = drop._find_column(list(frame.columns), (drop.LABEL_COLUMN,), "话术分类")
    drop_ids, xlsx_stats = drop.drop_file_ids(frame, label=args.label)

    table = pq.read_table(parquet_path)
    parent_ids, match_source = drop.parent_audio_ids(table)
    filtered, parquet_stats = drop.filter_table(table, drop_ids)

    _print_match_report(
        xlsx_path=xlsx_path,
        sheet_name=sheet_name,
        file_id_column=file_id_column,
        label_column=label_column,
        label=args.label,
        frame=frame,
        drop_ids=drop_ids,
        xlsx_stats=xlsx_stats,
        parquet_path=parquet_path,
        parent_ids=parent_ids,
        match_source=match_source,
        parquet_stats=parquet_stats,
    )

    if args.dry_run:
        print("[OK] dry-run：匹配已核对，未写盘")
        return 0

    if parquet_stats["drop_ids_hit"] == 0 and xlsx_stats["drop_ids"] and not args.allow_zero_hits:
        raise SystemExit(
            "[ERROR] 待删 file_id 在 parquet 父源上 0 命中。"
            "请先核对匹配字段；确认无误可加 --allow-zero-hits"
        )

    out_path = Path(args.output) if args.output is not None else parquet_path
    backup_path = None if args.no_backup else (
        Path(args.backup) if args.backup is not None else _backup_path(parquet_path)
    )

    if backup_path is not None:
        if backup_path.resolve() == parquet_path.resolve():
            raise SystemExit("[ERROR] 备份路径不能与输入 parquet 相同")
        if backup_path.exists() and not args.force:
            raise SystemExit(f"[ERROR] 备份已存在（加 --force 覆盖）: {backup_path}")
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(parquet_path, backup_path)
        print(f"[INFO] backup_full={backup_path}")

    if out_path.resolve() != parquet_path.resolve() and out_path.exists() and not args.force:
        raise SystemExit(f"[ERROR] 输出已存在（加 --force 覆盖）: {out_path}")

    # 写回原路径时先有备份；原子写避免半截文件被下游读到。
    drop.write_table(filtered, out_path, force=True)
    print(
        f"[OK] dropped={parquet_stats['dropped']} kept={filtered.num_rows} "
        f"cols={filtered.num_columns} -> {out_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
