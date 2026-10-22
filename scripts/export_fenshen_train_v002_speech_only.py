#!/usr/bin/env python3
# coding=utf-8
"""从 gold_fenshen_1015_full_v002 筛 train + speech，写出加速输入 parquet。

筛选：
  label_source_gold_kind == speech
  label_eval_split == train

输出（默认）::

  datasets/stage1/derived/gold_fenshen_1015_train_v002_speech_only.parquet
  datasets/stage1/segments/segments_fenshen_1015_train_v002_speech.parquet
  datasets/stage1/derived/gold_fenshen_1015_train_v002_speech_only.meta.json

``segments_*`` 是 ``augment_batch_speed --source-name`` 的默认输入槽位，必须存在。

服务器加速（需可读 /data2 wav）::

  audio-data pipeline run pipelines/augment_batch_speed.yaml \\
    --source-name fenshen_1015_train_v002_speech \\
    --speed-factor 1.2

产物目录形如::

  datasets/stage1/augmented/fenshen_1015_train_v002_speech/tempo_1p2_<digest>/manifest.parquet

合并训练集::

  python data/training_sets/exploratory/fenshen_1015/train_v008/prepare.py \\
    --augmented-manifest <上述 manifest.parquet>
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audio_engine.core.artifacts import atomic_path  # noqa: E402
from audio_engine.core.manifest import file_sha256  # noqa: E402

SOURCE_NAME = "fenshen_1015_train_v002_speech"
DEFAULT_SRC = ROOT / "datasets" / "stage1" / "derived" / "gold_fenshen_1015_full_v002.parquet"
DEFAULT_OUT = (
    ROOT / "datasets" / "stage1" / "derived" / "gold_fenshen_1015_train_v002_speech_only.parquet"
)
DEFAULT_SEGMENTS = (
    ROOT / "datasets" / "stage1" / "segments" / f"segments_{SOURCE_NAME}.parquet"
)
EXPECTED_ROWS = 5149


def _write_table(table, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(path) as tmp:
        pq.write_table(table, tmp, compression="zstd")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--segments-out",
        type=Path,
        default=DEFAULT_SEGMENTS,
        help="augment_batch_speed --source-name 默认输入槽位",
    )
    parser.add_argument(
        "--expected-rows",
        type=int,
        default=EXPECTED_ROWS,
        help="筛后期望行数；不符则失败（0=不检查）",
    )
    args = parser.parse_args()

    src = args.src.resolve()
    out = args.out.resolve()
    segments_out = args.segments_out.resolve()
    if not src.is_file():
        raise SystemExit(f"[ERROR] missing source parquet: {src}")

    table = pq.read_table(src)
    kind = pc.equal(table["label_source_gold_kind"], "speech")
    split = pc.equal(table["label_eval_split"], "train")
    filtered = table.filter(pc.and_(kind, split))
    n = filtered.num_rows
    if args.expected_rows and n != args.expected_rows:
        raise SystemExit(
            f"[ERROR] filtered rows={n}, expected {args.expected_rows} "
            f"(src={src})"
        )

    ids = filtered.column("id").to_pylist()
    if len(ids) != len(set(ids)):
        raise SystemExit("[ERROR] duplicate id in filtered parquet")

    _write_table(filtered, out)
    _write_table(filtered, segments_out)

    accel_cmd = (
        "audio-data pipeline run pipelines/augment_batch_speed.yaml "
        f"--source-name {SOURCE_NAME} "
        "--speed-factor 1.2"
    )
    meta = {
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "name": "fenshen_1015_train_v002_speech_only",
        "purpose": "tempo_augment_input",
        "sample_count": n,
        "filter": {
            "label_source_gold_kind": "speech",
            "label_eval_split": "train",
        },
        "source": {
            "parquet": str(src),
            "sha256": file_sha256(src),
            "rows": table.num_rows,
        },
        "output": {
            "parquet": str(out),
            "sha256": file_sha256(out),
            "segments_parquet": str(segments_out),
            "segments_sha256": file_sha256(segments_out),
        },
        "accelerate": {
            "pipeline": "pipelines/augment_batch_speed.yaml",
            "source_name": SOURCE_NAME,
            "speed_factor": 1.2,
            "default_input": str(segments_out),
            "command": accel_cmd,
            "product_root_glob": (
                f"datasets/stage1/augmented/{SOURCE_NAME}/"
                "tempo_1p2_*/manifest.parquet"
            ),
        },
        "merge": {
            "trainset": "exploratory/fenshen_1015/train_v008",
            "prepare": (
                "python data/training_sets/exploratory/fenshen_1015/train_v008/prepare.py "
                "--augmented-manifest <product_root>/manifest.parquet"
            ),
        },
        "schema_columns": filtered.column_names,
    }
    meta_path = out.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"[OK] rows={n} -> {out}")
    print(f"[OK] segments slot -> {segments_out}")
    print(f"[OK] meta -> {meta_path}")
    print(f"[NEXT] {accel_cmd}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
