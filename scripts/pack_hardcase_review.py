#!/usr/bin/env python3
# coding=utf-8
"""把两份 hardcase 切段合成标注平台「转换 parquet」能导入的一张表。

播放的是切段 wav（``audio.resampled_16k``）。``source_path`` 只放父音频。
7000 那份 hardcase 的 ``source_path`` 和 ``audio.normalized_16k`` 写的是切段
自己，原样导入会被当成子音频指向父音频，这里会改掉。

``transcripts.qwen`` 统一指向原来的 ``qwen1`` 或 ``qwen_1``。导入时数据单位
选 VAD 分片，预标注来源勾选 ``qwen``。Ai话术仍由 xlsx 导入。

再切段时 ``label_source_audio_id`` 可能仍是上一条 ``seg_``。父 wav 文件名里
有 ``fenshen-...`` 时，父源改成这个编号，并带上分片序号和 VAD 起止。

示例::

  python scripts/pack_hardcase_review.py
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import pandas as pd

_DEFAULT_0814 = Path("tmp/1023/0814-train/0814-train-hardcase-vad.parquet")
_DEFAULT_7000 = Path("tmp/1023/label_buckets/hardcase.parquet")
_DEFAULT_OUTPUT = Path("tmp/1023/0814-train/hardcase_review.parquet")
_QWEN_SOURCE_KEYS = ("qwen", "qwen1", "qwen_1")
_PARENT_FILE_ID = re.compile(r"(fenshen-[0-9a-fA-F]+-[0-9a-fA-F]+)")


def _loads(value: object) -> dict:
    if isinstance(value, dict):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return {}
    text = str(value).strip()
    if not text:
        return {}
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError(f"expected JSON object, got {type(parsed).__name__}")
    return parsed


def _text(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _optional_int(value: object) -> int | None:
    if value is None or (isinstance(value, float) and (math.isnan(value) or pd.isna(value))):
        return None
    if isinstance(value, bool):
        raise ValueError(f"expected integer, got bool")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value) or value != int(value):
            raise ValueError(f"expected integer, got {value}")
        return int(value)
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    number = float(text)
    if not math.isfinite(number) or number != int(number):
        raise ValueError(f"expected integer, got {value}")
    return int(number)


def parent_file_id(path: str | None) -> str:
    """父 wav 文件名里的 ``fenshen-<hex>-<hex>``。没有则返回空串。"""
    match = _PARENT_FILE_ID.search(_text(path))
    return match.group(1) if match else ""


def resolve_source_audio_id(source_audio_id: str, parent_path: str | None) -> str:
    """再切段留下的 ``seg_`` 父源，改成父 wav 文件名里的 ``fenshen-...``。"""
    file_id = parent_file_id(parent_path)
    if source_audio_id.startswith("seg_") and file_id:
        return file_id
    return source_audio_id


def _child_wav(audio: dict) -> str:
    return _text(audio.get("resampled_16k"))


def _parent_wav(row: pd.Series, audio: dict, child: str) -> str | None:
    candidates = (
        row.get("source_path"),
        audio.get("parent_resampled_16k"),
        row.get("label_normalized_audio_path"),
        audio.get("normalized_16k"),
        audio.get("parent_raw"),
    )
    for raw in candidates:
        path = _text(raw)
        if path and path != child:
            return path
    return None


def _with_qwen(transcripts: dict) -> dict:
    out = dict(transcripts)
    if _text((out.get("qwen") or {}).get("text") if isinstance(out.get("qwen"), dict) else out.get("qwen")):
        return out
    for key in _QWEN_SOURCE_KEYS[1:]:
        entry = out.get(key)
        if entry is None:
            continue
        text = entry.get("text") if isinstance(entry, dict) else entry
        if _text(text):
            out["qwen"] = entry
            return out
    return out


def _row(row: pd.Series, review_source: str) -> dict:
    audio = _loads(row.get("audio"))
    child = _child_wav(audio)
    if not child:
        raise ValueError(f"{row.get('id')}: missing audio.resampled_16k")
    sample_id = _text(row.get("id"))
    segment_id = _text(row.get("label_segment_id")) or sample_id
    if segment_id != sample_id:
        raise ValueError(f"{sample_id}: segment_id {segment_id} != id")
    source_audio_id = _text(row.get("label_source_audio_id"))
    if not source_audio_id:
        raise ValueError(f"{sample_id}: missing source_audio_id")
    if source_audio_id == sample_id:
        raise ValueError(f"{sample_id}: source_audio_id equals child id")
    sha = _text(row.get("sha256"))
    if not sha:
        raise ValueError(f"{sample_id}: missing sha256")
    parent = _parent_wav(row, audio, child)
    source_audio_id = resolve_source_audio_id(source_audio_id, parent)
    if source_audio_id == sample_id:
        raise ValueError(f"{sample_id}: source_audio_id equals child id")
    parent_pcm = _text(row.get("label_parent_pcm_sha256")) or _text(row.get("label_source_pcm_sha256"))
    transcripts = _with_qwen(_loads(row.get("transcripts")))
    return {
        "id": sample_id,
        "segment_id": sample_id,
        "is_vad_segment": True,
        "source_audio_id": source_audio_id,
        "parent_pcm_sha256": parent_pcm or None,
        "sha256": sha,
        "source_path": parent,
        "audio": json.dumps({"resampled_16k": child}, ensure_ascii=False),
        "transcripts": json.dumps(transcripts, ensure_ascii=False),
        "duration": None if pd.isna(row.get("duration")) else float(row.get("duration")),
        "review_source": review_source,
        "segment_index": _optional_int(row.get("label_segment_index")),
        "vad_start_ms": _optional_int(row.get("label_vad_start_ms")),
        "vad_end_ms": _optional_int(row.get("label_vad_end_ms")),
        "vad_inferred_end_ms": _optional_int(row.get("label_vad_inferred_end_ms")),
        "vad_end_reason": _text(row.get("label_vad_end_reason")) or None,
    }


_BUCKET_LINEAGE = (
    "label_segment_index",
    "label_vad_start_ms",
    "label_vad_end_ms",
    "label_vad_inferred_end_ms",
    "label_vad_end_reason",
)


def repair_packed_parents(packed: pd.DataFrame, bucket: pd.DataFrame) -> pd.DataFrame:
    """把已打包标注表里的 ``seg_`` 父源改回父 wav 上的 ``fenshen-...``，并补上分片区间。"""
    missing_cols = [name for name in ("id", "source_audio_id", "source_path") if name not in packed.columns]
    if missing_cols:
        raise ValueError(f"packed parquet missing {missing_cols}")
    needed = ["id", *_BUCKET_LINEAGE]
    missing_bucket = [name for name in needed if name not in bucket.columns]
    if missing_bucket:
        raise ValueError(f"bucket parquet missing {missing_bucket}")
    merged = packed.merge(
        bucket[needed],
        on="id",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    unmatched = merged["_merge"] != "both"
    if unmatched.any():
        ids = merged.loc[unmatched, "id"].head(5).tolist()
        raise ValueError(f"{int(unmatched.sum())} rows missing from bucket, e.g. {ids}")
    merged = merged.drop(columns=["_merge"])
    old = merged["source_audio_id"].map(_text)
    merged["source_audio_id"] = [
        resolve_source_audio_id(source_id, path)
        for source_id, path in zip(old, merged["source_path"], strict=True)
    ]
    unresolved = merged["source_audio_id"].map(_text).str.startswith("seg_")
    if unresolved.any():
        ids = merged.loc[unresolved, "id"].head(5).tolist()
        raise ValueError(f"{int(unresolved.sum())} rows still have seg_ parent, e.g. {ids}")
    merged["segment_index"] = merged["label_segment_index"].map(_optional_int)
    merged["vad_start_ms"] = merged["label_vad_start_ms"].map(_optional_int)
    merged["vad_end_ms"] = merged["label_vad_end_ms"].map(_optional_int)
    merged["vad_inferred_end_ms"] = merged["label_vad_inferred_end_ms"].map(_optional_int)
    merged["vad_end_reason"] = merged["label_vad_end_reason"].map(lambda value: _text(value) or None)
    return merged.drop(columns=list(_BUCKET_LINEAGE))


def build_frame(path_0814: Path, path_7000: Path) -> pd.DataFrame:
    left = pd.read_parquet(path_0814)
    right = pd.read_parquet(path_7000)
    rows = [_row(row, "0814-train") for _, row in left.iterrows()]
    rows.extend(_row(row, "label-7000") for _, row in right.iterrows())
    frame = pd.DataFrame(rows)
    dup = frame["id"].duplicated()
    if dup.any():
        ids = frame.loc[dup, "id"].head(5).tolist()
        raise ValueError(f"duplicate id: {ids}")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-0814", type=Path, default=_DEFAULT_0814)
    parser.add_argument("--input-7000", type=Path, default=_DEFAULT_7000)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument(
        "--repair",
        type=Path,
        help="已打包的标注 parquet。按 id 回源头桶，改父源并补分片区间，写到 --output",
    )
    parser.add_argument(
        "--bucket",
        type=Path,
        default=_DEFAULT_7000,
        help="带 label_source_audio_id / label_vad_* 的源头 hardcase parquet",
    )
    args = parser.parse_args()
    if args.repair is not None:
        packed = pd.read_parquet(args.repair)
        before = packed["source_audio_id"].map(_text)
        frame = repair_packed_parents(packed, pd.read_parquet(args.bucket))
        changed = int((before.to_numpy() != frame["source_audio_id"].map(_text).to_numpy()).sum())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(args.output, index=False)
        print(f"rows={len(frame)} rewritten_parents={changed} -> {args.output}")
        return
    frame = build_frame(args.input_0814, args.input_7000)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False)
    counts = frame["review_source"].value_counts().to_dict()
    print(f"rows={len(frame)} {counts} -> {args.output}")


if __name__ == "__main__":
    main()
