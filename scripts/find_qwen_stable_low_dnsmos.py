#!/usr/bin/env python3
r"""按 id 找出 Qwen 双跑稳定有字、DNSMOS SIG 与 OVRL 同时低分的候选。

两份单跑文件（各自只有一个 Qwen transcript key，可同为 qwen）：
    python scripts/find_qwen_stable_low_dnsmos.py \
        --qwen qwen1.parquet qwen2.parquet --dnsmos dnsmos.parquet -o candidates.txt

一份合并双跑文件（自动发现两个 Qwen key，也可显式指定）：
    python scripts/find_qwen_stable_low_dnsmos.py \
        --qwen merged.parquet --run-keys qwen1 qwen2 \
        --dnsmos dnsmos.parquet -o candidates.txt

稳定：两路成功、清洗后的 text 非空，按项目现有 harmless_equivalent 比较。
低分：dnsmos_status=success、SIG <= --sig-threshold 且 OVRL <= --ovrl-threshold。
两个阈值默认都是 2.0，分数必须有效（1～5）；这是待人工验证的试筛起点。
输入直接使用已有 VAD 后的结果，不增加 VAD 输入，不重新切段或计算分数。
默认 audio.resampled_16k，缺失时回退 source_path；相对路径以项目根目录解析。
同时输出同名 Parquet，供 audio-label-master 待标池导入；--output-parquet 可另选路径。
--batch 可指定实际批次；不传则保留源数据 batch，缺失时在平台导入界面指定。
Parquet 保留原样本身份，不以父音频摘要或 WAV 文件摘要替换片段摘要。
category=semantic_risk 仅用于平台待标模板路由，原分类保存在 source_category/source_labels。
平台选择 wav_path 为播放字段；转写仅作候选参考，不是人工金标。
只读已有结果，不启动任何推理。候选不等于已确认的噪声或幻觉。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from audio_engine.core.artifacts import atomic_path
from audio_engine.core.evidence.acoustic_facts import sig_is_low
from audio_engine.core.evidence.asr_status import classify_run_status, original_audio_sha256
from audio_engine.core.evidence.family_states import harmless_equivalent
from audio_engine.core.evidence.text import comparison_text
from audio_engine.core.sample import Sample

LINEAGE_FIELDS = (
    "is_vad_segment", "segment_id", "source_audio_id", "parent_pcm_sha256",
    "source_pcm_sha256", "segment_pcm_sha256", "original_audio_sha256", "segment_index",
    "vad_start_ms", "vad_end_ms", "vad_inferred_end_ms", "vad_end_reason",
    "clip_start_sample", "clip_end_sample", "clip_duration_ms", "vad_processing_version",
    "business_record_id", "business_file_id",
)
ANNOTATION_SCHEMA = pa.schema([
    (name, pa.string()) for name in (
        "id", "sample_id", "sha256", "original_audio_sha256", "batch", "wav_path",
        "source_path", "candidate_text", "selected_family", "category", "source_category",
        "business_record_id", "business_file_id", "source_audio_id", "segment_id",
        "qwen_run_1_key", "qwen_run_2_key", "qwen1_text", "qwen2_text",
        "dnsmos_status", "selection_reason", "audio", "labels", "source_labels",
        "transcripts", "quality",
    )
] + [
    ("queue_index", pa.int64()), ("is_vad_segment", pa.bool_()),
    ("duration", pa.float64()), ("dnsmos_sig", pa.float64()),
    ("dnsmos_bak", pa.float64()), ("dnsmos_ovrl", pa.float64()),
    ("sig_threshold", pa.float64()), ("ovrl_threshold", pa.float64()),
])


def as_mapping(value: Any, *, context: str) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{context}: 非法 JSON") from exc
        if value is None:
            return {}
    if not isinstance(value, dict):
        raise TypeError(f"{context}: 需要对象或 JSON 对象字符串")
    return dict(value)


def load_rows(path: Path, *, dnsmos: bool = False) -> dict[str, dict[str, Any]]:
    """仅投影需要的列，分批读取；每个输入文件中的 id 必须唯一。"""
    parquet = pq.ParquetFile(path)
    names = parquet.schema_arrow.names
    if "id" not in names:
        raise ValueError(f"{path}: 缺少 id 列")
    base = {"id", "source_path", "sha256", "audio", "transcripts", "status", "errors"}
    base.update({"labels", "duration", "batch", "category", "original_audio_sha256"})
    base.update(LINEAGE_FIELDS)
    if dnsmos:
        base = {"id", "sha256", "quality"}
    columns = [
        name for name in names
        if name in base
        or (dnsmos and name.startswith(("quality_dnsmos_", "dnsmos_")))
        or (not dnsmos and name.startswith("label_"))
        or (not dnsmos and name.endswith("_text"))
    ]
    result: dict[str, dict[str, Any]] = {}
    for batch in parquet.iter_batches(batch_size=4096, columns=columns):
        for row in batch.to_pylist():
            raw_id = row.get("id")
            if raw_id is None or not str(raw_id).strip():
                raise ValueError(f"{path}: 存在空 id")
            sample_id = str(raw_id).strip()
            if sample_id in result:
                raise ValueError(f"{path}: 重复 id={sample_id}，不能进行一对一对齐")
            row["id"] = sample_id
            for name in (("quality",) if dnsmos else ("audio", "labels", "transcripts", "status", "errors")):
                row[name] = as_mapping(row.get(name), context=f"{path} id={sample_id} {name}")
            if dnsmos:
                for key in ("dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl", "dnsmos_status"):
                    if key not in row["quality"]:
                        row["quality"][key] = row.get(f"quality_{key}", row.get(key))
            else:
                # Canonical nested entries win, including failed/empty entries.
                for key, value in row.items():
                    if key.endswith("_text") and value is not None:
                        route = key[:-5]
                        if route not in row["transcripts"]:
                            row["transcripts"][route] = {"text": value}
            result[sample_id] = row
    return result


def route_keys(rows: dict[str, dict[str, Any]]) -> list[str]:
    return sorted({
        key for row in rows.values() for key in row["transcripts"]
        if key.lower().startswith("qwen")
    })


def choose_keys(
    first: dict[str, dict[str, Any]],
    second: dict[str, dict[str, Any]],
    *,
    merged: bool,
    requested: list[str] | None,
) -> tuple[str, str]:
    if requested:
        keys = tuple(requested)
        if merged and keys[0] == keys[1]:
            raise ValueError("合并文件必须指定两个不同的 --run-keys，不能把同一路当作双跑")
        for rows, key in zip((first, second), keys):
            if rows and not any(key in row["transcripts"] for row in rows.values()):
                raise ValueError(f"找不到转写 key={key}，请检查 --run-keys")
        return keys
    left, right = route_keys(first), route_keys(second)
    if merged and len(left) == 2:
        return left[0], left[1]
    if not merged and len(left) == len(right) == 1:
        return left[0], right[0]
    if not first and not second:
        return "qwen1", "qwen2"
    raise ValueError(
        f"无法唯一确定两次 Qwen 结果：文件1 keys={left}，文件2 keys={right}；"
        "请传 --run-keys <第一次的key> <第二次的key>"
    )


def successful_text(row: dict[str, Any], key: str) -> str:
    entry = row["transcripts"].get(key)
    if isinstance(entry, str):
        entry = {"text": entry}
    if not isinstance(entry, dict):
        return ""
    sample = Sample(
        id=row["id"], source_path=str(row.get("source_path") or ""),
        transcripts={key: entry}, status=row["status"], errors=row["errors"],
    )
    if classify_run_status(sample, key) != "success_text":
        return ""
    # Use persisted cleaned text: raw_text must not revive a cleaned-empty result.
    return comparison_text(entry.get("text"))


def absolute_audio_path(row: dict[str, Any], *, audio_key: str, path_root: Path) -> str:
    value = (
        row.get("source_path") if audio_key == "source_path"
        else row["audio"].get(audio_key) or row.get("source_path")
    )
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"id={row['id']}: 缺少 audio.{audio_key} 和 source_path")
    value = value.strip()
    if "\n" in value or "\r" in value:
        raise ValueError(f"id={row['id']}: 音频路径包含换行，无法写入逐行 TXT")
    # Preserve server absolute paths when inspecting Linux results on Windows (or vice versa).
    if PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute():
        return value
    return str((path_root / value).resolve())


def annotation_row(
    row: dict[str, Any], peer: dict[str, Any], score_row: dict[str, Any],
    *, keys: tuple[str, str], wav_path: str, queue_index: int,
    batch: str | None, sig_threshold: float, ovrl_threshold: float, path_root: Path,
) -> dict[str, Any]:
    """Build a pending annotation record; never present the candidate as gold."""
    source_labels = {**peer["labels"], **row["labels"]}
    for source in (peer, row):
        for key, value in source.items():
            if key.startswith("label_") and value is not None:
                source_labels.setdefault(key[6:], value)
        for key in LINEAGE_FIELDS:
            if source.get(key) is not None:
                source_labels.setdefault(key, source[key])
    lineage = {key: source_labels[key] for key in LINEAGE_FIELDS if key in source_labels}
    flag = lineage.get("is_vad_segment")
    is_segment = flag is True or flag == 1 or str(flag).lower() == "true"
    audio = {**peer["audio"], **row["audio"]}
    if is_segment:
        # A missing child must never turn into a playable parent recording.
        parent_paths = [row.get("source_path"), peer.get("source_path")]
        parent_paths.extend(audio.get(key) for key in ("normalized_16k", "parent_resampled_16k", "parent_raw"))
        for parent in parent_paths:
            if parent and absolute_audio_path(
                {"id": row["id"], "source_path": parent},
                audio_key="source_path", path_root=path_root,
            ) == wav_path:
                raise ValueError(f"id={row['id']}: VAD 片段不能导出父录音作为 wav_path，请检查子片段音频路径")
        if not lineage.get("source_audio_id"):
            raise ValueError(f"id={row['id']}: VAD 片段缺少 source_audio_id，无法保留父源关联")
        if lineage.get("segment_id") not in (None, row["id"]):
            raise ValueError(f"id={row['id']}: segment_id 与样本 id 不一致")
        lineage["segment_id"] = row["id"]
    lineage["is_vad_segment"] = is_segment
    sample = Sample(id=row["id"], source_path=str(row.get("source_path") or ""),
                    sha256=str(row.get("sha256") or peer.get("sha256") or score_row.get("sha256") or ""),
                    labels=source_labels)
    digest = row.get("original_audio_sha256") or original_audio_sha256(sample)
    if not digest:
        raise ValueError(f"id={row['id']}: 缺少音频 sha256，不能生成可回填的标注 Parquet")
    for key in ("original_audio_sha256", "segment_pcm_sha256"):
        if is_segment and lineage.get(key) and str(lineage[key]) != str(digest):
            raise ValueError(f"id={row['id']}: {key} 与片段身份摘要不一致")
    source_batch = row.get("batch") or peer.get("batch") or source_labels.get("batch")
    if batch and source_batch and batch != str(source_batch):
        raise ValueError(f"id={row['id']}: --batch 与源数据批次不一致")
    entries = []
    for source, key in zip((row, peer), keys):
        entry = source["transcripts"][key]
        entries.append(dict(entry) if isinstance(entry, dict) else {"text": str(entry)})
    text1, text2 = (str(entry["text"]) for entry in entries)
    quality = score_row["quality"]
    bak = quality.get("dnsmos_bak")
    try:
        bak = float(bak) if bak is not None else None
        if bak is not None and not math.isfinite(bak):
            bak = None
    except (TypeError, ValueError):
        bak = None
    audio["resampled_16k"] = wav_path
    return {
        "id": row["id"], "sample_id": row["id"], "sha256": str(digest),
        "original_audio_sha256": str(digest),
        "batch": str(source_batch or batch) if source_batch or batch else None,
        "queue_index": queue_index, "wav_path": wav_path,
        "source_path": row.get("source_path"), "duration": row.get("duration"),
        "candidate_text": text1, "selected_family": "qwen",
        # Routing category required by the platform's annotation template, not an upstream verdict.
        "category": "semantic_risk",
        "source_category": row.get("category") or source_labels.get("category") or source_labels.get("type"),
        "business_record_id": lineage.get("business_record_id"),
        "business_file_id": lineage.get("business_file_id"),
        "is_vad_segment": is_segment, "segment_id": lineage.get("segment_id"),
        "source_audio_id": lineage.get("source_audio_id"),
        "qwen_run_1_key": keys[0], "qwen_run_2_key": keys[1],
        "qwen1_text": text1, "qwen2_text": text2,
        "transcripts": json.dumps(dict(zip(("qwen_run_1", "qwen_run_2"), entries)), ensure_ascii=False),
        "dnsmos_sig": float(quality["dnsmos_sig"]), "dnsmos_bak": bak,
        "dnsmos_ovrl": float(quality["dnsmos_ovrl"]), "dnsmos_status": quality["dnsmos_status"],
        "sig_threshold": sig_threshold, "ovrl_threshold": ovrl_threshold,
        "selection_reason": "qwen_dual_stable_text_and_low_dnsmos",
        "audio": json.dumps(audio, ensure_ascii=False),
        "labels": json.dumps(lineage, ensure_ascii=False),
        "source_labels": json.dumps(source_labels, ensure_ascii=False),
        "quality": json.dumps(quality, ensure_ascii=False),
    }


def find_candidates(
    qwen_paths: list[Path],
    dnsmos_path: Path,
    output: Path,
    *,
    run_keys: list[str] | None = None,
    sig_threshold: float = 2.0,
    ovrl_threshold: float = 2.0,
    audio_key: str = "resampled_16k",
    path_root: Path = ROOT,
    output_parquet: Path | None = None,
    batch: str | None = None,
) -> dict[str, Any]:
    if len(qwen_paths) not in {1, 2}:
        raise ValueError("--qwen 需要一份合并双跑文件，或两份单跑文件")
    for option, threshold in (("sig-threshold", sig_threshold), ("ovrl-threshold", ovrl_threshold)):
        if not math.isfinite(threshold) or not 1 <= threshold <= 5:
            raise ValueError(f"--{option} 必须是 1～5 的有限数值")
    inputs = [path.resolve() for path in [*qwen_paths, dnsmos_path]]
    if len(set(inputs)) != len(inputs):
        raise ValueError("输入文件不能重复，不能把同一次 Qwen 结果当作两次独立运行")
    output_parquet = output_parquet or output.with_suffix(".parquet")
    if output.resolve() == output_parquet.resolve():
        raise ValueError("TXT 与 Parquet 输出路径不能相同，请使用 --output-parquet")
    if output.resolve() in inputs or output_parquet.resolve() in inputs:
        raise ValueError("输出路径不能覆盖输入 Parquet")
    first = load_rows(qwen_paths[0])
    merged = len(qwen_paths) == 1
    second = first if merged else load_rows(qwen_paths[1])
    dnsmos = load_rows(dnsmos_path, dnsmos=True)
    key1, key2 = choose_keys(first, second, merged=merged, requested=run_keys)
    counts: Counter[str] = Counter()
    paths: list[str] = []
    annotation_rows: list[dict[str, Any]] = []
    for sample_id, row in first.items():
        peer = second.get(sample_id)
        if peer is None:
            continue
        counts["aligned_qwen"] += 1
        score_row = dnsmos.get(sample_id)
        known_hashes = {
            item["sha256"] for item in (row, peer, score_row)
            if item is not None and item.get("sha256")
        }
        if len(known_hashes) > 1:
            raise ValueError(f"id={sample_id}: 输入文件的音频 sha256 不一致，拒绝错误对齐")
        text1, text2 = successful_text(row, key1), successful_text(peer, key2)
        if not text1 or not text2 or not harmless_equivalent(text1, text2):
            continue
        counts["stable_text"] += 1
        if score_row is None:
            counts["missing_dnsmos"] += 1
            continue
        counts["aligned_all"] += 1
        quality = score_row["quality"]
        low, _ = sig_is_low(quality, sig_threshold)
        try:
            score = float(quality.get("dnsmos_sig"))
            ovrl = float(quality.get("dnsmos_ovrl"))
        except (TypeError, ValueError):
            counts["invalid_dnsmos"] += 1
            continue
        if low is None or any(not math.isfinite(value) or not 1 <= value <= 5 for value in (score, ovrl)):
            counts["invalid_dnsmos"] += 1
            continue
        counts["sig_low"] += int(low)
        counts["ovrl_low"] += int(ovrl <= ovrl_threshold)
        if not low or ovrl > ovrl_threshold:
            continue
        wav_path = absolute_audio_path(row, audio_key=audio_key, path_root=path_root)
        annotation_rows.append(annotation_row(
            row, peer, score_row, keys=(key1, key2), wav_path=wav_path,
            queue_index=len(paths) + 1, batch=batch, sig_threshold=sig_threshold,
            ovrl_threshold=ovrl_threshold, path_root=path_root,
        ))
        paths.append(wav_path)
    # One line per matched id; do not merge different ids merely because paths repeat.
    table = pa.Table.from_pylist(annotation_rows, schema=ANNOTATION_SCHEMA)
    with atomic_path(output) as temporary, atomic_path(output_parquet) as temporary_parquet:
        temporary.write_text("".join(path + "\n" for path in paths), encoding="utf-8")
        pq.write_table(table, temporary_parquet, compression="zstd")
    return {
        "qwen_rows": [len(first), len(second)], "run_keys": [key1, key2],
        "dnsmos_rows": len(dnsmos), "aligned_qwen": counts["aligned_qwen"],
        "qwen_unaligned": len(set(first) ^ set(second)),
        "stable_text": counts["stable_text"], "aligned_all": counts["aligned_all"],
        "missing_dnsmos": counts["missing_dnsmos"],
        "sig_low": counts["sig_low"], "ovrl_low": counts["ovrl_low"],
        "invalid_dnsmos": counts["invalid_dnsmos"], "matched": len(paths),
        "output": str(output.resolve()),
        "output_parquet": str(output_parquet.resolve()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--qwen", nargs="+", type=Path, required=True, help="两份单跑或一份合并双跑 Parquet")
    parser.add_argument("--dnsmos", type=Path, required=True, help="独立 DNSMOS Parquet")
    parser.add_argument("--run-keys", nargs=2, help="两次转写的 key；不传则自动识别")
    parser.add_argument("--sig-threshold", type=float, default=2.0, help="SIG 上限，包含边界，默认 2.0")
    parser.add_argument("--ovrl-threshold", type=float, default=2.0, help="OVRL 上限，包含边界，默认 2.0；两项需同时满足")
    parser.add_argument("--audio-key", default="resampled_16k", help="默认实际识别音频 resampled_16k；可选 source_path")
    parser.add_argument("--path-root", type=Path, default=ROOT, help="相对音频路径的根目录，默认项目目录")
    parser.add_argument("-o", "--output", type=Path, default=Path("qwen_stable_low_dnsmos_paths.txt"))
    parser.add_argument("--output-parquet", type=Path, help="标注 Parquet 路径；默认与 TXT 同名")
    parser.add_argument("--batch", help="实际源批次；未指定且源数据也没有时，在平台导入界面填写")
    args = parser.parse_args(argv)
    try:
        result = find_candidates(
            args.qwen, args.dnsmos, args.output, run_keys=args.run_keys,
            sig_threshold=args.sig_threshold, ovrl_threshold=args.ovrl_threshold,
            audio_key=args.audio_key,
            path_root=args.path_root.resolve(),
            output_parquet=args.output_parquet, batch=args.batch,
        )
    except (OSError, TypeError, ValueError) as exc:
        parser.exit(2, f"错误：{exc}\n")
    print(f"Qwen 两路 key: {result['run_keys'][0]} / {result['run_keys'][1]}")
    print(f"Qwen 两文件记录数: {result['qwen_rows']}；双方对齐: {result['aligned_qwen']}")
    print(f"Qwen 未对齐 ID 数: {result['qwen_unaligned']}")
    print(f"Qwen 双跑稳定有字: {result['stable_text']}")
    print(f"其中缺少 DNSMOS: {result['missing_dnsmos']}；评分失败/缺失/非法: {result['invalid_dnsmos']}")
    print(f"筛选条件: DNSMOS 计算成功且 SIG <= {args.sig_threshold:g} 且 OVRL <= {args.ovrl_threshold:g}")
    print(f"有效双评分中，SIG 低分: {result['sig_low']}；OVRL 低分: {result['ovrl_low']}（最终取交集）")
    print(f"命中总个数: {result['matched']}")
    print(f"绝对路径 TXT（每行一条）: {result['output']}")
    print(f"audio-label-master 标注 Parquet: {result['output_parquet']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
