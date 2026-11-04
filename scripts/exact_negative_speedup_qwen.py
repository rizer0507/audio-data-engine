#!/usr/bin/env python3
# coding=utf-8
"""否定词精准命中后保音高加速，再用 5555 上的 Qwen 转写，筛出字准不为 1 的标注 parquet。

输入是已经去掉「接洽（身份核实）」的两份 prepared_evidence parquet。
某一路转写去掉标点、空白、控制标签后，整段等于否定词表中的一条，该 VAD 切片准入。
label 就是这条命中的原文。两个文件按 id 去重后合成一份，交给保音高加速。

加速倍率默认 1.4，流水线与 ``scripts/speedup_negative_qwen_diff.py`` 相同。
转写前请求 ``/v1/models``。端口不通就不转写。

字准比较的是 label 和加速后的 Qwen。两边都先做 plain，再删掉
``configs/normalization/fenshen_1015_filler_tokens.yaml`` 里的语气词。
删完之后编辑距离仍不为 0（字准不为 1）才进入标注 parquet。
只差语气词的行字准为 1，不进入。

标注 parquet 给 audio-label-master「转换 parquet」导入：

- ``id`` / ``sha256`` 是原 VAD 切片
- ``audio.resampled_16k`` 是加速后的 wav
- ``source_path`` 是与播放音频不同的父音频路径
- ``source_audio_id`` 是父音频，用来对 AI 话术
- ``transcripts.exact_negative_phrase`` 是准入时命中的否定词，导入时预标注来源选这一路
- ``transcripts.qwen_1p4`` 是加速后的 Qwen，只作对照
- ``transcripts.<加速前 qwen 键>`` 是加速前的 Qwen
- ``audio_before`` / ``audio_after`` / ``qwen_before`` / ``qwen_after`` / ``label`` / ``char_acc`` 是对照列

示例::

  python scripts/exact_negative_speedup_qwen.py ^
    --output-dir tmp/1027/exact_negative_speed1p4

只出准入清单、不加速::

  python scripts/exact_negative_speedup_qwen.py --filter-only
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import filter_prepared_evidence_negative_en as exact_filter  # noqa: E402
import speedup_negative_qwen_diff as speedup  # noqa: E402
from audio_engine.core.artifacts import atomic_path, atomic_write_json  # noqa: E402
from audio_engine.core.augmentation.params import factor_tag, normalize_speed_factor  # noqa: E402
from audio_engine.core.manifest import Manifest  # noqa: E402
from audio_engine.core.sample import Sample  # noqa: E402
from audio_engine.core.source_naming import validate_source_name  # noqa: E402
from audio_engine.core.stage1.identity import fetch_model_identity  # noqa: E402
from audio_engine.core.transcript_reconcile import levenshtein_ops, plain_transcript_text  # noqa: E402

DEFAULT_INPUTS = (
    ROOT / "datasets" / "stage1" / "derived" / "prepared_evidence_20260814-1-n-s.parquet",
    ROOT / "datasets" / "stage1" / "derived" / "prepared_evidence_20260813-1-n-s.parquet",
)
DEFAULT_API_BASE = "http://127.0.0.1:5555"
DEFAULT_SOURCE_NAME = "exact_negative_20260813_20260814"
HIT_ROUTE_KEY = "exact_negative_route"
HIT_PHRASE_KEY = "exact_negative_phrase"
HIT_LABEL_KEY = "exact_negative_label"
QWEN_OPERATOR = speedup.QWEN_OPERATOR


def best_hit(routes: dict[str, str]) -> tuple[str, str] | None:
    """多路同时精准命中时保留最长的一条，并记下是哪一路。"""
    chosen_route: str | None = None
    chosen_phrase: str | None = None
    for name, text in routes.items():
        found = exact_filter.exact_negative(text)
        if found is None:
            continue
        if chosen_phrase is None or len(found) > len(chosen_phrase):
            chosen_route = name
            chosen_phrase = found
    if chosen_route is None or chosen_phrase is None:
        return None
    return chosen_route, chosen_phrase


def char_accuracy(reference: str, hypothesis: str) -> float:
    """plain 并删掉语气词之后的字准。1 表示删完后没有编辑距离。"""
    left = speedup.drop_spoken_tokens(plain_transcript_text(reference))
    right = speedup.drop_spoken_tokens(plain_transcript_text(hypothesis))
    ops = levenshtein_ops(left, right)
    return round(max(0.0, 1.0 - float(ops["cer"] or 0.0)), 6)


def char_acc_mismatch(reference: str, hypothesis: str) -> tuple[bool, float]:
    """字准不为 1 时返回 True。"""
    accuracy = char_accuracy(reference, hypothesis)
    return accuracy != 1.0, accuracy


def _labels_object(value: Any) -> dict[str, Any]:
    parsed = exact_filter._load_mapping(value)
    return dict(parsed) if parsed else {}


def filter_exact_table(table: pa.Table) -> tuple[pa.Table, Counter[str]]:
    """留下精准命中的行，并把命中的路和 label 写进 labels。"""
    names = table.schema.names
    route_columns = [name for name in names if exact_filter.is_route_text_column(name)]
    needed = list(route_columns)
    if "transcripts" in names and "transcripts" not in needed:
        needed.append("transcripts")
    if "labels" not in names:
        raise SystemExit("[ERROR] 输入 parquet 没有 labels 列，无法记下命中的 label")
    if not needed:
        raise SystemExit(
            "[ERROR] 找不到转写列。需要 transcripts，或任意一路的 *_text 列。"
            f" 现有列: {names}"
        )
    columns = {name: table.column(name).to_pylist() for name in needed}
    labels_in = table.column("labels").to_pylist()
    keep_index: list[int] = []
    stamped: list[str] = []
    hits: Counter[str] = Counter()
    for index in range(table.num_rows):
        values = {name: columns[name][index] for name in needed}
        found = best_hit(exact_filter.route_plains(values, route_columns))
        if found is None:
            continue
        route, phrase = found
        labels = _labels_object(labels_in[index])
        labels[HIT_ROUTE_KEY] = route
        labels[HIT_PHRASE_KEY] = phrase
        labels[HIT_LABEL_KEY] = phrase
        keep_index.append(index)
        stamped.append(json.dumps(labels, ensure_ascii=False))
        hits[phrase] += 1
    filtered = table.take(pa.array(keep_index, type=pa.int64()))
    if filtered.num_rows:
        filtered = filtered.set_column(
            filtered.schema.get_field_index("labels"),
            "labels",
            pa.array(stamped, type=table.schema.field("labels").type),
        )
    return filtered, hits


def concat_tables(tables: list[pa.Table]) -> pa.Table:
    if len(tables) == 1:
        return tables[0]
    try:
        return pa.concat_tables(tables, promote_options="permissive")
    except TypeError:
        return pa.concat_tables(tables, promote=True)


def dedupe_by_id(table: pa.Table) -> tuple[pa.Table, int]:
    if "id" not in table.schema.names or table.num_rows == 0:
        return table, 0
    seen: set[str] = set()
    mask: list[bool] = []
    dropped = 0
    for item in table.column("id").to_pylist():
        key = "" if item is None else str(item)
        if key in seen:
            mask.append(False)
            dropped += 1
            continue
        seen.add(key)
        mask.append(True)
    return table.filter(pa.array(mask, type=pa.bool_())), dropped


def write_admitted(tables: list[tuple[Path, pa.Table]], output: Path) -> dict[str, Any]:
    pieces: list[pa.Table] = []
    total_hits: Counter[str] = Counter()
    per_file: list[dict[str, Any]] = []
    for path, table in tables:
        filtered, hits = filter_exact_table(table)
        total_hits.update(hits)
        pieces.append(filtered)
        per_file.append({"path": str(path), "rows": table.num_rows, "hits": filtered.num_rows})
        detail = ", ".join(f"{phrase}={count}" for phrase, count in hits.most_common())
        speedup._info(f"{path.name} rows={table.num_rows} hits={filtered.num_rows}")
        if detail:
            speedup._info(f"  {detail}")
    if not pieces or all(piece.num_rows == 0 for piece in pieces):
        raise SystemExit("[ERROR] 两份 parquet 都没有精准命中否定词的行")
    merged = concat_tables(pieces)
    merged, dropped = dedupe_by_id(merged)
    output.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(output) as tmp:
        pq.write_table(merged, tmp)
    speedup._info(f"admitted {merged.num_rows} rows, dropped_duplicate_id={dropped} -> {output}")
    return {
        "admitted": str(output),
        "rows": merged.num_rows,
        "dropped_duplicate_id": dropped,
        "files": per_file,
        "hits": dict(total_hits),
    }


def assert_qwen_reachable(
    api_base: str,
    model: str | None,
    *,
    timeout_s: float = 5.0,
    fetch: Any = fetch_model_identity,
) -> list[str]:
    """转写前确认 /v1/models 能返回。配置了模型名时，服务里必须有这个 id。"""
    try:
        identity = fetch(api_base, timeout_s=timeout_s)
    except Exception as exc:
        raise SystemExit(f"[ERROR] Qwen 端口未联通: {api_base} ({exc})") from exc
    model_ids = [str(item) for item in identity.model_ids]
    if model and not identity.matches(model):
        raise SystemExit(
            f"[ERROR] Qwen 已连通，但没有模型 {model!r}。"
            f" 实际 ids={model_ids} @ {api_base}"
        )
    speedup._info(f"qwen reachable models={model_ids} @ {api_base}")
    return model_ids


def longest_contained_phrase(text: str) -> str | None:
    """整段相等优先；否则在 plain 文本里取最长的否定词子串。"""
    plain = plain_transcript_text(text)
    if not plain:
        return None
    exact = exact_filter.exact_negative(plain)
    if exact is not None:
        return exact
    best: str | None = None
    for phrase in exact_filter.PHRASE_SET:
        if phrase in plain and (best is None or len(phrase) > len(best)):
            best = phrase
    return best


def resolve_preannotation_phrase(archived: str, *texts: str) -> str:
    """预标注用准入时命中的否定词。没有归档命中时，再从转写里找最长的一条。"""
    stored = str(archived or "").strip()
    if stored:
        return stored
    best: str | None = None
    for text in texts:
        found = longest_contained_phrase(text)
        if found is not None and (best is None or len(found) > len(best)):
            best = found
    return "" if best is None else best


def _hit_label(sample: Sample, source: dict[str, Any]) -> str:
    value = speedup._archived_label(sample, source, HIT_LABEL_KEY)
    if value is None:
        value = speedup._archived_label(sample, source, HIT_PHRASE_KEY)
    return "" if value is None else str(value)


def review_row(
    sample: Sample,
    *,
    before_key: str,
    after_key: str,
) -> dict[str, Any]:
    """一条加速样本：对照 label 算字准，并留下加速前后的路径和 Qwen 文本。"""
    source = speedup._source_namespace(sample)
    source_transcripts = source.get("transcripts") or {}
    if not isinstance(source_transcripts, dict):
        source_transcripts = {}
    source_audio = source.get("audio") or {}
    if not isinstance(source_audio, dict):
        source_audio = {}

    status = str((sample.status or {}).get(QWEN_OPERATOR) or "")
    error = str((sample.errors or {}).get(QWEN_OPERATOR) or "")
    before_missing = before_key not in source_transcripts
    before_stored = "" if before_missing else speedup._entry_text(source_transcripts.get(before_key))
    after_entry = (sample.transcripts or {}).get(after_key)
    after_missing = after_entry is None
    after_stored = "" if after_missing else speedup._entry_text(after_entry)
    label = _hit_label(sample, source)
    route = speedup._archived_label(sample, source, HIT_ROUTE_KEY)
    phrase = speedup._archived_label(sample, source, HIT_PHRASE_KEY)

    accuracy: float | None = None
    if not label:
        bucket = "failed"
        if not error:
            error = f"missing archived label {HIT_LABEL_KEY}"
    elif status != "completed" or after_missing:
        bucket = "failed"
        if not error:
            error = status or f"missing transcript {after_key}"
    else:
        changed, accuracy = char_acc_mismatch(label, after_stored)
        bucket = "mismatch" if changed else "same"

    audio_after = str(
        (sample.audio or {}).get("aug_speed")
        or (sample.audio or {}).get("resampled_16k")
        or ""
    )
    audio_before = str(source_audio.get("resampled_16k") or "")
    source_id = str(source.get("sample_id") or sample.id)
    source_sha = str(source.get("sha256") or "")
    source_audio_id = speedup._archived_label(sample, source, "source_audio_id")
    parent_pcm = speedup._archived_label(sample, source, "parent_pcm_sha256")
    if parent_pcm is None:
        parent_pcm = speedup._archived_label(sample, source, "source_pcm_sha256")
    parent_path = speedup._archived_label(sample, source, "parent_source_path")
    if parent_path is None:
        parent_path = speedup._archived_label(sample, source, "normalized_audio_path")
    parent_path_text = None if parent_path is None else str(parent_path)
    if parent_path_text and audio_after and parent_path_text == audio_after:
        parent_path_text = None

    preannotation = resolve_preannotation_phrase(
        "" if phrase is None else str(phrase),
        label,
        before_stored,
        after_stored,
    )
    transcripts: dict[str, Any] = {}
    if preannotation:
        transcripts[HIT_PHRASE_KEY] = {"text": preannotation}
    if before_key:
        transcripts[before_key] = {"text": before_stored}
    if after_key and after_key != before_key:
        transcripts[after_key] = {"text": after_stored}
    audio = {"resampled_16k": audio_after} if audio_after else {}
    return {
        "id": source_id,
        "source_id": source_id,
        "segment_id": source_id,
        "is_vad_segment": True,
        "source_audio_id": None if source_audio_id is None else str(source_audio_id),
        "parent_pcm_sha256": None if parent_pcm is None else str(parent_pcm),
        "derived_id": sample.id,
        "bucket": bucket,
        "label": label,
        "matched_route": None if route is None else str(route),
        "matched_phrase": None if phrase is None else str(phrase),
        "char_acc": accuracy,
        "qwen_before_key": before_key,
        "qwen_after_key": after_key,
        "qwen_before": before_stored,
        "qwen_after": after_stored,
        "audio_before": audio_before,
        "audio_after": audio_after,
        "qwen_status": status,
        "error": error,
        "source_path": parent_path_text,
        "candidate_text": before_stored,
        "audio": json.dumps(audio, ensure_ascii=False),
        "transcripts": json.dumps(transcripts, ensure_ascii=False),
        "sha256": source_sha,
        "speed_sha256": sample.sha256 or "",
        "duration": sample.duration if sample.duration is not None else source.get("duration"),
    }


def write_review(
    samples: list[Sample],
    *,
    output_dir: Path,
    tag: str,
    before_key: str,
    after_key: str,
    meta: dict[str, Any],
) -> dict[str, Any]:
    rows = [review_row(sample, before_key=before_key, after_key=after_key) for sample in samples]
    compare_path = output_dir / f"qwen_{tag}_compare.parquet"
    mismatch_path = output_dir / f"qwen_{tag}_char_acc_ne1.parquet"
    failed_path = output_dir / f"qwen_{tag}_failed.parquet"
    speedup._write_parquet(rows, compare_path)
    speedup._write_parquet([row for row in rows if row["bucket"] == "mismatch"], mismatch_path)
    speedup._write_parquet([row for row in rows if row["bucket"] == "failed"], failed_path)
    counts = {
        "rows": len(rows),
        "same": sum(row["bucket"] == "same" for row in rows),
        "mismatch": sum(row["bucket"] == "mismatch" for row in rows),
        "failed": sum(row["bucket"] == "failed" for row in rows),
    }
    summary = {
        **meta,
        "counts": counts,
        "compare": str(compare_path),
        "label_parquet": str(mismatch_path),
        "failed": str(failed_path),
        "compare_rule": (
            "参考文本是准入 label（精准命中的那一路原文）。"
            "假设文本是加速后的 Qwen。两边 plain 后删口语语气词，"
            "字准 = 1 - 编辑距离/max(参考字数, 1)。字准不为 1 写入标注 parquet。"
        ),
        "spoken_drop_tokens": list(speedup._SPOKEN_DROP_TOKENS),
        "convert_import": {
            "file": str(mismatch_path),
            "data_unit": "VAD 分片",
            "id": "原切段 id",
            "sha256": "原切段 sha256",
            "source_path": "父音频路径",
            "playable_audio": "audio.resampled_16k = 加速后 wav",
            "audio_before": "加速前 VAD wav",
            "audio_after": "加速后 wav",
            "qwen_before": f"transcripts.{before_key}",
            "qwen_after": f"transcripts.{after_key}",
            "candidate_text": "未加速 Qwen，导入后作为原始机转参考",
            "preannotation_key": HIT_PHRASE_KEY,
            "label": "准入时精准命中的原文，同时写入 transcripts.exact_negative_phrase",
        },
    }
    summary_path = output_dir / "summary.json"
    atomic_write_json(summary_path, summary)
    speedup._info(
        f"rows={counts['rows']} same={counts['same']} "
        f"char_acc_ne1={counts['mismatch']} failed={counts['failed']}"
    )
    speedup._info(f"compare  {compare_path}")
    speedup._info(f"label    {mismatch_path}")
    speedup._info(f"failed   {failed_path}")
    speedup._info(f"summary  {summary_path}")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help="两份 prepared_evidence parquet。省略时用 datasets/stage1/derived 下 20260814 和 20260813 的 -s 文件",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "tmp" / "1027" / "exact_negative_speed1p4",
        help="准入清单、Qwen 结果和字准 parquet 的目录",
    )
    parser.add_argument("--source-name", default=DEFAULT_SOURCE_NAME)
    parser.add_argument("--speed-factor", type=float, default=1.4)
    parser.add_argument("--before-key", default=None, help="加速前 Qwen 键。默认自动选 qwen*")
    parser.add_argument("--after-key", default=None, help="加速后 Qwen 键。默认 qwen_<倍率>")
    parser.add_argument(
        "--qwen-api-base",
        default=None,
        help=f"Qwen 地址。默认环境变量 QWEN_ASR_API_BASE，否则 {DEFAULT_API_BASE}",
    )
    parser.add_argument("--qwen-model", default=None, help="须等于 vLLM --served-model-name")
    parser.add_argument("--qwen-manifest", type=Path, default=None, help="已有加速后 Qwen parquet，只重算字准")
    parser.add_argument("--filter-only", action="store_true", help="只写准入 parquet")
    parser.add_argument("--skip-filter", action="store_true", help="沿用输出目录里已有的准入 parquet")
    parser.add_argument("--skip-tempo", action="store_true")
    parser.add_argument("--skip-qwen", action="store_true")
    parser.add_argument("--no-sharding", action="store_true", help="加速阶段不分片。Qwen 阶段始终不分片")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    return parser.parse_args(argv)


def resolve_api_base(args: argparse.Namespace) -> str:
    import os

    return (
        args.qwen_api_base
        or os.environ.get("QWEN_ASR_API_BASE")
        or DEFAULT_API_BASE
    ).rstrip("/")


def resolve_model(args: argparse.Namespace) -> str:
    import os

    model = args.qwen_model or os.environ.get("QWEN_ASR_MODEL") or "qwen3-asr"
    return str(model).strip()


def run_from_qwen_manifest(args: argparse.Namespace, api_base: str, model: str | None) -> int:
    qwen_manifest = args.qwen_manifest.resolve()
    if not qwen_manifest.is_file():
        raise SystemExit(f"[ERROR] 找不到 Qwen 结果: {qwen_manifest}")
    speed = normalize_speed_factor(args.speed_factor)
    tag = factor_tag(speed)
    after_key = args.after_key or f"qwen_{tag}"
    validate_source_name(after_key)
    transcribed = Manifest.load(qwen_manifest)
    if not transcribed.samples:
        raise SystemExit(f"[ERROR] Qwen 结果没有样本: {qwen_manifest}")
    before_key = speedup.resolve_archived_before_key(transcribed.samples, args.before_key)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    speedup._info(
        f"qwen={qwen_manifest} rows={len(transcribed.samples)} "
        f"before_key={before_key} after_key={after_key}"
    )
    write_review(
        transcribed.samples,
        output_dir=output_dir,
        tag=tag,
        before_key=before_key,
        after_key=after_key,
        meta={
            "qwen_manifest": str(qwen_manifest),
            "speed_factor": speed,
            "before_key": before_key,
            "after_key": after_key,
            "qwen_api_base": api_base,
            "qwen_model": model,
            "diff_only": True,
        },
    )
    return 0


def load_inputs(paths: list[Path]) -> list[tuple[Path, pa.Table]]:
    loaded: list[tuple[Path, pa.Table]] = []
    for path in paths:
        if not path.is_file():
            raise SystemExit(f"[ERROR] 输入 parquet 不存在: {path}")
        loaded.append((path, pq.read_table(path)))
    return loaded


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    api_base = resolve_api_base(args)
    model = resolve_model(args)
    if args.qwen_manifest is not None:
        return run_from_qwen_manifest(args, api_base, model)

    inputs = [Path(item) for item in args.inputs] or [Path(item) for item in DEFAULT_INPUTS]
    if len(inputs) < 2:
        raise SystemExit("[ERROR] 至少需要两份输入 parquet")
    speed = normalize_speed_factor(args.speed_factor)
    tag = factor_tag(speed)
    try:
        source_name = validate_source_name(args.source_name)
    except ValueError as exc:
        raise SystemExit(f"[ERROR] {exc}") from exc
    after_key = args.after_key or f"qwen_{tag}"
    try:
        validate_source_name(after_key)
    except ValueError as exc:
        raise SystemExit(f"[ERROR] --after-key 不合法: {exc}") from exc

    output_dir = args.output_dir.resolve()
    admitted_path = output_dir / "admitted.parquet"
    qwen_manifest = output_dir / f"qwen_{tag}_asr.parquet"
    runs_dir = args.runs_dir.resolve()

    if args.skip_filter:
        if not admitted_path.is_file():
            raise SystemExit(f"[ERROR] --skip-filter 但找不到准入清单: {admitted_path}")
        speedup._info(f"沿用准入清单: {admitted_path}")
    else:
        loaded = load_inputs(inputs)
        write_admitted(loaded, admitted_path)

    if args.filter_only:
        return 0

    evidence = Manifest.load(admitted_path)
    if not evidence.samples:
        raise SystemExit(f"[ERROR] 准入 parquet 没有样本: {admitted_path}")
    before_key = speedup.resolve_before_key(evidence.samples, args.before_key)
    speedup.preflight_audio(evidence.samples, "resampled_16k")
    tempo_bind = speedup.predict_tempo_manifest(
        source_name=source_name,
        input_manifest=admitted_path,
        speed_factor=speed,
    )
    tempo_manifest = Path(tempo_bind["output_manifest"])
    speedup._info(f"rows={len(evidence.samples)} before_key={before_key} after_key={after_key}")
    speedup._info(f"tempo manifest={tempo_manifest}")
    speedup._info(f"qwen manifest={qwen_manifest}")

    env = speedup._child_env()
    env["QWEN_ASR_API_BASE"] = api_base
    if model:
        env["QWEN_ASR_MODEL"] = model

    tempo_cmd = speedup._audio_data_cmd() + [
        "pipeline",
        "run",
        str(speedup.TEMPO_YAML),
        "--source-name",
        source_name,
        "--input-manifest",
        str(admitted_path),
        "--speed-factor",
        str(speed),
        "--runs-dir",
        str(runs_dir / f"{source_name}_tempo_{tag}"),
    ]
    if args.no_sharding:
        tempo_cmd.append("--no-sharding")
    if args.force:
        tempo_cmd.append("--force")

    qwen_cmd = speedup._audio_data_cmd() + [
        "pipeline",
        "run",
        str(speedup.QWEN_YAML),
        "--no-sharding",
        "--input-manifest",
        str(tempo_manifest),
        "--output-manifest",
        str(qwen_manifest),
        "--transcript-key",
        after_key,
        "--runs-dir",
        str(runs_dir / f"{source_name}_qwen_{tag}"),
    ]
    if args.force:
        qwen_cmd.append("--force")

    if not args.skip_qwen:
        assert_qwen_reachable(api_base, model)

    if args.dry_run:
        if not args.skip_tempo:
            speedup._info("dry-run tempo: " + " ".join(tempo_cmd))
        if not args.skip_qwen:
            speedup._info("dry-run qwen: " + " ".join(qwen_cmd))
        return 0

    if not args.skip_tempo:
        if tempo_manifest.is_file() and not args.force:
            speedup._info(f"已有加速清单，跳过加速: {tempo_manifest}")
        else:
            speedup._run(tempo_cmd, env)
        if not tempo_manifest.is_file():
            raise SystemExit(f"[ERROR] 加速未发布正式清单: {tempo_manifest}")
    elif not tempo_manifest.is_file():
        raise SystemExit(f"[ERROR] --skip-tempo 但找不到加速清单: {tempo_manifest}")

    if not args.skip_qwen:
        if speedup.qwen_matches_tempo(qwen_manifest, tempo_manifest, after_key) and not args.force:
            speedup._info(f"已有对应当前加速清单的 Qwen 结果，跳过转写: {qwen_manifest}")
        else:
            assert_qwen_reachable(api_base, model)
            output_dir.mkdir(parents=True, exist_ok=True)
            speedup._run(qwen_cmd, env)
            if not qwen_manifest.is_file():
                raise SystemExit(f"[ERROR] Qwen 未写出清单: {qwen_manifest}")
            speedup.write_qwen_stamp(qwen_manifest, tempo_manifest, after_key)
    elif not qwen_manifest.is_file():
        raise SystemExit(f"[ERROR] --skip-qwen 但找不到转写清单: {qwen_manifest}")

    transcribed = Manifest.load(qwen_manifest)
    write_review(
        transcribed.samples,
        output_dir=output_dir,
        tag=tag,
        before_key=before_key,
        after_key=after_key,
        meta={
            "input": [str(path) for path in inputs],
            "admitted": str(admitted_path),
            "source_name": source_name,
            "speed_factor": speed,
            "before_key": before_key,
            "after_key": after_key,
            "qwen_api_base": api_base,
            "qwen_model": model,
            "tempo_manifest": str(tempo_manifest),
            "qwen_manifest": str(qwen_manifest),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
