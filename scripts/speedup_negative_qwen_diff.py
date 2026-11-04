#!/usr/bin/env python3
# coding=utf-8
"""否定词证据 parquet：1.4 倍保音高加速、Qwen 转写、与加速前 Qwen 做差值。

输入是筛选脚本产出的 ``prepared_evidence_*_negative.parquet``（证据行，含
``audio.resampled_16k`` 和原来的 Qwen 转写）。本脚本不改原始 wav。

步骤（在能读到 wav、且 Qwen vLLM 已启动的机器上执行）：

  1. ``pipelines/augment_batch_speed.yaml`` 用 FFmpeg atempo 写出额外 wav
  2. ``pipelines/qwen_asr_batch.yaml`` 只转写这些加速 wav
  3. 先把两边做成 plain 文本（去标点、空白），再删掉列出的口语词，删完仍不一致才选出

加速前的 Qwen 在派生样本的 ``labels.augmentation.source.transcripts`` 里，
不从加速后的音频上读取。

删词不是替换。词来自 ``configs/normalization/fenshen_1015_filler_tokens.yaml``，
长词优先整段删除。只因为这些语气词不同的行不再进入 mismatch。

服务器上已经有 ``qwen_1p4_asr.parquet`` 时，只重算差值::

  python scripts/speedup_negative_qwen_diff.py \\
    --qwen-manifest prepared_evidence__20260813-1-n_20260814-1-n_negative_speed1p4/qwen_1p4_asr.parquet

服务器示例::

  export QWEN_ASR_API_BASE=http://127.0.0.1:5555
  export QWEN_ASR_API_KEY=dummy
  export QWEN_ASR_MODEL=qwen3-asr

  python scripts/speedup_negative_qwen_diff.py \\
    --input datasets/stage1/derived/prepared_evidence_b1_b2_negative.parquet \\
    --source-name b1_b2_negative

产物（默认在 ``datasets/stage1/derived/<source-name>_speed<tag>/``）：

  qwen_<tag>_asr.parquet       加速样本 + 新 Qwen
  qwen_<tag>_compare.parquet   每条的加速前/后文本
  qwen_<tag>_mismatch.parquet  删词后仍不一致；用标注平台「转换 parquet」导入
  qwen_<tag>_failed.parquet    加速结果里缺转写，或 Qwen 失败
  summary.json

转换导入认 ``id``、``sha256``、``audio.resampled_16k``。
``audio.resampled_16k`` 放加速后的 wav。``source_path`` 只放与它不同的父音频路径，
不能写成加速 wav，否则 VAD 分片会认为子音频指向父音频。
预标注来源来自 ``transcripts`` 的键。``transcripts.qwen_1p4`` 是加速后的转写，
加速前原文仍留在原来的 qwen 键上。``source_audio_id`` 是父音频 ID，Ai话术仍由 xlsx 导入。
导入时数据单位选 VAD 分片，预标注来源勾选 ``qwen_1p4``。

Qwen 阶段固定 ``--no-sharding``，避免流水线按 YAML 里的 GPU 列表再占卡。
加速阶段沿用 YAML 分片；正式 ``manifest.parquet`` 已在对应产物目录时跳过。
Qwen 阶段在转写清单旁写 stamp，只有这份加速清单未变才跳过。
``--force`` 会把 ``--force`` 传给两段流水线并重做差值。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audio_engine.core.artifacts import atomic_path, atomic_write_json  # noqa: E402
from audio_engine.core.augmentation.params import factor_tag, normalize_speed_factor  # noqa: E402
from audio_engine.core.manifest import Manifest, file_sha256  # noqa: E402
from audio_engine.core.sample import Sample  # noqa: E402
from audio_engine.core.source_naming import validate_source_name  # noqa: E402
from audio_engine.core.transcript_reconcile import plain_transcript_text  # noqa: E402

TEMPO_YAML = ROOT / "pipelines" / "augment_batch_speed.yaml"
QWEN_YAML = ROOT / "pipelines" / "qwen_asr_batch.yaml"
QWEN_OPERATOR = "asr.qwen_batch"
DEFAULT_BEFORE_KEY = "qwen"
FILLER_TOKENS_YAML = ROOT / "configs" / "normalization" / "fenshen_1015_filler_tokens.yaml"


def load_filler_tokens(path: Path | None = None) -> tuple[str, ...]:
    """读取语气词 yaml。``tokens`` 为非空字符串列表，重复项忽略。删除时按字数从长到短。"""
    target = FILLER_TOKENS_YAML if path is None else path
    if not target.is_file():
        raise SystemExit(f"[ERROR] 缺少语气词表: {target}")
    raw = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    tokens = raw.get("tokens") if isinstance(raw, dict) else None
    if not isinstance(tokens, list) or not tokens:
        raise SystemExit(f"[ERROR] 语气词表 tokens 为空: {target}")
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in tokens:
        text = str(item).strip()
        if not text:
            raise SystemExit(f"[ERROR] 语气词表含空词: {target}")
        if text in seen:
            continue
        seen.add(text)
        cleaned.append(text)
    return tuple(cleaned)


# 长词优先，避免「你妈」「什么」被更短的词拆开。
_SPOKEN_DROP_TOKENS: tuple[str, ...] = load_filler_tokens()


def drop_spoken_tokens(text: str) -> str:
    """Delete listed spoken tokens wherever they occur. Longer tokens first."""
    dropped = str(text or "")
    for token in sorted(_SPOKEN_DROP_TOKENS, key=len, reverse=True):
        if token:
            dropped = dropped.replace(token, "")
    return dropped


def _info(msg: str) -> None:
    print(f"[INFO] {msg}", flush=True)


def _child_env() -> dict[str, str]:
    env = os.environ.copy()
    src = str(ROOT / "src")
    prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src if not prev else src + os.pathsep + prev
    return env


def _audio_data_cmd() -> list[str]:
    return [sys.executable, "-m", "audio_engine.cli.main"]


def _run(cmd: list[str], env: dict[str, str]) -> None:
    _info(" ".join(cmd))
    proc = subprocess.run(cmd, cwd=ROOT, env=env)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def _entry_text(entry: Any) -> str:
    if isinstance(entry, dict):
        value = entry.get("text")
        if value is None:
            return ""
        return str(value)
    if entry is None:
        return ""
    return str(entry)


def choose_before_key(names: set[str], preferred: str | None) -> str:
    """Pick the pre-speedup Qwen key. Explicit --before-key must exist."""
    if preferred:
        if preferred not in names:
            raise SystemExit(
                f"[ERROR] 输入里没有转写键 {preferred!r}。"
                f" 现有键: {sorted(names)[:30]}"
            )
        return preferred
    qwen_keys = sorted(key for key in names if "qwen" in key.lower())
    if DEFAULT_BEFORE_KEY in qwen_keys:
        return DEFAULT_BEFORE_KEY
    if len(qwen_keys) == 1:
        return qwen_keys[0]
    if not qwen_keys:
        raise SystemExit(
            "[ERROR] 输入 parquet 没有 qwen 转写。用 --before-key 指定要比对的键。"
        )
    raise SystemExit(
        "[ERROR] 输入里有多路 Qwen，用 --before-key 指定要比对的那一路: "
        + ", ".join(qwen_keys)
    )


def resolve_before_key(samples: list[Sample], preferred: str | None) -> str:
    names = {str(key) for sample in samples for key in (sample.transcripts or {})}
    return choose_before_key(names, preferred)


def resolve_archived_before_key(samples: list[Sample], preferred: str | None) -> str:
    names: set[str] = set()
    for sample in samples:
        transcripts = _source_namespace(sample).get("transcripts") or {}
        if isinstance(transcripts, dict):
            names.update(str(key) for key in transcripts)
    return choose_before_key(names, preferred)


def preflight_audio(samples: list[Sample], audio_key: str) -> None:
    missing = [
        sample.id
        for sample in samples
        if not str((sample.audio or {}).get(audio_key) or "").strip()
    ]
    if not missing:
        return
    preview = ", ".join(missing[:8])
    raise SystemExit(
        f"[ERROR] {len(missing)} 条缺少 audio.{audio_key}，加速不会改用别的路径。"
        f" 例如: {preview}"
    )


def predict_tempo_manifest(
    *,
    source_name: str,
    input_manifest: Path,
    speed_factor: float,
) -> dict[str, Any]:
    from audio_engine.core.augmentation.cli_bind import bind_tempo_pipeline
    from audio_engine.core.pipeline import PipelineConfig

    cfg = PipelineConfig.from_yaml(TEMPO_YAML)
    return bind_tempo_pipeline(
        cfg,
        source_batch=source_name,
        speed_factor_cli=speed_factor,
        input_manifest_override=str(input_manifest.resolve()),
        mock=False,
        defer_publish=cfg.sharding is not None,
    )


def _source_namespace(sample: Sample) -> dict[str, Any]:
    labels = sample.labels or {}
    augmentation = labels.get("augmentation") or {}
    if not isinstance(augmentation, dict):
        return {}
    source = augmentation.get("source") or {}
    return source if isinstance(source, dict) else {}


def _archived_label(sample: Sample, source: dict[str, Any], key: str) -> Any:
    """Read a segment label from the derived sample or the tempo archive."""
    bags: list[dict[str, Any]] = []
    if isinstance(sample.labels, dict):
        bags.append(sample.labels)
    for name in ("labels_extra", "identity_labels"):
        bag = source.get(name)
        if isinstance(bag, dict):
            bags.append(bag)
    for bag in bags:
        value = bag.get(key)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def compare_sample(
    sample: Sample,
    *,
    before_key: str,
    after_key: str,
) -> dict[str, Any]:
    """One accelerated row against the Qwen archived from the source audio."""
    source = _source_namespace(sample)
    source_transcripts = source.get("transcripts") or {}
    if not isinstance(source_transcripts, dict):
        source_transcripts = {}
    source_audio = source.get("audio") or {}
    if not isinstance(source_audio, dict):
        source_audio = {}

    status = str((sample.status or {}).get(QWEN_OPERATOR) or "")
    error = str((sample.errors or {}).get(QWEN_OPERATOR) or "")
    before_stored = ""
    before_missing = before_key not in source_transcripts
    if not before_missing:
        before_stored = _entry_text(source_transcripts.get(before_key))
    after_entry = (sample.transcripts or {}).get(after_key)
    after_missing = after_entry is None
    after_stored = "" if after_missing else _entry_text(after_entry)
    before_plain = plain_transcript_text(before_stored)
    after_plain = plain_transcript_text(after_stored)
    before_stripped = drop_spoken_tokens(before_plain)
    after_stripped = drop_spoken_tokens(after_plain)
    plain_changed = before_plain != after_plain

    if before_missing:
        bucket = "failed"
        if not error:
            error = f"missing archived transcript {before_key}"
    elif status != "completed" or after_missing:
        bucket = "failed"
        if not error:
            error = status or f"missing transcript {after_key}"
    elif before_stripped != after_stripped:
        bucket = "mismatch"
    else:
        bucket = "same"

    speed_wav = str(
        (sample.audio or {}).get("aug_speed")
        or (sample.audio or {}).get("resampled_16k")
        or sample.source_path
        or ""
    )
    source_wav = str(
        source_audio.get("resampled_16k")
        or source.get("source_path")
        or ""
    )
    source_sha = str(source.get("sha256") or "")
    source_id = str(source.get("sample_id") or "")
    source_audio_id = _archived_label(sample, source, "source_audio_id")
    parent_pcm = _archived_label(sample, source, "parent_pcm_sha256")
    if parent_pcm is None:
        parent_pcm = _archived_label(sample, source, "source_pcm_sha256")
    parent_path = _archived_label(sample, source, "parent_source_path")
    if parent_path is None:
        parent_path = _archived_label(sample, source, "normalized_audio_path")
    parent_path_text = None if parent_path is None else str(parent_path)
    play_wav = speed_wav or source_wav
    if parent_path_text and play_wav and parent_path_text == play_wav:
        parent_path_text = None
    transcripts = {
        before_key: {"text": before_stored},
    }
    if after_key and after_key != before_key:
        transcripts[after_key] = {"text": after_stored}
    audio = {"resampled_16k": play_wav} if play_wav else {}
    return {
        "id": source_id,
        "source_id": source_id,
        "segment_id": source_id,
        "is_vad_segment": True,
        "source_audio_id": None if source_audio_id is None else str(source_audio_id),
        "parent_pcm_sha256": None if parent_pcm is None else str(parent_pcm),
        "derived_id": sample.id,
        "bucket": bucket,
        "changed": bucket == "mismatch",
        "qwen_before_key": before_key,
        "qwen_after_key": after_key,
        "qwen_before": before_plain,
        "qwen_after": after_plain,
        "qwen_before_stripped": before_stripped,
        "qwen_after_stripped": after_stripped,
        "plain_changed": plain_changed and bucket != "failed",
        "qwen_status": status,
        "error": error,
        "source_path": parent_path_text,
        "source_wav": source_wav,
        "speed_wav": speed_wav,
        "audio": json.dumps(audio, ensure_ascii=False),
        "transcripts": json.dumps(transcripts, ensure_ascii=False),
        "sha256": source_sha,
        "speed_sha256": sample.sha256 or "",
        "source_sha256": source_sha,
        "duration": sample.duration if sample.duration is not None else source.get("duration"),
        "source_duration": source.get("duration"),
    }


def compare_samples(
    samples: list[Sample],
    *,
    before_key: str,
    after_key: str,
) -> list[dict[str, Any]]:
    return [
        compare_sample(sample, before_key=before_key, after_key=after_key)
        for sample in samples
    ]


def _write_parquet(rows: list[dict[str, Any]], path: Path) -> None:
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    with atomic_path(path) as tmp:
        frame.to_parquet(tmp, index=False)


def _qwen_stamp_path(qwen_manifest: Path) -> Path:
    return qwen_manifest.with_suffix(".stamp.json")


def _tempo_stamp(tempo_manifest: Path, after_key: str) -> dict[str, str]:
    return {
        "tempo_manifest": str(tempo_manifest.resolve()),
        "tempo_sha256": file_sha256(tempo_manifest),
        "after_key": after_key,
    }


def qwen_matches_tempo(qwen_manifest: Path, tempo_manifest: Path, after_key: str) -> bool:
    """Reuse a Qwen manifest only when it was produced from this tempo snapshot."""
    stamp_path = _qwen_stamp_path(qwen_manifest)
    if not qwen_manifest.is_file() or not stamp_path.is_file():
        return False
    try:
        saved = json.loads(stamp_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return saved == _tempo_stamp(tempo_manifest, after_key)


def write_qwen_stamp(qwen_manifest: Path, tempo_manifest: Path, after_key: str) -> None:
    atomic_write_json(
        _qwen_stamp_path(qwen_manifest),
        _tempo_stamp(tempo_manifest, after_key),
    )


def write_diff(
    samples: list[Sample],
    *,
    output_dir: Path,
    tag: str,
    before_key: str,
    after_key: str,
    meta: dict[str, Any],
) -> dict[str, Any]:
    rows = compare_samples(samples, before_key=before_key, after_key=after_key)
    compare_path = output_dir / f"qwen_{tag}_compare.parquet"
    mismatch_path = output_dir / f"qwen_{tag}_mismatch.parquet"
    failed_path = output_dir / f"qwen_{tag}_failed.parquet"
    _write_parquet(rows, compare_path)
    _write_parquet([row for row in rows if row["bucket"] == "mismatch"], mismatch_path)
    _write_parquet([row for row in rows if row["bucket"] == "failed"], failed_path)
    counts = {
        "rows": len(rows),
        "same": sum(row["bucket"] == "same" for row in rows),
        "plain_mismatch": sum(bool(row["plain_changed"]) for row in rows),
        "absorbed_by_spoken_drop": sum(
            row["bucket"] == "same" and bool(row["plain_changed"]) for row in rows
        ),
        "mismatch": sum(row["bucket"] == "mismatch" for row in rows),
        "failed": sum(row["bucket"] == "failed" for row in rows),
    }
    summary = {
        **meta,
        "counts": counts,
        "compare": str(compare_path),
        "mismatch": str(mismatch_path),
        "failed": str(failed_path),
        "convert_import": {
            "file": str(mismatch_path),
            "id": "原切段 id",
            "sha256": "原切段 sha256",
            "source_path": "父音频路径；加速 wav 在 audio.resampled_16k",
            "playable_audio": f"audio.resampled_16k = 加速后 wav",
            "preannotation_key": f"transcripts.{after_key} = 加速后 Qwen",
            "before_key": f"transcripts.{before_key} = 加速前 Qwen 原文",
        },
        "compare_rule": (
            "plain_transcript_text 之后删除语气词表 "
            f"{FILLER_TOKENS_YAML}："
            + "、".join(_SPOKEN_DROP_TOKENS)
            + "。删完整句仍不等才进入 mismatch"
        ),
        "spoken_drop_tokens": list(_SPOKEN_DROP_TOKENS),
    }
    summary_path = output_dir / "summary.json"
    atomic_write_json(summary_path, summary)
    _info(
        f"rows={counts['rows']} same={counts['same']} "
        f"plain_mismatch={counts['plain_mismatch']} "
        f"absorbed={counts['absorbed_by_spoken_drop']} "
        f"mismatch={counts['mismatch']} failed={counts['failed']}"
    )
    _info(f"compare  {compare_path}")
    _info(f"mismatch {mismatch_path}")
    _info(f"failed   {failed_path}")
    _info(f"summary  {summary_path}")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="否定词证据 parquet。已有 Qwen 结果时改传 --qwen-manifest",
    )
    parser.add_argument(
        "--qwen-manifest",
        type=Path,
        default=None,
        help="已完成的 qwen_*_asr.parquet。只重算差值，不再加速、不再转写",
    )
    parser.add_argument(
        "--source-name",
        default=None,
        help="加速批次名。默认用输入文件名（须为字母数字下划线或短横线）",
    )
    parser.add_argument("--speed-factor", type=float, default=1.4)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Qwen 与差值 parquet 目录。默认 datasets/stage1/derived/<source>_speed<tag>/",
    )
    parser.add_argument(
        "--before-key",
        default=None,
        help="加速前 Qwen 转写键。默认 qwen；只有一路 qwen* 时用那一路",
    )
    parser.add_argument(
        "--after-key",
        default=None,
        help="加速后写入的转写键。默认 qwen_<倍率标签>，例如 1.4 -> qwen_1p4",
    )
    parser.add_argument(
        "--qwen-api-base",
        default=None,
        help="覆盖环境变量 QWEN_ASR_API_BASE。Qwen 阶段必填其一",
    )
    parser.add_argument(
        "--qwen-model",
        default=None,
        help="覆盖环境变量 QWEN_ASR_MODEL，须等于 vLLM --served-model-name",
    )
    parser.add_argument(
        "--augmented-manifest",
        type=Path,
        default=None,
        help="已有加速 manifest.parquet。传入后不再推算产物路径",
    )
    parser.add_argument("--skip-tempo", action="store_true", help="跳过加速，使用已有 manifest")
    parser.add_argument("--skip-qwen", action="store_true", help="跳过 Qwen，只对已有转写做差值")
    parser.add_argument(
        "--no-sharding",
        action="store_true",
        help="加速阶段不分片。Qwen 阶段始终不分片",
    )
    parser.add_argument("--force", action="store_true", help="忽略已有产物，重跑加速和 Qwen")
    parser.add_argument("--dry-run", action="store_true", help="只检查输入并打印将执行的命令")
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=ROOT / "runs",
        help="两段流水线的运行目录父路径",
    )
    return parser.parse_args(argv)


def run_diff_from_qwen(args: argparse.Namespace) -> int:
    """Recompute the stripped diff from an existing qwen_*_asr.parquet."""
    qwen_manifest = args.qwen_manifest.resolve()
    if not qwen_manifest.is_file():
        raise SystemExit(f"[ERROR] 找不到 Qwen 结果: {qwen_manifest}")
    speed = normalize_speed_factor(args.speed_factor)
    tag = factor_tag(speed)
    after_key = args.after_key or f"qwen_{tag}"
    try:
        validate_source_name(after_key)
    except ValueError as exc:
        raise SystemExit(f"[ERROR] --after-key 不合法: {exc}") from exc

    transcribed = Manifest.load(qwen_manifest)
    if not transcribed.samples:
        raise SystemExit(f"[ERROR] Qwen 结果没有样本: {qwen_manifest}")
    before_key = resolve_archived_before_key(transcribed.samples, args.before_key)
    present_after = {
        str(key)
        for sample in transcribed.samples
        for key in (sample.transcripts or {})
    }
    if after_key not in present_after:
        raise SystemExit(
            f"[ERROR] Qwen 结果里没有转写键 {after_key!r}。"
            f" 现有键: {sorted(present_after)[:30]}。"
            " 用 --after-key 指定加速后的那一路。"
        )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else qwen_manifest.parent
    )
    _info(
        f"qwen={qwen_manifest} rows={len(transcribed.samples)} "
        f"before_key={before_key} after_key={after_key}"
    )
    write_diff(
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
            "diff_only": True,
        },
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.qwen_manifest is not None:
        if args.input is not None:
            _info("已传入 --qwen-manifest，忽略 --input，只重算差值")
        return run_diff_from_qwen(args)
    if args.input is None:
        raise SystemExit("[ERROR] 请传 --input，或传 --qwen-manifest 只重算差值")
    input_path = args.input.resolve()
    if not input_path.is_file():
        raise SystemExit(f"[ERROR] 找不到输入 parquet: {input_path}")
    if not TEMPO_YAML.is_file() or not QWEN_YAML.is_file():
        raise SystemExit("[ERROR] 缺少 pipelines/augment_batch_speed.yaml 或 qwen_asr_batch.yaml")

    speed = normalize_speed_factor(args.speed_factor)
    tag = factor_tag(speed)
    source_name = args.source_name or input_path.stem
    try:
        source_name = validate_source_name(source_name)
    except ValueError as exc:
        raise SystemExit(f"[ERROR] {exc}。请传 --source-name") from exc

    after_key = args.after_key or f"qwen_{tag}"
    try:
        validate_source_name(after_key)
    except ValueError as exc:
        raise SystemExit(f"[ERROR] --after-key 不合法: {exc}") from exc

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (ROOT / "datasets" / "stage1" / "derived" / f"{source_name}_speed{tag}").resolve()
    )
    qwen_manifest = output_dir / f"qwen_{tag}_asr.parquet"
    runs_dir = args.runs_dir.resolve()

    evidence = Manifest.load(input_path)
    if not evidence.samples:
        raise SystemExit(f"[ERROR] 输入 parquet 没有样本: {input_path}")
    before_key = resolve_before_key(evidence.samples, args.before_key)
    preflight_audio(evidence.samples, "resampled_16k")
    _info(f"input={input_path} rows={len(evidence.samples)} before_key={before_key}")

    if args.augmented_manifest is not None:
        tempo_manifest = args.augmented_manifest.resolve()
        tempo_bind: dict[str, Any] = {"output_manifest": str(tempo_manifest)}
    else:
        tempo_bind = predict_tempo_manifest(
            source_name=source_name,
            input_manifest=input_path,
            speed_factor=speed,
        )
        tempo_manifest = Path(tempo_bind["output_manifest"])
    _info(f"tempo manifest={tempo_manifest}")
    _info(f"qwen manifest={qwen_manifest}")
    _info(f"after_key={after_key} speed_factor={speed}")

    env = _child_env()
    if args.qwen_api_base:
        env["QWEN_ASR_API_BASE"] = args.qwen_api_base
    if args.qwen_model:
        env["QWEN_ASR_MODEL"] = args.qwen_model

    tempo_cmd = _audio_data_cmd() + [
        "pipeline",
        "run",
        str(TEMPO_YAML),
        "--source-name",
        source_name,
        "--input-manifest",
        str(input_path),
        "--speed-factor",
        str(speed),
        "--runs-dir",
        str(runs_dir / f"{source_name}_tempo_{tag}"),
    ]
    if args.no_sharding:
        tempo_cmd.append("--no-sharding")
    if args.force:
        tempo_cmd.append("--force")

    qwen_cmd = _audio_data_cmd() + [
        "pipeline",
        "run",
        str(QWEN_YAML),
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

    if args.dry_run:
        if not args.skip_tempo:
            _info("dry-run tempo: " + " ".join(tempo_cmd))
        if not args.skip_qwen:
            if not env.get("QWEN_ASR_API_BASE"):
                raise SystemExit(
                    "[ERROR] 未设置 QWEN_ASR_API_BASE 或 --qwen-api-base"
                )
            _info("dry-run qwen: " + " ".join(qwen_cmd))
        _info(f"dry-run diff -> {output_dir}")
        return 0

    if not args.skip_tempo:
        if tempo_manifest.is_file() and not args.force:
            _info(f"已有加速清单，跳过加速: {tempo_manifest}")
        else:
            _run(tempo_cmd, env)
        if not tempo_manifest.is_file():
            partial = tempo_manifest.with_name("manifest.partial.parquet")
            hint = f" 部分成功清单: {partial}" if partial.is_file() else ""
            raise SystemExit(f"[ERROR] 加速未发布正式清单: {tempo_manifest}.{hint}")
    elif not tempo_manifest.is_file():
        raise SystemExit(f"[ERROR] --skip-tempo 但找不到加速清单: {tempo_manifest}")

    if not args.skip_qwen:
        if not env.get("QWEN_ASR_API_BASE"):
            raise SystemExit("[ERROR] 未设置 QWEN_ASR_API_BASE 或 --qwen-api-base")
        if qwen_matches_tempo(qwen_manifest, tempo_manifest, after_key) and not args.force:
            _info(f"已有对应当前加速清单的 Qwen 结果，跳过转写: {qwen_manifest}")
        else:
            output_dir.mkdir(parents=True, exist_ok=True)
            _run(qwen_cmd, env)
            if not qwen_manifest.is_file():
                raise SystemExit(f"[ERROR] Qwen 未写出清单: {qwen_manifest}")
            write_qwen_stamp(qwen_manifest, tempo_manifest, after_key)
        if not qwen_manifest.is_file():
            raise SystemExit(f"[ERROR] Qwen 未写出清单: {qwen_manifest}")
    elif not qwen_manifest.is_file():
        raise SystemExit(f"[ERROR] --skip-qwen 但找不到转写清单: {qwen_manifest}")

    transcribed = Manifest.load(qwen_manifest)
    write_diff(
        transcribed.samples,
        output_dir=output_dir,
        tag=tag,
        before_key=before_key,
        after_key=after_key,
        meta={
            "input": str(input_path),
            "source_name": source_name,
            "speed_factor": speed,
            "before_key": before_key,
            "after_key": after_key,
            "tempo_manifest": str(tempo_manifest),
            "qwen_manifest": str(qwen_manifest),
            "product_root": tempo_bind.get("product_root"),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
