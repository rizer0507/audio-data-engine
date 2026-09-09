#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从若干 cleaned_*.parquet 随机抽取固定条数，写成新的清洗 Manifest。

输出默认落在 ``datasets/stage1/cleaned/cleaned_{source-name}.parquet``，
后续工序可直接用 ``--source-name`` 接入，例如：

    python scripts/sample_cleaned_parquet.py \\
      datasets/stage1/cleaned/cleaned_mt3000.parquet \\
      datasets/stage1/cleaned/cleaned_fenshen.parquet \\
      -n 10000 --source-name mix10k

    audio-data pipeline run pipelines/qwen_asr_batch.yaml --source-name mix10k

抽取前会丢掉时长为 0 / 缺失、音频文件为空或不存在的样本；同一 id 跨文件去重。
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from glob import glob
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from audio_engine.core.manifest import Manifest
from audio_engine.core.sample import Sample
from audio_engine.core.source_naming import (
    cleaned_output_path,
    manifest_stem,
    resolve_existing_manifest,
    validate_source_name,
)

SCRIPT_VERSION = "1.0"
DEFAULT_AUDIO_KEY = "resampled_16k"


class SampleError(ValueError):
    """User-facing sampling / IO error."""


def expand_input_patterns(patterns: Iterable[str]) -> list[Path]:
    """Expand globs; keep explicit files as-is. Raises if a pattern matches nothing."""
    resolved: list[Path] = []
    seen: set[Path] = set()
    for raw in patterns:
        text = str(raw).strip()
        if not text:
            continue
        if any(ch in text for ch in "*?["):
            matches = [Path(item) for item in sorted(glob(text, recursive=True))]
            files = [path for path in matches if path.is_file()]
            if not files:
                raise SampleError(f"glob 没有匹配到文件: {text}")
        else:
            path = Path(text)
            if not path.is_file():
                raise SampleError(f"输入文件不存在: {path}")
            files = [path]
        for path in files:
            key = path.resolve()
            if key in seen:
                continue
            seen.add(key)
            resolved.append(path)
    return resolved


def resolve_from_sources(source_names: Iterable[str]) -> list[Path]:
    """Map --from-source names to existing cleaned_{name}.parquet|.jsonl."""
    paths: list[Path] = []
    seen: set[Path] = set()
    for raw in source_names:
        name = validate_source_name(raw)
        path = resolve_existing_manifest(manifest_stem("cleaned", name))
        key = path.resolve()
        if key in seen:
            continue
        seen.add(key)
        paths.append(path)
    return paths


def audio_path_for(sample: Sample, audio_key: str = DEFAULT_AUDIO_KEY) -> Path | None:
    raw = (sample.audio or {}).get(audio_key) or sample.source_path
    if not raw:
        return None
    return Path(str(raw))


def skip_reason(
    sample: Sample,
    *,
    audio_key: str = DEFAULT_AUDIO_KEY,
    probe_wav: bool = False,
) -> str | None:
    """Return why a sample cannot enter the pool; None means keep."""
    if sample.labels.get("broken") is True:
        return "broken"
    if sample.labels.get("audio_pass") is False:
        return "audio_pass_false"
    duration = sample.duration
    if duration is None or duration <= 0:
        return "zero_duration"

    path = audio_path_for(sample, audio_key)
    if path is None:
        return "missing_audio_path"
    try:
        exists = path.is_file()
    except OSError:
        return "missing_audio"
    if not exists:
        return "missing_audio"
    try:
        size = path.stat().st_size
    except OSError:
        return "missing_audio"
    if size <= 0:
        return "empty_audio"

    if probe_wav:
        try:
            import soundfile as sf

            info = sf.info(str(path))
        except Exception:
            return "unreadable_audio"
        if info.frames <= 0 or info.duration <= 0:
            return "zero_duration"

    return None


def load_manifests(paths: Iterable[Path]) -> list[tuple[Path, Manifest]]:
    loaded: list[tuple[Path, Manifest]] = []
    for path in paths:
        try:
            loaded.append((path, Manifest.load(path)))
        except Exception as exc:
            raise SampleError(f"无法读取 Manifest: {path} ({exc})") from exc
    return loaded


def collect_pool(
    loaded: list[tuple[Path, Manifest]],
    *,
    audio_key: str = DEFAULT_AUDIO_KEY,
    probe_wav: bool = False,
) -> tuple[list[Sample], Counter[str]]:
    """Union samples, drop unusable / duplicate ids (first occurrence wins)."""
    pool: list[Sample] = []
    reasons: Counter[str] = Counter()
    seen_ids: set[str] = set()
    for path, manifest in loaded:
        reasons["input_rows"] += len(manifest)
        for sample in manifest.samples:
            sample_id = str(sample.id)
            if sample_id in seen_ids:
                reasons["duplicate_id"] += 1
                continue
            reason = skip_reason(sample, audio_key=audio_key, probe_wav=probe_wav)
            if reason is not None:
                reasons[reason] += 1
                continue
            seen_ids.add(sample_id)
            copied = sample.model_copy(deep=True)
            copied.add_lineage(
                operator="scripts.sample_cleaned_parquet",
                version=SCRIPT_VERSION,
                params={"from": path.name},
            )
            pool.append(copied)
    return pool, reasons


def draw_samples(pool: list[Sample], n: int, seed: int) -> list[Sample]:
    if n < 1:
        raise SampleError("--n 必须是正整数")
    if len(pool) < n:
        raise SampleError(
            f"可用样本不足：需要 {n} 条，过滤后只剩 {len(pool)} 条"
        )
    rng = random.Random(seed)
    picked = rng.sample(pool, n)
    rng.shuffle(picked)
    return picked


def default_output_path(source_name: str) -> Path:
    return Path(cleaned_output_path(validate_source_name(source_name)))


def write_sampled_manifest(
    samples: list[Sample],
    output: Path,
    *,
    overwrite: bool = False,
) -> Path:
    if output.exists() and not overwrite:
        raise SampleError(f"输出已存在（加 --overwrite 覆盖）: {output}")
    Manifest(samples).save(output)
    return output


def _print_summary(
    *,
    inputs: list[Path],
    reasons: Counter[str],
    pool_size: int,
    n: int,
    seed: int,
    source_name: str,
    output: Path,
    dry_run: bool,
) -> None:
    print("输入:")
    for path in inputs:
        print(f"  {path}")
    print("-" * 55)
    print(f"读取行数:          {reasons.get('input_rows', 0)}")
    print(f"时长<=0 / 缺失:    {reasons.get('zero_duration', 0)}")
    print(f"空音频文件:        {reasons.get('empty_audio', 0)}")
    print(f"音频文件不存在:    {reasons.get('missing_audio', 0) + reasons.get('missing_audio_path', 0)}")
    print(f"无法读取音频:      {reasons.get('unreadable_audio', 0)}")
    print(f"broken / 未通过:   {reasons.get('broken', 0) + reasons.get('audio_pass_false', 0)}")
    print(f"跨文件重复 id:     {reasons.get('duplicate_id', 0)}")
    print(f"可用池:            {pool_size}")
    print(f"抽取:              {n}  (seed={seed})")
    print(f"--source-name:     {source_name}")
    print(f"输出:              {output}")
    if dry_run:
        print("dry-run: 未写文件")
    else:
        print("-" * 55)
        print("后续接入示例:")
        print(f"  audio-data pipeline run pipelines/qwen_asr_batch.yaml --source-name {source_name}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="从若干 cleaned parquet 随机抽取固定条数，写成可被 --source-name 接入的新清洗集"
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help="cleaned_*.parquet 路径，支持 glob（PowerShell 请给通配符加引号）",
    )
    parser.add_argument(
        "--from-source",
        nargs="+",
        default=[],
        metavar="NAME",
        help="按 source-name 解析 datasets/.../cleaned_{NAME}.parquet",
    )
    parser.add_argument(
        "-n",
        "--count",
        dest="count",
        type=int,
        required=True,
        help="抽取条数（过滤零时长之后）",
    )
    parser.add_argument(
        "--source-name",
        required=True,
        help="新批次名；默认写出 datasets/stage1/cleaned/cleaned_{name}.parquet",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子（默认 42）")
    parser.add_argument(
        "--output",
        type=Path,
        help="覆盖默认输出路径（仍建议保持 cleaned_{source-name}.parquet 命名）",
    )
    parser.add_argument(
        "--audio-key",
        default=DEFAULT_AUDIO_KEY,
        help=f"用于检查空音频的 audio 键（默认 {DEFAULT_AUDIO_KEY}）",
    )
    parser.add_argument(
        "--probe-wav",
        action="store_true",
        help="用 soundfile 再核验真实帧数（更慢，适合不信任 duration 字段时）",
    )
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖已有输出")
    parser.add_argument("--dry-run", action="store_true", help="只打印统计，不写文件")
    return parser


def run(args: argparse.Namespace) -> Path | None:
    source_name = validate_source_name(args.source_name)
    inputs = expand_input_patterns(args.inputs)
    if args.from_source:
        inputs.extend(resolve_from_sources(args.from_source))
    # Dedup after mixing positional files and --from-source.
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in inputs:
        key = path.resolve()
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    inputs = unique
    if not inputs:
        raise SampleError("请至少提供一个输入 parquet，或使用 --from-source")

    loaded = load_manifests(inputs)
    pool, reasons = collect_pool(
        loaded, audio_key=args.audio_key, probe_wav=args.probe_wav
    )
    output = Path(args.output) if args.output else default_output_path(source_name)
    _print_summary(
        inputs=inputs,
        reasons=reasons,
        pool_size=len(pool),
        n=args.count,
        seed=args.seed,
        source_name=source_name,
        output=output,
        dry_run=args.dry_run,
    )
    expected = default_output_path(source_name)
    if output.resolve() != expected.resolve() and output.name != expected.name:
        print(
            f"[WARN] 输出文件名不是 {expected.name}，"
            f"--source-name {source_name} 可能解析不到该文件。"
            f"建议放到 {expected.as_posix()}",
            file=sys.stderr,
        )
    picked = draw_samples(pool, args.count, args.seed)
    if args.dry_run:
        return None
    write_sampled_manifest(picked, output, overwrite=args.overwrite)
    print(f"\n写出 {len(picked)} 条: {output}")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run(args)
    except SampleError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
