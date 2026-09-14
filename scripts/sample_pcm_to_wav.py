# -*- coding: utf-8 -*-
"""
sample_pcm_to_wav.py
====================
从两个（或多个）PCM 目录中随机抽取 n 个文件，按指定源采样率解读，
并以 16 kHz（可改）写成 WAV，输出到指定目录。

默认按无头 PCM：int16 单声道、源采样率 16000。若扩展名是 .pcm/.raw 但已带
RIFF/WAVE 头，则探测真实采样率后再重采样到目标率。

Usage
-----
python scripts/sample_pcm_to_wav.py \\
    --source-dir /path/to/pcm_a /path/to/pcm_b \\
    -n 50 \\
    --output-dir /path/to/out_wav

# 源其实是 8k 无头 PCM，仍输出 16k WAV
python scripts/sample_pcm_to_wav.py \\
    -i /data/pcm_a -i /data/pcm_b \\
    -n 20 -o ./sampled_wav \\
    --source-sample-rate 8000 --seed 42
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from audio_engine.operators.audio.pcm import looks_like_wav
from audio_engine.operators.audio.resample import DEFAULT_SAMPLE_RATE, resample_audio

_PCM_EXTS = {".pcm", ".raw"}


def collect_pcm_files(source_dirs: list[Path], *, recursive: bool) -> list[Path]:
    files: list[Path] = []
    for d in source_dirs:
        if not d.is_dir():
            print(f"[ERROR] 源目录不存在或不是目录: {d}", file=sys.stderr)
            sys.exit(1)
        it = d.rglob("*") if recursive else d.iterdir()
        for p in it:
            if p.is_file() and p.suffix.lower() in _PCM_EXTS:
                files.append(p.resolve())
    # 去重，保持稳定顺序便于 seed 复现
    seen: set[Path] = set()
    unique: list[Path] = []
    for p in files:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def unique_output_name(src: Path, source_dirs: list[Path], used: set[str]) -> str:
    """用「最近源目录名_stem」避免多目录同名冲突。"""
    parent_label = src.parent.name
    for d in source_dirs:
        try:
            rel = src.relative_to(d.resolve())
            parent_label = d.name if rel.parent == Path(".") else f"{d.name}_{'_'.join(rel.parent.parts)}"
            break
        except ValueError:
            continue
    base = f"{parent_label}_{src.stem}.wav"
    if base not in used:
        return base
    i = 2
    while True:
        cand = f"{parent_label}_{src.stem}_{i}.wav"
        if cand not in used:
            return cand
        i += 1


def load_pcm(
    path: Path,
    *,
    source_sample_rate: int,
    channels: int,
    dtype: str,
) -> tuple[np.ndarray, int]:
    """返回 (audio, source_sr)。有 WAV 头则探测；无头则按 source_sample_rate。"""
    if looks_like_wav(path):
        data, sr = sf.read(str(path), always_2d=False)
        return np.asanyarray(data), int(sr)

    raw = np.fromfile(path, dtype=dtype)
    if channels > 1:
        if raw.size % channels != 0:
            raise ValueError(f"样本数不能整除 channels={channels}: {path}")
        raw = raw.reshape(-1, channels)
    return raw, source_sample_rate


def convert_one(
    src: Path,
    dst: Path,
    *,
    source_sample_rate: int,
    target_sample_rate: int,
    channels: int,
    dtype: str,
) -> tuple[int, int, float]:
    data, src_sr = load_pcm(
        src,
        source_sample_rate=source_sample_rate,
        channels=channels,
        dtype=dtype,
    )
    if src_sr != target_sample_rate:
        data = resample_audio(data, src_sr, target_sample_rate)
    sf.write(str(dst), data, target_sample_rate, subtype="PCM_16")
    n_frames = data.shape[0] if getattr(data, "ndim", 1) > 1 else len(data)
    duration = n_frames / target_sample_rate if target_sample_rate else 0.0
    return src_sr, target_sample_rate, duration


def main() -> None:
    parser = argparse.ArgumentParser(
        description="从多个 PCM 目录随机抽 n 条，按 16k 写成 WAV"
    )
    parser.add_argument(
        "-i",
        "--source-dir",
        dest="source_dirs",
        nargs="+",
        action="append",
        required=True,
        help="PCM 源目录（可多次传入，或一次写多个路径）",
    )
    parser.add_argument(
        "-n",
        "--count",
        type=int,
        required=True,
        help="随机抽取数量",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        required=True,
        help="输出 WAV 目录（不存在则创建）",
    )
    parser.add_argument(
        "--source-sample-rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help=f"无头 PCM 的源采样率（默认: {DEFAULT_SAMPLE_RATE}）",
    )
    parser.add_argument(
        "--target-sample-rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help=f"输出 WAV 采样率（默认: {DEFAULT_SAMPLE_RATE}）",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=1,
        help="无头 PCM 声道数（默认: 1）",
    )
    parser.add_argument(
        "--dtype",
        default="int16",
        help="无头 PCM dtype（默认: int16）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子（可复现）",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="递归扫描子目录（默认只扫各源目录一层）",
    )
    args = parser.parse_args()

    if args.count <= 0:
        print("[ERROR] -n/--count 必须 > 0", file=sys.stderr)
        sys.exit(1)

    # flatten: -i a b 与 -i a -i b 都支持
    flat_dirs: list[str] = []
    for group in args.source_dirs:
        flat_dirs.extend(group)
    source_dirs = [Path(p).expanduser().resolve() for p in flat_dirs]
    if len(source_dirs) < 1:
        print("[ERROR] 至少指定一个 --source-dir", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_pcm = collect_pcm_files(source_dirs, recursive=args.recursive)
    if not all_pcm:
        print("[ERROR] 未找到任何 .pcm/.raw 文件", file=sys.stderr)
        sys.exit(1)

    n = min(args.count, len(all_pcm))
    if n < args.count:
        print(
            f"[WARN] 仅找到 {len(all_pcm)} 个 PCM，少于请求的 {args.count}，将全部抽取",
            file=sys.stderr,
        )

    rng = random.Random(args.seed)
    picked = rng.sample(all_pcm, n)

    print(f"源目录:     {', '.join(str(d) for d in source_dirs)}")
    print(f"候选 PCM:   {len(all_pcm)}")
    print(f"抽取:       {n}" + (f" (seed={args.seed})" if args.seed is not None else ""))
    print(f"源采样率:   {args.source_sample_rate}（仅无头）")
    print(f"目标采样率: {args.target_sample_rate}")
    print(f"输出目录:   {output_dir}")
    print("=" * 55)

    used_names: set[str] = set()
    ok = 0
    failed = 0
    for src in picked:
        name = unique_output_name(src, source_dirs, used_names)
        used_names.add(name)
        dst = output_dir / name
        try:
            src_sr, tgt_sr, dur = convert_one(
                src,
                dst,
                source_sample_rate=args.source_sample_rate,
                target_sample_rate=args.target_sample_rate,
                channels=args.channels,
                dtype=args.dtype,
            )
            print(f"[OK] {src.name} -> {name}  ({src_sr}Hz -> {tgt_sr}Hz, {dur:.2f}s)")
            ok += 1
        except Exception as exc:
            print(f"[FAIL] {src}: {exc}", file=sys.stderr)
            failed += 1

    print("=" * 55)
    print(f"完成: 成功 {ok}，失败 {failed}，输出 {output_dir}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
