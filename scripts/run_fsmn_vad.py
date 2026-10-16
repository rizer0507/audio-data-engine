#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对单个音频或目录跑与流水线同源的 FSMN VAD，输出拼接语音 WAV 或片段目录。

规则与 ``audio.fsmn_vad_segment`` 一致：16 kHz / 单声道 ch0、200 ms 流式分块、
模型 end 后保留 675 ms 原音频、EOF 无终点与 endpoint_at_eof 丢弃。

配置与环境（与 ``pipelines/data_cleaning_source_A.yaml`` 一致）
----------------------------------------------------------
- 模型运行目录：``FSMN_VAD_RUNTIME_DIR`` 或 ``--runtime-dir``
- 配置：``configs/vad/fenshen_vad_config.yaml``（须已 overlay 进 runtime 的 ``config.yaml``）
- 工程 mock：``AUDIO_ENGINE_VAD_MOCK=1`` 或 ``--mock``

首次准备 runtime（服务器原始权重）::

  export FSMN_VAD_SOURCE_MODEL_DIR=/data2/data-cp/xijujun/models/fsmn-vad
  python scripts/prepare_fsmn_vad_runtime.py \\
    --source-model-dir "$FSMN_VAD_SOURCE_MODEL_DIR" \\
    --fenshen-config configs/vad/fenshen_vad_config.yaml \\
    --runtime-dir var/models/fsmn-vad-fenshen-v1

用法
----
单文件 → 单 WAV（拼接全部保留片段）::

  export FSMN_VAD_RUNTIME_DIR="$PWD/var/models/fsmn-vad-fenshen-v1"
  PYTHONPATH=src python scripts/run_fsmn_vad.py -i a.wav -o a_speech.wav

单文件 → 目录（写出 ``{stem}.wav``，可选片段）::

  PYTHONPATH=src python scripts/run_fsmn_vad.py -i a.wav -o out_dir --write-segments

目录 → 目录（每个源文件一个拼接 WAV）::

  PYTHONPATH=src python scripts/run_fsmn_vad.py -i in_dir -o out_dir -r
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import wave
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from audio_engine.core.artifacts import atomic_path  # noqa: E402
from audio_engine.core.audio_normalize import take_channel_0  # noqa: E402
from audio_engine.core.fsmn_vad.adapter import (  # noqa: E402
    ScheduleVadModel,
    load_funasr_vad,
    run_streaming_vad,
)
from audio_engine.core.fsmn_vad.types import SR, VadKeptSegment, VadStreamResult  # noqa: E402
from audio_engine.core.manifest import file_sha256  # noqa: E402
from audio_engine.operators.audio.pcm import looks_like_wav  # noqa: E402
from audio_engine.operators.audio.resample import resample_audio  # noqa: E402

DEFAULT_FENSHEN_CONFIG = _ROOT / "configs" / "vad" / "fenshen_vad_config.yaml"
DEFAULT_RUNTIME_DIR = _ROOT / "var" / "models" / "fsmn-vad-fenshen-v1"

_AUDIO_EXTS = {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".pcm", ".raw"}
_PCM_EXTS = {".pcm", ".raw"}
_WAV_SUFFIXES = {".wav"}


class VadCliError(ValueError):
    """用户可见的参数 / 路径错误。"""


def _env_truthy(name: str) -> bool:
    return str(os.environ.get(name) or "").strip() in {"1", "true", "TRUE", "yes"}


def _info(msg: str) -> None:
    print(f"[INFO] {msg}", flush=True)


def _ok(msg: str) -> None:
    print(f"[OK]   {msg}", flush=True)


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}", flush=True)


def _fail(msg: str) -> None:
    print(f"[FAIL] {msg}", file=sys.stderr, flush=True)


def resolve_runtime_dir(cli_value: str | None) -> Path:
    raw = (cli_value or "").strip() or str(os.environ.get("FSMN_VAD_RUNTIME_DIR") or "").strip()
    if not raw:
        raw = str(DEFAULT_RUNTIME_DIR)
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (_ROOT / path).resolve()
    else:
        path = path.resolve()
    return path


def collect_inputs(input_path: Path, *, recursive: bool) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in _AUDIO_EXTS and not looks_like_wav(input_path):
            raise VadCliError(f"不支持的音频后缀: {input_path}")
        return [input_path.resolve()]
    if not input_path.is_dir():
        raise VadCliError(f"输入不存在: {input_path}")

    it = input_path.rglob("*") if recursive else input_path.iterdir()
    files: list[Path] = []
    for p in it:
        if not p.is_file():
            continue
        if p.suffix.lower() in _AUDIO_EXTS or looks_like_wav(p):
            files.append(p.resolve())
    files.sort()
    if not files:
        raise VadCliError(f"目录下未找到音频: {input_path}")
    return files


def load_pcm16le_mono(
    path: Path,
    *,
    source_sample_rate: int,
    channels: int,
) -> tuple[bytes, dict[str, Any]]:
    """读入并规范为 16 kHz mono PCM16LE（与流水线 ch0 / 16k 策略一致）。"""
    suffix = path.suffix.lower()
    headered = suffix not in _PCM_EXTS or looks_like_wav(path)

    if headered:
        data, sr = sf.read(str(path), always_2d=True, dtype="int16")
        mono, source_channels, channel_policy = take_channel_0(data)
        rate_source = "file_header"
        source_sr = int(sr)
    else:
        raw = np.fromfile(path, dtype="<i2")
        if channels > 1:
            if raw.size % channels != 0:
                raise ValueError(f"无头 PCM 长度不能整除 channels={channels}: {path}")
            data = raw.reshape(-1, channels)
        else:
            data = raw.reshape(-1, 1)
        mono, source_channels, channel_policy = take_channel_0(data)
        source_sr = int(source_sample_rate)
        rate_source = "source_config"

    mono_arr = np.asanyarray(mono)
    if source_sr != SR:
        # resample_audio 期望浮点或任意数值；写回 int16
        floated = mono_arr.astype(np.float32)
        floated = resample_audio(floated, source_sr, SR)
        mono_i16 = np.clip(np.rint(floated), -32768, 32767).astype("<i2")
        resampled = True
    else:
        mono_i16 = mono_arr.astype("<i2", copy=False)
        resampled = False

    meta = {
        "source_sample_rate": source_sr,
        "normalized_sample_rate": SR,
        "source_channels": source_channels,
        "selected_channel": 0,
        "channel_policy": channel_policy,
        "sample_rate_source": rate_source,
        "normalize_resampled": resampled,
        "n_samples": int(mono_i16.shape[0]),
        "duration_sec": float(mono_i16.shape[0] / SR) if mono_i16.shape[0] else 0.0,
    }
    return mono_i16.tobytes(), meta


def write_wav_pcm16le(path: Path, pcm16le: bytes, *, sample_rate: int = SR) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(path) as tmp:
        with wave.open(str(tmp), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm16le)


def concat_kept(kept: list[VadKeptSegment]) -> bytes:
    return b"".join(seg.pcm16le for seg in kept)


def segment_record(seg: VadKeptSegment, *, index: int) -> dict[str, Any]:
    return {
        "index": index,
        "start_ms": seg.start_ms,
        "end_ms": seg.end_ms,
        "inferred_end_ms": seg.inferred_end_ms,
        "end_reason": seg.end_reason,
        "start_sample": seg.start_sample,
        "end_sample": seg.end_sample,
        "duration_ms": seg.duration_ms,
        "boundary_clamped": seg.boundary_clamped,
        "n_samples": len(seg.pcm16le) // 2,
    }


def result_summary(result: VadStreamResult) -> dict[str, Any]:
    return {
        "status": result.status,
        "error": result.error,
        "candidate_count": len(result.candidates),
        "kept_count": len(result.kept),
        "discarded_count": len(result.discarded),
        "eof_fallback_count": len(result.eof_fallback),
        "kept": [segment_record(s, index=i) for i, s in enumerate(result.kept)],
        "discarded": [
            {
                "start_ms": d.start_ms,
                "end_ms": d.end_ms,
                "end_reason": d.end_reason,
                "reason": d.reason,
                "start_sample": d.start_sample,
                "end_sample": d.end_sample,
                "duration_ms": d.duration_ms,
            }
            for d in result.discarded
        ],
    }


def build_vad(*, runtime_dir: Path, mock: bool, mock_intervals: list[list[float]] | None) -> Any:
    if mock:
        intervals = mock_intervals or [[200.0, 1800.0]]
        pairs = [(float(a), float(b)) for a, b in intervals]
        return ScheduleVadModel(pairs, sample_rate=SR)
    if not runtime_dir.is_dir():
        raise VadCliError(
            f"FSMN VAD runtime 不存在: {runtime_dir}\n"
            "请先运行 scripts/prepare_fsmn_vad_runtime.py，或设置 FSMN_VAD_RUNTIME_DIR"
        )
    cfg = runtime_dir / "config.yaml"
    if not cfg.is_file():
        raise VadCliError(f"runtime 缺少 config.yaml: {runtime_dir}")
    return load_funasr_vad(runtime_dir)


def is_wav_output(path: Path) -> bool:
    return path.suffix.lower() in _WAV_SUFFIXES


def output_plan(
    *,
    inputs: list[Path],
    input_root: Path,
    output: Path,
    write_segments: bool,
) -> list[dict[str, Path | None]]:
    """为每个输入规划 concat wav / segments 目录。"""
    single_in = len(inputs) == 1 and input_root.is_file()
    out_is_wav = is_wav_output(output)

    if out_is_wav and not single_in:
        raise VadCliError("输出为单个 .wav 时，输入只能是单个音频文件")

    plans: list[dict[str, Path | None]] = []
    if out_is_wav:
        stem = output.stem
        seg_dir = output.parent / f"{stem}_segments" if write_segments else None
        plans.append({"src": inputs[0], "concat": output, "segments_dir": seg_dir})
        return plans

    # 输出目录
    output.mkdir(parents=True, exist_ok=True)
    for src in inputs:
        if input_root.is_dir():
            try:
                rel = src.relative_to(input_root.resolve())
            except ValueError:
                rel = Path(src.name)
            rel_stem = Path(*rel.parts[:-1]) / rel.stem if rel.parts else Path(src.stem)
            concat_path = output / Path(f"{rel_stem}.wav")
            seg_dir = output / Path(f"{rel_stem}_segments") if write_segments else None
        else:
            concat_path = output / f"{src.stem}.wav"
            seg_dir = output / f"{src.stem}_segments" if write_segments else None
        plans.append({"src": src, "concat": concat_path, "segments_dir": seg_dir})
    return plans


def process_one(
    *,
    src: Path,
    concat_path: Path,
    segments_dir: Path | None,
    vad: Any,
    source_sample_rate: int,
    channels: int,
    mock: bool,
) -> dict[str, Any]:
    pcm16le, norm_meta = load_pcm16le_mono(
        src,
        source_sample_rate=source_sample_rate,
        channels=channels,
    )
    if isinstance(vad, ScheduleVadModel):
        # ScheduleVadModel 按累计样本推进；每文件重建一份状态。
        intervals = list(vad.intervals)
        vad = ScheduleVadModel(intervals, sample_rate=SR)

    result = run_streaming_vad(vad, pcm16le, include_eof_fallback_in_kept=False)
    record: dict[str, Any] = {
        "source": str(src),
        "normalize": norm_meta,
        "normalized_sha256": "",
        "vad": result_summary(result),
        "concat_wav": None,
        "segment_wavs": [],
        "mock": mock,
    }

    record["normalized_sha256"] = hashlib.sha256(pcm16le).hexdigest()

    if result.kept:
        speech = concat_kept(result.kept)
        write_wav_pcm16le(concat_path, speech)
        record["concat_wav"] = str(concat_path.resolve())
        record["concat_sha256"] = file_sha256(concat_path)
        record["concat_duration_sec"] = len(speech) / 2 / SR
    else:
        # 无保留片段：不写空 wav，仅记录
        record["concat_wav"] = None
        record["concat_skipped_reason"] = result.status or "no_kept_segments"

    if segments_dir is not None and result.kept:
        segments_dir.mkdir(parents=True, exist_ok=True)
        for i, seg in enumerate(result.kept):
            seg_path = segments_dir / f"{i:03d}_{int(seg.start_ms)}_{int(seg.end_ms or 0)}.wav"
            write_wav_pcm16le(seg_path, seg.pcm16le)
            record["segment_wavs"].append(str(seg_path.resolve()))

    return record


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="FSMN VAD：单文件/目录切段，输出拼接 WAV 或片段目录",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "-i",
        "--input",
        required=True,
        type=Path,
        help="输入音频文件或目录",
    )
    p.add_argument(
        "-o",
        "--output",
        required=True,
        type=Path,
        help="输出：.wav 文件路径，或目录路径",
    )
    p.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="输入为目录时递归扫描",
    )
    p.add_argument(
        "--write-segments",
        action="store_true",
        help="额外写出每个保留片段 WAV（目录名为 {stem}_segments）",
    )
    p.add_argument(
        "--runtime-dir",
        default=None,
        help="FSMN VAD runtime 目录（默认环境变量 FSMN_VAD_RUNTIME_DIR 或 "
        "var/models/fsmn-vad-fenshen-v1）",
    )
    p.add_argument(
        "--fenshen-config",
        type=Path,
        default=DEFAULT_FENSHEN_CONFIG,
        help=f"fenshen 配置（仅作校验提示，默认 {DEFAULT_FENSHEN_CONFIG}）",
    )
    p.add_argument(
        "--source-sample-rate",
        type=int,
        default=SR,
        help=f"无头 PCM 源采样率（默认 {SR}）",
    )
    p.add_argument(
        "--channels",
        type=int,
        default=1,
        help="无头 PCM 声道数（默认 1；多声道取 ch0）",
    )
    p.add_argument(
        "--mock",
        action="store_true",
        help="使用工程 mock（等价 AUDIO_ENGINE_VAD_MOCK=1），不加载 FunASR",
    )
    p.add_argument(
        "--mock-intervals",
        default=None,
        help='mock 区间 JSON，如 [[200,1800],[2000,3500]]（毫秒）',
    )
    p.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="写出处理报告 JSON（默认：输出目录下 vad_run_report.json，"
        "或与输出 wav 同目录的 {stem}_vad_report.json）",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser()
    if not output_path.is_absolute():
        output_path = (Path.cwd() / output_path).resolve()
    else:
        output_path = output_path.resolve()

    mock = bool(args.mock) or _env_truthy("AUDIO_ENGINE_VAD_MOCK")
    mock_intervals: list[list[float]] | None = None
    if args.mock_intervals:
        mock_intervals = json.loads(args.mock_intervals)

    try:
        inputs = collect_inputs(input_path, recursive=args.recursive)
        plans = output_plan(
            inputs=inputs,
            input_root=input_path,
            output=output_path,
            write_segments=args.write_segments,
        )
        runtime_dir = resolve_runtime_dir(args.runtime_dir)
        vad = build_vad(runtime_dir=runtime_dir, mock=mock, mock_intervals=mock_intervals)
    except (VadCliError, json.JSONDecodeError, OSError) as exc:
        _fail(str(exc))
        return 1

    _info(f"input={input_path}  files={len(inputs)}")
    _info(f"output={output_path}")
    _info(f"mock={mock}  runtime={runtime_dir if not mock else '(unused)'}")
    fenshen = Path(args.fenshen_config).expanduser()
    if not fenshen.is_absolute():
        fenshen = (_ROOT / fenshen).resolve()
    if fenshen.is_file():
        _info(f"fenshen_config={fenshen} (sha256={file_sha256(fenshen)[:12]}…)")
    else:
        _warn(f"fenshen 配置文件不存在（不影响已准备好的 runtime）: {fenshen}")

    report: dict[str, Any] = {
        "schema_version": "run_fsmn_vad/v1",
        "input": str(input_path),
        "output": str(output_path),
        "mock": mock,
        "runtime_dir": str(runtime_dir) if not mock else None,
        "fenshen_config": str(fenshen),
        "files": [],
        "summary": {
            "sources": 0,
            "with_speech": 0,
            "no_kept": 0,
            "failed": 0,
            "kept_segments": 0,
        },
    }

    for plan in plans:
        src = Path(plan["src"])  # type: ignore[arg-type]
        concat_path = Path(plan["concat"])  # type: ignore[arg-type]
        segments_dir = plan["segments_dir"]
        seg_dir = Path(segments_dir) if segments_dir else None
        report["summary"]["sources"] += 1
        try:
            # 真实 FunASR 模型可复用；mock 在 process_one 内按文件重建。
            row = process_one(
                src=src,
                concat_path=concat_path,
                segments_dir=seg_dir,
                vad=vad,
                source_sample_rate=int(args.source_sample_rate),
                channels=int(args.channels),
                mock=mock,
            )
            report["files"].append(row)
            kept_n = int(row["vad"]["kept_count"])
            report["summary"]["kept_segments"] += kept_n
            if kept_n > 0:
                report["summary"]["with_speech"] += 1
                _ok(
                    f"{src.name}: kept={kept_n} → {row['concat_wav']} "
                    f"({row.get('concat_duration_sec', 0):.2f}s)"
                )
            else:
                report["summary"]["no_kept"] += 1
                _warn(f"{src.name}: 无保留片段 status={row['vad']['status']}")
        except Exception as exc:  # noqa: BLE001
            report["summary"]["failed"] += 1
            report["files"].append(
                {
                    "source": str(src),
                    "error": str(exc),
                    "status": "failed",
                }
            )
            _fail(f"{src}: {exc}")

    # 报告路径
    if args.json_out is not None:
        json_out = args.json_out.expanduser()
        if not json_out.is_absolute():
            json_out = (Path.cwd() / json_out).resolve()
    elif is_wav_output(output_path):
        json_out = output_path.with_name(f"{output_path.stem}_vad_report.json")
    else:
        json_out = output_path / "vad_run_report.json"

    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    _info(f"report={json_out}")
    _info(
        "summary: sources={sources} with_speech={with_speech} "
        "no_kept={no_kept} failed={failed} kept_segments={kept_segments}".format(
            **report["summary"]
        )
    )

    if report["summary"]["failed"]:
        return 1
    if report["summary"]["with_speech"] == 0 and report["summary"]["sources"] > 0:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
