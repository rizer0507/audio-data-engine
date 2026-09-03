#!/usr/bin/env python3
"""Probe local Kimi-Audio (kimia_infer) with one WAV file or a WAV directory.

This is the local-weight counterpart of scripts/probe_kimi_vllm.py. It loads the
model once in-process, then transcribes one file at a time. Multi-GPU throughput
belongs to pipelines/kimi_audio_asr_batch.yaml (one shard process per card).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from audio_engine.core.operator import OperatorConfig
from audio_engine.operators.asr import kimi_audio


def discover_wavs(input_path: Path, recursive: bool = False) -> list[Path]:
    """Return a stable list of WAV inputs, rejecting unsupported paths early."""
    path = input_path.expanduser().resolve()
    if path.is_file():
        if path.suffix.lower() != ".wav":
            raise ValueError(f"仅支持 WAV 文件: {path}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"输入不存在: {path}")
    pattern = "**/*" if recursive else "*"
    wavs = sorted(
        item for item in path.glob(pattern) if item.is_file() and item.suffix.lower() == ".wav"
    )
    if not wavs:
        raise ValueError(f"目录中没有 WAV 文件: {path}")
    return wavs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="用单个 WAV 或 WAV 文件夹探测 Kimi-Audio 本地 kimia_infer 推理"
    )
    parser.add_argument("input", type=Path, help="单个 .wav 文件或包含 .wav 的目录")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/asr/kimi_audio.yaml")
    parser.add_argument("--model-path", help="覆盖 KIMI_AUDIO_MODEL_PATH / 配置文件")
    parser.add_argument("--device", help="例如 cuda、cuda:0 或 cpu")
    parser.add_argument("--recursive", action="store_true", help="递归查找子目录中的 WAV")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="目录探针最多识别的文件数（0=全部）；单文件探针忽略此参数",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    params: dict[str, object] = {"config_path": str(args.config)}
    if args.model_path:
        params["model_path"] = args.model_path
    if args.device:
        params["device"] = args.device

    try:
        paths = discover_wavs(args.input, args.recursive)
        if args.limit and args.limit > 0:
            paths = paths[: args.limit]
        settings = kimi_audio._resolve_settings(OperatorConfig(params=params))
        model = kimi_audio._load_kimi_audio_model(settings)
    except Exception as exc:  # noqa: BLE001 - probe must print a concise operational error
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1

    empty = 0
    failed = 0
    for path in paths:
        try:
            result = kimi_audio._transcribe_one(model, str(path), settings)
            text = str(result.get("text", "")).strip()
            empty += not bool(text)
            record = {"ok": bool(text), "audio": str(path), "text": text}
        except Exception as exc:  # noqa: BLE001 - isolate one bad WAV in a folder probe
            failed += 1
            record = {"ok": False, "audio": str(path), "error": str(exc)}
        print(json.dumps(record, ensure_ascii=False))

    summary = {
        "ok": empty == 0 and failed == 0,
        "total": len(paths),
        "non_empty": len(paths) - empty - failed,
        "empty": empty,
        "failed": failed,
        "model_path": settings.get("model_path"),
        "device": settings.get("device"),
    }
    print(json.dumps({"summary": summary}, ensure_ascii=False), file=sys.stderr)
    if failed:
        return 1
    return 0 if empty == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
