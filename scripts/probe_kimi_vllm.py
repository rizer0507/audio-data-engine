#!/usr/bin/env python3
"""Probe a Kimi-Audio vLLM service with one WAV file or a WAV directory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from audio_engine.core.operator import OperatorConfig  # noqa: E402
from audio_engine.operators.asr import kimi  # noqa: E402


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
        description="用单个 WAV 或 WAV 文件夹探测 Kimi-Audio vLLM transcription 接口"
    )
    parser.add_argument("input", type=Path, help="单个 .wav 文件或包含 .wav 的目录")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/asr/kimi.yaml")
    parser.add_argument("--api-base", help="覆盖 KIMI_ASR_API_BASE / 配置文件")
    parser.add_argument("--model", help="覆盖 KIMI_ASR_MODEL / 配置文件")
    parser.add_argument("--concurrency", type=int, help="目录探针并发请求数")
    parser.add_argument("--recursive", action="store_true", help="递归查找子目录中的 WAV")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    params: dict[str, object] = {"config_path": str(args.config)}
    for key in ("api_base", "model", "concurrency"):
        value = getattr(args, key)
        if value is not None:
            params[key] = value

    try:
        paths = discover_wavs(args.input, args.recursive)
        settings = kimi._resolve_settings(OperatorConfig(params=params))
        results = kimi._transcribe_many([str(path) for path in paths], settings)
    except Exception as exc:  # noqa: BLE001 - probe must print a concise operational error
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1

    empty = 0
    for path, result in zip(paths, results):
        text = str(result.get("text", "")).strip()
        empty += not bool(text)
        print(
            json.dumps(
                {
                    "ok": bool(text),
                    "audio": str(path),
                    "text": text,
                    "language": result.get("language"),
                },
                ensure_ascii=False,
            )
        )
    summary = {
        "ok": empty == 0,
        "total": len(paths),
        "non_empty": len(paths) - empty,
        "empty": empty,
    }
    print(json.dumps({"summary": summary}, ensure_ascii=False), file=sys.stderr)
    return 0 if empty == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
