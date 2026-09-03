#!/usr/bin/env python3
"""Run one Kimi-Audio transcription by loading local weights directly.

Prefer scripts/probe_kimi_audio.py (single WAV or a WAV folder). This script is a
thin --audio wrapper around the same local operator.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Kimi-Audio 本地模型单条识别测试")
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--config", default=ROOT / "configs/asr/kimi_audio.yaml", type=Path)
    parser.add_argument("--model-path", help="覆盖配置中的本地模型目录")
    parser.add_argument("--device", help="例如 cuda、cuda:0 或 cpu")
    args = parser.parse_args(argv)

    audio = args.audio.expanduser().resolve()
    if not audio.is_file():
        parser.error(f"音频文件不存在: {audio}")

    probe = ROOT / "scripts" / "probe_kimi_audio.py"
    command = [sys.executable, str(probe), str(audio), "--config", str(args.config)]
    if args.model_path:
        command.extend(["--model-path", args.model_path])
    if args.device:
        command.extend(["--device", args.device])
    import subprocess

    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
