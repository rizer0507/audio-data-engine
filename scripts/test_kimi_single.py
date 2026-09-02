#!/usr/bin/env python3
"""Run one Kimi-Audio transcription by loading local weights directly."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audio_engine.core.operator import OperatorConfig  # noqa: E402
from audio_engine.operators.asr import kimi  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Kimi-Audio 本地模型单条识别测试")
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--config", default=ROOT / "configs/asr/kimi.yaml", type=Path)
    parser.add_argument("--model-path", help="覆盖配置中的本地模型目录")
    parser.add_argument("--device", help="例如 cuda、cuda:0 或 cpu")
    args = parser.parse_args()

    audio = args.audio.expanduser().resolve()
    if not audio.is_file():
        parser.error(f"音频文件不存在: {audio}")
    params = {"config_path": str(args.config)}
    if args.model_path:
        params["model_path"] = args.model_path
    if args.device:
        params["device"] = args.device

    settings = kimi._resolve_settings(OperatorConfig(params=params))
    result = kimi._transcribe_one(kimi._load_kimi_audio_model(settings), str(audio), settings)
    print(result["text"])


if __name__ == "__main__":
    main()
