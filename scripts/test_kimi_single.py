#!/usr/bin/env python3
"""Kimi-Audio 单条 wav 识别测试（vLLM /v1/audio/transcriptions）。

前置：先启动 vLLM 服务（见 docs/流水线改进/构造kimi-audio识别流水线.md）。

示例：
  export KIMI_ASR_API_BASE=http://127.0.0.1:5554
  export KIMI_ASR_MODEL=kimi-audio

  PYTHONPATH=src python scripts/test_kimi_single.py --audio /path/to/sample.wav

  PYTHONPATH=src python scripts/test_kimi_single.py \\
    --audio sample.wav --api-base http://127.0.0.1:5554 --json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from audio_engine.core.operator import OperatorConfig  # noqa: E402
from audio_engine.operators.asr import kimi as kimi_mod  # noqa: E402


def _probe_audio(path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {
        "path": str(path.resolve()),
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else None,
    }
    if not path.is_file():
        return info
    try:
        import soundfile as sf

        meta = sf.info(str(path))
        info.update(
            {
                "sample_rate": meta.samplerate,
                "channels": meta.channels,
                "duration_sec": round(float(meta.duration), 4),
                "format": meta.format,
            }
        )
    except Exception as exc:  # noqa: BLE001
        info["probe_error"] = str(exc)
    return info


def _build_settings(args: argparse.Namespace) -> dict[str, Any]:
    params: dict[str, Any] = {"config_path": str(args.config)}
    if args.api_base:
        params["api_base"] = args.api_base
    if args.model:
        params["model"] = args.model
    if args.prompt is not None:
        params["prompt"] = args.prompt
    if args.language:
        params["language"] = args.language
    if args.timeout is not None:
        params["timeout"] = args.timeout
    return kimi_mod._resolve_settings(OperatorConfig(params=params))


def _ping_api(api_base: str, timeout: float = 5.0) -> dict[str, Any]:
    import urllib.error
    import urllib.request

    url = f"{api_base.rstrip('/')}/health"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return {"ok": True, "status": response.status, "url": url}
    except urllib.error.URLError as exc:
        return {"ok": False, "url": url, "error": str(exc.reason)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Kimi-Audio 单条 wav 识别测试")
    parser.add_argument("--audio", type=Path, required=True, help="待识别 wav 文件路径")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/asr/kimi.yaml",
        help="ASR 配置（默认 configs/asr/kimi.yaml）",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default=None,
        help="vLLM 地址（默认读配置或 KIMI_ASR_API_BASE）",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="模型名（默认读配置或 KIMI_ASR_MODEL）",
    )
    parser.add_argument("--language", type=str, default=None, help="语言，默认 zh")
    parser.add_argument("--prompt", type=str, default=None, help="覆盖识别提示词")
    parser.add_argument("--timeout", type=float, default=None, help="HTTP 超时秒数")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出完整结果")
    parser.add_argument("--no-ping", action="store_true", help="跳过 vLLM /health 探测")
    args = parser.parse_args()

    audio_path = args.audio.resolve()
    if not audio_path.is_file():
        print(f"错误：音频不存在: {audio_path}", file=sys.stderr)
        return 2

    settings = _build_settings(args)
    report: dict[str, Any] = {
        "ok": False,
        "audio": _probe_audio(audio_path),
        "settings": {
            "api_base": settings.get("api_base"),
            "model": settings.get("model"),
            "language": settings.get("language"),
            "prompt": settings.get("prompt"),
            "timeout": settings.get("timeout"),
        },
        "env": {
            "KIMI_ASR_API_BASE": os.environ.get("KIMI_ASR_API_BASE"),
            "KIMI_ASR_MODEL": os.environ.get("KIMI_ASR_MODEL"),
        },
    }

    if not args.no_ping:
        report["health"] = _ping_api(str(settings["api_base"]))
        if not report["health"].get("ok"):
            report["error"] = (
                f"vLLM 不可达: {report['health'].get('error')} "
                f"（请先启动 vLLM，或检查 --api-base / KIMI_ASR_API_BASE）"
            )
            if args.json:
                print(json.dumps(report, ensure_ascii=False, indent=2))
            else:
                print(report["error"], file=sys.stderr)
            return 2

    started = time.perf_counter()
    try:
        result = kimi_mod._call_vllm_transcription(str(audio_path), settings)
        elapsed = time.perf_counter() - started
        text = str(result.get("text", ""))
        report.update(
            {
                "ok": True,
                "text": text,
                "text_empty": not bool(text.strip()),
                "elapsed_sec": round(elapsed, 4),
                "raw_response": result.get("extra", {}).get("raw_response"),
            }
        )
    except Exception as exc:  # noqa: BLE001
        report.update(
            {
                "error": str(exc),
                "traceback": traceback.format_exc(limit=8),
                "elapsed_sec": round(time.perf_counter() - started, 4),
            }
        )
        if args.json:
            print(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            print(f"识别失败: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"音频: {audio_path}")
        print(f"模型: {settings.get('model')} @ {settings.get('api_base')}")
        print(f"耗时: {report['elapsed_sec']}s")
        print("-" * 40)
        print(report["text"])

    return 0 if report.get("text") else 1


if __name__ == "__main__":
    raise SystemExit(main())
