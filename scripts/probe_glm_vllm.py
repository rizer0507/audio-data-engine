#!/usr/bin/env python3
"""Probe a GLM-ASR vLLM service with one WAV file or a WAV directory."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from audio_engine.core.operator import OperatorConfig  # noqa: E402
from audio_engine.operators.asr import glm  # noqa: E402
from audio_engine.operators.asr.vllm import call_vllm_transcription  # noqa: E402


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
        description="用单个 WAV 或 WAV 文件夹探测 GLM-ASR vLLM transcription 接口"
    )
    parser.add_argument("input", type=Path, help="单个 .wav 文件或包含 .wav 的目录")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/asr/glm.yaml")
    parser.add_argument(
        "--api-base",
        help="覆盖 GLM_ASR_API_BASE / 配置文件；多台用逗号分隔",
    )
    parser.add_argument("--model", help="覆盖 GLM_ASR_MODEL / 配置文件")
    parser.add_argument("--concurrency", type=int, help="目录探针并发请求数")
    parser.add_argument("--recursive", action="store_true", help="递归查找子目录中的 WAV")
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="同一批音频连续识别次数；2 用于文本稳定性探针",
    )
    return parser


def transcribe_probe_paths(paths: list[Path], settings: dict) -> list[dict]:
    """Keep source order. Concurrent HTTP across the directory."""
    concurrency = min(max(1, int(settings.get("concurrency", 8))), max(1, len(paths)))
    ordered: list[dict | None] = [None] * len(paths)

    def transcribe(index: int, path: Path) -> tuple[int, dict]:
        request = glm._request_settings(settings, path.stem)
        return index, call_vllm_transcription(str(path), request)

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(transcribe, index, path) for index, path in enumerate(paths)]
        for future in as_completed(futures):
            index, result = future.result()
            ordered[index] = result
    if any(item is None for item in ordered):
        raise RuntimeError("探针识别结果不完整")
    return [item for item in ordered if item is not None]


def _print_result(path: Path, result: dict, extra: dict[str, object] | None = None) -> str:
    text = str(result.get("text", "")).strip()
    payload: dict[str, object] = {
        "ok": bool(text),
        "audio": str(path),
        "text": text,
        "language": result.get("language"),
    }
    if extra:
        payload.update(extra)
    print(json.dumps(payload, ensure_ascii=False))
    return text


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.repeat < 1:
        print(json.dumps({"ok": False, "error": "--repeat 必须 >= 1"}, ensure_ascii=False), file=sys.stderr)
        return 1
    params: dict[str, object] = {"config_path": str(args.config)}
    for key in ("api_base", "model", "concurrency"):
        value = getattr(args, key)
        if value is not None:
            params[key] = value

    try:
        source_paths = discover_wavs(args.input, args.recursive)
        settings = glm._resolve_batch_settings(OperatorConfig(params=params))
        texts_by_round: list[list[str]] = []
        empty = 0
        for round_index in range(args.repeat):
            results = transcribe_probe_paths(source_paths, settings)
            round_texts: list[str] = []
            for source, result in zip(source_paths, results):
                extra = {"round": round_index + 1} if args.repeat > 1 else None
                text = _print_result(source, result, extra)
                round_texts.append(text)
                if round_index == 0:
                    empty += not bool(text)
            texts_by_round.append(round_texts)
    except Exception as exc:  # noqa: BLE001 - probe must print a concise operational error
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1

    unstable = 0
    if args.repeat > 1:
        first = texts_by_round[0]
        for later in texts_by_round[1:]:
            for left, right in zip(first, later):
                unstable += left != right

    summary = {
        "ok": empty == 0 and unstable == 0,
        "total": len(source_paths),
        "non_empty": len(source_paths) - empty,
        "empty": empty,
        "repeat": args.repeat,
        "unstable": unstable,
        "api_bases": list(settings.get("api_bases") or []),
    }
    print(json.dumps({"summary": summary}, ensure_ascii=False), file=sys.stderr)
    if empty:
        return 2
    if unstable:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
