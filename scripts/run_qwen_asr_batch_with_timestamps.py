#!/usr/bin/env python3
"""调用 qwen-base 或 qwen-asr-demo，输出转写和时间戳。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import run_qwen_asr_batch as qwen

DEFAULT_API_BASE = "http://127.0.0.1:5555"
DEFAULT_ALIGNER_DIR = Path("/data2/data-cp/models/Qwen3-ForcedAligner-0.6b")
OPERATOR = "asr.qwen_batch_with_timestamps"
VERSION = "1.0.0"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("input", nargs="?", type=Path, help="输入 Parquet 或 JSONL Manifest")
    source.add_argument("--audio", type=Path, help="单条音频，结果以 JSON 输出到 stdout")
    parser.add_argument("-o", "--output", type=Path, help="批量输出 Manifest")
    parser.add_argument("--audio-key", default="resampled_16k", help="Manifest 中的音频路径键")
    parser.add_argument("--transcript-key", default="qwen-base-timestamps", help="转写结果别名")
    parser.add_argument("--config", type=Path, default=qwen.ROOT / "configs/asr/qwen_asr.yaml")
    parser.add_argument(
        "--backend", choices=("vllm", "gradio"), default="vllm",
        help="vllm 使用 OpenAI Chat 和本地对齐器；gradio 使用 qwen-asr-demo 服务端时间戳",
    )
    parser.add_argument("--api-base", help="覆盖 QWEN_ASR_API_BASE，默认本机 5555")
    parser.add_argument("--api-key", help="覆盖 QWEN_ASR_API_KEY")
    parser.add_argument("--model", help="覆盖 QWEN_ASR_MODEL，须匹配 served-model-name")
    parser.add_argument("--timeout", type=float, help="单条 ASR 请求超时秒数")
    parser.add_argument("--batch-size", type=int, help="每批样本数，默认读取 ASR 配置")
    parser.add_argument("--concurrency", type=int, help="ASR 并发数，默认读取 ASR 配置")
    parser.add_argument("--aligner-model-dir", type=Path, default=DEFAULT_ALIGNER_DIR)
    parser.add_argument("--aligner-device", default="cuda:0")
    parser.add_argument(
        "--aligner-dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    parser.add_argument("--aligner-language", help="覆盖对齐语言，Gradio 模式也用于指定转写语言")
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--limit", type=int, help="本次最多处理多少条待转写样本")
    parser.add_argument("--force", action="store_true", help="重新转写和对齐已有结果")
    return parser


def load_aligner(args: argparse.Namespace):
    if not (args.aligner_model_dir / "config.json").is_file():
        raise FileNotFoundError(f"对齐模型缺少配置文件: {args.aligner_model_dir / 'config.json'}")

    import torch
    from qwen_asr import Qwen3ForcedAligner

    return Qwen3ForcedAligner.from_pretrained(
        str(args.aligner_model_dir),
        dtype=getattr(torch, args.aligner_dtype),
        device_map=args.aligner_device,
        local_files_only=True,
    )


def timestamp_ms(text: str, start: float, end: float) -> dict:
    if not (math.isfinite(start) and math.isfinite(end) and 0 <= start <= end):
        raise ValueError(f"无效时间戳: {start}, {end}")
    return {"text": text, "start_ms": round(start * 1000), "end_ms": round(end * 1000)}


def demo_language(settings, args) -> str:
    language = args.aligner_language or qwen.chat_template_kwargs(settings)["force_language"] or "Auto"
    return "Chinese" if language.lower() == "zh" else language


def make_transcriber(settings, args):
    if args.backend == "vllm":
        return lambda audio_path: qwen.call_vllm_chat_asr(audio_path, settings)

    try:
        from gradio_client import Client, handle_file
    except ImportError as exc:
        raise RuntimeError("Gradio 模式需要先执行 python -m pip install gradio-client") from exc

    timeout = float(settings.get("timeout", 120))
    client = Client(settings["api_base"], verbose=False, httpx_kwargs={"timeout": timeout})
    language = demo_language(settings, args)

    def transcribe(audio_path):
        if not Path(audio_path).is_file():
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")
        job = client.submit(handle_file(audio_path), language, True, api_name="/run")
        try:
            outputs = job.result(timeout=timeout)
        except TimeoutError:
            job.cancel()
            raise TimeoutError(f"Gradio 转写超时（{timeout:g} 秒）") from None
        if not isinstance(outputs, (tuple, list)) or len(outputs) < 3:
            raise ValueError("Gradio /run 没有返回时间戳输出，请确认 Demo 启动时指定了 --aligner-checkpoint")
        detected_language, text, timestamps = outputs[:3]
        if isinstance(timestamps, dict) and timestamps.get("__type__") == "update":
            timestamps = timestamps.get("value")
        if not isinstance(text, str) or not isinstance(timestamps, list):
            raise ValueError("Gradio /run 返回了无效的转写或时间戳格式")
        return {
            "text": text,
            "language": detected_language,
            "timestamps": [
                timestamp_ms(item["text"], item["start_time"], item["end_time"])
                for item in timestamps
            ],
        }

    return transcribe


def add_timestamps(result, audio_path, settings, args, aligner):
    entry = qwen.transcript_entry(result, settings)
    text = entry["text"].strip()
    timestamps = result.get("timestamps", [])
    if text != "【听不清】" and any(char.isalnum() for char in text):
        if aligner is not None:
            language = args.aligner_language or result.get("language") or "Chinese"
            if language.lower() == "zh":
                language = "Chinese"
            aligned = aligner.align(audio=audio_path, text=text, language=language)[0]
            timestamps = [
                timestamp_ms(item.text, item.start_time, item.end_time)
                for item in aligned.items
            ]
        if not timestamps:
            raise ValueError("对齐结果为空")
    else:
        timestamps = []
    entry["extra"].update(
        timestamps=timestamps,
        timestamp_origin="audio_start",
        alignment_status="aligned" if timestamps else "skipped_no_text",
    )
    return entry


def run_identity(settings, args):
    if args.backend == "gradio":
        return {
            "backend": "gradio",
            "api_base": settings["api_base"],
            "model": settings["model"],
            "model_version": settings.get("model_version"),
            "language": demo_language(settings, args),
            "audio_key": args.audio_key,
            "transcript_key": args.transcript_key,
        }
    request_options = {
        key: settings.get(key)
        for key in ("context", "prompt", "temperature", "language", "model_version")
    }
    request_options["chat_template_kwargs"] = qwen.chat_template_kwargs(settings)
    request_digest = hashlib.sha256(
        json.dumps(request_options, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return {
        "api_base": settings["api_base"],
        "model": settings["model"],
        "request_sha256": request_digest,
        "aligner_model_dir": str(args.aligner_model_dir),
        "aligner_device": args.aligner_device,
        "aligner_dtype": args.aligner_dtype,
        "aligner_language": args.aligner_language,
        "audio_key": args.audio_key,
        "transcript_key": args.transcript_key,
    }


def pending_samples(samples, args, identity):
    previous = {}
    if args.output.is_file() and not args.force:
        previous = {sample.id: sample for sample in qwen.Manifest.load(args.output)}
    pending = []
    for sample in samples:
        existing = previous.get(sample.id, sample)
        entry = existing.transcripts.get(args.transcript_key, {})
        record = next(
            (
                item for item in reversed(existing.lineage)
                if item.operator == OPERATOR and item.output_key == args.transcript_key
            ),
            None,
        )
        if (
            not args.force
            and existing.is_completed(OPERATOR)
            and isinstance(entry, dict)
            and isinstance(entry.get("extra"), dict)
            and isinstance(entry["extra"].get("timestamps"), list)
            and existing.sha256 == sample.sha256
            and existing.source_path == sample.source_path
            and existing.audio.get(args.audio_key) == sample.audio.get(args.audio_key)
            and record is not None
            and record.version == VERSION
            and record.params == identity
        ):
            sample.transcripts[args.transcript_key] = entry
            if record not in sample.lineage:
                sample.lineage.append(record)
            sample.mark_completed(OPERATOR)
            continue
        pending.append(sample)
    return pending


def run_batch(settings, args):
    manifest = qwen.Manifest.load(args.input)
    if len({sample.id for sample in manifest}) != len(manifest):
        raise ValueError("输入 Manifest 中的样本 id 必须唯一，才能按 id 恢复检查点")
    identity = run_identity(settings, args)
    pending = pending_samples(manifest.samples, args, identity)
    selected = pending[: args.limit]
    counts = {"processed": 0, "failed": 0, "remaining": len(pending) - len(selected)}
    if selected:
        request_asr = make_transcriber(settings, args)
        aligner = load_aligner(args) if args.backend == "vllm" else None
        batch_size = max(1, int(settings.get("batch_size", 8)))
        concurrency = min(batch_size, max(1, int(settings.get("concurrency", 4))))
        last_checkpoint = 0

        def transcribe(sample):
            path = sample.audio_path(args.audio_key)
            return path, request_asr(path)

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            for start in range(0, len(selected), batch_size):
                futures = {
                    pool.submit(transcribe, sample): sample
                    for sample in selected[start : start + batch_size]
                }
                for future in as_completed(futures):
                    sample = futures[future]
                    try:
                        audio_path, result = future.result()
                        entry = add_timestamps(result, audio_path, settings, args, aligner)
                        sample.transcripts[args.transcript_key] = entry
                        sample.add_lineage(
                            OPERATOR, VERSION, identity, input_key=args.audio_key,
                            output_key=args.transcript_key,
                        )
                        sample.mark_completed(OPERATOR)
                        counts["processed"] += 1
                    except Exception as exc:
                        sample.mark_failed(OPERATOR, str(exc))
                        counts["failed"] += 1
                        print(f"failed id={sample.id}: {exc}", file=sys.stderr)
                done = counts["processed"] + counts["failed"]
                print(f"progress {done}/{len(selected)} failed={counts['failed']}", file=sys.stderr)
                if done - last_checkpoint >= args.checkpoint_every:
                    manifest.save(args.output)
                    last_checkpoint = done
    manifest.save(args.output)
    print(json.dumps({"output": str(args.output), **counts}, ensure_ascii=False), file=sys.stderr)
    return 1 if counts["failed"] else 0


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.input:
        if not args.output:
            parser.error("批量模式需要 --output")
        if args.input.resolve() == args.output.resolve():
            parser.error("输入和输出路径必须不同")
        if args.output.suffix not in {".parquet", ".jsonl"}:
            parser.error("输出须为 .parquet 或 .jsonl")
    elif args.output:
        parser.error("单条模式将 JSON 打到 stdout，请使用重定向保存")
    for name in ("limit", "checkpoint_every", "batch_size", "concurrency", "timeout"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} 必须大于 0")

    args.api_base = args.api_base or os.environ.get("QWEN_ASR_API_BASE") or DEFAULT_API_BASE
    args.aligner_model_dir = args.aligner_model_dir.expanduser().resolve()
    args.mock = False
    try:
        settings = qwen.load_settings(args)
        if args.audio:
            audio_path = str(args.audio.expanduser().resolve())
            request_asr = make_transcriber(settings, args)
            aligner = load_aligner(args) if args.backend == "vllm" else None
            result = request_asr(audio_path)
            entry = add_timestamps(result, audio_path, settings, args, aligner)
            print(json.dumps(entry, ensure_ascii=False, indent=2))
            return 0
        return run_batch(settings, args)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
