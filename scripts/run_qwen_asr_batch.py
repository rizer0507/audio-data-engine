#!/usr/bin/env python3
"""Qwen3-ASR 批处理转写。

把流水线 ``pipelines/qwen_asr_batch.yaml`` 里 ``asr.qwen_batch`` 的转写路径收成一个脚本。
请求和调度与下面两处保持一致：

- ``src/audio_engine/operators/asr/vllm.py`` 的 ``call_vllm_chat_asr``
- ``src/audio_engine/operators/asr/qwen.py`` 的 ``QwenBatchASROperator``

行为：读 parquet，按 ``batch_size`` 分块，块内用 ``concurrency`` 个线程请求
vLLM ``/v1/chat/completions``，一条失败只记在该样本上。已有 ``transcript_key``
的样本默认跳过，输出文件已存在时按 ``id`` 接续。

不包含分片进程、算子磁盘缓存，也不做落盘清洗。
分片在 ``src/audio_engine/core/sharded_run.py``，
清洗是流水线下一步 ``quality.normalize_transcripts``。

示例::

    set QWEN_ASR_API_BASE=http://127.0.0.1:8000
    set QWEN_ASR_MODEL=qwen3-asr
    python scripts/run_qwen_asr_batch.py datasets/stage1/cleaned/cleaned_source_A.parquet ^
        -o datasets/stage1/asr/qwen_asr_source_A.parquet
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from audio_engine.core.manifest import Manifest  # noqa: E402
from audio_engine.core.sample import Sample  # noqa: E402

OPERATOR = "asr.qwen_batch"
OPERATOR_VERSION = "1.0.0"
_ASR_TEXT_TAG = "<asr_text>"
_LANG_PREFIX = "language "


def load_settings(args: argparse.Namespace) -> dict[str, Any]:
    """解析顺序与 ``_resolve_batch_settings`` 相同：命令行，环境变量，yaml。"""
    settings: dict[str, Any] = {}
    if args.config.is_file():
        settings.update(yaml.safe_load(args.config.read_text(encoding="utf-8")) or {})

    settings["api_base"] = (
        args.api_base
        or os.environ.get("QWEN_ASR_API_BASE")
        or settings.get("api_base")
    )
    settings["api_key"] = (
        args.api_key
        or os.environ.get("QWEN_ASR_API_KEY")
        or settings.get("api_key")
    )
    settings["model"] = (
        args.model
        or os.environ.get("QWEN_ASR_MODEL")
        or settings.get("model")
        or "qwen3-asr"
    )
    if args.batch_size is not None:
        settings["batch_size"] = args.batch_size
    if args.concurrency is not None:
        settings["concurrency"] = args.concurrency
    if args.timeout is not None:
        settings["timeout"] = args.timeout
    if not settings.get("api_base") and not args.mock:
        raise ValueError(
            "Qwen 批处理只走 vLLM。请设置 QWEN_ASR_API_BASE，或传入 --api-base。"
        )
    return settings


def chat_completions_url(api_base: str) -> str:
    base = api_base.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    if base.endswith("/audio/transcriptions"):
        root = base[: -len("/audio/transcriptions")]
        return f"{root}/chat/completions"
    return f"{base}/v1/chat/completions"


def chat_template_kwargs(settings: dict[str, Any]) -> dict[str, Any]:
    raw = settings.get("chat_template_kwargs")
    kwargs: dict[str, Any] = dict(raw) if isinstance(raw, dict) else {}
    force_language = settings.get("force_language")
    if force_language and "force_language" not in kwargs:
        kwargs["force_language"] = str(force_language)
    if "force_language" not in kwargs:
        kwargs["force_language"] = "Chinese"
    return kwargs


def parse_qwen_chat_content(content: str) -> tuple[str, str]:
    """拆掉可选的 ``language X<asr_text>`` 前缀，正文才是转写。"""
    text = str(content or "").strip()
    if not text or _ASR_TEXT_TAG not in text:
        return "", text
    meta, body = text.split(_ASR_TEXT_TAG, 1)
    language = ""
    for line in meta.splitlines():
        line = line.strip()
        if line.lower().startswith(_LANG_PREFIX):
            language = line[len(_LANG_PREFIX) :].strip()
            break
    return language, body.strip()


def audio_data_url(path: Path) -> str:
    mime_type = mimetypes.guess_type(path.name)[0] or "audio/wav"
    if not mime_type.startswith("audio/"):
        mime_type = "audio/wav"
    audio_b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{audio_b64}"


def call_vllm_chat_asr(audio_path: str, settings: dict[str, Any]) -> dict[str, Any]:
    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(f"音频文件不存在: {audio_path}")

    context = settings.get("context")
    if context is None:
        context = settings.get("prompt") or ""
    payload: dict[str, Any] = {
        "messages": [
            {"role": "system", "content": f"{context}"},
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio_url",
                        "audio_url": {"url": audio_data_url(path)},
                    }
                ],
            },
        ],
        "chat_template_kwargs": chat_template_kwargs(settings),
        "temperature": float(settings.get("temperature", 0)),
    }
    if settings.get("model"):
        payload = {"model": str(settings["model"]), **payload}

    headers = {"Content-Type": "application/json"}
    if settings.get("api_key"):
        headers["Authorization"] = f"Bearer {settings['api_key']}"

    request = urllib.request.Request(
        chat_completions_url(str(settings["api_base"])),
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    timeout = float(settings.get("timeout", 120))
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"vLLM ASR HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"vLLM ASR 请求失败: {exc.reason}") from exc
    if not isinstance(body, dict):
        raise RuntimeError(f"vLLM ASR 返回格式无效: {body!r}")

    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"vLLM chat ASR 返回格式无效: {body!r}") from exc

    language, text = parse_qwen_chat_content(str(content or ""))
    return {
        "text": text,
        "language": language
        or (settings.get("chat_template_kwargs") or {}).get("force_language")
        or settings.get("force_language")
        or settings.get("language"),
    }


def transcript_entry(result: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    language = result.get("language")
    return {
        "text": result["text"],
        "model": settings.get("model", "qwen3-asr"),
        "version": settings.get("model_version", settings.get("version", "unknown")),
        "extra": {"language": language} if language else {},
    }


def apply_success(sample: Sample, result: dict[str, Any], settings: dict[str, Any], args: argparse.Namespace) -> None:
    sample.transcripts[args.transcript_key] = transcript_entry(result, settings)
    sample.add_lineage(
        operator=OPERATOR,
        version=OPERATOR_VERSION,
        params={
            "input_audio_key": args.audio_key,
            "transcript_key": args.transcript_key,
            "model": settings.get("model", "qwen3-asr"),
            "api_base": settings.get("api_base"),
        },
        input_key=args.audio_key,
    )
    sample.mark_completed(OPERATOR)


def resume_from_output(samples: list[Sample], output: Path, transcript_key: str) -> int:
    """把已写出的转写按 id 盖回输入，方便中断后续跑。"""
    if not output.is_file() or output.stat().st_size == 0:
        return 0
    previous = {sample.id: sample for sample in Manifest.load(output).samples}
    restored = 0
    for sample in samples:
        old = previous.get(sample.id)
        if old is None or transcript_key not in old.transcripts:
            continue
        sample.transcripts[transcript_key] = old.transcripts[transcript_key]
        if old.status.get(OPERATOR):
            sample.status[OPERATOR] = old.status[OPERATOR]
        if OPERATOR in old.errors:
            sample.errors[OPERATOR] = old.errors[OPERATOR]
        restored += 1
    return restored


def pending_indexes(samples: list[Sample], transcript_key: str, force: bool) -> list[int]:
    if force:
        return list(range(len(samples)))
    return [index for index, sample in enumerate(samples) if transcript_key not in sample.transcripts]


def transcribe_one(
    sample: Sample,
    settings: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    if args.mock:
        return {"text": f"[mock:qwen:{sample.id}]", "language": settings.get("language")}
    return call_vllm_chat_asr(sample.audio_path(args.audio_key), settings)


def run_batch(samples: list[Sample], settings: dict[str, Any], args: argparse.Namespace) -> dict[str, int]:
    indexes = pending_indexes(samples, args.transcript_key, args.force)
    if args.limit is not None:
        indexes = indexes[: args.limit]
    batch_size = max(1, int(settings.get("batch_size", 8)))
    counts = {"pending": len(indexes), "processed": 0, "failed": 0, "skipped": len(samples) - len(indexes)}
    started = time.perf_counter()
    since_checkpoint = 0

    print(
        f"Qwen batch backend=vllm api_base={settings.get('api_base')} "
        f"model={settings.get('model')} pending={len(indexes)} skipped={counts['skipped']}",
        file=sys.stderr,
    )

    for start in range(0, len(indexes), batch_size):
        chunk = indexes[start : start + batch_size]
        concurrency = min(max(1, int(settings.get("concurrency", 4))), len(chunk))

        def transcribe(index: int) -> tuple[int, dict[str, Any]]:
            return index, transcribe_one(samples[index], settings, args)

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {pool.submit(transcribe, index): index for index in chunk}
            for future in as_completed(futures):
                index = futures[future]
                sample = samples[index]
                try:
                    _, result = future.result()
                    apply_success(sample, result, settings, args)
                    counts["processed"] += 1
                except Exception as exc:  # noqa: BLE001 - 单条失败不拖垮同批
                    sample.mark_failed(OPERATOR, str(exc))
                    counts["failed"] += 1
                    print(f"failed id={sample.id} error={exc}", file=sys.stderr)
                since_checkpoint += 1

        done = counts["processed"] + counts["failed"]
        elapsed = time.perf_counter() - started
        rate = done / elapsed if elapsed else 0.0
        print(
            f"progress {done}/{len(indexes)} processed={counts['processed']} "
            f"failed={counts['failed']} rate={rate:.2f}/s",
            file=sys.stderr,
        )
        if since_checkpoint >= args.checkpoint_every:
            Manifest(samples).save(args.output)
            since_checkpoint = 0
            print(f"checkpoint {args.output}", file=sys.stderr)

    return counts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="对 parquet 里的音频做 Qwen3-ASR vLLM 批处理转写")
    parser.add_argument("input", type=Path, help="输入 manifest，需含音频路径列")
    parser.add_argument("-o", "--output", type=Path, required=True, help="输出 parquet")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/asr/qwen_asr.yaml",
        help="ASR 配置，默认 configs/asr/qwen_asr.yaml",
    )
    parser.add_argument("--audio-key", default="resampled_16k", help="样本 audio 字典里的路径键")
    parser.add_argument("--transcript-key", default="qwen", help="写入 transcripts 的键")
    parser.add_argument("--api-base", help="覆盖 QWEN_ASR_API_BASE / 配置文件")
    parser.add_argument("--api-key", help="覆盖 QWEN_ASR_API_KEY / 配置文件")
    parser.add_argument("--model", help="覆盖 QWEN_ASR_MODEL，须等于 vLLM --served-model-name")
    parser.add_argument("--batch-size", type=int, help="每轮送去并发的样本数，默认读配置")
    parser.add_argument("--concurrency", type=int, help="每轮并发请求数，默认读配置")
    parser.add_argument("--timeout", type=float, help="单条 HTTP 超时秒数")
    parser.add_argument("--checkpoint-every", type=int, default=500, help="每处理多少条落一次盘")
    parser.add_argument("--limit", type=int, help="只转写这么多条尚未完成的样本")
    parser.add_argument("--force", action="store_true", help="已有转写也重跑")
    parser.add_argument("--mock", action="store_true", help="不请求服务，写入 [mock:qwen:样本id]")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.checkpoint_every < 1:
        print("--checkpoint-every 必须 >= 1", file=sys.stderr)
        return 2
    try:
        settings = load_settings(args)
        samples = list(Manifest.load(args.input).samples)
        restored = 0 if args.force else resume_from_output(samples, args.output, args.transcript_key)
        if restored:
            print(f"resume restored={restored} from {args.output}", file=sys.stderr)
        counts = run_batch(samples, settings, args)
        Manifest(samples).save(args.output)
    except Exception as exc:  # noqa: BLE001 - 脚本入口把错误打到 stderr
        print(str(exc), file=sys.stderr)
        return 1

    print(
        json.dumps({"output": str(args.output), **counts}, ensure_ascii=False),
        file=sys.stderr,
    )
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
