#!/usr/bin/env python3
"""用原 qwen-base 参数启动 vLLM，并在 API 服务启动时加载 ForcedAligner。"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
from pathlib import Path

from starlette.concurrency import run_in_threadpool
from starlette.datastructures import MutableHeaders
from starlette.requests import ClientDisconnect, Request
from starlette.responses import JSONResponse

MODEL = "/data2/data-cp/zcl/models/Qwen3-ASR-1___7B"
CHAT_TEMPLATE = "/data2/data-cp/zcl/project/vllm-test/qwen3_asr_language.jinja"
ALIGNER_DIR = Path("/data2/data-cp/models/Qwen3-ForcedAligner-0.6b")
ALIGNER_ENV = "QWEN_BASE_ALIGNER_MODEL_DIR"
LOGGER = logging.getLogger("uvicorn.error")


def vllm_command(binary: str) -> list[str]:
    middleware = f"{Path(__file__).stem}.ForcedAlignmentMiddleware"
    return [
        binary, "serve", MODEL,
        "--gpu-memory-utilization", "0.5",
        "--host", "127.0.0.1",
        "--port", "5555",
        "--tensor-parallel-size", "1",
        "--served-model-name", "qwen3-asr",
        "--chat-template", CHAT_TEMPLATE,
        "--middleware", middleware,
    ]


def load_aligner():
    import torch
    from qwen_asr import Qwen3ForcedAligner

    path = os.environ.get(ALIGNER_ENV, str(ALIGNER_DIR))
    LOGGER.info("Loading Qwen3-ForcedAligner from %s on cuda:0", path)
    return Qwen3ForcedAligner.from_pretrained(
        path, device_map="cuda:0", dtype=torch.bfloat16, local_files_only=True
    )


def embedded_audio(payload: dict) -> str | None:
    urls = []
    for message in payload.get("messages", []):
        content = message.get("content")
        if isinstance(content, list):
            for part in content:
                if part.get("type") == "audio_url":
                    item = part.get("audio_url")
                    urls.append(item.get("url") if isinstance(item, dict) else None)
    if len(urls) == 1 and isinstance(urls[0], str):
        url = urls[0]
        if url.startswith("data:audio/") and ";base64," in url:
            return url
    return None


def has_transcript_fields(body: object) -> bool:
    if not isinstance(body, dict):
        return False
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return False
    for choice in choices:
        message = choice.get("message") if isinstance(choice, dict) else None
        if not isinstance(message, dict) or "content" not in message:
            return False
        content = message["content"]
        if content is not None and not isinstance(content, str):
            return False
    return True


def add_timestamps(body: dict, audio: str, language: str, aligner) -> None:
    for choice in body["choices"]:
        message = choice["message"]
        text = (message["content"] or "").strip()
        detected_language = language
        if "<asr_text>" in text:
            prefix, text = text.split("<asr_text>", 1)
            text = text.strip()
            if prefix.strip().lower().startswith("language "):
                detected_language = prefix.strip()[len("language "):].strip() or language
        timestamps = []
        if text != "【听不清】" and any(char.isalnum() for char in text):
            if detected_language.lower() == "zh":
                detected_language = "Chinese"
            result = aligner.align(audio=audio, text=text, language=detected_language)[0]
            for item in result.items:
                start, end = item.start_time, item.end_time
                if not (math.isfinite(start) and math.isfinite(end) and 0 <= start <= end):
                    raise ValueError("ForcedAligner 返回无效时间戳")
                timestamps.append({
                    "text": item.text,
                    "start_ms": round(start * 1000),
                    "end_ms": round(end * 1000),
                })
            if not timestamps:
                raise ValueError("ForcedAligner 未返回时间戳")
        message["timestamps"] = timestamps


def error_response(status: int, message: str) -> JSONResponse:
    error_type = "invalid_request_error" if status < 500 else "server_error"
    return JSONResponse({"error": {"message": message, "type": error_type}}, status_code=status)


class ForcedAlignmentMiddleware:
    def __init__(self, app):
        self.app = app
        self.aligner = None
        self.alignment_lock = asyncio.Lock()

    async def __call__(self, scope, receive, send):
        if scope["type"] == "lifespan":
            await self.run_lifespan(scope, receive, send)
        elif (
            scope["type"] == "http"
            and scope["method"] == "POST"
            and scope["path"] == "/v1/chat/completions"
        ):
            await self.chat(scope, receive, send)
        else:
            await self.app(scope, receive, send)

    async def run_lifespan(self, scope, receive, send):
        event = await receive()
        if event["type"] == "lifespan.startup":
            try:
                self.aligner = await run_in_threadpool(load_aligner)
            except Exception:
                LOGGER.exception("ForcedAligner 启动失败")
                await send({
                    "type": "lifespan.startup.failed",
                    "message": "ForcedAligner 启动失败，请查看服务日志",
                })
                raise

        async def replay_startup():
            nonlocal event
            if event is not None:
                message, event = event, None
                return message
            return await receive()

        await self.app(scope, replay_startup, send)

    async def chat(self, scope, receive, send):
        try:
            raw_body = await Request(scope, receive).body()
        except ClientDisconnect:
            return

        async def replay_body():
            nonlocal raw_body
            if raw_body is not None:
                body, raw_body = raw_body, None
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        try:
            payload = json.loads(raw_body)
            if not isinstance(payload, dict):
                raise ValueError
            explicit = "return_time_stamps" in payload
            timestamps = payload.pop("return_time_stamps", None)
            if explicit and not isinstance(timestamps, bool):
                return await error_response(400, "return_time_stamps 必须是布尔值")(
                    scope, replay_body, send
                )
            streaming = payload.get("stream") is True
            audio = embedded_audio(payload) if timestamps is not False and not streaming else None
        except (ValueError, TypeError, AttributeError):
            # Let vLLM validate its own request schema and return its native errors.
            await self.app(scope, replay_body, send)
            return

        if timestamps is True and audio is None:
            return await error_response(
                400, "时间戳需要非流式请求，且只接收一段 base64 audio_url"
            )(scope, replay_body, send)
        if explicit:
            raw_body = json.dumps(payload, ensure_ascii=False).encode()
            scope = {**scope, "headers": list(scope["headers"])}
            MutableHeaders(scope=scope)["content-length"] = str(len(raw_body))
        if audio is None:
            await self.app(scope, replay_body, send)
            return
        if self.aligner is None:
            return await error_response(503, "ForcedAligner 尚未就绪")(scope, replay_body, send)

        template_kwargs = payload.get("chat_template_kwargs")
        language = template_kwargs.get("force_language") if isinstance(template_kwargs, dict) else None
        await self.align_response(scope, replay_body, send, audio, language or "Chinese")

    async def align_response(self, scope, receive, send, audio, language):
        response_start = None
        chunks = []

        async def add_to_response(message):
            nonlocal response_start
            if message["type"] == "http.response.start":
                headers = MutableHeaders(raw=list(message["headers"]))
                if message["status"] == 200 and headers.get("content-type", "").startswith(
                    "application/json"
                ) and not headers.get("content-encoding"):
                    response_start = {**message, "headers": headers.raw}
                    return
            elif message["type"] == "http.response.body" and response_start is not None:
                chunks.append(message.get("body", b""))
                if message.get("more_body", False):
                    return
                try:
                    result = json.loads(b"".join(chunks))
                except (ValueError, RecursionError):
                    result = None
                if not has_transcript_fields(result):
                    detail = "vLLM 返回无效的转写响应"
                    LOGGER.warning("%s（HTTP 200）", detail)
                    await error_response(502, detail)(scope, receive, send)
                    return

                try:
                    async with self.alignment_lock:
                        await run_in_threadpool(add_timestamps, result, audio, language, self.aligner)
                    response_body = json.dumps(
                        result, ensure_ascii=False, separators=(",", ":")
                    ).encode()
                except Exception:
                    LOGGER.exception("ForcedAligner 请求失败")
                    await error_response(502, "ForcedAligner 处理失败，请查看服务日志")(
                        scope, receive, send
                    )
                    return
                MutableHeaders(scope=response_start)["content-length"] = str(len(response_body))
                await send(response_start)
                await send({"type": "http.response.body", "body": response_body})
                return
            await send(message)

        await self.app(scope, receive, add_to_response)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-bin", default="vllm", help="vLLM 可执行文件")
    parser.add_argument("--aligner-model-dir", type=Path, default=ALIGNER_DIR)
    args = parser.parse_args()

    aligner_dir = args.aligner_model_dir.expanduser().resolve()
    if not (aligner_dir / "config.json").is_file():
        parser.error(f"对齐模型缺少 config.json: {aligner_dir}")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "4"
    env[ALIGNER_ENV] = str(aligner_dir)
    script_dir = str(Path(__file__).resolve().parent)
    python_path = env.get("PYTHONPATH")
    env["PYTHONPATH"] = script_dir + os.pathsep + python_path if python_path else script_dir
    try:
        os.execvpe(args.vllm_bin, vllm_command(args.vllm_bin), env)
    except OSError as exc:
        parser.exit(1, f"无法执行 vLLM: {exc}\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
