#!/usr/bin/env python3
"""对同一段长 wav，并行测 Sophon /v1/chat/completions 与 /v1/realtime（5 档 VAD）。

默认端口 6179。Realtime 五档从严到松仅改 Silero turn_detection；
chat 无 VAD，单独记一条对照。每条结果含 VAD 配置、转写/回复文本、带时间戳的模型事件日志。

用法:
  python scripts/test_sophon_vad_dual_api.py /path/to/long.wav
  python scripts/test_sophon_vad_dual_api.py /path/to/long.wav --host 127.0.0.1 --port 6179
  python scripts/test_sophon_vad_dual_api.py /path/to/long.wav --out tmp/sophon_vad_probe.txt

依赖: soundfile, numpy, websockets
  pip install soundfile numpy websockets
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

try:
    import websockets
    from websockets.exceptions import ConnectionClosed
except ImportError as exc:  # pragma: no cover
    raise SystemExit("缺少 websockets，请先: pip install websockets") from exc


TRANSCRIBE_INSTRUCTIONS = (
    "逐字转写音频中清楚可辨的词句，保留实际说出的语气词和重复。"
    "跳过听不清的部分，禁止猜测、补全或回答。忽略背景噪声和非语言声音。"
    "若没有可辨认的词句，返回空文本。"
    "只输出转写正文，禁止输出笑声、叹气、呼吸、咳嗽、情绪或重读标签，"
    "也不要输出“无语音”或“听不清”等说明。"
)

# 从严 → 逐步递减。越高门槛 / 越长最短语音 / 越长静音 = 越严。
VAD_TIERS: list[dict[str, Any]] = [
    {
        "tier": 1,
        "name": "最严",
        "turn_detection": {
            "type": "server_vad",
            "threshold": 0.70,
            "min_speech_duration_ms": 200,
            "silence_duration_ms": 900,
            "prefix_padding_ms": 200,
        },
    },
    {
        "tier": 2,
        "name": "偏严",
        "turn_detection": {
            "type": "server_vad",
            "threshold": 0.60,
            "min_speech_duration_ms": 160,
            "silence_duration_ms": 700,
            "prefix_padding_ms": 250,
        },
    },
    {
        "tier": 3,
        "name": "模板默认附近",
        "turn_detection": {
            "type": "server_vad",
            "threshold": 0.50,
            "min_speech_duration_ms": 96,
            "silence_duration_ms": 500,
            "prefix_padding_ms": 300,
        },
    },
    {
        "tier": 4,
        "name": "偏松",
        "turn_detection": {
            "type": "server_vad",
            "threshold": 0.40,
            "min_speech_duration_ms": 64,
            "silence_duration_ms": 350,
            "prefix_padding_ms": 300,
        },
    },
    {
        "tier": 5,
        "name": "最松",
        "turn_detection": {
            "type": "server_vad",
            "threshold": 0.30,
            "min_speech_duration_ms": 32,
            "silence_duration_ms": 200,
            "prefix_padding_ms": 400,
        },
    },
]


@dataclass
class ProbeResult:
    api: str
    label: str
    vad_config: dict[str, Any]
    transcript: str = ""
    events: list[str] = field(default_factory=list)
    error: str = ""
    started_at: str = ""
    finished_at: str = ""
    elapsed_sec: float = 0.0


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def _ts_line(msg: str) -> str:
    return f"[{_now()}] {msg}"


def fetch_model_id(base_url: str, timeout: float = 15.0) -> str:
    url = f"{base_url.rstrip('/')}/v1/models"
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8", errors="replace"))
    names = [item.get("id") for item in payload.get("data") or [] if item.get("id")]
    if not names:
        raise RuntimeError(f"{url} 未返回可用模型 ID: {payload!r}")
    return str(names[0])


def load_pcm16(path: Path, target_sr: int) -> tuple[bytes, float]:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    if int(sr) != int(target_sr):
        n = max(1, int(round(len(mono) * float(target_sr) / float(sr))))
        x_old = np.linspace(0.0, 1.0, num=len(mono), endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=n, endpoint=False)
        mono = np.interp(x_new, x_old, mono).astype(np.float32)
        sr = target_sr
    pcm = (np.clip(mono, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
    duration = len(pcm) / 2 / float(sr)
    return pcm, duration


def _shrink_event(obj: Any) -> Any:
    """日志里去掉大段 base64，保留事件结构。"""
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for key, value in obj.items():
            if key in {"audio", "delta", "data"} and isinstance(value, str) and len(value) > 120:
                out[key] = f"<omitted len={len(value)}>"
            elif key == "content" and isinstance(value, str) and len(value) > 500:
                out[key] = value[:500] + f"...<truncated total={len(value)}>"
            else:
                out[key] = _shrink_event(value)
        return out
    if isinstance(obj, list):
        return [_shrink_event(item) for item in obj]
    return obj


def _extract_text_from_realtime_event(event: dict[str, Any], bucket: list[str]) -> None:
    etype = str(event.get("type") or "")
    if etype in {
        "response.audio_transcript.delta",
        "response.text.delta",
        "conversation.item.input_audio_transcription.delta",
        "response.output_audio_transcript.delta",
        "response.output_text.delta",
    }:
        delta = event.get("delta")
        if isinstance(delta, str) and delta:
            bucket.append(delta)
        return
    if etype in {
        "response.audio_transcript.done",
        "response.text.done",
        "conversation.item.input_audio_transcription.completed",
        "response.output_audio_transcript.done",
        "response.output_text.done",
    }:
        text = event.get("transcript") or event.get("text")
        if isinstance(text, str) and text:
            bucket.append(text)
        return
    if etype == "response.done":
        response = event.get("response") or {}
        for item in response.get("output") or []:
            for part in item.get("content") or []:
                for key in ("transcript", "text"):
                    value = part.get(key)
                    if isinstance(value, str) and value:
                        bucket.append(value)


async def run_realtime(
    *,
    host: str,
    port: int,
    model: str,
    pcm: bytes,
    sample_rate: int,
    tier: dict[str, Any],
    chunk_ms: int,
    pace: float,
    trailing_silence_ms: int,
    idle_timeout: float,
    overall_timeout: float,
) -> ProbeResult:
    label = f"realtime-tier{tier['tier']}-{tier['name']}"
    vad = dict(tier["turn_detection"])
    result = ProbeResult(
        api="/v1/realtime",
        label=label,
        vad_config={
            "tier": tier["tier"],
            "tier_name": tier["name"],
            "turn_detection": vad,
            "extra_body": {"auto_response": True, "language": "zh"},
            "instructions": TRANSCRIBE_INSTRUCTIONS,
            "voice": "chelsie",
            "pcm_sample_rate": sample_rate,
            "chunk_ms": chunk_ms,
            "pace": pace,
            "trailing_silence_ms": trailing_silence_ms,
        },
        started_at=_now(),
    )
    t0 = time.perf_counter()
    uri = f"ws://{host}:{port}/v1/realtime"
    text_parts: list[str] = []
    saw_terminal = False

    def log(msg: str, payload: Any | None = None) -> None:
        if payload is None:
            result.events.append(_ts_line(msg))
            return
        compact = json.dumps(_shrink_event(payload), ensure_ascii=False, separators=(",", ":"))
        result.events.append(_ts_line(f"{msg} {compact}"))

    try:
        async with websockets.connect(
            uri,
            max_size=16 * 1024 * 1024,
            ping_interval=20,
            ping_timeout=20,
            open_timeout=30,
        ) as ws:
            log(f"connected {uri}")
            # Sophon 连上后短时内必须收到 session.update，否则 config_timeout。
            # 因此先发配置，再后台收事件；不要先阻塞等 session.created。
            session_update = {
                "type": "session.update",
                "session": {
                    "instructions": TRANSCRIBE_INSTRUCTIONS,
                    "voice": "chelsie",
                    "turn_detection": vad,
                    "extra_body": {"auto_response": True, "language": "zh"},
                },
            }
            if model:
                session_update["model"] = model
            await ws.send(json.dumps(session_update, ensure_ascii=False))
            log("send", session_update)

            incoming: asyncio.Queue[Any] = asyncio.Queue()

            async def _reader() -> None:
                try:
                    while True:
                        incoming.put_nowait(await ws.recv())
                except Exception as exc:  # noqa: BLE001
                    incoming.put_nowait(exc)

            reader_task = asyncio.create_task(_reader())

            async def _consume_one(raw: Any) -> str:
                nonlocal saw_terminal
                if isinstance(raw, Exception):
                    raise raw
                if isinstance(raw, bytes):
                    log("recv_binary", {"nbytes": len(raw)})
                    return "binary"
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    log("recv_non_json", {"raw": str(raw)[:300]})
                    return "non_json"
                log("recv", event)
                _extract_text_from_realtime_event(event, text_parts)
                etype = str(event.get("type") or "")
                if etype == "error":
                    result.error = json.dumps(event.get("error") or event, ensure_ascii=False)
                    saw_terminal = True
                elif etype in {"response.done", "session.closed"}:
                    saw_terminal = True
                return etype

            # 等 session.updated；期间若已 error 则直接结束本路
            configured = False
            config_deadline = time.perf_counter() + 15.0
            while time.perf_counter() < config_deadline and not configured:
                timeout = max(0.05, config_deadline - time.perf_counter())
                try:
                    raw = await asyncio.wait_for(incoming.get(), timeout=timeout)
                except asyncio.TimeoutError:
                    break
                etype = await _consume_one(raw)
                if etype == "error":
                    raise RuntimeError(result.error or "session.update rejected")
                if etype in {"session.updated", "transcription_session.updated"}:
                    configured = True
                elif etype in {"session.created", "transcription_session.created"}:
                    log("session.created seen; continue waiting for session.updated")
            if not configured:
                log("session.updated not seen within 15s; continue sending audio")

            bytes_per_chunk = max(2, int(sample_rate * (chunk_ms / 1000.0)) * 2)
            total = len(pcm)
            sent = 0
            chunk_idx = 0
            while sent < total:
                # 发送期间顺手排空服务端事件，避免缓冲积压
                while True:
                    try:
                        raw = incoming.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    etype = await _consume_one(raw)
                    if etype == "error":
                        raise RuntimeError(result.error or "realtime error during audio send")

                chunk = pcm[sent : sent + bytes_per_chunk]
                sent += len(chunk)
                chunk_idx += 1
                event = {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(chunk).decode("ascii"),
                }
                await ws.send(json.dumps(event))
                if chunk_idx == 1 or sent >= total or chunk_idx % 20 == 0:
                    log(
                        "send_audio_chunk",
                        {
                            "type": "input_audio_buffer.append",
                            "chunk_index": chunk_idx,
                            "bytes": len(chunk),
                            "sent_bytes": sent,
                            "total_bytes": total,
                        },
                    )
                if pace > 0:
                    await asyncio.sleep((len(chunk) / 2 / sample_rate) / pace)

            if trailing_silence_ms > 0:
                silence = b"\x00\x00" * int(sample_rate * (trailing_silence_ms / 1000.0))
                await ws.send(
                    json.dumps(
                        {
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(silence).decode("ascii"),
                        }
                    )
                )
                log(
                    "send_trailing_silence",
                    {"trailing_silence_ms": trailing_silence_ms, "bytes": len(silence)},
                )

            # server_vad 下一般自动 commit；再显式 commit 一次兜底
            commit = {"type": "input_audio_buffer.commit"}
            await ws.send(json.dumps(commit))
            log("send", commit)

            deadline = time.perf_counter() + overall_timeout
            last_msg = time.perf_counter()
            while time.perf_counter() < deadline:
                remaining = min(idle_timeout, deadline - time.perf_counter())
                if remaining <= 0:
                    break
                try:
                    raw = await asyncio.wait_for(incoming.get(), timeout=remaining)
                except asyncio.TimeoutError:
                    if time.perf_counter() - last_msg >= idle_timeout:
                        log(f"idle_timeout {idle_timeout}s, stop receiving")
                        break
                    continue
                last_msg = time.perf_counter()
                etype = await _consume_one(raw)
                if etype in {"response.done", "error", "session.closed"}:
                    deadline = min(deadline, time.perf_counter() + 2.0)
            if not saw_terminal and not text_parts:
                log("finished without response.done / transcript")

            reader_task.cancel()
            try:
                await reader_task
            except asyncio.CancelledError:
                pass
    except ConnectionClosed as exc:
        result.error = f"websocket closed: {exc}"
        result.events.append(_ts_line(f"error {result.error}"))
    except Exception as exc:  # noqa: BLE001 - 探测脚本需落盘异常
        result.error = f"{type(exc).__name__}: {exc}"
        result.events.append(_ts_line(f"error {result.error}"))

    # 去重拼接：delta 累加优先；若只有 done 全文则用最后一段
    if text_parts:
        # 若出现完整 done 文本覆盖，保留最长非空片段拼接观感
        joined = "".join(text_parts)
        # 简单去重连续重复块
        result.transcript = joined.strip()
    result.finished_at = _now()
    result.elapsed_sec = time.perf_counter() - t0
    return result


def run_chat_completions(
    *,
    base_url: str,
    model: str,
    wav_path: Path,
    max_tokens: int,
    temperature: float,
    timeout: float,
) -> ProbeResult:
    result = ProbeResult(
        api="/v1/chat/completions",
        label="chat-completions",
        vad_config={
            "note": "HTTP /v1/chat/completions 无 Silero VAD / turn_detection",
            "modalities": ["text"],
            "stream": False,
            "language": "zh",
            "temperature": temperature,
            "max_tokens": max_tokens,
            "system_prompt": TRANSCRIBE_INSTRUCTIONS,
            "user_content": ["text:<|audio|>", "input_audio:wav-base64"],
        },
        started_at=_now(),
    )
    t0 = time.perf_counter()
    result.events.append(_ts_line(f"prepare request model={model} wav={wav_path}"))
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": TRANSCRIBE_INSTRUCTIONS},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "<|audio|>"},
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": base64.b64encode(wav_path.read_bytes()).decode("ascii"),
                            "format": wav_path.suffix.lower().lstrip(".") or "wav",
                        },
                    },
                ],
            },
        ],
        "modalities": ["text"],
        "stream": False,
        "language": "zh",
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    result.events.append(
        _ts_line(
            "send "
            + json.dumps(
                {
                    "url": req.full_url,
                    "model": model,
                    "modalities": ["text"],
                    "language": "zh",
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "audio_bytes": wav_path.stat().st_size,
                    "body_bytes": len(body),
                },
                ensure_ascii=False,
            )
        )
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            result.events.append(_ts_line(f"http_status {resp.status}"))
        data = json.loads(raw)
        result.events.append(
            _ts_line("recv " + json.dumps(_shrink_event(data), ensure_ascii=False, separators=(",", ":")))
        )
        content = (((data.get("choices") or [{}])[0].get("message") or {}).get("content")) or ""
        result.transcript = str(content).strip()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        result.error = f"HTTP {exc.code}: {detail}"
        result.events.append(_ts_line(f"error {result.error}"))
    except Exception as exc:  # noqa: BLE001
        result.error = f"{type(exc).__name__}: {exc}"
        result.events.append(_ts_line(f"error {result.error}"))
    result.finished_at = _now()
    result.elapsed_sec = time.perf_counter() - t0
    return result


def format_report(
    *,
    wav: Path,
    host: str,
    port: int,
    model: str,
    audio_duration: float,
    results: list[ProbeResult],
) -> str:
    lines: list[str] = []
    lines.append("Sophon 双接口 VAD 对照测试报告")
    lines.append(f"生成时间: {_now()}")
    lines.append(f"音频: {wav}")
    lines.append(f"音频时长秒: {audio_duration:.3f}")
    lines.append(f"服务: http://{host}:{port}")
    lines.append(f"模型: {model}")
    lines.append(f"并行任务数: {len(results)}")
    lines.append("")
    lines.append("Realtime 五档 VAD 一览（从严到松）:")
    for tier in VAD_TIERS:
        td = tier["turn_detection"]
        lines.append(
            f"  tier{tier['tier']} {tier['name']}: "
            f"threshold={td['threshold']}, "
            f"min_speech_duration_ms={td['min_speech_duration_ms']}, "
            f"silence_duration_ms={td['silence_duration_ms']}, "
            f"prefix_padding_ms={td['prefix_padding_ms']}"
        )
    lines.append("")
    lines.append("=" * 88)

    for item in results:
        lines.append("")
        lines.append(f"## {item.label}  api={item.api}")
        lines.append(f"开始: {item.started_at}  结束: {item.finished_at}  耗时秒: {item.elapsed_sec:.3f}")
        if item.error:
            lines.append(f"错误: {item.error}")
        lines.append(
            "VAD配置："
            + json.dumps(item.vad_config, ensure_ascii=False, separators=(",", ":"))
        )
        lines.append("转写结果：" + (item.transcript if item.transcript else "<空>"))
        lines.append("模型解析日志：")
        if item.events:
            lines.extend(item.events)
        else:
            lines.append("<无事件>")
        lines.append("-" * 88)
    lines.append("")
    lines.append("完成。")
    return "\n".join(lines) + "\n"


async def run_all(args: argparse.Namespace) -> Path:
    wav = Path(args.wav).expanduser().resolve()
    if not wav.is_file():
        raise FileNotFoundError(f"找不到 wav: {wav}")

    base_url = f"http://{args.host}:{args.port}"
    model = args.model or fetch_model_id(base_url)
    print(f"model={model}", flush=True)

    pcm, duration = load_pcm16(wav, args.sample_rate)
    print(
        f"pcm_sr={args.sample_rate} duration={duration:.3f}s bytes={len(pcm)}",
        flush=True,
    )

    out = Path(args.out).expanduser().resolve() if args.out else None
    if out is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = Path("tmp") / f"sophon_vad_dual_{stamp}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)

    print("并行启动: chat/completions + realtime×5", flush=True)
    chat_future = asyncio.to_thread(
        run_chat_completions,
        base_url=base_url,
        model=model,
        wav_path=wav,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        timeout=args.chat_timeout,
    )
    realtime_futures = [
        run_realtime(
            host=args.host,
            port=args.port,
            model=model,
            pcm=pcm,
            sample_rate=args.sample_rate,
            tier=tier,
            chunk_ms=args.chunk_ms,
            pace=args.pace,
            trailing_silence_ms=args.trailing_silence_ms,
            idle_timeout=args.idle_timeout,
            overall_timeout=args.realtime_timeout,
        )
        for tier in VAD_TIERS
    ]
    gathered = await asyncio.gather(chat_future, *realtime_futures, return_exceptions=False)
    # 报告顺序：先五档 realtime，再 chat
    results = list(gathered[1:]) + [gathered[0]]
    report = format_report(
        wav=wav,
        host=args.host,
        port=args.port,
        model=model,
        audio_duration=duration,
        results=results,
    )
    out.write_text(report, encoding="utf-8")
    print(f"报告: {out}", flush=True)
    for item in results:
        preview = (item.transcript or item.error or "<空>")[:80].replace("\n", " ")
        print(f"  {item.label}: {preview}", flush=True)
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sophon chat + realtime 五档 VAD 并行探测")
    p.add_argument("wav", help="长语音 wav 路径")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=6179)
    p.add_argument("--model", default="", help="默认从 /v1/models 取第一个 ID")
    p.add_argument("--out", default="", help="输出 txt；默认 tmp/sophon_vad_dual_<时间戳>.txt")
    p.add_argument("--sample-rate", type=int, default=24000, help="realtime PCM 采样率")
    p.add_argument("--chunk-ms", type=int, default=100, help="realtime 每包音频毫秒")
    p.add_argument(
        "--pace",
        type=float,
        default=0.0,
        help="发送倍速；0=尽快推完，1=近似实时，2=两倍速",
    )
    p.add_argument("--trailing-silence-ms", type=int, default=1500, help="音频结束后补静音触发 VAD")
    p.add_argument("--idle-timeout", type=float, default=20.0, help="realtime 无新消息超时秒")
    p.add_argument("--realtime-timeout", type=float, default=600.0, help="单路 realtime 总超时秒")
    p.add_argument("--chat-timeout", type=float, default=600.0, help="chat/completions 超时秒")
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.0)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        asyncio.run(run_all(args))
    except KeyboardInterrupt:
        print("中断", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001
        print(f"失败: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
