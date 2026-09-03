from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml

from audio_engine.core.operator import BatchOperator, OperatorConfig, OperatorResult
from audio_engine.core.registry import register_operator
from audio_engine.core.sample import Sample
from audio_engine.operators.asr.base import BaseASROperator

SUBMIT_URL = "https://openspeech.bytedance.com/api/v3/auc/bigmodel/submit"
QUERY_URL = "https://openspeech.bytedance.com/api/v3/auc/bigmodel/query"

PROCESSING_STATUS_CODES = {"20000001", "20000002"}
SUCCESS_STATUS_CODE = "20000000"

FORMAT_BY_EXT = {
    ".wav": "wav",
    ".mp3": "mp3",
    ".ogg": "ogg",
    ".pcm": "pcm",
    ".spx": "spx",
    ".amr": "amr",
    ".aac": "aac",
    ".m4a": "m4a",
    ".raw": "raw",
}


def _load_asr_config(config_path: str | None) -> dict[str, Any]:
    if not config_path:
        return {}
    path = Path(config_path)
    if path.exists():
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {}


def _resolve_settings(config: OperatorConfig) -> dict[str, Any]:
    settings = _load_asr_config(config.params.get("config_path", "configs/asr/doubao.yaml"))
    settings.update({key: value for key, value in config.params.items() if key != "config_path"})

    settings["api_key"] = (
        config.params.get("api_key")
        or os.environ.get("DOUBAO_ASR_API_KEY")
        or settings.get("api_key")
    )
    settings["app_key"] = (
        config.params.get("app_key")
        or os.environ.get("DOUBAO_ASR_APP_KEY")
        or settings.get("app_key")
    )
    settings["access_key"] = (
        config.params.get("access_key")
        or os.environ.get("DOUBAO_ASR_ACCESS_KEY")
        or settings.get("access_key")
    )
    settings["resource_id"] = (
        config.params.get("resource_id")
        or os.environ.get("DOUBAO_ASR_RESOURCE_ID")
        or settings.get("resource_id")
        or "volc.seedasr.auc"
    )
    settings["poll_interval"] = float(settings.get("poll_interval", 2.0))
    settings["max_polls"] = max(1, int(settings.get("max_polls", 150)))
    settings["timeout"] = float(settings.get("timeout", 300))
    settings["user_id"] = str(settings.get("user_id", "audio-data-engine"))
    settings["concurrency"] = max(
        1,
        int(settings.get("concurrency", settings.get("batch_size", 4))),
    )
    return settings


def _cache_config(config: OperatorConfig, settings: dict[str, Any]) -> OperatorConfig:
    params = dict(config.params)
    for key in (
        "resource_id",
        "model",
        "model_version",
        "language",
        "format",
        "enable_itn",
        "enable_punc",
        "show_utterances",
    ):
        if key in settings and settings[key] is not None:
            params[f"resolved_{key}"] = settings[key]
    return config.model_copy(update={"params": params})


def _detect_format(path: Path, settings: dict[str, Any]) -> str:
    if settings.get("format"):
        return str(settings["format"])
    return FORMAT_BY_EXT.get(path.suffix.lower(), "wav")


def _build_headers(settings: dict[str, Any], request_id: str) -> dict[str, str]:
    headers = {
        "Content-Type": "application/json",
        "X-Api-Resource-Id": str(settings["resource_id"]),
        "X-Api-Request-Id": request_id,
        "X-Api-Sequence": "-1",
    }
    if settings.get("api_key"):
        headers["X-Api-Key"] = str(settings["api_key"])
    elif settings.get("app_key") and settings.get("access_key"):
        headers["X-Api-App-Key"] = str(settings["app_key"])
        headers["X-Api-Access-Key"] = str(settings["access_key"])
    else:
        raise RuntimeError(
            "豆包 ASR 鉴权未配置：请设置 DOUBAO_ASR_API_KEY，"
            "或 DOUBAO_ASR_APP_KEY + DOUBAO_ASR_ACCESS_KEY"
        )
    return headers


def _normalize_headers(raw_headers: Any) -> dict[str, str]:
    return {key.lower(): value for key, value in raw_headers.items()}


def _http_post(url: str, headers: dict[str, str], payload: dict[str, Any], timeout: float) -> tuple[dict[str, str], bytes]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return _normalize_headers(response.headers), response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"豆包 ASR HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"豆包 ASR 请求失败: {exc.reason}") from exc


def _build_audio_payload(path: Path, settings: dict[str, Any]) -> dict[str, Any]:
    audio_format = _detect_format(path, settings)
    audio: dict[str, Any] = {"format": audio_format}

    if settings.get("audio_url"):
        audio["url"] = str(settings["audio_url"])
    else:
        if not path.is_file():
            raise FileNotFoundError(f"音频文件不存在: {path}")
        audio["data"] = base64.b64encode(path.read_bytes()).decode("ascii")

    if settings.get("language"):
        audio["language"] = str(settings["language"])
    for key in ("codec", "rate", "bits", "channel"):
        if settings.get(key) is not None:
            audio[key] = settings[key]
    return audio


def _build_request_payload(settings: dict[str, Any]) -> dict[str, Any]:
    request: dict[str, Any] = {"model_name": "bigmodel"}
    for key in (
        "enable_itn",
        "enable_punc",
        "enable_ddc",
        "enable_speaker_info",
        "enable_channel_split",
        "show_utterances",
        "show_speech_rate",
        "show_volume",
        "enable_auto_lang",
        "enable_lid",
        "enable_emotion_detection",
        "enable_gender_detection",
        "enable_age_detection",
        "vad_segment",
        "end_window_size",
        "ssd_version",
        "ssd_mode",
        "enable_poi_fc",
        "enable_music_fc",
    ):
        if settings.get(key) is not None:
            request[key] = settings[key]
    if settings.get("corpus") is not None:
        request["corpus"] = settings["corpus"]
    return request


def _parse_json_body(body: bytes) -> dict[str, Any]:
    if not body:
        return {}
    text = body.decode("utf-8").strip()
    if not text or text == "{}":
        return {}
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise RuntimeError(f"豆包 ASR 返回格式无效: {payload!r}")
    return payload


def _extract_transcript(payload: dict[str, Any]) -> dict[str, Any]:
    result = payload.get("result")
    if not isinstance(result, dict):
        return {}
    text = result.get("text")
    if not text:
        return {}
    extra: dict[str, Any] = {}
    if payload.get("audio_info"):
        extra["audio_info"] = payload["audio_info"]
    if result.get("utterances"):
        extra["utterances"] = result["utterances"]
    return {"text": str(text), "extra": extra}


def _submit_task(headers: dict[str, str], payload: dict[str, Any], timeout: float) -> dict[str, str]:
    response_headers, _ = _http_post(SUBMIT_URL, headers, payload, timeout)
    status_code = response_headers.get("x-api-status-code", "")
    message = response_headers.get("x-api-message", "")
    if status_code and status_code not in {SUCCESS_STATUS_CODE, *PROCESSING_STATUS_CODES}:
        log_id = response_headers.get("x-tt-logid", "")
        raise RuntimeError(
            f"豆包 ASR 提交失败: status={status_code}, message={message}, logid={log_id}"
        )
    return response_headers


def _poll_result(headers: dict[str, str], settings: dict[str, Any]) -> dict[str, Any]:
    poll_interval = float(settings["poll_interval"])
    max_polls = int(settings["max_polls"])
    timeout = float(settings["timeout"])

    for _ in range(max_polls):
        time.sleep(poll_interval)
        response_headers, body = _http_post(QUERY_URL, headers, {}, timeout)
        status_code = response_headers.get("x-api-status-code", "")
        message = response_headers.get("x-api-message", "")
        payload = _parse_json_body(body)

        if status_code == SUCCESS_STATUS_CODE:
            transcript = _extract_transcript(payload)
            if transcript:
                return transcript
            if payload:
                raise RuntimeError(f"豆包 ASR 任务完成但未返回文本: {payload!r}")

        if status_code in PROCESSING_STATUS_CODES or not status_code:
            transcript = _extract_transcript(payload)
            if transcript:
                return transcript
            continue

        log_id = response_headers.get("x-tt-logid", "")
        raise RuntimeError(
            f"豆包 ASR 查询失败: status={status_code}, message={message}, logid={log_id}"
        )

    raise TimeoutError(
        f"豆包 ASR 轮询超时（{max_polls} 次 × {poll_interval}s）"
    )


def call_doubao_transcription(audio_path: str, settings: dict[str, Any]) -> dict[str, Any]:
    path = Path(audio_path)
    request_id = str(settings.get("request_id") or uuid4())
    headers = _build_headers(settings, request_id)
    payload = {
        "user": {"uid": settings["user_id"]},
        "audio": _build_audio_payload(path, settings),
        "request": _build_request_payload(settings),
    }
    if settings.get("callback"):
        payload["callback"] = settings["callback"]
    if settings.get("callback_data"):
        payload["callback_data"] = settings["callback_data"]

    _submit_task(headers, payload, float(settings["timeout"]))
    result = _poll_result(headers, settings)
    return {
        "text": result["text"],
        "language": settings.get("language"),
        "extra": result.get("extra", {}),
    }


def _transcribe_many(audio_paths: list[str], settings: dict[str, Any]) -> list[dict[str, Any]]:
    if not audio_paths:
        return []

    concurrency = min(int(settings["concurrency"]), len(audio_paths))
    if concurrency <= 1:
        return [call_doubao_transcription(path, settings) for path in audio_paths]

    results: list[dict[str, Any] | None] = [None] * len(audio_paths)
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        future_map = {
            pool.submit(call_doubao_transcription, path, settings): index
            for index, path in enumerate(audio_paths)
        }
        for future in as_completed(future_map):
            index = future_map[future]
            results[index] = future.result()
    return [item for item in results if item is not None]


@register_operator
class DoubaoASROperator(BaseASROperator):
    """Doubao (Volcengine Seed ASR) file transcription via submit + poll API."""

    name = "doubao"
    version = "1.0.0"

    def compute_cache_key(self, sample: Sample, config: OperatorConfig) -> str:
        settings = _resolve_settings(config)
        return super().compute_cache_key(sample, _cache_config(config, settings))

    def transcribe(self, sample: Sample, config: OperatorConfig) -> dict[str, Any]:
        settings = _resolve_settings(config)
        if config.mock or config.params.get("mock"):
            text = self._mock_transcript(sample, config)
            language = settings.get("language")
            extra: dict[str, Any] = {}
        else:
            input_key = config.params.get("input_audio_key", "raw")
            result = call_doubao_transcription(sample.audio_path(input_key), settings)
            text = result["text"]
            language = result.get("language")
            extra = result.get("extra", {})

        return {
            "text": text,
            "model": settings.get("model", settings.get("resource_id", "volc.seedasr.auc")),
            "version": settings.get("model_version", settings.get("version", "unknown")),
            "extra": {"language": language, **extra} if language or extra else extra,
        }


@register_operator
class DoubaoBatchASROperator(BatchOperator):
    """Concurrent Doubao ASR via submit + poll HTTP API."""

    name = "doubao_batch"
    version = "1.0.0"
    category = "asr"

    def should_skip(self, sample: Sample, config: OperatorConfig) -> bool:
        transcript_key = config.params.get("transcript_key", "doubao")
        return not config.force and str(transcript_key) in sample.transcripts

    def _execute(self, sample: Sample, config: OperatorConfig) -> dict[str, Any]:
        settings = _resolve_settings(config)
        input_key = config.params.get("input_audio_key", "raw")
        result = _transcribe_many([sample.audio_path(input_key)], settings)[0]
        return self._updates(result, config, settings)

    def compute_cache_key(self, sample: Sample, config: OperatorConfig) -> str:
        settings = _resolve_settings(config)
        return super().compute_cache_key(sample, _cache_config(config, settings))

    def process_batch(
        self, samples: list[Sample], config: OperatorConfig
    ) -> list[OperatorResult]:
        settings = _resolve_settings(config)
        cache_config = _cache_config(config, settings)
        results: list[OperatorResult | None] = [None] * len(samples)
        pending: list[tuple[int, Sample, str]] = []

        for index, sample in enumerate(samples):
            if self.should_skip(sample, config):
                results[index] = OperatorResult(sample=sample, skipped=True)
                continue
            cache_key = super().compute_cache_key(sample, cache_config)
            if not config.force:
                cached = self.load_cache(cache_key, config)
                if cached is not None:
                    results[index] = OperatorResult(
                        sample=self.apply_cached(sample, cached),
                        cache_hit=True,
                    )
                    continue
            pending.append((index, sample, cache_key))

        if pending and (config.mock or config.params.get("mock")):
            for index, sample, cache_key in pending:
                result = {
                    "text": f"[mock:doubao:{sample.id}]",
                    "language": settings.get("language"),
                    "extra": {},
                }
                results[index] = self._finalize(sample, result, cache_key, config, settings)
        elif pending:
            input_key = config.params.get("input_audio_key", "raw")
            inference_batch_size = max(1, int(settings.get("batch_size", settings["concurrency"])))
            for start in range(0, len(pending), inference_batch_size):
                chunk = pending[start : start + inference_batch_size]
                self._process_inference_chunk(chunk, input_key, config, settings, results)

        if any(result is None for result in results):
            raise RuntimeError("Doubao batch operator produced an incomplete result set")
        return [result for result in results if result is not None]

    def _process_inference_chunk(
        self,
        chunk: list[tuple[int, Sample, str]],
        input_key: str,
        config: OperatorConfig,
        settings: dict[str, Any],
        results: list[OperatorResult | None],
    ) -> None:
        try:
            paths = [sample.audio_path(input_key) for _, sample, _ in chunk]
            transcripts = _transcribe_many(paths, settings)
        except Exception:  # noqa: BLE001 - retry a failed batch one sample at a time
            for index, sample, cache_key in chunk:
                try:
                    transcript = _transcribe_many(
                        [sample.audio_path(input_key)], settings
                    )[0]
                    results[index] = self._finalize(
                        sample, transcript, cache_key, config, settings
                    )
                except Exception as exc:  # noqa: BLE001 - isolate corrupt audio
                    results[index] = self._failed(sample, exc)
            return

        for (index, sample, cache_key), transcript in zip(chunk, transcripts):
            results[index] = self._finalize(sample, transcript, cache_key, config, settings)

    def _updates(
        self,
        result: dict[str, Any],
        config: OperatorConfig,
        settings: dict[str, Any],
    ) -> dict[str, Any]:
        language = result.get("language")
        extra = result.get("extra") or {}
        transcript = {
            "text": result["text"],
            "model": settings.get("model", settings.get("resource_id", "volc.seedasr.auc")),
            "version": settings.get("model_version", settings.get("version", "unknown")),
            "extra": {"language": language, **extra} if language or extra else extra,
        }
        transcript_key = str(config.params.get("transcript_key", "doubao"))
        input_key = config.params.get("input_audio_key", "raw")
        return {
            "transcripts": {transcript_key: transcript},
            "lineage_entry": {
                "operator": self.full_name,
                "version": self.version,
                "params": dict(config.params),
                "input_key": input_key,
            },
        }

    def _finalize(
        self,
        sample: Sample,
        result: dict[str, Any],
        cache_key: str,
        config: OperatorConfig,
        settings: dict[str, Any],
    ) -> OperatorResult:
        updates = self._updates(result, config, settings)
        updated = self._apply_updates(sample, updates)
        entry = updates["lineage_entry"]
        updated.add_lineage(
            operator=entry["operator"],
            version=entry["version"],
            params=entry["params"],
            input_key=entry["input_key"],
            cache_key=cache_key,
        )
        updated.mark_completed(self.full_name)
        self.save_cache(cache_key, config, updates)
        return OperatorResult(sample=updated, message="processed")

    def _failed(self, sample: Sample, exc: Exception) -> OperatorResult:
        failed = sample.model_copy(deep=True)
        failed.mark_failed(self.full_name, str(exc))
        return OperatorResult(sample=failed, message=str(exc))
