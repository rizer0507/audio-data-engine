from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml

from audio_engine.core.operator import BatchOperator, OperatorConfig, OperatorResult
from audio_engine.core.registry import register_operator
from audio_engine.core.sample import Sample
from audio_engine.operators.asr.base import BaseASROperator
from audio_engine.operators.asr.vllm import call_vllm_transcription


def _load_asr_config(config_path: str | None) -> dict[str, Any]:
    if not config_path:
        return {}
    path = Path(config_path)
    if path.exists():
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {}


def _resolve_settings(config: OperatorConfig) -> dict[str, Any]:
    settings = _load_asr_config(config.params.get("config_path", "configs/asr/kimi.yaml"))
    settings.update({key: value for key, value in config.params.items() if key != "config_path"})
    settings["api_base"] = (
        config.params.get("api_base")
        or os.environ.get("KIMI_ASR_API_BASE")
        or settings.get("api_base")
        or "http://127.0.0.1:5554"
    )
    settings["model"] = (
        config.params.get("model")
        or os.environ.get("KIMI_ASR_MODEL")
        or settings.get("model")
        or "kimi-audio"
    )
    settings["concurrency"] = max(
        1,
        int(settings.get("concurrency", settings.get("batch_size", 4))),
    )
    return settings


def _cache_config(config: OperatorConfig, settings: dict[str, Any]) -> OperatorConfig:
    params = dict(config.params)
    # Keep all output-affecting server/request settings in the fingerprint while
    # excluding throughput-only knobs (concurrency, timeout and batch_size).
    fingerprint_keys = (
        "model",
        "model_version",
        "version",
        "api_base",
        "language",
        "prompt",
        "temperature",
        "response_format",
    )
    params["resolved_kimi_vllm_settings"] = {
        key: settings[key] for key in fingerprint_keys if key in settings
    }
    return config.model_copy(update={"params": params})


def _call_vllm_transcription(audio_path: str, settings: dict[str, Any]) -> dict[str, Any]:
    return call_vllm_transcription(audio_path, settings)


def _transcribe_many(audio_paths: list[str], settings: dict[str, Any]) -> list[dict[str, Any]]:
    if not audio_paths:
        return []

    concurrency = min(int(settings["concurrency"]), len(audio_paths))
    if concurrency <= 1:
        return [_call_vllm_transcription(path, settings) for path in audio_paths]

    results: list[dict[str, Any] | None] = [None] * len(audio_paths)
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        future_map = {
            pool.submit(_call_vllm_transcription, path, settings): index
            for index, path in enumerate(audio_paths)
        }
        for future in as_completed(future_map):
            index = future_map[future]
            results[index] = future.result()
    return [item for item in results if item is not None]


@register_operator
class KimiASROperator(BaseASROperator):
    name = "kimi"
    version = "3.0.0"

    def compute_cache_key(self, sample: Sample, config: OperatorConfig) -> str:
        settings = _resolve_settings(config)
        return super().compute_cache_key(sample, _cache_config(config, settings))

    def transcribe(self, sample: Sample, config: OperatorConfig) -> dict[str, Any]:
        settings = _resolve_settings(config)
        if config.mock or config.params.get("mock"):
            text = self._mock_transcript(sample, config)
            language = settings.get("language")
        else:
            input_key = config.params.get("input_audio_key", "raw")
            result = _call_vllm_transcription(sample.audio_path(input_key), settings)
            text = result["text"]
            language = result.get("language")

        return {
            "text": text,
            "model": settings.get("model", "kimi-audio"),
            "version": settings.get("model_version", settings.get("version", "unknown")),
            "extra": {"language": language} if language else {},
        }


@register_operator
class KimiBatchASROperator(BatchOperator):
    """Concurrent Kimi-Audio ASR via vLLM /v1/audio/transcriptions."""

    name = "kimi_batch"
    version = "3.0.0"
    category = "asr"

    def _execute(self, sample: Sample, config: OperatorConfig) -> dict[str, Any]:
        settings = _resolve_settings(config)
        input_key = config.params.get("input_audio_key", "raw")
        result = _transcribe_many([sample.audio_path(input_key)], settings)[0]
        return self._updates(result, config, settings)

    def compute_cache_key(self, sample: Sample, config: OperatorConfig) -> str:
        settings = _resolve_settings(config)
        return super().compute_cache_key(sample, _cache_config(config, settings))

    def process_batch(self, samples: list[Sample], config: OperatorConfig) -> list[OperatorResult]:
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
                    "text": f"[mock:kimi:{sample.id}]",
                    "language": settings.get("language"),
                }
                results[index] = self._finalize(sample, result, cache_key, config, settings)
        elif pending:
            input_key = config.params.get("input_audio_key", "raw")
            inference_batch_size = max(1, int(settings.get("batch_size", settings["concurrency"])))
            for start in range(0, len(pending), inference_batch_size):
                chunk = pending[start : start + inference_batch_size]
                self._process_inference_chunk(chunk, input_key, config, settings, results)

        if any(result is None for result in results):
            raise RuntimeError("Kimi batch operator produced an incomplete result set")
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
                    transcript = _transcribe_many([sample.audio_path(input_key)], settings)[0]
                    results[index] = self._finalize(sample, transcript, cache_key, config, settings)
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
        transcript = {
            "text": result["text"],
            "model": settings.get("model", "kimi-audio"),
            "version": settings.get("model_version", settings.get("version", "unknown")),
            "extra": {"language": result.get("language")} if result.get("language") else {},
        }
        input_key = config.params.get("input_audio_key", "raw")
        return {
            "transcripts": {"kimi": transcript},
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
