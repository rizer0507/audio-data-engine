from __future__ import annotations

import hashlib
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from audio_engine.core.operator import BatchOperator, OperatorConfig, OperatorResult
from audio_engine.core.registry import register_operator
from audio_engine.core.sample import Sample
from audio_engine.operators.asr.base import BaseASROperator
from audio_engine.operators.asr.vllm import call_vllm_transcription

_API_BASE_SPLIT = re.compile(r"[\s,;]+")


def _load_asr_config(config_path: str | None) -> dict[str, Any]:
    if not config_path:
        return {}
    path = Path(config_path)
    if path.exists():
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {}


def parse_glm_api_bases(raw: Any) -> list[str]:
    """Parse one or more vLLM bases from a string, list, or comma-separated env value."""
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        items = [str(item).strip() for item in raw]
    else:
        items = [part.strip() for part in _API_BASE_SPLIT.split(str(raw).strip())]
    seen: set[str] = set()
    bases: list[str] = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        bases.append(item)
    return bases


def _first_defined(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return None


def _resolve_settings(config: OperatorConfig) -> dict[str, Any]:
    settings = _load_asr_config(config.params.get("config_path", "configs/asr/glm.yaml"))
    settings.update({key: value for key, value in config.params.items() if key != "config_path"})
    bases = parse_glm_api_bases(
        _first_defined(
            config.params.get("api_bases"),
            config.params.get("api_base"),
            os.environ.get(str(config.params.get("api_base_env") or "")),
            os.environ.get("GLM_ASR_API_BASES"),
            os.environ.get("GLM_ASR_API_BASE"),
            settings.get("api_bases"),
            settings.get("api_base"),
        )
    )
    settings["api_bases"] = bases
    settings["api_base"] = bases[0] if bases else None
    settings["api_key"] = (
        config.params.get("api_key")
        or os.environ.get(str(config.params.get("api_key_env") or ""))
        or os.environ.get("GLM_ASR_API_KEY")
        or settings.get("api_key")
        or "dummy"
    )
    settings["model"] = (
        config.params.get("model")
        or os.environ.get(str(config.params.get("model_env") or ""))
        or os.environ.get("GLM_ASR_MODEL")
        or settings.get("model")
        or "glm-asr"
    )
    settings["concurrency"] = max(1, int(settings.get("concurrency", 8)))
    return settings


def _resolve_batch_settings(config: OperatorConfig) -> dict[str, Any]:
    """Resolve GLM batch settings; batch inference is vLLM-only."""
    settings = _resolve_settings(config)
    if not settings.get("api_base") and not (config.mock or config.params.get("mock")):
        raise ValueError(
            "GLM batch inference only supports vLLM. Export GLM_ASR_API_BASE "
            "or set params.api_base."
        )
    return settings


def _cache_config(config: OperatorConfig, settings: dict[str, Any]) -> OperatorConfig:
    params = dict(config.params)
    fingerprint_keys = (
        "model",
        "model_version",
        "version",
        "language",
        "prompt",
        "temperature",
        "response_format",
    )
    params["resolved_glm_vllm_settings"] = {
        key: settings[key] for key in fingerprint_keys if key in settings
    }
    params["resolved_glm_api_bases"] = list(settings.get("api_bases") or [])
    params["input_audio_key"] = config.params.get("input_audio_key", "raw")
    if settings.get("api_base") is not None:
        params["resolved_api_base"] = settings["api_base"]
    if settings.get("model") is not None:
        params["resolved_model"] = settings["model"]
    return config.model_copy(update={"params": params})


def select_glm_api_base(settings: dict[str, Any], sample_id: str) -> str:
    """Sticky-assign a replica by sample id so retries stay on the same endpoint."""
    bases = list(settings.get("api_bases") or [])
    if settings.get("api_base") and settings["api_base"] not in bases:
        bases = [str(settings["api_base"]), *bases]
    if not bases:
        raise ValueError("GLM ASR requires at least one vLLM api_base")
    if len(bases) == 1:
        return bases[0]
    digest = hashlib.sha256(str(sample_id).encode("utf-8")).digest()
    return bases[int.from_bytes(digest[:4], "big") % len(bases)]


def _request_settings(settings: dict[str, Any], sample_id: str) -> dict[str, Any]:
    request = dict(settings)
    request["api_base"] = select_glm_api_base(settings, sample_id)
    return request


@register_operator
class GlmASROperator(BaseASROperator):
    name = "glm"
    version = "1.0.0"

    def compute_cache_key(self, sample: Sample, config: OperatorConfig) -> str:
        settings = _resolve_settings(config)
        return super().compute_cache_key(sample, _cache_config(config, settings))

    def transcribe(self, sample: Sample, config: OperatorConfig) -> dict[str, Any]:
        settings = _resolve_settings(config)
        model_name = settings.get("model", "glm-asr")
        model_version = settings.get("model_version", settings.get("version", "unknown"))

        if config.mock or config.params.get("mock"):
            text = self._mock_transcript(sample, config)
            language = settings.get("language")
        elif settings.get("api_base"):
            result = call_vllm_transcription(
                sample.audio_path(config.params.get("input_audio_key", "raw")),
                _request_settings(settings, sample.id),
            )
            text = result["text"]
            language = result.get("language")
        else:
            raise ValueError(
                "GLM ASR only supports vLLM. Export GLM_ASR_API_BASE or set params.api_base."
            )

        return {
            "text": text,
            "model": model_name,
            "version": model_version,
            "extra": {"language": language} if language else {},
        }


@register_operator
class GlmBatchASROperator(BatchOperator):
    """vLLM-only GLM-ASR batch inference with cache and failure isolation."""

    name = "glm_batch"
    version = "1.0.0"
    category = "asr"

    def should_skip(self, sample: Sample, config: OperatorConfig) -> bool:
        transcript_key = config.params.get("transcript_key")
        if transcript_key:
            return not config.force and str(transcript_key) in sample.transcripts
        return super().should_skip(sample, config)

    def _execute(self, sample: Sample, config: OperatorConfig) -> dict[str, Any]:
        settings = _resolve_batch_settings(config)
        path = sample.audio_path(config.params.get("input_audio_key", "raw"))
        result = call_vllm_transcription(path, _request_settings(settings, sample.id))
        return self._updates(result, config, settings)

    def compute_cache_key(self, sample: Sample, config: OperatorConfig) -> str:
        settings = _resolve_batch_settings(config)
        return super().compute_cache_key(sample, _cache_config(config, settings))

    def process_batch(self, samples: list[Sample], config: OperatorConfig) -> list[OperatorResult]:
        results: list[OperatorResult | None] = [None] * len(samples)
        pending: list[tuple[int, Sample, str]] = []
        settings: dict[str, Any] | None = None

        def ensure_settings() -> dict[str, Any]:
            nonlocal settings
            if settings is None:
                settings = _resolve_batch_settings(config)
            return settings

        for index, sample in enumerate(samples):
            if self.should_skip(sample, config):
                results[index] = OperatorResult(sample=sample, skipped=True)
                continue
            cache_config = _cache_config(config, ensure_settings())
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

        skipped = sum(result is not None and result.skipped for result in results)
        cache_hits = sum(result is not None and result.cache_hit for result in results)
        if pending:
            settings = ensure_settings()
            logger.info(
                "GLM batch backend=vllm api_base={} model={} force={} samples={}",
                settings.get("api_base"),
                settings.get("model", "glm-asr"),
                config.force,
                len(samples),
            )
        logger.info(
            "GLM batch decision: pending={} skipped={} cache_hits={}",
            len(pending),
            skipped,
            cache_hits,
        )

        if pending and (config.mock or config.params.get("mock")):
            for index, sample, cache_key in pending:
                result = {
                    "text": f"[mock:glm:{sample.id}]",
                    "language": settings.get("language") if settings else None,
                }
                results[index] = self._finalize(sample, result, cache_key, config, settings or {})
        elif pending:
            assert settings is not None
            inference_batch_size = max(1, int(settings.get("batch_size", 8)))
            for start in range(0, len(pending), inference_batch_size):
                chunk = pending[start : start + inference_batch_size]
                self._process_vllm_chunk(chunk, config, settings, results)

        if any(result is None for result in results):
            raise RuntimeError("GLM batch operator produced an incomplete result set")
        return [result for result in results if result is not None]

    def _process_vllm_chunk(
        self,
        chunk: list[tuple[int, Sample, str]],
        config: OperatorConfig,
        settings: dict[str, Any],
        results: list[OperatorResult | None],
    ) -> None:
        input_key = config.params.get("input_audio_key", "raw")
        concurrency = min(max(1, int(settings.get("concurrency", 8))), len(chunk))

        def transcribe(item: tuple[int, Sample, str]) -> tuple[int, dict[str, Any]]:
            index, sample, _ = item
            return index, call_vllm_transcription(
                sample.audio_path(input_key),
                _request_settings(settings, sample.id),
            )

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {pool.submit(transcribe, item): item for item in chunk}
            for future in as_completed(futures):
                index, sample, cache_key = futures[future]
                try:
                    _, transcript = future.result()
                    results[index] = self._finalize(
                        sample, transcript, cache_key, config, settings
                    )
                except Exception as exc:  # noqa: BLE001 - isolate request failures
                    results[index] = self._failed(sample, exc)

    def _updates(
        self,
        result: dict[str, Any],
        config: OperatorConfig,
        settings: dict[str, Any],
    ) -> dict[str, Any]:
        transcript = {
            "text": result["text"],
            "model": settings.get("model", "glm-asr"),
            "version": settings.get("model_version", settings.get("version", "unknown")),
            "extra": {"language": result.get("language")} if result.get("language") else {},
        }
        input_key = config.params.get("input_audio_key", "raw")
        transcript_key = str(config.params.get("transcript_key", "glm"))
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
