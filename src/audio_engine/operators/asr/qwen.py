from __future__ import annotations

import json
import os
import threading
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

_MODEL_CACHE: dict[str, Any] = {}
_MODEL_LOCK = threading.Lock()


def _load_asr_config(config_path: str | None) -> dict[str, Any]:
    if not config_path:
        return {}
    path = Path(config_path)
    if path.exists():
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {}


def _resolve_settings(config: OperatorConfig) -> dict[str, Any]:
    settings = _load_asr_config(config.params.get("config_path", "configs/asr/qwen_asr.yaml"))
    settings.update({key: value for key, value in config.params.items() if key != "config_path"})
    registered_model = os.path.expandvars(str(config.params.get("registered_model") or ""))
    if registered_model:
        from audio_engine.core.catalog import ArtifactCatalog

        model = ArtifactCatalog(config.params.get("catalog_dir", "data/catalog")).get_model(
            str(registered_model)
        )
        if model.status != "ready":
            raise ValueError(f"registered model is not ready: {model.model_id} ({model.status})")
        settings.update(
            {
                "model_path": model.checkpoint_uri,
                "model": model.model_id,
                "model_version": model.model_id,
            }
        )
    settings["model_path"] = (
        config.params.get("model_path")
        or os.environ.get("QWEN_ASR_MODEL_PATH")
        or settings.get("model_path")
        or settings.get("model", "Qwen/Qwen3-ASR-1.7B")
    )
    settings["api_base"] = (
        config.params.get("api_base")
        or os.environ.get(str(config.params.get("api_base_env") or ""))
        or os.environ.get("QWEN_ASR_API_BASE")
        or settings.get("api_base")
    )
    settings["api_key"] = (
        config.params.get("api_key")
        or os.environ.get(str(config.params.get("api_key_env") or ""))
        or os.environ.get("QWEN_ASR_API_KEY")
        or settings.get("api_key")
    )
    return settings


def _resolve_batch_settings(config: OperatorConfig) -> dict[str, Any]:
    """Resolve Qwen batch settings; batch inference is vLLM-only."""
    settings = _resolve_settings(config)
    if not settings.get("api_base") and not (config.mock or config.params.get("mock")):
        raise ValueError(
            "Qwen batch inference only supports vLLM. Export QWEN_ASR_API_BASE "
            "or set params.api_base."
        )
    return settings


def _cache_config(config: OperatorConfig, settings: dict[str, Any]) -> OperatorConfig:
    params = dict(config.params)
    for key in (
        "model",
        "model_version",
        "model_path",
        "dtype",
        "device_map",
        "api_base",
        "language",
        "prompt",
    ):
        if key in settings:
            params[f"resolved_{key}"] = settings[key]
    return config.model_copy(update={"params": params})


def _load_qwen_model(settings: dict[str, Any]) -> Any:
    model_kwargs = {
        "dtype": settings.get("dtype", "float16"),
        "device_map": settings.get("device_map", "cuda:0"),
        "max_inference_batch_size": int(settings.get("batch_size", 8)),
        "max_new_tokens": int(settings.get("max_new_tokens", 256)),
    }
    cache_key = json.dumps(
        {"model_path": settings["model_path"], **model_kwargs},
        sort_keys=True,
        ensure_ascii=False,
    )
    with _MODEL_LOCK:
        if cache_key in _MODEL_CACHE:
            return _MODEL_CACHE[cache_key]
        try:
            import torch
            from qwen_asr import Qwen3ASRModel
        except ImportError as exc:
            raise RuntimeError(
                "本地 Qwen3-ASR 推理依赖未安装，请执行：pip install 'audio-data-engine[asr]'"
            ) from exc

        dtype_name = str(model_kwargs["dtype"]).removeprefix("torch.")
        try:
            model_kwargs["dtype"] = getattr(torch, dtype_name)
        except AttributeError as exc:
            raise ValueError(f"不支持的 Qwen ASR dtype: {dtype_name}") from exc

        model = Qwen3ASRModel.from_pretrained(settings["model_path"], **model_kwargs)
        _MODEL_CACHE[cache_key] = model
        return model


def release_cached_models() -> int:
    """Drop in-process Qwen model cache so the next ASR stage can reclaim GPU memory."""
    with _MODEL_LOCK:
        count = len(_MODEL_CACHE)
        _MODEL_CACHE.clear()
    return count


def _transcribe_many(
    model: Any, audio_paths: list[str], settings: dict[str, Any]
) -> list[dict[str, Any]]:
    kwargs: dict[str, Any] = {
        "audio": audio_paths,
        "return_time_stamps": bool(settings.get("return_time_stamps", False)),
    }
    language = settings.get("language")
    if language:
        kwargs["language"] = [language] * len(audio_paths)
    context = settings.get("context")
    if context:
        kwargs["context"] = [context] * len(audio_paths)

    raw_results = model.transcribe(**kwargs)
    if len(raw_results) != len(audio_paths):
        raise RuntimeError(
            f"Qwen3-ASR 返回 {len(raw_results)} 条结果，输入为 {len(audio_paths)} 条"
        )

    results = []
    for raw in raw_results:
        if isinstance(raw, dict):
            text = str(raw.get("text", ""))
            detected_language = raw.get("language")
        else:
            text = str(getattr(raw, "text", ""))
            detected_language = getattr(raw, "language", None)
        results.append({"text": text, "language": detected_language})
    return results


@register_operator
class QwenASROperator(BaseASROperator):
    name = "qwen"
    version = "2.0.0"

    def compute_cache_key(self, sample: Sample, config: OperatorConfig) -> str:
        settings = _resolve_settings(config)
        return super().compute_cache_key(sample, _cache_config(config, settings))

    def transcribe(self, sample: Sample, config: OperatorConfig) -> dict[str, Any]:
        settings = _resolve_settings(config)
        model_name = settings.get("model", "qwen3-asr")
        model_version = settings.get("model_version", settings.get("version", "unknown"))
        api_url = settings.get("api_url")
        api_base = settings.get("api_base")

        if config.mock or config.params.get("mock"):
            text = self._mock_transcript(sample, config)
            language = settings.get("language")
        elif api_base:
            result = call_vllm_transcription(
                sample.audio_path(config.params.get("input_audio_key", "raw")), settings
            )
            text = result["text"]
            language = result.get("language")
        elif api_url:
            text = self._call_remote_api(sample, config, api_url, settings)
            language = settings.get("language")
        else:
            local_result = self._call_local_model(sample, config, settings)
            text = local_result["text"]
            language = local_result.get("language")

        return {
            "text": text,
            "model": model_name,
            "version": model_version,
            "extra": {"language": language} if language else {},
        }

    def _call_remote_api(
        self,
        sample: Sample,
        config: OperatorConfig,
        api_url: str,
        asr_cfg: dict[str, Any],
    ) -> str:
        import urllib.request

        input_key = config.params.get("input_audio_key", "raw")
        audio_path = sample.audio_path(input_key)
        req = urllib.request.Request(
            api_url,
            data=json.dumps({"path": audio_path}, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=asr_cfg.get("timeout", 60)) as resp:
            body = json.loads(resp.read().decode())
            return str(body.get("text", ""))

    def _call_local_model(
        self,
        sample: Sample,
        config: OperatorConfig,
        asr_cfg: dict[str, Any],
    ) -> dict[str, Any]:
        input_key = config.params.get("input_audio_key", "raw")
        model = _load_qwen_model(asr_cfg)
        return _transcribe_many(model, [sample.audio_path(input_key)], asr_cfg)[0]


@register_operator
class QwenBatchASROperator(BatchOperator):
    """vLLM-only Qwen3-ASR batch inference with cache and failure isolation."""

    name = "qwen_batch"
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
        result = call_vllm_transcription(path, settings)
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
                "Qwen batch backend=vllm api_base={} model={} force={} samples={}",
                settings.get("api_base"),
                settings.get("model", "qwen3-asr"),
                config.force,
                len(samples),
            )
        logger.info(
            "Qwen batch decision: pending={} skipped={} cache_hits={}",
            len(pending),
            skipped,
            cache_hits,
        )

        if pending and (config.mock or config.params.get("mock")):
            for index, sample, cache_key in pending:
                result = {
                    "text": f"[mock:qwen:{sample.id}]",
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
            raise RuntimeError("Qwen batch operator produced an incomplete result set")
        return [result for result in results if result is not None]

    def _process_vllm_chunk(
        self,
        chunk: list[tuple[int, Sample, str]],
        config: OperatorConfig,
        settings: dict[str, Any],
        results: list[OperatorResult | None],
    ) -> None:
        input_key = config.params.get("input_audio_key", "raw")
        concurrency = min(max(1, int(settings.get("concurrency", 4))), len(chunk))

        def transcribe(item: tuple[int, Sample, str]) -> tuple[int, dict[str, Any]]:
            index, sample, _ = item
            return index, call_vllm_transcription(sample.audio_path(input_key), settings)

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
            "model": settings.get("model", "qwen3-asr"),
            "version": settings.get("model_version", settings.get("version", "unknown")),
            "extra": {"language": result.get("language")} if result.get("language") else {},
        }
        input_key = config.params.get("input_audio_key", "raw")
        transcript_key = str(config.params.get("transcript_key", "qwen"))
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
