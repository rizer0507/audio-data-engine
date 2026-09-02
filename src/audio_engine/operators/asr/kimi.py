from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from audio_engine.core.operator import BatchOperator, OperatorConfig, OperatorResult
from audio_engine.core.registry import register_operator
from audio_engine.core.sample import Sample
from audio_engine.operators.asr.base import BaseASROperator

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
    settings = _load_asr_config(
        config.params.get("config_path", "configs/asr/kimi.yaml")
    )
    settings.update({key: value for key, value in config.params.items() if key != "config_path"})
    settings["model_path"] = (
        os.environ.get("KIMI_AUDIO_MODEL_PATH")
        or config.params.get("model_path")
        or settings.get("model_path")
        or settings.get("model", "moonshotai/Kimi-Audio-7B-Instruct")
    )
    settings["device"] = str(settings.get("device", "cuda"))
    settings["batch_size"] = max(1, int(settings.get("batch_size", 1)))
    settings["inference_threads"] = max(1, int(settings.get("inference_threads", 1)))
    return settings


def _cache_config(config: OperatorConfig, settings: dict[str, Any]) -> OperatorConfig:
    params = dict(config.params)
    # Persist every output-affecting setting in the cache fingerprint. Runtime-only
    # throughput knobs deliberately stay out so batch tuning can reuse transcripts.
    fingerprint_keys = (
        "model",
        "model_version",
        "version",
        "model_path",
        "device",
        "load_detokenizer",
        "prompt",
        "model_kwargs",
        "generation_kwargs",
    )
    params["resolved_kimi_settings"] = {
        key: settings[key] for key in fingerprint_keys if key in settings
    }
    return config.model_copy(update={"params": params})


def _load_kimi_audio_model(settings: dict[str, Any]) -> Any:
    model_path = str(settings["model_path"])
    device = str(settings.get("device", "cuda"))
    path = Path(model_path).expanduser()
    if path.is_absolute() and not path.is_dir():
        raise FileNotFoundError(
            f"Kimi-Audio 本地模型目录不存在: {path}. "
            "请检查 KIMI_AUDIO_MODEL_PATH 或 configs/asr/kimi.yaml"
        )
    model_kwargs = {
        "model_path": model_path,
        "load_detokenizer": bool(settings.get("load_detokenizer", False)),
    }
    extra_model_kwargs = settings.get("model_kwargs") or {}
    if isinstance(extra_model_kwargs, dict):
        model_kwargs.update(extra_model_kwargs)
    cache_key = json.dumps({"model_kwargs": model_kwargs, "device": device}, sort_keys=True)

    with _MODEL_LOCK:
        if cache_key in _MODEL_CACHE:
            logger.debug(
                "[DIAG][MODEL] Kimi-Audio cache hit: model_path={} device={}",
                model_path,
                device,
            )
            return _MODEL_CACHE[cache_key]
        try:
            from kimia_infer.api.kimia import KimiAudio  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Kimi-Audio 推理依赖未安装，请执行："
                "pip install 'audio-data-engine[kimi-audio]'"
            ) from exc
        logger.info(
            "[DIAG][MODEL] loading Kimi-Audio: path={} device={} load_detokenizer={}",
            model_path,
            device,
            model_kwargs["load_detokenizer"],
        )
        t0 = time.perf_counter()
        model = KimiAudio(**model_kwargs)
        if hasattr(model, "to"):
            model.to(device)
        elapsed = time.perf_counter() - t0
        logger.info(
            "[DIAG][MODEL] Kimi-Audio loaded OK in {:.2f}s | type={} | path={} | device={}",
            elapsed,
            type(model).__name__,
            model_path,
            device,
        )
        _MODEL_CACHE[cache_key] = model
        return model


def release_cached_models() -> int:
    """Drop in-process Kimi-Audio model cache so GPU memory can be reclaimed."""
    with _MODEL_LOCK:
        count = len(_MODEL_CACHE)
        _MODEL_CACHE.clear()
    return count


def _build_messages(audio_path: str, settings: dict[str, Any]) -> list[dict[str, Any]]:
    prompt = settings.get(
        "prompt",
        "Please transcribe the audio accurately. Output only the transcription.",
    )
    return [
        {"role": "user", "message_type": "text", "content": prompt},
        {"role": "user", "message_type": "audio", "content": audio_path},
    ]


def _transcribe_one(model: Any, audio_path: str, settings: dict[str, Any]) -> dict[str, Any]:
    messages = _build_messages(audio_path, settings)
    generation_kwargs = dict(settings.get("generation_kwargs") or {})
    _, text_output = model.generate(messages, **generation_kwargs, output_type="text")
    return {"text": str(text_output).strip()}


def _normalize_batch_results(
    raw_results: Any, audio_paths: list[str]
) -> list[dict[str, Any]]:
    if isinstance(raw_results, list) and len(raw_results) == len(audio_paths):
        normalized: list[dict[str, Any]] = []
        for item in raw_results:
            if isinstance(item, dict):
                normalized.append({"text": str(item.get("text", "")).strip()})
            else:
                normalized.append({"text": str(item).strip()})
        return normalized
    raise RuntimeError(
        f"Kimi-Audio batch 返回 {len(raw_results) if isinstance(raw_results, list) else 1} "
        f"条结果，输入为 {len(audio_paths)} 条"
    )


def _transcribe_many(
    model: Any, audio_paths: list[str], settings: dict[str, Any]
) -> list[dict[str, Any]]:
    if not audio_paths:
        return []

    if hasattr(model, "transcribe_batch"):
        prompt = settings.get("prompt")
        batch_kwargs: dict[str, Any] = {"output_type": "text"}
        batch_kwargs.update(settings.get("generation_kwargs") or {})
        raw_results = model.transcribe_batch(audio_paths, prompt=prompt, **batch_kwargs)
        return _normalize_batch_results(raw_results, audio_paths)

    if hasattr(model, "transcribe"):
        try:
            raw_results = model.transcribe(list(audio_paths))
            return _normalize_batch_results(raw_results, audio_paths)
        except TypeError:
            pass

    threads = min(int(settings["inference_threads"]), len(audio_paths))
    if threads <= 1:
        return [_transcribe_one(model, path, settings) for path in audio_paths]

    results: list[dict[str, Any] | None] = [None] * len(audio_paths)
    with ThreadPoolExecutor(max_workers=threads) as pool:
        future_map = {
            pool.submit(_transcribe_one, model, path, settings): index
            for index, path in enumerate(audio_paths)
        }
        for future in as_completed(future_map):
            index = future_map[future]
            results[index] = future.result()
    return [item for item in results if item is not None]


class _KimiAudioUpdates:
    category = "asr"
    name: str
    version: str

    @property
    def full_name(self) -> str:
        return f"{self.category}.{self.name}"

    def _updates(
        self, result: dict[str, Any], config: OperatorConfig, settings: dict[str, Any]
    ) -> dict[str, Any]:
        transcript = {
            "text": result["text"],
            "model": settings.get("model", "kimi-audio-7b-instruct"),
            "version": settings.get("model_version", settings.get("version", "unknown")),
            "extra": {},
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


@register_operator
class KimiASROperator(_KimiAudioUpdates, BaseASROperator):
    name = "kimi"
    version = "2.0.0"

    def compute_cache_key(self, sample: Sample, config: OperatorConfig) -> str:
        settings = _resolve_settings(config)
        return super().compute_cache_key(sample, _cache_config(config, settings))

    def transcribe(self, sample: Sample, config: OperatorConfig) -> dict[str, Any]:
        settings = _resolve_settings(config)
        if config.mock or config.params.get("mock"):
            text = f"[mock:kimi:{sample.id}]"
        else:
            input_key = config.params.get("input_audio_key", "raw")
            text = _transcribe_one(
                _load_kimi_audio_model(settings),
                sample.audio_path(input_key),
                settings,
            )["text"]
        return {
            "text": text,
            "model": settings.get("model", "kimi-audio-7b-instruct"),
            "version": settings.get("model_version", settings.get("version", "unknown")),
        }


@register_operator
class KimiBatchASROperator(_KimiAudioUpdates, BatchOperator):
    """Persistent-model, batched Kimi-Audio inference with failure isolation."""

    name = "kimi_batch"
    version = "2.0.0"
    category = "asr"

    def _execute(self, sample: Sample, config: OperatorConfig) -> dict[str, Any]:
        settings = _resolve_settings(config)
        input_key = config.params.get("input_audio_key", "raw")
        result = _transcribe_many(
            _load_kimi_audio_model(settings),
            [sample.audio_path(input_key)],
            settings,
        )[0]
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
            cached = None if config.force else self.load_cache(cache_key, config)
            if cached is not None:
                results[index] = OperatorResult(
                    sample=self.apply_cached(sample, cached),
                    cache_hit=True,
                )
            else:
                pending.append((index, sample, cache_key))

        if pending and (config.mock or config.params.get("mock")):
            for index, sample, cache_key in pending:
                result = {"text": f"[mock:kimi:{sample.id}]"}
                results[index] = self._finalize(sample, result, cache_key, config, settings)
        elif pending:
            try:
                model = _load_kimi_audio_model(settings)
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "Kimi-Audio model initialization failed; all {} pending samples will fail: {}",
                    len(pending),
                    exc,
                )
                for index, sample, _ in pending:
                    results[index] = self._failed(sample, exc)
            else:
                size = max(1, int(settings.get("batch_size", 1)))
                for start in range(0, len(pending), size):
                    chunk = pending[start : start + size]
                    self._process_chunk(chunk, model, config, settings, results)
        else:
            logger.warning(
                "Kimi-Audio inference was not called: all {} samples were already "
                "completed or cache hits. Use --force to re-run inference.",
                len(samples),
            )

        if any(result is None for result in results):
            raise RuntimeError("Kimi-Audio batch operator produced an incomplete result set")
        return [result for result in results if result is not None]

    def _process_chunk(
        self,
        chunk: list[tuple[int, Sample, str]],
        model: Any,
        config: OperatorConfig,
        settings: dict[str, Any],
        results: list[OperatorResult | None],
    ) -> None:
        input_key = config.params.get("input_audio_key", "raw")
        audio_paths = [sample.audio_path(input_key) for _, sample, _ in chunk]
        try:
            transcripts = _transcribe_many(model, audio_paths, settings)
        except Exception:  # noqa: BLE001
            logger.warning(
                "Kimi-Audio batch inference failed for {} samples, falling back to per-sample retry",
                len(chunk),
            )
            for index, sample, cache_key in chunk:
                try:
                    transcript = _transcribe_many(
                        model, [sample.audio_path(input_key)], settings
                    )[0]
                    results[index] = self._finalize(
                        sample, transcript, cache_key, config, settings
                    )
                except Exception as exc:  # noqa: BLE001
                    results[index] = self._failed(sample, exc)
            return
        for (index, sample, cache_key), transcript in zip(chunk, transcripts):
            results[index] = self._finalize(sample, transcript, cache_key, config, settings)

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
