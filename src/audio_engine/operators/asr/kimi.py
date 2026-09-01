from __future__ import annotations

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import yaml

from audio_engine.core.operator import BatchOperator, OperatorConfig, OperatorResult
from audio_engine.core.registry import register_operator
from audio_engine.core.sample import Sample

_MODEL_CACHE: dict[str, Any] = {}
_MODEL_LOCK = threading.Lock()


def _load_asr_config(config_path: str | None) -> dict[str, Any]:
    if not config_path:
        return {}
    path = Path(config_path)
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _resolve_settings(config: OperatorConfig) -> dict[str, Any]:
    settings = _load_asr_config(config.params.get("config_path", "configs/asr/kimi_audio.yaml"))
    settings.update({key: value for key, value in config.params.items() if key != "config_path"})
    settings["model_path"] = (
        config.params.get("model_path")
        or os.environ.get("KIMI_AUDIO_MODEL_PATH")
        or settings.get("model_path")
        or settings.get("model", "moonshotai/Kimi-Audio-7B-Instruct")
    )
    return settings


def _cache_config(config: OperatorConfig, settings: dict[str, Any]) -> OperatorConfig:
    params = dict(config.params)
    for key in ("model", "model_version", "model_path", "dtype", "device"):
        if key in settings:
            params[f"resolved_{key}"] = settings[key]
    return config.model_copy(update={"params": params})


def _load_kimi_model(settings: dict[str, Any]) -> Any:
    """Load the official Kimi-Audio inference wrapper once in each shard process."""
    kwargs = dict(settings.get("model_kwargs") or {})
    if settings.get("device") is not None:
        kwargs.setdefault("device", settings["device"])
    cache_key = json.dumps(
        {"model_path": settings["model_path"], "model_kwargs": kwargs},
        sort_keys=True,
        ensure_ascii=False,
    )
    with _MODEL_LOCK:
        if cache_key in _MODEL_CACHE:
            return _MODEL_CACHE[cache_key]
        try:
            from kimia_infer.api.kimia import KimiAudio
        except ImportError as exc:
            raise RuntimeError(
                "Kimi-Audio 推理依赖未安装；请按官方 Kimi-Audio 仓库安装 kimia_infer"
            ) from exc
        model = KimiAudio(model_path=settings["model_path"], **kwargs)
        _MODEL_CACHE[cache_key] = model
        return model


def release_cached_models() -> int:
    """Release this process' Kimi models between memory-heavy ASR stages."""
    with _MODEL_LOCK:
        count = len(_MODEL_CACHE)
        _MODEL_CACHE.clear()
    return count


def _normalise_result(raw: Any) -> dict[str, Any]:
    if isinstance(raw, str):
        return {"text": raw, "language": None}
    if isinstance(raw, dict):
        # Kimi wrappers and inference services commonly use one of these fields.
        text = raw.get("text", raw.get("transcription", raw.get("output_text", "")))
        return {"text": str(text), "language": raw.get("language")}
    return {
        "text": str(getattr(raw, "text", getattr(raw, "transcription", ""))),
        "language": getattr(raw, "language", None),
    }


def _official_generate(model: Any, path: str, settings: dict[str, Any]) -> dict[str, Any]:
    prompt_text = str(settings.get("prompt", "Please transcribe the audio accurately."))
    prompt = model.build_prompt(path, prompt_text) if hasattr(model, "build_prompt") else path
    kwargs = dict(settings.get("generation_kwargs") or {})
    raw = model.generate(prompt, **kwargs)
    # Official generate variants may return (text, audio) or a list containing text.
    if isinstance(raw, tuple):
        raw = raw[0]
    if isinstance(raw, list) and len(raw) == 1:
        raw = raw[0]
    return _normalise_result(raw)


def _transcribe_many(
    model: Any, audio_paths: list[str], settings: dict[str, Any]
) -> list[dict[str, Any]]:
    """Use a native batch API when present, with a compatible official-API fallback."""
    kwargs = dict(settings.get("transcribe_kwargs") or {})
    if hasattr(model, "transcribe_batch"):
        raw_results = model.transcribe_batch(audio_paths, **kwargs)
    elif hasattr(model, "transcribe"):
        try:
            raw_results = model.transcribe(audio=audio_paths, **kwargs)
        except TypeError:
            raw_results = model.transcribe(audio_paths, **kwargs)
    else:

        def generate(path: str) -> dict[str, Any]:
            return _official_generate(model, path, settings)

        threads = max(1, int(settings.get("inference_threads", 1)))
        if threads > 1 and len(audio_paths) > 1:
            with ThreadPoolExecutor(max_workers=min(threads, len(audio_paths))) as pool:
                return list(pool.map(generate, audio_paths))
        return [generate(path) for path in audio_paths]

    if isinstance(raw_results, dict) and "results" in raw_results:
        raw_results = raw_results["results"]
    raw_results = list(raw_results)
    if len(raw_results) != len(audio_paths):
        raise RuntimeError(
            f"Kimi-Audio 返回 {len(raw_results)} 条结果，输入为 {len(audio_paths)} 条"
        )
    return [_normalise_result(raw) for raw in raw_results]


@register_operator
class KimiBatchASROperator(BatchOperator):
    """Cached, fault-isolated Kimi-Audio batch inference for large manifests."""

    name = "kimi_batch"
    version = "1.0.0"
    category = "asr"

    def compute_cache_key(self, sample: Sample, config: OperatorConfig) -> str:
        settings = _resolve_settings(config)
        return super().compute_cache_key(sample, _cache_config(config, settings))

    def _execute(self, sample: Sample, config: OperatorConfig) -> dict[str, Any]:
        settings = _resolve_settings(config)
        result = _transcribe_many(
            _load_kimi_model(settings),
            [sample.audio_path(config.params.get("input_audio_key", "raw"))],
            settings,
        )[0]
        return self._updates(result, config, settings)

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
                        sample=self.apply_cached(sample, cached), cache_hit=True
                    )
                    continue
            pending.append((index, sample, cache_key))

        if pending and (config.mock or config.params.get("mock")):
            for index, sample, cache_key in pending:
                results[index] = self._finalize(
                    sample,
                    {"text": f"[mock:kimi:{sample.id}]", "language": settings.get("language")},
                    cache_key,
                    config,
                    settings,
                )
        elif pending:
            try:
                model = _load_kimi_model(settings)
            except Exception as exc:  # noqa: BLE001
                for index, sample, _ in pending:
                    results[index] = self._failed(sample, exc)
            else:
                batch_size = max(1, int(settings.get("batch_size", 8)))
                for start in range(0, len(pending), batch_size):
                    self._process_chunk(
                        pending[start : start + batch_size], model, config, settings, results
                    )

        if any(result is None for result in results):
            raise RuntimeError("Kimi batch operator produced an incomplete result set")
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
        try:
            transcripts = _transcribe_many(
                model, [sample.audio_path(input_key) for _, sample, _ in chunk], settings
            )
        except Exception:  # A bad file must not poison its whole GPU batch.
            for index, sample, cache_key in chunk:
                try:
                    transcript = _transcribe_many(model, [sample.audio_path(input_key)], settings)[
                        0
                    ]
                    results[index] = self._finalize(sample, transcript, cache_key, config, settings)
                except Exception as exc:  # noqa: BLE001
                    results[index] = self._failed(sample, exc)
            return
        for (index, sample, cache_key), transcript in zip(chunk, transcripts):
            results[index] = self._finalize(sample, transcript, cache_key, config, settings)

    def _updates(
        self, result: dict[str, Any], config: OperatorConfig, settings: dict[str, Any]
    ) -> dict[str, Any]:
        extra = {"language": result.get("language")} if result.get("language") else {}
        return {
            "transcripts": {
                "kimi": {
                    "text": result["text"],
                    "model": settings.get("model", "kimi-audio"),
                    "version": settings.get("model_version", settings.get("version", "unknown")),
                    "extra": extra,
                }
            },
            "lineage_entry": {
                "operator": self.full_name,
                "version": self.version,
                "params": dict(config.params),
                "input_key": config.params.get("input_audio_key", "raw"),
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
