from __future__ import annotations

import hashlib
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml

from audio_engine.core.operator import BatchOperator, OperatorConfig, OperatorResult
from audio_engine.core.registry import register_operator
from audio_engine.core.sample import Sample
from audio_engine.operators.asr.base import BaseASROperator
from audio_engine.operators.asr.vllm import call_vllm_transcription

_API_BASE_SPLIT = re.compile(r"[\s,;]+")
_SHARD_DIR_RE = re.compile(r"^shard-(\d+)$")


def _load_asr_config(config_path: str | None) -> dict[str, Any]:
    if not config_path:
        return {}
    path = Path(config_path)
    if path.exists():
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {}


def parse_kimi_api_bases(raw: Any) -> list[str]:
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
    settings = _load_asr_config(config.params.get("config_path", "configs/asr/kimi.yaml"))
    settings.update({key: value for key, value in config.params.items() if key != "config_path"})
    bases = parse_kimi_api_bases(
        _first_defined(
            config.params.get("api_bases"),
            config.params.get("api_base"),
            os.environ.get("KIMI_ASR_API_BASES"),
            os.environ.get("KIMI_ASR_API_BASE"),
            settings.get("api_bases"),
            settings.get("api_base"),
            "http://127.0.0.1:5554",
        )
    )
    if not bases:
        bases = ["http://127.0.0.1:5554"]
    settings["api_bases"] = bases
    settings["api_base"] = bases[0]
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


def _cache_config(config: OperatorConfig, settings: dict[str, Any], sample: Sample) -> OperatorConfig:
    params = dict(config.params)
    # Keep all output-affecting server/request settings in the fingerprint while
    # excluding throughput-only knobs (concurrency, timeout and batch_size).
    # Pad bucket/mode must be present so unpadded ASR cache cannot be reused.
    fingerprint_keys = (
        "model",
        "model_version",
        "version",
        "language",
        "prompt",
        "temperature",
        "response_format",
    )
    params["resolved_kimi_vllm_settings"] = {
        key: settings[key] for key in fingerprint_keys if key in settings
    }
    params["resolved_kimi_api_bases"] = list(settings.get("api_bases") or [])
    params["kimi_pad_target_s"] = sample.labels.get("kimi_pad_target_s")
    params["kimi_pad_mode"] = sample.labels.get("kimi_pad_mode")
    params["input_audio_key"] = config.params.get("input_audio_key", "raw")
    return config.model_copy(update={"params": params})


def select_kimi_api_base(settings: dict[str, Any], sample_id: str) -> str:
    """Sticky-assign a replica by sample id so retries stay on the same endpoint."""
    bases = list(settings.get("api_bases") or [settings["api_base"]])
    if len(bases) == 1:
        return bases[0]
    digest = hashlib.sha256(str(sample_id).encode("utf-8")).digest()
    return bases[int.from_bytes(digest[:4], "big") % len(bases)]


def pad_bucket_key_from_values(mode: Any, target: Any) -> tuple[str, int | None]:
    """Normalize a pad-bucket key used to keep one vLLM encoder step on a single T."""
    mode_s = str(mode or "unknown")
    if target is not None and target != "":
        target = int(target)
    else:
        target = None
    return (mode_s, target)


def pad_bucket_key(sample: Sample) -> tuple[str, int | None]:
    """Group key so one vLLM step never mixes different padded durations."""
    return pad_bucket_key_from_values(
        sample.labels.get("kimi_pad_mode"),
        sample.labels.get("kimi_pad_target_s"),
    )


def group_by_pad_bucket(
    pending: list[tuple[int, Sample, str]],
) -> list[tuple[tuple[str, int | None], list[tuple[int, Sample, str]]]]:
    """Preserve first-seen bucket order; keep each bucket contiguous."""
    grouped: dict[tuple[str, int | None], list[tuple[int, Sample, str]]] = {}
    order: list[tuple[str, int | None]] = []
    for item in pending:
        key = pad_bucket_key(item[1])
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(item)
    return [(key, grouped[key]) for key in order]


def iter_pad_bucket_windows(
    keys: list[tuple[str, int | None]],
    *,
    batch_size: int,
) -> list[list[int]]:
    """Split indices into HTTP windows: one pad bucket per window, ``over_30s`` size 1."""
    grouped: dict[tuple[str, int | None], list[int]] = {}
    order: list[tuple[str, int | None]] = []
    for index, key in enumerate(keys):
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(index)

    windows: list[list[int]] = []
    default_size = max(1, int(batch_size))
    for key in order:
        size = 1 if key[0] == "over_30s" else default_size
        group = grouped[key]
        for start in range(0, len(group), size):
            windows.append(group[start : start + size])
    return windows


def bind_kimi_api_bases_to_shard(
    settings: dict[str, Any],
    config: OperatorConfig,
) -> dict[str, Any]:
    """Pin one shard process to one replica so two shards cannot mix T on one GPU."""
    bases = list(settings.get("api_bases") or [])
    if len(bases) <= 1 or config.run_dir is None:
        return settings
    match = _SHARD_DIR_RE.match(Path(config.run_dir).name)
    if not match:
        return settings
    chosen = bases[int(match.group(1)) % len(bases)]
    bound = dict(settings)
    bound["api_bases"] = [chosen]
    bound["api_base"] = chosen
    bound["kimi_shard_bind_index"] = int(match.group(1))
    return bound


def _call_vllm_transcription(audio_path: str, settings: dict[str, Any]) -> dict[str, Any]:
    return call_vllm_transcription(audio_path, settings)


def _transcribe_one(audio_path: str, settings: dict[str, Any], api_base: str) -> dict[str, Any]:
    per_request = dict(settings)
    per_request["api_base"] = api_base
    return _call_vllm_transcription(audio_path, per_request)


def _transcribe_many(
    audio_paths: list[str],
    settings: dict[str, Any],
    *,
    sample_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    if not audio_paths:
        return []

    ids = sample_ids if sample_ids is not None else [str(index) for index in range(len(audio_paths))]
    if len(ids) != len(audio_paths):
        raise ValueError("sample_ids must match audio_paths")
    bases = [select_kimi_api_base(settings, sample_id) for sample_id in ids]

    concurrency = min(int(settings["concurrency"]), len(audio_paths))
    if concurrency <= 1:
        return [
            _transcribe_one(path, settings, base)
            for path, base in zip(audio_paths, bases)
        ]

    results: list[dict[str, Any] | None] = [None] * len(audio_paths)
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        future_map = {
            pool.submit(_transcribe_one, path, settings, base): index
            for index, (path, base) in enumerate(zip(audio_paths, bases))
        }
        for future in as_completed(future_map):
            index = future_map[future]
            results[index] = future.result()
    return [item for item in results if item is not None]


@register_operator
class KimiASROperator(BaseASROperator):
    name = "kimi"
    version = "4.0.0"

    def compute_cache_key(self, sample: Sample, config: OperatorConfig) -> str:
        settings = _resolve_settings(config)
        return super().compute_cache_key(sample, _cache_config(config, settings, sample))

    def transcribe(self, sample: Sample, config: OperatorConfig) -> dict[str, Any]:
        settings = _resolve_settings(config)
        if config.mock or config.params.get("mock"):
            text = self._mock_transcript(sample, config)
            language = settings.get("language")
        else:
            input_key = config.params.get("input_audio_key", "raw")
            result = _transcribe_one(
                sample.audio_path(input_key),
                settings,
                select_kimi_api_base(settings, sample.id),
            )
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
    version = "4.0.0"
    category = "asr"

    def should_skip(self, sample: Sample, config: OperatorConfig) -> bool:
        transcript_key = config.params.get("transcript_key")
        if transcript_key:
            return not config.force and str(transcript_key) in sample.transcripts
        return super().should_skip(sample, config)

    def _execute(self, sample: Sample, config: OperatorConfig) -> dict[str, Any]:
        settings = _resolve_settings(config)
        input_key = config.params.get("input_audio_key", "raw")
        result = _transcribe_many(
            [sample.audio_path(input_key)],
            settings,
            sample_ids=[sample.id],
        )[0]
        return self._updates(result, config, settings)

    def compute_cache_key(self, sample: Sample, config: OperatorConfig) -> str:
        settings = _resolve_settings(config)
        return super().compute_cache_key(sample, _cache_config(config, settings, sample))

    def process_batch(self, samples: list[Sample], config: OperatorConfig) -> list[OperatorResult]:
        settings = _resolve_settings(config)
        route_settings = bind_kimi_api_bases_to_shard(settings, config)
        results: list[OperatorResult | None] = [None] * len(samples)
        pending: list[tuple[int, Sample, str]] = []

        for index, sample in enumerate(samples):
            if self.should_skip(sample, config):
                results[index] = OperatorResult(sample=sample, skipped=True)
                continue
            cache_config = _cache_config(config, settings, sample)
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
            inference_batch_size = max(
                1, int(route_settings.get("batch_size", route_settings["concurrency"]))
            )
            for key, group in group_by_pad_bucket(pending):
                chunk_settings = dict(route_settings)
                chunk_size = inference_batch_size
                if key[0] == "over_30s":
                    chunk_settings["concurrency"] = 1
                    chunk_size = 1
                for start in range(0, len(group), chunk_size):
                    chunk = group[start : start + chunk_size]
                    self._process_inference_chunk(
                        chunk, input_key, config, chunk_settings, results
                    )

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
        ready: list[tuple[int, Sample, str, str]] = []
        for index, sample, cache_key in chunk:
            try:
                path = sample.audio_path(input_key)
            except Exception as exc:  # noqa: BLE001 - missing pad artifact is a sample failure
                results[index] = self._failed(sample, exc)
                continue
            ready.append((index, sample, cache_key, path))
        if not ready:
            return

        try:
            transcripts = _transcribe_many(
                [path for *_rest, path in ready],
                settings,
                sample_ids=[sample.id for _index, sample, _cache_key, _path in ready],
            )
        except Exception:  # noqa: BLE001 - retry a failed batch one sample at a time
            for index, sample, cache_key, path in ready:
                try:
                    transcript = _transcribe_many(
                        [path], settings, sample_ids=[sample.id]
                    )[0]
                    results[index] = self._finalize(sample, transcript, cache_key, config, settings)
                except Exception as exc:  # noqa: BLE001 - isolate corrupt audio
                    results[index] = self._failed(sample, exc)
            return

        for (index, sample, cache_key, _path), transcript in zip(ready, transcripts):
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
        transcript_key = str(config.params.get("transcript_key", "kimi"))
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
