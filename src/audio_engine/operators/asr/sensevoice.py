from __future__ import annotations

import json
import os
import re
import threading
import time
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
_TAG_RE = re.compile(r"<\|([^|<>]+)\|>")
_LANGUAGES = {"zh", "en", "yue", "ja", "ko", "nospeech"}
_EMOTIONS = {"HAPPY", "SAD", "ANGRY", "NEUTRAL"}
_EVENTS = {"Speech", "BGM", "Applause", "Laughter", "Cry", "Sneeze", "Breath", "Cough"}

# ── 诊断采样配置 ─────────────────────────────────────────────────────────────
# 每处理 _DIAG_INTERVAL 条样本打印一次完整诊断快照（模型输出 + 写入结果）。
# 40 万条数据约产生 4000 行诊断日志，不会撑爆日志文件。
_DIAG_INTERVAL: int = 100
_diag_lock = threading.Lock()
_diag_counter: int = 0  # 跨 chunk 的全局已处理样本计数（含 cache hit 和 skip）


def _load_asr_config(config_path: str | None) -> dict[str, Any]:
    if not config_path:
        return {}
    path = Path(config_path)
    if path.exists():
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {}


def _resolve_settings(config: OperatorConfig) -> dict[str, Any]:
    settings = _load_asr_config(
        config.params.get("config_path", "configs/asr/sensevoice.yaml")
    )
    settings.update({key: value for key, value in config.params.items() if key != "config_path"})
    settings["model_path"] = (
        os.environ.get("SENSEVOICE_MODEL_PATH")
        or config.params.get("model_path")
        or settings.get("model_path")
        or settings.get("model", "iic/SenseVoiceSmall")
    )
    return settings


def _cache_config(config: OperatorConfig, settings: dict[str, Any]) -> OperatorConfig:
    params = dict(config.params)
    for key in ("model", "model_version", "version", "model_path", "device", "language"):
        if key in settings:
            params[f"resolved_{key}"] = settings[key]
    return config.model_copy(update={"params": params})


def _load_sensevoice_model(settings: dict[str, Any]) -> Any:
    model_path = str(settings["model_path"])
    path = Path(model_path).expanduser()
    if path.is_absolute() and not path.is_dir():
        raise FileNotFoundError(
            f"SenseVoice 本地模型目录不存在: {path}. "
            "请检查 SENSEVOICE_MODEL_PATH 或 configs/asr/sensevoice.yaml"
        )
    model_kwargs = {
        "model": model_path,
        "device": settings.get("device", "cuda:0"),
        "disable_update": bool(settings.get("disable_update", True)),
    }
    for key in ("model_revision", "vad_model", "punc_model"):
        if settings.get(key) is not None:
            model_kwargs[key] = settings[key]
    cache_key = json.dumps(model_kwargs, sort_keys=True, ensure_ascii=False)
    with _MODEL_LOCK:
        if cache_key in _MODEL_CACHE:
            logger.debug(
                "[DIAG][MODEL] cache hit: model_path={} device={}",
                model_path, model_kwargs["device"],
            )
            return _MODEL_CACHE[cache_key]
        try:
            from funasr import AutoModel  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "SenseVoice 推理依赖未安装，请执行：pip install 'audio-data-engine[sensevoice]'"
            ) from exc
        logger.info(
            "[DIAG][MODEL] loading SenseVoice: path={} device={} disable_update={}",
            model_path, model_kwargs["device"], model_kwargs["disable_update"],
        )
        t0 = time.perf_counter()
        model = AutoModel(**model_kwargs)
        elapsed = time.perf_counter() - t0
        logger.info(
            "[DIAG][MODEL] SenseVoice loaded OK in {:.2f}s | type={} | path={} | device={}",
            elapsed, type(model).__name__, model_path, model_kwargs["device"],
        )
        _MODEL_CACHE[cache_key] = model
        return model


def parse_sensevoice_text(raw_text: str) -> dict[str, Any]:
    """Split SenseVoice control tags from comparable transcript text."""
    tags = _TAG_RE.findall(raw_text)
    language = next((tag for tag in tags if tag.lower() in _LANGUAGES), None)
    emotion = next((tag for tag in tags if tag.upper() in _EMOTIONS), None)
    events = [tag for tag in tags if tag in _EVENTS]
    known = {tag for tag in tags if tag.lower() in _LANGUAGES}
    known.update(tag for tag in tags if tag.upper() in _EMOTIONS)
    known.update(events)
    unknown = [tag for tag in tags if tag not in known]
    text = _TAG_RE.sub(lambda match: "" if match.group(1) in known else match.group(0), raw_text)
    extra: dict[str, Any] = {
        "raw_text": raw_text,
        "language": language.lower() if language else None,
        "emotion": emotion.upper() if emotion else None,
        "events": events,
    }
    if unknown:
        extra["unknown_tags"] = unknown
    return {"text": text.strip(), "extra": extra}


def _transcribe_many(
    model: Any, audio_paths: list[str], settings: dict[str, Any],
    *, _diag: bool = False,
) -> list[dict[str, Any]]:
    kwargs: dict[str, Any] = {
        "input": audio_paths,
        "language": settings.get("language", "auto"),
        "use_itn": bool(settings.get("use_itn", True)),
        "batch_size": int(settings.get("batch_size", 32)),
    }
    logger.debug(
        "[DIAG][INFER] generate() called: n_inputs={} language={} use_itn={} batch_size={}",
        len(audio_paths), kwargs["language"], kwargs["use_itn"], kwargs["batch_size"],
    )
    raw_results = model.generate(**kwargs)

    # ── 原始返回结构诊断（每次触发 _diag 时打印，便于确认 FunASR 版本行为）──
    if _diag:
        preview = raw_results[:2] if isinstance(raw_results, list) else raw_results
        logger.info(
            "[DIAG][INFER] raw generate() output (first 2): type={} count={} preview={}",
            type(raw_results).__name__,
            len(raw_results) if isinstance(raw_results, list) else "N/A",
            repr(preview),
        )

    if not isinstance(raw_results, list) or len(raw_results) != len(audio_paths):
        count = len(raw_results) if isinstance(raw_results, list) else 1
        raise RuntimeError(f"SenseVoice 返回 {count} 条结果，输入为 {len(audio_paths)} 条")

    results = []
    for i, raw in enumerate(raw_results):
        raw_text = str(raw.get("text", "")) if isinstance(raw, dict) else str(raw)
        parsed = parse_sensevoice_text(raw_text)
        if isinstance(raw, dict) and raw.get("confidence") is not None:
            parsed["confidence"] = raw["confidence"]
        # 空文本告警：模型有返回但 text 为空（VAD 未命中 或 raw 字段名不对）
        if not parsed["text"].strip():
            logger.warning(
                "[DIAG][INFER] empty text at index {}: raw_type={} raw_keys={} raw_text={!r}",
                i,
                type(raw).__name__,
                list(raw.keys()) if isinstance(raw, dict) else "N/A",
                raw_text[:200],
            )
        results.append(parsed)
    return results


class _SenseVoiceUpdates:
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
            "model": settings.get("model", "sensevoice-small"),
            "version": settings.get("model_version", settings.get("version", "unknown")),
            "confidence": result.get("confidence"),
            "extra": result["extra"],
        }
        input_key = config.params.get("input_audio_key", "raw")
        return {
            "transcripts": {"sensevoice": transcript},
            "lineage_entry": {
                "operator": self.full_name,
                "version": self.version,
                "params": dict(config.params),
                "input_key": input_key,
            },
        }


@register_operator
class SenseVoiceOperator(_SenseVoiceUpdates, BaseASROperator):
    name = "sensevoice"
    version = "2.0.0"

    def compute_cache_key(self, sample: Sample, config: OperatorConfig) -> str:
        settings = _resolve_settings(config)
        return super().compute_cache_key(sample, _cache_config(config, settings))

    def transcribe(self, sample: Sample, config: OperatorConfig) -> dict[str, Any]:
        settings = _resolve_settings(config)
        if config.mock or config.params.get("mock"):
            parsed = parse_sensevoice_text(f"[mock:sensevoice:{sample.id}]")
        else:
            input_key = config.params.get("input_audio_key", "raw")
            parsed = _transcribe_many(
                _load_sensevoice_model(settings), [sample.audio_path(input_key)], settings
            )[0]
        return {
            **parsed,
            "model": settings.get("model", "sensevoice-small"),
            "version": settings.get("model_version", settings.get("version", "unknown")),
            "confidence": parsed.get("confidence"),
        }


@register_operator
class SenseVoiceBatchASROperator(_SenseVoiceUpdates, BatchOperator):
    """Persistent-model, batched SenseVoice inference with failure isolation."""

    name = "sensevoice_batch"
    version = "1.0.0"
    category = "asr"

    def _execute(self, sample: Sample, config: OperatorConfig) -> dict[str, Any]:
        settings = _resolve_settings(config)
        input_key = config.params.get("input_audio_key", "raw")
        result = _transcribe_many(
            _load_sensevoice_model(settings), [sample.audio_path(input_key)], settings
        )[0]
        return self._updates(result, config, settings)

    def compute_cache_key(self, sample: Sample, config: OperatorConfig) -> str:
        settings = _resolve_settings(config)
        return super().compute_cache_key(sample, _cache_config(config, settings))

    def process_batch(self, samples: list[Sample], config: OperatorConfig) -> list[OperatorResult]:
        global _diag_counter  # noqa: PLW0603
        settings = _resolve_settings(config)
        cache_config = _cache_config(config, settings)
        results: list[OperatorResult | None] = [None] * len(samples)
        pending: list[tuple[int, Sample, str]] = []
        skipped_count = 0
        cache_hit_count = 0

        for index, sample in enumerate(samples):
            if self.should_skip(sample, config):
                results[index] = OperatorResult(sample=sample, skipped=True)
                skipped_count += 1
                continue
            cache_key = super().compute_cache_key(sample, cache_config)
            cached = None if config.force else self.load_cache(cache_key, config)
            if cached is not None:
                results[index] = OperatorResult(sample=self.apply_cached(sample, cached), cache_hit=True)
                cache_hit_count += 1
            else:
                pending.append((index, sample, cache_key))

        logger.info(
            "[DIAG][BATCH] process_batch: total={} pending={} skipped={} cache_hit={}",
            len(samples), len(pending), skipped_count, cache_hit_count,
        )

        if pending and (config.mock or config.params.get("mock")):
            for index, sample, cache_key in pending:
                parsed = parse_sensevoice_text(f"[mock:sensevoice:{sample.id}]")
                results[index] = self._finalize(sample, parsed, cache_key, config, settings)
        elif pending:
            logger.info(
                "[DIAG][BATCH] SenseVoice inference starting: pending={} model={} device={} batch_size={}",
                len(pending), settings["model_path"], settings.get("device", "cuda:0"),
                settings.get("batch_size", 32),
            )
            try:
                model = _load_sensevoice_model(settings)
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "[DIAG][BATCH] SenseVoice model initialization failed; all {} pending samples will fail: {}",
                    len(pending), exc,
                )
                for index, sample, _ in pending:
                    results[index] = self._failed(sample, exc)
            else:
                size = max(1, int(settings.get("batch_size", 32)))
                for start in range(0, len(pending), size):
                    chunk = pending[start : start + size]
                    # 判断本 chunk 是否需要触发诊断采样
                    with _diag_lock:
                        _diag_counter += len(chunk)
                        do_diag = (_diag_counter % _DIAG_INTERVAL) < len(chunk)
                    self._process_chunk(chunk, model, config, settings, results, diag=do_diag)
        else:
            logger.warning(
                "[DIAG][BATCH] SenseVoice inference was not called: all {} samples were already "
                "completed or cache hits. Use --force to re-run inference.",
                len(samples),
            )

        if any(result is None for result in results):
            raise RuntimeError("SenseVoice batch operator produced an incomplete result set")
        return [result for result in results if result is not None]

    def _process_chunk(self, chunk, model, config, settings, results, *, diag: bool = False) -> None:
        input_key = config.params.get("input_audio_key", "raw")
        audio_paths = [sample.audio_path(input_key) for _, sample, _ in chunk]
        try:
            transcripts = _transcribe_many(model, audio_paths, settings, _diag=diag)
        except Exception:  # noqa: BLE001
            logger.warning(
                "[DIAG][CHUNK] batch inference failed for {} samples, falling back to per-sample retry",
                len(chunk),
            )
            for index, sample, cache_key in chunk:
                try:
                    transcript = _transcribe_many(
                        model, [sample.audio_path(input_key)], settings, _diag=diag
                    )[0]
                    results[index] = self._finalize(
                        sample, transcript, cache_key, config, settings, diag=diag
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        "[DIAG][CHUNK] per-sample retry failed: sample={} error={}",
                        sample.id, exc,
                    )
                    results[index] = self._failed(sample, exc)
            return
        for (index, sample, cache_key), transcript in zip(chunk, transcripts):
            results[index] = self._finalize(sample, transcript, cache_key, config, settings, diag=diag)

    def _finalize(self, sample, result, cache_key, config, settings, *, diag: bool = False) -> OperatorResult:
        updates = self._updates(result, config, settings)
        updated = self._apply_updates(sample, updates)
        entry = updates["lineage_entry"]
        updated.add_lineage(
            operator=entry["operator"], version=entry["version"], params=entry["params"],
            input_key=entry["input_key"], cache_key=cache_key,
        )
        updated.mark_completed(self.full_name)
        self.save_cache(cache_key, config, updates)
        # ── 诊断采样：打印写入确认 ──────────────────────────────────────────
        if diag:
            sv = updated.transcripts.get("sensevoice", {})
            logger.info(
                "[DIAG][WRITE] sample={} | text={!r} | model={} | version={} | "
                "language={} | emotion={} | events={} | raw_text={!r}",
                updated.id,
                sv.get("text", "<MISSING>"),
                sv.get("model", "<MISSING>"),
                sv.get("version", "<MISSING>"),
                sv.get("extra", {}).get("language"),
                sv.get("extra", {}).get("emotion"),
                sv.get("extra", {}).get("events"),
                sv.get("extra", {}).get("raw_text", "")[:200],
            )
        return OperatorResult(sample=updated, message="processed")

    def _failed(self, sample: Sample, exc: Exception) -> OperatorResult:
        failed = sample.model_copy(deep=True)
        failed.mark_failed(self.full_name, str(exc))
        return OperatorResult(sample=failed, message=str(exc))
