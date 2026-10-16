"""semantic.judge — template-driven multi-family ASR semantic judgment."""

from __future__ import annotations

import copy
import threading
from pathlib import Path
from typing import Any

from loguru import logger

from audio_engine.core.operator import BaseOperator, OperatorConfig, OperatorResult
from audio_engine.core.registry import register_operator
from audio_engine.core.sample import Sample
from audio_engine.core.semantic.client import ConcurrencyGate, SemanticChatClient
from audio_engine.core.semantic.concurrency import validate_concurrency_budget
from audio_engine.core.semantic.config import FrozenSemanticBundle, freeze_bundle
from audio_engine.core.semantic.engine import SemanticJudgeEngine
from audio_engine.core.semantic.fingerprint import input_context_digest, semantic_cache_key
from audio_engine.core.semantic.types import (
    ENGINE_VERSION,
    ERROR_STATUSES,
    STATUS_MOCK,
    STATUS_SKIPPED_INPUT,
    STATUS_SUCCESS,
    SUCCESS_LIKE,
)


def _task_config_path(params: dict[str, Any]) -> Path:
    raw = params.get("task_config") or params.get("task_config_path")
    if not raw:
        raise ValueError("semantic.judge requires params.task_config")
    return Path(str(raw))


def _merge_semantic_labels(sample: Sample, labels_update: dict[str, Any]) -> dict[str, Any]:
    """Merge only the current task node under labels.semantic_results."""
    merged = copy.deepcopy(dict(sample.labels or {}))
    for key, value in labels_update.items():
        if key == "semantic_results" and isinstance(value, dict):
            existing = merged.get("semantic_results")
            if not isinstance(existing, dict):
                existing = {}
            else:
                existing = copy.deepcopy(existing)
            existing.update(value)
            merged["semantic_results"] = existing
        else:
            merged[key] = value
    return merged


@register_operator
class SemanticJudgeOperator(BaseOperator):
    """One sample → one Chat request with multi-family evidence in the prompt."""

    name = "judge"
    version = ENGINE_VERSION
    category = "semantic"

    _bundle_cache: dict[str, FrozenSemanticBundle] = {}
    _bundle_lock = threading.Lock()
    _runtime: dict[str, Any] = {}
    _runtime_lock = threading.Lock()

    def dependency_fingerprint(self, config: OperatorConfig) -> dict[str, Any] | None:
        bundle = self._get_bundle(config)
        return {
            "engine_version": ENGINE_VERSION,
            "task_id": bundle.task.task_id,
            "task_version": bundle.task.task_version,
            "content_fingerprint": bundle.content_fingerprint,
            "model": bundle.service.model,
            "model_revision": bundle.service.model_revision,
            "cache_namespace": bundle.service.cache_namespace,
            "structured_output_mode": bundle.service.structured_output_mode,
            "mock": bool(config.mock or config.params.get("mock")),
        }

    def input_fingerprint(self, samples: list[Sample], config: OperatorConfig) -> str | None:
        bundle = self._get_bundle(config)
        return input_context_digest(samples, bundle)

    def should_retry_on_resume(self, sample: Sample, config: OperatorConfig) -> bool:
        if config.force:
            return True
        bundle = self._get_bundle(config)
        task_id = bundle.task.task_id
        if sample.status.get(self.full_name) == "failed":
            return True
        results = (sample.labels or {}).get("semantic_results")
        if not isinstance(results, dict):
            return True
        record = results.get(task_id)
        if not isinstance(record, dict):
            return True
        status = str(record.get("status") or "")
        if status in ERROR_STATUSES:
            return True
        # Mock must not satisfy a real resume.
        mock_run = bool(config.mock or config.params.get("mock"))
        if record.get("mock") and not mock_run:
            return True
        if status == STATUS_MOCK and not mock_run:
            return True
        return False

    def compute_cache_key(self, sample: Sample, config: OperatorConfig) -> str:
        bundle = self._get_bundle(config)
        mock = bool(config.mock or config.params.get("mock"))
        return semantic_cache_key(sample, bundle, mock=mock)

    def should_skip(self, sample: Sample, config: OperatorConfig) -> bool:
        if config.force:
            return False
        bundle = self._get_bundle(config)
        task_id = bundle.task.task_id
        results = (sample.labels or {}).get("semantic_results")
        if not isinstance(results, dict):
            return False
        record = results.get(task_id)
        if not isinstance(record, dict):
            return False
        status = str(record.get("status") or "")
        if status not in SUCCESS_LIKE:
            return False
        mock_run = bool(config.mock or config.params.get("mock"))
        if bool(record.get("mock")) != mock_run:
            return False
        # Require matching input fingerprint when present.
        expected = self.compute_cache_key(sample, config)
        return record.get("input_fingerprint") == expected

    def process(self, sample: Sample, config: OperatorConfig) -> OperatorResult:
        if self.should_skip(sample, config):
            return OperatorResult(sample=sample, skipped=True, message="semantic result reusable")

        cache_key = self.compute_cache_key(sample, config)
        mock = bool(config.mock or config.params.get("mock"))
        # Mock and live use different cache keys (mock flag in fingerprint); both may hit.
        if not config.force:
            cached = self.load_cache(cache_key, config)
            if cached is not None and self._cache_acceptable(cached, mock=mock):
                updated = self._apply_semantic_updates(sample, cached)
                if "lineage_entry" in cached:
                    entry = cached["lineage_entry"]
                    updated.add_lineage(
                        operator=entry["operator"],
                        version=entry["version"],
                        params=entry.get("params", {}),
                        input_key=entry.get("input_key"),
                        output_key=entry.get("output_key"),
                        output_path=entry.get("output_path"),
                        cache_key=entry.get("cache_key") or cache_key,
                    )
                results = updated.labels.get("semantic_results")
                if isinstance(results, dict):
                    task_id = cached.get("_semantic_task_id")
                    if task_id and isinstance(results.get(task_id), dict):
                        results[task_id]["cache_hit"] = True
                updated.mark_completed(self.full_name)
                return OperatorResult(sample=updated, cache_hit=True, message="cache hit")

        updates = self._execute(sample, config)
        updated = self._apply_semantic_updates(sample, updates)
        if "lineage_entry" in updates:
            entry = updates["lineage_entry"]
            updated.add_lineage(
                operator=entry["operator"],
                version=entry["version"],
                params=entry.get("params", {}),
                input_key=entry.get("input_key"),
                output_key=entry.get("output_key"),
                output_path=entry.get("output_path"),
                cache_key=cache_key,
            )

        status = str(updates.get("_semantic_status") or "")
        if status in {STATUS_SUCCESS, STATUS_SKIPPED_INPUT, STATUS_MOCK}:
            updated.mark_completed(self.full_name)
            # Only cache reusable successes (real success / skipped). Never cache mock
            # into the real namespace; mock has its own cache key via mock flag.
            if status in {STATUS_SUCCESS, STATUS_SKIPPED_INPUT}:
                cache_data = {
                    "labels": updates["labels"],
                    "lineage_entry": updates.get("lineage_entry"),
                    "_semantic_status": status,
                    "_semantic_task_id": updates.get("_semantic_task_id"),
                }
                self.save_cache(cache_key, config, cache_data)
            elif status == STATUS_MOCK:
                cache_data = {
                    "labels": updates["labels"],
                    "lineage_entry": updates.get("lineage_entry"),
                    "_semantic_status": status,
                    "_semantic_task_id": updates.get("_semantic_task_id"),
                }
                self.save_cache(cache_key, config, cache_data)
            return OperatorResult(sample=updated, message=f"semantic {status}")

        # Technical / input errors: keep the row, mark failed for resume retry.
        msg = ""
        record = updates.get("_semantic_record") or {}
        if isinstance(record, dict):
            msg = str(record.get("error_message") or record.get("error_code") or status)
        updated.mark_failed(self.full_name, msg or status or "semantic error")
        return OperatorResult(sample=updated, message=f"semantic {status}")

    def _cache_acceptable(self, cached: dict[str, Any], *, mock: bool) -> bool:
        status = str(cached.get("_semantic_status") or "")
        if status not in {STATUS_SUCCESS, STATUS_SKIPPED_INPUT, STATUS_MOCK}:
            return False
        if status == STATUS_MOCK and not mock:
            return False
        if status != STATUS_MOCK and mock:
            return False
        return True

    def _apply_semantic_updates(self, sample: Sample, updates: dict[str, Any]) -> Sample:
        updated = sample.model_copy(deep=True)
        if "labels" in updates and isinstance(updates["labels"], dict):
            updated.labels = _merge_semantic_labels(sample, updates["labels"])
        for key in ("audio", "transcripts", "quality"):
            if key in updates and isinstance(updates[key], dict):
                getattr(updated, key).update(updates[key])
        for field in self._SCALAR_FIELDS:
            if field in updates:
                setattr(updated, field, updates[field])
        return updated

    def _execute(self, sample: Sample, config: OperatorConfig) -> dict[str, Any]:
        engine = self._get_engine(config)
        return engine.judge(sample)

    def _get_bundle(self, config: OperatorConfig) -> FrozenSemanticBundle:
        path = str(_task_config_path(config.params).resolve())
        with self._bundle_lock:
            if path not in self._bundle_cache:
                self._bundle_cache[path] = freeze_bundle(path)
            return self._bundle_cache[path]

    def _runtime_key(self, config: OperatorConfig) -> str:
        bundle = self._get_bundle(config)
        mock = bool(config.mock or config.params.get("mock"))
        return f"{bundle.content_fingerprint}|mock={mock}|run={config.run_dir}"

    def _get_engine(self, config: OperatorConfig) -> SemanticJudgeEngine:
        key = self._runtime_key(config)
        with self._runtime_lock:
            runtime = self._runtime.get(key)
            if runtime is not None:
                return runtime["engine"]

            bundle = self._get_bundle(config)
            mock = bool(config.mock or config.params.get("mock"))
            workers = int(config.params.get("workers") or 0)
            # Per-process gate: shard workers already budgeted against total.
            limit = workers if workers > 0 else min(bundle.service.max_concurrency_total, 16)
            limit = max(1, min(limit, bundle.service.max_concurrency_total))
            gate = ConcurrencyGate(limit)
            client = None if mock else SemanticChatClient(bundle.service, gate=gate)
            engine = SemanticJudgeEngine(bundle, client=client, mock=mock)
            if not mock and bundle.service.probe_on_start:
                try:
                    info = engine.probe()
                    logger.info(
                        "semantic.judge probe ok task_id={} model={} info={}",
                        bundle.task.task_id,
                        bundle.service.model,
                        {k: info[k] for k in info if k != "models" or True},
                    )
                except Exception as exc:  # noqa: BLE001
                    # Fail fast on probe when not mock — configuration/service issue.
                    raise RuntimeError(f"semantic.judge probe failed: {exc}") from exc
            logger.info(
                "semantic.judge ready task_id={} workers_gate={} max_total={} mock={}",
                bundle.task.task_id,
                limit,
                bundle.service.max_concurrency_total,
                mock,
            )
            self._runtime[key] = {"engine": engine, "client": client, "gate": gate}
            return engine

    def preflight_concurrency(
        self,
        config: OperatorConfig,
        *,
        workers: int,
        parallel_shards: int = 1,
    ) -> dict[str, int]:
        bundle = self._get_bundle(config)
        return validate_concurrency_budget(
            max_concurrency_total=bundle.service.max_concurrency_total,
            workers_per_shard=workers,
            parallel_shards=parallel_shards,
            context=f"semantic.judge[{bundle.task.task_id}]",
        )
