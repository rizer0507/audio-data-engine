from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from audio_engine.core.sample import Sample


class OperatorConfig(BaseModel):
    """Runtime configuration passed to an operator."""

    params: dict[str, Any] = Field(default_factory=dict)
    output_dir: Path = Path("data/derived")
    cache_dir: Path = Path("data/cache")
    force: bool = False
    mock: bool = False

    model_config = {"arbitrary_types_allowed": True}


class OperatorResult(BaseModel):
    """Result returned by operator.process()."""

    sample: Sample
    cache_hit: bool = False
    skipped: bool = False
    message: str = ""

    model_config = {"arbitrary_types_allowed": True}


class ManifestOperator(ABC):
    """Operator that transforms the whole sample collection at once.

    Used for steps that cannot be expressed per-sample, e.g. ingest
    (creates samples from an external directory). Registered in the same
    OperatorRegistry and dispatched by PipelineRunner like any other step.
    """

    name: str = "base"
    version: str = "1.0.0"
    category: str = "manifest"

    @property
    def full_name(self) -> str:
        return f"{self.category}.{self.name}"

    @abstractmethod
    def run(self, samples: list[Sample], config: OperatorConfig) -> list[Sample]:
        """Take the current sample list and return the updated list."""


class BaseOperator(ABC):
    """Unified interface for all audio processing capabilities."""

    name: str = "base"
    version: str = "1.0.0"
    category: str = "base"

    @property
    def full_name(self) -> str:
        return f"{self.category}.{self.name}"

    def compute_cache_key(self, sample: Sample, config: OperatorConfig) -> str:
        """Cache key = hash(input_sha256 + operator + version + params + model)."""
        model_version = str(config.params.get("model", "")) + str(
            config.params.get("model_version", "")
        )
        payload = {
            "sha256": sample.sha256,
            "operator": self.full_name,
            "version": self.version,
            "params": config.params,
            "model_version": model_version,
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode()).hexdigest()

    def _cache_path(self, cache_key: str, config: OperatorConfig) -> Path:
        return config.cache_dir / self.full_name.replace(".", "_") / f"{cache_key}.json"

    def load_cache(self, cache_key: str, config: OperatorConfig) -> dict[str, Any] | None:
        path = self._cache_path(cache_key, config)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def save_cache(self, cache_key: str, config: OperatorConfig, data: dict[str, Any]) -> None:
        path = self._cache_path(cache_key, config)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def should_skip(self, sample: Sample, config: OperatorConfig) -> bool:
        if config.force:
            return False
        return sample.is_completed(self.full_name)

    _SCALAR_FIELDS = ("sample_rate", "channels", "duration", "source_path", "sha256")

    def _apply_updates(self, sample: Sample, updates: dict[str, Any]) -> Sample:
        updated = sample.model_copy(deep=True)
        for key in ("audio", "transcripts", "quality", "labels"):
            if key in updates:
                getattr(updated, key).update(updates[key])
        for field in self._SCALAR_FIELDS:
            if field in updates:
                setattr(updated, field, updates[field])
        return updated

    def apply_cached(self, sample: Sample, cached: dict[str, Any]) -> Sample:
        updated = self._apply_updates(sample, cached)
        if "lineage_entry" in cached:
            entry = cached["lineage_entry"]
            updated.add_lineage(
                operator=entry["operator"],
                version=entry["version"],
                params=entry.get("params", {}),
                input_key=entry.get("input_key"),
                output_key=entry.get("output_key"),
                output_path=entry.get("output_path"),
                cache_key=entry.get("cache_key"),
            )
        updated.mark_completed(self.full_name)
        return updated

    @abstractmethod
    def _execute(self, sample: Sample, config: OperatorConfig) -> dict[str, Any]:
        """Run operator logic and return update dict for cache/sample merge."""

    def process(self, sample: Sample, config: OperatorConfig) -> OperatorResult:
        if self.should_skip(sample, config):
            return OperatorResult(sample=sample, skipped=True, message="already completed")

        cache_key = self.compute_cache_key(sample, config)
        if not config.force:
            cached = self.load_cache(cache_key, config)
            if cached is not None:
                updated = self.apply_cached(sample, cached)
                return OperatorResult(
                    sample=updated,
                    cache_hit=True,
                    message="cache hit",
                )

        try:
            updates = self._execute(sample, config)
        except Exception as exc:
            failed = sample.model_copy(deep=True)
            failed.mark_failed(self.full_name, str(exc))
            raise

        updated = self._apply_updates(sample, updates)

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

        updated.mark_completed(self.full_name)
        cache_data = {
            k: updates[k]
            for k in (
                "audio",
                "transcripts",
                "quality",
                "labels",
                "lineage_entry",
                *self._SCALAR_FIELDS,
            )
            if k in updates
        }
        self.save_cache(cache_key, config, cache_data)

        return OperatorResult(sample=updated, message="processed")
