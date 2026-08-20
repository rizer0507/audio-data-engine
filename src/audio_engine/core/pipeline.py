from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from audio_engine.core.manifest import Manifest
from audio_engine.core.operator import ManifestOperator, OperatorConfig
from audio_engine.core.registry import OperatorRegistry
from audio_engine.core.sample import Sample


@dataclass
class PipelineStep:
    name: str
    operator: str
    params: dict[str, Any] = field(default_factory=dict)
    input_audio_key: str = "raw"
    output_audio_key: str | None = None


@dataclass
class PipelineConfig:
    name: str
    input_manifest: str
    steps: list[PipelineStep]
    source_dir: str | None = None
    source_id: str | None = None
    output_manifest: str | None = None
    output_dir: Path = Path("data/derived")
    cache_dir: Path = Path("data/cache")
    runs_dir: Path = Path("runs")
    force: bool = False
    mock: bool = False
    filter_expr: str | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> PipelineConfig:
        from audio_engine.core.source import resolve_source_input

        path = Path(path)
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        steps = [
            PipelineStep(
                name=s.get("name", s["operator"]),
                operator=s["operator"],
                params=s.get("params", {}),
                input_audio_key=s.get("input_audio_key", "raw"),
                output_audio_key=s.get("output_audio_key"),
            )
            for s in data.get("pipeline", [])
        ]
        input_cfg = data.get("input", {}) or {}
        output_cfg = data.get("output", {}) or {}
        manifest = input_cfg.get("manifest", "") or ""
        source_dir = input_cfg.get("source_dir")
        source_id = input_cfg.get("source")

        if source_id and not manifest and not source_dir:
            resolved = resolve_source_input(source_id)
            manifest = resolved.get("manifest", "")
            source_dir = resolved.get("source_dir")

        if not manifest and not source_dir:
            raise ValueError(
                f"Pipeline '{path}' needs input.manifest, input.source_dir, or input.source"
            )

        output_manifest = output_cfg.get("manifest") or data.get("output_manifest")
        return cls(
            name=data.get("name", path.stem),
            input_manifest=manifest,
            steps=steps,
            source_dir=source_dir,
            source_id=source_id,
            output_manifest=output_manifest,
            output_dir=Path(data.get("output_dir", "data/derived")),
            cache_dir=Path(data.get("cache_dir", "data/cache")),
            runs_dir=Path(data.get("runs_dir", "runs")),
            force=data.get("force", False),
            mock=data.get("mock", False),
            filter_expr=data.get("filter"),
        )


@dataclass
class RunMetrics:
    total: int = 0
    processed: int = 0
    skipped: int = 0
    cache_hits: int = 0
    failed: int = 0
    by_step: dict[str, dict[str, int]] = field(default_factory=dict)

    def record(self, step_name: str, *, processed: int = 0, skipped: int = 0, cache_hits: int = 0, failed: int = 0) -> None:
        if step_name not in self.by_step:
            self.by_step[step_name] = {"processed": 0, "skipped": 0, "cache_hits": 0, "failed": 0}
        self.by_step[step_name]["processed"] += processed
        self.by_step[step_name]["skipped"] += skipped
        self.by_step[step_name]["cache_hits"] += cache_hits
        self.by_step[step_name]["failed"] += failed
        self.processed += processed
        self.skipped += skipped
        self.cache_hits += cache_hits
        self.failed += failed

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "processed": self.processed,
            "skipped": self.skipped,
            "cache_hits": self.cache_hits,
            "failed": self.failed,
            "by_step": self.by_step,
        }


class PipelineRunner:
    """Execute a configured pipeline over a manifest with resume/cache support."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.metrics = RunMetrics()
        self.run_dir = self._create_run_dir()

    def _create_run_dir(self) -> Path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.config.runs_dir / f"{ts}_{self.config.name}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _setup_logging(self) -> None:
        log_path = self.run_dir / "run.log"
        logger.add(str(log_path), rotation="50 MB", level="DEBUG")

    def run(self) -> Manifest:
        self._setup_logging()
        if self.config.input_manifest:
            manifest_path = Path(self.config.input_manifest)
            if not manifest_path.is_absolute() and not manifest_path.exists():
                manifest_path = Manifest.resolve_path(self.config.input_manifest)
            manifest = Manifest.load(manifest_path)
        else:
            # No manifest yet: pipeline starts empty and an ingest step
            # (e.g. ingest.scan on input.source_dir) creates the samples.
            manifest_path = None
            manifest = Manifest([])
        if self.config.filter_expr:
            manifest = manifest.filter(self.config.filter_expr)

        self.metrics.total = len(manifest.samples)
        logger.info(
            "Pipeline '{}' started: {} samples, {} steps",
            self.config.name,
            len(manifest.samples),
            len(self.config.steps),
        )

        samples = list(manifest.samples)
        for step in self.config.steps:
            samples = self._run_step(step, samples)

        result = Manifest(samples)
        out_manifest = self.run_dir / "manifest.parquet"
        result.save(out_manifest)

        config_out = self.run_dir / "config.yaml"
        config_out.write_text(
            yaml.dump(
                {
                    "name": self.config.name,
                    "input_manifest": str(manifest_path) if manifest_path else "",
                    "source_dir": self.config.source_dir,
                    "source_id": self.config.source_id,
                    "output_manifest": self.config.output_manifest,
                    "filter": self.config.filter_expr,
                    "pipeline": [
                        {"name": s.name, "operator": s.operator, "params": s.params}
                        for s in self.config.steps
                    ],
                },
                allow_unicode=True,
            ),
            encoding="utf-8",
        )

        metrics_path = self.run_dir / "metrics.json"
        metrics_path.write_text(
            json.dumps(self.metrics.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        logger.info("Pipeline finished. Run dir: {}", self.run_dir)
        return result

    def _run_step(self, step: PipelineStep, samples: list[Sample]) -> list[Sample]:
        operator = OperatorRegistry.get(step.operator)
        logger.info("Step '{}' ({})", step.name, step.operator)

        op_config = OperatorConfig(
            params=dict(step.params),
            output_dir=self.config.output_dir,
            cache_dir=self.config.cache_dir,
            force=self.config.force,
            mock=self.config.mock,
        )
        if step.input_audio_key:
            op_config.params.setdefault("input_audio_key", step.input_audio_key)
        if step.output_audio_key:
            op_config.params.setdefault("output_audio_key", step.output_audio_key)

        if isinstance(operator, ManifestOperator):
            if self.config.source_dir:
                op_config.params.setdefault("source_dir", self.config.source_dir)
            updated = operator.run(list(samples), op_config)
            delta = len(updated) - len(samples)
            if delta > 0:
                self.metrics.record(step.name, processed=delta)
            elif delta < 0:
                self.metrics.record(step.name, processed=len(updated), skipped=-delta)
            else:
                self.metrics.record(step.name, processed=0)
            self.metrics.total = len(updated)
            return updated

        updated_samples: list[Sample] = []
        for idx, sample in enumerate(samples, start=1):
            try:
                result = operator.process(sample, op_config)
                updated_samples.append(result.sample)
                if result.skipped:
                    self.metrics.record(step.name, skipped=1)
                elif result.cache_hit:
                    self.metrics.record(step.name, cache_hits=1)
                else:
                    self.metrics.record(step.name, processed=1)
            except Exception as exc:
                logger.error("Sample {} failed at step '{}': {}", sample.id, step.name, exc)
                failed = sample.model_copy(deep=True)
                failed.mark_failed(step.operator, str(exc))
                updated_samples.append(failed)
                self.metrics.record(step.name, failed=1)

            if idx % 100 == 0:
                logger.debug("Step '{}' progress: {}/{}", step.name, idx, len(samples))

        return updated_samples

    def run_single_operator(
        self,
        operator_name: str,
        manifest: Manifest,
        params: dict[str, Any] | None = None,
    ) -> Manifest:
        """Run a single operator over all samples (CLI 'run' command)."""
        step = PipelineStep(name=operator_name, operator=operator_name, params=params or {})
        samples = self._run_step(step, list(manifest.samples))
        return Manifest(samples)
