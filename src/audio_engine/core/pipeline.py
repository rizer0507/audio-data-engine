from __future__ import annotations

import json
import os
import time
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from audio_engine.core.checkpoint import (
    COUNT_KEYS,
    StepCheckpoint,
    digest_payload,
    digest_samples,
    empty_counts,
)
from audio_engine.core.manifest import Manifest
from audio_engine.core.operator import BaseOperator, BatchOperator, ManifestOperator, OperatorConfig
from audio_engine.core.registry import OperatorRegistry
from audio_engine.core.sample import Sample

EXECUTORS = ("thread", "process", "sequential")


def _process_one_task(
    operator_name: str, sample: Sample, config: OperatorConfig
) -> tuple[Sample, str]:
    """Top-level worker for ProcessPoolExecutor (must be picklable on Windows spawn)."""
    # Child processes start empty; re-import so OperatorRegistry is populated.
    import audio_engine.operators  # noqa: F401

    operator = OperatorRegistry.get(operator_name)
    try:
        result = operator.process(sample, config)
    except Exception as exc:
        failed = sample.model_copy(deep=True)
        failed.mark_failed(operator_name, str(exc))
        return failed, "failed"

    if result.skipped:
        return result.sample, "skipped"
    if result.cache_hit:
        return result.sample, "cache_hits"
    return result.sample, "processed"


@dataclass(frozen=True)
class ExecutionConfig:
    """Concurrency settings for one step; a step-level value overrides the pipeline default."""

    executor: str = "thread"
    workers: int = 4
    max_in_flight: int = 0  # 0 → workers * 4
    fail_fast: bool = False
    checkpoint_every: int = 1000  # samples per checkpoint part; 0 disables checkpointing

    def __post_init__(self) -> None:
        if self.executor not in EXECUTORS:
            raise ValueError(
                f"Unsupported execution.executor '{self.executor}'. Use one of {EXECUTORS}."
            )
        if self.workers < 1:
            raise ValueError(f"execution.workers must be >= 1, got {self.workers}")
        if self.checkpoint_every < 0:
            raise ValueError(
                f"execution.checkpoint_every must be >= 0, got {self.checkpoint_every}"
            )

    @property
    def concurrent(self) -> bool:
        return self.executor in ("thread", "process") and self.workers > 1

    @property
    def in_flight(self) -> int:
        return self.max_in_flight if self.max_in_flight > 0 else self.workers * 4

    def merged(self, override: dict[str, Any] | None) -> ExecutionConfig:
        if not override:
            return self
        known = {f.name for f in fields(self)}
        unknown = set(override) - known
        if unknown:
            raise ValueError(f"Unknown execution keys: {sorted(unknown)}. Allowed: {sorted(known)}")
        return replace(self, **override)


@dataclass
class PipelineStep:
    name: str
    operator: str
    params: dict[str, Any] = field(default_factory=dict)
    input_audio_key: str = "raw"
    output_audio_key: str | None = None
    execution: dict[str, Any] = field(default_factory=dict)


SHARD_STRATEGIES = ("hash", "duration-balanced")


@dataclass(frozen=True)
class ShardingConfig:
    """When set on a pipeline, ``pipeline run`` does split → parallel shards → merge."""

    shards: int
    strategy: str = "hash"
    parallel_shards: int | None = None
    gpus: tuple[str, ...] = ()
    instances_per_gpu: int = 1
    workers: int | None = None
    executor: str | None = None

    def __post_init__(self) -> None:
        if self.shards < 1:
            raise ValueError(f"sharding.shards must be >= 1, got {self.shards}")
        if self.strategy not in SHARD_STRATEGIES:
            raise ValueError(
                f"Unknown sharding.strategy '{self.strategy}'. Use one of {SHARD_STRATEGIES}."
            )
        if self.instances_per_gpu < 1:
            raise ValueError(
                f"sharding.instances_per_gpu must be >= 1, got {self.instances_per_gpu}"
            )
        if self.parallel_shards is not None and self.parallel_shards < 1:
            raise ValueError(f"sharding.parallel_shards must be >= 1, got {self.parallel_shards}")
        if self.executor is not None and self.executor not in EXECUTORS:
            raise ValueError(
                f"Unsupported sharding.executor '{self.executor}'. Use one of {EXECUTORS}."
            )
        if len(self.gpus) != len(set(self.gpus)):
            raise ValueError("sharding.gpus must not contain duplicate ids")

    @property
    def effective_parallel(self) -> int:
        return self.parallel_shards if self.parallel_shards is not None else self.shards

    @property
    def gpu_slots(self) -> int:
        return len(self.gpus) * self.instances_per_gpu if self.gpus else 0

    def validate_parallel(self) -> None:
        parallel = self.effective_parallel
        if self.gpus and parallel > self.gpu_slots:
            raise ValueError(
                "sharding.parallel_shards cannot exceed GPUs * instances_per_gpu "
                f"({len(self.gpus)} * {self.instances_per_gpu} = {self.gpu_slots}); "
                "also written as --instances-per-gpu on the CLI"
            )

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> ShardingConfig | None:
        if not data:
            return None
        raw_gpus = data.get("gpus") or []
        if isinstance(raw_gpus, str):
            gpus = tuple(item.strip() for item in raw_gpus.split(",") if item.strip())
        else:
            gpus = tuple(str(item).strip() for item in raw_gpus if str(item).strip())
        cfg = cls(
            shards=int(data["shards"]),
            strategy=str(data.get("strategy", "hash")),
            parallel_shards=(
                int(data["parallel_shards"]) if data.get("parallel_shards") is not None else None
            ),
            gpus=gpus,
            instances_per_gpu=int(data.get("instances_per_gpu", 1)),
            workers=int(data["workers"]) if data.get("workers") is not None else None,
            executor=str(data["executor"]) if data.get("executor") is not None else None,
        )
        cfg.validate_parallel()
        return cfg


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
    catalog_dir: Path = Path("data/catalog")
    force: bool = False
    mock: bool = False
    filter_expr: str | None = None
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    # Highest priority (CLI flags): applied after pipeline default + step override.
    execution_override: dict[str, Any] = field(default_factory=dict)
    # Existing run dir (or run id under runs_dir) to continue from its checkpoints.
    resume: str | None = None
    # Exact run dir to use, created when missing; reuses checkpoints found inside.
    run_dir: str | None = None
    sharding: ShardingConfig | None = None
    # Original YAML path; used by sharded runs to re-invoke ``pipeline run``.
    config_path: str | None = None

    def step_execution(self, step: PipelineStep) -> ExecutionConfig:
        return self.execution.merged(step.execution).merged(self.execution_override)

    def digest(self) -> str:
        """Fingerprint the executable pipeline, including operator implementations."""
        return digest_payload(
            {
                "name": self.name,
                "mock": self.mock,
                "steps": [
                    {
                        "name": step.name,
                        "operator": step.operator,
                        "version": getattr(OperatorRegistry.get(step.operator), "version", ""),
                        "params": step.params,
                        "input_audio_key": step.input_audio_key,
                        "output_audio_key": step.output_audio_key,
                    }
                    for step in self.steps
                ],
            }
        )

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
                execution=s.get("execution") or {},
            )
            for s in data.get("pipeline", [])
        ]
        input_cfg = data.get("input", {}) or {}
        output_cfg = data.get("output", {}) or {}
        manifest = os.path.expandvars(str(input_cfg.get("manifest", "") or ""))
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

        raw_output_manifest = output_cfg.get("manifest") or data.get("output_manifest")
        output_manifest = (
            os.path.expandvars(str(raw_output_manifest)) if raw_output_manifest else None
        )
        sharding = ShardingConfig.from_mapping(data.get("sharding"))
        if sharding is not None and not output_manifest:
            raise ValueError(f"Pipeline '{path}' enables sharding but has no output.manifest")
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
            catalog_dir=Path(data.get("catalog_dir", "data/catalog")),
            force=data.get("force", False),
            mock=data.get("mock", False),
            filter_expr=data.get("filter"),
            execution=ExecutionConfig().merged(data.get("execution") or {}),
            sharding=sharding,
            config_path=str(path.resolve()),
        )


@dataclass
class RunMetrics:
    total: int = 0
    processed: int = 0
    skipped: int = 0
    cache_hits: int = 0
    failed: int = 0
    by_step: dict[str, dict[str, int]] = field(default_factory=dict)

    def record_counts(self, step_name: str, counts: dict[str, int]) -> None:
        self.record(step_name, **{key: int(counts.get(key, 0)) for key in COUNT_KEYS})

    def record(
        self,
        step_name: str,
        *,
        processed: int = 0,
        skipped: int = 0,
        cache_hits: int = 0,
        failed: int = 0,
    ) -> None:
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
        if self.config.resume:
            candidate = Path(self.config.resume)
            if not candidate.exists():
                candidate = self.config.runs_dir / self.config.resume
            if not candidate.is_dir():
                raise FileNotFoundError(f"Cannot resume: run dir not found ({self.config.resume})")
            return candidate

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.config.runs_dir / f"{ts}_{self.config.name}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _config_digest(self) -> str:
        """Any change to the step list, params or operator versions invalidates checkpoints."""
        return self.config.digest()

    def _open_checkpoint(
        self,
        order: int,
        step: PipelineStep,
        operator: BaseOperator | ManifestOperator,
        op_config: OperatorConfig,
        samples: list[Sample],
        execution: ExecutionConfig,
    ) -> StepCheckpoint | None:
        if execution.checkpoint_every <= 0:
            return None

        fingerprint = {
            "pipeline": self._config_digest(),
            "operator": step.operator,
            "version": operator.version,
            "params": digest_payload(op_config.params),
            "input": digest_samples(samples),
            "checkpoint_every": execution.checkpoint_every,
        }
        directory = self.run_dir / "checkpoints" / f"{order:02d}_{step.name}"
        directory.mkdir(parents=True, exist_ok=True)
        checkpoint = StepCheckpoint(directory, fingerprint)
        # --force means "recompute": write fresh checkpoints, never restore old ones.
        return checkpoint if self.config.force else checkpoint.load()

    def _setup_logging(self) -> None:
        log_path = self.run_dir / "run.log"
        logger.add(str(log_path), rotation="50 MB", level="DEBUG")

    def run(self) -> Manifest:
        self._setup_logging()
        if self.config.input_manifest:
            if self.config.input_manifest.startswith("manifest_"):
                from audio_engine.core.catalog import ArtifactCatalog

                manifest_path = Path(
                    ArtifactCatalog(self.config.catalog_dir)
                    .get(self.config.input_manifest, verify=True)
                    .uri
                )
            else:
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
            "Pipeline '{}' {}: {} samples, {} steps, execution={}",
            self.config.name,
            f"resumed from {self.run_dir}" if self.config.resume else "started",
            len(manifest.samples),
            len(self.config.steps),
            self.config.execution,
        )

        samples = list(manifest.samples)
        for order, step in enumerate(self.config.steps):
            samples = self._run_step(step, samples, order=order)

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
                    "execution": asdict(self.config.execution),
                    "config_digest": self._config_digest(),
                    "pipeline": [
                        {
                            "name": s.name,
                            "operator": s.operator,
                            "params": s.params,
                            "execution": asdict(self.config.step_execution(s)),
                        }
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

    def _run_step(self, step: PipelineStep, samples: list[Sample], order: int = 0) -> list[Sample]:
        operator = OperatorRegistry.get(step.operator)

        op_config = OperatorConfig(
            params=dict(step.params),
            output_dir=self.config.output_dir,
            cache_dir=self.config.cache_dir,
            force=self.config.force,
            mock=self.config.mock,
            run_dir=self.run_dir,
            step_name=step.name,
        )
        if step.input_audio_key:
            op_config.params.setdefault("input_audio_key", step.input_audio_key)
        if step.output_audio_key:
            op_config.params.setdefault("output_audio_key", step.output_audio_key)

        execution = self.config.step_execution(step)
        checkpoint = self._open_checkpoint(order, step, operator, op_config, samples, execution)

        if checkpoint is not None and checkpoint.complete:
            restored = checkpoint.read_samples()
            self.metrics.record_counts(step.name, checkpoint.restored_counts())
            self.metrics.total = len(restored)
            logger.info(
                "Step '{}' ({}): restored {} samples from checkpoint, not re-run",
                step.name,
                step.operator,
                len(restored),
            )
            return restored

        if isinstance(operator, ManifestOperator):
            return self._run_manifest_step(step, operator, op_config, samples, checkpoint)
        return self._run_sample_step(step, operator, op_config, samples, execution, checkpoint)

    def _run_manifest_step(
        self,
        step: PipelineStep,
        operator: ManifestOperator,
        op_config: OperatorConfig,
        samples: list[Sample],
        checkpoint: StepCheckpoint | None,
    ) -> list[Sample]:
        """Whole-collection operators run in one shot, so they checkpoint as a single part."""
        logger.info("Step '{}' ({}): manifest operator, sequential", step.name, step.operator)
        if self.config.source_dir:
            op_config.params.setdefault("source_dir", self.config.source_dir)

        started = time.perf_counter()
        updated = operator.run(list(samples), op_config)
        delta = len(updated) - len(samples)
        if delta > 0:
            counts = {"processed": delta}
        elif delta < 0:
            counts = {"processed": len(updated), "skipped": -delta}
        else:
            counts = {}
        self.metrics.record_counts(step.name, counts)
        self.metrics.total = len(updated)

        if checkpoint is not None:
            checkpoint.append(updated, count_in=len(samples), counts=counts)
            checkpoint.finish(len(updated))

        logger.info(
            "Step '{}' finished: {} -> {} samples in {:.2f}s",
            step.name,
            len(samples),
            len(updated),
            time.perf_counter() - started,
        )
        return updated

    def _run_sample_step(
        self,
        step: PipelineStep,
        operator: BaseOperator,
        op_config: OperatorConfig,
        samples: list[Sample],
        execution: ExecutionConfig,
        checkpoint: StepCheckpoint | None,
    ) -> list[Sample]:
        done: list[Sample] = []
        if checkpoint is not None and checkpoint.restored:
            done = checkpoint.read_samples()
            self.metrics.record_counts(step.name, checkpoint.restored_counts())
            logger.info(
                "Step '{}' ({}): resuming after {} checkpointed samples",
                step.name,
                step.operator,
                len(done),
            )

        pending = samples[checkpoint.consumed :] if checkpoint is not None else samples
        batch_size = (
            execution.checkpoint_every
            if checkpoint is not None and execution.checkpoint_every > 0
            else len(pending) or 1
        )

        mode = (
            f"{execution.workers} {execution.executor}s, max_in_flight={execution.in_flight}"
            if execution.concurrent
            else "sequential"
        )
        logger.info(
            "Step '{}' ({}): {} samples, {}{}",
            step.name,
            step.operator,
            len(pending),
            mode,
            f", checkpoint every {batch_size}" if checkpoint is not None else "",
        )

        started = time.perf_counter()
        total = len(samples)
        already = len(done)
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            if isinstance(operator, BatchOperator):
                updated, counts = self._run_batch_operator(
                    step, operator, op_config, batch, execution
                )
            elif execution.concurrent and len(batch) > 1:
                updated, counts = self._run_concurrent(step, operator, op_config, batch, execution)
            else:
                updated, counts = self._run_sequential(step, operator, op_config, batch, execution)
            self.metrics.record_counts(step.name, counts)
            done.extend(updated)
            if checkpoint is not None:
                checkpoint.append(updated, count_in=len(batch), counts=counts)

            elapsed = time.perf_counter() - started
            done_now = already + (start + len(batch))
            # Only count freshly processed pending for rate (resume-friendly).
            processed_pending = start + len(batch)
            rate = processed_pending / elapsed if elapsed > 0 else 0.0
            remaining = max(len(pending) - processed_pending, 0)
            eta = remaining / rate if rate > 0 else None
            pct = (100.0 * done_now / total) if total else 0.0
            eta_txt = f"{eta:.0f}s" if eta is not None else "n/a"
            logger.info(
                "[PROGRESS] step={} done={}/{} ({:.1f}%) rate={:.2f} samples/s "
                "elapsed={:.1f}s eta={} checkpoint_every={}",
                step.name,
                done_now,
                total,
                pct,
                rate,
                elapsed,
                eta_txt,
                execution.checkpoint_every,
            )

        if checkpoint is not None:
            checkpoint.finish(len(done))

        elapsed = time.perf_counter() - started
        rate = len(pending) / elapsed if elapsed > 0 else 0.0
        logger.info(
            "Step '{}' finished: {} samples in {:.2f}s ({:.1f} samples/s)",
            step.name,
            len(pending),
            elapsed,
            rate,
        )
        return done

    def _run_batch_operator(
        self,
        step: PipelineStep,
        operator: BatchOperator,
        op_config: OperatorConfig,
        samples: list[Sample],
        execution: ExecutionConfig,
    ) -> tuple[list[Sample], dict[str, int]]:
        """Run a GPU/model batch without adding thread/process concurrency."""
        if execution.concurrent:
            logger.warning(
                "Step '{}' uses batch operator {}; workers/executor are ignored",
                step.name,
                step.operator,
            )
        try:
            results = operator.process_batch(samples, op_config)
        except Exception as exc:  # noqa: BLE001 - isolate a failed model batch
            logger.error("Batch failed at step '{}': {}", step.name, exc)
            if execution.fail_fast:
                raise RuntimeError(f"Step '{step.name}' batch aborted: {exc}") from exc
            failed_samples = []
            for sample in samples:
                failed = sample.model_copy(deep=True)
                failed.mark_failed(step.operator, str(exc))
                failed_samples.append(failed)
            return failed_samples, {**empty_counts(), "failed": len(failed_samples)}

        if len(results) != len(samples):
            raise RuntimeError(
                f"Batch operator {step.operator} returned {len(results)} results "
                f"for {len(samples)} samples"
            )

        updated: list[Sample] = []
        counts = empty_counts()
        for result in results:
            updated.append(result.sample)
            if result.skipped:
                outcome = "skipped"
            elif result.cache_hit:
                outcome = "cache_hits"
            elif result.sample.status.get(step.operator) == "failed":
                outcome = "failed"
            else:
                outcome = "processed"
            counts[outcome] += 1
            if outcome == "failed" and execution.fail_fast:
                raise RuntimeError(
                    f"Step '{step.name}' aborted (fail_fast) on sample {result.sample.id}: "
                    f"{result.sample.errors.get(step.operator, 'unknown error')}"
                )
        return updated, counts

    def _process_one(
        self,
        step: PipelineStep,
        operator: BaseOperator,
        op_config: OperatorConfig,
        sample: Sample,
    ) -> tuple[Sample, str]:
        """Run one sample and classify the outcome. Never raises — runs inside workers."""
        try:
            result = operator.process(sample, op_config)
        except Exception as exc:
            logger.error("Sample {} failed at step '{}': {}", sample.id, step.name, exc)
            failed = sample.model_copy(deep=True)
            failed.mark_failed(step.operator, str(exc))
            return failed, "failed"

        if result.skipped:
            return result.sample, "skipped"
        if result.cache_hit:
            return result.sample, "cache_hits"
        return result.sample, "processed"

    def _run_sequential(
        self,
        step: PipelineStep,
        operator: BaseOperator,
        op_config: OperatorConfig,
        samples: list[Sample],
        execution: ExecutionConfig,
    ) -> tuple[list[Sample], dict[str, int]]:
        updated_samples: list[Sample] = []
        counts = empty_counts()
        for idx, sample in enumerate(samples, start=1):
            processed, outcome = self._process_one(step, operator, op_config, sample)
            updated_samples.append(processed)
            counts[outcome] += 1
            if outcome == "failed" and execution.fail_fast:
                raise RuntimeError(
                    f"Step '{step.name}' aborted (fail_fast) on sample {processed.id}: "
                    f"{processed.errors.get(step.operator, 'unknown error')}"
                )
            if idx % 100 == 0:
                logger.debug("Step '{}' progress: {}/{}", step.name, idx, len(samples))
        return updated_samples, counts

    def _run_concurrent(
        self,
        step: PipelineStep,
        operator: BaseOperator,
        op_config: OperatorConfig,
        samples: list[Sample],
        execution: ExecutionConfig,
    ) -> tuple[list[Sample], dict[str, int]]:
        """Bounded worker pool: at most `in_flight` tasks queued, output order preserved."""
        total = len(samples)
        results: list[Sample | None] = [None] * total
        counts = empty_counts()
        pending = iter(enumerate(samples))
        completed = 0
        abort: str | None = None

        if execution.executor == "process":
            pool_cm = ProcessPoolExecutor(max_workers=execution.workers)

            def submit(pool, sample):
                return pool.submit(_process_one_task, step.operator, sample, op_config)
        else:
            pool_cm = ThreadPoolExecutor(
                max_workers=execution.workers, thread_name_prefix=f"ade-{step.name}"
            )

            def submit(pool, sample):
                return pool.submit(self._process_one, step, operator, op_config, sample)

        with pool_cm as pool:
            futures: dict[Future[tuple[Sample, str]], int] = {}

            def submit_next() -> bool:
                item = next(pending, None)
                if item is None:
                    return False
                idx, sample = item
                futures[submit(pool, sample)] = idx
                return True

            for _ in range(execution.in_flight):
                if not submit_next():
                    break

            while futures:
                done, _ = wait(list(futures), return_when=FIRST_COMPLETED)
                for future in done:
                    idx = futures.pop(future)
                    processed, outcome = future.result()
                    results[idx] = processed
                    # counts are only touched by this (main) thread
                    counts[outcome] += 1
                    completed += 1
                    if outcome == "failed" and execution.fail_fast and abort is None:
                        abort = (
                            f"sample {processed.id}: "
                            f"{processed.errors.get(step.operator, 'unknown error')}"
                        )
                    if completed % 100 == 0:
                        logger.debug("Step '{}' progress: {}/{}", step.name, completed, total)
                if abort is not None:
                    break
                for _ in range(len(done)):
                    if not submit_next():
                        break

        if abort is not None:
            raise RuntimeError(f"Step '{step.name}' aborted (fail_fast) on {abort}")
        return [s for s in results if s is not None], counts

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
