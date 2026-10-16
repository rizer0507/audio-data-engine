"""Bounded sample scheduling; workers never write tag state or checkpoints."""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Iterator
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from contextvars import copy_context
from dataclasses import dataclass, field
from itertools import islice

from loguru import logger

from audio_engine.core.sample import Sample
from audio_engine.core.tagging.adapters import AcousticAdapter, RealAcousticAdapter
from audio_engine.core.tagging.config import TaggingConfig
from audio_engine.core.tagging.engine import EngineStats, MultiTagEngine, SampleTaggingResult
from audio_engine.core.tagging.monitoring import TaggingMonitor
from audio_engine.core.tagging.state import SampleTagState

_local = threading.local()
_AUTO_PROCESS_MIN_SAMPLES = 4096


@dataclass(frozen=True)
class TaggingExecution:
    workers: int
    executor: str
    max_in_flight: int
    batch_size: int = 1

    @classmethod
    def resolve(
        cls,
        workers: int | None,
        executor: str,
        max_in_flight: int | None,
        *,
        compute: bool,
        total: int,
        batch_size: int = 1,
    ) -> TaggingExecution:
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError("tagging batch_size must be >= 1")
        if workers is not None and (
            not isinstance(workers, int) or isinstance(workers, bool) or workers < 1
        ):
            raise ValueError("tagging workers must be >= 1")
        if executor not in {"auto", "sequential", "thread", "process"}:
            raise ValueError("tagging executor must be auto, sequential, thread or process")
        if max_in_flight is not None and (
            not isinstance(max_in_flight, int)
            or isinstance(max_in_flight, bool)
            or max_in_flight < 1
        ):
            raise ValueError("tagging max_in_flight must be >= 1")
        if compute and executor == "process":
            raise ValueError("compute_acoustic requires thread or sequential executor")
        count = min(workers or min(8, os.cpu_count() or 1), max(1, total))
        # Spawn + config transfer costs more than evaluation on small text batches.
        # Threads also overlap the coordinator's per-sample result file writes.
        mode = (
            ("thread" if compute or total < _AUTO_PROCESS_MIN_SAMPLES else "process")
            if executor == "auto"
            else executor
        )
        if count == 1 or mode == "sequential":
            count, mode = 1, "sequential"
        size = min(batch_size, max(1, (total + count - 1) // count))
        limit = max_in_flight or count * 2 * size
        size = min(size, max(1, limit // count))
        return cls(count, mode, limit, size)


def _init_worker(config: TaggingConfig, adapter: AcousticAdapter | None, monitor=None) -> None:
    # Each thread owns its model sessions (Silero has mutable recurrent state).
    if adapter is None and config.evidence_mode == "compute_acoustic":
        adapter = RealAcousticAdapter()
    _local.engine = MultiTagEngine(config, acoustic_adapter=adapter)
    _local.monitor = monitor


@dataclass
class BlockResult:
    items: list = field(default_factory=list)
    error: str | None = None
    failed_sample_id: str | None = None
    traceback: str = ""


def _evaluate_block(jobs, snapshot_id, rule_ids, preview):
    output = BlockResult()
    engine, monitor = _local.engine, _local.monitor
    for index, sample, prior in jobs:
        engine.stats = EngineStats()
        started = time.perf_counter()
        if monitor:
            monitor.engine("text", queued=-1, running=1)
        try:
            result = engine.evaluate_sample(
                sample,
                prior_state=prior,
                snapshot_id=snapshot_id,
                rule_ids=rule_ids,
                dry_run=preview,
            )
        except Exception as exc:  # noqa: BLE001 - transport failures with partial block results
            import traceback

            output.error = f"{type(exc).__name__}: {exc}"
            output.traceback = traceback.format_exc()
            output.failed_sample_id = sample.id
            if monitor:
                monitor.finish_engine("text", time.perf_counter() - started, failed=True)
                monitor.engine(
                    "text",
                    queued=-(len(jobs) - len(output.items) - 1),
                    cancelled=len(jobs) - len(output.items) - 1,
                )
            return output
        if monitor:
            monitor.finish_engine("text", time.perf_counter() - started)
        engine.stats.timings["total"] = time.perf_counter() - started
        output.items.append((index, sample, prior, result, engine.stats))
    return output


class _LockedAdapter:
    """Injected adapters may own mutable sessions; serialize only their compute calls."""

    def __init__(self, adapter: AcousticAdapter):
        self.adapter = adapter
        self.lock = threading.Lock()

    def compute(self, sample, *, deps, params):
        with self.lock:
            return self.adapter.compute(sample, deps=deps, params=params)


def evaluate_samples(
    jobs: Iterator[tuple[int, Sample, SampleTagState]],
    *,
    config: TaggingConfig,
    execution: TaggingExecution,
    adapter: AcousticAdapter | None,
    snapshot_id: str,
    rule_ids: set[str] | None,
    preview: bool,
    monitor: TaggingMonitor | None = None,
) -> Iterator[tuple[int, Sample, SampleTagState, SampleTaggingResult, EngineStats]]:
    """Yield completions without input-order blocking; bound all submitted work."""
    if execution.executor == "sequential":
        # Do not replace the caller thread's engine when two services overlap.
        engine = MultiTagEngine(
            config,
            acoustic_adapter=adapter
            or (RealAcousticAdapter() if config.evidence_mode == "compute_acoustic" else None),
        )
        for index, sample, prior in jobs:
            if monitor:
                monitor.engine("text", submitted=1, running=1)
            started = time.perf_counter()
            engine.stats = EngineStats()
            try:
                result = engine.evaluate_sample(
                    sample,
                    prior_state=prior,
                    snapshot_id=snapshot_id,
                    rule_ids=rule_ids,
                    dry_run=preview,
                )
            except Exception:
                if monitor:
                    monitor.finish_engine("text", time.perf_counter() - started, failed=True)
                logger.exception("tagging sample failed sample_id={}", sample.id)
                raise
            if monitor:
                monitor.finish_engine("text", time.perf_counter() - started)
            engine.stats.timings["total"] = time.perf_counter() - started
            yield index, sample, prior, result, engine.stats
        return

    is_process = execution.executor == "process"
    factory = ProcessPoolExecutor if is_process else ThreadPoolExecutor
    shared_adapter = adapter
    if adapter is not None and not getattr(adapter, "thread_safe", False):
        shared_adapter = _LockedAdapter(adapter)
    pool = factory(
        max_workers=execution.workers,
        initializer=_init_worker,
        initargs=(config, None if is_process else shared_adapter, None if is_process else monitor),
    )
    pending = {}
    in_flight = 0
    last_heartbeat = time.perf_counter()
    exhausted = False
    try:
        while pending or not exhausted:
            while not exhausted and in_flight < execution.max_in_flight:
                block = list(
                    islice(jobs, min(execution.batch_size, execution.max_in_flight - in_flight))
                )
                if not block:
                    exhausted = True
                    break
                if monitor:
                    monitor.engine(
                        "text", submitted=len(block), queued=len(block), blocks_submitted=1
                    )
                logger.bind(
                    event="block_submitted",
                    sample_count=len(block),
                    first_sample_id=block[0][1].id,
                    last_sample_id=block[-1][1].id,
                ).trace("tagging block submitted")
                if is_process:
                    future = pool.submit(_evaluate_block, block, snapshot_id, rule_ids, preview)
                else:
                    future = pool.submit(
                        copy_context().run, _evaluate_block, block, snapshot_id, rule_ids, preview
                    )
                pending[future] = block
                in_flight += len(block)
            if not pending:
                break
            completed, _ = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
            if time.perf_counter() - last_heartbeat >= 15:
                logger.info(
                    "tagging blocks active executor={} workers={} blocks={} samples_in_flight={}",
                    execution.executor,
                    execution.workers,
                    len(pending),
                    in_flight,
                )
                last_heartbeat = time.perf_counter()
            for future in completed:
                block = pending.pop(future)
                in_flight -= len(block)
                try:
                    output = future.result()
                except Exception as exc:
                    if monitor:
                        monitor.engine(
                            "text", queued=-len(block) if is_process else 0, cancelled=len(block)
                        )
                    raise RuntimeError(
                        f"tagging block failed sample_id={block[0][1].id}: {exc}"
                    ) from exc
                if monitor:
                    monitor.engine("text", blocks_completed=1)
                    if is_process:
                        monitor.engine(
                            "text",
                            queued=-len(block),
                            completed=len(output.items) + int(output.error is not None),
                            failed=int(output.error is not None),
                            cancelled=len(block)
                            - len(output.items)
                            - int(output.error is not None),
                            elapsed_seconds=sum(x[4].timings.get("total", 0) for x in output.items),
                        )
                for item in output.items:
                    yield item
                if output.error:
                    logger.bind(event="sample_failed", sample_id=output.failed_sample_id).error(
                        "tagging sample failed sample_id={} error={} traceback={}",
                        output.failed_sample_id,
                        output.error,
                        output.traceback,
                    )
                    raise RuntimeError(
                        f"tagging sample failed sample_id={output.failed_sample_id}: {output.error}"
                    )
    finally:
        for future, block in pending.items():
            if future.cancel() and monitor:
                monitor.engine("text", queued=-len(block), cancelled=len(block))
        pool.shutdown(wait=True, cancel_futures=True)
        if monitor and is_process:
            # Process state is observed at block completion, never invent active worker counts.
            with monitor.lock:
                item = monitor.engines.get("text", {})
                remaining = item.get("queued", 0)
            monitor.engine("text", queued=-remaining, cancelled=remaining)
