"""One bounded acoustic scheduler per run, independent of tag rule count."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from typing import Any

from loguru import logger

from audio_engine.core.evidence.store import merge_quality_fields
from audio_engine.core.sample import Sample
from audio_engine.core.tagging.adapters import RealAcousticAdapter
from audio_engine.core.tagging.evidence.acoustic import acoustic_from_quality
from audio_engine.core.tagging.monitoring import TaggingMonitor

DEPS = ("vad", "energy", "dnsmos")


def resolve_engine_limits(workers, max_in_flight, retries):
    if not isinstance(workers, dict) or not isinstance(max_in_flight, dict):
        raise TypeError("engine workers and in-flight limits must be mappings")
    if set(workers) - set(DEPS) or set(max_in_flight) - set(DEPS):
        raise ValueError("engine limits accept only vad, energy, dnsmos")
    if not isinstance(retries, int) or isinstance(retries, bool) or retries < 0:
        raise ValueError("engine_retries must be >= 0")
    resolved = {dep: workers.get(dep, 4) for dep in DEPS}
    limits = {dep: max_in_flight.get(dep, resolved[dep] * 2) for dep in DEPS}
    if any(
        not isinstance(n, int) or isinstance(n, bool) or n < 1
        for n in (*resolved.values(), *limits.values())
    ):
        raise ValueError("engine workers and in-flight limits must be positive integers")
    return resolved, limits


class AcousticScheduler:
    name = "scheduled_acoustic"
    thread_safe = True

    def __init__(
        self,
        *,
        workers: dict[str, int],
        max_in_flight: dict[str, int],
        retries: int,
        monitor: TaggingMonitor,
        adapter_factory=None,
    ):
        self.workers, self.limits = resolve_engine_limits(workers, max_in_flight, retries)
        self.monitor, self.retries = monitor, retries
        self.factory = adapter_factory or RealAcousticAdapter
        self.local = threading.local()
        self.gates = {dep: threading.BoundedSemaphore(self.limits[dep]) for dep in DEPS}
        self.pools = {
            dep: ThreadPoolExecutor(
                max_workers=self.workers[dep], thread_name_prefix=f"tagging-{dep}"
            )
            for dep in DEPS
        }

    def _compute(self, dep: str, sample: Sample, params: dict[str, Any]):
        self.monitor.engine(dep, queued=-1, running=1)
        started = time.perf_counter()
        logger.bind(event="engine_started", engine=dep, sample_id=sample.id).trace(
            "tagging engine started"
        )
        failed = True
        try:
            adapter = getattr(self.local, "adapter", None)
            if adapter is None:
                adapter = self.factory()
                self.local.adapter = adapter
            for attempt in range(self.retries + 1):
                self.monitor.engine(dep, attempts=1)
                try:
                    view = adapter.compute(sample, deps={dep}, params=params)
                    failed = getattr(view, f"{dep}_status", None) == "failed"
                    if not failed or attempt == self.retries:
                        return view, dict(sample.quality or {})
                except Exception:
                    if attempt == self.retries:
                        raise
                self.monitor.engine(dep, retries=1)
                logger.bind(event="engine_retry", engine=dep, sample_id=sample.id).warning(
                    "tagging engine retry engine={} sample_id={} attempt={}",
                    dep,
                    sample.id,
                    attempt + 1,
                )
        except Exception:
            logger.bind(event="engine_failed", engine=dep, sample_id=sample.id).exception(
                "tagging engine failed engine={} sample_id={}", dep, sample.id
            )
            raise
        finally:
            elapsed = time.perf_counter() - started
            self.monitor.finish_engine(dep, elapsed, failed=failed)
            logger.bind(
                event="engine_finished",
                engine=dep,
                sample_id=sample.id,
                elapsed_seconds=elapsed,
                failed=failed,
            ).trace("tagging engine finished")
            # Keep the model session, release facts/history from this sample.
            if getattr(self.local, "adapter", None) is not None:
                scorer = getattr(self.local.adapter, "scorer", None)
                if scorer is not None:
                    from audio_engine.core.evidence.scoring import ScoreLedger
                    from audio_engine.core.evidence.store import SharedAcousticEvidenceStore

                    scorer.store = SharedAcousticEvidenceStore()
                    scorer.ledger = ScoreLedger()
                    calls = getattr(self.local.adapter, "model_calls", None)
                    if calls is not None:
                        calls.clear()

    def compute(self, sample: Sample, *, deps: set[str], params: dict[str, Any]):
        futures = []
        try:
            for dep in sorted(deps):
                self.gates[dep].acquire()
                self.monitor.engine(dep, submitted=1, queued=1)
                try:
                    # Different engines never mutate the same Sample concurrently.
                    isolated = sample.model_copy(deep=True)
                    context = copy_context()
                    future = self.pools[dep].submit(
                        context.run, self._compute, dep, isolated, params
                    )
                    future.add_done_callback(lambda _f, gate=self.gates[dep]: gate.release())
                    futures.append((dep, future))
                    logger.bind(event="engine_submitted", engine=dep, sample_id=sample.id).trace(
                        "tagging engine submitted"
                    )
                except BaseException:
                    self.monitor.engine(dep, queued=-1, cancelled=1)
                    self.gates[dep].release()
                    raise
            details = {}
            quality = dict(sample.quality or {})
            for dep, future in futures:
                view, fields = future.result()
                # Isolated copies contain other deps' old fields; merge only this engine's fields.
                from audio_engine.core.evidence.store import DEP_STATUS_KEYS, DEP_VALUE_KEYS

                keys = (DEP_STATUS_KEYS[dep], *DEP_VALUE_KEYS[dep])
                quality = merge_quality_fields(quality, {k: fields[k] for k in keys if k in fields})
                from audio_engine.core.tagging.adapters import apply_view_to_sample

                temporary = sample.model_copy(deep=False)
                temporary.quality = quality
                apply_view_to_sample(temporary, view)
                quality = merge_quality_fields(quality, temporary.quality)
                details.update(view.details or {})
            sample.quality = quality
            view = acoustic_from_quality(quality)
            view.details = details
            return view
        finally:
            # Drain all sibling calls before the caller releases this sample/block.
            for _, future in futures:
                if not future.done():
                    try:
                        future.result()
                    except Exception:  # noqa: BLE001 - drain sibling calls without masking the primary failure
                        logger.opt(exception=True).debug(
                            "tagging sibling engine call drained after failure"
                        )

    def close(self) -> None:
        for pool in self.pools.values():
            pool.shutdown(wait=True, cancel_futures=True)
