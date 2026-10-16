"""Run lifecycle and live, bounded engine metrics shared by tagging workers."""

from __future__ import annotations

import threading
import time
from contextvars import copy_context
from pathlib import Path
from typing import Any

from loguru import logger

from audio_engine.core.artifacts import atomic_write_json


class TaggingMonitor:
    def __init__(
        self, directory: Path, metadata: dict[str, Any], *, interval: float = 15, mirrors=()
    ):
        self.directory = directory
        self.mirrors = list(mirrors)
        self.metadata = {
            **metadata,
            "status": "running",
            "stage": "initializing",
            "started_at": time.time(),
        }
        self.interval = interval
        self.started = time.perf_counter()
        self.lock = threading.RLock()
        self.write_lock = threading.Lock()
        self.stop = threading.Event()
        self.engines: dict[str, dict[str, Any]] = {}
        self.progress = {
            "total": 0,
            "processed": 0,
            "restored": 0,
            "committed": 0,
            "blocks_completed": 0,
        }
        self.thread: threading.Thread | None = None

    def engine(self, name: str, **changes: Any) -> None:
        with self.lock:
            item = self.engines.setdefault(
                name,
                {
                    "submitted": 0,
                    "queued": 0,
                    "running": 0,
                    "completed": 0,
                    "failed": 0,
                    "cancelled": 0,
                    "reused": 0,
                    "retries": 0,
                    "attempts": 0,
                    "elapsed_seconds": 0.0,
                    "max_seconds": 0.0,
                    "peak_running": 0,
                    "peak_in_flight": 0,
                },
            )
            for key, value in changes.items():
                item[key] = item.get(key, 0) + value
            item["peak_running"] = max(item["peak_running"], item["running"])
            item["peak_in_flight"] = max(item["peak_in_flight"], item["running"] + item["queued"])

    def finish_engine(self, name: str, elapsed: float, *, failed: bool = False) -> None:
        self.engine(name, running=-1, completed=1, failed=int(failed), elapsed_seconds=elapsed)
        with self.lock:
            self.engines[name]["max_seconds"] = max(self.engines[name]["max_seconds"], elapsed)

    def update(self, *, stage: str | None = None, **progress: Any) -> None:
        changed = False
        with self.lock:
            self.progress.update(progress)
            if stage:
                changed = stage != self.metadata["stage"]
                self.metadata["stage"] = stage
        if changed:
            logger.bind(event="tagging_stage", stage=stage).info("tagging stage={}", stage)

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            elapsed = max(time.perf_counter() - self.started, 1e-6)
            engines = {}
            for name, item in self.engines.items():
                entry = dict(item)
                completed = entry["completed"]
                entry["rate_per_second"] = completed / elapsed
                entry["mean_seconds"] = entry["elapsed_seconds"] / completed if completed else 0.0
                entry["in_flight"] = entry["submitted"] - entry["completed"] - entry["cancelled"]
                if name == "text" and self.metadata.get("text_running_observable") is False:
                    entry["queued"] = None
                    entry["running"] = None
                    entry["peak_running"] = None
                engines[name] = entry
            progress = dict(self.progress)
            progress["blocks_completed"] = self.engines.get("text", {}).get("blocks_completed", 0)
            rate = progress["processed"] / elapsed
            return {
                **self.metadata,
                "updated_at": time.time(),
                "elapsed_seconds": elapsed,
                "progress": {
                    **progress,
                    "rate_per_second": rate,
                    "eta_seconds": max(0, progress["total"] - progress["processed"]) / rate
                    if rate
                    else None,
                },
                "engines": engines,
            }

    def persist(self, *, report: bool = False) -> None:
        with self.write_lock:
            snapshot = self.snapshot()
            for directory in (self.directory, *self.mirrors):
                atomic_write_json(
                    directory / "run.json",
                    {k: v for k, v in snapshot.items() if k not in {"engines", "progress"}},
                )
                atomic_write_json(directory / "metrics.json", snapshot)
                atomic_write_json(directory / "progress_view.json", snapshot["progress"])
        if report:
            p = snapshot["progress"]
            logger.bind(event="tagging_progress", engines=snapshot["engines"]).info(
                "tagging stage={} progress={}/{} committed={} restored={} rate={:.1f}/s engines={}",
                snapshot["stage"],
                p["processed"],
                p["total"],
                p["committed"],
                p["restored"],
                p["rate_per_second"],
                {
                    name: {
                        key: values[key] for key in ("in_flight", "completed", "failed", "reused")
                    }
                    for name, values in snapshot["engines"].items()
                },
            )

    def start(self) -> None:
        self.persist()

        def heartbeat():
            while not self.stop.wait(self.interval):
                try:
                    self.persist(report=True)
                except Exception:  # noqa: BLE001 - keep heartbeat alive after a metrics write error
                    logger.exception("tagging metrics write failed")

        context = copy_context()
        self.thread = threading.Thread(
            target=lambda: context.run(heartbeat), daemon=True, name="tagging-progress"
        )
        self.thread.start()

    def finish(self, status: str, **metadata: Any) -> None:
        self.stop.set()
        if self.thread:
            self.thread.join()
        with self.lock:
            self.metadata.update(status=status, stage=status, finished_at=time.time(), **metadata)
        self.persist(report=True)
