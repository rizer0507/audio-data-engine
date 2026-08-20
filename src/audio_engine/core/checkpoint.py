from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from audio_engine.core.artifacts import atomic_path, atomic_write_json, sample_digest
from audio_engine.core.manifest import Manifest
from audio_engine.core.sample import Sample

STATE_FILE = "_state.json"
COUNT_KEYS = ("processed", "skipped", "cache_hits", "failed")


def empty_counts() -> dict[str, int]:
    return dict.fromkeys(COUNT_KEYS, 0)


def digest_samples(samples: list[Sample]) -> str:
    """Fingerprint a step's input so a checkpoint is never replayed onto other data."""
    hasher = hashlib.sha256()
    for sample in samples:
        hasher.update(sample.id.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(sample_digest(sample).encode("utf-8"))
        hasher.update(b"\n")
    hasher.update(f"count={len(samples)}".encode())
    return hasher.hexdigest()


def digest_payload(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class StepCheckpoint:
    """Committed parts of one step, resumable across process restarts.

    A part covers a contiguous range of the step's input, written atomically and
    only then recorded in `_state.json`. A part file without a state entry is
    therefore treated as absent: crash-safe at the cost of recomputing at most
    one batch (which the operator cache usually makes cheap anyway).
    """

    def __init__(self, directory: Path, fingerprint: dict[str, Any]):
        self.directory = Path(directory)
        self.fingerprint = fingerprint
        self.parts: list[dict[str, Any]] = []
        self.complete = False
        self.output_count = 0
        self.restored = False

    @property
    def state_path(self) -> Path:
        return self.directory / STATE_FILE

    @property
    def consumed(self) -> int:
        """How many input samples the committed parts already cover."""
        return sum(int(part["count_in"]) for part in self.parts)

    def load(self) -> StepCheckpoint:
        if not self.state_path.exists():
            return self
        try:
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Checkpoint {}: unreadable state ({}), starting over", self.name, exc)
            return self

        if state.get("fingerprint") != self.fingerprint:
            logger.info(
                "Checkpoint {}: pipeline config, operator version or input changed, not reused",
                self.name,
            )
            return self

        recorded = state.get("parts") or []
        usable: list[dict[str, Any]] = []
        for part in recorded:
            if not (self.directory / part["file"]).exists():
                break  # only a contiguous prefix is usable
            usable.append(part)
        if len(usable) != len(recorded):
            logger.warning(
                "Checkpoint {}: {} of {} part files missing, keeping the first {}",
                self.name,
                len(recorded) - len(usable),
                len(recorded),
                len(usable),
            )

        self.parts = usable
        self.complete = bool(state.get("complete")) and len(usable) == len(recorded)
        self.output_count = int(state.get("output_count", 0))
        self.restored = bool(usable)
        return self

    @property
    def name(self) -> str:
        return self.directory.name

    def read_samples(self) -> list[Sample]:
        restored: list[Sample] = []
        for part in self.parts:
            restored.extend(Manifest.load(self.directory / part["file"]).samples)
        return restored

    def restored_counts(self) -> dict[str, int]:
        totals = empty_counts()
        for part in self.parts:
            for key in COUNT_KEYS:
                totals[key] += int((part.get("metrics") or {}).get(key, 0))
        return totals

    def append(self, samples: list[Sample], *, count_in: int, counts: dict[str, int]) -> None:
        file_name = f"part-{len(self.parts):06d}.parquet"
        with atomic_path(self.directory / file_name) as tmp:
            Manifest(samples).save(tmp)
        self.parts.append(
            {
                "file": file_name,
                "count_in": count_in,
                "count_out": len(samples),
                "metrics": {key: int(counts.get(key, 0)) for key in COUNT_KEYS},
            }
        )
        self._write_state()

    def finish(self, output_count: int) -> None:
        self.complete = True
        self.output_count = output_count
        self._write_state()

    def _write_state(self) -> None:
        atomic_write_json(
            self.state_path,
            {
                "fingerprint": self.fingerprint,
                "parts": self.parts,
                "complete": self.complete,
                "output_count": self.output_count,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            },
        )
