"""Immutable result blocks with a transactional sample index and lazy result access."""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path

from audio_engine.core.artifacts import atomic_path


class RunLease:
    """OS-released SQLite lock prevents two processes from resuming the same run."""

    def __init__(self, directory: Path):
        self.path = directory / "run.lock.sqlite3"
        self.db = None

    def __enter__(self):
        self.db = sqlite3.connect(self.path, timeout=0)
        try:
            self.db.execute("BEGIN EXCLUSIVE")
        except sqlite3.OperationalError as exc:
            self.db.close()
            self.db = None
            raise RuntimeError(f"tagging run already active: {self.path.parent}") from exc
        return self

    def __exit__(self, *args):
        if self.db is not None:
            self.db.close()


class ResultArchive:
    def __init__(self, directory: Path, *, readonly: bool = False):
        self.directory = directory
        self.path = directory / "sample_results.sqlite3"
        self.readonly = readonly
        self.pending = []
        self.legacy = []
        self.streams = OrderedDict()
        self.db = (
            sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True)
            if readonly
            else sqlite3.connect(self.path)
        )
        if not readonly:
            self.db.execute(
                "CREATE TABLE IF NOT EXISTS results (sample_id TEXT PRIMARY KEY, "
                "block TEXT NOT NULL, offset INTEGER NOT NULL, length INTEGER NOT NULL, "
                "committed INTEGER NOT NULL DEFAULT 0)"
            )
            self.db.commit()

    def add(self, sample_id: str, payload: dict) -> Path:
        self.pending.append((sample_id, payload))
        return self.path

    def flush(self) -> None:
        if not self.pending:
            return
        relative = f"sample_results/blocks/{uuid.uuid4().hex}.jsonl"
        rows = []
        with atomic_path(self.directory / relative) as tmp, tmp.open("wb") as stream:
            for sample_id, payload in self.pending:
                raw = (
                    json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
                ).encode("utf-8")
                offset = stream.tell()
                stream.write(raw)
                rows.append((sample_id, relative, offset, len(raw)))
        with self.db:
            self.db.executemany("INSERT OR REPLACE INTO results VALUES (?, ?, ?, ?, 0)", rows)
        self.pending.clear()

    def commit(self, sample_ids) -> None:
        with self.db:
            self.db.executemany(
                "UPDATE results SET committed=1 WHERE sample_id=?", ((sid,) for sid in sample_ids)
            )

    def invalidate(self, sample_ids) -> None:
        with self.db:
            self.db.executemany(
                "UPDATE results SET committed=0 WHERE sample_id=?", ((sid,) for sid in sample_ids)
            )

    def completed_ids(self) -> set[str]:
        self.flush_legacy()
        return {
            row[0] for row in self.db.execute("SELECT sample_id FROM results WHERE committed=1")
        }

    def load(self, sample_id: str) -> dict | None:
        row = self.db.execute(
            "SELECT block, offset, length FROM results WHERE sample_id=?", (sample_id,)
        ).fetchone()
        if row is None:
            return None
        stream = self.streams.pop(row[0], None)
        if stream is None:
            stream = (self.directory / row[0]).open("rb")
        self.streams[row[0]] = stream
        if len(self.streams) > 32:
            self.streams.popitem(last=False)[1].close()
        stream.seek(row[1])
        return json.loads(stream.read(row[2]))

    def import_legacy(self, sample_id: str, path: Path) -> None:
        self.legacy.append(
            (sample_id, str(path.relative_to(self.directory)), 0, path.stat().st_size)
        )
        if len(self.legacy) >= 1000:
            self.flush_legacy()

    def flush_legacy(self) -> None:
        if self.legacy:
            with self.db:
                self.db.executemany(
                    "INSERT OR IGNORE INTO results VALUES (?, ?, ?, ?, 1)", self.legacy
                )
            self.legacy.clear()

    def close(self) -> None:
        self.flush_legacy()
        for stream in self.streams.values():
            stream.close()
        self.streams.clear()
        self.db.close()


class ArchivedResults(Sequence):
    """Keep only output sample references in memory, not millions of rule evidence trees."""

    def __init__(self, ws, samples):
        self.ws, self.samples = ws, samples

    def __len__(self):
        return len(self.samples)

    def __iter__(self):
        from audio_engine.core.tagging.engine import result_from_dict
        from audio_engine.core.tagging.persist import sample_result_path

        archive = ResultArchive(self.ws.work_dir, readonly=True)
        try:
            for sample in self.samples:
                payload = archive.load(sample.id)
                if payload is None:
                    payload = json.loads(
                        sample_result_path(self.ws, sample.id).read_text(encoding="utf-8")
                    )
                yield result_from_dict(
                    {
                        **payload,
                        "effective_tags": list(sample.labels.get("tags") or []),
                        "eval_status": sample.labels.get("tagging", {}).get("eval_status")
                        or payload.get("eval_status"),
                    }
                )
        finally:
            archive.close()

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        from audio_engine.core.tagging.engine import result_from_dict
        from audio_engine.core.tagging.persist import load_sample_result

        payload = load_sample_result(self.ws, self.samples[index].id)
        if payload is None:
            raise ValueError(f"tagging result missing: {self.samples[index].id}")
        sample = self.samples[index]
        return result_from_dict(
            {
                **payload,
                "effective_tags": list(sample.labels.get("tags") or []),
                "eval_status": sample.labels.get("tagging", {}).get("eval_status")
                or payload.get("eval_status"),
            }
        )
