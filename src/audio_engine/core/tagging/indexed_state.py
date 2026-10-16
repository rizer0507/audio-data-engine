"""Indexed tag states; legacy JSON remains an interoperable end-of-run snapshot."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from audio_engine.core.artifacts import atomic_path
from audio_engine.core.tagging.state import ConcurrentModificationError, SampleTagState


class IndexedTagStateStore:
    def __init__(self, path: Path):
        self.path = path
        self.index_path = path.with_suffix(".sqlite3")
        self.db = sqlite3.connect(self.index_path, timeout=30)
        # Bootstrap under the same database lock as writers. Two new runs must
        # not each publish an old JSON snapshot over the other's committed rows.
        self.db.execute("BEGIN IMMEDIATE")
        try:
            self.db.execute(
                "CREATE TABLE IF NOT EXISTS states (sample_id TEXT PRIMARY KEY, version INTEGER NOT NULL, payload TEXT NOT NULL)"
            )
            self.db.execute("CREATE TABLE IF NOT EXISTS metadata (revision INTEGER NOT NULL)")
            if self.db.execute("SELECT revision FROM metadata").fetchone() is None:
                existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
                self.db.executemany(
                    "INSERT INTO states VALUES (?, ?, ?)",
                    (
                        (sid, int(raw.get("version") or 0), json.dumps(raw, ensure_ascii=False))
                        for sid, raw in existing.items()
                        if isinstance(raw, dict)
                    ),
                )
                self.db.execute("INSERT INTO metadata VALUES (0)")
            self.db.commit()
        except BaseException:
            self.db.rollback()
            self.db.close()
            raise

    def get_many(self, sample_ids) -> dict[str, SampleTagState]:
        ids = list(sample_ids)
        output = {}
        for start in range(0, len(ids), 900):
            block = ids[start : start + 900]
            marks = ",".join("?" for _ in block)
            for sid, raw in self.db.execute(
                f"SELECT sample_id, payload FROM states WHERE sample_id IN ({marks})", block
            ):
                output[sid] = SampleTagState.from_dict(json.loads(raw))
        return output

    def get(self, sample_id):
        return self.get_many([sample_id]).get(sample_id) or SampleTagState(sample_id)

    def put_many(self, states, *, check_versions=True):
        if not states:
            return
        self.db.execute("BEGIN IMMEDIATE")
        try:
            current = self.get_many(s.sample_id for s in states)
            for state in states:
                version = current.get(state.sample_id, SampleTagState(state.sample_id)).version
                if check_versions and version != state.version:
                    raise ConcurrentModificationError(
                        f"sample {state.sample_id} version conflict: expected {state.version}, got {version}"
                    )
            self.db.executemany(
                "INSERT OR REPLACE INTO states VALUES (?, ?, ?)",
                (
                    (
                        s.sample_id,
                        s.version + 1,
                        json.dumps({**s.as_dict(), "version": s.version + 1}, ensure_ascii=False),
                    )
                    for s in states
                ),
            )
            self.db.execute("UPDATE metadata SET revision=revision+1")
            self.db.commit()
        except BaseException:
            self.db.rollback()
            raise
        for state in states:
            state.version += 1

    def all_states(self):
        return {
            sid: SampleTagState.from_dict(json.loads(raw))
            for sid, raw in self.db.execute("SELECT sample_id, payload FROM states")
        }

    def export_snapshot(self):
        with atomic_path(self.path) as tmp, tmp.open("w", encoding="utf-8") as stream:
            stream.write("{")
            separator = ""
            for sid, raw in self.db.execute(
                "SELECT sample_id, payload FROM states ORDER BY sample_id"
            ):
                stream.write(separator + json.dumps(sid, ensure_ascii=False) + ":" + raw)
                separator = ",\n"
            stream.write("}\n")

    def close(self):
        self.db.close()
