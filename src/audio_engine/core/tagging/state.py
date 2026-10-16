"""Tag sources, manual ops, effective tags, reconcile (040 §6)."""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from audio_engine.core.tagging.types import (
    SampleEvalStatus,
    TagSource,
    Ternary,
)

SOURCE_AUTO = "auto_rule"
SOURCE_MANUAL_ADD = "manual_add"
SOURCE_MANUAL_BLOCK = "manual_block"


@dataclass
class SampleTagState:
    sample_id: str
    audio_sha256: str = ""
    sources: list[TagSource] = field(default_factory=list)
    version: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "audio_sha256": self.audio_sha256,
            "sources": [s.as_dict() for s in self.sources],
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SampleTagState:
        sources = [
            TagSource(
                tag_id=str(s.get("tag_id") or ""),
                source_type=str(s.get("source_type") or ""),
                rule_id=s.get("rule_id"),
                rule_version=s.get("rule_version"),
                snapshot_id=s.get("snapshot_id"),
                stale=bool(s.get("stale")),
                details=dict(s.get("details") or {}),
            )
            for s in (data.get("sources") or [])
            if isinstance(s, dict)
        ]
        return cls(
            sample_id=str(data.get("sample_id") or ""),
            audio_sha256=str(data.get("audio_sha256") or ""),
            sources=sources,
            version=int(data.get("version") or 0),
        )


def effective_tags(state: SampleTagState, *, enabled_tag_ids: set[str] | None = None) -> list[str]:
    """有效标签 =（自动命中 ∪ 人工添加）－ 人工屏蔽；同 tag 去重。"""
    auto: set[str] = set()
    manual_add: set[str] = set()
    blocked: set[str] = set()
    for src in state.sources:
        if src.stale:
            continue
        if enabled_tag_ids is not None and src.tag_id not in enabled_tag_ids:
            if src.source_type != SOURCE_MANUAL_BLOCK:
                continue
        if src.source_type == SOURCE_AUTO:
            auto.add(src.tag_id)
        elif src.source_type == SOURCE_MANUAL_ADD:
            manual_add.add(src.tag_id)
        elif src.source_type == SOURCE_MANUAL_BLOCK:
            blocked.add(src.tag_id)
    enabled = enabled_tag_ids
    result = (auto | manual_add) - blocked
    if enabled is not None:
        result = {t for t in result if t in enabled}
    return sorted(result)


def sample_eval_status(
    *,
    effective: list[str],
    rule_ternaries: dict[str, Ternary],
    has_enabled_rules: bool,
    has_stale: bool = False,
) -> SampleEvalStatus:
    if effective:
        return SampleEvalStatus.TAGGED
    if not has_enabled_rules and not effective:
        return SampleEvalStatus.NOT_EVALUATED
    unknowns = [t for t in rule_ternaries.values() if t == Ternary.UNKNOWN]
    if unknowns or has_stale:
        return SampleEvalStatus.UNRESOLVED
    return SampleEvalStatus.UNTAGGED


@dataclass
class ManualOpRecord:
    op_id: str
    op: str
    sample_id: str
    tag_id: str
    batch_id: str
    audio_sha256: str
    actor: str
    reason: str
    ts: float
    before: dict[str, Any]
    after: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "op_id": self.op_id,
            "op": self.op,
            "sample_id": self.sample_id,
            "tag_id": self.tag_id,
            "batch_id": self.batch_id,
            "audio_sha256": self.audio_sha256,
            "actor": self.actor,
            "reason": self.reason,
            "ts": self.ts,
            "before": self.before,
            "after": self.after,
        }


class TagStateStore:
    """File-backed per-batch tag source store with simple lock for concurrency."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._indexed_store = None
        self._lock = threading.RLock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write({})

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _write(self, data: dict[str, Any]) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.path)

    def get(self, sample_id: str) -> SampleTagState:
        with self._lock:
            if self.path.with_suffix(".sqlite3").exists():
                return self.enable_indexed().get(sample_id)
            data = self._read()
            raw = data.get(sample_id)
            if not isinstance(raw, dict):
                return SampleTagState(sample_id=sample_id)
            return SampleTagState.from_dict(raw)

    def put(self, state: SampleTagState, *, expected_version: int | None = None) -> SampleTagState:
        with self._lock:
            if self.path.with_suffix(".sqlite3").exists():
                indexed = self.enable_indexed()
                if expected_version is not None and indexed.get(state.sample_id).version != expected_version:
                    raise ConcurrentModificationError(f"sample {state.sample_id} version conflict")
                indexed.put_many([state], check_versions=expected_version is not None)
                indexed.export_snapshot()
                return state
            data = self._read()
            current = data.get(state.sample_id)
            if expected_version is not None and isinstance(current, dict):
                if int(current.get("version") or 0) != expected_version:
                    raise ConcurrentModificationError(
                        f"sample {state.sample_id} version conflict: "
                        f"expected {expected_version}, got {current.get('version')}"
                    )
            state.version = int(state.version) + 1
            data[state.sample_id] = state.as_dict()
            self._write(data)
            return state

    def all_states(self) -> dict[str, SampleTagState]:
        with self._lock:
            if self.path.with_suffix(".sqlite3").exists():
                return self.enable_indexed().all_states()
            data = self._read()
            return {
                sid: SampleTagState.from_dict(raw)
                for sid, raw in data.items()
                if isinstance(raw, dict)
            }

    def put_many(self, states: list[SampleTagState]) -> None:
        """Commit a checkpoint in one atomic rewrite instead of one per sample."""
        if not states:
            return
        with self._lock:
            if self.path.with_suffix(".sqlite3").exists():
                self.enable_indexed().put_many(states)
                return
            data = self._read()
            for state in states:
                current_version = int((data.get(state.sample_id) or {}).get("version") or 0)
                if current_version != state.version:
                    raise ConcurrentModificationError(
                        f"sample {state.sample_id} version conflict: "
                        f"expected {state.version}, got {current_version}"
                    )
            for state in states:
                state.version += 1
                data[state.sample_id] = state.as_dict()
            self._write(data)

    def enable_indexed(self):
        from audio_engine.core.tagging.indexed_state import IndexedTagStateStore
        if self._indexed_store is None:
            self._indexed_store = IndexedTagStateStore(self.path)
        return self._indexed_store

    def get_many(self, sample_ids) -> dict[str, SampleTagState]:
        with self._lock:
            if self.path.with_suffix(".sqlite3").exists():
                return self.enable_indexed().get_many(sample_ids)
            data = self._read()
            return {sid: SampleTagState.from_dict(data[sid]) for sid in sample_ids
                    if isinstance(data.get(sid), dict)}

    def close(self):
        if self._indexed_store is not None:
            self._indexed_store.close()
            self._indexed_store = None


class ConcurrentModificationError(RuntimeError):
    pass


class ManualOpLog:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def append(self, record: ManualOpRecord) -> None:
        with self._lock, self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record.as_dict(), ensure_ascii=False) + "\n")


def reconcile_auto_sources(
    state: SampleTagState,
    *,
    matched_rules: list[tuple[str, str, str]],  # rule_id, rule_version, tag_id
    snapshot_id: str,
    scope_rule_ids: set[str] | None = None,
    mark_stale_outside: bool = False,
    executed_rule_ids: set[str] | None = None,
) -> SampleTagState:
    """
    Replace auto sources for executed rules with new matched set.
    Local runs must not delete sources for rules outside scope.
    Rules that became unknown should not keep confirmed auto sources.
    """
    executed = executed_rule_ids if executed_rule_ids is not None else (
        {r[0] for r in matched_rules} | (scope_rule_ids or set())
    )
    kept: list[TagSource] = []
    for src in state.sources:
        if src.source_type != SOURCE_AUTO:
            kept.append(src)
            continue
        rid = str(src.rule_id or "")
        if scope_rule_ids is not None and rid not in scope_rule_ids:
            # Outside local scope — keep (optionally mark stale if versions drift).
            if mark_stale_outside:
                src = TagSource(
                    tag_id=src.tag_id,
                    source_type=src.source_type,
                    rule_id=src.rule_id,
                    rule_version=src.rule_version,
                    snapshot_id=src.snapshot_id,
                    stale=True,
                    details=dict(src.details),
                )
            kept.append(src)
            continue
        if rid in executed:
            # Drop old auto for executed rules; re-add matched below.
            continue
        kept.append(src)

    for rule_id, rule_version, tag_id in matched_rules:
        kept.append(
            TagSource(
                tag_id=tag_id,
                source_type=SOURCE_AUTO,
                rule_id=rule_id,
                rule_version=rule_version,
                snapshot_id=snapshot_id,
                stale=False,
            )
        )
    state.sources = kept
    return state


def apply_manual_add(
    state: SampleTagState,
    *,
    tag_id: str,
    audio_sha256: str,
) -> SampleTagState:
    _assert_audio(state, audio_sha256)
    # Re-add after block: remove block first.
    state.sources = [
        s
        for s in state.sources
        if not (s.source_type == SOURCE_MANUAL_BLOCK and s.tag_id == tag_id)
    ]
    if any(s.source_type == SOURCE_MANUAL_ADD and s.tag_id == tag_id and not s.stale for s in state.sources):
        return state
    state.sources.append(TagSource(tag_id=tag_id, source_type=SOURCE_MANUAL_ADD))
    return state


def apply_manual_remove(
    state: SampleTagState,
    *,
    tag_id: str,
    audio_sha256: str,
) -> SampleTagState:
    """移除人工添加并创建持续屏蔽。"""
    _assert_audio(state, audio_sha256)
    state.sources = [
        s
        for s in state.sources
        if not (s.source_type == SOURCE_MANUAL_ADD and s.tag_id == tag_id)
    ]
    if not any(s.source_type == SOURCE_MANUAL_BLOCK and s.tag_id == tag_id and not s.stale for s in state.sources):
        state.sources.append(TagSource(tag_id=tag_id, source_type=SOURCE_MANUAL_BLOCK))
    return state


def apply_retract_manual(
    state: SampleTagState,
    *,
    tag_id: str,
    audio_sha256: str,
) -> SampleTagState:
    """只撤销人工添加，不创建屏蔽。"""
    _assert_audio(state, audio_sha256)
    state.sources = [
        s
        for s in state.sources
        if not (s.source_type == SOURCE_MANUAL_ADD and s.tag_id == tag_id)
    ]
    return state


def apply_restore_auto(
    state: SampleTagState,
    *,
    tag_id: str,
    audio_sha256: str,
) -> SampleTagState:
    """撤销屏蔽，按当前自动结果重新决定显示。"""
    _assert_audio(state, audio_sha256)
    state.sources = [
        s
        for s in state.sources
        if not (s.source_type == SOURCE_MANUAL_BLOCK and s.tag_id == tag_id)
    ]
    return state


def _assert_audio(state: SampleTagState, audio_sha256: str) -> None:
    if state.audio_sha256 and audio_sha256 and state.audio_sha256 != audio_sha256:
        raise ValueError(
            f"audio identity mismatch for {state.sample_id}: "
            f"stored={state.audio_sha256} got={audio_sha256}"
        )
    if audio_sha256 and not state.audio_sha256:
        state.audio_sha256 = audio_sha256


def new_op_id() -> str:
    return uuid.uuid4().hex


def now_ts() -> float:
    return time.time()
