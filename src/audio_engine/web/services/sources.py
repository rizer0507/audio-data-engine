"""Source registry backed solely by resources/manifest.yaml."""

from __future__ import annotations

import os
import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from audio_engine.core.source_naming import validate_source_name

try:
    import msvcrt
except ImportError:  # pragma: no cover
    msvcrt = None  # type: ignore

try:
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore


class DuplicateKeyLoader(yaml.SafeLoader):
    pass


def _no_duplicates(loader: yaml.SafeLoader, node: yaml.nodes.MappingNode, deep: bool = False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            mark = key_node.start_mark
            raise yaml.constructor.ConstructorError(
                None,
                None,
                f"duplicate key {key!r} at line {mark.line + 1} column {mark.column + 1}",
                mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


DuplicateKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _no_duplicates,
)


class SourceRegistryError(Exception):
    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}


def _lock_file(fh) -> None:
    if msvcrt is not None:
        msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
    elif fcntl is not None:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)


def _unlock_file(fh) -> None:
    if msvcrt is not None:
        try:
            msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
    elif fcntl is not None:
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def load_sources_strict(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load sources with duplicate-key detection.

    If the file currently has duplicate top-level `sources` keys (known issue),
    parse with a recovery path that surfaces both lists instead of silent drop.
    """

    text = path.read_text(encoding="utf-8") if path.is_file() else "sources: []\n"
    try:
        data = yaml.load(text, Loader=DuplicateKeyLoader)
        if not isinstance(data, dict):
            raise SourceRegistryError("SOURCES_INVALID", "manifest root must be a mapping")
        sources = data.get("sources") or []
        if not isinstance(sources, list):
            raise SourceRegistryError("SOURCES_INVALID", "sources must be a list")
        return list(sources), data
    except yaml.constructor.ConstructorError as exc:
        # Recovery: manually split duplicate sources sequences for migration.
        return _recover_duplicate_sources(path, text, str(exc))


def _recover_duplicate_sources(
    path: Path, text: str, error: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Collect all `sources:` blocks; merge by source_id with conflict detection."""

    # Backup once beside the file.
    backup = path.with_suffix(path.suffix + ".bak_duplicate_sources")
    if path.is_file() and not backup.exists():
        shutil.copy2(path, backup)

    # Use unsafe multi-doc approach: rewrite duplicate keys to sources__N then load.
    lines = text.splitlines(keepends=True)
    count = 0
    rewritten: list[str] = []
    for line in lines:
        if line.startswith("sources:") or line.startswith("sources :"):
            if count == 0:
                rewritten.append(line)
            else:
                rewritten.append(line.replace("sources", f"sources__dup{count}", 1))
            count += 1
        else:
            rewritten.append(line)
    data = yaml.safe_load("".join(rewritten)) or {}
    if not isinstance(data, dict):
        raise SourceRegistryError("SOURCES_INVALID", "manifest root must be a mapping", {"parse_error": error})

    lists: list[list[dict[str, Any]]] = []
    primary = data.get("sources") or []
    if isinstance(primary, list):
        lists.append(primary)
    for key, value in data.items():
        if str(key).startswith("sources__dup") and isinstance(value, list):
            lists.append(value)

    merged: dict[str, dict[str, Any]] = {}
    conflicts: list[dict[str, Any]] = []
    for block in lists:
        for item in block:
            if not isinstance(item, dict):
                continue
            sid = str(item.get("source_id") or "").strip()
            if not sid:
                continue
            if sid in merged and merged[sid] != item:
                conflicts.append({"source_id": sid, "a": merged[sid], "b": item})
            else:
                merged[sid] = item

    if conflicts:
        raise SourceRegistryError(
            "SOURCES_ID_CONFLICT",
            "duplicate sources key contained conflicting source_id entries; backup kept",
            {"conflicts": conflicts, "backup": str(backup), "parse_error": error},
        )

    normalized = {"sources": list(merged.values())}
    # Persist normalized form atomically so subsequent loads are strict-clean.
    atomic_write_yaml(path, normalized)
    return list(merged.values()), normalized


def atomic_write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with open(lock_path, "a+", encoding="utf-8") as lock_fh:
        _lock_file(lock_fh)
        try:
            fd, tmp_name = tempfile.mkstemp(
                prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    yaml.safe_dump(data, fh, allow_unicode=True, sort_keys=False)
                Path(tmp_name).replace(path)
            finally:
                if Path(tmp_name).exists():
                    Path(tmp_name).unlink(missing_ok=True)
        finally:
            _unlock_file(lock_fh)


def path_allowed(path: Path, allowed_roots: list[Path]) -> bool:
    resolved = path.resolve()
    if not allowed_roots:
        return True
    for root in allowed_roots:
        try:
            resolved.relative_to(root.resolve())
            return True
        except ValueError:
            continue
    return False


def new_source_id() -> str:
    return f"src_{uuid.uuid4().hex[:8]}"


def normalize_absolute_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/")


def try_validate_batch(value: str | None) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return validate_source_name(text)
    except ValueError:
        return None


def effective_batch(item: dict[str, Any]) -> str | None:
    """Resolve engine source_name from batch, with legacy fallbacks. Never invent."""

    if "batch" in item and item.get("batch") is not None:
        return try_validate_batch(str(item.get("batch")))
    # Legacy: source_name that already matches engine naming rules.
    legacy_name = try_validate_batch(str(item.get("source_name") or ""))
    if legacy_name:
        return legacy_name
    # Legacy hand-written IDs that were used as --source-name.
    return try_validate_batch(str(item.get("source_id") or ""))


def public_source(item: dict[str, Any]) -> dict[str, Any]:
    out = dict(item)
    batch = effective_batch(item)
    if batch is not None:
        out["batch"] = batch
    elif "batch" not in out:
        out["batch"] = item.get("batch")
    # Display name: prefer explicit name/source_name; fall back to batch.
    display = str(item.get("name") or item.get("source_name") or batch or item.get("source_id") or "")
    out["source_name"] = display
    out["name"] = display
    return out


class SourceService:
    def __init__(self, manifest_path: Path, allowed_roots: list[Path]) -> None:
        self.manifest_path = manifest_path
        self.allowed_roots = allowed_roots
        self._etag = 1

    def list_sources(self) -> dict[str, Any]:
        sources, _ = load_sources_strict(self.manifest_path)
        items = [public_source(s) for s in sources]
        return {"items": items, "next_cursor": None, "total": len(items), "revision": self._etag}

    def get(self, source_id: str) -> dict[str, Any]:
        sources, _ = load_sources_strict(self.manifest_path)
        for item in sources:
            if item.get("source_id") == source_id:
                return public_source(item)
        raise SourceRegistryError("SOURCE_NOT_FOUND", f"source_id not found: {source_id}")

    def create(self, payload: dict[str, Any]) -> dict[str, Any]:
        sources, data = load_sources_strict(self.manifest_path)
        try:
            batch = validate_source_name(str(payload.get("batch") or "").strip())
        except ValueError as exc:
            raise SourceRegistryError("SOURCE_BATCH_INVALID", str(exc)) from exc

        path = Path(str(payload.get("path") or ""))
        if not path.is_absolute():
            raise SourceRegistryError("SOURCE_PATH_NOT_ABSOLUTE", "path must be absolute server path")
        if not path_allowed(path, self.allowed_roots):
            raise SourceRegistryError(
                "SOURCE_PATH_NOT_ALLOWED",
                "path outside allowed_source_roots",
                {"path": str(path)},
            )
        norm_path = normalize_absolute_path(path)

        # Dedup by batch + normalized absolute path (same lock / write cycle).
        for existing in sources:
            existing_batch = effective_batch(existing)
            existing_path = normalize_absolute_path(Path(str(existing.get("path") or ""))) if existing.get("path") else ""
            if existing_batch == batch and existing_path == norm_path:
                return {**public_source(existing), "revision": self._etag, "reused": True}
            if existing_batch == batch and existing_path and existing_path != norm_path:
                raise SourceRegistryError(
                    "SOURCE_BATCH_PATH_CONFLICT",
                    "same batch already registered with a different path; edit existing or use another batch",
                    {
                        "batch": batch,
                        "existing_source_id": existing.get("source_id"),
                        "existing_path": existing.get("path"),
                        "requested_path": norm_path,
                    },
                )

        sid = new_source_id()
        while any(s.get("source_id") == sid for s in sources):
            sid = new_source_id()

        display = str(payload.get("name") or payload.get("source_name") or batch).strip() or batch
        item = {
            "source_id": sid,
            "batch": batch,
            "source_name": display,
            "path": norm_path,
            "format": payload.get("format") or "wav",
            "sample_rate": payload.get("sample_rate"),
            "bit_depth": payload.get("bit_depth"),
            "channels": payload.get("channels"),
            "note": payload.get("note") or "",
            "origin": payload.get("origin") or "",
            "ingested_at": datetime.now(timezone.utc).isoformat(),
            "file_count": None,
        }
        if item["format"] == "pcm":
            if not item["sample_rate"] or not item["bit_depth"] or not item["channels"]:
                raise SourceRegistryError(
                    "PCM_PARAMS_REQUIRED",
                    "pcm sources require sample_rate, bit_depth, channels",
                )
        sources.append(item)
        data["sources"] = sources
        atomic_write_yaml(self.manifest_path, data)
        self._etag += 1
        return {**public_source(item), "revision": self._etag, "reused": False}

    def patch(self, source_id: str, payload: dict[str, Any], *, expected_revision: int | None) -> dict[str, Any]:
        if expected_revision is not None and expected_revision != self._etag:
            raise SourceRegistryError(
                "REVISION_CONFLICT",
                "source registry revision mismatch; reload and retry",
                {"current_revision": self._etag},
            )
        sources, data = load_sources_strict(self.manifest_path)
        found = None
        for i, item in enumerate(sources):
            if item.get("source_id") == source_id:
                found = i
                break
        if found is None:
            raise SourceRegistryError("SOURCE_NOT_FOUND", f"source_id not found: {source_id}")
        item = dict(sources[found])
        # Internal ID never changes on patch.
        if payload.get("batch") is not None:
            try:
                item["batch"] = validate_source_name(str(payload["batch"]).strip())
            except ValueError as exc:
                raise SourceRegistryError("SOURCE_BATCH_INVALID", str(exc)) from exc
        display = payload.get("name")
        if display is None and payload.get("source_name") is not None:
            display = payload.get("source_name")
        if display is not None:
            item["source_name"] = str(display).strip() or item.get("batch") or source_id
        for key in ("format", "sample_rate", "bit_depth", "channels", "note", "origin"):
            if key in payload and payload[key] is not None:
                item[key] = payload[key]
        if payload.get("path") is not None:
            path = Path(str(payload["path"]))
            if not path.is_absolute():
                raise SourceRegistryError("SOURCE_PATH_NOT_ABSOLUTE", "path must be absolute")
            if not path_allowed(path, self.allowed_roots):
                raise SourceRegistryError("SOURCE_PATH_NOT_ALLOWED", "path outside allowed roots")
            item["path"] = normalize_absolute_path(path)

        new_batch = effective_batch(item)
        new_path = normalize_absolute_path(Path(str(item.get("path") or ""))) if item.get("path") else ""
        for other in sources:
            if other.get("source_id") == source_id:
                continue
            other_batch = effective_batch(other)
            other_path = normalize_absolute_path(Path(str(other.get("path") or ""))) if other.get("path") else ""
            if new_batch and other_batch == new_batch and other_path != new_path:
                raise SourceRegistryError(
                    "SOURCE_BATCH_PATH_CONFLICT",
                    "same batch already registered with a different path",
                    {"batch": new_batch, "existing_source_id": other.get("source_id")},
                )

        sources[found] = item
        data["sources"] = sources
        atomic_write_yaml(self.manifest_path, data)
        self._etag += 1
        return {**public_source(item), "revision": self._etag}

    def check(self, source_id: str, *, sample_limit: int = 5) -> dict[str, Any]:
        item = self.get(source_id)
        path = Path(item["path"])
        result: dict[str, Any] = {
            "source_id": source_id,
            "path": str(path),
            "exists": path.exists(),
            "readable": False,
            "sample_file_count": 0,
            "sampled_files": [],
            "full_scan": False,
            "status": "unchecked",
            "checked_at": datetime.now(timezone.utc).isoformat(),
        }
        if not path.exists():
            result["status"] = "check_failed"
            result["error"] = "path does not exist"
            return result
        if not os.access(path, os.R_OK):
            result["status"] = "check_failed"
            result["error"] = "path not readable"
            return result
        result["readable"] = True
        files: list[str] = []
        if path.is_file():
            files = [str(path)]
        elif path.is_dir():
            for p in sorted(path.rglob("*")):
                if p.is_file():
                    files.append(str(p))
                if len(files) >= sample_limit:
                    break
        result["sample_file_count"] = len(files)
        result["sampled_files"] = files
        result["status"] = "sample_ok" if files else "sample_empty"
        result["note"] = "抽样可读，非全量扫描结论"
        return result
