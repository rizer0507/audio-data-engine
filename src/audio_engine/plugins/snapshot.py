"""Recursive definition snapshots and implementation digests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sha256_obj(obj: Any) -> str:
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)
    return sha256_text(payload)


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def _resolve_rel(path: str | Path, *, base: Path, project_root: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute() and candidate.exists():
        return candidate.resolve()
    for root in (base, project_root, Path.cwd()):
        joined = (root / candidate).resolve()
        if joined.exists():
            return joined
    return (project_root / candidate).resolve()


def freeze_yaml_tree(
    entry: Path,
    *,
    project_root: Path,
    visited: set[str] | None = None,
) -> dict[str, Any]:
    """Recursively snapshot a YAML file and referenced local YAML/config paths."""

    visited = visited if visited is not None else set()
    resolved = entry.resolve()
    key = str(resolved)
    if key in visited:
        return {"path": key, "cycle": True, "content_digest": sha256_file(resolved)}
    visited.add(key)

    raw_text = resolved.read_text(encoding="utf-8")
    try:
        data = yaml.safe_load(raw_text)
    except Exception as exc:  # noqa: BLE001
        return {
            "path": _rel(resolved, project_root),
            "absolute_path": key,
            "content_digest": sha256_text(raw_text),
            "parse_error": str(exc),
            "text": raw_text,
            "children": {},
        }

    children: dict[str, Any] = {}
    if isinstance(data, dict):
        for ref_path in _collect_path_refs(data):
            child = _resolve_rel(ref_path, base=resolved.parent, project_root=project_root)
            if not child.exists() or not child.is_file():
                children[ref_path] = {"path": ref_path, "missing": True}
                continue
            if child.suffix.lower() in {".yaml", ".yml", ".json"}:
                children[ref_path] = freeze_yaml_tree(
                    child, project_root=project_root, visited=visited
                )
            else:
                children[ref_path] = {
                    "path": _rel(child, project_root),
                    "absolute_path": str(child),
                    "content_digest": sha256_file(child),
                    "bytes": child.stat().st_size,
                }

    return {
        "path": _rel(resolved, project_root),
        "absolute_path": key,
        "content_digest": sha256_text(raw_text),
        "data": data,
        "children": children,
    }


def _rel(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _collect_path_refs(node: Any) -> list[str]:
    refs: list[str] = []
    path_keys = {
        "pipeline",
        "config",
        "config_path",
        "definition",
        "path",
        "manifest",
        "normalization_path",
        "blank_exact_hotwords_path",
        "evidence_config",
        "voicemail_patterns_path",
        "silero_vad_config",
        "audio_energy_config",
        "dnsmos_config",
        "tagging_config",
        "fragments",
        "runtime_config",
    }

    def walk(n: Any, key: str | None = None) -> None:
        if isinstance(n, dict):
            for k, v in n.items():
                walk(v, str(k))
        elif isinstance(n, list):
            for item in n:
                walk(item, key)
        elif isinstance(n, str) and key:
            text = n.strip()
            if not text:
                return
            if key in path_keys or key.endswith("_path") or key.endswith("_config"):
                if any(text.endswith(ext) for ext in (".yaml", ".yml", ".json", ".parquet")):
                    refs.append(text)
            if key == "lexicons" or key == "library":
                if text.endswith((".yaml", ".yml")) or "/" in text or "\\" in text:
                    refs.append(text)

    walk(node)
    # de-dupe preserving order
    seen: set[str] = set()
    out: list[str] = []
    for item in refs:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def digest_implementation_files(paths: list[Path]) -> str:
    """Digest workspace implementation files (detects uncommitted edits)."""

    parts: list[dict[str, str]] = []
    for path in sorted(paths, key=lambda p: str(p)):
        if not path.exists() or not path.is_file():
            parts.append({"path": str(path), "missing": "1"})
            continue
        parts.append(
            {
                "path": str(path.resolve()),
                "digest": sha256_file(path),
                "mtime_ns": str(path.stat().st_mtime_ns),
                "size": str(path.stat().st_size),
            }
        )
    return sha256_obj(parts)


def write_snapshot_dir(snapshot: dict[str, Any], target: Path) -> str:
    target.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True, default=str)
    out = target / "definition_snapshot.json"
    out.write_text(payload, encoding="utf-8")
    digest = sha256_text(payload)
    (target / "definition_snapshot.digest").write_text(digest + "\n", encoding="utf-8")
    return digest


def materialize_frozen_files(snapshot: dict[str, Any], target: Path) -> None:
    """Write recursive YAML text copies under target for offline execution."""

    target.mkdir(parents=True, exist_ok=True)

    def walk(node: dict[str, Any]) -> None:
        rel = node.get("path")
        data = node.get("data")
        if rel and data is not None and not node.get("cycle") and not node.get("missing"):
            dest = target / Path(str(rel))
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open("w", encoding="utf-8") as fh:
                yaml.safe_dump(data, fh, allow_unicode=True, sort_keys=False)
        for child in (node.get("children") or {}).values():
            if isinstance(child, dict):
                walk(child)

    walk(snapshot)
