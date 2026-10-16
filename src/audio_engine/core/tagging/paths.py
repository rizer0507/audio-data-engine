"""Stable path resolution for tagging configs (042).

Resolution order for relative paths (does not depend on accidental CWD alone):
1. Absolute path as-is if it exists
2. ``base_dir / path`` (usually the run-config directory)
3. ``repo_root / path`` (repo root discovered via configs/tagging + src/audio_engine)
4. ``cwd / path`` as last resort

Errors always include the attempted candidates.
"""

from __future__ import annotations

from pathlib import Path


def find_repo_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    for p in [start, *start.parents]:
        if (p / "configs" / "tagging").is_dir() and (p / "src" / "audio_engine").is_dir():
            return p
    return start


def resolve_tagging_path(
    path: str | Path,
    *,
    base_dir: Path | None = None,
    repo_root: Path | None = None,
    must_exist: bool = True,
) -> Path:
    raw = Path(str(path))
    root = (repo_root or find_repo_root(base_dir)).resolve()
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        if base_dir is not None:
            candidates.append((base_dir / raw).resolve())
        candidates.append((root / raw).resolve())
        candidates.append((Path.cwd() / raw).resolve())

    seen: set[str] = set()
    uniq: list[Path] = []
    for c in candidates:
        key = str(c)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)

    for c in uniq:
        if c.exists():
            return c
    if not must_exist and uniq:
        return uniq[0]
    tried = ", ".join(str(c) for c in uniq)
    raise FileNotFoundError(f"tagging path not found: {path} (tried: {tried})")
