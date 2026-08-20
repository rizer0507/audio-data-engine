from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from audio_engine.core.sample import Sample


def _temp_sibling(path: Path) -> Path:
    """Temp path in the same directory, keeping the suffix so writers can infer format."""
    unique = f"{os.getpid()}.{uuid.uuid4().hex}"
    return path.with_name(f"{path.stem}.{unique}.tmp{path.suffix}")


@contextmanager
def atomic_path(path: Path) -> Iterator[Path]:
    """Yield a temp path and publish it to `path` only after the block succeeds.

    Concurrent readers therefore see either the previous complete file or the new
    complete file, never a partially written one.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _temp_sibling(path)
    try:
        yield tmp
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    if not tmp.exists():
        raise FileNotFoundError(f"atomic_path: nothing was written to {tmp}")
    os.replace(tmp, path)


def atomic_write_json(path: Path, data: Any) -> None:
    with atomic_path(Path(path)) as tmp:
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def sample_digest(sample: Sample) -> str:
    """Content hash used to build collision-free derived paths."""
    if sample.sha256:
        return sample.sha256
    return hashlib.sha256(sample.source_path.encode("utf-8")).hexdigest()


def derived_audio_path(
    output_dir: Path,
    subdir: str,
    sample: Sample,
    *,
    stem_suffix: str = "",
    ext: str = ".wav",
) -> Path:
    """`<output_dir>/<subdir>/<digest[:2]>/<digest[:16]>_<id><stem_suffix><ext>`.

    The digest prevents same-named inputs from overwriting each other, and the
    two-char fan-out keeps directories small at 100k+ samples.
    """
    digest = sample_digest(sample)
    path = Path(output_dir) / subdir / digest[:2] / f"{digest[:16]}_{sample.id}{stem_suffix}{ext}"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
