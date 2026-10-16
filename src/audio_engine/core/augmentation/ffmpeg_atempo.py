"""FFmpeg atempo backend: pitch-preserving tempo change."""

from __future__ import annotations

import re
import shutil
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from audio_engine.core.artifacts import atomic_path
from audio_engine.core.augmentation.params import TempoParams, normalize_speed_factor

_ATEMPO_MAX = 2.0
_ATEMPO_MIN = 0.5
_VERSION_RE = re.compile(r"ffmpeg version\s+(\S+)", re.IGNORECASE)
_FILTER_RE = re.compile(r"\batempo\b", re.IGNORECASE)

# Soft cap on concurrent FFmpeg child processes (shared across workers).
_FFMPEG_GATE: threading.BoundedSemaphore | None = None
_FFMPEG_GATE_SIZE = 0
_FFMPEG_GATE_LOCK = threading.Lock()


def _ensure_gate(limit: int) -> threading.BoundedSemaphore:
    global _FFMPEG_GATE, _FFMPEG_GATE_SIZE
    with _FFMPEG_GATE_LOCK:
        if _FFMPEG_GATE is None or _FFMPEG_GATE_SIZE != limit:
            _FFMPEG_GATE = threading.BoundedSemaphore(limit)
            _FFMPEG_GATE_SIZE = limit
        return _FFMPEG_GATE


@dataclass(frozen=True)
class FFmpegBackendInfo:
    binary: str
    version: str
    atempo_available: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "binary": self.binary,
            "version": self.version,
            "atempo_available": self.atempo_available,
        }


def decompose_atempo(factor: float) -> list[float]:
    """Split factor into atempo stages within [0.5, 2.0] whose product equals factor."""
    value = normalize_speed_factor(factor)
    stages: list[float] = []
    remaining = value
    # Only acceleration (>1) in first version; keep loop general for chaining.
    while remaining > _ATEMPO_MAX + 1e-12:
        stages.append(_ATEMPO_MAX)
        remaining /= _ATEMPO_MAX
    if remaining < _ATEMPO_MIN - 1e-12:
        raise ValueError(f"cannot decompose speed_factor={value} into atempo stages")
    # Avoid tiny residual drift from float division.
    stages.append(float(f"{remaining:.12g}"))
    product = 1.0
    for stage in stages:
        product *= stage
    if abs(product - value) / value > 1e-6:
        raise ValueError(
            f"atempo stage product {product} diverges from speed_factor {value}"
        )
    return stages


def atempo_filter_graph(factor: float) -> str:
    stages = decompose_atempo(factor)
    return ",".join(f"atempo={stage:.12g}" for stage in stages)


def probe_ffmpeg(ffmpeg_bin: str = "ffmpeg") -> FFmpegBackendInfo:
    binary = shutil.which(ffmpeg_bin) or ffmpeg_bin
    path = Path(binary)
    if not path.is_file() and shutil.which(ffmpeg_bin) is None:
        raise FileNotFoundError(
            f"FFmpeg not found ({ffmpeg_bin!r}). Install FFmpeg and ensure it is on PATH, "
            "or set params.ffmpeg_bin to the absolute executable path."
        )
    try:
        version_proc = subprocess.run(
            [binary, "-version"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except OSError as exc:
        raise FileNotFoundError(
            f"failed to execute FFmpeg ({binary}): {exc}"
        ) from exc
    version_text = (version_proc.stdout or "") + "\n" + (version_proc.stderr or "")
    match = _VERSION_RE.search(version_text)
    version = match.group(1) if match else "unknown"

    filters_proc = subprocess.run(
        [binary, "-hide_banner", "-filters"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    filters_text = (filters_proc.stdout or "") + "\n" + (filters_proc.stderr or "")
    if not _FILTER_RE.search(filters_text):
        raise RuntimeError(
            f"FFmpeg at {binary} does not expose the atempo filter; "
            "install a build with libavfilter atempo enabled."
        )
    return FFmpegBackendInfo(binary=str(Path(binary)), version=version, atempo_available=True)


def run_atempo(
    *,
    input_path: Path,
    output_path: Path,
    params: TempoParams,
    backend: FFmpegBackendInfo | None = None,
    gate_limit: int | None = None,
) -> dict[str, Any]:
    """Run FFmpeg atempo with atomic output. Returns filter/process metadata."""
    info = backend or probe_ffmpeg(params.ffmpeg_bin)
    filter_graph = atempo_filter_graph(params.speed_factor)
    threads = max(1, int(params.backend_threads))
    command = [
        info.binary,
        "-hide_banner",
        "-nostdin",
        "-y",
        "-threads",
        str(threads),
        "-i",
        str(input_path),
        "-filter:a",
        filter_graph,
        "-acodec",
        "pcm_s16le",
        "-f",
        "wav",
        # Placeholder filled inside atomic_path
        "OUTPUT",
    ]
    limit = gate_limit if gate_limit is not None else max(1, int(params.max_in_flight))
    gate = _ensure_gate(limit)
    stderr_tail = ""
    with gate:
        with atomic_path(output_path) as tmp:
            cmd = list(command)
            cmd[-1] = str(tmp)
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=float(params.timeout_seconds),
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise TimeoutError(
                    f"FFmpeg atempo timed out after {params.timeout_seconds}s "
                    f"for {input_path}"
                ) from exc
            stderr_tail = (proc.stderr or "")[-4000:]
            if proc.returncode != 0:
                raise RuntimeError(
                    f"FFmpeg atempo failed (exit {proc.returncode}) for {input_path}: "
                    f"{stderr_tail or proc.stdout}"
                )
            if not tmp.is_file() or tmp.stat().st_size <= 0:
                raise RuntimeError(f"FFmpeg produced empty output for {input_path}")
    return {
        "filter_graph": filter_graph,
        "stages": decompose_atempo(params.speed_factor),
        "ffmpeg_version": info.version,
        "ffmpeg_bin": info.binary,
        "command": command[:-1] + [str(output_path)],
        "stderr_tail": stderr_tail,
    }
