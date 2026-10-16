"""Streaming FSMN VAD adapter (FunASR) with injectable mock for tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np

from audio_engine.core.fsmn_vad.pairing import pair_and_filter
from audio_engine.core.fsmn_vad.runtime import validate_runtime_model_dir
from audio_engine.core.fsmn_vad.types import (
    CHUNK_MS,
    CHUNK_SAMPLES,
    SR,
    VadStreamResult,
)


class VadModel(Protocol):
    def generate(self, **kwargs: Any) -> Any: ...


GenerateFn = Callable[..., Any]


def pcm16le_to_float32(pcm16le: bytes) -> np.ndarray:
    """Source formula: int16.astype(float32) / 32767 (iinfo.max)."""
    if len(pcm16le) % 2:
        raise ValueError("PCM16 byte length must be even")
    samples = np.frombuffer(pcm16le, dtype="<i2")
    return samples.astype(np.float32) / np.iinfo(np.int16).max


def run_streaming_vad(
    vad: VadModel | GenerateFn,
    pcm16le: bytes,
    *,
    chunk_samples: int = CHUNK_SAMPLES,
    chunk_ms: int = CHUNK_MS,
    generate_options: dict[str, Any] | None = None,
    include_eof_fallback_in_kept: bool = False,
    sample_rate: int = SR,
) -> VadStreamResult:
    """Feed normalized PCM16 in 200 ms chunks; last non-empty chunk is final."""
    if len(pcm16le) % 2:
        raise ValueError("PCM16 byte length must be even")
    samples = np.frombuffer(pcm16le, dtype="<i2")
    total = len(samples)
    if total == 0:
        return VadStreamResult(status="empty_audio", event_log=[])

    options = dict(generate_options or {})
    reserved = {"input", "cache", "is_final", "chunk_size"}
    if reserved.intersection(options):
        raise ValueError("generate_options cannot replace stream arguments")

    cache: dict[str, Any] = {}
    event_log: list[dict[str, Any]] = []
    generate = vad.generate if hasattr(vad, "generate") else vad  # type: ignore[assignment]

    for left in range(0, total, chunk_samples):
        right = min(total, left + chunk_samples)
        final = right == total
        chunk = samples[left:right].astype(np.float32) / np.iinfo(np.int16).max
        result = generate(
            input=chunk,
            cache=cache,
            is_final=final,
            chunk_size=chunk_ms,
            **options,
        )
        events = result[0].get("value", []) if result else []
        event_log.append(
            {
                "input_start_sample": left,
                "input_end_sample": right,
                "is_final": final,
                "events": [list(pair) for pair in events],
            }
        )

    return pair_and_filter(
        event_log,
        pcm16le,
        total_samples=total,
        include_eof_fallback_in_kept=include_eof_fallback_in_kept,
        sample_rate=sample_rate,
    )


class MockVadModel:
    """Deterministic mock: emits scripted events keyed by chunk index or absolute time.

    ``script`` maps ``(input_start_sample, input_end_sample)`` or chunk index
    to a list of ``[start_ms, end_ms]`` events. Use ``by_chunk_index=True`` for
    simpler tests.
    """

    def __init__(
        self,
        script: dict[Any, list[list[float]]] | None = None,
        *,
        by_chunk_index: bool = True,
        default_events: list[list[float]] | None = None,
    ):
        self.script = script or {}
        self.by_chunk_index = by_chunk_index
        self.default_events = default_events or []
        self.calls: list[dict[str, Any]] = []
        self._chunk_i = 0

    def generate(self, **kwargs: Any) -> list[dict[str, Any]]:
        inp = kwargs.get("input")
        n = 0 if inp is None else len(inp)
        call = {
            "n_samples": n,
            "is_final": bool(kwargs.get("is_final")),
            "chunk_size": kwargs.get("chunk_size"),
            "cache_id": id(kwargs.get("cache")),
        }
        self.calls.append(call)

        if self.by_chunk_index:
            events = self.script.get(self._chunk_i, self.default_events)
        else:
            # Caller may pass absolute sample range via side channel — not used here.
            events = self.default_events
        self._chunk_i += 1
        return [{"value": [list(e) for e in events]}]

    def reset(self) -> None:
        self.calls.clear()
        self._chunk_i = 0


class ScheduleVadModel:
    """Emit events when cumulative input covers their start; complete ends at/after end_ms.

    Useful for multi-chunk streaming tests without hard-coding chunk indices.
    """

    def __init__(self, intervals: list[tuple[float, float]], *, sample_rate: int = SR):
        self.intervals = list(intervals)
        self.sample_rate = sample_rate
        self._emitted: set[int] = set()
        self._pending_starts: dict[int, float] = {}
        self.calls: list[dict[str, Any]] = []
        self._cum_samples = 0
        self._cache_ids: list[int] = []

    def generate(self, **kwargs: Any) -> list[dict[str, Any]]:
        inp = kwargs["input"]
        n = len(inp)
        left = self._cum_samples
        right = left + n
        self._cum_samples = right
        cache = kwargs.get("cache")
        self._cache_ids.append(id(cache))
        self.calls.append(
            {
                "left": left,
                "right": right,
                "is_final": bool(kwargs.get("is_final")),
                "cache_id": id(cache),
            }
        )

        events: list[list[float]] = []
        covered_ms = right * 1000.0 / self.sample_rate
        for idx, (start_ms, end_ms) in enumerate(self.intervals):
            if idx in self._emitted:
                continue
            # Split-style: emit start when covered, then end later.
            if idx not in self._pending_starts and covered_ms >= start_ms:
                # Prefer complete event if end also covered in same chunk.
                if covered_ms >= end_ms:
                    events.append([start_ms, end_ms])
                    self._emitted.add(idx)
                else:
                    events.append([start_ms, -1])
                    self._pending_starts[idx] = start_ms
            elif idx in self._pending_starts and covered_ms >= end_ms:
                events.append([-1, end_ms])
                self._emitted.add(idx)
                self._pending_starts.pop(idx, None)
        return [{"value": events}]


def load_funasr_vad(runtime_dir: str | Path, *, ncpu: int = 1, **kwargs: Any) -> Any:
    """Load CPU VAD with bounded intra-op threads; parallelism lives across sources."""
    if isinstance(ncpu, bool) or not isinstance(ncpu, int) or ncpu < 1:
        raise ValueError("FSMN VAD ncpu must be a positive integer")
    path = str(validate_runtime_model_dir(runtime_dir))
    if "model" in kwargs:
        raise ValueError("Use runtime_dir to select the local FSMN VAD model")
    try:
        from funasr import AutoModel
    except ImportError as exc:  # pragma: no cover - env dependent
        raise RuntimeError(
            "funasr is not installed; cannot load real FSMN VAD. "
            "Use mock=True for engineering tests."
        ) from exc

    params = {
        "model": path,
        "trust_remote_code": True,
        "disable_tqdm": True,
        "disable_pbar": True,
        "device": "cpu",
        "ncpu": ncpu,
        "disable_update": True,
    }
    params.update(kwargs)
    return AutoModel(**params)
