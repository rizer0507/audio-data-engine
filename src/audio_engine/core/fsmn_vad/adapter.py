"""Streaming FSMN VAD adapter aligned with fenshen client packet / end scheduling."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np

from audio_engine.core.fsmn_vad.pairing import FenshenEventPairer
from audio_engine.core.fsmn_vad.runtime import (
    assert_fenshen_runtime_config,
    validate_loaded_vad_opts,
    validate_runtime_model_dir,
)
from audio_engine.core.fsmn_vad.types import (
    MODEL_CHUNK_MS,
    PACKET_MS,
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
    packet_ms: int = PACKET_MS,
    chunk_samples: int | None = None,
    chunk_ms: int = MODEL_CHUNK_MS,
    generate_options: dict[str, Any] | None = None,
    include_eof_fallback_in_kept: bool = True,
    send_end_result: bool = True,
    sample_rate: int = SR,
) -> VadStreamResult:
    """Replay PCM as 100 ms client packets; infer every two packets; then signal=end.

    Matches ``process_audio_chunk`` / ``process_str_message(end)`` with
    ``send_end_result=true``: leftover non-empty accum uses ``is_final=True``;
    empty accum skips ``generate`` but still runs application EOF logic.

    ``chunk_samples`` is accepted only as an alias for packet size in samples;
    if set it must equal ``packet_ms`` at ``sample_rate``.
    ``include_eof_fallback_in_kept`` is ignored when ``send_end_result`` is True.
    """
    del include_eof_fallback_in_kept
    if len(pcm16le) % 2:
        raise ValueError("PCM16 byte length must be even")
    if sample_rate != SR:
        raise ValueError(f"run_streaming_vad requires sample_rate={SR}, got {sample_rate}")
    if packet_ms < 1:
        raise ValueError("packet_ms must be >= 1")
    packet_samples = sample_rate * packet_ms // 1000
    if chunk_samples is not None and int(chunk_samples) != packet_samples:
        raise ValueError(
            f"chunk_samples={chunk_samples} conflicts with packet_ms={packet_ms} "
            f"({packet_samples} samples)"
        )
    packet_bytes = packet_samples * 2
    total = len(pcm16le) // 2
    if total == 0:
        return VadStreamResult(status="empty_audio", event_log=[])

    options = dict(generate_options or {})
    reserved = {"input", "cache", "is_final", "chunk_size"}
    if reserved.intersection(options):
        raise ValueError("generate_options cannot replace stream arguments")

    cache: dict[str, Any] = {}
    event_log: list[dict[str, Any]] = []
    generate = vad.generate if hasattr(vad, "generate") else vad  # type: ignore[assignment]
    pairer = FenshenEventPairer(sample_rate=sample_rate)

    received = 0
    chunk_idx = 0
    vad_accum = b""
    call_index = 0

    def _flush(*, is_final: bool, application_end: bool) -> None:
        nonlocal call_index, vad_accum
        audio = pcm16le_to_float32(vad_accum) if vad_accum else np.asarray([], dtype=np.float32)
        input_end = received // 2
        input_start = (received - len(vad_accum)) // 2 if vad_accum else input_end
        entry: dict[str, Any] = {
            "call_index": call_index,
            "input_start_sample": input_start,
            "input_end_sample": input_end,
            "received_bytes": received,
            "is_final": is_final,
            "application_end": application_end,
            "packet_ms": packet_ms,
            "model_chunk_size": chunk_ms,
        }
        if len(audio) > 0:
            result = generate(
                input=audio,
                cache=cache,
                is_final=is_final,
                chunk_size=chunk_ms,
                **options,
            )
            events = result[0].get("value", []) if result else []
            entry["skipped_generate"] = False
            entry["events"] = [list(pair) for pair in events]
            pairer.consume_events(events, pcm16le, received_bytes=received)
            call_index += 1
        else:
            entry["skipped_generate"] = True
            entry["events"] = []
            # Distinguish app-end-without-generate from a model final call.
            entry["application_end_only"] = True
        event_log.append(entry)
        vad_accum = b""
        if application_end:
            pairer.finalize_application_end(
                pcm16le,
                received_bytes=received,
                send_end_result=send_end_result,
            )

    while received < len(pcm16le):
        packet = pcm16le[received : received + packet_bytes]
        received += len(packet)
        if packet:
            vad_accum += packet
        chunk_idx += 1
        if chunk_idx >= 2:
            chunk_idx = 0
            _flush(is_final=False, application_end=False)

    # Client signal=end with send_end_result=true → process_audio_chunk(b'').
    if send_end_result:
        _flush(is_final=True, application_end=True)
    elif vad_accum:
        # Non-fenshen path: flush leftover without application EOF publication.
        _flush(is_final=True, application_end=False)

    return pairer.to_result(event_log)


class MockVadModel:
    """Deterministic mock: emits scripted events keyed by generate-call index.

    ``script`` maps call index (after two-packet merges / final flush) to
    ``[start_ms, end_ms]`` events.
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
            events = self.default_events
        self._chunk_i += 1
        return [{"value": [list(e) for e in events]}]

    def reset(self) -> None:
        self.calls.clear()
        self._chunk_i = 0


class ScheduleVadModel:
    """Emit events when cumulative input covers their start; complete ends at/after end_ms."""

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
            if idx not in self._pending_starts and covered_ms >= start_ms:
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

    def reset(self) -> None:
        self.calls.clear()
        self._emitted.clear()
        self._pending_starts.clear()
        self._cum_samples = 0
        self._cache_ids.clear()


def load_funasr_vad(
    runtime_dir: str | Path,
    *,
    ncpu: int = 1,
    verify_loaded_config: bool = True,
    **kwargs: Any,
) -> Any:
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
    model = AutoModel(**params)
    if verify_loaded_config:
        expected = assert_fenshen_runtime_config(path)
        detector = getattr(model, "model", None)
        opts = dict(vars(detector.vad_opts)) if detector is not None and hasattr(detector, "vad_opts") else {}
        validate_loaded_vad_opts(expected, opts)
    return model


__all__ = [
    "MockVadModel",
    "ScheduleVadModel",
    "VadModel",
    "load_funasr_vad",
    "pcm16le_to_float32",
    "run_streaming_vad",
]
