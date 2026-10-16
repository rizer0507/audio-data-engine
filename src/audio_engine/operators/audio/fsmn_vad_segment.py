"""ManifestOperator: expand normalized parent samples into FSMN VAD segments."""

from __future__ import annotations

import os
import threading
import time
import wave
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from queue import SimpleQueue
from typing import Any

import soundfile as sf
from loguru import logger

from audio_engine.core.artifacts import atomic_path, atomic_write_json
from audio_engine.core.fsmn_vad.adapter import (
    MockVadModel,
    ScheduleVadModel,
    load_funasr_vad,
    run_streaming_vad,
)
from audio_engine.core.fsmn_vad.identity import (
    build_segment_labels,
    parent_source_audio_id,
    segment_content_sha256,
    stable_segment_id,
)
from audio_engine.core.fsmn_vad.runtime import (
    dependency_versions,
    prepare_runtime_model_dir,
    validate_runtime_model_dir,
)
from audio_engine.core.fsmn_vad.types import (
    END_SILENCE_KEEP_MS,
    MIN_KEEP_MS,
    SOURCE_STATUS_DECODE_FAILED,
    SOURCE_STATUS_VAD_FAILED,
    SR,
    VAD_FRAME_LENGTH_MS,
    VAD_FRAME_SHIFT_MS,
    VAD_PROCESSING_VERSION,
    VadKeptSegment,
    VadStreamResult,
)
from audio_engine.core.manifest import file_sha256
from audio_engine.core.operator import ManifestOperator, OperatorConfig
from audio_engine.core.registry import register_operator
from audio_engine.core.sample import Sample


def _read_pcm16le_mono(path: Path, *, expect_sr: int = SR) -> tuple[bytes, int, dict[str, Any]]:
    """Load audio as 16-bit mono PCM16LE bytes at expect_sr (resample if needed)."""
    data, sr = sf.read(str(path), always_2d=True, dtype="int16")
    sr = int(sr)
    # Take first channel only (channel strategy: mono / ch0). Multi-channel
    # independent VAD can be enabled later via params.channel_mode.
    mono = data[:, 0]
    meta = {"source_sr": sr, "frames": int(mono.shape[0]), "channels_read": int(data.shape[1])}
    if sr != expect_sr:
        # Should already be 16k from upstream resample; refuse silent mis-rate.
        raise ValueError(
            f"normalized audio sample_rate={sr} != {expect_sr}; "
            "run audio.resample before FSMN VAD"
        )
    return mono.astype("<i2").tobytes(), sr, meta


def _write_wav_pcm16le(path: Path, pcm16le: bytes, *, sample_rate: int = SR) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(path) as tmp:
        with wave.open(str(tmp), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm16le)
    return file_sha256(path)


def _segment_to_dict(seg: VadKeptSegment) -> dict[str, Any]:
    return {
        "start_ms": seg.start_ms,
        "end_ms": seg.end_ms,
        "inferred_end_ms": seg.inferred_end_ms,
        "end_reason": seg.end_reason,
        "start_sample": seg.start_sample,
        "end_sample": seg.end_sample,
        "duration_ms": seg.duration_ms,
        "boundary_clamped": seg.boundary_clamped,
        "extra_padding_ms": seg.extra_padding_ms,
        "pcm_sha256": segment_content_sha256(seg.pcm16le),
        "n_samples": len(seg.pcm16le) // 2,
    }


def _result_to_source_record(
    *,
    source_audio_id: str,
    source_pcm_sha256: str,
    normalized_path: str,
    normalized_sha256: str,
    channel_index: int,
    result: VadStreamResult,
    published_ids: list[str],
    fingerprint: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "vad-source/v1",
        "source_audio_id": source_audio_id,
        "source_pcm_sha256": source_pcm_sha256,
        "normalized_audio_uri": normalized_path,
        "normalized_audio_sha256": normalized_sha256,
        "channel_index": channel_index,
        "status": result.status,
        "error": result.error,
        "candidate_count": len(result.candidates),
        "kept_count": len(result.kept),
        "discarded_count": len(result.discarded),
        "eof_fallback_count": len(result.eof_fallback),
        "published_segment_ids": published_ids,
        "candidates": [
            {
                "start_ms": c.start_ms,
                "end_ms": c.end_ms,
                "end_reason": c.end_reason,
                "start_sample": c.start_sample,
                "end_sample": c.end_sample,
                "boundary_clamped": c.boundary_clamped,
            }
            for c in result.candidates
        ],
        "discarded": [
            {
                "start_ms": d.start_ms,
                "end_ms": d.end_ms,
                "end_reason": d.end_reason,
                "reason": d.reason,
                "start_sample": d.start_sample,
                "end_sample": d.end_sample,
                "duration_ms": d.duration_ms,
            }
            for d in result.discarded
        ],
        "eof_fallback": [_segment_to_dict(s) for s in result.eof_fallback],
        "kept": [_segment_to_dict(s) for s in result.kept],
        "event_log": result.event_log,
        "fingerprint": fingerprint,
    }


@register_operator
class FsmnVadSegmentOperator(ManifestOperator):
    """One-to-many: parent normalized PCM → kept VAD segment Samples.

    Params
    ------
    input_audio_key: default ``resampled_16k``
    output_audio_key: default ``resampled_16k`` (segment WAV; ASR default key)
    source_results_dir: relative to run/output; default ``vad_source_results``
    segments_subdir: default ``vad_segments``
    model_runtime_dir / source_model_dir / fenshen_config: real-model paths
    mock: use scripted mock (tests / dry-run without FunASR)
    mock_intervals: list of [start_ms, end_ms] for ScheduleVadModel
    include_eof_fallback: legacy parameter; starts without an end are always discarded
    Model endpoints at the last frontend frame / file EOF are also always discarded.
    min_keep_ms: 1000 — discard clips with actual duration < 1s (exactly 1s kept)
    end_silence_keep_ms: fixed via END_SILENCE_KEEP_MS (675); retain real audio after model end
    channel_index: default 0（与规范化「只取第0声道」一致；不按声道拆多路）
    prepare_runtime: if True, copy/overlay model into runtime dir
    ncpu: positive integer, default 1; FunASR/PyTorch intra-op threads per invocation
    """

    name = "fsmn_vad_segment"
    version = "1.8.0"
    category = "audio"

    def run(self, samples: list[Sample], config: OperatorConfig) -> list[Sample]:
        params, mock = self.preflight(config.params, mock=config.mock)
        input_key = str(params.get("input_audio_key", "resampled_16k"))
        output_key = str(params.get("output_audio_key", "resampled_16k"))
        include_eof = False
        channel_index = int(params.get("channel_index", 0))
        processing_version = str(params.get("vad_processing_version") or VAD_PROCESSING_VERSION)

        base_dir = Path(config.run_dir) if config.run_dir else Path(config.output_dir)
        source_results_dir = base_dir / str(params.get("source_results_dir", "vad_source_results"))
        segments_dir = Path(config.output_dir) / str(params.get("segments_subdir", "vad_segments"))
        source_results_dir.mkdir(parents=True, exist_ok=True)
        segments_dir.mkdir(parents=True, exist_ok=True)

        fingerprint = self._build_fingerprint(
            params, mock=mock, processing_version=processing_version,
        )
        source_workers = max(1, int(params.get("source_workers") or 1))
        max_in_flight = int(params.get("max_in_flight_sources") or source_workers)
        max_in_flight = max(source_workers, max_in_flight)
        active_workers = min(source_workers, len(samples))
        logger.info(
            "FSMN VAD initializing: sources={} source_workers={} max_in_flight={} ncpu={} runtime={}",
            len(samples), active_workers, max_in_flight, params.get("ncpu", 1),
            fingerprint.get("runtime_dir", "mock"),
        )
        # Initialize before queueing work: a failed load aborts once, rather than
        # retrying on every queued source while executor shutdown drains the shard.
        models: SimpleQueue = SimpleQueue()
        for index in range(active_workers):
            models.put(self._build_vad(params, mock=mock, fingerprint=fingerprint))
            logger.info("FSMN VAD model ready: {}/{}", index + 1, active_workers)

        out_segments: list[Sample] = []
        summary = {
            "sources": 0,
            "kept_segments": 0,
            "discarded": 0,
            "eof_published": 0,
            "eof_held": 0,
            "eof_discarded": 0,
            "endpoint_at_eof_discarded": 0,
            "no_speech": 0,
            "all_short": 0,
            "failed": 0,
            "source_workers": source_workers,
            "max_in_flight_sources": max_in_flight,
            "ncpu": params.get("ncpu", 1),
        }
        lock = threading.Lock()
        # Thread-local VAD sessions: one model + streaming cache per worker thread.
        thread_vad = threading.local()
        completed = 0
        last_progress = time.monotonic()

        def _init_worker() -> None:
            thread_vad.model = models.get_nowait()

        def _handle_one(sample: Sample) -> list[Sample]:
            nonlocal completed, last_progress
            segments = self._process_one_source(
                sample,
                params=params,
                input_key=input_key,
                output_key=output_key,
                include_eof=include_eof,
                channel_index=channel_index,
                processing_version=processing_version,
                fingerprint=fingerprint,
                source_results_dir=source_results_dir,
                segments_dir=segments_dir,
                vad=thread_vad.model,
                summary=summary,
                summary_lock=lock,
            )
            with lock:
                completed += 1
                now = time.monotonic()
                if now - last_progress >= 10 or completed == len(samples):
                    logger.info(
                        "FSMN VAD progress: sources={}/{} kept={} failed={}",
                        completed, len(samples), summary["kept_segments"], summary["failed"],
                    )
                    last_progress = now
            return segments

        if active_workers == 1:
            _init_worker()
            for sample in samples:
                out_segments.extend(_handle_one(sample))
        elif active_workers > 1:
            # Keep both queued and running tasks bounded; reuse each worker's model.
            pending_sources = iter(samples)
            with ThreadPoolExecutor(
                max_workers=active_workers, initializer=_init_worker,
            ) as pool:
                pending = set()
                try:
                    while True:
                        while len(pending) < max_in_flight:
                            sample = next(pending_sources, None)
                            if sample is None:
                                break
                            pending.add(pool.submit(_handle_one, sample))
                        if not pending:
                            break
                        finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                        for future in finished:
                            out_segments.extend(future.result())
                except BaseException:
                    for future in pending:
                        future.cancel()
                    raise

        summary_path = source_results_dir / "_batch_summary.json"
        atomic_write_json(
            summary_path,
            {
                "operator": self.full_name,
                "version": self.version,
                "fingerprint": fingerprint,
                "summary": summary,
                "include_eof_fallback": include_eof,
                "note": (
                    f"Actual segment duration < {MIN_KEEP_MS}ms discarded; "
                    f"exactly {MIN_KEEP_MS}ms kept. "
                    f"Model end retains {END_SILENCE_KEEP_MS}ms real audio (EOF truncate). "
                    "EOF without an end and model endpoints at the last frontend frame "
                    "are always discarded."
                ),
            },
        )
        logger.info(
            "audio.fsmn_vad_segment: sources={} kept={} discarded={} "
            "eof_published={} eof_discarded={} endpoint_at_eof_discarded={} "
            "no_speech={} all_short={} failed={} source_workers={}",
            summary["sources"],
            summary["kept_segments"],
            summary["discarded"],
            summary["eof_published"],
            summary["eof_discarded"],
            summary["endpoint_at_eof_discarded"],
            summary["no_speech"],
            summary["all_short"],
            summary["failed"],
            source_workers,
        )
        return out_segments

    def _process_one_source(
        self,
        sample: Sample,
        *,
        params: dict[str, Any],
        input_key: str,
        output_key: str,
        include_eof: bool,
        channel_index: int,
        processing_version: str,
        fingerprint: dict[str, Any],
        source_results_dir: Path,
        segments_dir: Path,
        vad: Any,
        summary: dict[str, Any],
        summary_lock: threading.Lock,
    ) -> list[Sample]:
        out_segments: list[Sample] = []
        with summary_lock:
            summary["sources"] += 1
        source_audio_id = parent_source_audio_id(sample)
        source_pcm_sha = str(
            sample.labels.get("source_pcm_sha256")
            or sample.labels.get("parent_pcm_sha256")
            or sample.sha256
            or ""
        )

        def _fail(status: str, error: str) -> list[Sample]:
            self._persist_failure(
                source_results_dir, source_audio_id, status, error, fingerprint
            )
            with summary_lock:
                summary["failed"] += 1
            return []

        try:
            audio_path = Path(sample.audio_path(input_key))
        except KeyError as exc:
            return _fail(SOURCE_STATUS_DECODE_FAILED, str(exc))

        try:
            pcm16le, _sr, _meta = _read_pcm16le_mono(audio_path)
            normalized_sha = file_sha256(audio_path)
        except Exception as exc:
            logger.exception("FSMN VAD decode failed for {}", source_audio_id)
            return _fail(SOURCE_STATUS_DECODE_FAILED, str(exc))

        # Per-source model: reset mock call state; FunASR uses fresh cache per call.
        local_vad = vad
        if isinstance(local_vad, (MockVadModel, ScheduleVadModel)):
            if hasattr(local_vad, "reset"):
                local_vad.reset()
            elif isinstance(local_vad, ScheduleVadModel):
                local_vad = ScheduleVadModel(
                    list(params.get("mock_intervals") or []),
                    sample_rate=SR,
                )

        try:
            result = run_streaming_vad(
                local_vad,
                pcm16le,
                include_eof_fallback_in_kept=include_eof,
                generate_options=params.get("generate_options") or None,
            )
        except Exception as exc:
            logger.exception("FSMN VAD failed for {}", source_audio_id)
            return _fail(SOURCE_STATUS_VAD_FAILED, str(exc))

        config_sha = str(fingerprint.get("config_sha256") or "")
        published_ids: list[str] = []

        for idx, seg in enumerate(result.kept):
            pcm_sha = segment_content_sha256(seg.pcm16le)
            seg_id = stable_segment_id(
                source_audio_id=source_audio_id,
                normalized_sha256=normalized_sha,
                channel_index=channel_index,
                start_sample=seg.start_sample,
                end_sample=seg.end_sample,
                vad_processing_version=processing_version,
                config_sha256=config_sha,
            )
            out_path = segments_dir / seg_id[:2] / f"{seg_id}.wav"
            if out_path.is_file():
                wav_sha = file_sha256(out_path)
            else:
                wav_sha = _write_wav_pcm16le(out_path, seg.pcm16le)

            work_sha = pcm_sha
            labels = build_segment_labels(
                source_audio_id=source_audio_id,
                source_pcm_sha256=source_pcm_sha,
                normalized_sha256=normalized_sha,
                normalized_audio_path=str(audio_path.resolve()),
                channel_index=channel_index,
                segment=seg,
                segment_sha256=work_sha,
                segment_index=idx,
                vad_processing_version=processing_version,
                config_sha256=config_sha,
                model_fingerprint=fingerprint,
            )
            labels["segment_id"] = seg_id
            labels["segment_wav_sha256"] = wav_sha
            for k in (
                "source_sample_rate",
                "normalized_sample_rate",
                "source_channels",
                "selected_channel",
                "channel_policy",
                "sample_rate_source",
            ):
                if k in sample.labels and k not in labels:
                    labels[k] = sample.labels[k]
                elif k in sample.quality and k not in labels:
                    labels[k] = sample.quality[k]

            child = Sample(
                id=seg_id,
                source_path=str(audio_path.resolve()),
                sha256=work_sha,
                sample_rate=SR,
                channels=1,
                duration=seg.duration_ms / 1000.0,
                audio={
                    output_key: str(out_path.resolve()),
                    "normalized_16k": str(audio_path.resolve()),
                    "parent_resampled_16k": str(audio_path.resolve()),
                },
                labels=labels,
                quality={
                    "vad_end_reason": seg.end_reason,
                    "clip_start_sample": seg.start_sample,
                    "clip_end_sample": seg.end_sample,
                },
            )
            if "raw" in sample.audio:
                child.audio["parent_raw"] = sample.audio["raw"]
            elif sample.source_path:
                child.audio["parent_raw"] = sample.source_path

            child.mark_completed(self.full_name)
            child.add_lineage(
                operator=self.full_name,
                version=self.version,
                params={
                    "vad_processing_version": processing_version,
                    "config_sha256": config_sha,
                    "start_sample": seg.start_sample,
                    "end_sample": seg.end_sample,
                    "source_audio_id": source_audio_id,
                },
                input_key=input_key,
                output_key=output_key,
                output_path=str(out_path),
            )
            out_segments.append(child)
            published_ids.append(seg_id)

        with summary_lock:
            summary["kept_segments"] += len(result.kept)
            summary["discarded"] += len(result.discarded)
            eof_n = len(result.eof_fallback)
            if include_eof:
                summary["eof_published"] += sum(
                    1 for s in result.kept if s.end_reason == "eof_fallback"
                )
            else:
                summary["eof_held"] += eof_n
            if result.status == "no_speech":
                summary["no_speech"] += 1
            elif result.status == "all_short":
                summary["all_short"] += 1
            summary["eof_discarded"] += sum(
                d.end_reason == "eof_fallback" or d.reason == "endpoint_at_eof"
                for d in result.discarded
            )
            summary["endpoint_at_eof_discarded"] += sum(
                d.reason == "endpoint_at_eof" for d in result.discarded
            )

        record = _result_to_source_record(
            source_audio_id=source_audio_id,
            source_pcm_sha256=source_pcm_sha,
            normalized_path=str(audio_path.resolve()),
            normalized_sha256=normalized_sha,
            channel_index=channel_index,
            result=result,
            published_ids=published_ids,
            fingerprint=fingerprint,
        )
        record["source_channels"] = sample.labels.get("source_channels") or sample.quality.get(
            "source_channels"
        )
        record["selected_channel"] = channel_index
        record["channel_policy"] = sample.labels.get("channel_policy") or sample.quality.get(
            "channel_policy"
        )
        record["source_sample_rate"] = (
            sample.labels.get("source_sample_rate") or sample.quality.get("source_sample_rate")
        )
        record["normalized_sample_rate"] = SR
        atomic_write_json(source_results_dir / f"{source_audio_id}.json", record)
        return out_segments

    @staticmethod
    def preflight(params: dict[str, Any], *, mock: bool = False) -> tuple[dict[str, Any], bool]:
        """Validate model paths in the parent before any shard processes start."""
        params = dict(params)
        FsmnVadSegmentOperator._resolve_env_params(params)
        mock = bool(params.get("mock", mock)) or str(
            os.environ.get("AUDIO_ENGINE_VAD_MOCK", "")
        ).strip() in {"1", "true", "TRUE", "yes"}
        if mock:
            return params, True
        runtime_dir = str(params.get("model_runtime_dir") or "").strip()
        if not runtime_dir:
            raise ValueError(
                "Set FSMN_VAD_RUNTIME_DIR/model_runtime_dir to a prepared local directory"
            )
        if params.get("prepare_runtime"):
            source = Path(str(params.get("source_model_dir") or "")).expanduser()
            config = Path(str(params.get("fenshen_config") or "")).expanduser()
            if not params.get("source_model_dir") or not source.is_dir():
                raise FileNotFoundError(f"FSMN VAD source model directory not found: {source}")
            if not config.is_file():
                raise FileNotFoundError(f"fenshen VAD config not found: {config}")
        else:
            runtime_dir = str(validate_runtime_model_dir(runtime_dir))
        params["model_runtime_dir"] = runtime_dir
        return params, False

    @staticmethod
    def _resolve_env_params(params: dict[str, Any]) -> None:
        """Fill model paths from env when YAML params omit them (no ${} expansion)."""
        env_map = {
            "model_runtime_dir": "FSMN_VAD_RUNTIME_DIR",
            "source_model_dir": "FSMN_VAD_SOURCE_MODEL_DIR",
            "fenshen_config": "FSMN_VAD_FENSHEN_CONFIG",
        }
        for key, env_name in env_map.items():
            current = str(params.get(key) or "").strip()
            if current:
                continue
            value = str(os.environ.get(env_name) or "").strip()
            if value:
                params[key] = value
        if "prepare_runtime" not in params and params.get("source_model_dir") and params.get(
            "fenshen_config"
        ):
            # Auto-prepare only when explicitly requested via env.
            if str(os.environ.get("FSMN_VAD_PREPARE_RUNTIME", "")).strip() in {
                "1",
                "true",
                "TRUE",
                "yes",
            }:
                params["prepare_runtime"] = True

    def _build_fingerprint(
        self, params: dict[str, Any], *, mock: bool, processing_version: str
    ) -> dict[str, Any]:
        fp: dict[str, Any] = {
            "vad_processing_version": processing_version,
            "operator_version": self.version,
            "mock": mock,
            "chunk_ms": 200,
            "min_keep_ms": MIN_KEEP_MS,
            "eof_without_end_policy": "discard",
            "endpoint_at_eof_policy": "discard",
            "endpoint_at_eof_boundary": "last_frontend_frame",
            "vad_frame_length_ms": VAD_FRAME_LENGTH_MS,
            "vad_frame_shift_ms": VAD_FRAME_SHIFT_MS,
            "end_silence_keep_ms": END_SILENCE_KEEP_MS,
            "dependencies": dependency_versions(),
        }
        if mock:
            fp["config_sha256"] = "mock"
            fp["engine"] = "mock"
            return fp

        runtime_dir = params.get("model_runtime_dir")
        source_model_dir = params.get("source_model_dir")
        fenshen_config = params.get("fenshen_config")
        if params.get("prepare_runtime") and source_model_dir and fenshen_config and runtime_dir:
            prepared = prepare_runtime_model_dir(
                source_model_dir=source_model_dir,
                fenshen_config_path=fenshen_config,
                runtime_dir=runtime_dir,
                overwrite=bool(params.get("overwrite_runtime", False)),
                verify_fingerprints=bool(params.get("verify_fingerprints", True)),
            )
            fp.update(prepared)
            validate_runtime_model_dir(prepared["runtime_dir"])
            fp["engine"] = "funasr/FsmnVADStreaming"
        elif runtime_dir:
            runtime = Path(str(runtime_dir))
            cfg = runtime / "config.yaml"
            fp["runtime_dir"] = str(runtime.resolve())
            fp["config_sha256"] = file_sha256(cfg) if cfg.is_file() else ""
            fp["engine"] = "funasr/FsmnVADStreaming"
        else:
            fp["config_sha256"] = ""
            fp["engine"] = "funasr/FsmnVADStreaming"
            fp["warning"] = "model_runtime_dir not set"
        return fp

    def _build_vad(self, params: dict[str, Any], *, mock: bool, fingerprint: dict[str, Any]) -> Any:
        if mock:
            intervals = params.get("mock_intervals")
            if intervals:
                pairs = [(float(a), float(b)) for a, b in intervals]
                return ScheduleVadModel(pairs, sample_rate=SR)
            script = params.get("mock_script") or {}
            # JSON keys become strings — normalize to int chunk indices when possible.
            normalized: dict[Any, list[list[float]]] = {}
            for k, v in script.items():
                key: Any = int(k) if str(k).isdigit() else k
                normalized[key] = [list(map(float, e)) for e in v]
            return MockVadModel(normalized, by_chunk_index=True)

        runtime_dir = params.get("model_runtime_dir") or fingerprint.get("runtime_dir")
        if not runtime_dir:
            raise RuntimeError(
                "audio.fsmn_vad_segment requires model_runtime_dir "
                "(or mock=true for engineering tests)"
            )
        return load_funasr_vad(str(runtime_dir), ncpu=params.get("ncpu", 1))

    def _persist_failure(
        self,
        source_results_dir: Path,
        source_audio_id: str,
        status: str,
        error: str,
        fingerprint: dict[str, Any],
    ) -> None:
        atomic_write_json(
            source_results_dir / f"{source_audio_id}.json",
            {
                "schema_version": "vad-source/v1",
                "source_audio_id": source_audio_id,
                "status": status,
                "error": error,
                "candidate_count": 0,
                "kept_count": 0,
                "discarded_count": 0,
                "eof_fallback_count": 0,
                "published_segment_ids": [],
                "fingerprint": fingerprint,
            },
        )
