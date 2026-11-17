"""ManifestOperator: approval parents → FireRedVAD HTTP timestamps → local 16 kHz clips."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from audio_engine.core.artifacts import atomic_write_json
from audio_engine.core.audio_scene import (
    SCENE_SHENPI_INTERVIEW,
    SCENE_SHENPI_OUTBOUND,
    VAD_SHENPI_SCENES,
    requires_reingest_for_shenpi,
    scene_from_sample_labels,
)
from audio_engine.core.fsmn_vad.clip import clip_pcm16le, ms_to_sample
from audio_engine.core.fsmn_vad.identity import parent_source_audio_id, segment_content_sha256
from audio_engine.core.fsmn_vad.publish import publish_kept_segments, read_pcm16le_mono
from audio_engine.core.fsmn_vad.types import (
    END_REASON_VAD,
    SOURCE_STATUS_COMPLETED,
    SOURCE_STATUS_DECODE_FAILED,
    SOURCE_STATUS_NO_SPEECH,
    SOURCE_STATUS_VAD_FAILED,
    SR,
    VadKeptSegment,
)
from audio_engine.core.manifest import file_sha256
from audio_engine.core.operator import ManifestOperator, OperatorConfig
from audio_engine.core.registry import register_operator
from audio_engine.core.sample import Sample
from audio_engine.operators.asr.vllm import _encode_multipart

SHENPI_VAD_PROCESSING_VERSION = "shenpi_vad_segment/v1"
DEFAULT_PROFILES_PATH = Path("configs/vad/shenpi_vad_profiles.yaml")
BRANCH_HTTP = "http_timestamps"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_shenpi_profiles(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else _repo_root() / DEFAULT_PROFILES_PATH
    if not cfg_path.is_file():
        raise FileNotFoundError(f"shenpi VAD profiles not found: {cfg_path}")
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"invalid shenpi VAD profiles: {cfg_path}")
    return data


def profile_fingerprint(profiles: dict[str, Any]) -> str:
    raw = json.dumps(profiles, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _select_shenpi_parents(samples: list[Sample]) -> tuple[list[Sample], dict[str, int]]:
    selected: list[Sample] = []
    stats = {
        "input": len(samples),
        "selected": 0,
        "other_scene": 0,
        "unknown_scene": 0,
        "missing_scene": 0,
    }
    blocked: list[str] = []
    for sample in samples:
        if requires_reingest_for_shenpi(sample.id, sample.labels):
            blocked.append(sample.id)
            continue
        scene = scene_from_sample_labels(sample.labels)
        if scene is None:
            stats["missing_scene"] += 1
            continue
        if scene in VAD_SHENPI_SCENES:
            selected.append(sample)
            continue
        if scene == "unknown":
            stats["unknown_scene"] += 1
        else:
            stats["other_scene"] += 1
    if blocked:
        preview = ", ".join(blocked[:5])
        more = "" if len(blocked) <= 5 else f" (+{len(blocked) - 5} more)"
        raise ValueError(
            "audio.shenpi_vad_segment: approval parents lack labels.audio_scene and must be "
            f"re-ingested before either VAD branch: {preview}{more}"
        )
    stats["selected"] = len(selected)
    return selected, stats


def _validate_timestamps(raw: Any) -> list[tuple[int, int]]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("timestamps must be a list of [start_ms, end_ms]")
    parsed: list[tuple[int, int]] = []
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"malformed timestamp entry: {item!r}")
        start_ms, end_ms = item
        if not isinstance(start_ms, int) or isinstance(start_ms, bool):
            raise ValueError(f"timestamp start_ms must be int: {item!r}")
        if not isinstance(end_ms, int) or isinstance(end_ms, bool):
            raise ValueError(f"timestamp end_ms must be int: {item!r}")
        if start_ms < 0 or end_ms < 0:
            raise ValueError(f"negative timestamp: {item!r}")
        if end_ms <= start_ms:
            raise ValueError(f"non-positive interval: {item!r}")
        parsed.append((start_ms, end_ms))
    # Sort + drop exact duplicate intervals; keep overlapping distinct ranges.
    parsed.sort(key=lambda pair: (pair[0], pair[1]))
    unique: list[tuple[int, int]] = []
    for pair in parsed:
        if unique and unique[-1] == pair:
            continue
        unique.append(pair)
    return unique


def timestamps_to_kept(
    timestamps: list[tuple[int, int]],
    pcm16le: bytes,
    *,
    sample_rate: int = SR,
) -> list[VadKeptSegment]:
    total_samples = len(pcm16le) // 2
    kept: list[VadKeptSegment] = []
    for start_ms, end_ms in timestamps:
        start_sample = ms_to_sample(start_ms, sample_rate=sample_rate)
        end_sample = ms_to_sample(end_ms, sample_rate=sample_rate)
        if start_sample >= total_samples:
            raise ValueError(
                f"timestamp start beyond parent audio: start_ms={start_ms} "
                f"start_sample={start_sample} total={total_samples}"
            )
        clipped_end = min(total_samples, end_sample)
        if clipped_end <= start_sample:
            raise ValueError(
                f"empty clip after bounds: [{start_ms},{end_ms}] → "
                f"[{start_sample},{clipped_end}) total={total_samples}"
            )
        pcm = clip_pcm16le(pcm16le, start_sample, clipped_end)
        if not pcm:
            raise ValueError(f"empty PCM after clip [{start_sample},{clipped_end})")
        n_samples = len(pcm) // 2
        duration_ms = n_samples * 1000.0 / sample_rate
        kept.append(
            VadKeptSegment(
                start_ms=float(start_sample * 1000.0 / sample_rate),
                end_ms=float(end_ms),
                inferred_end_ms=float(end_ms),
                end_reason=END_REASON_VAD,
                start_sample=start_sample,
                end_sample=clipped_end,
                duration_ms=duration_ms,
                boundary_clamped=clipped_end != end_sample or start_sample < 0,
                pcm16le=pcm,
                extra_padding_ms=0.0,
                clip_branch=BRANCH_HTTP,
                reported_start_ms=float(start_ms),
                reported_end_ms=float(end_ms),
                buffer_end_sample=total_samples,
            )
        )
    return kept


def _http_upload(
    *,
    api_base: str,
    sample_id: str,
    pcm_bytes: bytes,
    sample_rate: int,
    vad_config: dict[str, Any],
    timeout_s: float,
    mode: str = "stream",
) -> dict[str, Any]:
    url = api_base.rstrip("/") + "/vad/upload"
    fields = {
        "id": sample_id,
        "audio_type": "pcm",
        "mode": mode,
        "sample_rate": str(int(sample_rate)),
        "vad_config": json.dumps(vad_config, ensure_ascii=False, separators=(",", ":")),
        "return_probs": "false",
        "return_frames": "false",
    }
    body, content_type = _encode_multipart(
        fields,
        {"file": ("audio.pcm", pcm_bytes, "application/octet-stream")},
    )
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": content_type, "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            payload = response.read()
            status = int(getattr(response, "status", 200) or 200)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"network error calling {url}: {exc}") from exc
    try:
        data = json.loads(payload.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"non-JSON response from {url} (http={status})") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"unexpected response type from {url}")
    return data


def _is_retryable(exc: Exception) -> bool:
    text = str(exc).lower()
    if "HTTP 4" in str(exc) and "HTTP 429" not in str(exc):
        # 400/413/422 etc. are not retryable except rate limit.
        if "HTTP 429" in str(exc):
            return True
        return False
    markers = ("timeout", "timed out", "temporarily", "connection", "reset", "503", "502", "429")
    return any(m in text for m in markers)


@register_operator
class ShenpiVadSegmentOperator(ManifestOperator):
    """One-to-many: approval parents → HTTP VAD timestamps → local 16 kHz segment Samples."""

    name = "shenpi_vad_segment"
    version = "1.0.0"
    category = "audio"

    def run(self, samples: list[Sample], config: OperatorConfig) -> list[Sample]:
        params = self.preflight(config.params, mock=config.mock)
        selected, select_stats = _select_shenpi_parents(samples)
        logger.info(
            "audio.shenpi_vad_segment scene select: input={} selected={} other={} "
            "unknown={} missing_scene={}",
            select_stats["input"],
            select_stats["selected"],
            select_stats["other_scene"],
            select_stats["unknown_scene"],
            select_stats["missing_scene"],
        )
        if not selected:
            logger.info("audio.shenpi_vad_segment: no approval parents selected; empty segments")
            return []

        input_key = str(params.get("input_audio_key", "resampled_16k"))
        raw_key = str(params.get("raw_audio_key", "raw"))
        output_key = str(params.get("output_audio_key", "resampled_16k"))
        channel_index = int(params.get("channel_index", 0))
        processing_version = str(
            params.get("vad_processing_version") or SHENPI_VAD_PROCESSING_VERSION
        )
        profiles = load_shenpi_profiles(params.get("profiles_path"))
        config_sha = profile_fingerprint(profiles)
        fingerprint = {
            "vad_processing_version": processing_version,
            "config_sha256": config_sha,
            "profiles_path": str(params.get("profiles_path") or DEFAULT_PROFILES_PATH),
            "api_base": params.get("api_base"),
            "deploy_id": params.get("deploy_id"),
            "mode": params.get("mode", "stream"),
            "scene_select": select_stats,
            "mock": bool(params.get("mock")),
        }

        base_dir = Path(config.run_dir) if config.run_dir else Path(config.output_dir)
        source_results_dir = base_dir / str(params.get("source_results_dir", "vad_source_results"))
        segments_dir = Path(config.output_dir) / str(params.get("segments_subdir", "vad_segments"))
        source_results_dir.mkdir(parents=True, exist_ok=True)
        segments_dir.mkdir(parents=True, exist_ok=True)

        source_workers = max(1, int(params.get("source_workers") or 2))
        max_in_flight = max(source_workers, int(params.get("max_in_flight_sources") or source_workers))
        active_workers = min(source_workers, len(selected))

        out_segments: list[Sample] = []
        summary = {
            "sources": 0,
            "kept_segments": 0,
            "no_speech": 0,
            "failed": 0,
            "interview": 0,
            "outbound": 0,
            "source_workers": source_workers,
            "max_in_flight_sources": max_in_flight,
        }
        lock = threading.Lock()
        failures: list[str] = []

        def _handle_one(sample: Sample) -> list[Sample]:
            try:
                return self._process_one_source(
                    sample,
                    params=params,
                    profiles=profiles,
                    fingerprint=fingerprint,
                    input_key=input_key,
                    raw_key=raw_key,
                    output_key=output_key,
                    channel_index=channel_index,
                    processing_version=processing_version,
                    config_sha=config_sha,
                    source_results_dir=source_results_dir,
                    segments_dir=segments_dir,
                    summary=summary,
                    summary_lock=lock,
                )
            except Exception as exc:
                with lock:
                    summary["failed"] += 1
                    failures.append(f"{sample.id}: {exc}")
                logger.exception("shenpi VAD failed for {}", sample.id)
                self._persist_failure(
                    source_results_dir,
                    sample_id=str(sample.id),
                    error=str(exc),
                    fingerprint=fingerprint,
                )
                return []

        if active_workers == 1:
            for sample in selected:
                out_segments.extend(_handle_one(sample))
        else:
            pending_sources = iter(selected)
            with ThreadPoolExecutor(max_workers=active_workers) as pool:
                pending = set()
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

        atomic_write_json(
            source_results_dir / "_batch_summary.json",
            {
                "operator": self.full_name,
                "version": self.version,
                "fingerprint": fingerprint,
                "summary": summary,
                "failures": failures,
            },
        )
        logger.info(
            "audio.shenpi_vad_segment: sources={} kept={} no_speech={} failed={} "
            "interview={} outbound={}",
            summary["sources"],
            summary["kept_segments"],
            summary["no_speech"],
            summary["failed"],
            summary["interview"],
            summary["outbound"],
        )
        if failures and not bool(params.get("allow_partial", False)):
            preview = "; ".join(failures[:3])
            raise RuntimeError(
                f"audio.shenpi_vad_segment incomplete: {len(failures)} parent failure(s): {preview}"
            )
        return out_segments

    def _process_one_source(
        self,
        sample: Sample,
        *,
        params: dict[str, Any],
        profiles: dict[str, Any],
        fingerprint: dict[str, Any],
        input_key: str,
        raw_key: str,
        output_key: str,
        channel_index: int,
        processing_version: str,
        config_sha: str,
        source_results_dir: Path,
        segments_dir: Path,
        summary: dict[str, Any],
        summary_lock: threading.Lock,
    ) -> list[Sample]:
        with summary_lock:
            summary["sources"] += 1
        source_audio_id = parent_source_audio_id(sample)
        scene = scene_from_sample_labels(sample.labels)
        if scene not in profiles:
            raise ValueError(f"no HTTP profile for scene={scene!r} id={sample.id}")
        profile = profiles[scene]
        expected_sr = int(profile["source_sample_rate"])
        vad_config = dict(profile.get("vad_config") or {})
        label_sr = sample.labels.get("source_sample_rate")
        if label_sr is not None and int(label_sr) != expected_sr:
            raise ValueError(
                f"source_sample_rate mismatch for {sample.id}: "
                f"labels={label_sr} profile={expected_sr}"
            )

        raw_path = Path(sample.audio.get(raw_key) or sample.source_path)
        if not raw_path.is_file():
            raise FileNotFoundError(f"raw PCM missing for {sample.id}: {raw_path}")
        raw_bytes = raw_path.read_bytes()
        if sample.sha256:
            actual = hashlib.sha256(raw_bytes).hexdigest()
            if actual != sample.sha256:
                raise ValueError(
                    f"raw PCM sha256 mismatch for {sample.id}: "
                    f"manifest={sample.sha256} file={actual}"
                )

        audio_path = Path(sample.audio_path(input_key))
        try:
            pcm16le, _, meta = read_pcm16le_mono(audio_path, expect_sr=SR)
        except Exception as exc:
            self._persist_failure(
                source_results_dir,
                sample_id=source_audio_id,
                error=str(exc),
                fingerprint=fingerprint,
                status=SOURCE_STATUS_DECODE_FAILED,
            )
            raise
        normalized_sha = file_sha256(audio_path)
        source_pcm_sha = sample.sha256 or file_sha256(raw_path)

        if bool(params.get("mock")):
            response = self._mock_response(
                sample_id=source_audio_id,
                sample_rate=expected_sr,
                vad_config=vad_config,
                params=params,
            )
        else:
            response = self._call_with_retries(
                api_base=str(params["api_base"]),
                sample_id=source_audio_id,
                pcm_bytes=raw_bytes,
                sample_rate=expected_sr,
                vad_config=vad_config,
                timeout_s=float(params.get("timeout_s") or 120),
                retries=int(params.get("retries") or 2),
                mode=str(params.get("mode") or "stream"),
            )

        self._validate_response(
            response,
            sample_id=source_audio_id,
            expected_sr=expected_sr,
            vad_config=vad_config,
        )
        timestamps = _validate_timestamps(response.get("timestamps"))
        if int(response.get("num_segments", len(timestamps))) != len(timestamps):
            raise ValueError(
                f"num_segments mismatch for {sample.id}: "
                f"{response.get('num_segments')} vs {len(timestamps)}"
            )

        if not timestamps:
            with summary_lock:
                summary["no_speech"] += 1
                if scene == SCENE_SHENPI_INTERVIEW:
                    summary["interview"] += 1
                elif scene == SCENE_SHENPI_OUTBOUND:
                    summary["outbound"] += 1
            atomic_write_json(
                source_results_dir / f"{source_audio_id}.json",
                {
                    "schema_version": "vad-source/v1",
                    "source_audio_id": source_audio_id,
                    "status": SOURCE_STATUS_NO_SPEECH,
                    "audio_scene": scene,
                    "upload_sample_rate": expected_sr,
                    "published_segment_ids": [],
                    "timestamps": [],
                    "fingerprint": fingerprint,
                    "response_meta": {
                        "sample_rate": response.get("sample_rate"),
                        "model_sample_rate": response.get("model_sample_rate"),
                        "elapsed_ms": response.get("elapsed_ms"),
                    },
                },
            )
            return []

        kept = timestamps_to_kept(timestamps, pcm16le, sample_rate=SR)
        children, published_ids = publish_kept_segments(
            parent=sample,
            kept=kept,
            source_audio_id=source_audio_id,
            source_pcm_sha256=str(source_pcm_sha),
            normalized_path=audio_path,
            normalized_sha256=normalized_sha,
            channel_index=channel_index,
            segments_dir=segments_dir,
            processing_version=processing_version,
            config_sha256=config_sha,
            operator_full_name=self.full_name,
            operator_version=self.version,
            input_key=input_key,
            output_key=output_key,
            model_fingerprint=fingerprint,
            lineage_extra={"audio_scene": scene, "upload_sample_rate": expected_sr},
        )
        with summary_lock:
            summary["kept_segments"] += len(children)
            if scene == SCENE_SHENPI_INTERVIEW:
                summary["interview"] += 1
            elif scene == SCENE_SHENPI_OUTBOUND:
                summary["outbound"] += 1
        atomic_write_json(
            source_results_dir / f"{source_audio_id}.json",
            {
                "schema_version": "vad-source/v1",
                "source_audio_id": source_audio_id,
                "source_pcm_sha256": source_pcm_sha,
                "normalized_audio_uri": str(audio_path.resolve()),
                "normalized_audio_sha256": normalized_sha,
                "channel_index": channel_index,
                "status": SOURCE_STATUS_COMPLETED,
                "audio_scene": scene,
                "upload_sample_rate": expected_sr,
                "candidate_count": len(timestamps),
                "kept_count": len(children),
                "published_segment_ids": published_ids,
                "timestamps": [list(pair) for pair in timestamps],
                "kept": [
                    {
                        "start_ms": seg.start_ms,
                        "end_ms": seg.end_ms,
                        "reported_start_ms": seg.reported_start_ms,
                        "reported_end_ms": seg.reported_end_ms,
                        "start_sample": seg.start_sample,
                        "end_sample": seg.end_sample,
                        "buffer_end_sample": seg.buffer_end_sample,
                        "duration_ms": seg.duration_ms,
                        "boundary_clamped": seg.boundary_clamped,
                        "pcm_sha256": segment_content_sha256(seg.pcm16le),
                        "n_samples": len(seg.pcm16le) // 2,
                    }
                    for seg in kept
                ],
                "fingerprint": fingerprint,
                "decode_meta": meta,
                "response_meta": {
                    "sample_rate": response.get("sample_rate"),
                    "model_sample_rate": response.get("model_sample_rate"),
                    "vad_config": response.get("vad_config"),
                    "elapsed_ms": response.get("elapsed_ms"),
                },
            },
        )
        return children

    def _call_with_retries(
        self,
        *,
        api_base: str,
        sample_id: str,
        pcm_bytes: bytes,
        sample_rate: int,
        vad_config: dict[str, Any],
        timeout_s: float,
        retries: int,
        mode: str,
    ) -> dict[str, Any]:
        attempt = 0
        last_exc: Exception | None = None
        while attempt <= max(0, retries):
            try:
                return _http_upload(
                    api_base=api_base,
                    sample_id=sample_id,
                    pcm_bytes=pcm_bytes,
                    sample_rate=sample_rate,
                    vad_config=vad_config,
                    timeout_s=timeout_s,
                    mode=mode,
                )
            except Exception as exc:
                last_exc = exc
                if attempt >= retries or not _is_retryable(exc):
                    raise
                sleep_s = min(2.0 ** attempt, 8.0)
                logger.warning(
                    "shenpi VAD retry {}/{} for {}: {}",
                    attempt + 1,
                    retries,
                    sample_id,
                    exc,
                )
                time.sleep(sleep_s)
                attempt += 1
        assert last_exc is not None
        raise last_exc

    @staticmethod
    def _validate_response(
        response: dict[str, Any],
        *,
        sample_id: str,
        expected_sr: int,
        vad_config: dict[str, Any],
    ) -> None:
        if response.get("status") != "ok":
            raise RuntimeError(
                f"VAD status={response.get('status')!r} error={response.get('error')!r}"
            )
        if str(response.get("id")) != str(sample_id):
            raise ValueError(
                f"response id mismatch: expected={sample_id!r} got={response.get('id')!r}"
            )
        if response.get("mode") not in (None, "stream", "offline"):
            raise ValueError(f"unexpected mode={response.get('mode')!r}")
        if int(response.get("sample_rate")) != int(expected_sr):
            raise ValueError(
                f"sample_rate echo mismatch: expected={expected_sr} "
                f"got={response.get('sample_rate')}"
            )
        if int(response.get("model_sample_rate", SR)) != SR:
            raise ValueError(
                f"model_sample_rate expected {SR}, got {response.get('model_sample_rate')}"
            )
        echoed = response.get("vad_config")
        if isinstance(echoed, dict):
            for key, value in vad_config.items():
                if key in echoed and echoed[key] != value:
                    raise ValueError(
                        f"vad_config echo mismatch for {key}: "
                        f"sent={value!r} got={echoed[key]!r}"
                    )

    @staticmethod
    def _mock_response(
        *,
        sample_id: str,
        sample_rate: int,
        vad_config: dict[str, Any],
        params: dict[str, Any],
    ) -> dict[str, Any]:
        script = params.get("mock_by_id") or {}
        if isinstance(script, dict) and sample_id in script:
            timestamps = script[sample_id]
        else:
            timestamps = params.get("mock_intervals")
            if timestamps is None:
                timestamps = [[250, 750]]
        return {
            "id": sample_id,
            "status": "ok",
            "mode": str(params.get("mode") or "stream"),
            "sample_rate": int(sample_rate),
            "model_sample_rate": SR,
            "timestamps": timestamps,
            "num_segments": len(timestamps or []),
            "vad_config": dict(vad_config),
            "elapsed_ms": 0.0,
        }

    @staticmethod
    def _persist_failure(
        source_results_dir: Path,
        *,
        sample_id: str,
        error: str,
        fingerprint: dict[str, Any],
        status: str = SOURCE_STATUS_VAD_FAILED,
    ) -> None:
        atomic_write_json(
            source_results_dir / f"{sample_id}.json",
            {
                "schema_version": "vad-source/v1",
                "source_audio_id": sample_id,
                "status": status,
                "error": error,
                "published_segment_ids": [],
                "fingerprint": fingerprint,
            },
        )

    @staticmethod
    def preflight(params: dict[str, Any], *, mock: bool = False) -> dict[str, Any]:
        params = dict(params)
        mock = bool(params.get("mock", mock)) or str(
            os.environ.get("AUDIO_ENGINE_SHENPI_VAD_MOCK", "")
        ).strip() in {"1", "true", "TRUE", "yes"}
        params["mock"] = mock
        if not mock:
            api_base = str(params.get("api_base") or os.environ.get("SHENPI_VAD_API_BASE") or "").strip()
            if not api_base:
                raise ValueError(
                    "Set SHENPI_VAD_API_BASE or params.api_base for audio.shenpi_vad_segment"
                )
            params["api_base"] = api_base
        else:
            params.setdefault("api_base", "http://mock.local")
        # Ensure profiles loadable early.
        load_shenpi_profiles(params.get("profiles_path"))
        return params
