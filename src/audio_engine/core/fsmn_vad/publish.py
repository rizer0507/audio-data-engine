"""Shared VAD segment WAV write + Sample construction (FSMN + shenpi HTTP)."""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Any, Iterable

from audio_engine.core.artifacts import atomic_path
from audio_engine.core.audio_scene import SCENE_FENSHEN, infer_legacy_fenshen_scene
from audio_engine.core.fsmn_vad.identity import (
    build_segment_labels,
    segment_content_sha256,
    stable_segment_id,
)
from audio_engine.core.fsmn_vad.types import SR, VadKeptSegment
from audio_engine.core.manifest import file_sha256
from audio_engine.core.sample import Sample

PARENT_METADATA_KEYS = (
    "source_sample_rate",
    "normalized_sample_rate",
    "source_channels",
    "selected_channel",
    "channel_policy",
    "sample_rate_source",
    "audio_scene",
    "audio_scene_rule_version",
    "audio_scene_prefix",
)


def read_pcm16le_mono(path: Path, *, expect_sr: int = SR) -> tuple[bytes, int, dict[str, Any]]:
    """Load audio as 16-bit mono PCM16LE bytes at expect_sr."""
    import soundfile as sf

    data, sr = sf.read(str(path), always_2d=True, dtype="int16")
    sr = int(sr)
    mono = data[:, 0]
    meta = {"source_sr": sr, "frames": int(mono.shape[0]), "channels_read": int(data.shape[1])}
    if sr != expect_sr:
        raise ValueError(
            f"normalized audio sample_rate={sr} != {expect_sr}; "
            "run audio.resample before VAD segment publish"
        )
    return mono.astype("<i2").tobytes(), sr, meta


def write_wav_pcm16le(path: Path, pcm16le: bytes, *, sample_rate: int = SR) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(path) as tmp:
        with wave.open(str(tmp), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm16le)
    return file_sha256(path)


def inherit_parent_metadata(parent: Sample, labels: dict[str, Any]) -> dict[str, Any]:
    """Copy selected parent labels/quality onto segment labels; fill legacy fenshen scene."""
    for key in PARENT_METADATA_KEYS:
        if key in labels:
            continue
        if key in parent.labels:
            labels[key] = parent.labels[key]
        elif key in parent.quality:
            labels[key] = parent.quality[key]
    if "audio_scene" not in labels:
        legacy = infer_legacy_fenshen_scene(parent.id, parent.labels)
        if legacy == SCENE_FENSHEN:
            labels["audio_scene"] = SCENE_FENSHEN
    return labels


def publish_kept_segments(
    *,
    parent: Sample,
    kept: Iterable[VadKeptSegment],
    source_audio_id: str,
    source_pcm_sha256: str,
    normalized_path: Path,
    normalized_sha256: str,
    channel_index: int,
    segments_dir: Path,
    processing_version: str,
    config_sha256: str,
    operator_full_name: str,
    operator_version: str,
    input_key: str,
    output_key: str,
    model_fingerprint: dict[str, Any] | None = None,
    lineage_extra: dict[str, Any] | None = None,
) -> tuple[list[Sample], list[str]]:
    """Write segment WAVs and construct child Samples. Returns (children, published_ids)."""
    out_segments: list[Sample] = []
    published_ids: list[str] = []
    for idx, seg in enumerate(kept):
        pcm_sha = segment_content_sha256(seg.pcm16le)
        seg_id = stable_segment_id(
            source_audio_id=source_audio_id,
            normalized_sha256=normalized_sha256,
            channel_index=channel_index,
            start_sample=seg.start_sample,
            end_sample=seg.end_sample,
            vad_processing_version=processing_version,
            config_sha256=config_sha256,
        )
        out_path = segments_dir / seg_id[:2] / f"{seg_id}.wav"
        if out_path.is_file():
            wav_sha = file_sha256(out_path)
        else:
            wav_sha = write_wav_pcm16le(out_path, seg.pcm16le)

        labels = build_segment_labels(
            source_audio_id=source_audio_id,
            source_pcm_sha256=source_pcm_sha256,
            normalized_sha256=normalized_sha256,
            normalized_audio_path=str(normalized_path.resolve()),
            channel_index=channel_index,
            segment=seg,
            segment_sha256=pcm_sha,
            segment_index=idx,
            vad_processing_version=processing_version,
            config_sha256=config_sha256,
            model_fingerprint=model_fingerprint,
        )
        labels["segment_id"] = seg_id
        labels["segment_wav_sha256"] = wav_sha
        inherit_parent_metadata(parent, labels)

        child = Sample(
            id=seg_id,
            source_path=str(normalized_path.resolve()),
            sha256=pcm_sha,
            sample_rate=SR,
            channels=1,
            duration=seg.duration_ms / 1000.0,
            audio={
                output_key: str(out_path.resolve()),
                "normalized_16k": str(normalized_path.resolve()),
                "parent_resampled_16k": str(normalized_path.resolve()),
            },
            labels=labels,
            quality={
                "vad_end_reason": seg.end_reason,
                "vad_clip_branch": seg.clip_branch,
                "clip_start_sample": seg.start_sample,
                "clip_end_sample": seg.end_sample,
            },
        )
        if "raw" in parent.audio:
            child.audio["parent_raw"] = parent.audio["raw"]
        elif parent.source_path:
            child.audio["parent_raw"] = parent.source_path

        child.mark_completed(operator_full_name)
        params = {
            "vad_processing_version": processing_version,
            "config_sha256": config_sha256,
            "start_sample": seg.start_sample,
            "end_sample": seg.end_sample,
            "source_audio_id": source_audio_id,
        }
        if lineage_extra:
            params.update(lineage_extra)
        child.add_lineage(
            operator=operator_full_name,
            version=operator_version,
            params=params,
            input_key=input_key,
            output_key=output_key,
            output_path=str(out_path),
        )
        out_segments.append(child)
        published_ids.append(seg_id)
    return out_segments, published_ids
