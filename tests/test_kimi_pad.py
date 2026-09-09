from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import audio_engine.operators  # noqa: F401
from audio_engine.core.operator import OperatorConfig
from audio_engine.core.pipeline import ExecutionConfig, PipelineConfig, PipelineRunner, PipelineStep
from audio_engine.core.manifest import Manifest
from audio_engine.core.registry import OperatorRegistry
from audio_engine.core.sample import Sample
from audio_engine.operators.audio.kimi_pad import (
    KIMI_PAD_BUCKETS,
    plan_kimi_pad,
    resolve_kimi_pad_target,
)


def _write_wav(path: Path, seconds: float, sr: int = 16000, channels: int = 1) -> Path:
    frames = int(round(seconds * sr))
    if channels == 1:
        data = np.full(frames, 0.1, dtype=np.float32)
    else:
        data = np.full((frames, channels), 0.1, dtype=np.float32)
    sf.write(str(path), data, sr, subtype="PCM_16")
    return path


def _sample(path: Path, duration: float, sample_id: str = "s0") -> Sample:
    return Sample(
        id=sample_id,
        source_path=str(path),
        sha256="abc123",
        audio={"resampled_16k": str(path)},
        sample_rate=16000,
        duration=duration,
        channels=1,
    )


def _config(tmp_path: Path, **params) -> OperatorConfig:
    return OperatorConfig(
        params={
            "input_audio_key": "resampled_16k",
            "output_audio_key": "kimi_padded_16k",
            **params,
        },
        cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "derived",
    )


@pytest.mark.parametrize(
    ("duration", "expected"),
    [
        (0.0, 3),
        (3.0, 3),
        (3.0001, 6),
        (6.0, 6),
        (6.1, 10),
        (10.0, 10),
        (10.1, 15),
        (15.0, 15),
        (15.1, 30),
        (30.0, 30),
        (30.0001, None),
        (45.0, None),
    ],
)
def test_kimi_pad_bucket_table(duration: float, expected: int | None):
    assert resolve_kimi_pad_target(duration) == expected
    assert KIMI_PAD_BUCKETS == (3, 6, 10, 15, 30)


def test_kimi_pad_writes_new_wav_and_keeps_original(tmp_path: Path):
    src = _write_wav(tmp_path / "short.wav", 1.5)
    original = src.read_bytes()
    sample = _sample(src, 1.5)
    result = OperatorRegistry.get("audio.kimi_duration_pad").process(sample, _config(tmp_path))

    padded = Path(result.sample.audio["kimi_padded_16k"])
    assert padded != src
    assert padded.is_file()
    info = sf.info(str(padded))
    assert info.samplerate == 16000
    assert info.frames == 3 * 16000
    assert result.sample.duration == 1.5
    assert result.sample.audio["resampled_16k"] == str(src)
    assert src.read_bytes() == original
    assert result.sample.labels["kimi_pad_mode"] == "padded"
    assert result.sample.labels["kimi_pad_target_s"] == 3


def test_kimi_pad_passthrough_when_already_at_bucket(tmp_path: Path):
    src = _write_wav(tmp_path / "exact3.wav", 3.0)
    sample = _sample(src, 3.0)
    result = OperatorRegistry.get("audio.kimi_duration_pad").process(sample, _config(tmp_path))

    assert Path(result.sample.audio["kimi_padded_16k"]).resolve() == src.resolve()
    assert result.sample.labels["kimi_pad_mode"] == "passthrough"
    assert result.sample.duration == 3.0
    derived = list((tmp_path / "derived").rglob("*.wav")) if (tmp_path / "derived").exists() else []
    assert derived == []


def test_kimi_pad_over_30s_does_not_truncate(tmp_path: Path):
    src = _write_wav(tmp_path / "long.wav", 31.0)
    sample = _sample(src, 31.0)
    result = OperatorRegistry.get("audio.kimi_duration_pad").process(sample, _config(tmp_path))

    assert Path(result.sample.audio["kimi_padded_16k"]).resolve() == src.resolve()
    assert result.sample.labels["kimi_pad_mode"] == "over_30s"
    assert result.sample.labels["kimi_pad_target_s"] is None
    assert result.sample.duration == 31.0
    assert sf.info(str(src)).frames == 31 * 16000


def test_kimi_pad_preserves_stereo_and_sample_rate(tmp_path: Path):
    src = _write_wav(tmp_path / "stereo.wav", 2.0, channels=2)
    sample = _sample(src, 2.0)
    result = OperatorRegistry.get("audio.kimi_duration_pad").process(sample, _config(tmp_path))
    info = sf.info(result.sample.audio["kimi_padded_16k"])
    assert info.channels == 2
    assert info.samplerate == 16000
    assert info.frames == 3 * 16000


def test_kimi_pad_cache_includes_bucket_and_reuses_file(tmp_path: Path):
    src = _write_wav(tmp_path / "mid.wav", 4.0)
    sample = _sample(src, 4.0)
    operator = OperatorRegistry.get("audio.kimi_duration_pad")
    first = operator.process(sample, _config(tmp_path))
    second = operator.process(sample, _config(tmp_path))
    assert first.sample.labels["kimi_pad_target_s"] == 6
    assert second.cache_hit
    assert first.sample.audio["kimi_padded_16k"] == second.sample.audio["kimi_padded_16k"]

    other_key = operator.compute_cache_key(sample, _config(tmp_path, buckets=[3, 10, 30]))
    default_key = operator.compute_cache_key(sample, _config(tmp_path))
    assert other_key != default_key


def test_kimi_pad_failure_marks_sample_only(tmp_path: Path):
    missing = tmp_path / "missing.wav"
    sample = _sample(missing, 1.0)
    with pytest.raises(Exception):
        OperatorRegistry.get("audio.kimi_duration_pad").process(sample, _config(tmp_path))


def test_kimi_pad_then_kimi_asr_pipeline_keeps_duration(tmp_path: Path):
    src = _write_wav(tmp_path / "clip.wav", 2.2)
    sample = _sample(src, 2.2, sample_id="clip")
    input_path = tmp_path / "input.parquet"
    Manifest([sample]).save(input_path)
    config = PipelineConfig(
        name="kimi_pad_asr_test",
        input_manifest=str(input_path),
        output_manifest=str(tmp_path / "out.parquet"),
        steps=[
            PipelineStep(
                name="kimi_duration_pad",
                operator="audio.kimi_duration_pad",
                params={
                    "input_audio_key": "resampled_16k",
                    "output_audio_key": "kimi_padded_16k",
                },
            ),
            PipelineStep(
                name="kimi_asr",
                operator="asr.kimi_batch",
                params={
                    "input_audio_key": "kimi_padded_16k",
                    "mock": True,
                    "concurrency": 1,
                },
            ),
        ],
        cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "derived",
        runs_dir=tmp_path / "runs",
        execution=ExecutionConfig(executor="sequential", workers=1, fail_fast=False),
    )
    result = PipelineRunner(config).run()
    assert len(result) == 1
    sample = result.samples[0]
    assert sample.duration == 2.2
    assert sample.audio["resampled_16k"] == str(src)
    assert "kimi_padded_16k" in sample.audio
    assert sample.get_transcript_text("kimi").startswith("[mock:kimi:")
    assert plan_kimi_pad(src, duration=2.2).target_s == 3
