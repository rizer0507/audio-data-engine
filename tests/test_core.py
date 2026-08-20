from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import audio_engine.operators  # noqa: F401
from audio_engine.core.manifest import Manifest
from audio_engine.core.operator import OperatorConfig
from audio_engine.core.registry import OperatorRegistry
from audio_engine.core.sample import Sample


@pytest.fixture
def sample_wav(tmp_path: Path) -> Path:
    sr = 16000
    t = np.linspace(0, 1, sr, endpoint=False)
    data = 0.3 * np.sin(2 * np.pi * 440 * t)
    path = tmp_path / "test001.wav"
    sf.write(str(path), data.astype(np.float32), sr)
    return path


def test_ingest(sample_wav: Path):
    manifest = Manifest.ingest(sample_wav.parent)
    assert len(manifest) == 1
    assert manifest.samples[0].sample_rate == 16000
    assert manifest.samples[0].duration == pytest.approx(1.0, abs=0.01)


def test_manifest_roundtrip(sample_wav: Path, tmp_path: Path):
    manifest = Manifest.ingest(sample_wav.parent)
    out = tmp_path / "test.parquet"
    manifest.save(out)
    loaded = Manifest.load(out)
    assert len(loaded) == 1
    assert loaded.samples[0].id == manifest.samples[0].id


def test_resample_operator(sample_wav: Path):
    sample = Sample(
        id="test001",
        source_path=str(sample_wav),
        sha256="abc",
        audio={"raw": str(sample_wav)},
        sample_rate=16000,
        duration=1.0,
    )
    op = OperatorRegistry.get("audio.resample")
    config = OperatorConfig(
        params={"sample_rate": 8000, "input_audio_key": "raw", "output_audio_key": "resampled_8k"},
        output_dir=sample_wav.parent / "derived",
        cache_dir=sample_wav.parent / "cache",
    )
    result = op.process(sample, config)
    assert "resampled_8k" in result.sample.audio


def test_cache_hit(sample_wav: Path):
    sample = Sample(
        id="test001",
        source_path=str(sample_wav),
        sha256="deadbeef",
        audio={"raw": str(sample_wav)},
    )
    op = OperatorRegistry.get("quality.snr")
    config = OperatorConfig(
        params={"input_audio_key": "raw"},
        output_dir=sample_wav.parent / "derived",
        cache_dir=sample_wav.parent / "cache",
    )
    r1 = op.process(sample, config)
    r2 = op.process(sample, config)
    assert r1.cache_hit is False
    assert r2.cache_hit is True


def test_operator_registry():
    names = OperatorRegistry.list_operators()
    assert "audio.resample" in names
    assert "asr.qwen" in names
    assert "asr.sensevoice" in names
    assert "ingest.scan" in names


def test_ingest_pipeline(sample_wav: Path, tmp_path: Path):
    """Ingest runs through the unified PipelineRunner -> Operator path."""
    from audio_engine.core.pipeline import PipelineConfig, PipelineRunner, PipelineStep

    cfg = PipelineConfig(
        name="test_ingest",
        input_manifest="",
        source_dir=str(sample_wav.parent),
        steps=[PipelineStep(name="ingest", operator="ingest.scan")],
        output_dir=tmp_path / "derived",
        cache_dir=tmp_path / "cache",
        runs_dir=tmp_path / "runs",
    )
    runner = PipelineRunner(cfg)
    result = runner.run()

    assert len(result) == 1
    sample = result.samples[0]
    assert sample.sample_rate == 16000
    assert sample.is_completed("ingest.scan")
    assert sample.lineage[0].operator == "ingest.scan"
    assert runner.metrics.processed == 1


def test_filter_manifest(sample_wav: Path):
    manifest = Manifest.ingest(sample_wav.parent)
    manifest.samples[0].labels["badcase"] = "noise"
    filtered = manifest.filter("label_badcase == 'noise'")
    assert len(filtered) == 1

    empty = manifest.filter("label_badcase == 'silence'")
    assert len(empty) == 0
