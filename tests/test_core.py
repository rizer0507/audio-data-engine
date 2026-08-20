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
    assert result.sample.labels.get("resampled") is True


def test_pcm_and_resample_passthrough(sample_wav: Path):
    """Already-wav / already-16k should not convert or rewrite."""
    sample = Sample(
        id="test001",
        source_path=str(sample_wav),
        sha256="abc",
        audio={"raw": str(sample_wav)},
        sample_rate=16000,
        duration=1.0,
    )
    pcm = OperatorRegistry.get("audio.pcm_to_wav")
    pcm_cfg = OperatorConfig(
        params={
            "sample_rate": 8000,
            "input_audio_key": "raw",
            "output_audio_key": "pcm_wav",
        },
        output_dir=sample_wav.parent / "derived",
        cache_dir=sample_wav.parent / "cache_pcm",
    )
    pcm_result = pcm.process(sample, pcm_cfg)
    assert pcm_result.sample.labels.get("pcm_converted") is False
    assert pcm_result.sample.audio["pcm_wav"] == str(sample_wav.resolve())

    rs = OperatorRegistry.get("audio.resample")
    rs_cfg = OperatorConfig(
        params={
            "sample_rate": 16000,
            "input_audio_key": "pcm_wav",
            "output_audio_key": "resampled_16k",
        },
        output_dir=sample_wav.parent / "derived",
        cache_dir=sample_wav.parent / "cache_rs",
    )
    rs_result = rs.process(pcm_result.sample, rs_cfg)
    assert rs_result.sample.labels.get("resampled") is False
    assert rs_result.sample.audio["resampled_16k"] == str(sample_wav.resolve())
    assert not (sample_wav.parent / "derived" / "resample_16k").exists()


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


def test_probe_and_select(sample_wav: Path, tmp_path: Path):
    from audio_engine.core.pipeline import PipelineConfig, PipelineRunner, PipelineStep

    # corrupt wav should fail probe → broken → filtered out（输入已是 VAD 后数据，不测 speech_ratio）
    broken = tmp_path / "broken.wav"
    broken.write_bytes(b"not a real wav file")

    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    import shutil

    shutil.copy2(sample_wav, audio_dir / "good.wav")
    shutil.copy2(broken, audio_dir / "broken.wav")

    cfg = PipelineConfig(
        name="test_cleaning",
        input_manifest="",
        source_dir=str(audio_dir),
        steps=[
            PipelineStep(name="ingest", operator="ingest.scan"),
            PipelineStep(
                name="resample",
                operator="audio.resample",
                params={
                    "sample_rate": 16000,
                    "input_audio_key": "raw",
                    "output_audio_key": "resampled_16k",
                },
            ),
            PipelineStep(
                name="probe",
                operator="quality.probe",
                params={"input_audio_key": "resampled_16k"},
            ),
            PipelineStep(
                name="audio_pass",
                operator="quality.filter",
                params={
                    "expr": "label_broken != True and duration > 0",
                    "label_key": "audio_pass",
                },
            ),
            PipelineStep(
                name="select_pass",
                operator="quality.select",
                params={"expr": "label_audio_pass == True"},
            ),
        ],
        output_dir=tmp_path / "derived",
        cache_dir=tmp_path / "cache",
        runs_dir=tmp_path / "runs",
    )
    result = PipelineRunner(cfg).run()
    assert len(result) == 1
    assert result.samples[0].id == "good"
    assert result.samples[0].labels.get("audio_pass") is True


def test_resolve_source(tmp_path: Path):
    from audio_engine.core.source import resolve_source_input

    resources = tmp_path / "resources"
    resources.mkdir()
    manifest = resources / "manifest.yaml"
    manifest.write_text(
        "sources:\n"
        "  - source_id: source_A\n"
        "    source_name: A\n"
        "    ingested_at: '2026-08-20T10:00:00+08:00'\n"
        "    origin: test\n"
        "    path: D:/Data/batch_A\n",
        encoding="utf-8",
    )
    # no sample index → fall back to source_dir
    resolved = resolve_source_input("source_A", resources_manifest=manifest)
    assert resolved["source_dir"] == "D:/Data/batch_A"

    samples_dir = tmp_path / "resources" / "sources" / "source_A"
    samples_dir.mkdir(parents=True)
    (samples_dir / "samples.jsonl").write_text(
        '{"id":"x","source_path":"a.wav","sha256":"1"}\n',
        encoding="utf-8",
    )
    # monkeypatch cwd-relative paths used by resolver: create under real relative path
    # Prefer testing lookup with absolute by temporarily chdir
    import os

    old = os.getcwd()
    os.chdir(tmp_path)
    try:
        resolved2 = resolve_source_input("source_A", resources_manifest=manifest)
        assert resolved2["manifest"].endswith("samples.jsonl")
    finally:
        os.chdir(old)
