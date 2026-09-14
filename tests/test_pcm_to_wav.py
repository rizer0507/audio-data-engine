from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import audio_engine.operators  # noqa: F401
from audio_engine.core.operator import OperatorConfig
from audio_engine.core.registry import OperatorRegistry
from audio_engine.core.sample import Sample
from audio_engine.operators.audio.pcm import looks_like_wav
from audio_engine.operators.audio.resample import DEFAULT_SAMPLE_RATE


def _sine(sr: int, seconds: float = 1.0, freq: float = 440.0) -> np.ndarray:
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _sample(path: Path, sr: int | None = None) -> Sample:
    return Sample(
        id=path.stem,
        source_path=str(path),
        sha256="abc",
        audio={"raw": str(path)},
        sample_rate=sr,
        duration=1.0 if sr else None,
    )


def _cfg(tmp_path: Path, **params: object) -> OperatorConfig:
    merged = {
        "input_audio_key": "raw",
        "output_audio_key": "pcm_wav",
        **params,
    }
    return OperatorConfig(
        params=merged,
        output_dir=tmp_path / "derived",
        cache_dir=tmp_path / "cache",
    )


def test_default_target_is_16k():
    assert DEFAULT_SAMPLE_RATE == 16000


def test_16k_wav_passthrough(tmp_path: Path):
    path = tmp_path / "ok.wav"
    sf.write(str(path), _sine(16000), 16000)
    op = OperatorRegistry.get("audio.pcm_to_wav")
    result = op.process(_sample(path, 16000), _cfg(tmp_path))
    assert result.sample.labels.get("pcm_converted") is False
    assert result.sample.audio["pcm_wav"] == str(path.resolve())
    assert result.sample.sample_rate == 16000
    assert result.sample.quality["pcm_to_wav"] == "passthrough"


def test_8k_wav_probed_and_resampled_to_16k(tmp_path: Path):
    path = tmp_path / "narrow.wav"
    sf.write(str(path), _sine(8000), 8000)
    op = OperatorRegistry.get("audio.pcm_to_wav")
    result = op.process(_sample(path, 8000), _cfg(tmp_path))
    out = Path(result.sample.audio["pcm_wav"])
    assert out != path.resolve()
    info = sf.info(str(out))
    assert info.samplerate == 16000
    assert info.duration == pytest.approx(1.0, abs=0.05)
    assert result.sample.sample_rate == 16000
    assert result.sample.quality["source_sample_rate"] == 8000
    assert result.sample.quality["pcm_to_wav"] == "resampled"
    assert result.sample.labels.get("pcm_converted") is False


def test_16k_headerless_pcm_wraps_at_16k_not_8k(tmp_path: Path):
    pcm_path = tmp_path / "raw16.pcm"
    samples = (_sine(16000) * 32767).astype(np.int16)
    samples.tofile(pcm_path)
    op = OperatorRegistry.get("audio.pcm_to_wav")
    result = op.process(_sample(pcm_path), _cfg(tmp_path))
    out = Path(result.sample.audio["pcm_wav"])
    info = sf.info(str(out))
    assert info.samplerate == 16000
    assert info.duration == pytest.approx(1.0, abs=0.05)
    assert result.sample.labels.get("pcm_converted") is True
    assert result.sample.quality["source_sample_rate"] == 16000
    assert result.sample.quality["probed"] is False


def test_8k_headerless_pcm_needs_explicit_source_rate(tmp_path: Path):
    pcm_path = tmp_path / "raw8.pcm"
    samples = (_sine(8000) * 32767).astype(np.int16)
    samples.tofile(pcm_path)
    op = OperatorRegistry.get("audio.pcm_to_wav")
    result = op.process(
        _sample(pcm_path),
        _cfg(tmp_path, source_sample_rate=8000),
    )
    out = Path(result.sample.audio["pcm_wav"])
    info = sf.info(str(out))
    assert info.samplerate == 16000
    assert info.duration == pytest.approx(1.0, abs=0.05)
    assert result.sample.quality["source_sample_rate"] == 8000
    assert result.sample.quality["pcm_to_wav"] == "resampled"


def test_pcm_extension_with_wav_header_is_probed(tmp_path: Path):
    wav_path = tmp_path / "hidden.wav"
    pcm_path = tmp_path / "hidden.pcm"
    sf.write(str(wav_path), _sine(16000), 16000)
    pcm_path.write_bytes(wav_path.read_bytes())
    assert looks_like_wav(pcm_path)

    op = OperatorRegistry.get("audio.pcm_to_wav")
    result = op.process(_sample(pcm_path), _cfg(tmp_path))
    assert result.sample.labels.get("pcm_converted") is False
    assert result.sample.audio["pcm_wav"] == str(pcm_path.resolve())
    assert result.sample.sample_rate == 16000
    assert result.sample.quality["probed"] is True


def test_stale_8k_metadata_does_not_override_16k_pcm_default(tmp_path: Path):
    pcm_path = tmp_path / "stale.pcm"
    samples = (_sine(16000) * 32767).astype(np.int16)
    samples.tofile(pcm_path)
    op = OperatorRegistry.get("audio.pcm_to_wav")
    result = op.process(_sample(pcm_path, sr=8000), _cfg(tmp_path))
    info = sf.info(str(result.sample.audio["pcm_wav"]))
    assert info.samplerate == 16000
    assert info.duration == pytest.approx(1.0, abs=0.05)
    assert result.sample.quality["source_sample_rate"] == 16000


def test_pcm_extension_with_8k_wav_header_resamples(tmp_path: Path):
    wav_path = tmp_path / "hidden8.wav"
    pcm_path = tmp_path / "hidden8.pcm"
    sf.write(str(wav_path), _sine(8000), 8000)
    pcm_path.write_bytes(wav_path.read_bytes())

    op = OperatorRegistry.get("audio.pcm_to_wav")
    result = op.process(_sample(pcm_path), _cfg(tmp_path))
    out = Path(result.sample.audio["pcm_wav"])
    assert sf.info(str(out)).samplerate == 16000
    assert result.sample.quality["source_sample_rate"] == 8000
    assert result.sample.quality["pcm_to_wav"] == "resampled"
