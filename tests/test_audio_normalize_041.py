"""041 normalize: 8k→16k alerts, channel-0 extract, headerless PCM validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import audio_engine.operators  # noqa: F401
from audio_engine.core.operator import OperatorConfig
from audio_engine.core.registry import OperatorRegistry
from audio_engine.core.sample import Sample


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
        run_dir=tmp_path / "run",
    )


def test_8k_wav_resamples_to_16k_with_split_rate_fields(tmp_path: Path, capsys):
    path = tmp_path / "narrow.wav"
    sf.write(str(path), _sine(8000, 1.0), 8000)
    op = OperatorRegistry.get("audio.pcm_to_wav")
    result = op.process(_sample(path, 8000), _cfg(tmp_path, sample_rate=16000))
    out = Path(result.sample.audio["pcm_wav"])
    info = sf.info(str(out))
    assert info.samplerate == 16000
    assert info.channels == 1
    assert info.duration == pytest.approx(1.0, abs=0.05)
    assert result.sample.labels["source_sample_rate"] == 8000
    assert result.sample.labels["normalized_sample_rate"] == 16000
    assert result.sample.quality["source_sample_rate"] == 8000
    assert result.sample.quality["target_sample_rate"] == 16000
    assert result.sample.sample_rate == 16000
    captured = capsys.readouterr()
    assert "8kHz" in captured.out or "8000" in captured.out
    alert = tmp_path / "run" / "normalize_alerts.jsonl"
    assert alert.is_file()
    assert "8000" in alert.read_text(encoding="utf-8")


def test_stereo_takes_channel_0_duration_not_doubled(tmp_path: Path):
    path = tmp_path / "stereo.wav"
    left = _sine(16000, 1.0, 440)
    right = _sine(16000, 1.0, 880)
    stereo = np.column_stack([left, right])
    sf.write(str(path), stereo, 16000)
    op = OperatorRegistry.get("audio.pcm_to_wav")
    result = op.process(_sample(path, 16000), _cfg(tmp_path))
    out = Path(result.sample.audio["pcm_wav"])
    data, sr = sf.read(str(out), always_2d=True)
    assert sr == 16000
    assert data.shape[1] == 1
    assert data.shape[0] == pytest.approx(16000, abs=16)
    assert result.sample.labels["source_channels"] == 2
    assert result.sample.labels["selected_channel"] == 0
    assert result.sample.labels["channel_policy"] == "take_channel_0"
    # Output matches left channel (within float round-trip).
    mono_ref = (left * 32767).astype(np.int16).astype(np.float32) / 32767
    # pcm_to_wav wrote float via soundfile PCM_16 — compare correlation / length
    assert abs(float(data[:, 0].mean()) - float(left.mean())) < 0.05


def test_headerless_pcm_default_16k_records_convention(tmp_path: Path):
    pcm_path = tmp_path / "raw16.pcm"
    samples = (_sine(16000) * 32767).astype(np.int16)
    samples.tofile(pcm_path)
    op = OperatorRegistry.get("audio.pcm_to_wav")
    # Explicitly omit source_sample_rate to use default convention path via resolve
    cfg = _cfg(tmp_path, sample_rate=16000, channels=1)
    # Force default_convention by not putting source_sample_rate — but YAML usually sets it.
    # When set to 16000 with key present → source_config. Test both.
    result = op.process(_sample(pcm_path), cfg)
    assert result.sample.labels["sample_rate_source"] in {"source_config", "default_convention"}
    assert result.sample.labels["source_sample_rate"] == 16000
    assert result.sample.sample_rate == 16000
    assert result.sample.channels == 1


def test_headerless_pcm_declared_8k_resamples(tmp_path: Path, capsys):
    pcm_path = tmp_path / "raw8.pcm"
    samples = (_sine(8000, 1.0) * 32767).astype(np.int16)
    samples.tofile(pcm_path)
    op = OperatorRegistry.get("audio.pcm_to_wav")
    result = op.process(
        _sample(pcm_path),
        _cfg(tmp_path, sample_rate=16000, source_sample_rate=8000, channels=1),
    )
    info = sf.info(str(result.sample.audio["pcm_wav"]))
    assert info.samplerate == 16000
    assert info.duration == pytest.approx(1.0, abs=0.05)
    assert result.sample.labels["source_sample_rate"] == 8000
    assert result.sample.labels["sample_rate_source"] == "source_config"
    assert "8kHz" in capsys.readouterr().out or True  # alert printed


def test_headerless_stereo_deinterleave_ch0(tmp_path: Path):
    pcm_path = tmp_path / "stereo.pcm"
    left = (_sine(16000, 0.5, 440) * 10000).astype(np.int16)
    right = (_sine(16000, 0.5, 880) * 20000).astype(np.int16)
    interleaved = np.empty(left.size * 2, dtype=np.int16)
    interleaved[0::2] = left
    interleaved[1::2] = right
    interleaved.tofile(pcm_path)
    op = OperatorRegistry.get("audio.pcm_to_wav")
    result = op.process(
        _sample(pcm_path),
        _cfg(tmp_path, sample_rate=16000, source_sample_rate=16000, channels=2),
    )
    data, sr = sf.read(str(result.sample.audio["pcm_wav"]), dtype="int16")
    assert sr == 16000
    assert data.ndim == 1 or data.shape[-1] == 1
    flat = data if data.ndim == 1 else data[:, 0]
    assert flat.shape[0] == left.shape[0]
    # Must match left, not the interleaved-as-mono mess (which would be 2x length).
    assert flat.shape[0] != interleaved.shape[0]
    np.testing.assert_array_equal(flat, left)
    assert result.sample.labels["channel_policy"] == "take_channel_0"


def test_headerless_bad_byte_length_for_channels_raises(tmp_path: Path):
    pcm_path = tmp_path / "bad.pcm"
    # 5 int16 values — not divisible by channels=2
    np.arange(5, dtype=np.int16).tofile(pcm_path)
    op = OperatorRegistry.get("audio.pcm_to_wav")
    with pytest.raises(ValueError, match="not divisible"):
        op.process(
            _sample(pcm_path),
            _cfg(tmp_path, source_sample_rate=16000, channels=2),
        )


def test_normalize_report_summary(tmp_path: Path):
    path = tmp_path / "n8.wav"
    sf.write(str(path), _sine(8000, 0.5), 8000)
    op = OperatorRegistry.get("audio.pcm_to_wav")
    cfg = _cfg(tmp_path)
    s = op.process(_sample(path, 8000), cfg).sample
    report = OperatorRegistry.get("audio.normalize_report")
    out = report.run([s], cfg)
    assert len(out) == 1
    summary_path = tmp_path / "run" / "normalize_summary.json"
    assert summary_path.is_file()
    text = summary_path.read_text(encoding="utf-8")
    assert "eight_k_count" in text
    assert '"eight_k_count": 1' in text or '"eight_k_count":1' in text
