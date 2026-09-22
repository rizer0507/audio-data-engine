"""Silero VAD audio I/O uses soundfile (not torchaudio/torchcodec)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from audio_engine.core.quality.silero_vad import (
    SileroVadConfig,
    SileroVadSession,
    _classify_speech_state,
    read_wav_for_silero,
)


def _write_wav(path: Path, *, sr: int, channels: int = 1, seconds: float = 0.2) -> None:
    n = max(1, int(sr * seconds))
    t = np.linspace(0.0, seconds, n, endpoint=False, dtype=np.float32)
    tone = (0.2 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)
    if channels == 1:
        data = tone
    else:
        data = np.column_stack([tone * (0.5 + 0.5 * c / channels) for c in range(channels)])
    sf.write(str(path), data, sr)


def test_classify_speech_state_thresholds() -> None:
    cfg = SileroVadConfig(
        min_speech_ratio_present=0.10,
        max_speech_ratio_absent=0.02,
        min_speech_duration_ms=120,
    )
    assert (
        _classify_speech_state(speech_ratio=0.2, speech_duration_ms=200, cfg=cfg)
        == "present"
    )
    assert (
        _classify_speech_state(speech_ratio=0.01, speech_duration_ms=50, cfg=cfg)
        == "absent"
    )
    assert (
        _classify_speech_state(speech_ratio=0.05, speech_duration_ms=200, cfg=cfg)
        == "uncertain"
    )


def test_analyze_missing_file_is_failed_not_forged() -> None:
    session = SileroVadSession(SileroVadConfig())
    out = session.analyze("/nonexistent/path/xxx.wav")
    assert out["vad_status"] == "failed"
    assert out["vad_speech_state"] == "failed"
    assert str(out["vad_error"]).startswith("audio_unreadable:")


def test_read_wav_for_silero_mono_16k(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    wav = tmp_path / "a.wav"
    _write_wav(wav, sr=16000, channels=1, seconds=0.25)
    tensor = read_wav_for_silero(wav, sampling_rate=16000)
    assert tensor.ndim == 1
    assert tensor.dtype == torch.float32
    assert tensor.shape[0] == 4000


def test_read_wav_for_silero_stereo_mean_and_resample(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    wav = tmp_path / "stereo_8k.wav"
    _write_wav(wav, sr=8000, channels=2, seconds=0.5)
    tensor = read_wav_for_silero(wav, sampling_rate=16000)
    assert tensor.ndim == 1
    assert tensor.dtype == torch.float32
    # 0.5s @ 8k → 4000; resample to 16k → ~8000
    assert abs(int(tensor.shape[0]) - 8000) <= 2
