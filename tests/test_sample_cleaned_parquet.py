"""Tests for scripts/sample_cleaned_parquet.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from audio_engine.core.manifest import Manifest
from audio_engine.core.sample import Sample

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "sample_cleaned_parquet.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("sample_cleaned_parquet", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_wav(path: Path, seconds: float, sr: int = 16000) -> Path:
    frames = int(round(seconds * sr))
    data = np.full(max(frames, 0), 0.1, dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), data, sr, subtype="PCM_16")
    return path


def _sample(
    path: Path,
    duration: float | None,
    sample_id: str,
    *,
    labels: dict | None = None,
    audio_path: str | None = None,
) -> Sample:
    return Sample(
        id=sample_id,
        source_path=str(path),
        sha256=sample_id,
        audio={"resampled_16k": audio_path if audio_path is not None else str(path)},
        sample_rate=16000,
        duration=duration,
        channels=1,
        labels=labels or {},
    )


def _save_cleaned(tmp_path: Path, name: str, samples: list[Sample]) -> Path:
    path = tmp_path / f"cleaned_{name}.parquet"
    Manifest(samples).save(path)
    return path


@pytest.fixture
def mod():
    return _load_script()


def test_skip_zero_and_missing_duration(mod, tmp_path: Path):
    wav = _write_wav(tmp_path / "ok.wav", 0.2)
    ok = _sample(wav, 0.2, "ok")
    zero = _sample(wav, 0.0, "zero")
    missing = _sample(wav, None, "missing")
    assert mod.skip_reason(ok) is None
    assert mod.skip_reason(zero) == "zero_duration"
    assert mod.skip_reason(missing) == "zero_duration"


def test_skip_empty_and_missing_audio(mod, tmp_path: Path):
    empty = tmp_path / "empty.wav"
    empty.write_bytes(b"")
    missing = tmp_path / "gone.wav"
    wav = _write_wav(tmp_path / "ok.wav", 0.2)
    assert mod.skip_reason(_sample(empty, 1.0, "empty")) == "empty_audio"
    assert mod.skip_reason(_sample(missing, 1.0, "gone")) == "missing_audio"
    assert mod.skip_reason(_sample(wav, 0.2, "ok")) is None


def test_probe_wav_catches_header_only_zero_frames(mod, tmp_path: Path):
    path = tmp_path / "header.wav"
    sf.write(str(path), np.zeros(0, dtype=np.float32), 16000, subtype="PCM_16")
    sample = _sample(path, 1.0, "lie")  # duration 字段撒谎
    assert mod.skip_reason(sample) is None
    assert mod.skip_reason(sample, probe_wav=True) == "zero_duration"


def test_sample_filters_and_writes_usable_source(mod, tmp_path: Path):
    wavs = [_write_wav(tmp_path / f"{i}.wav", 0.15 + 0.01 * i) for i in range(6)]
    a = _save_cleaned(
        tmp_path,
        "a",
        [
            _sample(wavs[0], 0.2, "keep_a0"),
            _sample(wavs[1], 0.0, "drop_zero"),
            _sample(wavs[2], 0.3, "keep_a2"),
            _sample(wavs[3], 0.25, "dup"),
        ],
    )
    b = _save_cleaned(
        tmp_path,
        "b",
        [
            _sample(wavs[3], 0.25, "dup"),
            _sample(wavs[4], None, "drop_none"),
            _sample(wavs[5], 0.4, "keep_b5", labels={"broken": True}),
            _sample(wavs[5], 0.4, "keep_b6"),
        ],
    )
    out = tmp_path / "cleaned_mix.parquet"
    code = mod.main(
        [
            str(a),
            str(b),
            "-n",
            "3",
            "--source-name",
            "mix",
            "--output",
            str(out),
            "--seed",
            "7",
        ]
    )
    assert code == 0
    loaded = Manifest.load(out)
    assert len(loaded) == 3
    ids = {sample.id for sample in loaded.samples}
    assert ids <= {"keep_a0", "keep_a2", "dup", "keep_b6"}
    assert "drop_zero" not in ids and "drop_none" not in ids and "keep_b5" not in ids
    assert len(ids) == 3
    assert all(sample.duration and sample.duration > 0 for sample in loaded.samples)
    assert all(sample.audio.get("resampled_16k") for sample in loaded.samples)
    assert all(
        any(entry.operator == "scripts.sample_cleaned_parquet" for entry in sample.lineage)
        for sample in loaded.samples
    )


def test_seed_is_reproducible(mod, tmp_path: Path):
    wavs = [_write_wav(tmp_path / f"{i}.wav", 0.2) for i in range(8)]
    src = _save_cleaned(
        tmp_path,
        "src",
        [_sample(wavs[i], 0.2, f"id{i}") for i in range(8)],
    )
    out1 = tmp_path / "cleaned_one.parquet"
    out2 = tmp_path / "cleaned_two.parquet"
    args = ["-n", "4", "--source-name", "one", "--seed", "99"]
    assert mod.main([str(src), *args, "--output", str(out1)]) == 0
    assert mod.main([str(src), "-n", "4", "--source-name", "two", "--seed", "99", "--output", str(out2)]) == 0
    ids1 = [s.id for s in Manifest.load(out1).samples]
    ids2 = [s.id for s in Manifest.load(out2).samples]
    assert ids1 == ids2


def test_not_enough_usable_samples(mod, tmp_path: Path):
    wav = _write_wav(tmp_path / "ok.wav", 0.2)
    src = _save_cleaned(
        tmp_path,
        "tiny",
        [_sample(wav, 0.2, "ok"), _sample(wav, 0.0, "zero")],
    )
    out = tmp_path / "cleaned_fail.parquet"
    code = mod.main(
        [str(src), "-n", "5", "--source-name", "fail", "--output", str(out)]
    )
    assert code == 1
    assert not out.exists()


def test_dry_run_does_not_write(mod, tmp_path: Path):
    wav = _write_wav(tmp_path / "ok.wav", 0.2)
    src = _save_cleaned(tmp_path, "src", [_sample(wav, 0.2, f"id{i}") for i in range(3)])
    out = tmp_path / "cleaned_dry.parquet"
    code = mod.main(
        [str(src), "-n", "2", "--source-name", "dry", "--output", str(out), "--dry-run"]
    )
    assert code == 0
    assert not out.exists()


def test_default_output_follows_source_name(mod):
    assert mod.default_output_path("mix10k").as_posix() == (
        "datasets/stage1/cleaned/cleaned_mix10k.parquet"
    )
