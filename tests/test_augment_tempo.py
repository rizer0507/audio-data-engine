"""047 pitch-preserving tempo augmentation — unit + small integration tests."""

from __future__ import annotations

import math
import shutil
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

HAS_FFMPEG = shutil.which("ffmpeg") is not None

from audio_engine.core.augmentation.ffmpeg_atempo import (
    atempo_filter_graph,
    decompose_atempo,
    probe_ffmpeg,
    run_atempo,
)
from audio_engine.core.augmentation.params import (
    normalize_speed_factor,
    resolve_tempo_params,
)
from audio_engine.core.augmentation.preflight import (
    AugmentPreflightError,
    detect_already_augmented,
    validate_input_snapshot,
)
from audio_engine.core.augmentation.publish import AugmentBatchIncomplete, publish_batch_artifacts
from audio_engine.core.manifest import Manifest, file_sha256
from audio_engine.core.operator import OperatorConfig
from audio_engine.core.sample import Sample
from audio_engine.operators.augmentation.tempo import TempoAugmentOperator


def _write_tone(
    path: Path,
    *,
    sr: int = 16000,
    duration: float = 1.0,
    freq: float = 440.0,
    channels: int = 1,
) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    mono = 0.2 * np.sin(2 * np.pi * freq * t).astype(np.float32)
    if channels == 1:
        data = mono
    else:
        data = np.stack([mono, mono * 0.5], axis=1)
    sf.write(str(path), data, sr, subtype="PCM_16")
    return {
        "path": path,
        "sr": sr,
        "duration": duration,
        "freq": freq,
        "channels": channels,
        "sha256": file_sha256(path),
    }


def _dominant_freq(path: Path, sr_hint: int | None = None) -> float:
    data, sr = sf.read(str(path), always_2d=False)
    if data.ndim > 1:
        data = data[:, 0]
    # Skip edges; use FFT peak.
    n = len(data)
    if n < 64:
        return float("nan")
    window = data[n // 8 : -n // 8] if n > 256 else data
    spectrum = np.abs(np.fft.rfft(window * np.hanning(len(window))))
    peak = int(np.argmax(spectrum[1:]) + 1)
    freqs = np.fft.rfftfreq(len(window), d=1.0 / sr)
    return float(freqs[peak])


def test_normalize_speed_factor_rejects_invalid():
    with pytest.raises(ValueError):
        normalize_speed_factor(1.0)
    with pytest.raises(ValueError):
        normalize_speed_factor(0)
    with pytest.raises(ValueError):
        normalize_speed_factor(-1.2)
    with pytest.raises(ValueError):
        normalize_speed_factor(True)
    with pytest.raises(ValueError):
        normalize_speed_factor("nan")
    assert normalize_speed_factor("1.20") == normalize_speed_factor(1.2)
    assert normalize_speed_factor(1.25) != normalize_speed_factor(1.249)


def test_decompose_atempo_chains_above_two():
    stages = decompose_atempo(3.0)
    assert all(s <= 2.0 + 1e-9 for s in stages)
    product = math.prod(stages)
    assert abs(product - 3.0) / 3.0 < 1e-6
    assert "atempo=2" in atempo_filter_graph(3.0)


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not on PATH")
def test_ffmpeg_atempo_preserves_pitch_and_channels(tmp_path: Path):
    probe_ffmpeg()
    tone = _write_tone(tmp_path / "in.wav", duration=2.0, freq=440.0, channels=2)
    out = tmp_path / "out.wav"
    params = resolve_tempo_params({"speed_factor": 1.2, "backend_threads": 1})
    meta = run_atempo(input_path=tone["path"], output_path=out, params=params)
    assert "atempo=" in meta["filter_graph"]
    info = sf.info(str(out))
    assert info.samplerate == tone["sr"]
    assert info.channels == 2
    expected = tone["duration"] / 1.2
    assert abs(info.duration - expected) <= max(0.05, 0.03 * expected)
    f0 = _dominant_freq(out)
    # Pitch should stay near 440 Hz (atempo); resample-style would shift ~528 Hz.
    assert abs(f0 - 440.0) / 440.0 <= 0.02


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not on PATH")
def test_tempo_operator_batch_conservation(tmp_path: Path):
    batch = "aug_demo"
    audio_dir = tmp_path / "audio"
    samples = []
    for i in range(3):
        tone = _write_tone(audio_dir / f"s{i}.wav", duration=1.0 + 0.2 * i, freq=330.0)
        samples.append(
            Sample(
                id=f"s{i}",
                source_path=str(tone["path"]),
                sha256=tone["sha256"],
                sample_rate=16000,
                channels=1,
                duration=tone["duration"],
                audio={"resampled_16k": str(tone["path"]), "raw": str(tone["path"])},
                labels={"batch": batch, "gold_text": f"text-{i}", "group_id": "g1"},
                transcripts={"qwen": {"text": "old", "model": "qwen"}},
                quality={"dnsmos": {"ovr": 3.0}},
                status={"asr.qwen": "completed"},
            )
        )
    manifest_path = tmp_path / "segments_aug_demo.parquet"
    Manifest(samples).save(manifest_path)

    product_root = tmp_path / "augmented" / batch / "tempo_1p2_test"
    op = TempoAugmentOperator()
    cfg = OperatorConfig(
        params={
            "speed_factor": 1.2,
            "source_batch": batch,
            "input_manifest": str(manifest_path),
            "product_root": str(product_root),
            "recipe_digest": "testdigest000001",
            "derived_batch": f"{batch}__tempo_1p2_testdigest",
            "workers": 2,
            "max_in_flight": 2,
            "backend_threads": 1,
        },
        output_dir=tmp_path / "derived",
        cache_dir=tmp_path / "cache",
        run_dir=tmp_path / "run",
        force=False,
        mock=False,
    )
    out = op.run(samples, cfg)
    assert len(out) == 3
    assert (product_root / "manifest.parquet").is_file()
    assert (product_root / "results.parquet").is_file()
    assert (product_root / "recipe.json").is_file()
    formal = Manifest.load(product_root / "manifest.parquet")
    assert len(formal) == 3
    for sample in out:
        assert sample.id.startswith("tempo_")
        assert sample.status.get("augmentation.tempo") == "completed"
        assert "asr.qwen" not in sample.status
        assert "qwen" not in sample.transcripts
        assert sample.transcripts == {}
        assert "dnsmos" not in sample.quality
        aug = sample.labels["augmentation"]
        assert aug["preserve_pitch"] is True
        assert aug["speed_factor"] == 1.2
        src_tx = (aug.get("source") or {}).get("transcripts") or {}
        assert "qwen" in src_tx
        assert sample.labels.get("gold_text") == f"text-{aug['source_id'][-1]}"
        assert sample.labels.get("group_id") == "g1"
        assert sample.labels.get("gold_text_inherited_from_source") is True
        assert Path(sample.source_path).is_file()
        assert sample.sha256 == file_sha256(Path(sample.source_path))
    for sample in formal.samples:
        assert sample.id.startswith("tempo_")
        assert "asr.qwen" not in sample.status
        assert "qwen" not in sample.transcripts

    # Idempotent re-run: same ids, cache hits, no duplicate files explosion.
    out2 = op.run(samples, cfg)
    assert [s.id for s in out2] == [s.id for s in out]
    wavs = list((product_root / "audio").rglob("*.wav"))
    assert len(wavs) == 3


def test_reject_already_augmented(tmp_path: Path):
    sample = Sample(
        id="tempo_abc",
        source_path="x.wav",
        labels={"augmentation": {"type": "tempo", "speed_factor": 1.2}},
    )
    assert detect_already_augmented(sample)
    with pytest.raises(AugmentPreflightError):
        validate_input_snapshot(
            [sample],
            source_batch="b1",
            input_manifest=tmp_path / "m.parquet",
        )


def test_publish_partial_does_not_write_formal(tmp_path: Path):
    root = tmp_path / "prod"
    recipe = {"recipe_digest": "abc", "speed_factor": 1.2, "source_batch": "b"}
    rows = [
        {"source_id": "a", "status": "ok", "input_duration": 1.0, "output_duration": 0.8},
        {
            "source_id": "b",
            "status": "failed",
            "error_code": "boom",
            "error_message": "x",
            "input_duration": 1.0,
        },
    ]
    with pytest.raises(AugmentBatchIncomplete):
        publish_batch_artifacts(
            product_root=root,
            recipe=recipe,
            results_rows=rows,
            success_samples=[],
            source_order=["a", "b"],
            started_at=0.0,
        )
    assert not (root / "manifest.parquet").exists()
    assert (root / "results.parquet").is_file()
    assert (root / "summary.json").is_file()


def test_cli_speed_factor_priority():
    from audio_engine.core.augmentation.cli_bind import effective_speed_factor

    assert effective_speed_factor(cli_value=1.3, step_params={"speed_factor": 1.5}) == 1.3
    assert effective_speed_factor(cli_value=None, step_params={"speed_factor": 1.5}) == 1.5
    assert effective_speed_factor(cli_value=None, step_params={}) == 1.2


def test_operator_failure_injection_no_formal(tmp_path: Path):
    batch = "aug_fail"
    good = _write_tone(tmp_path / "ok.wav", duration=1.0)
    samples = [
        Sample(
            id="ok",
            source_path=str(good["path"]),
            sha256=good["sha256"],
            sample_rate=16000,
            channels=1,
            duration=1.0,
            audio={"resampled_16k": str(good["path"])},
            labels={"batch": batch},
        ),
        Sample(
            id="bad",
            source_path=str(tmp_path / "missing.wav"),
            sha256="0" * 64,
            sample_rate=16000,
            channels=1,
            duration=1.0,
            audio={"resampled_16k": str(tmp_path / "missing.wav")},
            labels={"batch": batch},
        ),
    ]
    manifest_path = tmp_path / "segments.parquet"
    Manifest(samples).save(manifest_path)
    product_root = tmp_path / "out"
    op = TempoAugmentOperator()
    cfg = OperatorConfig(
        params={
            "speed_factor": 1.2,
            "source_batch": batch,
            "input_manifest": str(manifest_path),
            "product_root": str(product_root),
            "recipe_digest": "failtest00000001",
            "derived_batch": f"{batch}__tempo_1p2_fail",
            "workers": 2,
            "max_in_flight": 2,
        },
        output_dir=tmp_path / "derived",
        cache_dir=tmp_path / "cache",
        run_dir=tmp_path / "run",
        mock=not HAS_FFMPEG,
    )
    with pytest.raises(AugmentBatchIncomplete):
        op.run(samples, cfg)
    assert not (product_root / "manifest.parquet").exists()
    import pandas as pd

    results = pd.read_parquet(product_root / "results.parquet")
    assert len(results) == 2
    assert set(results["status"]) == {"ok", "failed"}
