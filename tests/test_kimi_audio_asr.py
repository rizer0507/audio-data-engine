from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

import audio_engine.operators  # noqa: F401
import audio_engine.operators.asr.kimi_audio as kimi_audio_module
from audio_engine.core.manifest import Manifest
from audio_engine.core.operator import OperatorConfig
from audio_engine.core.pipeline import (
    ExecutionConfig,
    PipelineConfig,
    PipelineRunner,
    PipelineStep,
)
from audio_engine.core.registry import OperatorRegistry
from audio_engine.core.sample import Sample


def _samples(count: int) -> list[Sample]:
    return [
        Sample(
            id=f"s{index}",
            source_path=f"/audio/s{index}.wav",
            sha256=f"{index:064x}",
            audio={"resampled_16k": f"/audio/s{index}.wav"},
        )
        for index in range(count)
    ]


class FakeKimiModel:
    def __init__(self, broken: str | None = None):
        self.calls: list[list[str]] = []
        self.broken = broken

    def generate(self, messages, **kwargs):
        audio_path = next(
            item["content"]
            for item in messages
            if item.get("message_type") == "audio"
        )
        if self.broken == audio_path:
            raise ValueError(f"broken audio: {self.broken}")
        return None, f"文本{Path(audio_path).stem}"

    def transcribe_batch(self, audio_paths, **kwargs):
        self.calls.append(list(audio_paths))
        if self.broken in audio_paths:
            raise ValueError(f"broken audio: {self.broken}")
        return [{"text": f"文本{Path(path).stem}"} for path in audio_paths]


class FakeGenerateOnlyModel:
    """Mirrors official KimiAudio: generate() only, one utterance at a time."""

    def __init__(self):
        self.calls: list[str] = []

    def generate(self, messages, **kwargs):
        audio_path = next(
            item["content"]
            for item in messages
            if item.get("message_type") == "audio"
        )
        self.calls.append(audio_path)
        return None, f"文本{Path(audio_path).stem}"


def _config(tmp_path: Path, **params) -> OperatorConfig:
    return OperatorConfig(
        params={
            "input_audio_key": "resampled_16k",
            "model_path": "/models/kimi-audio",
            "model_version": "test",
            "batch_size": 2,
            **params,
        },
        cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "derived",
    )


def test_batch_transcribes_and_reuses_cache(tmp_path: Path, monkeypatch):
    model = FakeKimiModel()
    loads: list[dict] = []
    monkeypatch.setattr(
        kimi_audio_module,
        "_load_kimi_audio_model",
        lambda settings: loads.append(settings) or model,
    )
    operator = OperatorRegistry.get("asr.kimi_audio_batch")

    first = operator.process_batch(_samples(3), _config(tmp_path))
    assert len(loads) == 1
    assert model.calls == [["/audio/s0.wav", "/audio/s1.wav"], ["/audio/s2.wav"]]
    assert first[0].sample.get_transcript_text("kimi") == "文本s0"
    assert all(result.sample.is_completed("asr.kimi_audio_batch") for result in first)

    second = operator.process_batch(_samples(3), _config(tmp_path))
    assert all(result.cache_hit for result in second)
    assert len(loads) == 1


def test_generate_only_runs_one_utterance_at_a_time(tmp_path: Path, monkeypatch):
    model = FakeGenerateOnlyModel()
    monkeypatch.setattr(kimi_audio_module, "_load_kimi_audio_model", lambda settings: model)
    results = OperatorRegistry.get("asr.kimi_audio_batch").process_batch(
        _samples(3), _config(tmp_path, batch_size=1)
    )

    assert model.calls == ["/audio/s0.wav", "/audio/s1.wav", "/audio/s2.wav"]
    assert [result.sample.get_transcript_text("kimi") for result in results] == [
        "文本s0",
        "文本s1",
        "文本s2",
    ]


def test_single_operator_writes_transcripts_kimi(tmp_path: Path, monkeypatch):
    model = FakeGenerateOnlyModel()
    monkeypatch.setattr(kimi_audio_module, "_load_kimi_audio_model", lambda settings: model)
    result = OperatorRegistry.get("asr.kimi_audio").process(_samples(1)[0], _config(tmp_path))
    assert result.sample.get_transcript_text("kimi") == "文本s0"
    assert "kimi_audio" not in result.sample.transcripts


def test_batch_isolates_broken_audio(tmp_path: Path, monkeypatch):
    model = FakeKimiModel("/audio/s1.wav")
    monkeypatch.setattr(kimi_audio_module, "_load_kimi_audio_model", lambda settings: model)
    results = OperatorRegistry.get("asr.kimi_audio_batch").process_batch(
        _samples(3), _config(tmp_path, batch_size=3)
    )

    assert results[0].sample.get_transcript_text("kimi") == "文本s0"
    assert results[1].sample.status["asr.kimi_audio_batch"] == "failed"
    assert "broken audio" in results[1].sample.errors["asr.kimi_audio_batch"]
    assert results[2].sample.get_transcript_text("kimi") == "文本s2"


def test_environment_model_path_has_highest_priority(tmp_path: Path, monkeypatch):
    config_file = tmp_path / "kimi_audio.yaml"
    config_file.write_text("model_path: /from/yaml\n", encoding="utf-8")
    monkeypatch.setenv("KIMI_AUDIO_MODEL_PATH", "/from/environment")
    config = OperatorConfig(params={"config_path": str(config_file), "model_path": "/from/operator"})

    assert kimi_audio_module._resolve_settings(config)["model_path"] == "/from/environment"


def test_visible_single_gpu_rewrites_device_to_cuda0(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    settings = kimi_audio_module._resolve_settings(
        OperatorConfig(params={"model_path": "/models/kimi-audio", "device": "cuda:1"})
    )
    assert settings["device"] == "cuda:0"


def test_missing_absolute_model_path_fails_before_kimia_import(tmp_path: Path):
    missing = tmp_path / "missing-model"

    with pytest.raises(FileNotFoundError, match="Kimi-Audio 本地模型目录不存在"):
        kimi_audio_module._load_kimi_audio_model(
            {"model_path": str(missing), "device": "cuda"}
        )


def test_kimi_audio_batch_runs_through_pipeline_with_metrics(tmp_path: Path):
    input_path = tmp_path / "input.parquet"
    Manifest(_samples(3)).save(input_path)
    config = PipelineConfig(
        name="kimi_audio_batch_test",
        input_manifest=str(input_path),
        steps=[
            PipelineStep(
                name="kimi_audio_asr",
                operator="asr.kimi_audio_batch",
                params={"input_audio_key": "resampled_16k", "mock": True, "batch_size": 2},
            )
        ],
        cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "derived",
        runs_dir=tmp_path / "runs",
        execution=ExecutionConfig(
            executor="sequential",
            workers=1,
            checkpoint_every=2,
        ),
    )

    runner = PipelineRunner(config)
    result = runner.run()

    assert len(result) == 3
    assert runner.metrics.to_dict()["by_step"]["kimi_audio_asr"]["processed"] == 3
    assert all(
        sample.get_transcript_text("kimi").startswith("[mock:kimi_audio:")
        for sample in result
    )


def test_kimi_audio_pipeline_yaml_fits_two_a800s():
    yaml_path = Path(__file__).parents[1] / "pipelines/kimi_audio_asr_batch.yaml"
    cfg = PipelineConfig.from_yaml(yaml_path)
    assert cfg.sharding is not None
    assert cfg.sharding.gpus == ("0", "1")
    assert cfg.sharding.shards == 2
    assert cfg.sharding.effective_parallel == 2
    assert cfg.sharding.gpu_slots == 2
    cfg.sharding.validate_parallel()


def _load_probe_module():
    script = Path(__file__).parents[1] / "scripts/probe_kimi_audio.py"
    spec = importlib.util.spec_from_file_location("probe_kimi_audio", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_probe_discovers_single_file_and_directory(tmp_path: Path):
    probe = _load_probe_module()
    first = tmp_path / "b.WAV"
    second = tmp_path / "a.wav"
    ignored = tmp_path / "notes.txt"
    nested = tmp_path / "nested" / "c.wav"
    nested.parent.mkdir()
    for path in (first, second, ignored, nested):
        path.write_bytes(b"test")

    assert probe.discover_wavs(first) == [first.resolve()]
    assert probe.discover_wavs(tmp_path) == [second.resolve(), first.resolve()]
    assert probe.discover_wavs(tmp_path, recursive=True) == [
        second.resolve(),
        first.resolve(),
        nested.resolve(),
    ]


def test_probe_transcribes_file_and_folder(tmp_path: Path, monkeypatch, capsys):
    probe = _load_probe_module()
    first = tmp_path / "a.wav"
    second = tmp_path / "b.wav"
    first.write_bytes(b"test")
    second.write_bytes(b"test")
    monkeypatch.setattr(probe.kimi_audio, "_load_kimi_audio_model", lambda settings: object())
    monkeypatch.setattr(
        probe.kimi_audio,
        "_transcribe_one",
        lambda model, path, settings: {"text": f"文本{Path(path).stem}"},
    )

    file_code = probe.main([str(first), "--model-path", "/models/kimi-audio"])
    file_out = capsys.readouterr()
    assert file_code == 0
    assert '"text": "文本a"' in file_out.out

    dir_code = probe.main([str(tmp_path), "--model-path", "/models/kimi-audio", "--limit", "1"])
    dir_out = capsys.readouterr()
    assert dir_code == 0
    assert dir_out.out.count('"ok": true') == 1
