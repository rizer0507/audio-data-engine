from __future__ import annotations

from pathlib import Path

import pytest

import audio_engine.operators  # noqa: F401
import audio_engine.operators.asr.kimi as kimi_module
from audio_engine.core.manifest import Manifest
from audio_engine.core.operator import OperatorConfig
from audio_engine.core.pipeline import ExecutionConfig, PipelineConfig, PipelineRunner, PipelineStep
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


def test_canonical_batch_transcribes_and_reuses_cache(tmp_path: Path, monkeypatch):
    model = FakeKimiModel()
    loads: list[dict] = []
    monkeypatch.setattr(
        kimi_module,
        "_load_kimi_audio_model",
        lambda settings: loads.append(settings) or model,
    )
    operator = OperatorRegistry.get("asr.kimi_batch")

    first = operator.process_batch(_samples(3), _config(tmp_path))
    assert len(loads) == 1
    assert model.calls == [["/audio/s0.wav", "/audio/s1.wav"], ["/audio/s2.wav"]]
    assert first[0].sample.get_transcript_text("kimi") == "文本s0"
    assert all(result.sample.is_completed("asr.kimi_batch") for result in first)

    second = operator.process_batch(_samples(3), _config(tmp_path))
    assert all(result.cache_hit for result in second)
    assert len(loads) == 1


def test_canonical_batch_isolates_broken_audio(tmp_path: Path, monkeypatch):
    model = FakeKimiModel("/audio/s1.wav")
    monkeypatch.setattr(kimi_module, "_load_kimi_audio_model", lambda settings: model)
    results = OperatorRegistry.get("asr.kimi_batch").process_batch(
        _samples(3), _config(tmp_path, batch_size=3)
    )

    assert results[0].sample.get_transcript_text("kimi") == "文本s0"
    assert results[1].sample.status["asr.kimi_batch"] == "failed"
    assert "broken audio" in results[1].sample.errors["asr.kimi_batch"]
    assert results[2].sample.get_transcript_text("kimi") == "文本s2"


def test_canonical_environment_model_path_has_highest_priority(tmp_path: Path, monkeypatch):
    config_file = tmp_path / "kimi_audio.yaml"
    config_file.write_text("model_path: /from/yaml\n", encoding="utf-8")
    monkeypatch.setenv("KIMI_AUDIO_MODEL_PATH", "/from/environment")
    config = OperatorConfig(params={"config_path": str(config_file), "model_path": "/from/operator"})

    assert kimi_module._resolve_settings(config)["model_path"] == "/from/environment"


def test_canonical_missing_absolute_model_path_fails_before_kimia_import(tmp_path: Path):
    missing = tmp_path / "missing-model"

    with pytest.raises(FileNotFoundError, match="Kimi-Audio 本地模型目录不存在"):
        kimi_module._load_kimi_audio_model(
            {"model_path": str(missing), "device": "cuda"}
        )


def test_canonical_kimi_audio_batch_runs_through_pipeline_with_metrics(tmp_path: Path):
    input_path = tmp_path / "input.parquet"
    Manifest(_samples(3)).save(input_path)
    config = PipelineConfig(
        name="kimi_batch_test",
        input_manifest=str(input_path),
        steps=[
            PipelineStep(
                name="kimi_asr",
                operator="asr.kimi_batch",
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
    assert runner.metrics.to_dict()["by_step"]["kimi_asr"]["processed"] == 3
    assert all(
        sample.get_transcript_text("kimi").startswith("[mock:kimi:")
        for sample in result
    )
