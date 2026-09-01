from __future__ import annotations

from pathlib import Path

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
    def __init__(self, broken_path: str | None = None):
        self.calls: list[list[str]] = []
        self.broken_path = broken_path

    def transcribe_batch(self, paths: list[str], **kwargs):
        self.calls.append(list(paths))
        if self.broken_path in paths:
            raise ValueError(f"broken audio: {self.broken_path}")
        return [{"text": f"text:{Path(path).stem}", "language": "zh"} for path in paths]


def _config(tmp_path: Path, **params) -> OperatorConfig:
    return OperatorConfig(
        params={
            "input_audio_key": "resampled_16k",
            "model_path": "/models/kimi",
            "model_version": "test",
            "batch_size": 2,
            **params,
        },
        cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "derived",
    )


def test_kimi_batch_transcribes_and_reuses_cache(tmp_path: Path, monkeypatch):
    model = FakeKimiModel()
    monkeypatch.setattr(kimi_module, "_load_kimi_model", lambda settings: model)
    operator = OperatorRegistry.get("asr.kimi_batch")

    first = operator.process_batch(_samples(3), _config(tmp_path))
    assert model.calls == [["/audio/s0.wav", "/audio/s1.wav"], ["/audio/s2.wav"]]
    assert [result.sample.get_transcript_text("kimi") for result in first] == [
        "text:s0",
        "text:s1",
        "text:s2",
    ]
    assert all(result.sample.is_completed("asr.kimi_batch") for result in first)

    second = operator.process_batch(_samples(3), _config(tmp_path))
    assert all(result.cache_hit for result in second)
    assert len(model.calls) == 2


def test_kimi_batch_isolates_corrupt_audio(tmp_path: Path, monkeypatch):
    model = FakeKimiModel(broken_path="/audio/s1.wav")
    monkeypatch.setattr(kimi_module, "_load_kimi_model", lambda settings: model)

    results = OperatorRegistry.get("asr.kimi_batch").process_batch(
        _samples(3), _config(tmp_path, batch_size=3)
    )

    assert results[0].sample.get_transcript_text("kimi") == "text:s0"
    assert results[1].sample.status["asr.kimi_batch"] == "failed"
    assert "broken audio" in results[1].sample.errors["asr.kimi_batch"]
    assert results[2].sample.get_transcript_text("kimi") == "text:s2"


def test_kimi_generate_only_official_api():
    class GenerateOnlyModel:
        def build_prompt(self, path, prompt):
            return f"{path}|{prompt}"

        def generate(self, prompt, **kwargs):
            return (f"generated:{prompt.split('|')[0]}", None)

    results = kimi_module._transcribe_many(
        GenerateOnlyModel(), ["a.wav", "b.wav"], {"prompt": "transcribe"}
    )
    assert [result["text"] for result in results] == ["generated:a.wav", "generated:b.wav"]


def test_kimi_batch_runs_through_pipeline(tmp_path: Path):
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
        execution=ExecutionConfig(executor="sequential", workers=1, checkpoint_every=2),
    )

    runner = PipelineRunner(config)
    result = runner.run()

    assert runner.metrics.to_dict()["by_step"]["kimi_asr"]["processed"] == 3
    assert all(sample.get_transcript_text("kimi").startswith("[mock:kimi:") for sample in result)
