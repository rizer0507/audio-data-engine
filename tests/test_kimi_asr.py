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


def _config(tmp_path: Path, **params) -> OperatorConfig:
    return OperatorConfig(
        params={
            "input_audio_key": "resampled_16k",
            "api_base": "http://127.0.0.1:5554",
            "model_version": "test",
            "concurrency": 2,
            **params,
        },
        cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "derived",
    )


def test_kimi_batch_transcribes_and_reuses_cache(tmp_path: Path, monkeypatch):
    calls: list[str] = []

    def fake_transcribe(audio_path: str, settings: dict) -> dict:
        calls.append(audio_path)
        return {"text": f"text:{Path(audio_path).stem}", "language": "zh"}

    monkeypatch.setattr(kimi_module, "_call_vllm_transcription", fake_transcribe)
    operator = OperatorRegistry.get("asr.kimi_batch")
    config = _config(tmp_path)

    first = operator.process_batch(_samples(3), config)
    assert sorted(calls) == ["/audio/s0.wav", "/audio/s1.wav", "/audio/s2.wav"]
    assert [result.sample.get_transcript_text("kimi") for result in first] == [
        "text:s0",
        "text:s1",
        "text:s2",
    ]
    assert all(result.sample.is_completed("asr.kimi_batch") for result in first)

    second = operator.process_batch(_samples(3), config)
    assert all(result.cache_hit for result in second)
    assert len(calls) == 3


def test_kimi_batch_isolates_corrupt_audio(tmp_path: Path, monkeypatch):
    def fake_transcribe(audio_path: str, settings: dict) -> dict:
        if audio_path.endswith("s1.wav"):
            raise ValueError("broken audio")
        return {"text": f"text:{Path(audio_path).stem}", "language": "zh"}

    monkeypatch.setattr(kimi_module, "_call_vllm_transcription", fake_transcribe)
    operator = OperatorRegistry.get("asr.kimi_batch")

    results = operator.process_batch(_samples(3), _config(tmp_path, concurrency=3))

    assert results[0].sample.get_transcript_text("kimi") == "text:s0"
    assert results[1].sample.status["asr.kimi_batch"] == "failed"
    assert "broken audio" in results[1].sample.errors["asr.kimi_batch"]
    assert results[2].sample.get_transcript_text("kimi") == "text:s2"


def test_kimi_batch_runs_through_pipeline_with_metrics(tmp_path: Path):
    input_path = tmp_path / "input.parquet"
    Manifest(_samples(3)).save(input_path)
    config = PipelineConfig(
        name="kimi_batch_test",
        input_manifest=str(input_path),
        steps=[
            PipelineStep(
                name="kimi_asr",
                operator="asr.kimi_batch",
                params={"input_audio_key": "resampled_16k", "mock": True, "concurrency": 2},
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
    assert all(sample.get_transcript_text("kimi").startswith("[mock:kimi:") for sample in result)
