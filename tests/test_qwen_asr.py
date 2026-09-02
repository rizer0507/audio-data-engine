from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import audio_engine.operators  # noqa: F401
import audio_engine.operators.asr.qwen as qwen_module
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


class FakeQwenModel:
    def __init__(self, broken_path: str | None = None):
        self.calls: list[list[str]] = []
        self.broken_path = broken_path

    def transcribe(self, **kwargs):
        paths = list(kwargs["audio"])
        self.calls.append(paths)
        if self.broken_path in paths:
            raise ValueError(f"broken audio: {self.broken_path}")
        return [
            SimpleNamespace(text=f"text:{Path(path).stem}", language="Chinese") for path in paths
        ]


def _config(tmp_path: Path, **params) -> OperatorConfig:
    return OperatorConfig(
        params={
            "input_audio_key": "resampled_16k",
            "model_path": "/models/qwen",
            "model_version": "test",
            "batch_size": 2,
            **params,
        },
        cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "derived",
    )


def test_qwen_batch_transcribes_and_reuses_cache(tmp_path: Path, monkeypatch):
    model = FakeQwenModel()
    monkeypatch.setattr(qwen_module, "_load_qwen_model", lambda settings: model)
    operator = OperatorRegistry.get("asr.qwen_batch")
    config = _config(tmp_path)

    first = operator.process_batch(_samples(3), config)
    assert model.calls == [
        ["/audio/s0.wav", "/audio/s1.wav"],
        ["/audio/s2.wav"],
    ]
    assert [result.sample.get_transcript_text("qwen") for result in first] == [
        "text:s0",
        "text:s1",
        "text:s2",
    ]
    assert all(result.sample.is_completed("asr.qwen_batch") for result in first)
    assert all(result.sample.lineage[-1].operator == "asr.qwen_batch" for result in first)

    second = operator.process_batch(_samples(3), config)
    assert all(result.cache_hit for result in second)
    assert len(model.calls) == 2


def test_qwen_batch_isolates_corrupt_audio(tmp_path: Path, monkeypatch):
    model = FakeQwenModel(broken_path="/audio/s1.wav")
    monkeypatch.setattr(qwen_module, "_load_qwen_model", lambda settings: model)
    operator = OperatorRegistry.get("asr.qwen_batch")

    results = operator.process_batch(_samples(3), _config(tmp_path, batch_size=3))

    assert results[0].sample.get_transcript_text("qwen") == "text:s0"
    assert results[1].sample.status["asr.qwen_batch"] == "failed"
    assert "broken audio" in results[1].sample.errors["asr.qwen_batch"]
    assert results[2].sample.get_transcript_text("qwen") == "text:s2"


def test_qwen_batch_runs_through_pipeline_with_metrics(tmp_path: Path):
    input_path = tmp_path / "input.parquet"
    Manifest(_samples(3)).save(input_path)
    config = PipelineConfig(
        name="qwen_batch_test",
        input_manifest=str(input_path),
        steps=[
            PipelineStep(
                name="qwen_asr",
                operator="asr.qwen_batch",
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
    assert runner.metrics.to_dict()["by_step"]["qwen_asr"]["processed"] == 3
    assert all(sample.get_transcript_text("qwen").startswith("[mock:qwen:") for sample in result)


def test_run_shards_refuses_more_processes_than_gpu_slots(tmp_path: Path):
    from typer.testing import CliRunner

    from audio_engine.cli.main import app

    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text("name: test\ninput:\n  manifest: unused.parquet\npipeline: []\n")
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    (shard_dir / "shard-000.parquet").touch()

    result = CliRunner().invoke(
        app,
        [
            "pipeline",
            "run-shards",
            str(config_path),
            "--shard-dir",
            str(shard_dir),
            "--parallel-shards",
            "2",
            "--gpus",
            "0",
        ],
    )

    assert result.exit_code != 0
    assert "cannot exceed GPUs" in result.output


def test_run_shards_gpu_slot_math_rejects_overflow(tmp_path: Path):
    from typer.testing import CliRunner

    from audio_engine.cli.main import app

    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text("name: test\ninput:\n  manifest: unused.parquet\npipeline: []\n")
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    (shard_dir / "shard-000.parquet").touch()

    result = CliRunner().invoke(
        app,
        [
            "pipeline",
            "run-shards",
            str(config_path),
            "--shard-dir",
            str(shard_dir),
            "--parallel-shards",
            "9",
            "--gpus",
            "0,1,2,3",
            "--instances-per-gpu",
            "2",
        ],
    )

    assert result.exit_code != 0
    assert "cannot exceed GPUs" in result.output
    assert "instances-per-gpu" in result.output


def test_qwen_batch_supports_evaluation_transcript_key(tmp_path: Path):
    operator = OperatorRegistry.get("asr.qwen_batch")
    results = operator.process_batch(
        _samples(1),
        _config(tmp_path, mock=True, transcript_key="new_model"),
    )
    assert results[0].sample.get_transcript_text("new_model") == "[mock:qwen:s0]"
    assert "qwen" not in results[0].sample.transcripts

    candidate = operator.process_batch(
        [results[0].sample],
        _config(tmp_path, mock=True, transcript_key="old_model"),
    )
    assert candidate[0].skipped is False
    assert set(candidate[0].sample.transcripts) == {"new_model", "old_model"}
