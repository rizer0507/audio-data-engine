from __future__ import annotations

from pathlib import Path

import pytest

import audio_engine.operators  # noqa: F401
import audio_engine.operators.asr.glm as glm_module
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
            "model_version": "test",
            "api_base": "http://127.0.0.1:5570",
            "batch_size": 2,
            **params,
        },
        cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "derived",
    )


def test_glm_batch_refuses_to_load_a_local_model(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("GLM_ASR_API_BASE", raising=False)
    monkeypatch.delenv("GLM_ASR_API_BASES", raising=False)
    operator = OperatorRegistry.get("asr.glm_batch")
    with pytest.raises(ValueError, match="only supports vLLM"):
        operator.process_batch(_samples(1), _config(tmp_path, api_base=None))


def test_glm_batch_reads_vllm_environment(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("GLM_ASR_API_BASES", raising=False)
    monkeypatch.setenv("GLM_ASR_API_BASE", "http://127.0.0.1:5570")
    settings = glm_module._resolve_batch_settings(_config(tmp_path, api_base=None))
    assert settings["api_base"] == "http://127.0.0.1:5570"


def test_glm_batch_reads_step_specific_vllm_environment(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("GLM_ASR_API_BASE", raising=False)
    monkeypatch.delenv("GLM_ASR_API_BASES", raising=False)
    monkeypatch.setenv("CANDIDATE_ASR_API_BASE", "http://127.0.0.1:5572")
    settings = glm_module._resolve_batch_settings(
        _config(
            tmp_path,
            api_base=None,
            api_base_env="CANDIDATE_ASR_API_BASE",
        )
    )
    assert settings["api_base"] == "http://127.0.0.1:5572"


def test_glm_batch_reads_served_model_environment(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("GLM_ASR_API_BASE", "http://127.0.0.1:5570")
    monkeypatch.setenv("GLM_ASR_MODEL", "glm-sft-ep100")
    settings = glm_module._resolve_batch_settings(_config(tmp_path, api_base=None))
    assert settings["model"] == "glm-sft-ep100"

    monkeypatch.setenv("CANDIDATE_ASR_MODEL", "candidate-sft")
    settings = glm_module._resolve_batch_settings(
        _config(tmp_path, api_base=None, model_env="CANDIDATE_ASR_MODEL")
    )
    assert settings["model"] == "candidate-sft"

    settings = glm_module._resolve_batch_settings(
        _config(tmp_path, api_base=None, model="explicit-model")
    )
    assert settings["model"] == "explicit-model"


def test_glm_batch_transcribes_and_reuses_cache(tmp_path: Path, monkeypatch):
    calls: list[str] = []

    def fake_transcribe(path: str, settings: dict) -> dict:
        calls.append(path)
        return {"text": f"text:{Path(path).stem}", "language": "Chinese"}

    monkeypatch.setattr(glm_module, "call_vllm_transcription", fake_transcribe)
    operator = OperatorRegistry.get("asr.glm_batch")
    config = _config(tmp_path)

    first = operator.process_batch(_samples(3), config)
    assert sorted(calls) == ["/audio/s0.wav", "/audio/s1.wav", "/audio/s2.wav"]
    assert [result.sample.get_transcript_text("glm") for result in first] == [
        "text:s0",
        "text:s1",
        "text:s2",
    ]
    assert all(result.sample.is_completed("asr.glm_batch") for result in first)
    assert all(result.sample.lineage[-1].operator == "asr.glm_batch" for result in first)

    second = operator.process_batch(_samples(3), config)
    assert all(result.cache_hit for result in second)
    assert len(calls) == 3


def test_glm_batch_cache_includes_model_and_api_base(tmp_path: Path, monkeypatch):
    calls: list[str] = []

    def fake_transcribe(path: str, settings: dict) -> dict:
        calls.append(f"{settings['model']}:{settings['api_base']}:{path}")
        return {"text": f"text:{Path(path).stem}", "language": "zh"}

    monkeypatch.setattr(glm_module, "call_vllm_transcription", fake_transcribe)
    operator = OperatorRegistry.get("asr.glm_batch")
    samples = _samples(1)

    operator.process_batch(samples, _config(tmp_path, model="glm-asr"))
    operator.process_batch(samples, _config(tmp_path, model="glm-sft"))
    operator.process_batch(
        samples, _config(tmp_path, model="glm-asr", api_base="http://127.0.0.1:5571")
    )
    assert len(calls) == 3


def test_glm_batch_isolates_corrupt_audio(tmp_path: Path, monkeypatch):
    def fake_transcribe(path: str, settings: dict) -> dict:
        if path == "/audio/s1.wav":
            raise ValueError(f"broken audio: {path}")
        return {"text": f"text:{Path(path).stem}", "language": "Chinese"}

    monkeypatch.setattr(glm_module, "call_vllm_transcription", fake_transcribe)
    operator = OperatorRegistry.get("asr.glm_batch")

    results = operator.process_batch(_samples(3), _config(tmp_path, batch_size=3))

    assert results[0].sample.get_transcript_text("glm") == "text:s0"
    assert results[1].sample.status["asr.glm_batch"] == "failed"
    assert "broken audio" in results[1].sample.errors["asr.glm_batch"]
    assert results[2].sample.get_transcript_text("glm") == "text:s2"


def test_glm_batch_runs_through_pipeline_with_metrics(tmp_path: Path):
    input_path = tmp_path / "input.parquet"
    Manifest(_samples(3)).save(input_path)
    config = PipelineConfig(
        name="glm_batch_test",
        input_manifest=str(input_path),
        steps=[
            PipelineStep(
                name="glm_asr",
                operator="asr.glm_batch",
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
    assert runner.metrics.to_dict()["by_step"]["glm_asr"]["processed"] == 3
    assert all(sample.get_transcript_text("glm").startswith("[mock:glm:") for sample in result)


def test_glm_batch_supports_evaluation_transcript_key(tmp_path: Path):
    operator = OperatorRegistry.get("asr.glm_batch")
    results = operator.process_batch(
        _samples(1),
        _config(tmp_path, mock=True, transcript_key="new_model"),
    )
    assert results[0].sample.get_transcript_text("new_model") == "[mock:glm:s0]"
    assert "glm" not in results[0].sample.transcripts

    candidate = operator.process_batch(
        [results[0].sample],
        _config(tmp_path, mock=True, transcript_key="old_model"),
    )
    assert candidate[0].skipped is False
    assert set(candidate[0].sample.transcripts) == {"new_model", "old_model"}


def test_glm_vllm_parses_multiple_api_bases(monkeypatch):
    monkeypatch.delenv("GLM_ASR_API_BASE", raising=False)
    monkeypatch.setenv(
        "GLM_ASR_API_BASES",
        "http://127.0.0.1:5570, http://127.0.0.1:5571",
    )
    settings = glm_module._resolve_settings(OperatorConfig(params={}))
    assert settings["api_bases"] == [
        "http://127.0.0.1:5570",
        "http://127.0.0.1:5571",
    ]
    first = glm_module.select_glm_api_base(settings, "s0")
    second = glm_module.select_glm_api_base(settings, "s1")
    assert first in settings["api_bases"]
    assert second in settings["api_bases"]
    assert glm_module.select_glm_api_base(settings, "s0") == first


def test_glm_batch_routes_samples_across_replicas(tmp_path: Path, monkeypatch):
    seen: list[str] = []

    def fake_transcribe(audio_path: str, settings: dict) -> dict:
        seen.append(settings["api_base"])
        return {"text": f"text:{Path(audio_path).stem}", "language": "zh"}

    monkeypatch.setattr(glm_module, "call_vllm_transcription", fake_transcribe)
    OperatorRegistry.get("asr.glm_batch").process_batch(
        _samples(4),
        _config(
            tmp_path,
            api_bases=["http://127.0.0.1:5570", "http://127.0.0.1:5571"],
            concurrency=4,
            batch_size=4,
        ),
    )
    assert set(seen) <= {"http://127.0.0.1:5570", "http://127.0.0.1:5571"}
    assert len(seen) == 4


def _load_probe_module():
    import importlib.util

    script = Path(__file__).parents[1] / "scripts/probe_glm_vllm.py"
    spec = importlib.util.spec_from_file_location("probe_glm_vllm", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_glm_probe_discovers_wavs_and_has_no_pad_flag(tmp_path: Path):
    probe = _load_probe_module()
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF")
    assert probe.discover_wavs(wav) == [wav.resolve()]
    assert probe.discover_wavs(tmp_path) == [wav.resolve()]
    assert "--pad" not in probe.build_parser().format_help()


def test_glm_probe_exits_1_without_api_base(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("GLM_ASR_API_BASE", raising=False)
    monkeypatch.delenv("GLM_ASR_API_BASES", raising=False)
    wav = tmp_path / "short.wav"
    wav.write_bytes(b"RIFF")
    probe = _load_probe_module()
    assert probe.main([str(wav)]) == 1
