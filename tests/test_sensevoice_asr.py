from __future__ import annotations

from pathlib import Path

import pytest

import audio_engine.operators  # noqa: F401
import audio_engine.operators.asr.sensevoice as sensevoice_module
from audio_engine.core.operator import OperatorConfig
from audio_engine.core.registry import OperatorRegistry
from audio_engine.core.sample import Sample


def _samples(count: int) -> list[Sample]:
    return [Sample(id=f"s{i}", source_path=f"/audio/s{i}.wav", sha256=f"{i:064x}", audio={"resampled_16k": f"/audio/s{i}.wav"}) for i in range(count)]


class FakeModel:
    def __init__(self, broken: str | None = None):
        self.calls = []
        self.broken = broken

    def generate(self, **kwargs):
        paths = list(kwargs["input"])
        self.calls.append(paths)
        if self.broken in paths:
            raise ValueError(f"broken audio: {self.broken}")
        return [{"text": f"<|zh|><|NEUTRAL|><|Speech|>文本{Path(path).stem}"} for path in paths]


def _config(tmp_path: Path, **params):
    return OperatorConfig(params={"input_audio_key": "resampled_16k", "model_path": "/models/sensevoice", "model_version": "test", "batch_size": 2, **params}, cache_dir=tmp_path / "cache")


def test_parse_sensevoice_tags_preserves_unknown_tags():
    parsed = sensevoice_module.parse_sensevoice_text("<|zh|><|HAPPY|><|Laughter|><|future|>你好")
    assert parsed["text"] == "<|future|>你好"
    assert parsed["extra"] == {"raw_text": "<|zh|><|HAPPY|><|Laughter|><|future|>你好", "language": "zh", "emotion": "HAPPY", "events": ["Laughter"], "unknown_tags": ["future"]}


def test_batch_transcribes_structured_results_and_reuses_cache(tmp_path, monkeypatch):
    model = FakeModel()
    loads = []
    monkeypatch.setattr(sensevoice_module, "_load_sensevoice_model", lambda settings: loads.append(settings) or model)
    operator = OperatorRegistry.get("asr.sensevoice_batch")
    first = operator.process_batch(_samples(3), _config(tmp_path))
    assert len(loads) == 1
    assert model.calls == [["/audio/s0.wav", "/audio/s1.wav"], ["/audio/s2.wav"]]
    transcript = first[0].sample.transcripts["sensevoice"]
    assert transcript["text"] == "文本s0"
    assert transcript["extra"] == {"raw_text": "<|zh|><|NEUTRAL|><|Speech|>文本s0", "language": "zh", "emotion": "NEUTRAL", "events": ["Speech"]}
    second = operator.process_batch(_samples(3), _config(tmp_path))
    assert all(result.cache_hit for result in second)
    assert len(loads) == 1


def test_batch_supports_transcript_key_alias(tmp_path, monkeypatch):
    model = FakeModel()
    monkeypatch.setattr(sensevoice_module, "_load_sensevoice_model", lambda settings: model)
    operator = OperatorRegistry.get("asr.sensevoice_batch")
    results = operator.process_batch(
        _samples(1), _config(tmp_path, transcript_key="sensevoice1", mock=True)
    )
    assert "sensevoice1" in results[0].sample.transcripts
    assert "sensevoice" not in results[0].sample.transcripts
    skipped = operator.process_batch(
        [results[0].sample], _config(tmp_path, transcript_key="sensevoice1", mock=True)
    )
    assert skipped[0].skipped is True


def test_batch_isolates_broken_audio(tmp_path, monkeypatch):
    model = FakeModel("/audio/s1.wav")
    monkeypatch.setattr(sensevoice_module, "_load_sensevoice_model", lambda settings: model)
    results = OperatorRegistry.get("asr.sensevoice_batch").process_batch(_samples(3), _config(tmp_path, batch_size=3))
    assert results[0].sample.get_transcript_text("sensevoice") == "文本s0"
    assert results[1].sample.status["asr.sensevoice_batch"] == "failed"
    assert "broken audio" in results[1].sample.errors["asr.sensevoice_batch"]
    assert results[2].sample.get_transcript_text("sensevoice") == "文本s2"


def test_environment_model_path_has_highest_priority(tmp_path, monkeypatch):
    config_file = tmp_path / "sensevoice.yaml"
    config_file.write_text("model_path: /from/yaml\n", encoding="utf-8")
    monkeypatch.setenv("SENSEVOICE_MODEL_PATH", "/from/environment")
    config = OperatorConfig(
        params={"config_path": str(config_file), "model_path": "/from/operator"}
    )

    assert sensevoice_module._resolve_settings(config)["model_path"] == "/from/environment"


def test_missing_absolute_model_path_fails_before_funasr_import(tmp_path):
    missing = tmp_path / "missing-model"

    with pytest.raises(FileNotFoundError, match="SenseVoice 本地模型目录不存在"):
        sensevoice_module._load_sensevoice_model(
            {"model_path": str(missing), "device": "cuda:0"}
        )
