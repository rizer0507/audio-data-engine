from __future__ import annotations

from pathlib import Path

import audio_engine.operators  # noqa: F401
import audio_engine.operators.asr.doubao as doubao_module
from audio_engine.core.operator import OperatorConfig
from audio_engine.core.registry import OperatorRegistry
from audio_engine.core.sample import Sample


def _sample() -> Sample:
    return Sample(
        id="single",
        source_path="/audio/sample.wav",
        sha256="0" * 64,
        audio={"raw": "/audio/sample.wav"},
    )


def _config(tmp_path: Path, **params) -> OperatorConfig:
    return OperatorConfig(
        params={
            "input_audio_key": "raw",
            "transcript_key": "doubao",
            "api_key": "test-api-key",
            "resource_id": "volc.seedasr.auc",
            **params,
        },
        cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "derived",
    )


def test_doubao_transcribes_single_audio(tmp_path: Path, monkeypatch):
    def fake_transcribe(audio_path: str, settings: dict) -> dict:
        assert audio_path == "/audio/sample.wav"
        assert settings["api_key"] == "test-api-key"
        return {
            "text": "你好，世界",
            "language": "zh-CN",
            "extra": {"audio_info": {"duration": 1200}},
        }

    monkeypatch.setattr(doubao_module, "call_doubao_transcription", fake_transcribe)
    operator = OperatorRegistry.get("asr.doubao")
    result = operator.process(_sample(), _config(tmp_path))

    assert result.sample.get_transcript_text("doubao") == "你好，世界"
    assert result.sample.transcripts["doubao"]["model"] == "volc.seedasr.auc"
    assert result.sample.is_completed("asr.doubao")


def test_doubao_mock_mode(tmp_path: Path, monkeypatch):
    calls: list[str] = []

    def fake_transcribe(audio_path: str, settings: dict) -> dict:
        calls.append(audio_path)
        return {"text": "unexpected", "extra": {}}

    monkeypatch.setattr(doubao_module, "call_doubao_transcription", fake_transcribe)
    operator = OperatorRegistry.get("asr.doubao")
    result = operator.process(_sample(), _config(tmp_path, mock=True))

    assert result.sample.get_transcript_text("doubao") == "[mock:doubao:single]"
    assert calls == []


def test_doubao_builds_base64_payload(tmp_path: Path, monkeypatch):
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"RIFFdemo")
    settings = {
        "api_key": "test-api-key",
        "resource_id": "volc.seedasr.auc",
        "user_id": "test-user",
        "poll_interval": 0.01,
        "max_polls": 1,
        "timeout": 5,
        "enable_itn": True,
        "enable_punc": True,
    }

    captured: dict[str, object] = {}

    def fake_submit(headers: dict[str, str], payload: dict, timeout: float) -> dict[str, str]:
        captured["headers"] = headers
        captured["payload"] = payload
        return {"x-api-status-code": "20000000"}

    def fake_poll(headers: dict[str, str], poll_settings: dict) -> dict:
        return {
            "text": "测试文本",
            "extra": {"audio_info": {"duration": 1000}},
        }

    monkeypatch.setattr(doubao_module, "_submit_task", fake_submit)
    monkeypatch.setattr(doubao_module, "_poll_result", fake_poll)

    result = doubao_module.call_doubao_transcription(str(audio_path), settings)

    assert result["text"] == "测试文本"
    payload = captured["payload"]
    assert payload["audio"]["format"] == "wav"
    assert payload["audio"]["data"]
    assert payload["request"]["model_name"] == "bigmodel"
    headers = captured["headers"]
    assert headers["X-Api-Key"] == "test-api-key"
    assert headers["X-Api-Resource-Id"] == "volc.seedasr.auc"


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


def _batch_config(tmp_path: Path, **params) -> OperatorConfig:
    return OperatorConfig(
        params={
            "input_audio_key": "resampled_16k",
            "transcript_key": "doubao",
            "app_key": "5769728651",
            "access_key": "test-token",
            "resource_id": "volc.seedasr.auc",
            "concurrency": 2,
            **params,
        },
        cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "derived",
    )


def test_doubao_batch_transcribes_and_reuses_cache(tmp_path: Path, monkeypatch):
    calls: list[str] = []

    def fake_transcribe(audio_path: str, settings: dict) -> dict:
        calls.append(audio_path)
        return {
            "text": f"text:{Path(audio_path).stem}",
            "language": "zh-CN",
            "extra": {},
        }

    monkeypatch.setattr(doubao_module, "call_doubao_transcription", fake_transcribe)
    operator = OperatorRegistry.get("asr.doubao_batch")
    config = _batch_config(tmp_path)

    first = operator.process_batch(_samples(3), config)
    assert sorted(calls) == ["/audio/s0.wav", "/audio/s1.wav", "/audio/s2.wav"]
    assert [result.sample.get_transcript_text("doubao") for result in first] == [
        "text:s0",
        "text:s1",
        "text:s2",
    ]
    assert all(result.sample.is_completed("asr.doubao_batch") for result in first)

    second = operator.process_batch(_samples(3), config)
    assert all(result.cache_hit for result in second)
    assert len(calls) == 3


def test_doubao_batch_isolates_corrupt_audio(tmp_path: Path, monkeypatch):
    def fake_transcribe(audio_path: str, settings: dict) -> dict:
        if audio_path.endswith("s1.wav"):
            raise ValueError("broken audio")
        return {"text": f"text:{Path(audio_path).stem}", "language": "zh-CN", "extra": {}}

    monkeypatch.setattr(doubao_module, "call_doubao_transcription", fake_transcribe)
    operator = OperatorRegistry.get("asr.doubao_batch")

    results = operator.process_batch(_samples(3), _batch_config(tmp_path, concurrency=3))

    assert results[0].sample.get_transcript_text("doubao") == "text:s0"
    assert results[1].sample.status["asr.doubao_batch"] == "failed"
    assert "broken audio" in results[1].sample.errors["asr.doubao_batch"]
    assert results[2].sample.get_transcript_text("doubao") == "text:s2"
