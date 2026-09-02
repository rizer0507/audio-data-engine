from __future__ import annotations

import json
from pathlib import Path

import pytest

from audio_engine.operators.asr import vllm


@pytest.mark.parametrize(
    ("base", "expected"),
    [
        ("http://localhost:8000", "http://localhost:8000/v1/audio/transcriptions"),
        ("http://localhost:8000/", "http://localhost:8000/v1/audio/transcriptions"),
        ("http://localhost:8000/v1", "http://localhost:8000/v1/audio/transcriptions"),
        (
            "http://localhost:8000/v1/audio/transcriptions",
            "http://localhost:8000/v1/audio/transcriptions",
        ),
    ],
)
def test_transcription_url_accepts_root_v1_and_full_url(base: str, expected: str):
    assert vllm.transcription_url(base) == expected


def test_vllm_transcription_sends_only_supported_fields(tmp_path: Path, monkeypatch):
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"RIFF-test")
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return json.dumps({"text": "识别结果", "language": "zh"}).encode()

    def fake_urlopen(request, timeout):
        captured.update(url=request.full_url, body=request.data, timeout=timeout)
        return Response()

    monkeypatch.setattr(vllm.urllib.request, "urlopen", fake_urlopen)
    result = vllm.call_vllm_transcription(
        str(audio),
        {
            "api_base": "http://localhost:8000/v1",
            "api_key": "secret",
            "model": "served-asr",
            "language": "zh",
            "prompt": "转写",
            "temperature": 0,
            "max_completion_tokens": 256,
        },
    )

    assert captured["url"] == "http://localhost:8000/v1/audio/transcriptions"
    assert b'name="model"' in captured["body"]
    assert b'name="prompt"' in captured["body"]
    assert b"max_completion_tokens" not in captured["body"]
    assert result["text"] == "识别结果"
