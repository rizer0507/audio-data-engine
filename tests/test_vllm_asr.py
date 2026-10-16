from __future__ import annotations

import base64
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


@pytest.mark.parametrize(
    ("base", "expected"),
    [
        ("http://localhost:8000", "http://localhost:8000/v1/chat/completions"),
        ("http://localhost:8000/", "http://localhost:8000/v1/chat/completions"),
        ("http://localhost:8000/v1", "http://localhost:8000/v1/chat/completions"),
        (
            "http://localhost:8000/v1/chat/completions",
            "http://localhost:8000/v1/chat/completions",
        ),
        (
            "http://localhost:8000/v1/audio/transcriptions",
            "http://localhost:8000/v1/chat/completions",
        ),
    ],
)
def test_chat_completions_url_accepts_root_v1_and_full_url(base: str, expected: str):
    assert vllm.chat_completions_url(base) == expected


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


def test_vllm_chat_asr_matches_force_language_payload(tmp_path: Path, monkeypatch):
    audio = tmp_path / "sample.wav"
    audio_bytes = b"RIFF-test-audio"
    audio.write_bytes(audio_bytes)
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "content": "language Chinese<asr_text>不需要谢谢",
                            }
                        }
                    ]
                }
            ).encode()

    def fake_urlopen(request, timeout):
        captured.update(
            url=request.full_url,
            body=request.data,
            headers=dict(request.headers),
            timeout=timeout,
        )
        return Response()

    monkeypatch.setattr(vllm.urllib.request, "urlopen", fake_urlopen)
    context = "vocabulary:没有，暂时不用"
    result = vllm.call_vllm_chat_asr(
        str(audio),
        {
            "api_base": "http://localhost:8000/v1",
            "api_key": "secret",
            "model": "qwen3-asr",
            "context": context,
            "temperature": 0,
            "chat_template_kwargs": {"force_language": "Chinese"},
        },
    )

    assert captured["url"] == "http://localhost:8000/v1/chat/completions"
    payload = json.loads(captured["body"].decode("utf-8"))
    expected_url = (
        "data:audio/wav;base64," + base64.b64encode(audio_bytes).decode("ascii")
    )
    assert payload["messages"] == [
        {"role": "system", "content": context},
        {
            "role": "user",
            "content": [
                {
                    "type": "audio_url",
                    "audio_url": {"url": expected_url},
                }
            ],
        },
    ]
    assert payload["chat_template_kwargs"] == {"force_language": "Chinese"}
    assert payload["temperature"] == 0
    assert payload["model"] == "qwen3-asr"
    assert result["text"] == "不需要谢谢"
    assert result["language"] == "Chinese"
