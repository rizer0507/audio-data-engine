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


def test_sophon_chat_asr_requests_text_only_transcription(tmp_path: Path, monkeypatch):
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"RIFF-test")
    context = "vocabulary:不用，不需要"

    def fake_http_json(url, *, body, headers, timeout):
        payload = json.loads(body)
        assert url == "http://127.0.0.1:6179/v1/chat/completions"
        assert payload["model"] == "/models/Sophon-Realtime-tag"
        assert payload["modalities"] == ["text"]
        assert payload["stream"] is False
        assert payload["language"] == "zh"
        assert payload["temperature"] == 0
        assert "chat_template_kwargs" not in payload
        system, user = payload["messages"]
        assert "逐字转写" in system["content"]
        assert context in system["content"]
        assert user["content"] == [
            {"type": "text", "text": "<|audio|>"},
            {
                "type": "input_audio",
                "input_audio": {
                    "data": base64.b64encode(audio.read_bytes()).decode(),
                    "format": "wav",
                },
            },
        ]
        return {"choices": [{"message": {"content": " 不用，谢谢。 "}}]}

    monkeypatch.setattr(vllm, "_http_json", fake_http_json)
    result = vllm.call_vllm_chat_asr(
        str(audio),
        {
            "api_base": "http://127.0.0.1:6179",
            "model": "/models/Sophon-Realtime-tag",
            "chat_format": "sophon",
            "context": context,
        },
    )
    assert result["text"] == "不用，谢谢。"
    assert result["language"] == "zh"


@pytest.mark.parametrize(
    ("raw_text", "expected"),
    [
        ("［laughter］（开心）<strong>嗯，不用不用。</strong>（叹气）", "嗯,不用不用。"),
        ("[sigh]<laughter>哈哈！</laughter>。", ""),
        ("<laughter>我知道</laughter>[unknown_event", "我知道"),
        ("[cough]听不清。", "听不清。"),
    ],
)
def test_sophon_v1_cleans_tags_and_keeps_original(tmp_path, monkeypatch, raw_text, expected):
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"RIFF-test")
    prompt = "只转写清楚可辨的词句。"

    def fake_http_json(_url, *, body, **_kwargs):
        assert json.loads(body)["messages"][0]["content"] == prompt
        return {"choices": [{"message": {"content": raw_text}}]}

    monkeypatch.setattr(vllm, "_http_json", fake_http_json)
    result = vllm.call_vllm_chat_asr(
        str(audio),
        {
            "api_base": "http://127.0.0.1:6179",
            "chat_format": "sophon",
            "system_prompt": prompt,
            "strip_sophon_tags": True,
            "context": "",
        },
    )
    assert result["text"] == expected
    assert result["extra"]["raw_text"] == raw_text
