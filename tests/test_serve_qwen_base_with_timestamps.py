import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from starlette.responses import Response

from scripts.serve_qwen_base_with_timestamps import ForcedAlignmentMiddleware


def post_audio(upstream_body: str, status: int = 200):
    async def upstream(scope, receive, send):
        await receive()
        await Response(upstream_body, status_code=status, media_type="application/json")(
            scope, receive, send
        )

    middleware = ForcedAlignmentMiddleware(upstream)
    item = SimpleNamespace(text="你好", start_time=0.1, end_time=0.25)
    middleware.aligner = Mock()
    middleware.aligner.align.return_value = [SimpleNamespace(items=[item])]

    async def request():
        transport = httpx.ASGITransport(app=middleware)
        audio = {"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64,AA=="}}
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post(
                "/v1/chat/completions", json={"messages": [{"content": [audio]}]}
            )

    return asyncio.run(request()), middleware.aligner


@pytest.mark.parametrize("upstream_body", [
    "null",
    "{",
    '{"choices":[]}',
    '{"choices":[{"message":{"content":"你好"}},{"message":{"content":42}}]}',
])
def test_rejects_invalid_upstream_chat_response(upstream_body, caplog):
    response, aligner = post_audio(upstream_body)

    assert response.status_code == 502
    assert "vLLM" in response.json()["error"]["message"]
    assert "ForcedAligner 请求失败" not in caplog.text
    aligner.align.assert_not_called()


def test_aligns_valid_upstream_chat_response():
    response, aligner = post_audio('{"choices":[{"message":{"content":"你好"}}]}')

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"] == {
        "content": "你好",
        "timestamps": [{"text": "你好", "start_ms": 100, "end_ms": 250}],
    }
    aligner.align.assert_called_once()


def test_preserves_empty_transcript_in_valid_chat_response():
    response, aligner = post_audio('{"choices":[{"message":{"content":null}}]}')

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"] == {"content": None, "timestamps": []}
    aligner.align.assert_not_called()


def test_preserves_native_upstream_error():
    response, aligner = post_audio('{"error":{"message":"engine error"}}', status=500)

    assert response.status_code == 500
    assert response.json() == {"error": {"message": "engine error"}}
    aligner.align.assert_not_called()
