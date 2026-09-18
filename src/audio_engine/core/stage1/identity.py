"""vLLM model identity checks via OpenAI-compatible /v1/models."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelIdentity:
    api_base: str
    model_ids: tuple[str, ...]
    raw: dict[str, Any]

    def matches(self, expected: str) -> bool:
        return expected in self.model_ids


def models_url(api_base: str) -> str:
    base = api_base.rstrip("/")
    if base.endswith("/v1/models"):
        return base
    if base.endswith("/v1"):
        return f"{base}/models"
    return f"{base}/v1/models"


def fetch_model_identity(api_base: str, *, timeout_s: float = 10.0) -> ModelIdentity:
    request = urllib.request.Request(models_url(api_base), method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"查询模型身份失败 HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"查询模型身份失败: {exc.reason}") from exc

    if not isinstance(payload, dict):
        raise RuntimeError(f"/v1/models 返回格式无效: {payload!r}")
    data = payload.get("data") or []
    if not isinstance(data, list):
        raise RuntimeError(f"/v1/models data 不是列表: {payload!r}")
    model_ids: list[str] = []
    for item in data:
        if isinstance(item, dict) and item.get("id") is not None:
            model_ids.append(str(item["id"]))
    if not model_ids:
        raise RuntimeError(f"/v1/models 未返回模型 id: {payload!r}")
    return ModelIdentity(api_base=api_base, model_ids=tuple(model_ids), raw=payload)


def assert_served_model(api_base: str, expected: str, *, timeout_s: float = 10.0) -> ModelIdentity:
    identity = fetch_model_identity(api_base, timeout_s=timeout_s)
    if not identity.matches(expected):
        raise RuntimeError(
            f"模型身份不匹配: 期望 served-model-name={expected!r}, "
            f"实际 ids={list(identity.model_ids)!r} @ {api_base}"
        )
    return identity
