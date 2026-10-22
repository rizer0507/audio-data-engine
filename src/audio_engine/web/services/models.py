"""Model connection management and two-level checks (not real vLLM lifecycle)."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
import uuid
from typing import Any
from urllib.parse import urlparse

from audio_engine.web.settings import ModelConnectionConfig, WebSettings, save_models_to_web_yaml


class ModelServiceError(Exception):
    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}


def normalize_v1_base(base_url: str) -> str:
    text = str(base_url or "").rstrip("/")
    if not text:
        raise ModelServiceError("BASE_URL_REQUIRED", "base_url is required")
    parsed = urlparse(text)
    if parsed.hostname in {"0.0.0.0", "::"}:
        raise ModelServiceError(
            "BASE_URL_INVALID_HOST",
            "0.0.0.0 is a bind address, not a client target; use 127.0.0.1",
            {"base_url": text},
        )
    if not parsed.scheme or not parsed.hostname:
        raise ModelServiceError("BASE_URL_INVALID", "base_url must be absolute http(s) URL")
    if parsed.port is not None and not (1 <= parsed.port <= 65535):
        raise ModelServiceError("BASE_URL_PORT_INVALID", "port must be 1..65535")
    # Avoid /v1/v1
    if text.endswith("/v1"):
        return text
    if text.endswith("/v1/"):
        return text.rstrip("/")
    return text + "/v1"


def new_connection_id() -> str:
    return f"conn_{uuid.uuid4().hex[:8]}"


def default_model_name(provider: str, base_url: str) -> str:
    parsed = urlparse(base_url)
    host = parsed.hostname or "unknown"
    port = parsed.port
    if port is None:
        port = 443 if parsed.scheme == "https" else 80
    return f"{provider} {host}:{port}"


class ModelService:
    def __init__(self, settings: WebSettings) -> None:
        self.settings = settings

    def list_models(self) -> dict[str, Any]:
        items = [m.public_dict() for m in self.settings.models]
        return {
            "items": items,
            "next_cursor": None,
            "total": len(items),
            "revision": self.settings.revision,
        }

    def get(self, connection_id: str) -> ModelConnectionConfig:
        for m in self.settings.models:
            if m.connection_id == connection_id:
                return m
        raise ModelServiceError("MODEL_NOT_FOUND", f"connection_id not found: {connection_id}")

    def create(self, payload: dict[str, Any]) -> dict[str, Any]:
        provider = str(payload.get("provider") or "").lower()
        if provider not in {"glm", "qwen"}:
            raise ModelServiceError("PROVIDER_UNSUPPORTED", f"unsupported provider: {provider}")
        base = normalize_v1_base(str(payload.get("base_url") or ""))
        model_name = str(payload.get("model") or "").strip()
        if not model_name:
            raise ModelServiceError("MODEL_NAME_REQUIRED", "model (served-model-name) is required")

        # Dedup by provider + normalized base_url + model; reuse connection_id.
        for existing in self.settings.models:
            if (
                existing.provider == provider
                and normalize_v1_base(existing.base_url) == base
                and existing.model == model_name
            ):
                out = existing.public_dict()
                out["revision"] = self.settings.revision
                out["reused"] = True
                return out

        cid = new_connection_id()
        while any(m.connection_id == cid for m in self.settings.models):
            cid = new_connection_id()

        display = str(payload.get("name") or "").strip() or default_model_name(provider, base)
        model = ModelConnectionConfig(
            connection_id=cid,
            name=display,
            provider=provider,
            base_url=base,
            model=model_name,
            model_revision=str(payload.get("model_revision") or "deployment_001"),
            api_key_env=str(payload.get("api_key_env") or ""),
            timeout_seconds=int(payload.get("timeout_seconds") or 120),
            concurrency=int(payload.get("concurrency") or 4),
        )
        models = list(self.settings.models) + [model]
        save_models_to_web_yaml(self.settings, models)
        out = model.public_dict()
        out["revision"] = self.settings.revision
        out["reused"] = False
        return out

    def patch(
        self, connection_id: str, payload: dict[str, Any], *, expected_revision: int | None
    ) -> dict[str, Any]:
        if expected_revision is not None and expected_revision != self.settings.revision:
            raise ModelServiceError(
                "REVISION_CONFLICT",
                "model config revision mismatch",
                {"current_revision": self.settings.revision},
            )
        models = list(self.settings.models)
        idx = next((i for i, m in enumerate(models) if m.connection_id == connection_id), None)
        if idx is None:
            raise ModelServiceError("MODEL_NOT_FOUND", f"connection_id not found: {connection_id}")
        cur = models[idx]
        data = {
            "connection_id": cur.connection_id,
            "name": payload.get("name", cur.name),
            "provider": payload.get("provider", cur.provider),
            "base_url": payload.get("base_url", cur.base_url),
            "model": payload.get("model", cur.model),
            "model_revision": payload.get("model_revision", cur.model_revision),
            "api_key_env": payload.get("api_key_env", cur.api_key_env),
            "timeout_seconds": payload.get("timeout_seconds", cur.timeout_seconds),
            "concurrency": payload.get("concurrency", cur.concurrency),
        }
        if data["base_url"]:
            data["base_url"] = normalize_v1_base(str(data["base_url"]))
        if str(data["provider"]).lower() not in {"glm", "qwen"}:
            raise ModelServiceError("PROVIDER_UNSUPPORTED", "provider must be glm|qwen")
        models[idx] = ModelConnectionConfig(**{**data, "provider": str(data["provider"]).lower()})
        save_models_to_web_yaml(self.settings, models)
        out = models[idx].public_dict()
        out["revision"] = self.settings.revision
        return out

    def check(
        self,
        connection_id: str,
        *,
        level: str = "models",
        sample_path: str | None = None,
    ) -> dict[str, Any]:
        model = self.get(connection_id)
        base = normalize_v1_base(model.base_url)
        started = time.time()
        result: dict[str, Any] = {
            "connection_id": connection_id,
            "level": level,
            "base_url": base,
            "configured_model": model.model,
            "provider": model.provider,
            "ok": False,
            "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        try:
            models_url = base + "/models"
            req = urllib.request.Request(models_url, method="GET")
            with urllib.request.urlopen(req, timeout=model.timeout_seconds) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                result["http_status"] = resp.status
            ids = []
            for item in body.get("data") or []:
                if isinstance(item, dict) and item.get("id"):
                    ids.append(str(item["id"]))
            result["returned_model_ids"] = ids
            result["latency_ms"] = int((time.time() - started) * 1000)
            if model.model and model.model not in ids:
                result["ok"] = False
                result["code"] = "MODEL_NAME_MISMATCH"
                result["message"] = "服务未提供所配置的模型名"
                return result
            result["ok"] = True
            result["identity_scope"] = "public /v1/models listing only; weights not verified"
            if level == "models":
                return result
            if level != "transcription":
                raise ModelServiceError("CHECK_LEVEL_INVALID", "level must be models|transcription")
            if not sample_path:
                result["ok"] = False
                result["code"] = "SAMPLE_REQUIRED"
                result["message"] = "transcription check requires sample_ref"
                return result
            # Protocol-accurate short call using existing helpers when possible.
            tx = self._transcription_probe(model, sample_path)
            result.update(tx)
            return result
        except ModelServiceError:
            raise
        except urllib.error.HTTPError as exc:
            result["ok"] = False
            result["http_status"] = exc.code
            result["code"] = "MODEL_HTTP_ERROR"
            result["message"] = str(exc)
            result["latency_ms"] = int((time.time() - started) * 1000)
            return result
        except Exception as exc:  # noqa: BLE001
            result["ok"] = False
            result["code"] = "MODEL_CHECK_FAILED"
            result["message"] = str(exc)
            result["latency_ms"] = int((time.time() - started) * 1000)
            return result

    def _transcription_probe(self, model: ModelConnectionConfig, sample_path: str) -> dict[str, Any]:
        settings = {
            "api_base": model.base_url,
            "model": model.model,
            "api_key": "dummy",
            "timeout": model.timeout_seconds,
        }
        import os

        if model.api_key_env and os.environ.get(model.api_key_env):
            settings["api_key"] = os.environ[model.api_key_env]
        try:
            if model.provider == "qwen":
                from audio_engine.operators.asr.vllm import call_vllm_chat_asr

                out = call_vllm_chat_asr(sample_path, settings)
            else:
                from audio_engine.operators.asr.vllm import call_vllm_transcription

                out = call_vllm_transcription(sample_path, settings)
            text = ""
            if isinstance(out, dict):
                text = str(out.get("text") or out.get("transcript") or "")[:500]
            return {
                "ok": True,
                "transcription_ok": True,
                "transcript_preview": text,
                "note": "protocol probe against configured endpoint; not a quality claim",
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "ok": False,
                "transcription_ok": False,
                "code": "TRANSCRIPTION_FAILED",
                "message": str(exc),
            }
