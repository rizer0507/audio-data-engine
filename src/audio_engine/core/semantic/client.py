"""Pure-text OpenAI-compatible Chat Completions client for semantic judge."""

from __future__ import annotations

import json
import random
import threading
import time
from dataclasses import dataclass
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest

from audio_engine.core.semantic.config import ServiceConfig
from audio_engine.core.semantic.types import STRUCTURED_OUTPUT_JSON_SCHEMA


def chat_completions_url(api_base: str) -> str:
    """Build OpenAI-compatible chat-completions URL (text-only; no ASR coupling)."""
    base = api_base.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


class SemanticRequestError(Exception):
    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        status_code: int | None = None,
        error_code: str = "request_error",
    ):
        super().__init__(message)
        self.retryable = retryable
        self.status_code = status_code
        self.error_code = error_code


@dataclass
class ChatResponse:
    content: str
    raw: dict[str, Any]
    latency_ms: float
    usage: dict[str, Any] | None
    attempts: int


class RateLimiter:
    """Optional token-bucket style rate limit shared across threads."""

    def __init__(self, rps: float | None):
        self._rps = rps
        self._lock = threading.Lock()
        self._next = 0.0

    def acquire(self) -> None:
        if not self._rps or self._rps <= 0:
            return
        with self._lock:
            now = time.monotonic()
            if now < self._next:
                time.sleep(self._next - now)
                now = time.monotonic()
            self._next = now + (1.0 / self._rps)


class ConcurrencyGate:
    """Bounded active HTTP request gate (retries share the same slot)."""

    def __init__(self, limit: int):
        if limit < 1:
            raise ValueError(f"concurrency limit must be >= 1, got {limit}")
        self.limit = limit
        self._sem = threading.BoundedSemaphore(limit)
        self._active = 0
        self._peak = 0
        self._lock = threading.Lock()

    def __enter__(self) -> ConcurrencyGate:
        self._sem.acquire()
        with self._lock:
            self._active += 1
            self._peak = max(self._peak, self._active)
        return self

    def __exit__(self, *args: Any) -> None:
        with self._lock:
            self._active = max(0, self._active - 1)
        self._sem.release()

    @property
    def active(self) -> int:
        with self._lock:
            return self._active

    @property
    def peak(self) -> int:
        with self._lock:
            return self._peak


class SemanticChatClient:
    def __init__(self, service: ServiceConfig, *, gate: ConcurrencyGate | None = None):
        self.service = service
        self.gate = gate or ConcurrencyGate(service.max_concurrency_total)
        self.rate_limiter = RateLimiter(service.rate_limit_rps)
        self._lock = threading.Lock()

    def close(self) -> None:
        return None

    def list_models(self, timeout_s: float = 10.0) -> list[str]:
        base = self.service.api_base.rstrip("/")
        if base.endswith("/v1"):
            url = f"{base}/models"
        elif base.endswith("/chat/completions"):
            url = base[: -len("/chat/completions")] + "/models"
        else:
            url = f"{base}/v1/models"
        headers = {"Accept": "application/json"}
        api_key = self.service.resolve_api_key()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        req = urlrequest.Request(url, headers=headers, method="GET")
        try:
            with urlrequest.urlopen(req, timeout=timeout_s) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            raise SemanticRequestError(
                f"model list probe failed: {exc}",
                retryable=False,
                error_code="probe_error",
            ) from exc
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list):
            return []
        names = []
        for item in data:
            if isinstance(item, dict) and item.get("id"):
                names.append(str(item["id"]))
        return names

    def build_payload(
        self,
        *,
        system: str | None,
        user: str,
        output_schema: dict[str, Any] | None,
        schema_name: str,
    ) -> dict[str, Any]:
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        payload: dict[str, Any] = {
            "model": self.service.model,
            "messages": messages,
            **self.service.generation.request_fields(),
            "stream": False,
        }
        if self.service.enable_thinking is not None:
            payload["chat_template_kwargs"] = {
                "enable_thinking": bool(self.service.enable_thinking)
            }
        if (
            self.service.structured_output_mode == STRUCTURED_OUTPUT_JSON_SCHEMA
            and output_schema is not None
        ):
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": output_schema,
                },
            }
        return payload

    def chat(
        self,
        *,
        system: str | None,
        user: str,
        output_schema: dict[str, Any] | None,
        schema_name: str = "semantic_result",
    ) -> ChatResponse:
        payload = self.build_payload(
            system=system,
            user=user,
            output_schema=output_schema,
            schema_name=schema_name,
        )
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        api_key = self.service.resolve_api_key()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        url = chat_completions_url(self.service.api_base)
        retry = self.service.retry
        attempt = 0
        last_exc: Exception | None = None
        started_all = time.perf_counter()
        deadline = started_all + retry.request_timeout_s

        with self.gate:
            while attempt < retry.max_attempts:
                attempt += 1
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                self.rate_limiter.acquire()
                timeout = min(
                    retry.read_timeout_s + retry.connect_timeout_s,
                    max(0.1, remaining),
                )
                started = time.perf_counter()
                try:
                    raw = self._post(url, body=body, headers=headers, timeout=timeout)
                    content = self._extract_content(raw)
                    usage = raw.get("usage") if isinstance(raw.get("usage"), dict) else None
                    return ChatResponse(
                        content=content,
                        raw=raw,
                        latency_ms=(time.perf_counter() - started) * 1000.0,
                        usage=usage,
                        attempts=attempt,
                    )
                except SemanticRequestError as exc:
                    last_exc = exc
                    if not exc.retryable or attempt >= retry.max_attempts:
                        raise
                    self._sleep_backoff(attempt, retry_after=None, retry=retry)
                except Exception as exc:  # noqa: BLE001
                    last_exc = SemanticRequestError(
                        str(exc), retryable=True, error_code="request_error"
                    )
                    if attempt >= retry.max_attempts:
                        raise last_exc from exc
                    self._sleep_backoff(attempt, retry_after=None, retry=retry)

        raise SemanticRequestError(
            f"request exhausted after {attempt} attempts: {last_exc}",
            retryable=False,
            error_code="request_error",
        )

    def _sleep_backoff(
        self,
        attempt: int,
        *,
        retry_after: float | None,
        retry: Any,
    ) -> None:
        if retry_after is not None and retry_after > 0:
            time.sleep(min(retry_after, retry.backoff_max_s))
            return
        base = min(retry.backoff_max_s, retry.backoff_base_s * (2 ** (attempt - 1)))
        time.sleep(base * (0.5 + random.random()))

    def _post(
        self,
        url: str,
        *,
        body: bytes,
        headers: dict[str, str],
        timeout: float,
    ) -> dict[str, Any]:
        req = urlrequest.Request(url, data=body, headers=headers, method="POST")
        try:
            with urlrequest.urlopen(req, timeout=timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urlerror.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            code = int(exc.code)
            retryable = code in {408, 429, 500, 502, 503, 504}
            retry_after = None
            if code == 429:
                raw_ra = exc.headers.get("Retry-After") if exc.headers else None
                if raw_ra:
                    try:
                        retry_after = float(raw_ra)
                    except ValueError:
                        retry_after = None
            err = SemanticRequestError(
                f"HTTP {code}: {detail[:500]}",
                retryable=retryable,
                status_code=code,
                error_code="http_error" if code not in {401, 403} else "auth_error",
            )
            if retryable and retry_after is not None:
                self._sleep_backoff(1, retry_after=retry_after, retry=self.service.retry)
            raise err from exc
        except urlerror.URLError as exc:
            raise SemanticRequestError(
                f"connection error: {exc.reason}",
                retryable=True,
                error_code="connection_error",
            ) from exc
        if not isinstance(payload, dict):
            raise SemanticRequestError(
                f"invalid response type: {type(payload).__name__}",
                retryable=False,
                error_code="parse_error",
            )
        return payload

    @staticmethod
    def _extract_content(payload: dict[str, Any]) -> str:
        try:
            message = payload["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise SemanticRequestError(
                f"invalid chat response: {payload!r}",
                retryable=False,
                error_code="parse_error",
            ) from exc
        content = message.get("content")
        if content is None:
            raise SemanticRequestError(
                "empty message.content",
                retryable=False,
                error_code="parse_error",
            )
        # Do not treat reasoning fields as the answer.
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text") or ""))
                elif isinstance(item, str):
                    parts.append(item)
            return "".join(parts)
        return str(content)
