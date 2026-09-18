"""Transient vs deterministic errors, backoff, and circuit breaker for stage1."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


DEFAULT_MAX_ATTEMPTS = 3  # initial try + 2 retries
DEFAULT_BACKOFF_S = (2.0, 5.0, 15.0)
CIRCUIT_CONSECUTIVE_LIMIT = 3


_DETERMINISTIC_PATTERNS = (
    r"部署配置缺项",
    r"authorized_gpus",
    r"不在 authorized_gpus",
    r"非法 batch",
    r"--model 缺少",
    r"未知家族",
    r"格式应为",
    r"启动器检查失败",
    r"配置不存在",
    r"MissingConfigError",
    r"identity 不是映射",
    r"output identity already exists with different content",
    r"declared input_audio_digest differs",
    r"audio base requires",
    r"ASR output has missing hashes",
    r"六路 execution_id",
    r"artifact_id 必须",
    r"对账门禁未通过",
)

_TRANSIENT_PATTERNS = (
    r"exit=\d+",
    r"失败 exit=",
    r"Timeout",
    r"timed out",
    r"Connection refused",
    r"Connection reset",
    r"Temporary failure",
    r"Unavailable",
    r"Broken pipe",
    r"CUDA out of memory",
    r"\bOOM\b",
    r"out of memory",
    r"服务在 .* 内未就绪",
    r"查询模型身份失败",
    r"vLLM ASR",
    r"HTTP 5\d\d",
    r"worker pid=",
)


@dataclass(frozen=True)
class ErrorClass:
    kind: str  # transient | deterministic | oom
    retryable: bool
    reason: str


def classify_error(exc: BaseException | str) -> ErrorClass:
    text = str(exc)
    lowered = text.lower()
    if any(re.search(pat, text, re.I) for pat in _DETERMINISTIC_PATTERNS):
        return ErrorClass("deterministic", False, "deterministic_config_or_contract")
    if "out of memory" in lowered or re.search(r"\boom\b", lowered) or "cuda out of memory" in lowered:
        return ErrorClass("oom", True, "oom_transient_may_retry_same_semantics")
    if isinstance(exc, (FileNotFoundError, MissingConfigLike)):
        # Missing outputs after a run may be transient; missing config paths are deterministic.
        if "配置" in text or "vllm_bin" in text or "engine_python" in text or "runtime" in text.lower():
            return ErrorClass("deterministic", False, "missing_required_path")
        return ErrorClass("transient", True, "missing_output_or_weight_transient")
    if isinstance(exc, (ValueError, TypeError, KeyError)) and not any(
        re.search(pat, text, re.I) for pat in _TRANSIENT_PATTERNS
    ):
        return ErrorClass("deterministic", False, "value_error")
    if any(re.search(pat, text, re.I) for pat in _TRANSIENT_PATTERNS):
        return ErrorClass("transient", True, "transient_runtime")
    # Default: treat unknown RuntimeError as retryable once-class transient.
    if isinstance(exc, RuntimeError):
        return ErrorClass("transient", True, "runtime_error_default_retry")
    return ErrorClass("deterministic", False, "unknown_non_retryable")


class MissingConfigLike(Exception):
    """Marker for tests / policy."""


def backoff_seconds(attempt_index: int, schedule: tuple[float, ...] = DEFAULT_BACKOFF_S) -> float:
    """attempt_index is 0-based for the retry about to sleep before next try."""
    if attempt_index < 0:
        return 0.0
    if attempt_index >= len(schedule):
        return schedule[-1]
    return schedule[attempt_index]


def should_trip_circuit(consecutive_failures: int, limit: int = CIRCUIT_CONSECUTIVE_LIMIT) -> bool:
    return consecutive_failures >= limit


def attempt_record(
    *,
    attempt: int,
    status: str,
    error: str | None = None,
    error_kind: str | None = None,
    worker_token: str | None = None,
) -> dict[str, Any]:
    from audio_engine.core.catalog import utc_now

    return {
        "attempt": attempt,
        "status": status,
        "error": error,
        "error_kind": error_kind,
        "worker_token": worker_token,
        "at": utc_now(),
    }
