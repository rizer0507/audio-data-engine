"""Semantic judge status constants and shared helpers."""

from __future__ import annotations

ENGINE_VERSION = "1.0.0"

# Execution statuses written under labels.semantic_results[task_id].status
STATUS_SUCCESS = "success"
STATUS_SKIPPED_INPUT = "skipped_input"
STATUS_INPUT_ERROR = "input_error"
STATUS_TEMPLATE_ERROR = "template_error"
STATUS_REQUEST_ERROR = "request_error"
STATUS_PARSE_ERROR = "parse_error"
STATUS_SCHEMA_ERROR = "schema_error"
STATUS_CONTEXT_LIMIT = "context_limit"
STATUS_MOCK = "mock"

SUCCESS_LIKE = frozenset({STATUS_SUCCESS, STATUS_SKIPPED_INPUT, STATUS_MOCK})
ERROR_STATUSES = frozenset(
    {
        STATUS_INPUT_ERROR,
        STATUS_TEMPLATE_ERROR,
        STATUS_REQUEST_ERROR,
        STATUS_PARSE_ERROR,
        STATUS_SCHEMA_ERROR,
        STATUS_CONTEXT_LIMIT,
    }
)

# Run evidence statuses (reuse neutral evidence vocabulary)
RUN_SUCCESS_TEXT = "success_text"
RUN_SUCCESS_EMPTY = "success_empty"
RUN_FAILED = "failed"
RUN_MISSING = "missing"

STRUCTURED_OUTPUT_JSON_SCHEMA = "json_schema"
STRUCTURED_OUTPUT_PROMPT_ONLY = "prompt_schema_validate"
