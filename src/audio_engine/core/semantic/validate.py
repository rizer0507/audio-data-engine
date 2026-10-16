"""Parse model content and validate against task JSON Schema."""

from __future__ import annotations

import json
import re
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError


class SemanticParseError(Exception):
    def __init__(self, message: str, *, error_code: str = "parse_error"):
        super().__init__(message)
        self.error_code = error_code


def load_validator(schema: dict[str, Any]) -> Draft202012Validator:
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        raise ValueError(f"invalid output JSON Schema: {exc}") from exc
    return Draft202012Validator(schema)


def extract_json_object(text: str) -> Any:
    raw = (text or "").strip()
    if not raw:
        raise SemanticParseError("empty model content", error_code="parse_error")
    # Strip optional ```json fences without silently inventing content.
    fence = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```$", raw, re.IGNORECASE)
    if fence:
        raw = fence.group(1).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SemanticParseError(f"JSON parse failed: {exc}", error_code="parse_error") from exc


def validate_result(data: Any, validator: Draft202012Validator) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise SemanticParseError(
            f"result must be a JSON object, got {type(data).__name__}",
            error_code="schema_error",
        )
    try:
        validator.validate(data)
    except ValidationError as exc:
        raise SemanticParseError(
            f"schema validation failed: {exc.message}",
            error_code="schema_error",
        ) from exc
    return data
