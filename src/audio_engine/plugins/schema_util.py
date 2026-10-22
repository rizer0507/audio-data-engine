"""JSON Schema helpers for plugin parameters."""

from __future__ import annotations

from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from audio_engine.plugins.contracts import ValidationIssue, ValidationReport


EMPTY_OBJECT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {},
    "title": "参数",
    "description": "本插件无可覆盖任务参数",
}


def validate_against_schema(
    parameters: dict[str, Any],
    schema: dict[str, Any],
    *,
    field_prefix: str = "parameters",
    plugin_ref: dict[str, str] | None = None,
) -> ValidationReport:
    report = ValidationReport(ok=True)
    if not isinstance(parameters, dict):
        report.ok = False
        report.errors.append(
            ValidationIssue(
                code="PARAMETERS_NOT_OBJECT",
                message="parameters must be an object",
                field_path=field_prefix,
                plugin_ref=plugin_ref,
            )
        )
        return report
    try:
        Draft202012Validator.check_schema(schema)
    except Exception as exc:  # noqa: BLE001
        report.ok = False
        report.errors.append(
            ValidationIssue(
                code="SCHEMA_INVALID",
                message=f"parameters_schema is invalid: {exc}",
                field_path="parameters_schema",
                plugin_ref=plugin_ref,
            )
        )
        return report

    validator = Draft202012Validator(schema)
    for error in sorted(validator.iter_errors(parameters), key=lambda e: list(e.path)):
        path = ".".join([field_prefix, *[str(p) for p in error.path]])
        report.ok = False
        report.errors.append(
            ValidationIssue(
                code="SCHEMA_VALIDATION",
                message=error.message,
                field_path=path.rstrip("."),
                plugin_ref=plugin_ref,
                details={"validator": error.validator},
            )
        )
    return report


def assert_schema_or_raise(schema: dict[str, Any], *, definition_path: str) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    try:
        Draft202012Validator.check_schema(schema)
    except Exception as exc:  # noqa: BLE001
        issues.append(
            ValidationIssue(
                code="SCHEMA_INVALID",
                message=str(exc),
                definition_path=definition_path,
                field_path="parameters_schema",
            )
        )
        return issues
    if schema.get("type") != "object":
        issues.append(
            ValidationIssue(
                code="SCHEMA_INVALID",
                message="parameters_schema.type must be object",
                definition_path=definition_path,
                field_path="parameters_schema.type",
            )
        )
    if schema.get("additionalProperties", True) is not False:
        issues.append(
            ValidationIssue(
                code="SCHEMA_ADDITIONAL_PROPERTIES",
                message="parameters_schema must set additionalProperties: false",
                definition_path=definition_path,
                field_path="parameters_schema.additionalProperties",
            )
        )
    return issues


def format_validation_error(exc: ValidationError) -> str:
    return exc.message
