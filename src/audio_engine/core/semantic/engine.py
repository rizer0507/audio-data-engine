"""Per-sample semantic judge execution."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

from audio_engine.core.sample import Sample
from audio_engine.core.semantic.client import SemanticChatClient, SemanticRequestError
from audio_engine.core.semantic.config import FrozenSemanticBundle
from audio_engine.core.semantic.evidence import (
    build_families_context,
    build_sample_context,
)
from audio_engine.core.semantic.fingerprint import result_fingerprint, semantic_cache_key
from audio_engine.core.semantic.templates import TemplateRenderer
from audio_engine.core.semantic.types import (
    ENGINE_VERSION,
    STATUS_CONTEXT_LIMIT,
    STATUS_INPUT_ERROR,
    STATUS_MOCK,
    STATUS_PARSE_ERROR,
    STATUS_REQUEST_ERROR,
    STATUS_SCHEMA_ERROR,
    STATUS_SKIPPED_INPUT,
    STATUS_SUCCESS,
    STATUS_TEMPLATE_ERROR,
    STRUCTURED_OUTPUT_JSON_SCHEMA,
)
from audio_engine.core.semantic.validate import (
    SemanticParseError,
    extract_json_object,
    load_validator,
    validate_result,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _truncate_raw(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    return text[:limit], True


class SemanticJudgeEngine:
    def __init__(
        self,
        bundle: FrozenSemanticBundle,
        *,
        client: SemanticChatClient | None = None,
        mock: bool = False,
    ):
        self.bundle = bundle
        self.mock = bool(mock)
        self.client = client
        self.renderer = TemplateRenderer(
            template_root=bundle.template_root,
            user_template=bundle.user_template,
            system_template=bundle.system_template,
            included=bundle.included_templates,
        )
        self.validator = load_validator(bundle.output_schema)
        self._probed = False

    def probe(self) -> dict[str, Any]:
        if self.mock:
            return {"ok": True, "mode": "mock"}
        if self.client is None:
            raise RuntimeError("semantic client not configured")
        info: dict[str, Any] = {"ok": True, "mode": "live"}
        try:
            models = self.client.list_models()
            info["models"] = models
            if models and self.bundle.service.model not in models:
                raise RuntimeError(
                    f"served model {self.bundle.service.model!r} not in /v1/models: {models}"
                )
        except SemanticRequestError as exc:
            # Model list is best-effort; still require a chat probe below.
            info["models_error"] = str(exc)

        system, user = "Return a JSON object.", '{"probe": true}'
        # Minimal probe without forcing business schema.
        resp = self.client.chat(
            system=system,
            user=user,
            output_schema=None,
            schema_name="probe",
        )
        info["probe_content_len"] = len(resp.content or "")
        self._probed = True
        return info

    def ensure_probed(self) -> None:
        if self.mock or self._probed or not self.bundle.service.probe_on_start:
            return
        self.probe()

    def cache_key(self, sample: Sample) -> str:
        return semantic_cache_key(sample, self.bundle, mock=self.mock)

    def judge(self, sample: Sample) -> dict[str, Any]:
        """Return labels/lineage updates for one sample. Never raises for row errors."""
        task = self.bundle.task
        service = self.bundle.service
        started = _utc_now()
        input_fp = self.cache_key(sample)

        try:
            sample_ctx = build_sample_context(sample, task.evidence.sample_context_fields)
            families, evidence_summary = build_families_context(sample, task.evidence)
        except Exception as exc:  # noqa: BLE001
            return self._result_updates(
                status=STATUS_INPUT_ERROR,
                result=None,
                raw_content=None,
                error_code="input_error",
                error_message=str(exc),
                attempts=0,
                latency_ms=0.0,
                evidence_summary={},
                input_fingerprint=input_fp,
                started=started,
            )

        if evidence_summary.get("insufficient_evidence"):
            mode = task.evidence.on_insufficient_evidence
            if mode == "skip":
                return self._result_updates(
                    status=STATUS_SKIPPED_INPUT,
                    result=None,
                    raw_content=None,
                    error_code="insufficient_evidence",
                    error_message="required evidence missing; skipped by policy",
                    attempts=0,
                    latency_ms=0.0,
                    evidence_summary=evidence_summary,
                    input_fingerprint=input_fp,
                    started=started,
                )
            if mode == "fail":
                return self._result_updates(
                    status=STATUS_INPUT_ERROR,
                    result=None,
                    raw_content=None,
                    error_code="insufficient_evidence",
                    error_message="required evidence missing; fail by policy",
                    attempts=0,
                    latency_ms=0.0,
                    evidence_summary=evidence_summary,
                    input_fingerprint=input_fp,
                    started=started,
                )

        context = {
            "sample": sample_ctx,
            "families": families,
            "task": {
                "task_id": task.task_id,
                "task_version": task.task_version,
                "parameters": task.parameters,
            },
            "output_schema": self.bundle.output_schema,
            "evidence_summary": evidence_summary,
        }
        try:
            system, user = self.renderer.render(context)
        except Exception as exc:  # noqa: BLE001
            return self._result_updates(
                status=STATUS_TEMPLATE_ERROR,
                result=None,
                raw_content=None,
                error_code="template_error",
                error_message=str(exc),
                attempts=0,
                latency_ms=0.0,
                evidence_summary=evidence_summary,
                input_fingerprint=input_fp,
                started=started,
            )

        prompt_chars = len(system or "") + len(user or "")
        if prompt_chars > service.max_input_chars:
            return self._result_updates(
                status=STATUS_CONTEXT_LIMIT,
                result=None,
                raw_content=None,
                error_code="context_limit",
                error_message=(
                    f"prompt length {prompt_chars} exceeds max_input_chars "
                    f"{service.max_input_chars}; refusing silent truncation"
                ),
                attempts=0,
                latency_ms=0.0,
                evidence_summary=evidence_summary,
                input_fingerprint=input_fp,
                started=started,
            )

        if self.mock:
            mock_result = self._mock_result(sample, evidence_summary)
            raw = __import__("json").dumps(mock_result, ensure_ascii=False)
            return self._result_updates(
                status=STATUS_MOCK,
                result=mock_result,
                raw_content=raw,
                error_code=None,
                error_message=None,
                attempts=1,
                latency_ms=0.0,
                evidence_summary=evidence_summary,
                input_fingerprint=input_fp,
                started=started,
                cache_hit=False,
            )

        if self.client is None:
            return self._result_updates(
                status=STATUS_REQUEST_ERROR,
                result=None,
                raw_content=None,
                error_code="client_missing",
                error_message="live mode requires chat client",
                attempts=0,
                latency_ms=0.0,
                evidence_summary=evidence_summary,
                input_fingerprint=input_fp,
                started=started,
            )

        schema_for_request = (
            self.bundle.output_schema
            if service.structured_output_mode == STRUCTURED_OUTPUT_JSON_SCHEMA
            else None
        )
        try:
            resp = self.client.chat(
                system=system,
                user=user,
                output_schema=schema_for_request,
                schema_name=task.task_id,
            )
        except SemanticRequestError as exc:
            status = STATUS_REQUEST_ERROR
            return self._result_updates(
                status=status,
                result=None,
                raw_content=None,
                error_code=exc.error_code,
                error_message=str(exc),
                attempts=getattr(exc, "attempts", service.retry.max_attempts),
                latency_ms=0.0,
                evidence_summary=evidence_summary,
                input_fingerprint=input_fp,
                started=started,
            )

        raw_content, truncated = _truncate_raw(
            resp.content or "", service.max_raw_content_chars
        )
        try:
            parsed = extract_json_object(resp.content or "")
            result = validate_result(parsed, self.validator)
        except SemanticParseError as exc:
            status = (
                STATUS_SCHEMA_ERROR
                if exc.error_code == "schema_error"
                else STATUS_PARSE_ERROR
            )
            # Optional single identical retry for parse/schema issues.
            if service.retry.parse_retry_once and resp.attempts < service.retry.max_attempts:
                try:
                    resp2 = self.client.chat(
                        system=system,
                        user=user,
                        output_schema=schema_for_request,
                        schema_name=task.task_id,
                    )
                    raw_content, truncated = _truncate_raw(
                        resp2.content or "", service.max_raw_content_chars
                    )
                    parsed = extract_json_object(resp2.content or "")
                    result = validate_result(parsed, self.validator)
                    return self._result_updates(
                        status=STATUS_SUCCESS,
                        result=result,
                        raw_content=raw_content,
                        error_code=None,
                        error_message=None,
                        attempts=resp.attempts + resp2.attempts,
                        latency_ms=resp.latency_ms + resp2.latency_ms,
                        evidence_summary=evidence_summary,
                        input_fingerprint=input_fp,
                        started=started,
                        usage=resp2.usage,
                        raw_truncated=truncated,
                    )
                except Exception as retry_exc:  # noqa: BLE001
                    return self._result_updates(
                        status=status,
                        result=None,
                        raw_content=raw_content,
                        error_code=getattr(retry_exc, "error_code", exc.error_code),
                        error_message=str(retry_exc),
                        attempts=resp.attempts + 1,
                        latency_ms=resp.latency_ms,
                        evidence_summary=evidence_summary,
                        input_fingerprint=input_fp,
                        started=started,
                        usage=resp.usage,
                        raw_truncated=truncated,
                    )
            return self._result_updates(
                status=status,
                result=None,
                raw_content=raw_content,
                error_code=exc.error_code,
                error_message=str(exc),
                attempts=resp.attempts,
                latency_ms=resp.latency_ms,
                evidence_summary=evidence_summary,
                input_fingerprint=input_fp,
                started=started,
                usage=resp.usage,
                raw_truncated=truncated,
            )

        return self._result_updates(
            status=STATUS_SUCCESS,
            result=result,
            raw_content=raw_content,
            error_code=None,
            error_message=None,
            attempts=resp.attempts,
            latency_ms=resp.latency_ms,
            evidence_summary=evidence_summary,
            input_fingerprint=input_fp,
            started=started,
            usage=resp.usage,
            raw_truncated=truncated,
        )

    def _mock_result(
        self, sample: Sample, evidence_summary: dict[str, Any]
    ) -> dict[str, Any]:
        """Deterministic mock shaped loosely; still validated when possible."""
        schema = self.bundle.output_schema
        props = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        required = schema.get("required") if isinstance(schema.get("required"), list) else []
        out: dict[str, Any] = {}
        for key in required:
            spec = props.get(key) if isinstance(props, dict) else None
            typ = (spec or {}).get("type") if isinstance(spec, dict) else None
            if typ == "boolean":
                out[key] = False
            elif typ == "integer":
                out[key] = 0
            elif typ == "number":
                out[key] = 0.0
            elif typ == "array":
                out[key] = []
            elif typ == "object":
                out[key] = {}
            else:
                # Prefer enum first value when present.
                enum = (spec or {}).get("enum") if isinstance(spec, dict) else None
                if isinstance(enum, list) and enum:
                    out[key] = enum[0]
                else:
                    out[key] = f"mock:{sample.id}:{key}"
        # Common optional fields used by example tasks.
        if "conflict" in props and "conflict" not in out:
            out["conflict"] = bool(evidence_summary.get("insufficient_evidence"))
        if "intent" in props and "intent" not in out:
            enum = props["intent"].get("enum") if isinstance(props["intent"], dict) else None
            out["intent"] = enum[0] if isinstance(enum, list) and enum else "unknown"
        if "insufficient_evidence" in props and "insufficient_evidence" not in out:
            out["insufficient_evidence"] = bool(
                evidence_summary.get("insufficient_evidence")
            )
        try:
            return validate_result(out, self.validator)
        except SemanticParseError:
            # Fall back to minimal object; engine marks mock even if schema-tight.
            return out

    def _result_updates(
        self,
        *,
        status: str,
        result: dict[str, Any] | None,
        raw_content: str | None,
        error_code: str | None,
        error_message: str | None,
        attempts: int,
        latency_ms: float,
        evidence_summary: dict[str, Any],
        input_fingerprint: str,
        started: str,
        usage: dict[str, Any] | None = None,
        raw_truncated: bool = False,
        cache_hit: bool = False,
    ) -> dict[str, Any]:
        task = self.bundle.task
        service = self.bundle.service
        record = {
            "task_id": task.task_id,
            "task_version": task.task_version,
            "engine_version": ENGINE_VERSION,
            "template_digest": self.bundle.content_fingerprint,
            "schema_digest": hashlib.sha256(
                __import__("json")
                .dumps(self.bundle.output_schema, sort_keys=True, ensure_ascii=False)
                .encode("utf-8")
            ).hexdigest(),
            "model": service.model,
            "model_revision": service.model_revision,
            "request_params": service.generation.fingerprint(),
            "structured_output_mode": service.structured_output_mode,
            "input_fingerprint": input_fingerprint,
            "result_fingerprint": result_fingerprint(result),
            "status": status,
            "result": result,
            "raw_content": raw_content,
            "raw_truncated": raw_truncated,
            "error_code": error_code,
            "error_message": error_message,
            "attempts": attempts,
            "latency_ms": latency_ms,
            "completed_at": _utc_now(),
            "started_at": started,
            "usage": usage,
            "cache_hit": cache_hit,
            "mock": status == STATUS_MOCK or self.mock,
            "evidence_summary": evidence_summary,
        }
        return {
            "labels": {"semantic_results": {task.task_id: record}},
            "lineage_entry": {
                "operator": "semantic.judge",
                "version": ENGINE_VERSION,
                "params": {
                    "task_id": task.task_id,
                    "task_version": task.task_version,
                    "content_fingerprint": self.bundle.content_fingerprint,
                    "model": service.model,
                    "model_revision": service.model_revision,
                    "mock": self.mock,
                },
                "input_key": None,
                "output_key": f"labels.semantic_results.{task.task_id}",
                "cache_key": input_fingerprint,
            },
            "_semantic_status": status,
            "_semantic_task_id": task.task_id,
            "_semantic_record": record,
        }
