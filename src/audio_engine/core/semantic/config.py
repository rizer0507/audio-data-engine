"""Load and freeze semantic task / service configuration."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from audio_engine.core.semantic.types import (
    STRUCTURED_OUTPUT_JSON_SCHEMA,
    STRUCTURED_OUTPUT_PROMPT_ONLY,
)


def _read_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON Schema root must be an object: {path}")
    return data


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(x) for x in value]


@dataclass(frozen=True)
class GenerationParams:
    max_tokens: int = 1024
    temperature: float = 0.0
    top_p: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def request_fields(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if self.top_p is not None:
            out["top_p"] = self.top_p
        out.update(self.extra)
        return out

    def fingerprint(self) -> dict[str, Any]:
        return {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "extra": self.extra,
        }


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3  # includes first try
    connect_timeout_s: float = 10.0
    read_timeout_s: float = 120.0
    request_timeout_s: float = 180.0
    backoff_base_s: float = 0.5
    backoff_max_s: float = 30.0
    parse_retry_once: bool = False


@dataclass(frozen=True)
class ServiceConfig:
    api_base: str
    model: str
    model_revision: str
    api_key_env: str = "SEMANTIC_JUDGE_API_KEY"
    generation: GenerationParams = field(default_factory=GenerationParams)
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    max_concurrency_total: int = 64
    rate_limit_rps: float | None = None
    structured_output_mode: str = STRUCTURED_OUTPUT_JSON_SCHEMA
    enable_thinking: bool | None = None
    cache_namespace: str = "default"
    max_input_chars: int = 120_000
    max_raw_content_chars: int = 32_000
    probe_on_start: bool = True

    def resolve_api_key(self) -> str | None:
        if not self.api_key_env:
            return None
        value = os.environ.get(self.api_key_env)
        if value is None or not str(value).strip():
            return None
        return str(value).strip()


@dataclass(frozen=True)
class EvidenceConfig:
    model_families: dict[str, list[str]]
    required_families: list[str] = field(default_factory=list)
    min_successful_runs: int = 1
    on_insufficient_evidence: str = "judge"  # judge | skip | fail
    sample_context_fields: list[str] = field(
        default_factory=lambda: ["id", "duration", "source_path"]
    )


@dataclass(frozen=True)
class TaskConfig:
    task_id: str
    task_version: str
    service_config_path: Path
    system_prompt_path: Path | None
    user_prompt_path: Path
    output_schema_path: Path
    evidence: EvidenceConfig
    parameters: dict[str, Any] = field(default_factory=dict)
    source_path: Path | None = None


@dataclass(frozen=True)
class FrozenSemanticBundle:
    """Immutable runtime bundle: configs + file contents + digests."""

    task: TaskConfig
    service: ServiceConfig
    output_schema: dict[str, Any]
    system_template: str | None
    user_template: str
    template_root: Path
    included_templates: dict[str, str]
    content_fingerprint: str


def load_service_config(path: str | Path) -> ServiceConfig:
    path = Path(path)
    raw = _read_yaml(path)
    gen_raw = raw.get("generation") or {}
    if not isinstance(gen_raw, dict):
        raise ValueError(f"{path}: generation must be a mapping")
    known_gen = {"max_tokens", "temperature", "top_p"}
    extra = {k: v for k, v in gen_raw.items() if k not in known_gen}
    generation = GenerationParams(
        max_tokens=int(gen_raw.get("max_tokens", 1024)),
        temperature=float(gen_raw.get("temperature", 0.0)),
        top_p=(float(gen_raw["top_p"]) if gen_raw.get("top_p") is not None else None),
        extra=extra,
    )
    retry_raw = raw.get("retry") or {}
    if not isinstance(retry_raw, dict):
        raise ValueError(f"{path}: retry must be a mapping")
    retry = RetryPolicy(
        max_attempts=max(1, int(retry_raw.get("max_attempts", 3))),
        connect_timeout_s=float(retry_raw.get("connect_timeout_s", 10.0)),
        read_timeout_s=float(retry_raw.get("read_timeout_s", 120.0)),
        request_timeout_s=float(retry_raw.get("request_timeout_s", 180.0)),
        backoff_base_s=float(retry_raw.get("backoff_base_s", 0.5)),
        backoff_max_s=float(retry_raw.get("backoff_max_s", 30.0)),
        parse_retry_once=bool(retry_raw.get("parse_retry_once", False)),
    )
    mode = str(raw.get("structured_output_mode", STRUCTURED_OUTPUT_JSON_SCHEMA)).strip()
    if mode not in {STRUCTURED_OUTPUT_JSON_SCHEMA, STRUCTURED_OUTPUT_PROMPT_ONLY}:
        raise ValueError(
            f"{path}: structured_output_mode must be "
            f"{STRUCTURED_OUTPUT_JSON_SCHEMA!r} or {STRUCTURED_OUTPUT_PROMPT_ONLY!r}"
        )
    api_base = str(
        os.environ.get("SEMANTIC_JUDGE_API_BASE")
        or os.path.expandvars(str(raw.get("api_base") or ""))
        or ""
    ).strip()
    model = str(
        os.environ.get("SEMANTIC_JUDGE_MODEL")
        or os.path.expandvars(str(raw.get("model") or ""))
        or ""
    ).strip()
    if not api_base:
        raise ValueError(f"{path}: api_base is required (or SEMANTIC_JUDGE_API_BASE)")
    if not model:
        raise ValueError(f"{path}: model (served-model-name) is required")
    revision = str(
        os.environ.get("SEMANTIC_JUDGE_MODEL_REVISION")
        or raw.get("model_revision")
        or ""
    ).strip()
    if not revision:
        raise ValueError(
            f"{path}: model_revision is required (explicit weight/cache identity)"
        )
    rate = raw.get("rate_limit_rps")
    return ServiceConfig(
        api_base=api_base,
        model=model,
        model_revision=revision,
        api_key_env=str(raw.get("api_key_env") or "SEMANTIC_JUDGE_API_KEY"),
        generation=generation,
        retry=retry,
        max_concurrency_total=int(raw.get("max_concurrency_total", 64)),
        rate_limit_rps=(float(rate) if rate is not None else None),
        structured_output_mode=mode,
        enable_thinking=(
            bool(raw["enable_thinking"]) if "enable_thinking" in raw else None
        ),
        cache_namespace=str(raw.get("cache_namespace") or "default"),
        max_input_chars=int(raw.get("max_input_chars", 120_000)),
        max_raw_content_chars=int(raw.get("max_raw_content_chars", 32_000)),
        probe_on_start=bool(raw.get("probe_on_start", True)),
    )


def load_task_config(path: str | Path) -> TaskConfig:
    path = Path(path)
    raw = _read_yaml(path)
    task_id = str(raw.get("task_id") or "").strip()
    if not task_id:
        raise ValueError(f"{path}: task_id is required")
    task_version = str(raw.get("task_version") or "1").strip()
    service_path = Path(str(raw.get("service_config") or "")).expanduser()
    if not str(service_path):
        raise ValueError(f"{path}: service_config is required")
    prompts = raw.get("prompts") or {}
    if not isinstance(prompts, dict) or not prompts.get("user"):
        raise ValueError(f"{path}: prompts.user is required")
    user_path = Path(str(prompts["user"])).expanduser()
    system_path = (
        Path(str(prompts["system"])).expanduser() if prompts.get("system") else None
    )
    schema_path = Path(str(raw.get("output_schema") or "")).expanduser()
    if not str(schema_path):
        raise ValueError(f"{path}: output_schema is required")

    evidence_raw = raw.get("evidence") or {}
    if not isinstance(evidence_raw, dict):
        raise ValueError(f"{path}: evidence must be a mapping")
    families_raw = evidence_raw.get("model_families") or {}
    if not isinstance(families_raw, dict) or not families_raw:
        raise ValueError(f"{path}: evidence.model_families is required")
    model_families: dict[str, list[str]] = {}
    for family, runs in families_raw.items():
        keys = _as_list(runs)
        if not keys:
            raise ValueError(f"{path}: family {family!r} has empty run list")
        model_families[str(family)] = keys

    on_insufficient = str(
        evidence_raw.get("on_insufficient_evidence") or "judge"
    ).strip()
    if on_insufficient not in {"judge", "skip", "fail"}:
        raise ValueError(
            f"{path}: on_insufficient_evidence must be judge|skip|fail"
        )

    sample_fields = _as_list(
        evidence_raw.get("sample_context_fields")
        or ["id", "duration", "source_path"]
    )
    if "id" not in sample_fields:
        sample_fields = ["id", *sample_fields]

    evidence = EvidenceConfig(
        model_families=model_families,
        required_families=_as_list(evidence_raw.get("required_families")),
        min_successful_runs=int(evidence_raw.get("min_successful_runs", 1)),
        on_insufficient_evidence=on_insufficient,
        sample_context_fields=sample_fields,
    )
    unknown_required = [f for f in evidence.required_families if f not in model_families]
    if unknown_required:
        raise ValueError(
            f"{path}: required_families not in model_families: {unknown_required}"
        )

    parameters = raw.get("parameters") or {}
    if not isinstance(parameters, dict):
        raise ValueError(f"{path}: parameters must be a mapping")

    return TaskConfig(
        task_id=task_id,
        task_version=task_version,
        service_config_path=service_path,
        system_prompt_path=system_path,
        user_prompt_path=user_path,
        output_schema_path=schema_path,
        evidence=evidence,
        parameters=dict(parameters),
        source_path=path,
    )


def _collect_includes(template_text: str, root: Path) -> dict[str, str]:
    """Best-effort collect {% include %} / {% import %} targets under root."""
    import re

    collected: dict[str, str] = {}
    pattern = re.compile(
        r"""\{%-?\s*(?:include|import)\s+['"]([^'"]+)['"]""",
        re.IGNORECASE,
    )
    pending = list(pattern.findall(template_text))
    seen: set[str] = set()
    while pending:
        rel = pending.pop()
        if rel in seen:
            continue
        seen.add(rel)
        target = (root / rel).resolve()
        try:
            target.relative_to(root.resolve())
        except ValueError as exc:
            raise ValueError(
                f"Template include escapes template root: {rel}"
            ) from exc
        if not target.is_file():
            raise FileNotFoundError(f"Included template not found: {target}")
        text = target.read_text(encoding="utf-8")
        collected[rel.replace("\\", "/")] = text
        pending.extend(pattern.findall(text))
    return collected


def freeze_bundle(task_config_path: str | Path) -> FrozenSemanticBundle:
    """Load task + service + templates + schema and freeze content digests."""
    from audio_engine.core.checkpoint import digest_payload

    task = load_task_config(task_config_path)
    for path in (
        task.service_config_path,
        task.user_prompt_path,
        task.output_schema_path,
        task.system_prompt_path,
    ):
        if path is None:
            continue
        if not path.is_file():
            raise FileNotFoundError(f"Semantic config file missing: {path}")

    service = load_service_config(task.service_config_path)
    output_schema = _read_json(task.output_schema_path)
    user_template = task.user_prompt_path.read_text(encoding="utf-8")
    system_template = (
        task.system_prompt_path.read_text(encoding="utf-8")
        if task.system_prompt_path
        else None
    )
    template_root = task.user_prompt_path.parent.resolve()
    included = _collect_includes(user_template, template_root)
    if system_template:
        included.update(_collect_includes(system_template, template_root))

    content_fingerprint = digest_payload(
        {
            "task_id": task.task_id,
            "task_version": task.task_version,
            "parameters": task.parameters,
            "evidence": {
                "model_families": task.evidence.model_families,
                "required_families": task.evidence.required_families,
                "min_successful_runs": task.evidence.min_successful_runs,
                "on_insufficient_evidence": task.evidence.on_insufficient_evidence,
                "sample_context_fields": task.evidence.sample_context_fields,
            },
            "service": {
                "api_base": service.api_base,
                "model": service.model,
                "model_revision": service.model_revision,
                "generation": service.generation.fingerprint(),
                "structured_output_mode": service.structured_output_mode,
                "enable_thinking": service.enable_thinking,
                "cache_namespace": service.cache_namespace,
                "max_input_chars": service.max_input_chars,
            },
            "user_template": user_template,
            "system_template": system_template,
            "included_templates": included,
            "output_schema": output_schema,
        }
    )
    return FrozenSemanticBundle(
        task=task,
        service=service,
        output_schema=output_schema,
        system_template=system_template,
        user_template=user_template,
        template_root=template_root,
        included_templates=included,
        content_fingerprint=content_fingerprint,
    )
