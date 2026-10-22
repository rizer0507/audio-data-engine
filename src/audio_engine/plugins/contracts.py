"""Unified plugin lifecycle contracts (api_version=1).

Pipeline and tag plugins share describe/validate/plan/execute.
Adapters wrap existing engines; they do not replace them.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Protocol, runtime_checkable

API_VERSION = "1"

PluginKind = Literal["pipeline", "tag"]
PluginStatus = Literal["enabled", "disabled", "archived"]
TaskType = Literal["pipeline", "tagging"]
InputKind = Literal["source", "dataset"]
Outcome = Literal["succeeded", "failed", "cancelled"]
EvidenceMode = Literal["existing", "compute_acoustic", "plan", "preview"]

PLUGIN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def validate_plugin_id(value: str, *, field_name: str = "id") -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} must be non-empty")
    if ".." in text or "/" in text or "\\" in text:
        raise ValueError(f"{field_name} must not contain path separators or '..'")
    if not PLUGIN_ID_RE.match(text):
        raise ValueError(
            f"{field_name} {text!r} must match [A-Za-z0-9][A-Za-z0-9_.-]*"
        )
    return text


@dataclass(frozen=True)
class PluginRef:
    kind: PluginKind
    id: str
    version: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", validate_plugin_id(self.id))
        object.__setattr__(self, "version", str(self.version or "").strip() or "1")
        if self.kind not in ("pipeline", "tag"):
            raise ValueError(f"unsupported plugin kind: {self.kind}")

    def as_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "id": self.id, "version": self.version}

    def key(self) -> str:
        return f"{self.kind}:{self.id}@{self.version}"


@dataclass
class PluginCapabilities:
    supports_resume: bool = False
    supports_cancel: bool = False
    supports_tag_selection: bool = False
    supports_isolated_output: bool = False
    supports_task_private_env: bool = False
    purpose: str = "processing"  # processing | tagging | evaluation
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PluginDependency:
    kind: str  # operator | model_role | tag | evidence | config | file
    ref: str
    required: bool = True
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PluginDescriptor:
    kind: PluginKind
    id: str
    version: str
    api_version: str
    name: str
    description: str
    adapter: str
    status: PluginStatus
    runnable: bool
    input_kinds: list[InputKind]
    output_kinds: list[str]
    parameters_schema: dict[str, Any]
    capabilities: PluginCapabilities
    dependencies: list[PluginDependency] = field(default_factory=list)
    definition_digest: str = ""
    implementation_digest: str = ""
    catalog_generation: int = 0
    definition_path: str = ""
    description_updated_at: str | None = None
    not_runnable_reasons: list[str] = field(default_factory=list)
    diagnostics: list[dict[str, Any]] = field(default_factory=list)
    detail: dict[str, Any] = field(default_factory=dict)

    def ref(self) -> PluginRef:
        return PluginRef(kind=self.kind, id=self.id, version=self.version)

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "id": self.id,
            "version": self.version,
            "api_version": self.api_version,
            "name": self.name,
            "description": self.description,
            "description_updated_at": self.description_updated_at,
            "adapter": self.adapter,
            "status": self.status,
            "runnable": self.runnable,
            "not_runnable_reasons": list(self.not_runnable_reasons),
            "input_kinds": list(self.input_kinds),
            "output_kinds": list(self.output_kinds),
            "parameters_schema": self.parameters_schema,
            "capabilities": self.capabilities.as_dict(),
            "dependencies": [d.as_dict() for d in self.dependencies],
            "definition_digest": self.definition_digest,
            "implementation_digest": self.implementation_digest,
            "catalog_generation": self.catalog_generation,
            "definition_path": self.definition_path,
            "diagnostics": list(self.diagnostics),
            "detail": dict(self.detail),
        }


@dataclass
class ValidationIssue:
    code: str
    message: str
    plugin_ref: dict[str, str] | None = None
    field_path: str | None = None
    definition_path: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "plugin_ref": self.plugin_ref,
            "field_path": self.field_path,
            "definition_path": self.definition_path,
            "details": dict(self.details),
        }


@dataclass
class ValidationReport:
    ok: bool
    errors: list[ValidationIssue] = field(default_factory=list)
    warnings: list[ValidationIssue] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "errors": [e.as_dict() for e in self.errors],
            "warnings": [w.as_dict() for w in self.warnings],
        }

    def extend(self, other: ValidationReport) -> None:
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        if other.errors:
            self.ok = False


@dataclass
class InputRef:
    kind: InputKind
    source_id: str | None = None
    dataset_id: str | None = None
    artifact_id: str | None = None
    # Filled by web layer after resolving source_id → registry (not a form field).
    batch: str | None = None
    path: str | None = None

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"kind": self.kind}
        if self.source_id:
            out["source_id"] = self.source_id
        if self.dataset_id:
            out["dataset_id"] = self.dataset_id
        if self.artifact_id:
            out["artifact_id"] = self.artifact_id
        if self.batch:
            out["batch"] = self.batch
        if self.path:
            out["path"] = self.path
        return out

    def engine_source_name(self) -> str | None:
        """Batch used as engine --source-name; never invent from display names."""
        if self.batch:
            return self.batch
        return self.source_id


@dataclass
class ModelBinding:
    route_id: str
    connection_id: str

    def as_dict(self) -> dict[str, str]:
        return {"route_id": self.route_id, "connection_id": self.connection_id}


@dataclass
class ExecutionOptions:
    evidence_mode: EvidenceMode | None = None
    workers: int | None = None
    concurrency: int | None = None
    resume: bool = False
    preview: bool = False
    force: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    ALLOWED_EXTRA = frozenset(
        {
            "shards",
            "parallel_shards",
            "fail_fast",
            "checkpoint_every",
            "batch_size",
            "mock",
        }
    )

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "resume": self.resume,
            "preview": self.preview,
            "force": self.force,
        }
        if self.evidence_mode is not None:
            out["evidence_mode"] = self.evidence_mode
        if self.workers is not None:
            out["workers"] = self.workers
        if self.concurrency is not None:
            out["concurrency"] = self.concurrency
        for key, value in self.extra.items():
            if key in self.ALLOWED_EXTRA:
                out[key] = value
        return out


@dataclass
class PluginRequest:
    """Single-plugin view used by adapters."""

    plugin: PluginRef
    input: InputRef
    parameters: dict[str, Any] = field(default_factory=dict)
    model_bindings: list[ModelBinding] = field(default_factory=list)
    tags: list[PluginRef] = field(default_factory=list)
    execution: ExecutionOptions = field(default_factory=ExecutionOptions)
    catalog_generation: int | None = None
    definition_digest: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "plugin": self.plugin.as_dict(),
            "input": self.input.as_dict(),
            "parameters": dict(self.parameters),
            "model_bindings": [b.as_dict() for b in self.model_bindings],
            "tags": [t.as_dict() for t in self.tags],
            "execution": self.execution.as_dict(),
            "catalog_generation": self.catalog_generation,
            "definition_digest": self.definition_digest,
        }


@dataclass
class TaskRequest:
    """HTTP/plugin shell for pipeline or tagging batch tasks."""

    task_type: TaskType
    input: InputRef
    root_plugin: PluginRef | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    model_bindings: list[ModelBinding] = field(default_factory=list)
    tags: list[PluginRef] = field(default_factory=list)
    execution: ExecutionOptions = field(default_factory=ExecutionOptions)
    catalog_generation: int | None = None
    definition_digest: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_type": self.task_type,
            "root_plugin": self.root_plugin.as_dict() if self.root_plugin else None,
            "input": self.input.as_dict(),
            "parameters": dict(self.parameters),
            "model_bindings": [b.as_dict() for b in self.model_bindings],
            "tags": [t.as_dict() for t in self.tags],
            "execution": self.execution.as_dict(),
            "catalog_generation": self.catalog_generation,
            "definition_digest": self.definition_digest,
        }


@dataclass
class ResolvedModelConnection:
    connection_id: str
    provider: str
    base_url: str
    model: str
    model_revision: str
    api_key_env: str
    timeout_seconds: int = 120
    concurrency: int = 4
    # Secrets stay as env refs only.
    secret_refs: dict[str, str] = field(default_factory=dict)

    def public_dict(self) -> dict[str, Any]:
        return {
            "connection_id": self.connection_id,
            "provider": self.provider,
            "base_url": self.base_url,
            "model": self.model,
            "model_revision": self.model_revision,
            "api_key_env": self.api_key_env,
            "timeout_seconds": self.timeout_seconds,
            "concurrency": self.concurrency,
            "secret_refs": dict(self.secret_refs),
        }


@dataclass
class PlanNode:
    node_id: str
    kind: str
    label: str
    depends_on: list[str] = field(default_factory=list)
    plugin_ref: dict[str, str] | None = None
    model_route: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExecutionPlan:
    plan_id: str
    plan_digest: str
    plugin: PluginRef
    catalog_generation: int
    request: dict[str, Any]
    nodes: list[PlanNode] = field(default_factory=list)
    resolved_models: list[ResolvedModelConnection] = field(default_factory=list)
    selected_tags: list[dict[str, str]] = field(default_factory=list)
    evidence_union: dict[str, Any] = field(default_factory=dict)
    output_path_template: str = ""
    expected_artifacts: list[str] = field(default_factory=list)
    estimated_steps: int | None = None
    estimated_samples: int | None = None
    supports_resume: bool = False
    supports_cancel: bool = False
    missing_conditions: list[str] = field(default_factory=list)
    definition_snapshot: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    # Present for tagging batch plans.
    task_request: dict[str, Any] | None = None
    sub_plans: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "plan_digest": self.plan_digest,
            "plugin": self.plugin.as_dict(),
            "catalog_generation": self.catalog_generation,
            "request": dict(self.request),
            "nodes": [n.as_dict() for n in self.nodes],
            "resolved_models": [m.public_dict() for m in self.resolved_models],
            "selected_tags": list(self.selected_tags),
            "evidence_union": dict(self.evidence_union),
            "output_path_template": self.output_path_template,
            "expected_artifacts": list(self.expected_artifacts),
            "estimated_steps": self.estimated_steps,
            "estimated_samples": self.estimated_samples,
            "supports_resume": self.supports_resume,
            "supports_cancel": self.supports_cancel,
            "missing_conditions": list(self.missing_conditions),
            "definition_snapshot": dict(self.definition_snapshot),
            "warnings": list(self.warnings),
            "task_request": self.task_request,
            "sub_plans": list(self.sub_plans),
        }


@dataclass
class ArtifactRef:
    artifact_kind: str
    path: str
    artifact_id: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PluginResult:
    outcome: Outcome
    artifacts: list[ArtifactRef] = field(default_factory=list)
    run_dir: str | None = None
    counts: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[ValidationIssue] = field(default_factory=list)
    child_results: list[dict[str, Any]] = field(default_factory=list)
    snapshot_digest: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome,
            "artifacts": [a.as_dict() for a in self.artifacts],
            "run_dir": self.run_dir,
            "counts": dict(self.counts),
            "metrics": dict(self.metrics),
            "warnings": list(self.warnings),
            "errors": [e.as_dict() for e in self.errors],
            "child_results": list(self.child_results),
            "snapshot_digest": self.snapshot_digest,
            "details": dict(self.details),
        }


ProgressCallback = Callable[[dict[str, Any]], None]
LogCallback = Callable[[str, dict[str, Any] | None], None]


class CancelToken:
    """Cooperative cancel signal for a single task attempt."""

    def __init__(self) -> None:
        self._cancelled = False
        self._reason: str | None = None

    def cancel(self, reason: str | None = None) -> None:
        self._cancelled = True
        self._reason = reason or self._reason

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    @property
    def reason(self) -> str | None:
        return self._reason

    def raise_if_cancelled(self) -> None:
        if self._cancelled:
            raise CancelledError(self._reason or "cancelled")


class CancelledError(RuntimeError):
    pass


@dataclass
class PluginContext:
    project_root: Path
    job_id: str | None = None
    attempt_id: str | None = None
    work_dir: Path | None = None
    snapshot_dir: Path | None = None
    model_connections: dict[str, ResolvedModelConnection] = field(default_factory=dict)
    secret_resolver: Callable[[str], str | None] | None = None
    cancel_token: CancelToken | None = None
    on_progress: ProgressCallback | None = None
    on_log: LogCallback | None = None
    readonly: bool = False

    def resolve_secret(self, env_name: str) -> str | None:
        if self.secret_resolver:
            return self.secret_resolver(env_name)
        import os

        return os.environ.get(env_name)

    def log(self, message: str, details: dict[str, Any] | None = None) -> None:
        if self.on_log:
            self.on_log(message, details)

    def progress(self, payload: dict[str, Any]) -> None:
        if self.on_progress:
            self.on_progress(payload)

    def check_cancelled(self) -> None:
        if self.cancel_token:
            self.cancel_token.raise_if_cancelled()


@runtime_checkable
class Plugin(Protocol):
    def describe(self) -> PluginDescriptor: ...

    def validate(self, request: PluginRequest, context: PluginContext) -> ValidationReport: ...

    def plan(self, request: PluginRequest, context: PluginContext) -> ExecutionPlan: ...

    def execute(self, plan: ExecutionPlan, context: PluginContext) -> PluginResult: ...


def parse_plugin_ref(data: dict[str, Any] | None, *, default_kind: PluginKind | None = None) -> PluginRef:
    if not isinstance(data, dict):
        raise ValueError("plugin ref must be an object")
    kind = data.get("kind") or default_kind
    if kind not in ("pipeline", "tag"):
        raise ValueError(f"invalid plugin kind: {kind!r}")
    return PluginRef(kind=kind, id=str(data.get("id") or ""), version=str(data.get("version") or "1"))


def parse_input_ref(data: dict[str, Any] | None) -> InputRef:
    if not isinstance(data, dict):
        raise ValueError("input must be an object")
    kind = data.get("kind")
    if kind not in ("source", "dataset"):
        raise ValueError(f"input.kind must be source|dataset, got {kind!r}")
    return InputRef(
        kind=kind,
        source_id=str(data["source_id"]) if data.get("source_id") else None,
        dataset_id=str(data["dataset_id"]) if data.get("dataset_id") else None,
        artifact_id=str(data["artifact_id"]) if data.get("artifact_id") else None,
        batch=str(data["batch"]).strip() if data.get("batch") else None,
        path=str(data["path"]).strip() if data.get("path") else None,
    )


def parse_execution_options(data: dict[str, Any] | None) -> ExecutionOptions:
    raw = dict(data or {})
    unknown = set(raw) - {
        "evidence_mode",
        "workers",
        "concurrency",
        "resume",
        "preview",
        "force",
        *ExecutionOptions.ALLOWED_EXTRA,
    }
    if unknown:
        raise ValueError(f"unknown execution fields: {sorted(unknown)}")
    extra = {k: raw[k] for k in ExecutionOptions.ALLOWED_EXTRA if k in raw}
    mode = raw.get("evidence_mode")
    if mode is not None and mode not in ("existing", "compute_acoustic", "plan", "preview"):
        raise ValueError(f"invalid evidence_mode: {mode!r}")
    return ExecutionOptions(
        evidence_mode=mode,
        workers=raw.get("workers"),
        concurrency=raw.get("concurrency"),
        resume=bool(raw.get("resume", False)),
        preview=bool(raw.get("preview", False)),
        force=bool(raw.get("force", False)),
        extra=extra,
    )


def parse_task_request(data: dict[str, Any]) -> TaskRequest:
    if not isinstance(data, dict):
        raise ValueError("task request must be an object")
    task_type = data.get("task_type")
    if task_type not in ("pipeline", "tagging"):
        raise ValueError(f"task_type must be pipeline|tagging, got {task_type!r}")
    root = data.get("root_plugin")
    root_plugin = parse_plugin_ref(root) if root else None
    tags = [parse_plugin_ref(t, default_kind="tag") for t in (data.get("tags") or [])]
    bindings = []
    for item in data.get("model_bindings") or []:
        if not isinstance(item, dict):
            raise ValueError("model_bindings items must be objects")
        bindings.append(
            ModelBinding(
                route_id=str(item.get("route_id") or ""),
                connection_id=str(item.get("connection_id") or ""),
            )
        )
    return TaskRequest(
        task_type=task_type,
        root_plugin=root_plugin,
        input=parse_input_ref(data.get("input")),
        parameters=dict(data.get("parameters") or {}),
        model_bindings=bindings,
        tags=tags,
        execution=parse_execution_options(data.get("execution")),
        catalog_generation=data.get("catalog_generation"),
        definition_digest=data.get("definition_digest"),
    )


def task_to_plugin_request(task: TaskRequest, plugin: PluginRef) -> PluginRequest:
    return PluginRequest(
        plugin=plugin,
        input=task.input,
        parameters=dict(task.parameters),
        model_bindings=list(task.model_bindings),
        tags=list(task.tags),
        execution=task.execution,
        catalog_generation=task.catalog_generation,
        definition_digest=task.definition_digest,
    )
