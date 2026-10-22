"""Built-in plugin adapters wrapping existing engine entry points."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml

from audio_engine.core.checkpoint import digest_payload
from audio_engine.core.production.config import is_production_yaml, load_production_config
from audio_engine.core.production.orchestrator import ProductionOrchestrator
from audio_engine.core.tagging.config import load_tagging_config
from audio_engine.core.tagging.library import (
    TagLibrary,
    TagLibraryError,
    build_definition_snapshot,
    load_tag_library,
    select_from_library,
)
from audio_engine.core.tagging.service import TaggingRunRequest, TaggingService
from audio_engine.plugins.contracts import (
    API_VERSION,
    ArtifactRef,
    CancelledError,
    ExecutionPlan,
    InputRef,
    PlanNode,
    PluginCapabilities,
    PluginContext,
    PluginDependency,
    PluginDescriptor,
    PluginRef,
    PluginRequest,
    PluginResult,
    ResolvedModelConnection,
    ValidationIssue,
    ValidationReport,
)
from audio_engine.plugins.schema_util import EMPTY_OBJECT_SCHEMA, validate_against_schema
from audio_engine.plugins.snapshot import (
    digest_implementation_files,
    freeze_yaml_tree,
    sha256_file,
    sha256_obj,
    write_snapshot_dir,
)

AdapterFactory = Callable[..., Any]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _impl_files(*relpaths: str, project_root: Path) -> list[Path]:
    return [(project_root / p).resolve() for p in relpaths]


COMMON_PIPELINE_IMPL = (
    "src/audio_engine/core/pipeline.py",
    "src/audio_engine/cli/main.py",
    "src/audio_engine/plugins/adapters.py",
)
PRODUCTION_IMPL = COMMON_PIPELINE_IMPL + (
    "src/audio_engine/core/production/orchestrator.py",
    "src/audio_engine/core/production/config.py",
)
TAG_IMPL = (
    "src/audio_engine/core/tagging/library.py",
    "src/audio_engine/core/tagging/service.py",
    "src/audio_engine/core/tagging/engine.py",
    "src/audio_engine/plugins/adapters.py",
)


def _safe_status(raw: str | None) -> str:
    text = str(raw or "enabled").strip().lower()
    if text in {"enabled", "active", "on"}:
        return "enabled"
    if text in {"disabled", "off", "inactive"}:
        return "disabled"
    if text in {"archived", "archive"}:
        return "archived"
    return "enabled"


def _engine_source_name(request_or_plan_input: dict[str, Any], parameters: dict[str, Any] | None = None) -> str | None:
    """Prefer resolved input.batch; fall back to parameters then legacy source_id."""

    inp = request_or_plan_input or {}
    batch = str(inp.get("batch") or "").strip()
    if batch:
        return batch
    params = parameters if parameters is not None else {}
    for key in ("batch", "source_name"):
        value = str(params.get(key) or "").strip()
        if value:
            return value
    return str(inp.get("source_id") or "").strip() or None


def _engine_source_dir(request_or_plan_input: dict[str, Any], parameters: dict[str, Any] | None = None) -> str | None:
    params = parameters if parameters is not None else {}
    if params.get("source_dir"):
        return str(params["source_dir"])
    path = str((request_or_plan_input or {}).get("path") or "").strip()
    return path or None


def _plan_ids(request: PluginRequest, generation: int) -> tuple[str, str]:
    plan_id = f"plan_{uuid.uuid4().hex[:16]}"
    digest = digest_payload(
        {
            "request": request.as_dict(),
            "catalog_generation": generation,
        }
    )
    return plan_id, digest


def _resolve_models(
    request: PluginRequest,
    context: PluginContext,
    *,
    allowed_routes: set[str] | None = None,
) -> tuple[list[ResolvedModelConnection], ValidationReport]:
    report = ValidationReport(ok=True)
    resolved: list[ResolvedModelConnection] = []
    seen_routes: set[str] = set()
    for binding in request.model_bindings:
        if not binding.route_id or not binding.connection_id:
            report.ok = False
            report.errors.append(
                ValidationIssue(
                    code="MODEL_BINDING_INCOMPLETE",
                    message="model_bindings require route_id and connection_id",
                    field_path="model_bindings",
                    plugin_ref=request.plugin.as_dict(),
                )
            )
            continue
        if binding.route_id in seen_routes:
            report.ok = False
            report.errors.append(
                ValidationIssue(
                    code="MODEL_ROUTE_DUPLICATE",
                    message=f"duplicate route_id {binding.route_id}",
                    field_path="model_bindings",
                    plugin_ref=request.plugin.as_dict(),
                )
            )
            continue
        seen_routes.add(binding.route_id)
        if allowed_routes is not None and binding.route_id not in allowed_routes:
            report.ok = False
            report.errors.append(
                ValidationIssue(
                    code="MODEL_ROUTE_UNKNOWN",
                    message=f"route_id {binding.route_id} is not declared by this plugin",
                    field_path="model_bindings",
                    plugin_ref=request.plugin.as_dict(),
                    details={"allowed_routes": sorted(allowed_routes)},
                )
            )
            continue
        conn = context.model_connections.get(binding.connection_id)
        if conn is None:
            report.warnings.append(
                ValidationIssue(
                    code="MODEL_CONNECTION_UNRESOLVED",
                    message=(
                        f"connection_id {binding.connection_id} not in context; "
                        "plan continues with binding only"
                    ),
                    field_path="model_bindings",
                    plugin_ref=request.plugin.as_dict(),
                )
            )
            resolved.append(
                ResolvedModelConnection(
                    connection_id=binding.connection_id,
                    provider="unknown",
                    base_url="",
                    model="",
                    model_revision="",
                    api_key_env="",
                )
            )
            continue
        resolved.append(conn)
    return resolved, report


def _task_env_from_models(
    models: list[ResolvedModelConnection],
    *,
    route_family: dict[str, str] | None = None,
    context: PluginContext,
) -> dict[str, str]:
    """Build a task-private env mapping. Does not mutate os.environ."""

    env = dict(os.environ)
    # Clear conflicting legacy multi-base vars so frozen bindings win.
    for key in (
        "GLM_ASR_API_BASE",
        "GLM_ASR_API_BASES",
        "GLM_ASR_MODEL",
        "QWEN_ASR_API_BASE",
        "QWEN_ASR_API_BASES",
        "QWEN_ASR_MODEL",
    ):
        env.pop(key, None)

    # Group by provider for single-route convenience; multi-route uses per-node env.
    by_provider: dict[str, list[ResolvedModelConnection]] = {}
    for m in models:
        by_provider.setdefault(m.provider, []).append(m)

    for provider, items in by_provider.items():
        if provider == "glm":
            bases = [m.base_url for m in items if m.base_url]
            if bases:
                env["GLM_ASR_API_BASES"] = ",".join(bases)
                env["GLM_ASR_API_BASE"] = bases[0]
            if items and items[0].model:
                env["GLM_ASR_MODEL"] = items[0].model
            if items and items[0].api_key_env:
                secret = context.resolve_secret(items[0].api_key_env)
                if secret:
                    env["GLM_ASR_API_KEY"] = secret
        elif provider == "qwen":
            bases = [m.base_url for m in items if m.base_url]
            if bases:
                env["QWEN_ASR_API_BASES"] = ",".join(bases)
                env["QWEN_ASR_API_BASE"] = bases[0]
            if items and items[0].model:
                env["QWEN_ASR_MODEL"] = items[0].model
            if items and items[0].api_key_env:
                secret = context.resolve_secret(items[0].api_key_env)
                if secret:
                    env["QWEN_ASR_API_KEY"] = secret
    _ = route_family  # reserved for per-route production env injection
    return env


def _read_pipeline_meta(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"pipeline YAML must be a mapping: {path}")
    return data


# ---------------------------------------------------------------------------
# YAML pipeline adapter
# ---------------------------------------------------------------------------


@dataclass
class YamlPipelinePlugin:
    registration: dict[str, Any]
    project_root: Path
    catalog_generation: int = 0
    definition_path: Path = field(init=False)
    _meta: dict[str, Any] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        rel = str(self.registration.get("definition") or "")
        candidate = Path(rel)
        self.definition_path = (
            candidate.resolve()
            if candidate.is_absolute()
            else (self.project_root / candidate).resolve()
        )
        if self.definition_path.is_file():
            self._meta = _read_pipeline_meta(self.definition_path)

    def describe(self) -> PluginDescriptor:
        schema = self.registration.get("parameters_schema") or EMPTY_OBJECT_SCHEMA
        status = _safe_status(self.registration.get("status"))
        purpose = str(self.registration.get("purpose") or "processing")
        input_kinds = list(self.registration.get("input_kinds") or ["dataset"])
        runnable = status == "enabled" and self.definition_path.is_file()
        reasons: list[str] = []
        if status != "enabled":
            reasons.append(f"status={status}")
        if not self.definition_path.is_file():
            reasons.append(f"definition missing: {self.definition_path}")

        steps = []
        for step in self._meta.get("pipeline") or []:
            if isinstance(step, dict):
                steps.append(
                    {
                        "name": step.get("name"),
                        "operator": step.get("operator"),
                    }
                )

        caps = PluginCapabilities(
            supports_resume=True,
            supports_cancel=False,
            supports_tag_selection=bool(self.registration.get("supports_tag_selection", False)),
            supports_isolated_output=True,
            supports_task_private_env=True,
            purpose=purpose,
            notes=[
                "cancel: YAML PipelineRunner has no cooperative cancel; capability=false",
                "resume: uses PipelineRunner --resume / checkpoints when work_dir provided",
                "isolated_output: adapter writes manifests under task work_dir when provided",
            ],
        )
        deps: list[PluginDependency] = []
        for step in steps:
            if step.get("operator"):
                deps.append(
                    PluginDependency(kind="operator", ref=str(step["operator"]), required=True)
                )

        digest = sha256_file(self.definition_path) if self.definition_path.is_file() else ""
        impl = digest_implementation_files(
            _impl_files(*COMMON_PIPELINE_IMPL, project_root=self.project_root)
        )
        updated_at = self.registration.get("description_updated_at")
        return PluginDescriptor(
            kind="pipeline",
            id=str(self.registration["id"]),
            version=str(self.registration.get("version") or "1"),
            api_version=API_VERSION,
            name=str(self.registration.get("name") or self.registration["id"]),
            description=str(self.registration.get("description") or self._meta.get("name") or ""),
            adapter="yaml_pipeline",
            status=status,  # type: ignore[arg-type]
            runnable=runnable,
            not_runnable_reasons=reasons,
            input_kinds=input_kinds,  # type: ignore[arg-type]
            output_kinds=list(self.registration.get("output_kinds") or ["manifest", "report"]),
            parameters_schema=schema,
            capabilities=caps,
            dependencies=deps,
            definition_digest=digest,
            implementation_digest=impl,
            catalog_generation=self.catalog_generation,
            definition_path=str(self.definition_path.relative_to(self.project_root).as_posix())
            if self.definition_path.is_file()
            else str(self.registration.get("definition") or ""),
            description_updated_at=str(updated_at).strip() if updated_at else None,
            detail={"steps": steps, "has_sharding": "sharding" in self._meta},
        )

    def validate(self, request: PluginRequest, context: PluginContext) -> ValidationReport:
        desc = self.describe()
        report = ValidationReport(ok=True)
        if request.plugin.id != desc.id or request.plugin.version != desc.version:
            report.ok = False
            report.errors.append(
                ValidationIssue(
                    code="PLUGIN_REF_MISMATCH",
                    message="request plugin ref does not match adapter instance",
                    plugin_ref=request.plugin.as_dict(),
                )
            )
        if desc.status != "enabled":
            report.ok = False
            report.errors.append(
                ValidationIssue(
                    code="PLUGIN_DISABLED",
                    message=f"plugin status is {desc.status}",
                    plugin_ref=request.plugin.as_dict(),
                )
            )
        if request.input.kind not in desc.input_kinds:
            report.ok = False
            report.errors.append(
                ValidationIssue(
                    code="INPUT_KIND_UNSUPPORTED",
                    message=f"input kind {request.input.kind} not in {desc.input_kinds}",
                    field_path="input.kind",
                    plugin_ref=request.plugin.as_dict(),
                )
            )
        if request.input.kind == "source" and not request.input.source_id:
            report.ok = False
            report.errors.append(
                ValidationIssue(
                    code="INPUT_SOURCE_REQUIRED",
                    message="source input requires source_id",
                    field_path="input.source_id",
                    plugin_ref=request.plugin.as_dict(),
                )
            )
        if request.input.kind == "dataset" and not (
            request.input.dataset_id or request.input.artifact_id
        ):
            report.ok = False
            report.errors.append(
                ValidationIssue(
                    code="INPUT_DATASET_REQUIRED",
                    message="dataset input requires dataset_id or artifact_id",
                    field_path="input.dataset_id",
                    plugin_ref=request.plugin.as_dict(),
                )
            )
        if request.tags and not desc.capabilities.supports_tag_selection:
            report.ok = False
            report.errors.append(
                ValidationIssue(
                    code="TAG_SELECTION_UNSUPPORTED",
                    message="this pipeline does not support tag selection",
                    field_path="tags",
                    plugin_ref=request.plugin.as_dict(),
                )
            )
        report.extend(
            validate_against_schema(
                request.parameters,
                desc.parameters_schema,
                plugin_ref=request.plugin.as_dict(),
            )
        )
        return report

    def plan(self, request: PluginRequest, context: PluginContext) -> ExecutionPlan:
        context.check_cancelled()
        report = self.validate(request, context)
        if not report.ok:
            raise ValueError(
                "validate failed: " + "; ".join(e.message for e in report.errors)
            )
        desc = self.describe()
        models, model_report = _resolve_models(request, context)
        plan_id, plan_digest = _plan_ids(request, self.catalog_generation)
        snapshot = freeze_yaml_tree(self.definition_path, project_root=self.project_root)
        nodes = [
            PlanNode(
                node_id=str(step.get("name") or f"step_{i}"),
                kind="operator",
                label=str(step.get("operator") or step.get("name")),
                depends_on=[] if i == 0 else [str((desc.detail["steps"][i - 1]).get("name"))],
                details={"operator": step.get("operator")},
            )
            for i, step in enumerate(desc.detail.get("steps") or [])
        ]
        work = context.work_dir or (self.project_root / "runs" / "plugins" / plan_id)
        warnings = [w.message for w in model_report.warnings]
        warnings.append("plan is read-only; no ASR/DNSMOS inference executed")
        return ExecutionPlan(
            plan_id=plan_id,
            plan_digest=plan_digest,
            plugin=request.plugin,
            catalog_generation=self.catalog_generation,
            request=request.as_dict(),
            nodes=nodes,
            resolved_models=models,
            selected_tags=[t.as_dict() for t in request.tags],
            output_path_template=str(work / "outputs" / "{artifact}"),
            expected_artifacts=["manifest"],
            estimated_steps=len(nodes) or None,
            estimated_samples=None,
            supports_resume=desc.capabilities.supports_resume,
            supports_cancel=desc.capabilities.supports_cancel,
            missing_conditions=[],
            definition_snapshot={
                "pipeline": snapshot,
                "implementation_digest": desc.implementation_digest,
            },
            warnings=warnings,
        )

    def execute(self, plan: ExecutionPlan, context: PluginContext) -> PluginResult:
        context.check_cancelled()
        work_dir = context.work_dir or (self.project_root / "runs" / "plugins" / plan.plan_id)
        work_dir.mkdir(parents=True, exist_ok=True)
        snap_dir = context.snapshot_dir or (work_dir / "snapshot")
        write_snapshot_dir(plan.definition_snapshot, snap_dir)

        from audio_engine.plugins.contracts import ModelBinding, parse_execution_options

        inp = plan.request.get("input") or {}
        request = PluginRequest(
            plugin=plan.plugin,
            input=InputRef(
                kind=inp["kind"],
                source_id=inp.get("source_id"),
                dataset_id=inp.get("dataset_id"),
                artifact_id=inp.get("artifact_id"),
                batch=inp.get("batch"),
                path=inp.get("path"),
            ),
            parameters=dict(plan.request.get("parameters") or {}),
            model_bindings=[
                ModelBinding(**b) for b in plan.request.get("model_bindings") or []
            ],
            tags=[PluginRef(**t) for t in plan.request.get("tags") or []],
            execution=parse_execution_options(plan.request.get("execution")),
        )

        out_manifest = work_dir / "outputs" / "output_manifest.parquet"
        out_manifest.parent.mkdir(parents=True, exist_ok=True)
        env = _task_env_from_models(plan.resolved_models, context=context)
        cmd = [
            sys.executable,
            "-m",
            "audio_engine.cli.main",
            "pipeline",
            "run",
            str(self.definition_path),
            "--output-manifest",
            str(out_manifest),
        ]
        if request.input.kind == "source":
            engine_name = request.input.engine_source_name()
            if engine_name:
                cmd.extend(["--source-name", engine_name])
            source_dir = _engine_source_dir(
                request.input.as_dict(), request.parameters
            )
            if source_dir and "source_dir" in (request.parameters or {}):
                # Only pass when explicitly in parameters (schema-declared).
                pass
        if request.input.dataset_id:
            # dataset_id may be a path relative to project or absolute parquet
            manifest = Path(request.input.dataset_id)
            if not manifest.is_absolute():
                manifest = self.project_root / manifest
            cmd.extend(["--input-manifest", str(manifest)])
        if request.execution.resume and context.work_dir:
            cmd.append("--resume")
        if request.execution.force:
            cmd.append("--force")
        if request.execution.extra.get("mock"):
            cmd.append("--mock")

        context.log(f"yaml_pipeline execute: {' '.join(cmd)}")
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(self.project_root),
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
        except CancelledError:
            return PluginResult(outcome="cancelled", run_dir=str(work_dir))

        log_path = work_dir / "run.log"
        log_path.write_text(
            (proc.stdout or "") + "\n" + (proc.stderr or ""),
            encoding="utf-8",
        )
        if proc.returncode != 0:
            return PluginResult(
                outcome="failed",
                run_dir=str(work_dir),
                errors=[
                    ValidationIssue(
                        code="PIPELINE_EXIT_NONZERO",
                        message=f"pipeline exited with code {proc.returncode}",
                        details={"stderr_tail": (proc.stderr or "")[-2000:]},
                    )
                ],
                snapshot_digest=plan.plan_digest,
            )
        artifacts = []
        if out_manifest.exists():
            artifacts.append(
                ArtifactRef(artifact_kind="manifest", path=str(out_manifest))
            )
        return PluginResult(
            outcome="succeeded",
            artifacts=artifacts,
            run_dir=str(work_dir),
            snapshot_digest=plan.plan_digest,
            details={"command": cmd},
        )


# ---------------------------------------------------------------------------
# Production DAG adapter
# ---------------------------------------------------------------------------


@dataclass
class ProductionPipelinePlugin:
    registration: dict[str, Any]
    project_root: Path
    catalog_generation: int = 0
    definition_path: Path = field(init=False)
    _config: Any = field(default=None, init=False)

    def __post_init__(self) -> None:
        rel = str(self.registration.get("definition") or "")
        candidate = Path(rel)
        self.definition_path = (
            candidate.resolve()
            if candidate.is_absolute()
            else (self.project_root / candidate).resolve()
        )
        if self.definition_path.is_file() and is_production_yaml(self.definition_path):
            self._config = load_production_config(self.definition_path)

    def describe(self) -> PluginDescriptor:
        schema = self.registration.get("parameters_schema") or {
            "type": "object",
            "additionalProperties": False,
            "title": "生产主链参数",
            "properties": {
                "source_dir": {
                    "type": "string",
                    "title": "源目录覆盖",
                    "description": "可选；默认使用数据源登记路径",
                },
                "asr_route_ids": {
                    "type": "array",
                    "title": "启用的 ASR 路次",
                    "description": "空表示使用定义中的全部路次；不得隐式补齐未选路次",
                    "items": {"type": "string"},
                },
            },
        }
        status = _safe_status(self.registration.get("status"))
        runnable = status == "enabled" and self._config is not None
        reasons: list[str] = []
        if status != "enabled":
            reasons.append(f"status={status}")
        if self._config is None:
            reasons.append("production config not loaded")

        route_ids = []
        dag_nodes = []
        if self._config is not None:
            route_ids = [r.route_id for r in self._config.asr_routes]
            dag_nodes = [
                {"id": "ingest", "pipeline": self._config.ingest.pipeline},
                {"id": "vad", "pipeline": self._config.vad.pipeline, "depends_on": ["ingest"]},
                *[
                    {
                        "id": f"asr_{r.route_id}",
                        "pipeline": r.pipeline,
                        "family": r.family,
                        "depends_on": ["vad"],
                    }
                    for r in self._config.asr_routes
                ],
                {
                    "id": "dnsmos",
                    "pipeline": self._config.dnsmos.pipeline,
                    "depends_on": ["vad"],
                },
                {
                    "id": "evidence_join",
                    "pipeline": self._config.evidence_join.pipeline,
                    "depends_on": [f"asr_{r.route_id}" for r in self._config.asr_routes]
                    + ["dnsmos"],
                },
                {
                    "id": "tagging",
                    "pipeline": self._config.tagging.pipeline,
                    "depends_on": ["evidence_join"],
                },
            ]

        # Accurate capability declaration based on audited code:
        # ProductionOrchestrator supports resume; no cancel API; dataset outputs
        # still use shared stage1 paths keyed by source_name. Adapter provides
        # task-private runs_dir and env, but cannot yet fully isolate published
        # datasets without engine changes.
        caps = PluginCapabilities(
            supports_resume=True,
            supports_cancel=False,
            supports_tag_selection=bool(self.registration.get("supports_tag_selection", True)),
            supports_isolated_output=False,
            supports_task_private_env=True,
            purpose=str(self.registration.get("purpose") or "processing"),
            notes=[
                "cancel: ProductionOrchestrator has no cancel/interrupt API; capability=false",
                "resume: supported via resume run root / succeeded node skip",
                "isolated_output: run logs under task runs_dir; published datasets still "
                "use shared datasets/stage1/<source_name> paths — not fully task-private",
                "task_private_env: child CLI subprocesses receive task-scoped env",
            ],
        )
        deps = [
            PluginDependency(kind="model_role", ref=rid, required=False) for rid in route_ids
        ]
        deps.append(PluginDependency(kind="config", ref=str(self.registration.get("definition")), required=True))
        digest = sha256_file(self.definition_path) if self.definition_path.is_file() else ""
        impl = digest_implementation_files(
            _impl_files(*PRODUCTION_IMPL, project_root=self.project_root)
        )
        updated_at = self.registration.get("description_updated_at")
        return PluginDescriptor(
            kind="pipeline",
            id=str(self.registration["id"]),
            version=str(self.registration.get("version") or "1"),
            api_version=API_VERSION,
            name=str(self.registration.get("name") or self.registration["id"]),
            description=str(self.registration.get("description") or ""),
            adapter="production_pipeline",
            status=status,  # type: ignore[arg-type]
            runnable=runnable,
            not_runnable_reasons=reasons,
            input_kinds=list(self.registration.get("input_kinds") or ["source"]),  # type: ignore[arg-type]
            output_kinds=list(
                self.registration.get("output_kinds")
                or ["manifest", "report", "annotation_package"]
            ),
            parameters_schema=schema,
            capabilities=caps,
            dependencies=deps,
            definition_digest=digest,
            implementation_digest=impl,
            catalog_generation=self.catalog_generation,
            definition_path=str(
                self.definition_path.relative_to(self.project_root).as_posix()
            )
            if self.definition_path.is_file()
            else "",
            description_updated_at=str(updated_at).strip() if updated_at else None,
            detail={"dag_nodes": dag_nodes, "asr_route_ids": route_ids},
        )

    def validate(self, request: PluginRequest, context: PluginContext) -> ValidationReport:
        desc = self.describe()
        report = ValidationReport(ok=True)
        if desc.status != "enabled":
            report.ok = False
            report.errors.append(
                ValidationIssue(
                    code="PLUGIN_DISABLED",
                    message=f"plugin status is {desc.status}",
                    plugin_ref=request.plugin.as_dict(),
                )
            )
        if request.input.kind not in desc.input_kinds:
            report.ok = False
            report.errors.append(
                ValidationIssue(
                    code="INPUT_KIND_UNSUPPORTED",
                    message=f"input kind {request.input.kind} unsupported",
                    field_path="input.kind",
                    plugin_ref=request.plugin.as_dict(),
                )
            )
        if request.input.kind == "source" and not request.input.source_id:
            report.ok = False
            report.errors.append(
                ValidationIssue(
                    code="INPUT_SOURCE_REQUIRED",
                    message="production pipeline requires source_id",
                    field_path="input.source_id",
                    plugin_ref=request.plugin.as_dict(),
                )
            )
        if request.tags and not desc.capabilities.supports_tag_selection:
            report.ok = False
            report.errors.append(
                ValidationIssue(
                    code="TAG_SELECTION_UNSUPPORTED",
                    message="tag selection not supported",
                    field_path="tags",
                    plugin_ref=request.plugin.as_dict(),
                )
            )
        report.extend(
            validate_against_schema(
                request.parameters, desc.parameters_schema, plugin_ref=request.plugin.as_dict()
            )
        )
        allowed = set(desc.detail.get("asr_route_ids") or [])
        _, model_report = _resolve_models(request, context, allowed_routes=allowed or None)
        report.extend(model_report)
        return report

    def plan(self, request: PluginRequest, context: PluginContext) -> ExecutionPlan:
        context.check_cancelled()
        report = self.validate(request, context)
        if not report.ok:
            raise ValueError(
                "validate failed: " + "; ".join(e.message for e in report.errors)
            )
        desc = self.describe()
        models, _ = _resolve_models(
            request, context, allowed_routes=set(desc.detail.get("asr_route_ids") or []) or None
        )
        plan_id, plan_digest = _plan_ids(request, self.catalog_generation)
        snapshot = freeze_yaml_tree(self.definition_path, project_root=self.project_root)

        selected_routes = request.parameters.get("asr_route_ids") or desc.detail.get(
            "asr_route_ids"
        )
        nodes = []
        for item in desc.detail.get("dag_nodes") or []:
            nid = item["id"]
            if nid.startswith("asr_") and selected_routes:
                rid = nid[len("asr_") :]
                if rid not in selected_routes:
                    continue
            nodes.append(
                PlanNode(
                    node_id=nid,
                    kind="production_node",
                    label=nid,
                    depends_on=list(item.get("depends_on") or []),
                    model_route=nid[len("asr_") :] if nid.startswith("asr_") else None,
                    details=item,
                )
            )

        # Fix evidence_join depends_on to selected ASR only
        selected_asr = [n.node_id for n in nodes if n.node_id.startswith("asr_")]
        for node in nodes:
            if node.node_id == "evidence_join":
                node.depends_on = selected_asr + ["dnsmos"]

        work = context.work_dir or (self.project_root / "runs" / "web" / plan_id)
        warnings = [
            "plan is read-only; no ASR/DNSMOS inference executed",
            "supports_cancel=false until ProductionOrchestrator gains cancel",
            "published datasets remain under shared stage1 paths (not fully isolated)",
        ]
        evidence = {
            "mode": request.execution.evidence_mode or "existing",
            "compute": False,
            "note": "planning never computes evidence",
        }
        return ExecutionPlan(
            plan_id=plan_id,
            plan_digest=plan_digest,
            plugin=request.plugin,
            catalog_generation=self.catalog_generation,
            request=request.as_dict(),
            nodes=nodes,
            resolved_models=models,
            selected_tags=[t.as_dict() for t in request.tags],
            evidence_union=evidence,
            output_path_template=str(work / "nodes" / "{node_id}"),
            expected_artifacts=["manifest", "report"],
            estimated_steps=len(nodes),
            estimated_samples=None,
            supports_resume=True,
            supports_cancel=False,
            definition_snapshot={
                "production": snapshot,
                "implementation_digest": desc.implementation_digest,
                "selected_asr_routes": list(selected_routes or []),
            },
            warnings=warnings,
        )

    def execute(self, plan: ExecutionPlan, context: PluginContext) -> PluginResult:
        context.check_cancelled()
        if self._config is None:
            return PluginResult(
                outcome="failed",
                errors=[
                    ValidationIssue(
                        code="PRODUCTION_CONFIG_MISSING",
                        message="production config not loaded",
                    )
                ],
            )
        work_dir = context.work_dir or (self.project_root / "runs" / "web" / plan.plan_id)
        work_dir.mkdir(parents=True, exist_ok=True)
        snap_dir = context.snapshot_dir or (work_dir / "snapshot")
        write_snapshot_dir(plan.definition_snapshot, snap_dir)

        from audio_engine.plugins.contracts import parse_execution_options

        req = plan.request
        params = dict(req.get("parameters") or {})
        source_name = _engine_source_name(req.get("input") or {}, params)
        if not source_name:
            return PluginResult(
                outcome="failed",
                errors=[
                    ValidationIssue(
                        code="INPUT_SOURCE_REQUIRED",
                        message="source_id required (with resolvable batch)",
                    )
                ],
            )
        source_dir = _engine_source_dir(req.get("input") or {}, params)
        execution = parse_execution_options(req.get("execution"))

        # Optionally filter ASR routes into a temporary production YAML under snapshot.
        selected = params.get("asr_route_ids")
        config_path = self.definition_path
        if selected:
            filtered = self._config.model_copy(deep=True)
            keep = set(selected)
            filtered.asr_routes = [r for r in filtered.asr_routes if r.route_id in keep]
            if not filtered.asr_routes:
                return PluginResult(
                    outcome="failed",
                    errors=[
                        ValidationIssue(
                            code="ASR_ROUTES_EMPTY",
                            message="asr_route_ids filtered to empty set",
                        )
                    ],
                )
            config_path = snap_dir / "effective_production.yaml"
            payload = yaml.safe_load(self.definition_path.read_text(encoding="utf-8"))
            payload["production"]["asr_routes"] = [
                r.model_dump() for r in filtered.asr_routes
            ]
            config_path.write_text(
                yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
            config = load_production_config(config_path)
        else:
            config = self._config

        # Inject per-route env from model bindings into production config copy.
        route_to_conn = {
            b["route_id"]: b["connection_id"] for b in req.get("model_bindings") or []
        }
        if route_to_conn:
            config = config.model_copy(deep=True)
            for route in config.asr_routes:
                cid = route_to_conn.get(route.route_id)
                if not cid:
                    continue
                conn = next(
                    (m for m in plan.resolved_models if m.connection_id == cid), None
                )
                if not conn or not conn.base_url:
                    continue
                env = dict(route.env)
                if conn.provider == "glm":
                    env["GLM_ASR_API_BASE"] = conn.base_url
                    env["GLM_ASR_API_BASES"] = conn.base_url
                    if conn.model:
                        env["GLM_ASR_MODEL"] = conn.model
                elif conn.provider == "qwen":
                    env["QWEN_ASR_API_BASE"] = conn.base_url
                    env["QWEN_ASR_API_BASES"] = conn.base_url
                    if conn.model:
                        env["QWEN_ASR_MODEL"] = conn.model
                route.env = env

        # Tag selection: write a private tagging runtime config if tags provided.
        tagging_config = None
        if req.get("tags"):
            tagging_config = str(
                self._write_private_tagging_config(
                    snap_dir,
                    tags=[PluginRef(**t) for t in req["tags"]],
                    base_runtime=config.tagging.config,
                )
            )

        # Prefer per-route env on ASR nodes. Avoid mutating process-global
        # os.environ when any route already carries an explicit binding — Worker
        # already runs this adapter in an isolated subprocess.
        context.log(f"production execute source={source_name} runs_dir={work_dir}")
        routes_have_binding = any(
            (r.env.get("GLM_ASR_API_BASE") or r.env.get("QWEN_ASR_API_BASE"))
            for r in config.asr_routes
        )
        env = _task_env_from_models(plan.resolved_models, context=context)
        old_env: dict[str, str | None] = {}
        try:
            if not routes_have_binding:
                for k, v in env.items():
                    if k.startswith(("GLM_ASR_", "QWEN_ASR_")):
                        old_env[k] = os.environ.get(k)
                        os.environ[k] = v
            else:
                # Still clear inherited conflicting vars without installing a
                # single global base that would fight multi-route isolation.
                for k in (
                    "GLM_ASR_API_BASE",
                    "GLM_ASR_API_BASES",
                    "GLM_ASR_MODEL",
                    "QWEN_ASR_API_BASE",
                    "QWEN_ASR_API_BASES",
                    "QWEN_ASR_MODEL",
                ):
                    if k in os.environ:
                        old_env[k] = os.environ.pop(k)
            orch = ProductionOrchestrator(
                config,
                config_path=config_path,
                source_name=source_name,
                source_dir=source_dir,
                runs_dir=work_dir,
                resume=str(work_dir) if execution.resume else None,
                force=execution.force,
                mock=bool(execution.extra.get("mock")),
                log=lambda msg: context.log(msg),
                tagging_config=tagging_config,
            )
            result = orch.run()
        except CancelledError:
            return PluginResult(outcome="cancelled", run_dir=str(work_dir))
        except Exception as exc:  # noqa: BLE001
            return PluginResult(
                outcome="failed",
                run_dir=str(work_dir),
                errors=[
                    ValidationIssue(code="PRODUCTION_EXECUTE_FAILED", message=str(exc))
                ],
            )
        finally:
            for k, v in old_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

        outcome = "succeeded" if result.status == "succeeded" else "failed"
        return PluginResult(
            outcome=outcome,  # type: ignore[arg-type]
            run_dir=str(result.run_root),
            counts={"nodes": len(result.node_states)},
            details={"node_states": result.node_states},
            snapshot_digest=plan.plan_digest,
            warnings=[
                "dataset outputs may reside under shared datasets/stage1 paths",
            ],
        )

    def _write_private_tagging_config(
        self,
        snap_dir: Path,
        *,
        tags: list[PluginRef],
        base_runtime: str | None,
    ) -> Path:
        runtime = base_runtime or "configs/tagging/zh_asr_free_tags_v2.yaml"
        src = self.project_root / runtime
        data = yaml.safe_load(src.read_text(encoding="utf-8"))
        data["select_tags"] = [{"tag_id": t.id, "version": t.version} for t in tags]
        out = snap_dir / "tagging_runtime.yaml"
        out.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
        return out


# ---------------------------------------------------------------------------
# Tag definition adapter + batch coordinator
# ---------------------------------------------------------------------------


@dataclass
class TagDefinitionPlugin:
    entry_raw: dict[str, Any]
    library_root: Path
    runtime_config: Path
    project_root: Path
    catalog_generation: int = 0
    source_path: str = ""

    @property
    def tag_id(self) -> str:
        return str(self.entry_raw["tag_id"])

    @property
    def version(self) -> str:
        return str(self.entry_raw.get("version") or "1")

    def describe(self) -> PluginDescriptor:
        from audio_engine.core.tagging.library import _status_flags

        enabled, archived = _status_flags(self.entry_raw, path=self.source_path or self.tag_id)
        if archived:
            status = "archived"
        elif not enabled:
            status = "disabled"
        else:
            status = "enabled"
        deps_raw = self.entry_raw.get("dependencies") or {}
        requires_acoustic = bool(deps_raw.get("requires_acoustic") or False)
        schema = {
            "type": "object",
            "additionalProperties": False,
            "title": "Tag 批次共享参数",
            "description": "单 tag 无独立覆盖参数；批次级参数由 tagging 任务 parameters 提供",
            "properties": {},
        }
        caps = PluginCapabilities(
            supports_resume=True,
            supports_cancel=False,
            supports_tag_selection=False,
            supports_isolated_output=True,
            supports_task_private_env=True,
            purpose="tagging",
            notes=[
                "execute delegates to TagBatchCoordinator / TaggingService",
                "cancel: TaggingService has no cooperative cancel API; capability=false",
                "multi-tag requests must use tagging task_type for shared evidence",
            ],
        )
        deps = [
            PluginDependency(kind="fragment", ref=str(x), required=True)
            for x in (deps_raw.get("fragments") or [])
        ]
        deps += [
            PluginDependency(kind="lexicon", ref=str(x), required=True)
            for x in (deps_raw.get("lexicons") or [])
        ]
        if requires_acoustic:
            deps.append(
                PluginDependency(kind="evidence", ref="acoustic", required=True)
            )
        digest = sha256_obj(self.entry_raw)
        impl = digest_implementation_files(
            _impl_files(*TAG_IMPL, project_root=self.project_root)
        )
        updated_at = self.entry_raw.get("description_updated_at")
        return PluginDescriptor(
            kind="tag",
            id=self.tag_id,
            version=self.version,
            api_version=API_VERSION,
            name=str(self.entry_raw.get("name") or self.tag_id),
            description=str(self.entry_raw.get("description") or ""),
            adapter="tag_definition",
            status=status,  # type: ignore[arg-type]
            runnable=status == "enabled",
            not_runnable_reasons=[] if status == "enabled" else [f"status={status}"],
            input_kinds=["dataset"],
            output_kinds=["manifest", "report"],
            parameters_schema=schema,
            capabilities=caps,
            dependencies=deps,
            definition_digest=digest,
            implementation_digest=impl,
            catalog_generation=self.catalog_generation,
            definition_path=self.source_path,
            description_updated_at=str(updated_at).strip() if updated_at else None,
            detail={
                "business_definition": self.entry_raw.get("business_definition") or "",
                "rules": self.entry_raw.get("rules") or [],
                "examples": self.entry_raw.get("examples") or {},
                "dependencies": deps_raw,
                "requires_acoustic": requires_acoustic,
            },
        )

    def validate(self, request: PluginRequest, context: PluginContext) -> ValidationReport:
        desc = self.describe()
        report = ValidationReport(ok=True)
        if desc.status != "enabled":
            report.ok = False
            report.errors.append(
                ValidationIssue(
                    code="PLUGIN_DISABLED",
                    message=f"tag status is {desc.status}",
                    plugin_ref=request.plugin.as_dict(),
                    definition_path=self.source_path,
                )
            )
        if request.plugin.version != desc.version:
            report.ok = False
            report.errors.append(
                ValidationIssue(
                    code="TAG_VERSION_MISMATCH",
                    message=f"requested {request.plugin.version}, library has {desc.version}",
                    plugin_ref=request.plugin.as_dict(),
                    definition_path=self.source_path,
                )
            )
        if request.input.kind != "dataset":
            report.ok = False
            report.errors.append(
                ValidationIssue(
                    code="INPUT_KIND_UNSUPPORTED",
                    message="tag plugins require dataset input",
                    field_path="input.kind",
                    plugin_ref=request.plugin.as_dict(),
                )
            )
        report.extend(
            validate_against_schema(
                request.parameters, desc.parameters_schema, plugin_ref=request.plugin.as_dict()
            )
        )
        return report

    def plan(self, request: PluginRequest, context: PluginContext) -> ExecutionPlan:
        # Single-tag plan delegates to batch coordinator with one tag.
        coordinator = TagBatchCoordinator(
            runtime_config=self.runtime_config,
            library_root=self.library_root,
            project_root=self.project_root,
            catalog_generation=self.catalog_generation,
        )
        return coordinator.plan_tags([request.plugin], request, context)

    def execute(self, plan: ExecutionPlan, context: PluginContext) -> PluginResult:
        coordinator = TagBatchCoordinator(
            runtime_config=self.runtime_config,
            library_root=self.library_root,
            project_root=self.project_root,
            catalog_generation=self.catalog_generation,
        )
        return coordinator.execute(plan, context)


@dataclass
class TagBatchCoordinator:
    """Batch multiple selected tags into one TaggingService run (shared evidence)."""

    runtime_config: Path
    library_root: Path
    project_root: Path
    catalog_generation: int = 0

    def plan_tags(
        self,
        tags: list[PluginRef],
        request: PluginRequest,
        context: PluginContext,
    ) -> ExecutionPlan:
        context.check_cancelled()
        if not tags:
            raise ValueError("tagging requires at least one tag")
        mode = request.execution.evidence_mode or "existing"
        if mode in ("plan", "preview") or context.readonly:
            compute = False
        else:
            compute = mode == "compute_acoustic"

        library = load_tag_library(self.library_root, base_dir=self.project_root)
        select = [{"tag_id": t.id, "version": t.version} for t in tags]
        try:
            tag_defs, rules = select_from_library(library, select, source_label="plugin.plan")
        except TagLibraryError as exc:
            raise ValueError(str(exc)) from exc

        # Build effective config snapshot without mutating the on-disk runtime.
        runtime_text = self.runtime_config.read_text(encoding="utf-8")
        runtime_data = yaml.safe_load(runtime_text)
        runtime_data = dict(runtime_data)
        runtime_data["select_tags"] = select
        if request.execution.evidence_mode:
            runtime_data["evidence_mode"] = (
                "existing" if mode in ("plan", "preview", "existing") else mode
            )

        # Load via temp file under work/snapshot if available, else ephemeral.
        tmp_dir = context.work_dir or (self.project_root / "runs" / "plugins" / "_plan")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_cfg = tmp_dir / f"tagging_plan_{uuid.uuid4().hex[:8]}.yaml"
        tmp_cfg.write_text(
            yaml.safe_dump(runtime_data, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        try:
            cfg = load_tagging_config(tmp_cfg)
            definition_snapshot = build_definition_snapshot(cfg)
        finally:
            if context.readonly or mode in ("plan", "preview"):
                try:
                    tmp_cfg.unlink(missing_ok=True)
                except OSError:
                    pass

        acoustic_tags = [
            tid
            for tid, t in tag_defs.items()
            if t.requires_acoustic
        ]
        evidence_union = {
            "mode": mode,
            "compute": bool(compute and acoustic_tags),
            "acoustic_tag_ids": acoustic_tags,
            "shared_evidence": True,
            "note": (
                "multiple tags share one TaggingService batch; "
                "ASR is never re-run by tagging; acoustic only if compute_acoustic"
            ),
        }
        if mode in ("existing", "plan", "preview"):
            evidence_union["inference_calls"] = 0

        plan_id, plan_digest = _plan_ids(request, self.catalog_generation)
        nodes = [
            PlanNode(
                node_id="tag_batch",
                kind="tagging",
                label="TaggingService batch",
                details={
                    "tag_ids": [t.id for t in tags],
                    "rule_count": len(rules),
                    "evidence_mode": mode,
                },
            )
        ]
        work = context.work_dir or (self.project_root / "runs" / "plugins" / plan_id)
        root_ref = request.plugin if request.plugin.kind == "tag" else tags[0]
        return ExecutionPlan(
            plan_id=plan_id,
            plan_digest=plan_digest,
            plugin=root_ref,
            catalog_generation=self.catalog_generation,
            request=request.as_dict(),
            nodes=nodes,
            selected_tags=[t.as_dict() for t in tags],
            evidence_union=evidence_union,
            output_path_template=str(work / "outputs" / "{artifact}"),
            expected_artifacts=["manifest", "report"],
            estimated_steps=1,
            estimated_samples=None,
            supports_resume=True,
            supports_cancel=False,
            definition_snapshot={
                "tagging": definition_snapshot,
                "runtime_config_digest": sha256_file(self.runtime_config),
                "select_tags": select,
                "implementation_digest": digest_implementation_files(
                    _impl_files(*TAG_IMPL, project_root=self.project_root)
                ),
            },
            warnings=[
                "plan/preview/existing modes do not invoke ASR or acoustic models",
                "supports_cancel=false for TaggingService",
            ],
        )

    def execute(self, plan: ExecutionPlan, context: PluginContext) -> PluginResult:
        context.check_cancelled()
        mode = (plan.evidence_union or {}).get("mode") or "existing"
        if mode in ("plan",):
            return PluginResult(
                outcome="succeeded",
                details={"note": "plan-only; no execution"},
                snapshot_digest=plan.plan_digest,
            )

        work_dir = context.work_dir or (self.project_root / "runs" / "plugins" / plan.plan_id)
        work_dir.mkdir(parents=True, exist_ok=True)
        snap_dir = context.snapshot_dir or (work_dir / "snapshot")
        write_snapshot_dir(plan.definition_snapshot, snap_dir)

        select = plan.definition_snapshot.get("select_tags") or plan.selected_tags
        runtime_data = yaml.safe_load(self.runtime_config.read_text(encoding="utf-8"))
        runtime_data = dict(runtime_data)
        runtime_data["select_tags"] = [
            {"tag_id": t["id"] if "id" in t else t["tag_id"], "version": t["version"]}
            for t in select
        ]
        if mode == "existing":
            runtime_data["evidence_mode"] = "existing"
        elif mode == "compute_acoustic":
            runtime_data["evidence_mode"] = "compute_acoustic"
        cfg_path = snap_dir / "tagging_runtime.yaml"
        cfg_path.write_text(
            yaml.safe_dump(runtime_data, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

        dataset = plan.request["input"].get("dataset_id") or plan.request["input"].get(
            "artifact_id"
        )
        if not dataset:
            return PluginResult(
                outcome="failed",
                errors=[
                    ValidationIssue(
                        code="INPUT_DATASET_REQUIRED",
                        message="tagging execute requires dataset_id",
                    )
                ],
            )
        manifest_path = Path(dataset)
        if not manifest_path.is_absolute():
            manifest_path = self.project_root / manifest_path
        if not manifest_path.is_file():
            return PluginResult(
                outcome="failed",
                errors=[
                    ValidationIssue(
                        code="DATASET_NOT_FOUND",
                        message=f"dataset not found: {manifest_path}",
                    )
                ],
            )

        from audio_engine.core.manifest import Manifest

        samples = Manifest.load(manifest_path).samples
        tag_params = dict(plan.request.get("parameters") or {})
        batch_id = (
            _engine_source_name(plan.request.get("input") or {}, tag_params)
            or plan.request["input"].get("source_id")
            or manifest_path.stem
        )
        out_manifest = work_dir / "outputs" / "tagged.parquet"
        out_manifest.parent.mkdir(parents=True, exist_ok=True)
        summary = work_dir / "outputs" / "summary.xlsx"

        service = TaggingService.from_path(cfg_path)
        try:
            result = service.run(
                TaggingRunRequest(
                    batch_id=batch_id,
                    config_path=cfg_path,
                    samples=list(samples),
                    evidence_mode=runtime_data.get("evidence_mode"),
                    work_root=work_dir / "tagging_work",
                    preview=bool(plan.request.get("execution", {}).get("preview"))
                    or mode == "preview",
                    publish=mode != "preview",
                    output_manifest=out_manifest,
                    summary_xlsx=summary,
                    pipeline_run_dir=work_dir,
                )
            )
        except CancelledError:
            return PluginResult(outcome="cancelled", run_dir=str(work_dir))
        except Exception as exc:  # noqa: BLE001
            return PluginResult(
                outcome="failed",
                run_dir=str(work_dir),
                errors=[
                    ValidationIssue(code="TAGGING_EXECUTE_FAILED", message=str(exc))
                ],
            )

        artifacts = []
        if out_manifest.exists():
            artifacts.append(ArtifactRef(artifact_kind="manifest", path=str(out_manifest)))
        if summary.exists():
            artifacts.append(ArtifactRef(artifact_kind="report", path=str(summary)))
        return PluginResult(
            outcome="succeeded",
            artifacts=artifacts,
            run_dir=str(work_dir),
            counts=dict(result.stats or {}),
            metrics=dict(result.metrics or {}),
            snapshot_digest=plan.plan_digest,
            details={
                "run_id": result.run_id,
                "shared_evidence": True,
                "evidence_mode": runtime_data.get("evidence_mode"),
            },
        )


# ---------------------------------------------------------------------------
# Adapter registration factories
# ---------------------------------------------------------------------------


def create_yaml_pipeline(
    registration: dict[str, Any], *, project_root: Path, catalog_generation: int
) -> YamlPipelinePlugin:
    return YamlPipelinePlugin(
        registration=registration,
        project_root=project_root,
        catalog_generation=catalog_generation,
    )


def create_production_pipeline(
    registration: dict[str, Any], *, project_root: Path, catalog_generation: int
) -> ProductionPipelinePlugin:
    return ProductionPipelinePlugin(
        registration=registration,
        project_root=project_root,
        catalog_generation=catalog_generation,
    )


def create_tag_definition(
    *,
    entry_raw: dict[str, Any],
    library_root: Path,
    runtime_config: Path,
    project_root: Path,
    catalog_generation: int,
    source_path: str,
) -> TagDefinitionPlugin:
    return TagDefinitionPlugin(
        entry_raw=entry_raw,
        library_root=library_root,
        runtime_config=runtime_config,
        project_root=project_root,
        catalog_generation=catalog_generation,
        source_path=source_path,
    )
