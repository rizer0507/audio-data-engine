"""Bridge from Web layer to unified plugin registry (no business-name dispatch)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from audio_engine.plugins.contracts import (
    PluginContext,
    PluginStatus,
    ResolvedModelConnection,
    parse_task_request,
)
from audio_engine.plugins.registry import PluginRegistry
from audio_engine.plugins.snapshot import write_snapshot_dir
from audio_engine.web.services.sources import SourceRegistryError, SourceService, effective_batch
from audio_engine.web.settings import ModelConnectionConfig, WebSettings


def connections_to_plugin_map(
    models: list[ModelConnectionConfig],
) -> dict[str, ResolvedModelConnection]:
    out: dict[str, ResolvedModelConnection] = {}
    for m in models:
        out[m.connection_id] = ResolvedModelConnection(
            connection_id=m.connection_id,
            provider=m.provider,
            base_url=m.base_url,
            model=m.model,
            model_revision=m.model_revision,
            api_key_env=m.api_key_env,
            timeout_seconds=m.timeout_seconds,
            concurrency=m.concurrency,
            secret_refs={"api_key": m.api_key_env} if m.api_key_env else {},
        )
    return out


class EngineService:
    def __init__(
        self,
        settings: WebSettings,
        registry: PluginRegistry,
        source_service: SourceService | None = None,
    ) -> None:
        self.settings = settings
        self.registry = registry
        self.source_service = source_service or SourceService(
            settings.sources_manifest, settings.allowed_source_roots
        )

    def ensure_catalog(self) -> Any:
        if getattr(self.registry, "_catalog", None) is None:
            self.registry.refresh(self.settings.plugins_config)
        return self.registry.catalog

    def refresh_plugins(self) -> dict[str, Any]:
        catalog = self.registry.refresh(self.settings.plugins_config)
        return catalog.as_dict()

    def list_plugins(self, **filters: Any) -> dict[str, Any]:
        catalog = self.ensure_catalog()
        items = [d.as_dict() for d in catalog.list(**filters)]
        return {
            "generation": catalog.generation,
            "revision": catalog.generation,
            "items": items,
            "next_cursor": None,
            "total": len(items),
            "diagnostics": catalog.diagnostics,
        }

    def get_plugin(self, kind: str, plugin_id: str, version: str | None = None) -> dict[str, Any]:
        self.ensure_catalog()
        return self.registry.describe_detail(kind, plugin_id, version)  # type: ignore[arg-type]

    def set_plugin_status(
        self,
        kind: str,
        plugin_id: str,
        status: PluginStatus,
        *,
        expected_revision: int | None = None,
    ) -> dict[str, Any]:
        return self.registry.set_status(
            kind,  # type: ignore[arg-type]
            plugin_id,
            status,
            expected_revision=expected_revision,
            plugins_config=self.settings.plugins_config,
        )

    def validate_plugin_definition(
        self,
        kind: str,
        plugin_id: str,
        draft: dict[str, Any],
        *,
        expected_revision: int | None = None,
        expected_definition_digest: str | None = None,
        version: str | None = None,
    ) -> dict[str, Any]:
        return self.registry.validate_definition(
            kind,  # type: ignore[arg-type]
            plugin_id,
            draft,
            expected_revision=expected_revision,
            expected_definition_digest=expected_definition_digest,
            version=version,
            plugins_config=self.settings.plugins_config,
        )

    def save_plugin_definition(
        self,
        kind: str,
        plugin_id: str,
        draft: dict[str, Any],
        *,
        expected_revision: int | None = None,
        expected_definition_digest: str | None = None,
        version: str | None = None,
    ) -> dict[str, Any]:
        return self.registry.save_definition(
            kind,  # type: ignore[arg-type]
            plugin_id,
            draft,
            expected_revision=expected_revision,
            expected_definition_digest=expected_definition_digest,
            version=version,
            plugins_config=self.settings.plugins_config,
        )

    def resolve_source_into_request(self, request_data: dict[str, Any]) -> dict[str, Any]:
        """Bind internal source_id to engine batch/path on input (not parameters schema)."""

        data = dict(request_data)
        inp = dict(data.get("input") or {})
        if inp.get("kind") != "source":
            return data
        source_id = str(inp.get("source_id") or "").strip()
        if not source_id:
            return data
        try:
            item = self.source_service.get(source_id)
        except SourceRegistryError as exc:
            raise ValueError(f"source not found: {source_id}") from exc
        batch = effective_batch(item)
        if not batch:
            raise ValueError(
                f"source {source_id} has no usable batch for engine source_name; "
                "set a valid batch (letters/digits/_/-) via PATCH"
            )
        inp["batch"] = batch
        if item.get("path"):
            inp["path"] = item["path"]
        data["input"] = inp
        return data

    def validate(self, request_data: dict[str, Any]) -> dict[str, Any]:
        self.ensure_catalog()
        resolved = self.resolve_source_into_request(request_data)
        task = parse_task_request(resolved)
        ctx = PluginContext(
            project_root=self.settings.project_root,
            model_connections=connections_to_plugin_map(self.settings.models),
            readonly=True,
        )
        return self.registry.validate_task(task, ctx).as_dict()

    def plan(self, request_data: dict[str, Any], *, store_dir: Path) -> dict[str, Any]:
        self.ensure_catalog()
        resolved = self.resolve_source_into_request(request_data)
        task = parse_task_request(resolved)
        if task.catalog_generation is None:
            task.catalog_generation = self.registry.catalog.generation
        ctx = PluginContext(
            project_root=self.settings.project_root,
            work_dir=store_dir,
            model_connections=connections_to_plugin_map(self.settings.models),
            readonly=True,
        )
        plan = self.registry.plan_task(task, ctx)
        snap_dir = store_dir / "plans" / plan.plan_id
        digest = write_snapshot_dir(plan.definition_snapshot, snap_dir)
        payload = plan.as_dict()
        payload["snapshot_path"] = str(snap_dir)
        payload["snapshot_digest"] = digest
        # Persist plan json beside snapshot for worker
        (snap_dir / "plan.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (snap_dir / "task_request.json").write_text(
            json.dumps(task.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return payload

    def execute_frozen_plan(
        self,
        plan_dict: dict[str, Any],
        *,
        work_dir: Path,
        job_id: str,
        attempt_id: str,
        on_log=None,
        on_progress=None,
        cancel_token=None,
    ) -> dict[str, Any]:
        from audio_engine.plugins.contracts import (
            CancelToken,
            ExecutionPlan,
            PlanNode,
            PluginRef,
            ResolvedModelConnection,
        )

        self.ensure_catalog()
        plugin = PluginRef(**plan_dict["plugin"])
        nodes = [PlanNode(**n) for n in plan_dict.get("nodes") or []]
        models = [
            ResolvedModelConnection(**m) for m in plan_dict.get("resolved_models") or []
        ]
        plan = ExecutionPlan(
            plan_id=plan_dict["plan_id"],
            plan_digest=plan_dict["plan_digest"],
            plugin=plugin,
            catalog_generation=int(plan_dict.get("catalog_generation") or 0),
            request=dict(plan_dict.get("request") or {}),
            nodes=nodes,
            resolved_models=models,
            selected_tags=list(plan_dict.get("selected_tags") or []),
            evidence_union=dict(plan_dict.get("evidence_union") or {}),
            output_path_template=str(plan_dict.get("output_path_template") or ""),
            expected_artifacts=list(plan_dict.get("expected_artifacts") or []),
            estimated_steps=plan_dict.get("estimated_steps"),
            estimated_samples=plan_dict.get("estimated_samples"),
            supports_resume=bool(plan_dict.get("supports_resume")),
            supports_cancel=bool(plan_dict.get("supports_cancel")),
            missing_conditions=list(plan_dict.get("missing_conditions") or []),
            definition_snapshot=dict(plan_dict.get("definition_snapshot") or {}),
            warnings=list(plan_dict.get("warnings") or []),
            task_request=plan_dict.get("task_request"),
            sub_plans=list(plan_dict.get("sub_plans") or []),
        )
        ctx = PluginContext(
            project_root=self.settings.project_root,
            job_id=job_id,
            attempt_id=attempt_id,
            work_dir=work_dir,
            snapshot_dir=work_dir / "snapshot",
            model_connections={m.connection_id: m for m in models},
            cancel_token=cancel_token or CancelToken(),
            on_log=on_log,
            on_progress=on_progress,
            readonly=False,
        )
        # Execute via registry using frozen plan content (adapters still resolve
        # plugin implementation by id; definition bodies come from snapshot on disk).
        result = self.registry.execute_plan(plan, ctx)
        return result.as_dict()
