"""Trusted test-only plugin adapter for job/lease/publish fault injection.

Referenced only from temporary plugins.yaml in tests via extension_modules.
Production configs/plugins.yaml must not list this module.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

from audio_engine.core.checkpoint import digest_payload
from audio_engine.plugins.contracts import (
    API_VERSION,
    ArtifactRef,
    ExecutionPlan,
    PlanNode,
    PluginCapabilities,
    PluginContext,
    PluginDescriptor,
    PluginRequest,
    PluginResult,
    ValidationIssue,
    ValidationReport,
)
from audio_engine.plugins.registry import register_extension_adapter


class HarnessPipelinePlugin:
    def __init__(
        self,
        registration: dict[str, Any],
        *,
        project_root: Path,
        catalog_generation: int,
    ) -> None:
        self.registration = registration
        self.project_root = project_root
        self.catalog_generation = catalog_generation

    def describe(self) -> PluginDescriptor:
        pid = str(self.registration["id"])
        version = str(self.registration.get("version") or "1")
        return PluginDescriptor(
            kind="pipeline",
            id=pid,
            version=version,
            api_version=API_VERSION,
            name=str(self.registration.get("name") or pid),
            description="test harness pipeline",
            adapter="test_harness",
            status="enabled",
            runnable=True,
            input_kinds=list(self.registration.get("input_kinds") or ["source", "dataset"]),
            output_kinds=["report"],
            parameters_schema=self.registration.get("parameters_schema")
            or {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["ok", "fail", "sleep", "crash"],
                        "default": "ok",
                        "title": "模式",
                    },
                    "sleep_seconds": {
                        "type": "number",
                        "default": 0.2,
                        "title": "睡眠秒数",
                    },
                },
            },
            capabilities=PluginCapabilities(
                supports_resume=bool(self.registration.get("supports_resume", True)),
                supports_cancel=True,
                supports_tag_selection=False,
                supports_isolated_output=True,
                supports_task_private_env=True,
                purpose="processing",
                notes=["test harness only"],
            ),
            definition_digest=f"harness:{pid}:{version}",
            implementation_digest="harness_adapter_v1",
            catalog_generation=self.catalog_generation,
            definition_path="audio_engine.web.testing.harness_adapter",
        )

    def validate(self, request: PluginRequest, context: PluginContext) -> ValidationReport:
        return ValidationReport(ok=True)

    def plan(self, request: PluginRequest, context: PluginContext) -> ExecutionPlan:
        plan_id = f"plan_{uuid.uuid4().hex[:16]}"
        digest = digest_payload(request.as_dict())
        return ExecutionPlan(
            plan_id=plan_id,
            plan_digest=digest,
            plugin=request.plugin,
            catalog_generation=self.catalog_generation,
            request=request.as_dict(),
            nodes=[PlanNode(node_id="harness", kind="test", label="harness")],
            output_path_template=str((context.work_dir or Path(".")) / "out"),
            supports_resume=True,
            supports_cancel=True,
            definition_snapshot={"harness": True, "request": request.as_dict()},
            warnings=["harness plan is read-only; no inference"],
        )

    def execute(self, plan: ExecutionPlan, context: PluginContext) -> PluginResult:
        params = dict(plan.request.get("parameters") or {})
        mode = str(params.get("mode") or "ok")
        sleep_seconds = float(params.get("sleep_seconds") or 0.2)
        work = context.work_dir or Path(".")
        work.mkdir(parents=True, exist_ok=True)
        out = work / "harness_out.json"

        if mode == "sleep":
            end = time.time() + sleep_seconds
            while time.time() < end:
                if context.cancel_token and context.cancel_token.cancelled:
                    return PluginResult(outcome="cancelled", run_dir=str(work))
                time.sleep(0.05)
            mode = "ok"

        if mode == "crash":
            raise RuntimeError("harness intentional crash")

        if mode == "fail":
            return PluginResult(
                outcome="failed",
                run_dir=str(work),
                errors=[ValidationIssue(code="HARNESS_FAIL", message="intentional fail")],
            )

        payload = {
            "ok": True,
            "job_id": context.job_id,
            "attempt_id": context.attempt_id,
            "env_glm": os.environ.get("GLM_ASR_API_BASE"),
            "env_qwen": os.environ.get("QWEN_ASR_API_BASE"),
        }
        out.write_text(json.dumps(payload), encoding="utf-8")
        context.log("harness succeeded")
        return PluginResult(
            outcome="succeeded",
            run_dir=str(work),
            artifacts=[
                ArtifactRef(artifact_kind="report", path=str(out), artifact_id=str(out))
            ],
            counts={"processed": 1, "succeeded": 1},
        )


def create_test_harness(
    registration: dict[str, Any], *, project_root: Path, catalog_generation: int
) -> HarnessPipelinePlugin:
    return HarnessPipelinePlugin(
        registration,
        project_root=project_root,
        catalog_generation=catalog_generation,
    )


register_extension_adapter("test_harness", create_test_harness)
