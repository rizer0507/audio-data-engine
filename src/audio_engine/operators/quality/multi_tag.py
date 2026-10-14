"""quality.multi_tag_v1 — composable multi-tag production operator (040)."""

from __future__ import annotations

from typing import Any

from audio_engine.core.operator import ManifestOperator, OperatorConfig
from audio_engine.core.registry import register_operator
from audio_engine.core.sample import Sample
from audio_engine.core.tagging.service import TaggingRunRequest, TaggingService


def _param(params: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in params and params[key] is not None:
            return params[key]
    return default


@register_operator
class MultiTagV1Operator(ManifestOperator):
    """Evaluate independent tag rules; write labels.tags / labels.tagging."""

    name = "multi_tag_v1"
    version = "1.0.0"
    category = "quality"

    def run(self, samples: list[Sample], config: OperatorConfig) -> list[Sample]:
        params = dict(config.params)
        config_path = _param(params, "config_path", "tagging_config")
        if not config_path:
            raise ValueError("quality.multi_tag_v1 requires config_path")

        batch_id = str(_param(params, "batch_id", "batch", default="unknown_batch"))
        evidence_mode = _param(params, "evidence_mode")
        preview = bool(_param(params, "preview", default=False))
        resume_run_id = _param(params, "resume_run_id", "resume")
        work_root = _param(params, "work_root", "work_dir")
        rule_ids = _param(params, "rule_ids")
        sample_ids = _param(params, "sample_ids")
        publish = bool(_param(params, "publish", default=not preview))
        output_manifest = _param(params, "output_manifest")
        summary_xlsx = _param(params, "summary_xlsx")
        overlap_tags = _param(params, "overlap_tags")

        if isinstance(rule_ids, str):
            rule_ids = [x.strip() for x in rule_ids.split(",") if x.strip()]
        if isinstance(sample_ids, str):
            sample_ids = [x.strip() for x in sample_ids.split(",") if x.strip()]
        if isinstance(overlap_tags, str):
            overlap_tags = [x.strip() for x in overlap_tags.split(",") if x.strip()]

        service = TaggingService.from_path(config_path)
        ctx = getattr(config, "context", None)
        if ctx is not None and hasattr(ctx, "log"):
            ctx.log("multi_tag_start", batch_id=batch_id, sample_count=len(samples))

        result = service.run(
            TaggingRunRequest(
                batch_id=batch_id,
                config_path=config_path,
                samples=samples,
                evidence_mode=evidence_mode,
                work_root=work_root,
                resume_run_id=resume_run_id,
                rule_ids=rule_ids,
                sample_ids=sample_ids,
                preview=preview,
                publish=publish,
                output_manifest=output_manifest,
                summary_xlsx=summary_xlsx,
                overlap_tags=overlap_tags,
            )
        )

        # Attach run meta onto each sample lineage-ish via labels.tagging.run
        out: list[Sample] = []
        for sample in result.samples:
            labels = dict(sample.labels or {})
            tagging = dict(labels.get("tagging") or {})
            tagging["run_id"] = result.run_id
            tagging["snapshot_id"] = result.snapshot_id
            tagging["stats"] = result.stats
            labels["tagging"] = tagging
            sample.labels = labels
            sample.mark_completed(self.full_name)
            sample.add_lineage(
                operator=self.full_name,
                version=self.version,
                params={
                    "config_path": str(config_path),
                    "batch_id": batch_id,
                    "evidence_mode": evidence_mode or service.config.evidence_mode,
                    "preview": preview,
                    "run_id": result.run_id,
                    "snapshot_id": result.snapshot_id,
                },
            )
            out.append(sample)

        if ctx is not None and hasattr(ctx, "log"):
            ctx.log(
                "multi_tag_finished",
                batch_id=batch_id,
                run_id=result.run_id,
                snapshot_id=result.snapshot_id,
                stats=result.stats,
                messages=result.messages,
            )
        return out
