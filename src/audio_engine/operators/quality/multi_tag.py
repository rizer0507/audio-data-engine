"""quality.multi_tag_v1 — composable multi-tag production operator (040)."""

from __future__ import annotations

from pathlib import Path
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
    version = "1.2.0"
    category = "quality"

    def dependency_fingerprint(self, config: OperatorConfig) -> dict[str, Any] | None:
        params = dict(config.params)
        config_path = _param(params, "config_path", "tagging_config")
        if not config_path:
            return None
        from audio_engine.core.tagging.config import load_tagging_config
        from audio_engine.core.tagging.persist import resolve_work_root

        cfg = load_tagging_config(config_path)
        batch_id = str(_param(params, "batch_id", "batch", default=""))
        work_root = _param(params, "work_root", "work_dir")
        manual_version = None
        state_revision = None
        if batch_id:
            root = resolve_work_root(
                batch_id=batch_id,
                policy_version=cfg.policy_version,
                template=cfg.work_dir_template,
                override=work_root,
            )
            state_path = Path(root) / "tag_state.json"
            if state_path.is_file():
                import hashlib

                manual_version = hashlib.sha256(
                    state_path.read_bytes()
                ).hexdigest()
            indexed_path = state_path.with_suffix(".sqlite3")
            if indexed_path.exists():
                import sqlite3
                from contextlib import closing
                with closing(sqlite3.connect(indexed_path)) as db:
                    state_revision = db.execute("SELECT revision FROM metadata").fetchone()[0]
        return {
            "tagging_content_fingerprint": cfg.content_fingerprint,
            "policy_version": cfg.policy_version,
            "evidence_mode": _param(params, "evidence_mode") or cfg.evidence_mode,
            "manual_tag_state_sha256": manual_version,
            "tag_state_revision": state_revision,
            "config_path": str(Path(str(config_path)).as_posix()),
        }

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
        acoustic_adapter = _param(params, "acoustic_adapter")

        if isinstance(rule_ids, str):
            rule_ids = [x.strip() for x in rule_ids.split(",") if x.strip()]
        if isinstance(sample_ids, str):
            sample_ids = [x.strip() for x in sample_ids.split(",") if x.strip()]
        if isinstance(overlap_tags, str):
            overlap_tags = [x.strip() for x in overlap_tags.split(",") if x.strip()]

        service = TaggingService.from_path(config_path)
        from audio_engine.core.checkpoint import digest_samples
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
                acoustic_adapter=acoustic_adapter,
                workers=_param(params, "workers"),
                executor=_param(params, "executor", default="auto"),
                max_in_flight=_param(params, "max_in_flight"),
                checkpoint_every=_param(params, "checkpoint_every", default=100),
                batch_size=_param(params, "batch_size", default=32),
                engine_workers=_param(params, "engine_workers", default={}),
                engine_max_in_flight=_param(params, "engine_max_in_flight", default={}),
                engine_retries=_param(params, "engine_retries", default=0),
                retain_results=bool(_param(params, "retain_results", default=False)),
                pipeline_run_dir=config.run_dir,
                step_name=config.step_name or "multi_tag",
                pipeline_resume=config.resume,
                force_recompute=config.force,
                input_fingerprint=digest_samples(samples),
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
            tagging["run_dir"] = result.run_dir
            tagging["pipeline_run_id"] = config.run_dir.name if config.run_dir else None
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
