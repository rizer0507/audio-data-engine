"""Unified multi-tag service used by pipeline operator and CLI (040 §12.2)."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from audio_engine.core.manifest import Manifest
from audio_engine.core.sample import Sample
from audio_engine.core.tagging.adapters import AcousticAdapter
from audio_engine.core.tagging.config import TaggingConfig, load_tagging_config
from audio_engine.core.tagging.engine import MultiTagEngine, SampleTaggingResult
from audio_engine.core.tagging.export import (
    apply_result_to_sample,
    filter_samples,
    write_rule_details,
    write_summary_xlsx,
)
from audio_engine.core.tagging.persist import (
    RunWorkspace,
    atomic_publish,
    create_run_workspace,
    load_progress,
    save_progress,
)
from audio_engine.core.tagging.state import (
    ManualOpLog,
    ManualOpRecord,
    TagStateStore,
    apply_manual_add,
    apply_manual_remove,
    apply_restore_auto,
    apply_retract_manual,
    effective_tags,
    new_op_id,
    now_ts,
)
from audio_engine.core.tagging.types import TagSource


@dataclass
class TaggingRunRequest:
    batch_id: str
    config_path: str | Path
    samples: list[Sample]
    evidence_mode: str | None = None
    work_root: str | Path | None = None
    resume_run_id: str | None = None
    rule_ids: list[str] | None = None
    sample_ids: list[str] | None = None
    preview: bool = False
    publish: bool = True
    output_manifest: str | Path | None = None
    summary_xlsx: str | Path | None = None
    overlap_tags: list[str] | None = None
    acoustic_adapter: AcousticAdapter | None = None
    force_recompute: bool = False


@dataclass
class TaggingRunResult:
    batch_id: str
    run_id: str
    snapshot_id: str | None
    preview: bool
    samples: list[Sample]
    results: list[SampleTaggingResult]
    stats: dict[str, Any]
    published: dict[str, Any] | None = None
    work_dir: str = ""
    messages: list[str] = field(default_factory=list)


class TaggingService:
    def __init__(self, config: TaggingConfig | None = None, *, config_path: str | Path | None = None):
        if config is None:
            if config_path is None:
                raise ValueError("TaggingService requires config or config_path")
            config = load_tagging_config(config_path)
        self.config = config
        self.config_path = str(config_path or config.source_path)

    @classmethod
    def from_path(cls, path: str | Path) -> TaggingService:
        return cls(config_path=path)

    def validate(self) -> dict[str, Any]:
        # load_tagging_config already validates; re-report summary.
        return {
            "ok": True,
            "policy_version": self.config.policy_version,
            "content_fingerprint": self.config.content_fingerprint,
            "tag_count": len(self.config.tags),
            "rule_count": len(self.config.rules),
            "enabled_rules": [r.rule_id for r in self.config.enabled_rules()],
            "lexicons": {k: v.as_dict() for k, v in self.config.lexicons.items()},
            "evidence_mode": self.config.evidence_mode,
        }

    def workspace(self, batch_id: str, *, resume_run_id: str | None = None, work_root: str | Path | None = None) -> RunWorkspace:
        return create_run_workspace(
            batch_id=batch_id,
            policy_version=self.config.policy_version,
            template=self.config.work_dir_template,
            work_root=work_root,
            resume_run_id=resume_run_id,
        )

    def _stores(self, ws: RunWorkspace) -> tuple[TagStateStore, ManualOpLog]:
        return TagStateStore(ws.state_path), ManualOpLog(ws.op_log_path)

    def run(self, request: TaggingRunRequest) -> TaggingRunResult:
        cfg = self.config
        if request.evidence_mode:
            # Shallow override without mutating shared config object unexpectedly.
            from dataclasses import replace

            cfg = replace(cfg, evidence_mode=str(request.evidence_mode))

        ws = self.workspace(
            request.batch_id,
            resume_run_id=request.resume_run_id,
            work_root=request.work_root,
        )
        store, _op_log = self._stores(ws)
        progress = load_progress(ws) if request.resume_run_id else None
        done_ids = set((progress or {}).get("completed_sample_ids") or [])

        samples = list(request.samples)
        if request.sample_ids:
            wanted = set(request.sample_ids)
            samples = [s for s in samples if s.id in wanted]

        rule_id_set = set(request.rule_ids) if request.rule_ids else None
        adapter = request.acoustic_adapter
        if adapter is None and cfg.evidence_mode == "compute_acoustic":
            # Default: passthrough that only uses existing quality unless hooks injected later.
            from audio_engine.core.tagging.adapters import QualityPassthroughAdapter

            adapter = QualityPassthroughAdapter()

        engine = MultiTagEngine(cfg, acoustic_adapter=adapter)
        results: list[SampleTaggingResult] = []
        out_samples: list[Sample] = []
        messages: list[str] = []

        for sample in samples:
            if sample.id in done_ids and not request.force_recompute and not request.preview:
                # Resume: keep prior labels if present; still include in output.
                out_samples.append(sample.model_copy(deep=True))
                continue
            prior = store.get(sample.id)
            result = engine.evaluate_sample(
                sample,
                prior_state=prior,
                snapshot_id="" if request.preview else ws.snapshot_id,
                rule_ids=rule_id_set,
                dry_run=request.preview,
            )
            results.append(result)
            updated = apply_result_to_sample(
                sample.model_copy(deep=True),
                result,
                policy_version=cfg.policy_version,
                snapshot_id="" if request.preview else ws.snapshot_id,
            )
            out_samples.append(updated)
            if not request.preview:
                prior.sources = [
                    TagSource(
                        tag_id=str(s.get("tag_id") or ""),
                        source_type=str(s.get("source_type") or ""),
                        rule_id=s.get("rule_id"),
                        rule_version=s.get("rule_version"),
                        snapshot_id=s.get("snapshot_id"),
                        stale=bool(s.get("stale")),
                        details=dict(s.get("details") or {}),
                    )
                    for s in result.sources
                ]
                if not prior.audio_sha256:
                    prior.audio_sha256 = str(sample.sha256 or "")
                store.put(prior)
                done_ids.add(sample.id)
                save_progress(
                    ws,
                    {
                        "completed_sample_ids": sorted(done_ids),
                        "run_id": ws.run_id,
                        "batch_id": request.batch_id,
                        "config_fingerprint": cfg.content_fingerprint,
                    },
                )

        published = None
        snapshot_id = None if request.preview else ws.snapshot_id
        if request.preview:
            messages.append("preview_only_no_publish")
            # Estimate acoustic needs
            planned = engine.stats.planned_calls
            messages.append(f"estimated_acoustic_samples={planned}")
        elif request.publish:
            # Write work artifacts then atomically publish.
            work_parquet = ws.work_dir / f"tagged_multi_tag_v1_{request.batch_id}.parquet"
            work_detail = ws.work_dir / f"rule_details_multi_tag_v1_{request.batch_id}.jsonl"
            work_xlsx = ws.work_dir / f"summary_multi_tag_v1_{request.batch_id}.xlsx"

            manifest = Manifest(samples=out_samples)
            # Prefer engine Manifest.save if available.
            if hasattr(manifest, "save"):
                manifest.save(work_parquet)
            else:
                import pandas as pd

                pd.DataFrame([s.to_flat_dict() for s in out_samples]).to_parquet(work_parquet, index=False)

            write_rule_details(
                work_detail,
                results,
                batch_id=request.batch_id,
                snapshot_id=ws.snapshot_id,
                policy_version=cfg.policy_version,
            )
            write_summary_xlsx(
                work_xlsx,
                out_samples,
                results,
                batch_id=request.batch_id,
                overlap_tags=request.overlap_tags,
            )

            published = atomic_publish(
                ws,
                artifacts={
                    "parquet": work_parquet,
                    "rule_details": work_detail,
                    "summary_xlsx": work_xlsx,
                },
                meta={
                    "config_path": self.config_path,
                    "content_fingerprint": cfg.content_fingerprint,
                    "stats": engine.stats.as_dict(),
                    "sample_count": len(out_samples),
                },
            )
            # Also copy to declared production paths if provided.
            if request.output_manifest:
                dest = Path(request.output_manifest)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(work_parquet, dest)
            if request.summary_xlsx:
                dest = Path(request.summary_xlsx)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(work_xlsx, dest)
            # Mark progress complete
            save_progress(
                ws,
                {
                    "completed_sample_ids": sorted(done_ids),
                    "run_id": ws.run_id,
                    "batch_id": request.batch_id,
                    "published_snapshot_id": ws.snapshot_id,
                    "status": "published",
                },
            )

        return TaggingRunResult(
            batch_id=request.batch_id,
            run_id=ws.run_id,
            snapshot_id=snapshot_id,
            preview=request.preview,
            samples=out_samples,
            results=results,
            stats=engine.stats.as_dict(),
            published=published,
            work_dir=str(ws.work_dir.as_posix()),
            messages=messages,
        )

    def explain(self, sample: Sample, *, rule_id: str | None = None) -> dict[str, Any]:
        engine = MultiTagEngine(self.config)
        result = engine.evaluate_sample(sample, dry_run=True)
        payload = result.as_dict()
        if rule_id:
            payload["rule_results"] = [
                r for r in payload["rule_results"] if r["rule_id"] == rule_id
            ]
        return payload

    def _mutate(
        self,
        *,
        batch_id: str,
        sample_id: str,
        tag_id: str,
        audio_sha256: str,
        actor: str,
        reason: str,
        op: str,
        work_root: str | Path | None = None,
    ) -> dict[str, Any]:
        ws = self.workspace(batch_id, work_root=work_root)
        # Reuse existing run folder root; ensure dirs.
        ws.ensure()
        store, op_log = self._stores(ws)
        state = store.get(sample_id)
        before = state.as_dict()
        expected = state.version
        if op == "add":
            state = apply_manual_add(state, tag_id=tag_id, audio_sha256=audio_sha256)
        elif op == "remove":
            state = apply_manual_remove(state, tag_id=tag_id, audio_sha256=audio_sha256)
        elif op == "restore":
            state = apply_restore_auto(state, tag_id=tag_id, audio_sha256=audio_sha256)
        elif op == "retract_manual":
            state = apply_retract_manual(state, tag_id=tag_id, audio_sha256=audio_sha256)
        else:
            raise ValueError(f"unknown op {op}")
        state = store.put(state, expected_version=expected)
        after = state.as_dict()
        record = ManualOpRecord(
            op_id=new_op_id(),
            op=op,
            sample_id=sample_id,
            tag_id=tag_id,
            batch_id=batch_id,
            audio_sha256=audio_sha256,
            actor=actor,
            reason=reason,
            ts=now_ts(),
            before=before,
            after=after,
        )
        op_log.append(record)
        enabled = {tid for tid, t in self.config.tags.items() if t.enabled and not t.archived}
        return {
            "ok": True,
            "op": op,
            "sample_id": sample_id,
            "tag_id": tag_id,
            "effective_tags": effective_tags(state, enabled_tag_ids=enabled),
            "sources": [s.as_dict() for s in state.sources],
            "op_id": record.op_id,
        }

    def add_tag(self, **kwargs: Any) -> dict[str, Any]:
        return self._mutate(op="add", **kwargs)

    def remove_tag(self, **kwargs: Any) -> dict[str, Any]:
        return self._mutate(op="remove", **kwargs)

    def restore_tag(self, **kwargs: Any) -> dict[str, Any]:
        return self._mutate(op="restore", **kwargs)

    def retract_manual(self, **kwargs: Any) -> dict[str, Any]:
        return self._mutate(op="retract_manual", **kwargs)

    def export_filtered(
        self,
        samples: list[Sample],
        *,
        output: str | Path,
        include_all: list[str] | None = None,
        include_any: list[str] | None = None,
        exclude_tags: list[str] | None = None,
        require_unknown: bool = False,
        require_manual_block: bool = False,
    ) -> dict[str, Any]:
        filtered = filter_samples(
            samples,
            include_all=include_all,
            include_any=include_any,
            exclude_tags=exclude_tags,
            require_unknown=require_unknown,
            require_manual_block=require_manual_block,
        )
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        manifest = Manifest(samples=filtered)
        if hasattr(manifest, "save"):
            manifest.save(out)
        else:
            import pandas as pd

            pd.DataFrame([s.to_flat_dict() for s in filtered]).to_parquet(out, index=False)
        return {"ok": True, "count": len(filtered), "output": str(out.as_posix())}
