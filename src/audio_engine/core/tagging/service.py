"""Unified multi-tag service used by pipeline operator and CLI (040 §12.2)."""

from __future__ import annotations

import shutil
import time
from collections.abc import Sequence
from contextlib import ExitStack, closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from audio_engine.core.artifacts import atomic_write_json
from audio_engine.core.checkpoint import digest_payload
from audio_engine.core.manifest import Manifest
from audio_engine.core.sample import Sample
from audio_engine.core.tagging.adapters import AcousticAdapter
from audio_engine.core.tagging.archive import ArchivedResults, ResultArchive, RunLease
from audio_engine.core.tagging.config import TaggingConfig, load_tagging_config
from audio_engine.core.tagging.engine import MultiTagEngine, SampleTaggingResult, result_from_dict
from audio_engine.core.tagging.execution import TaggingExecution, evaluate_samples
from audio_engine.core.tagging.export import (
    apply_result_to_sample,
    filter_samples,
    sort_samples_by_dnsmos_sig,
    write_rule_details,
    write_summary_xlsx,
    write_tagged_parquet,
)
from audio_engine.core.tagging.library import (
    assert_resume_compatible,
    save_definition_snapshot,
)
from audio_engine.core.tagging.monitoring import TaggingMonitor
from audio_engine.core.tagging.persist import (
    RunWorkspace,
    atomic_publish,
    create_run_workspace,
    load_progress,
    load_sample_result,
    save_progress,
    save_sample_result,
)
from audio_engine.core.tagging.scheduling import AcousticScheduler, resolve_engine_limits
from audio_engine.core.tagging.state import (
    ManualOpLog,
    ManualOpRecord,
    SampleTagState,
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

_PROGRESS_EVERY = 100
_PROGRESS_EVERY_SECONDS = 15.0


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
    workers: int | None = None
    executor: str = "auto"
    max_in_flight: int | None = None
    checkpoint_every: int = 100
    batch_size: int = 32
    engine_workers: dict[str, int] = field(default_factory=dict)
    engine_max_in_flight: dict[str, int] = field(default_factory=dict)
    engine_retries: int = 0
    retain_results: bool = True
    pipeline_run_dir: str | Path | None = None
    step_name: str = "multi_tag"
    pipeline_resume: bool = False
    input_fingerprint: str = ""


@dataclass
class TaggingRunResult:
    batch_id: str
    run_id: str
    snapshot_id: str | None
    preview: bool
    samples: list[Sample]
    results: Sequence[SampleTaggingResult]
    stats: dict[str, Any]
    published: dict[str, Any] | None = None
    work_dir: str = ""
    messages: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    run_dir: str = ""


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
            "library_mode": self.config.library_mode,
            "library_path": self.config.library_path,
            "select_tags": list(self.config.select_tags),
            "tag_count": len(self.config.tags),
            "rule_count": len(self.config.rules),
            "enabled_rules": [r.rule_id for r in self.config.enabled_rules()],
            "tags": {tid: t.as_dict() for tid, t in self.config.tags.items()},
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
        import re
        from dataclasses import replace
        step = re.sub(r"[^\w-]", "_", request.step_name)[:64] or "multi_tag"
        step += "_" + digest_payload(request.step_name)[:8]
        directory = Path(request.pipeline_run_dir) / "tagging" / step if request.pipeline_run_dir else None
        if directory:
            import hashlib
            import json
            from audio_engine.core.checkpoint import digest_samples
            raw_digest = hashlib.sha256()
            for sample in request.samples:
                raw_digest.update(json.dumps(
                    [(key, value.get("raw_text")) for key, value in sorted(sample.transcripts.items())
                     if isinstance(value, dict)], ensure_ascii=False, default=str).encode("utf-8"))
                raw_digest.update(b"\n")
            request = replace(request, input_fingerprint=digest_payload({
                "input": request.input_fingerprint or digest_samples(request.samples),
                "raw_text_fields": raw_digest.hexdigest(),
            }))
        identity = digest_payload({"definition": self.config.content_fingerprint,
                                   "batch_id": request.batch_id,
                                   "input": request.input_fingerprint,
                                   "evidence_mode": request.evidence_mode or self.config.evidence_mode,
                                   "rule_ids": sorted(request.rule_ids or []),
                                   "sample_ids": sorted(request.sample_ids or []), "preview": request.preview,
                                   "work_root": str(request.work_root or "")})
        if directory and request.pipeline_resume and not request.resume_run_id:
            import json
            prior_meta = directory / "run.json"
            if prior_meta.exists():
                prior = json.loads(prior_meta.read_text(encoding="utf-8"))
                if prior.get("identity") == identity and not request.preview:
                    request = replace(request, resume_run_id=prior["tagging_run_id"])
        ws = self.workspace(request.batch_id, resume_run_id=request.resume_run_id,
                            work_root=request.work_root)
        with RunLease(ws.work_dir):
            return self._run_with_context(request, ws, directory, identity)

    def _run_with_context(self, request, ws, directory, identity):
        import os
        directory = directory or ws.work_dir
        directory.mkdir(parents=True, exist_ok=True)
        log_path = directory / "tagging.log.jsonl"
        metadata = {"tagging_run_id": ws.run_id, "batch_id": request.batch_id, "pid": os.getpid(),
                    "pipeline_run_id": Path(request.pipeline_run_dir).name if request.pipeline_run_dir else None,
                    "step_name": request.step_name, "identity": identity,
                    "work_dir": str(ws.work_dir.resolve()), "log_path": str(log_path.resolve()),
                    "checkpoint_path": str((ws.work_dir / "progress.json").resolve()),
                    "definition_snapshot": str((ws.work_dir / "definition_snapshot.json").resolve()),
                    "config_path": self.config_path, "definition_fingerprint": self.config.content_fingerprint,
                    "input_fingerprint": request.input_fingerprint,
                    "evidence_mode": request.evidence_mode or self.config.evidence_mode,
                    "preview": request.preview, "force_recompute": request.force_recompute,
                    "rule_ids": request.rule_ids,
                    "sample_filter_count": len(request.sample_ids) if request.sample_ids else None}
        if directory != ws.work_dir:
            atomic_write_json(ws.work_dir / "pipeline_run.json", {
                **metadata, "run_dir": str(directory.resolve()), "metrics_path": str((directory / "metrics.json").resolve())})
        monitor = TaggingMonitor(directory, metadata, mirrors=[ws.work_dir] if directory != ws.work_dir else [])
        sink = logger.add(log_path, serialize=True, level="TRACE", rotation="50 MB",
                          filter=lambda record: record["extra"].get("tagging_run_id") == ws.run_id)
        try:
            with logger.contextualize(tagging_run_id=ws.run_id, batch_id=request.batch_id,
                                      pipeline_run_id=metadata["pipeline_run_id"], step_name=request.step_name):
                try:
                    monitor.start()
                    with ExitStack() as resources:
                        result = self._run(request, ws, monitor, resources)
                        monitor.update(stage="finalizing")
                    monitor.finish("completed", snapshot_id=result.snapshot_id,
                                   published=result.published, artifacts={"work_dir": result.work_dir})
                    result.metrics = monitor.snapshot()
                    result.run_dir = str(directory.resolve())
                    return result
                except BaseException as exc:
                    logger.exception("tagging run failed batch={} run_id={} log={}",
                                     request.batch_id, ws.run_id, log_path)
                    monitor.finish("interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                                   error=f"{type(exc).__name__}: {exc}")
                    raise
        finally:
            logger.remove(sink)

    def _run(self, request: TaggingRunRequest, ws: RunWorkspace,
             monitor: TaggingMonitor, resources: ExitStack) -> TaggingRunResult:
        cfg = self.config
        if request.evidence_mode:
            # Shallow override without mutating shared config object unexpectedly.
            from dataclasses import replace

            cfg = replace(cfg, evidence_mode=str(request.evidence_mode))

        store, _op_log = self._stores(ws)
        resources.callback(store.close)
        progress = load_progress(ws) if request.resume_run_id else None
        if request.resume_run_id:
            assert_resume_compatible(progress, cfg)
        done_ids = set((progress or {}).get("completed_sample_ids") or [])

        # Persist full effective definitions (not merely config path / hash).
        if not request.preview:
            save_definition_snapshot(ws.work_dir / "definition_snapshot.json", cfg)

        samples = list(request.samples)
        if request.sample_ids:
            wanted = set(request.sample_ids)
            samples = [s for s in samples if s.id in wanted]

        if len({s.id for s in samples}) != len(samples):
            raise ValueError("tagging requires unique sample IDs")
        if not isinstance(request.checkpoint_every, int) or isinstance(request.checkpoint_every, bool) or request.checkpoint_every < 1:
            raise ValueError("tagging checkpoint_every must be >= 1")
        engine_workers, engine_limits = resolve_engine_limits(
            request.engine_workers, request.engine_max_in_flight, request.engine_retries)
        execution = TaggingExecution.resolve(
            request.workers, request.executor, request.max_in_flight,
            compute=cfg.evidence_mode == "compute_acoustic", total=len(samples),
            batch_size=request.batch_size,
        )
        archive = ResultArchive(ws.work_dir)
        ws.result_archive = archive
        resources.callback(lambda: setattr(ws, "result_archive", None))
        resources.callback(archive.close)
        exported_snapshot = False
        if not request.preview:
            store.enable_indexed()
            resources.callback(lambda: None if exported_snapshot else store.enable_indexed().export_snapshot())
        monitor.update(stage="evaluating", total=len(samples))
        with monitor.lock:
            monitor.metadata.update(execution={"executor": execution.executor, "workers": execution.workers,
                                               "max_in_flight": execution.max_in_flight,
                                               "batch_size": execution.batch_size},
                                    engine_workers=engine_workers,
                                    engine_max_in_flight=engine_limits,
                                    engine_retries=request.engine_retries)
            if execution.executor == "process":
                monitor.metadata["text_running_observable"] = False
        monitor.persist()
        if request.force_recompute and request.resume_run_id and not request.preview:
            # Old completion markers must not authorize snapshots overwritten by
            # an interrupted, not-yet-committed recompute batch.
            done_ids.difference_update(sample.id for sample in samples)
            archive.invalidate(sample.id for sample in samples)
            save_progress(ws, {
                "completed_sample_ids": sorted(done_ids),
                "run_id": ws.run_id, "batch_id": request.batch_id,
                "config_fingerprint": cfg.content_fingerprint,
                "definition_fingerprint": cfg.content_fingerprint,
            })

        rule_id_set = set(request.rule_ids) if request.rule_ids else None
        adapter = request.acoustic_adapter
        if adapter is None and cfg.evidence_mode == "compute_acoustic":
            adapter = AcousticScheduler(workers=request.engine_workers,
                                        max_in_flight=request.engine_max_in_flight,
                                        retries=request.engine_retries, monitor=monitor)
            resources.callback(adapter.close)
            with monitor.lock:
                monitor.metadata.update(engine_workers=adapter.workers, engine_max_in_flight=adapter.limits)
        retain_results = request.retain_results or request.preview
        engine = MultiTagEngine(cfg, acoustic_adapter=adapter)
        results: list[SampleTaggingResult] = []
        out_samples: list[Sample] = []
        messages: list[str] = []
        total = len(samples)
        started = time.perf_counter()
        last_progress_at = started
        processed = 0
        restored = 0
        logger.info(
            "tagging run start batch={} run_id={} samples={} evidence_mode={} preview={} resume={} "
            "executor={} workers={} max_in_flight={} checkpoint_every={} batch_size={} log={}",
            request.batch_id,
            ws.run_id,
            total,
            cfg.evidence_mode,
            request.preview,
            bool(request.resume_run_id),
            execution.executor, execution.workers, execution.max_in_flight,
            request.checkpoint_every, execution.batch_size, monitor.directory / "tagging.log.jsonl",
        )
        if adapter is not None and execution.executor == "thread" and not getattr(adapter, "thread_safe", False):
            logger.info("tagging injected adapter compute calls serialized for session safety")

        def _emit_progress(*, force: bool = False) -> None:
            nonlocal last_progress_at
            monitor.update(processed=processed, restored=restored)
            now = time.perf_counter()
            if not force and not (
                processed == total
                or processed % _PROGRESS_EVERY == 0
                or (now - last_progress_at) >= _PROGRESS_EVERY_SECONDS
            ):
                return
            elapsed = max(now - started, 1e-6)
            rate = processed / elapsed
            eta = (total - processed) / rate if rate > 0 else 0.0
            logger.info(
                "tagging progress {}/{} ({:.1f}%) restored={} rate={:.1f}/s eta={:.0f}s batch={} run_id={}",
                processed,
                total,
                100.0 * processed / total if total else 100.0,
                restored,
                rate,
                eta,
                request.batch_id,
                ws.run_id,
            )
            last_progress_at = now
            monitor.persist()

        pending_states: list[SampleTagState] = []
        output_indices: list[int] = []
        checkpoint_failed = False
        monitor.update(committed=len(done_ids))

        def _flush_checkpoint() -> None:
            nonlocal checkpoint_failed
            if request.preview or not pending_states or checkpoint_failed:
                return
            checkpoint_started = time.perf_counter()
            monitor.engine("checkpoint", submitted=1, running=1)
            try:
                archive.flush()
                archive.flush_legacy()
                store.put_many(pending_states)
                archive.commit(state.sample_id for state in pending_states)
                # Results and state must be durable before marking samples completed.
                save_progress(ws, {
                    "completed_sample_ids": sorted(done_ids) if len(done_ids) <= 4096 else [],
                    "completed_count": len(done_ids),
                    "completed_store": "sample_results.sqlite3" if len(done_ids) > 4096 else None,
                    "run_id": ws.run_id, "batch_id": request.batch_id,
                    "config_fingerprint": cfg.content_fingerprint,
                    "definition_fingerprint": cfg.content_fingerprint,
                })
                logger.debug("tagging checkpoint run_id={} committed={} completed={}",
                             ws.run_id, len(pending_states), len(done_ids))
                pending_states.clear()
                monitor.update(committed=len(done_ids))
            except BaseException:
                checkpoint_failed = True
                raise
            finally:
                monitor.finish_engine("checkpoint", time.perf_counter() - checkpoint_started,
                                      failed=checkpoint_failed)

        def _is_restored(sample: Sample) -> bool:
            return sample.id in done_ids and not request.force_recompute and not request.preview

        def _jobs():
            for offset in range(0, len(samples), max(1000, execution.batch_size)):
                block = samples[offset:offset + max(1000, execution.batch_size)]
                states = store.get_many(s.id for s in block)
                for index, sample in enumerate(block, offset):
                    if not _is_restored(sample):
                        prior = states.get(sample.id) or SampleTagState(sample_id=sample.id)
                        if prior.audio_sha256 and sample.sha256 and prior.audio_sha256 != sample.sha256:
                            raise ValueError(f"refuse unsafe resume for sample {sample.id}: audio identity drifted")
                        yield index, sample, prior

        def _completions():
            for index, sample in enumerate(samples):
                if _is_restored(sample):
                    yield index, sample, store.get(sample.id), None
            for index, sample, prior, result, stats in evaluate_samples(
                iter(_jobs()), config=cfg, execution=execution, adapter=adapter,
                snapshot_id="" if request.preview else ws.snapshot_id,
                rule_ids=rule_id_set, preview=request.preview, monitor=monitor,
            ):
                for name in ("planned_calls", "actual_calls", "reused", "failed"):
                    target = getattr(engine.stats, name)
                    for dep, count in getattr(stats, name).items():
                        target[dep] = target.get(dep, 0) + count
                for dep, count in stats.reused.items():
                    if count:
                        monitor.engine(dep, reused=count)
                for phase in ("asr_text", "rules"):
                    monitor.engine(phase, submitted=1, completed=1,
                                   elapsed_seconds=stats.timings.get(phase, 0.0))
                yield index, sample, prior, result

        try:
            with closing(_completions()) as completions:
                for index, sample, prior, evaluated_result in completions:
                    output_indices.append(index)
                    audio_sha = str(sample.sha256 or "")
                    if prior.audio_sha256 and audio_sha and prior.audio_sha256 != audio_sha:
                        raise ValueError(
                            f"refuse unsafe resume for sample {sample.id}: "
                            f"stored audio_sha256={prior.audio_sha256} != current={audio_sha}"
                        )
                    if (
                        sample.id in done_ids
                        and not request.force_recompute
                        and not request.preview
                    ):
                        cached = load_sample_result(ws, sample.id)
                        if not cached:
                            raise ValueError(
                                f"resume incomplete for sample {sample.id}: marked completed but "
                                "sample_results snapshot missing; refuse to copy raw input as output"
                            )
                        if cached.get("audio_sha256") and audio_sha and cached["audio_sha256"] != audio_sha:
                            raise ValueError(
                                f"refuse unsafe resume for sample {sample.id}: "
                                "cached result audio identity drifted"
                            )
                        if cached.get("asr_fingerprint") and cached.get("input_asr_fingerprint"):
                            # Prefer engine rebuild of ASR fingerprint when available later.
                            pass
                        updated = sample.model_copy(deep=True)
                        labels = dict(updated.labels or {})
                        labels["tags"] = list(cached.get("effective_tags") or [])
                        labels["tagging"] = dict(cached.get("tagging") or {})
                        # Merge manual-effective tags from current state store (may have drifted ops).
                        from audio_engine.core.tagging.state import effective_tags as _eff

                        enabled = set(cfg.tags)
                        labels["tags"] = _eff(prior, enabled_tag_ids=enabled)
                        labels.setdefault("tagging", {})
                        labels["tagging"]["eval_status"] = cached.get("eval_status") or labels["tagging"].get(
                            "eval_status"
                        )
                        labels["tagging"]["restored_from"] = "sample_results"
                        labels["tagging"]["snapshot_id"] = cached.get("snapshot_id") or labels["tagging"].get(
                            "snapshot_id"
                        )
                        updated.labels = labels
                        out_samples.append(updated)
                        if cached.get("quality"):
                            updated.quality = dict(cached["quality"])
                        if retain_results:
                            results.append(result_from_dict({**cached, "effective_tags": labels["tags"]}))
                        messages.append(f"restored_sample:{sample.id}")
                        restored += 1
                        processed += 1
                        _emit_progress()
                        continue
                    result = evaluated_result
                    if retain_results:
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
                        prior.audio_sha256 = audio_sha or prior.audio_sha256
                        save_sample_result(
                            ws,
                            sample.id,
                            {
                                "sample_id": sample.id,
                                "audio_sha256": audio_sha,
                                "quality": dict(updated.quality or {}),
                                "acoustic_plan": result.acoustic_plan.as_dict() if result.acoustic_plan else None,
                                "asr_fingerprint": result.asr_fingerprint,
                                "effective_tags": list(result.effective_tags),
                                "eval_status": result.eval_status,
                                "untagged_reason": result.untagged_reason,
                                "sources": list(result.sources),
                                "rule_results": [rr.as_dict() for rr in result.rule_results],
                                "tagging": dict((updated.labels or {}).get("tagging") or {}),
                                "snapshot_id": ws.snapshot_id,
                                "definition_fingerprint": cfg.content_fingerprint,
                            },
                        )
                        done_ids.add(sample.id)
                        pending_states.append(prior)
                        if len(pending_states) >= request.checkpoint_every:
                            _flush_checkpoint()
                    logger.trace("tagging sample done run_id={} sample_id={} status={} tags={}",
                                 ws.run_id, sample.id, result.eval_status, result.effective_tags)
                    if result.acoustic_plan:
                        for dep in result.acoustic_plan.required:
                            error = (sample.quality or {}).get(f"{dep}_error")
                            if (sample.quality or {}).get(f"{dep}_status") == "failed":
                                logger.warning("tagging acoustic failed run_id={} sample_id={} dep={} error={}",
                                               ws.run_id, sample.id, dep, error or "backend_reported_failed")
                    processed += 1
                    _emit_progress()

        finally:
            _flush_checkpoint()
        if not request.preview:
            monitor.update(stage="checkpointing")
            archive.flush_legacy()
            store.enable_indexed().export_snapshot()
            exported_snapshot = True
        # Completion order drives checkpoints; exported results retain input order.
        order = sorted(range(len(output_indices)), key=output_indices.__getitem__)
        out_samples = [out_samples[i] for i in order]
        results = [results[i] for i in order] if retain_results else ArchivedResults(ws, out_samples)
        _emit_progress(force=True)
        logger.info(
            "tagging samples done batch={} processed={}/{} restored={} elapsed={:.1f}s",
            request.batch_id,
            processed,
            total,
            restored,
            time.perf_counter() - started,
        )

        published = None
        snapshot_id = None if request.preview else ws.snapshot_id
        if request.preview:
            messages.append("preview_only_no_publish")
            # Estimate acoustic needs
            planned = engine.stats.planned_calls
            messages.append(f"estimated_acoustic_samples={planned}")
        elif request.publish:
            monitor.update(stage="exporting")
            monitor.persist(report=True)
            # Write work artifacts then atomically publish.
            work_parquet = ws.work_dir / f"tagged_multi_tag_v1_{request.batch_id}.parquet"
            work_detail = ws.work_dir / f"rule_details_multi_tag_v1_{request.batch_id}.jsonl"
            work_xlsx = ws.work_dir / f"summary_multi_tag_v1_{request.batch_id}.xlsx"
            work_def_snap = ws.work_dir / "definition_snapshot.json"
            if not work_def_snap.exists():
                save_definition_snapshot(work_def_snap, cfg)

            write_tagged_parquet(work_parquet, out_samples)

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
                ordered=True,
            )

            monitor.update(stage="publishing")
            monitor.persist(report=True)
            published = atomic_publish(
                ws,
                artifacts={
                    "parquet": work_parquet,
                    "rule_details": work_detail,
                    "summary_xlsx": work_xlsx,
                    "definition_snapshot": work_def_snap,
                },
                meta={
                    "config_path": self.config_path,
                    "content_fingerprint": cfg.content_fingerprint,
                    "definition_fingerprint": cfg.content_fingerprint,
                    "library_mode": cfg.library_mode,
                    "library_path": cfg.library_path,
                    "select_tags": list(cfg.select_tags),
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
                    "completed_sample_ids": sorted(done_ids) if len(done_ids) <= 4096 else [],
                    "completed_count": len(done_ids),
                    "completed_store": "sample_results.sqlite3" if len(done_ids) > 4096 else None,
                    "run_id": ws.run_id,
                    "batch_id": request.batch_id,
                    "published_snapshot_id": ws.snapshot_id,
                    "config_fingerprint": cfg.content_fingerprint,
                    "definition_fingerprint": cfg.content_fingerprint,
                    "status": "published",
                },
            )

        logger.info("tagging run complete batch={} run_id={} snapshot_id={} published={} elapsed={:.2f}s stats={}",
                    request.batch_id, ws.run_id, snapshot_id, bool(published),
                    time.perf_counter() - started, engine.stats.as_dict())
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
        expected_version: int | None = None,
    ) -> dict[str, Any]:
        ws = self.workspace(batch_id, work_root=work_root)
        # Reuse existing run folder root; ensure dirs.
        ws.ensure()
        store, op_log = self._stores(ws)
        try:
            state = store.get(sample_id)
            before = state.as_dict()
            if expected_version is not None and int(state.version) != int(expected_version):
                from audio_engine.core.tagging.state import ConcurrentModificationError

                raise ConcurrentModificationError(
                    f"sample {sample_id} version conflict: "
                    f"expected {expected_version}, got {state.version}"
                )
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
                "revision": str(state.version),
                "version": state.version,
                "work_root": str(ws.root.as_posix()),
                "published": False,
                "state_kind": "working",
            }
        finally:
            store.close()

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
        sort_by_dnsmos_sig: bool = False,
    ) -> dict[str, Any]:
        filtered = filter_samples(
            samples,
            include_all=include_all,
            include_any=include_any,
            exclude_tags=exclude_tags,
            require_unknown=require_unknown,
            require_manual_block=require_manual_block,
        )
        if sort_by_dnsmos_sig:
            filtered = sort_samples_by_dnsmos_sig(filtered)
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        manifest = Manifest(samples=filtered)
        if hasattr(manifest, "save"):
            manifest.save(out)
        else:
            import pandas as pd

            pd.DataFrame([s.to_flat_dict() for s in filtered]).to_parquet(out, index=False)
        return {"ok": True, "count": len(filtered), "output": str(out.as_posix())}
