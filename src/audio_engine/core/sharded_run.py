"""Split → parallel shard pipelines → merge, driven by PipelineConfig.sharding."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Callable

from loguru import logger
import yaml

from audio_engine.core.manifest import Manifest
from audio_engine.core.catalog import register_manifest_output
from audio_engine.core.pipeline import PipelineConfig, PipelineRunner, ShardingConfig
from audio_engine.core.progress import (
    collect_sharded_progress,
    emit_progress,
    resolve_asr_batch_size,
    shard_input_count,
    sharding_config_summary,
)


class ShardedRunError(RuntimeError):
    """Raised when one or more shard child processes fail."""

    def __init__(self, message: str, *, failed: list[str], run_root: Path):
        super().__init__(message)
        self.failed = failed
        self.run_root = run_root


@dataclass
class ShardedRunResult:
    manifest: Manifest
    run_root: Path
    shard_count: int
    merge_report: dict


@dataclass
class StagedRunResult:
    """Result of running ``stages:`` sequentially (full model A, then full model B)."""

    run_root: Path
    stage_roots: list[Path]
    final_manifest: str | None


def load_stage_paths(config_path: Path) -> list[Path] | None:
    """Return absolute stage YAML paths when the root config defines ``stages:``."""
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    raw = data.get("stages")
    if not raw:
        return None
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{config_path}: stages must be a non-empty list of YAML paths")
    base = config_path.parent
    paths: list[Path] = []
    for item in raw:
        text = str(item).strip()
        if not text:
            raise ValueError(f"{config_path}: empty stages entry")
        candidate = Path(text)
        if not candidate.is_absolute():
            # Prefer paths relative to CWD (repo-root style: pipelines/foo.yaml),
            # then fall back to relative-to-orchestrator-dir.
            cwd_candidate = Path.cwd() / candidate
            local_candidate = base / candidate
            if cwd_candidate.is_file():
                candidate = cwd_candidate
            elif local_candidate.is_file():
                candidate = local_candidate
            else:
                candidate = cwd_candidate
        paths.append(candidate.resolve())
    return paths


def resolve_staged_run_root(
    name: str,
    runs_dir: Path,
    resume: str | Path | None = None,
) -> Path:
    if resume is not None:
        candidate = Path(resume)
        if not candidate.exists():
            candidate = runs_dir / str(resume)
        if not candidate.is_dir():
            raise FileNotFoundError(f"Cannot resume staged run: {resume}")
        return candidate
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = runs_dir / f"{ts}_{name}_stages"
    root.mkdir(parents=True, exist_ok=True)
    return root


def run_staged_pipelines(
    stage_paths: list[Path],
    *,
    name: str,
    runs_dir: Path = Path("runs"),
    resume: str | Path | None = None,
    force: bool = False,
    mock: bool = False,
    execution_override: dict | None = None,
    source_name: str | None = None,
    source_name_layout: list[dict] | None = None,
    log: Callable[[str], None] | None = None,
) -> StagedRunResult:
    """Run each stage YAML to completion (including its own sharding) before the next.

    This is how 「先全量 Qwen，再全量 SenseVoice」 is expressed as one CLI command.

    When ``source_name`` is set, each stage's input/output manifests are rewritten
    from ``source_name_layout`` (or the multi-ASR default layout).
    """
    from audio_engine.core.source_naming import (
        expand_layout_templates,
        resolve_existing_manifest,
        validate_source_name,
    )

    emit = log or (lambda msg: logger.info(msg))
    if not stage_paths:
        raise ValueError("run_staged_pipelines requires at least one stage")

    path_overrides: list[tuple[str, str]] | None = None
    if source_name is not None:
        validate_source_name(source_name)
        path_overrides = expand_layout_templates(source_name_layout, source_name)
        if len(path_overrides) != len(stage_paths):
            raise ValueError(
                f"--source-name layout has {len(path_overrides)} entries but "
                f"stages has {len(stage_paths)}; they must match"
            )
        emit(f"Source name: {source_name}")
        for index, (inp, out) in enumerate(path_overrides, start=1):
            emit(f"  Stage {index} paths: {inp} → {out}")

    root = resolve_staged_run_root(name, runs_dir, resume)
    emit(f"Staged run root: {root}")
    stage_roots: list[Path] = []
    final_manifest: str | None = None

    for index, stage_path in enumerate(stage_paths, start=1):
        if not stage_path.is_file():
            raise FileNotFoundError(f"Stage YAML not found: {stage_path}")
        cfg = PipelineConfig.from_yaml(stage_path)
        cfg.force = force or cfg.force
        cfg.mock = mock or cfg.mock
        if execution_override:
            cfg.execution_override = dict(execution_override)

        if path_overrides is not None:
            inp_stem, out_path = path_overrides[index - 1]
            resolved_in = resolve_existing_manifest(inp_stem)
            cfg.input_manifest = str(resolved_in)
            cfg.output_manifest = out_path
            cfg.source_dir = None
            cfg.source_id = None

        stage_root = root / f"stage-{index:02d}_{cfg.name}"
        stage_root.mkdir(parents=True, exist_ok=True)
        stage_roots.append(stage_root)
        emit(f"=== Stage {index}/{len(stage_paths)}: {cfg.name} ({stage_path.name}) ===")
        emit(f"  input:  {cfg.input_manifest}")
        emit(f"  output: {cfg.output_manifest}")

        if cfg.sharding is not None:
            # Resume stage subdir when it already has shard outputs / checkpoints.
            resume_stage = stage_root if any(stage_root.iterdir()) else None
            result = run_sharded_pipeline(
                cfg,
                config_path=stage_path,
                run_root=resume_stage or stage_root,
                log=emit,
            )
            emit(
                f"Stage {index} done: {result.shard_count} shards → {cfg.output_manifest} "
                f"({result.merge_report.get('total_out', len(result.manifest))} samples)"
            )
        else:
            cfg.run_dir = str(stage_root)
            runner = PipelineRunner(cfg)
            manifest = runner.run()
            if cfg.output_manifest:
                out = Path(cfg.output_manifest)
                out.parent.mkdir(parents=True, exist_ok=True)
                manifest.save(out)
                manifest.save(out.with_suffix(".jsonl"))
                artifact = register_manifest_output(
                    out,
                    catalog_dir=cfg.catalog_dir,
                    pipeline=cfg.name,
                    run_dir=stage_root,
                    config_digest=cfg.digest(),
                    sample_count=len(manifest),
                )
                emit(f"Stage {index} done: {len(manifest)} samples → {out}")
                emit(f"Registered artifact: {artifact.artifact_id}")
            else:
                emit(f"Stage {index} done: {len(manifest)} samples")

        final_manifest = cfg.output_manifest

    return StagedRunResult(
        run_root=root,
        stage_roots=stage_roots,
        final_manifest=final_manifest,
    )


def resolve_run_root(cfg: PipelineConfig, run_root: Path | None = None) -> Path:
    if run_root is not None:
        root = Path(run_root)
    elif cfg.resume:
        candidate = Path(cfg.resume)
        if not candidate.exists():
            candidate = cfg.runs_dir / cfg.resume
        if not candidate.is_dir():
            raise FileNotFoundError(f"Cannot resume sharded run: {cfg.resume}")
        root = candidate
    elif cfg.run_dir:
        root = Path(cfg.run_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        root = cfg.runs_dir / f"{ts}_{cfg.name}_shards"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _try_load_input_manifest(cfg: PipelineConfig) -> Manifest | None:
    if not cfg.input_manifest:
        return None
    candidates = [Path(cfg.input_manifest)]
    try:
        candidates.append(Manifest.resolve_path(cfg.input_manifest))
    except FileNotFoundError:
        pass
    for path in candidates:
        if path.is_file():
            return Manifest.load(path)
    return None


def prepare_full_manifest(cfg: PipelineConfig, run_root: Path) -> Manifest:
    """Load or ingest the full sample set before splitting."""
    ingest_steps = [s for s in cfg.steps if s.operator.startswith("ingest.")]
    manifest = _try_load_input_manifest(cfg)
    if manifest is None:
        manifest = Manifest([])

    if ingest_steps and (len(manifest) == 0 or cfg.source_dir):
        prepare_dir = run_root / "prepare"
        prepare_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Sharded prepare: running {} ingest step(s) on full set under {}",
            len(ingest_steps),
            prepare_dir,
        )
        prepare_cfg = replace(
            cfg,
            steps=list(ingest_steps),
            sharding=None,
            resume=str(prepare_dir),
            output_manifest=None,
            config_path=None,
        )
        manifest = PipelineRunner(prepare_cfg).run()

    if cfg.filter_expr:
        manifest = manifest.filter(cfg.filter_expr)

    if len(manifest) == 0:
        raise ValueError(
            "Sharded pipeline has 0 samples after prepare; "
            "check input.manifest / input.source_dir / ingest steps"
        )
    return manifest


def write_shards(
    manifest: Manifest,
    run_root: Path,
    sharding: ShardingConfig,
) -> list[Path]:
    shard_dir = run_root / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    parts = manifest.split(sharding.shards, strategy=sharding.strategy)
    paths: list[Path] = []
    for idx, part in enumerate(parts):
        if len(part) == 0:
            logger.warning("shard-{:03d}: empty, skipped", idx)
            continue
        out = shard_dir / f"shard-{idx:03d}.parquet"
        part.save(out)
        paths.append(out)
        logger.info("Wrote {} ({} samples)", out.name, len(part))
    if not paths:
        raise ValueError("All shards were empty after split")
    return paths


def _build_shard_command(
    *,
    config_path: Path,
    shard: Path,
    output: Path,
    shard_run_dir: Path,
    ingest_step_names: list[str],
    workers: int | None,
    executor: str | None,
    force: bool,
    mock: bool,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "audio_engine.cli.main",
        "pipeline",
        "run",
        str(config_path),
        "--input-manifest",
        str(shard),
        "--output-manifest",
        str(output),
        "--resume",
        str(shard_run_dir),
        "--no-sharding",
    ]
    for name in ingest_step_names:
        command += ["--skip-step", name]
    if workers is not None:
        command += ["--workers", str(workers)]
    if executor is not None:
        command += ["--executor", executor]
    if force:
        command.append("--force")
    if mock:
        command.append("--mock")
    return command


def spawn_shard_processes(
    *,
    config_path: Path,
    shards: list[Path],
    run_root: Path,
    sharding: ShardingConfig,
    ingest_step_names: list[str],
    force: bool = False,
    mock: bool = False,
    workers_override: int | None = None,
    executor_override: str | None = None,
    log: Callable[[str], None] | None = None,
    progress_config: dict | None = None,
    progress_interval_s: float = 10.0,
) -> dict[str, int]:
    """Run one child ``pipeline run --no-sharding`` per shard parquet.

    While shards run, periodically emit aggregate ``[PROGRESS]`` lines to the
    parent log and ``run_root/PROGRESS.log`` (done count + throughput).
    """
    emit = log or (lambda msg: logger.info(msg))
    parallel = sharding.effective_parallel
    workers = workers_override if workers_override is not None else sharding.workers
    executor = executor_override if executor_override is not None else sharding.executor
    gpu_ids = list(sharding.gpus)

    shard_totals = {shard.stem: shard_input_count(shard) for shard in shards}
    total_samples = sum(shard_totals.values())
    emit(f"Running {len(shards)} shards, {parallel} at a time ({total_samples} samples)")
    if gpu_ids:
        emit(f"  GPUs: {','.join(gpu_ids)}  instances-per-gpu: {sharding.instances_per_gpu}")
    if ingest_step_names:
        emit(f"  Dropping ingest steps: {', '.join(ingest_step_names)}")
    emit(f"  Progress log: {run_root / 'PROGRESS.log'}")

    queue = list(shards)
    running: list[tuple[Path, subprocess.Popen, object, str | None]] = []
    exit_codes: dict[str, int] = {}
    started_at = time.time()
    prev_done = 0
    prev_at = started_at
    last_progress_at = 0.0

    def _emit_progress(*, force_emit: bool = False) -> None:
        nonlocal prev_done, prev_at, last_progress_at
        now = time.time()
        if not force_emit and (now - last_progress_at) < progress_interval_s:
            return
        running_stems = {item[0].stem for item in running}
        finished_stems = set(exit_codes)
        queued_stems = {item.stem for item in queue}
        snapshot = collect_sharded_progress(
            run_root=run_root,
            shard_totals=shard_totals,
            running_stems=running_stems,
            finished_stems=finished_stems,
            queued_stems=queued_stems,
            started_at=started_at,
            prev_done=prev_done,
            prev_at=prev_at,
            config=progress_config,
        )
        emit_progress(snapshot, run_root=run_root, log=emit)
        prev_done = snapshot.done
        prev_at = snapshot.updated_at
        last_progress_at = now

    while queue or running:
        while queue and len(running) < parallel:
            gpu_id = None
            if gpu_ids:
                used: dict[str, int] = {}
                for _shard, _proc, _log, assigned in running:
                    if assigned is not None:
                        used[assigned] = used.get(assigned, 0) + 1
                gpu_id = next(
                    (item for item in gpu_ids if used.get(item, 0) < sharding.instances_per_gpu),
                    None,
                )
                if gpu_id is None:
                    break
            shard = queue.pop(0)
            shard_run_dir = run_root / shard.stem
            shard_run_dir.mkdir(parents=True, exist_ok=True)
            output = run_root / f"{shard.stem}.parquet"
            command = _build_shard_command(
                config_path=config_path,
                shard=shard,
                output=output,
                shard_run_dir=shard_run_dir,
                ingest_step_names=ingest_step_names,
                workers=workers,
                executor=executor,
                force=force,
                mock=mock,
            )
            log_file = (run_root / f"{shard.stem}.log").open("w", encoding="utf-8")
            env = None
            if gpu_id is not None:
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = gpu_id
            process = subprocess.Popen(
                command,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=env,
            )
            running.append((shard, process, log_file, gpu_id))
            gpu_label = f", GPU {gpu_id}" if gpu_id is not None else ""
            emit(f"  started {shard.name} (pid {process.pid}{gpu_label})")

        _emit_progress()
        time.sleep(0.2)
        for entry in list(running):
            shard, process, log_file, _gpu_id = entry
            if process.poll() is None:
                continue
            log_file.close()
            running.remove(entry)
            exit_codes[shard.stem] = process.returncode
            status = "ok" if process.returncode == 0 else "FAILED"
            emit(f"  {status} {shard.name} (exit {process.returncode})")
            _emit_progress(force_emit=True)

    _emit_progress(force_emit=True)
    return exit_codes


def merge_shard_outputs(
    run_root: Path,
    shard_stems: list[str],
    *,
    expected_shards: int | None = None,
) -> tuple[Manifest, dict]:
    paths = [run_root / f"{stem}.parquet" for stem in shard_stems]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing shard outputs: {missing}")
    if expected_shards is not None and len(paths) != expected_shards:
        raise ValueError(f"Expected {expected_shards} shard outputs, found {len(paths)}")
    manifests = [Manifest.load(p) for p in paths]
    return Manifest.merge(manifests)


def run_sharded_pipeline(
    cfg: PipelineConfig,
    *,
    config_path: Path | None = None,
    run_root: Path | None = None,
    shard_dir: Path | None = None,
    sharding_override: ShardingConfig | None = None,
    log: Callable[[str], None] | None = None,
) -> ShardedRunResult:
    """Execute prepare (optional) → split → parallel children → merge."""
    emit = log or (lambda msg: logger.info(msg))
    sharding = sharding_override or cfg.sharding
    if sharding is None:
        raise ValueError("run_sharded_pipeline requires PipelineConfig.sharding")
    sharding.validate_parallel()

    path = Path(config_path or cfg.config_path or "")
    if not path.is_file():
        raise ValueError(
            "Sharded run needs the pipeline YAML path (PipelineConfig.config_path or config_path=)"
        )
    if not cfg.output_manifest:
        raise ValueError("Sharded run requires output.manifest")

    for step in cfg.steps:
        if step.operator != "asr.qwen_batch":
            continue
        named_env = str(step.params.get("api_base_env") or "")
        api_base = (
            step.params.get("api_base")
            or os.environ.get(named_env)
            or os.environ.get("QWEN_ASR_API_BASE")
        )
        backend = "vllm" if api_base else "local"
        emit(
            f"Qwen preflight: backend={backend} "
            f"api_base={api_base or '<none>'} batch_mode=vllm-only"
        )
        if not api_base and not cfg.mock:
            raise ValueError(
                "Qwen batch inference only supports vLLM, but "
                "QWEN_ASR_API_BASE/api_base is empty in the parent process"
            )

    root = resolve_run_root(cfg, run_root)
    emit(f"Sharded run root: {root}")

    if shard_dir is not None:
        shards = sorted(Path(shard_dir).glob("shard-*.parquet"))
        if not shards:
            raise FileNotFoundError(f"No shard-*.parquet found in {shard_dir}")
    else:
        full = prepare_full_manifest(cfg, root)
        prepared_path = root / "prepared_input.parquet"
        full.save(prepared_path)
        emit(f"Prepared {len(full)} samples → {prepared_path}")
        shards = write_shards(full, root, sharding)

    ingest_names = sorted({s.name for s in cfg.steps if s.operator.startswith("ingest.")})
    progress_config = sharding_config_summary(
        sharding,
        pipeline=cfg.name,
        batch_size=resolve_asr_batch_size(cfg),
        checkpoint_every=cfg.execution.checkpoint_every,
    )
    exit_codes = spawn_shard_processes(
        config_path=path,
        shards=shards,
        run_root=root,
        sharding=sharding,
        ingest_step_names=ingest_names,
        force=cfg.force,
        mock=cfg.mock,
        workers_override=cfg.execution_override.get("workers"),
        executor_override=cfg.execution_override.get("executor"),
        log=emit,
        progress_config=progress_config,
    )

    failed = [name for name, code in exit_codes.items() if code != 0]
    if failed:
        raise ShardedRunError(
            f"{len(failed)} shard(s) failed: {', '.join(sorted(failed))}",
            failed=failed,
            run_root=root,
        )

    ok_stems = [name for name, code in exit_codes.items() if code == 0]
    # Preserve shard-000 ordering for merge determinism when stems sort well.
    ok_stems = sorted(ok_stems)
    merged, report = merge_shard_outputs(root, ok_stems, expected_shards=len(ok_stems))

    out = Path(cfg.output_manifest)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.save(out)
    merged.save(out.with_suffix(".jsonl"))
    artifact = register_manifest_output(
        out,
        catalog_dir=cfg.catalog_dir,
        pipeline=cfg.name,
        run_dir=root,
        config_digest=cfg.digest(),
        sample_count=len(merged),
    )
    emit(f"Merged {report['inputs']} shards → {out} ({report['total_out']} samples)")
    emit(f"Registered artifact: {artifact.artifact_id}")

    return ShardedRunResult(
        manifest=merged,
        run_root=root,
        shard_count=len(ok_stems),
        merge_report=report,
    )
