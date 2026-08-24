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

from audio_engine.core.manifest import Manifest
from audio_engine.core.pipeline import PipelineConfig, PipelineRunner, ShardingConfig


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
) -> dict[str, int]:
    """Run one child ``pipeline run --no-sharding`` per shard parquet."""
    emit = log or (lambda msg: logger.info(msg))
    parallel = sharding.effective_parallel
    workers = workers_override if workers_override is not None else sharding.workers
    executor = executor_override if executor_override is not None else sharding.executor
    gpu_ids = list(sharding.gpus)

    emit(f"Running {len(shards)} shards, {parallel} at a time")
    if gpu_ids:
        emit(
            f"  GPUs: {','.join(gpu_ids)}  instances-per-gpu: {sharding.instances_per_gpu}"
        )
    if ingest_step_names:
        emit(f"  Dropping ingest steps: {', '.join(ingest_step_names)}")

    queue = list(shards)
    running: list[tuple[Path, subprocess.Popen, object, str | None]] = []
    exit_codes: dict[str, int] = {}

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
        raise ValueError(
            f"Expected {expected_shards} shard outputs, found {len(paths)}"
        )
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
            "Sharded run needs the pipeline YAML path "
            "(PipelineConfig.config_path or config_path=)"
        )
    if not cfg.output_manifest:
        raise ValueError("Sharded run requires output.manifest")

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
    emit(f"Merged {report['inputs']} shards → {out} ({report['total_out']} samples)")

    return ShardedRunResult(
        manifest=merged,
        run_root=root,
        shard_count=len(ok_stems),
        merge_report=report,
    )
