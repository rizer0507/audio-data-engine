from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
import yaml
from rich.console import Console
from rich.table import Table

import audio_engine.operators  # noqa: F401 — register all operators
from audio_engine.core.manifest import Manifest
from audio_engine.core.pipeline import (
    EXECUTORS,
    ExecutionConfig,
    PipelineConfig,
    PipelineRunner,
    PipelineStep,
    ShardingConfig,
)
from audio_engine.core.registry import OperatorRegistry
from audio_engine.core.sharded_run import ShardedRunError, run_sharded_pipeline
from audio_engine.core.transcript_reconcile import reconcile_transcripts

app = typer.Typer(
    name="audio-data",
    help="Manifest-driven audio data processing engine",
    no_args_is_help=True,
)
console = Console()

MANIFESTS_DIR = Path("datasets/manifests")
EXPORTS_DIR = Path("data/exports")
RUNS_DIR = Path("runs")


def _resolve_dataset(name: str) -> Path:
    return Manifest.resolve_path(name, MANIFESTS_DIR)


def _execution_override(
    workers: Optional[int],
    executor: Optional[str],
    max_in_flight: Optional[int] = None,
    checkpoint_every: Optional[int] = None,
) -> dict:
    """CLI flags win over pipeline and step YAML values; unset flags change nothing."""
    override: dict = {}
    if workers is not None:
        override["workers"] = workers
    if executor is not None:
        override["executor"] = executor
    if max_in_flight is not None:
        override["max_in_flight"] = max_in_flight
    if checkpoint_every is not None:
        override["checkpoint_every"] = checkpoint_every
    try:
        ExecutionConfig().merged(override)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    return override


def _resolve_operator(name: str) -> str:
    """Allow shorthand names like qwen_asr -> asr.qwen."""
    aliases = {
        "scan": "ingest.scan",
        "qwen_asr": "asr.qwen",
        "qwen": "asr.qwen",
        "sensevoice": "asr.sensevoice",
        "pcm_to_wav": "audio.pcm_to_wav",
        "resample": "audio.resample",
        "denoise": "audio.denoise",
        "vad": "audio.vad",
        "snr": "quality.snr",
        "cer": "quality.cer",
        "filter": "quality.filter",
        "transcript_diff": "quality.transcript_diff",
        "probe": "quality.probe",
        "select": "quality.select",
        "add_noise": "augmentation.add_noise",
        "speed_perturb": "augmentation.speed_perturb",
        "volume_perturb": "augmentation.volume_perturb",
    }
    if name in OperatorRegistry.list_operators():
        return name
    if name in aliases:
        return aliases[name]
    candidate = f"asr.{name}"
    if candidate in OperatorRegistry.list_operators():
        return candidate
    raise typer.BadParameter(
        f"Unknown operator '{name}'. Available: {', '.join(OperatorRegistry.list_operators())}"
    )


@app.command("ingest")
def ingest(
    source: Path = typer.Argument(..., help="Directory containing raw audio files"),
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Dataset name"),
    copy_to_raw: bool = typer.Option(False, "--copy", help="Copy files to data/raw/"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output manifest path"),
) -> None:
    """Ingest audio files via the ingest.scan operator (unified pipeline path)."""
    if not source.is_dir():
        raise typer.BadParameter(f"Not a directory: {source}")

    dataset_name = name or f"raw_{datetime.now().strftime('%Y%m%d')}"
    params: dict = {"source_dir": str(source)}
    if copy_to_raw:
        params["copy_to"] = "data/raw"

    cfg = PipelineConfig(
        name=f"ingest_{dataset_name}",
        input_manifest="",
        source_dir=str(source),
        steps=[PipelineStep(name="ingest", operator="ingest.scan", params=params)],
    )
    runner = PipelineRunner(cfg)
    result = runner.run()

    out_path = output or (MANIFESTS_DIR / f"{dataset_name}.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(out_path)
    result.save(out_path.with_suffix(".jsonl"))

    stats = result.stats()
    console.print(f"[green]OK[/green] {stats['files']} files discovered")
    console.print(f"Manifest created: [cyan]{out_path}[/cyan]")
    console.print(f"  Run dir: [cyan]{runner.run_dir}[/cyan]")


@app.command("stats")
def stats(
    dataset: str = typer.Argument(..., help="Dataset name or manifest path"),
) -> None:
    """Show dataset statistics."""
    path = _resolve_dataset(dataset)
    manifest = Manifest.load(path)
    s = manifest.stats()

    table = Table(title=f"Dataset: {dataset}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Files", str(s.get("files", 0)))
    table.add_row("Duration (hours)", str(s.get("duration_hours", 0)))
    table.add_row("Broken", str(s.get("broken", 0)))
    for sr, count in (s.get("sample_rates") or {}).items():
        table.add_row(f"Sample rate {sr}", str(count))
    for fmt, count in (s.get("formats") or {}).items():
        table.add_row(fmt.upper(), str(count))
    console.print(table)


@app.command("run")
def run_operator(
    operator: str = typer.Argument(..., help="Operator name, e.g. sensevoice or asr.qwen"),
    dataset: str = typer.Option(..., "--dataset", "-d", help="Input dataset name"),
    filter_expr: Optional[str] = typer.Option(None, "--filter", "-f", help="Pandas query filter"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Operator config YAML"),
    force: bool = typer.Option(False, "--force", help="Ignore cache and re-run"),
    mock: bool = typer.Option(False, "--mock", help="Use mock ASR output"),
    input_audio_key: str = typer.Option("raw", "--input-key", help="Input audio artifact key"),
    workers: Optional[int] = typer.Option(None, "--workers", "-w", help="Concurrent workers"),
    executor: Optional[str] = typer.Option(
        None, "--executor", help=f"Executor: {', '.join(EXECUTORS)}"
    ),
) -> None:
    """Run a single operator over a dataset."""
    op_name = _resolve_operator(operator)
    manifest_path = _resolve_dataset(dataset)
    manifest = Manifest.load(manifest_path)
    if filter_expr:
        manifest = manifest.filter(filter_expr)

    params: dict = {"input_audio_key": input_audio_key}
    if config and config.exists():
        params.update(yaml.safe_load(config.read_text(encoding="utf-8")) or {})

    pipeline_cfg = PipelineConfig(
        name=f"run_{op_name.replace('.', '_')}",
        input_manifest=str(manifest_path),
        steps=[],
        force=force,
        mock=mock,
        filter_expr=filter_expr,
        execution_override=_execution_override(workers, executor),
    )
    runner = PipelineRunner(pipeline_cfg)
    result = runner.run_single_operator(op_name, manifest, params)

    out_path = runner.run_dir / "manifest.parquet"
    result.save(out_path)
    result.save(out_path.with_suffix(".jsonl"))

    # Update source manifest
    source_out = manifest_path
    result.save(source_out)
    result.save(source_out.with_suffix(".jsonl"))

    m = runner.metrics.to_dict()
    console.print(f"[green]OK[/green] Operator '{op_name}' finished")
    console.print(f"  Processed: {m['processed']}, Cache hits: {m['cache_hits']}, Skipped: {m['skipped']}")
    console.print(f"  Run dir: [cyan]{runner.run_dir}[/cyan]")


pipeline_app = typer.Typer(help="Pipeline commands")
app.add_typer(pipeline_app, name="pipeline")


@pipeline_app.command("run")
def pipeline_run(
    config_path: Path = typer.Argument(..., help="Pipeline YAML file"),
    force: bool = typer.Option(False, "--force", help="Ignore cache"),
    mock: bool = typer.Option(False, "--mock", help="Use mock ASR"),
    workers: Optional[int] = typer.Option(
        None, "--workers", "-w", help="Override per-step concurrency (default from YAML)"
    ),
    executor: Optional[str] = typer.Option(
        None, "--executor", help=f"Override executor: {', '.join(EXECUTORS)}"
    ),
    max_in_flight: Optional[int] = typer.Option(
        None, "--max-in-flight", help="Override queued+running tasks per step"
    ),
    checkpoint_every: Optional[int] = typer.Option(
        None, "--checkpoint-every", help="Samples per checkpoint part (0 disables checkpointing)"
    ),
    resume: Optional[str] = typer.Option(
        None, "--resume", help="Continue an existing run dir (path or run id under runs/)"
    ),
    input_manifest: Optional[Path] = typer.Option(
        None, "--input-manifest", help="Override input.manifest (e.g. run one shard)"
    ),
    output_manifest: Optional[Path] = typer.Option(
        None, "--output-manifest", help="Override output.manifest"
    ),
    runs_dir: Optional[Path] = typer.Option(
        None, "--runs-dir", help="Override run directory root (isolate parallel runs)"
    ),
    skip_step: Optional[list[str]] = typer.Option(
        None, "--skip-step", help="Skip a step by name or operator (repeatable)"
    ),
    no_sharding: bool = typer.Option(
        False,
        "--no-sharding",
        help="Ignore YAML sharding and run as a single process (used by shard workers)",
    ),
) -> None:
    """Run a pipeline from YAML configuration.

    When the YAML defines ``sharding:``, this command automatically splits the
    input, runs shard workers in parallel, and merges into ``output.manifest``.
    """
    cfg = PipelineConfig.from_yaml(config_path)
    cfg.force = force or cfg.force
    cfg.mock = mock or cfg.mock
    cfg.execution_override = _execution_override(
        workers, executor, max_in_flight, checkpoint_every
    )
    cfg.resume = resume or cfg.resume

    if input_manifest is not None:
        cfg.input_manifest = str(input_manifest)
        cfg.source_dir = None  # an explicit manifest replaces directory scanning
    if output_manifest is not None:
        cfg.output_manifest = str(output_manifest)
    if runs_dir is not None:
        cfg.runs_dir = runs_dir
    if skip_step:
        skipped = set(skip_step)
        kept = [s for s in cfg.steps if s.name not in skipped and s.operator not in skipped]
        if len(kept) != len(cfg.steps):
            console.print(f"  Skipping steps: {', '.join(sorted(skipped))}")
        cfg.steps = kept
    if no_sharding:
        cfg.sharding = None

    if cfg.sharding is not None:
        try:
            result = run_sharded_pipeline(
                cfg,
                config_path=config_path.resolve(),
                run_root=Path(cfg.resume) if cfg.resume else None,
                log=lambda msg: console.print(msg),
            )
        except ShardedRunError as exc:
            console.print(f"[red]{exc}[/red]")
            console.print(f"  Logs: [cyan]{exc.run_root}[/cyan]")
            raise typer.Exit(code=1) from exc
        except (ValueError, FileNotFoundError) as exc:
            raise typer.BadParameter(str(exc)) from exc

        report = result.merge_report
        console.print(f"[green]OK[/green] Sharded pipeline '{cfg.name}' finished")
        console.print(
            f"  Shards: {result.shard_count}, Samples: {report.get('total_out', len(result.manifest))}"
        )
        console.print(f"  Output: [cyan]{cfg.output_manifest}[/cyan]")
        console.print(f"  Run root: [cyan]{result.run_root}[/cyan]")
        return

    runner = PipelineRunner(cfg)
    result = runner.run()

    if cfg.output_manifest:
        out = Path(cfg.output_manifest)
        out.parent.mkdir(parents=True, exist_ok=True)
        result.save(out)
        result.save(out.with_suffix(".jsonl"))
        console.print(f"  Output: [cyan]{out}[/cyan]")
    elif cfg.input_manifest and not Path(cfg.input_manifest).is_absolute():
        # Save back to input manifest only when no dedicated output is configured
        try:
            resolved = Manifest.resolve_path(cfg.input_manifest)
            result.save(resolved)
            result.save(resolved.with_suffix(".jsonl"))
        except FileNotFoundError:
            pass

    m = runner.metrics.to_dict()
    console.print(f"[green]OK[/green] Pipeline '{cfg.name}' finished")
    console.print(f"  Total: {m['total']}, Processed: {m['processed']}, Failed: {m['failed']}")
    console.print(f"  Run dir: [cyan]{runner.run_dir}[/cyan]")


@pipeline_app.command("clean-temp")
def pipeline_clean_temp(
    config_path: Path = typer.Argument(..., help="Pipeline YAML file"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Only list what would be removed"),
) -> None:
    """Remove orphaned `*.tmp*` files left behind when a run was hard-killed.

    Atomic writes clean up after themselves on exceptions, but a SIGKILL leaves
    the temp file in place. It is never read (only `<name>` is), so this is
    housekeeping rather than a correctness fix.
    """
    cfg = PipelineConfig.from_yaml(config_path)
    targets = [cfg.cache_dir, cfg.output_dir, cfg.runs_dir]
    removed = 0
    freed = 0
    for directory in targets:
        if not directory.exists():
            continue
        for leftover in directory.rglob("*.tmp*"):
            if not leftover.is_file():
                continue
            size = leftover.stat().st_size
            console.print(f"  {'would remove' if dry_run else 'removed'} {leftover} ({size} B)")
            if not dry_run:
                leftover.unlink(missing_ok=True)
            removed += 1
            freed += size

    verb = "would remove" if dry_run else "removed"
    console.print(f"[green]OK[/green] {verb} {removed} temp file(s), {freed} bytes")


@pipeline_app.command("run-shards")
def pipeline_run_shards(
    config_path: Path = typer.Argument(..., help="Pipeline YAML file"),
    shard_dir: Path = typer.Option(..., "--shard-dir", help="Directory with shard-*.parquet"),
    parallel_shards: int = typer.Option(
        2, "--parallel-shards", "-p", help="Shard pipelines running at the same time"
    ),
    workers: Optional[int] = typer.Option(
        None, "--workers", "-w", help="Workers per shard pipeline"
    ),
    executor: Optional[str] = typer.Option(
        None, "--executor", help=f"Override executor: {', '.join(EXECUTORS)}"
    ),
    gpus: Optional[str] = typer.Option(
        None,
        "--gpus",
        help="Comma-separated GPU ids assigned round-robin to shard processes (e.g. 0,1,2,3)",
    ),
    instances_per_gpu: int = typer.Option(
        1,
        "--instances-per-gpu",
        help="Max concurrent shard processes sharing one GPU (A800 80G ASR can use 2)",
    ),
    force: bool = typer.Option(False, "--force", help="Ignore cache"),
    run_root: Optional[Path] = typer.Option(None, "--run-root", help="Output root for this batch"),
) -> None:
    """Run one independent pipeline process per pre-built shard (compat path).

    Prefer ``pipeline run`` with YAML ``sharding:`` for split+run+merge in one command.
    Ingest steps are dropped: the shard manifest already lists the samples.
    """
    cfg = PipelineConfig.from_yaml(config_path)
    cfg.force = force or cfg.force
    if workers is not None or executor is not None:
        cfg.execution_override = _execution_override(workers, executor)

    gpu_ids = tuple(item.strip() for item in (gpus or "").split(",") if item.strip())
    try:
        override = ShardingConfig(
            shards=max(parallel_shards, 1),
            strategy="hash",
            parallel_shards=parallel_shards,
            gpus=gpu_ids,
            instances_per_gpu=instances_per_gpu,
            workers=workers,
            executor=executor,
        )
        override.validate_parallel()
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    # Compat: ensure an output path so the shared runner can merge shard parquets.
    if not cfg.output_manifest:
        root_for_out = run_root or RUNS_DIR
        cfg.output_manifest = str(Path(root_for_out) / f"{cfg.name}_merged.parquet")
        auto_output = True
    else:
        auto_output = False

    root = run_root or (RUNS_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{cfg.name}_shards")
    try:
        result = run_sharded_pipeline(
            cfg,
            config_path=config_path.resolve(),
            run_root=root,
            shard_dir=shard_dir,
            sharding_override=override,
            log=lambda msg: console.print(msg),
        )
    except ShardedRunError as exc:
        console.print(f"[red]{exc}[/red]")
        console.print(f"  Logs: [cyan]{exc.run_root}[/cyan]")
        raise typer.Exit(code=1) from exc
    except (ValueError, FileNotFoundError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    console.print(f"[green]OK[/green] all {result.shard_count} shards finished")
    console.print(f"  Run root: [cyan]{result.run_root}[/cyan]")
    console.print(f"  Output: [cyan]{cfg.output_manifest}[/cyan]")
    if auto_output:
        console.print(
            "  (compat) also: [cyan]audio-data manifest merge "
            f'"{result.run_root}/shard-*.parquet" --output <path>[/cyan]'
        )


manifest_app = typer.Typer(help="Manifest sharding and merging for large-scale runs")
app.add_typer(manifest_app, name="manifest")


@manifest_app.command("shard")
def manifest_shard(
    dataset: str = typer.Argument(..., help="Dataset name or manifest path"),
    shards: int = typer.Option(..., "--shards", "-n", help="Number of shards"),
    output_dir: Path = typer.Option(..., "--output-dir", "-o", help="Where to write the shards"),
    strategy: str = typer.Option(
        "hash",
        "--strategy",
        help="hash (stable content buckets) or duration-balanced (even total duration)",
    ),
) -> None:
    """Split a manifest into shards (hash or duration-balanced)."""
    manifest = Manifest.load(_resolve_dataset(dataset))
    try:
        parts = manifest.split(shards, strategy=strategy)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for idx, part in enumerate(parts):
        if not len(part):
            console.print(f"  shard-{idx:03d}: empty, skipped")
            continue
        out = output_dir / f"shard-{idx:03d}.parquet"
        part.save(out)
        written += 1
        total_dur = sum(s.duration or 0.0 for s in part.samples)
        console.print(f"  {out.name}: {len(part)} samples, {total_dur:.1f}s audio")

    console.print(
        f"[green]OK[/green] {len(manifest)} samples -> {written} shards "
        f"({strategy}) in [cyan]{output_dir}[/cyan]"
    )


@manifest_app.command("merge")
def manifest_merge(
    inputs: list[str] = typer.Argument(..., help="Shard manifest paths or globs"),
    output: Path = typer.Option(..., "--output", "-o", help="Merged manifest path"),
    expected_shards: Optional[int] = typer.Option(
        None, "--expected-shards", help="Fail unless this many shard files are found"
    ),
    keep_duplicates: bool = typer.Option(
        False, "--keep-duplicates", help="Keep repeated (sha256, id) instead of dropping them"
    ),
) -> None:
    """Merge shard outputs with completeness, duplicate and count checks."""
    paths: list[Path] = []
    for pattern in inputs:
        candidate = Path(pattern)
        matched = (
            [candidate]
            if candidate.exists()
            else sorted(Path(candidate.parent or ".").glob(candidate.name))
        )
        if not matched:
            raise typer.BadParameter(f"No manifest matched: {pattern}")
        paths.extend(matched)

    if expected_shards is not None and len(paths) != expected_shards:
        raise typer.BadParameter(
            f"Expected {expected_shards} shard files, found {len(paths)}: "
            f"{[p.name for p in paths]}"
        )

    manifests = [Manifest.load(p) for p in paths]
    merged, report = Manifest.merge(manifests, keep_duplicates=keep_duplicates)

    output.parent.mkdir(parents=True, exist_ok=True)
    merged.save(output)
    merged.save(output.with_suffix(".jsonl"))

    console.print(f"[green]OK[/green] merged {report['inputs']} shard manifests")
    console.print(f"  In: {report['total_in']}, duplicates: {report['duplicates']}")
    console.print(f"  Out: {report['total_out']} -> [cyan]{output}[/cyan]")
    if report["duplicates"]:
        action = "kept" if keep_duplicates else "dropped"
        console.print(f"  [yellow]{report['duplicates']} duplicate (sha256, id) {action}[/yellow]")


@app.command("compare")
def compare(
    model_a: str = typer.Argument(..., help="First ASR model key, e.g. qwen or sensevoice"),
    model_b: str = typer.Argument(..., help="Second ASR model key"),
    dataset: str = typer.Option(..., "--dataset", "-d", help="Dataset with transcripts"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output run directory"),
) -> None:
    """Compare two ASR model transcripts and export badcases."""
    manifest_path = _resolve_dataset(dataset)
    manifest = Manifest.load(manifest_path)
    df = manifest.to_dataframe()

    key_a = model_a.split(".")[-1]
    key_b = model_b.split(".")[-1]
    col_a = f"{key_a}_text" if f"{key_a}_text" in df.columns else None
    col_b = f"{key_b}_text" if f"{key_b}_text" in df.columns else None

    if col_a is None:
        df[col_a := f"{key_a}_text"] = df["transcripts"].apply(
            lambda t: (t or {}).get(key_a, {}).get("text", "") if isinstance(t, dict) else ""
        )
    if col_b is None:
        df[col_b := f"{key_b}_text"] = df["transcripts"].apply(
            lambda t: (t or {}).get(key_b, {}).get("text", "") if isinstance(t, dict) else ""
        )

    df["match"] = df[col_a] == df[col_b]
    mismatches = df[~df["match"]]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output or (RUNS_DIR / f"{ts}_{key_a}_vs_{key_b}")
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest.save(run_dir / "manifest.parquet")
    summary = {
        "model_a": key_a,
        "model_b": key_b,
        "total": len(df),
        "match": int(df["match"].sum()),
        "mismatch": int(len(mismatches)),
        "match_rate": round(float(df["match"].mean()), 4) if len(df) else 0,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if len(mismatches):
        badcases_path = run_dir / "badcases.xlsx"
        mismatches[["id", "source_path", col_a, col_b]].to_excel(badcases_path, index=False)
        console.print(f"Badcases exported: [cyan]{badcases_path}[/cyan]")

    console.print(f"[green]OK[/green] Compare finished: {summary['match']}/{summary['total']} match")
    console.print(f"  Run dir: [cyan]{run_dir}[/cyan]")


@app.command("reconcile-transcripts")
def reconcile_transcript_files(
    xlsx: Path = typer.Option(..., "--xlsx", help="包含Qwen识别结果的Excel文件"),
    sensevoice_result: Path = typer.Option(
        ..., "--sensevoice-result", help="SenseVoice结果（fenp/JSONL/JSON/Parquet/Excel）"
    ),
    output: Path = typer.Option(..., "--output", "-o", help="清洗和比对后的Excel文件"),
    id_column: Optional[str] = typer.Option(None, "--id-column", help="Excel中的关联ID列"),
    sensevoice_id_column: Optional[str] = typer.Option(
        None, "--sensevoice-id-column", help="SenseVoice结果中的关联ID列"
    ),
    qwen_column: Optional[str] = typer.Option(None, "--qwen-column", help="Qwen文本列"),
    sensevoice_column: Optional[str] = typer.Option(
        None, "--sensevoice-column", help="SenseVoice文本列"
    ),
    threshold: float = typer.Option(
        0.9, "--threshold", min=0.0, max=1.0, help="可认为一致的最低字符相似度"
    ),
) -> None:
    """清洗SenseVoice控制字段，并与Qwen结果做字符级一致性比对。"""
    try:
        summary = reconcile_transcripts(
            xlsx,
            sensevoice_result,
            output,
            id_column=id_column,
            sensevoice_id_column=sensevoice_id_column,
            qwen_column=qwen_column,
            sensevoice_column=sensevoice_column,
            threshold=threshold,
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    console.print(f"[green]OK[/green] 转写清洗和比对完成: [cyan]{output}[/cyan]")
    console.print(
        f"  一致: {summary['consistent']}/{summary['total']} "
        f"({summary['consistent_rate']:.2%}), 缺失SenseVoice: {summary['missing_sensevoice']}"
    )
    console.print(f"  汇总: [cyan]{summary_path}[/cyan]")


@app.command("operators")
def list_operators() -> None:
    """List all registered operators."""
    for name in OperatorRegistry.list_operators():
        console.print(f"  - {name}")


@app.command("export")
def export_dataset(
    dataset: str = typer.Argument(..., help="Dataset name"),
    format: str = typer.Option("jsonl", "--format", "-f", help="Export format: jsonl, parquet, scp"),
    output: Optional[Path] = typer.Option(None, "--output", "-o"),
    filter_expr: Optional[str] = typer.Option(None, "--filter"),
) -> None:
    """Export dataset to training-ready formats."""
    manifest_path = _resolve_dataset(dataset)
    manifest = Manifest.load(manifest_path)
    if filter_expr:
        manifest = manifest.filter(filter_expr)

    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = output or (EXPORTS_DIR / f"{dataset}.{format}")

    if format == "jsonl":
        manifest.save(out)
    elif format == "parquet":
        manifest.save(out)
    elif format == "scp":
        lines = [f"{s.id} {s.audio_path()}" for s in manifest.samples]
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
        raise typer.BadParameter(f"Unsupported format: {format}")

    console.print(f"[green]OK[/green] Exported to [cyan]{out}[/cyan]")


if __name__ == "__main__":
    app()
