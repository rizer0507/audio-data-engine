from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
import yaml
from rich.console import Console
from rich.table import Table

from audio_engine.core.dotenv import load_dotenv
from audio_engine.core.manifest import Manifest
from audio_engine.core.operator import OperatorConfig
from audio_engine.core.catalog import (
    ArtifactCatalog,
    DatasetRelease,
    ModelVersion,
    current_git_commit,
    register_manifest_output,
)
from audio_engine.core.pipeline import (
    EXECUTORS,
    ExecutionConfig,
    PipelineConfig,
    PipelineRunner,
    PipelineStep,
    ShardingConfig,
)
from audio_engine.core.registry import OperatorRegistry
from audio_engine.core.sharded_run import (
    ShardedRunError,
    load_stage_paths,
    run_sharded_pipeline,
    run_staged_pipelines,
)
from audio_engine.core.source_naming import (
    apply_eval_name_to_single_pipeline,
    apply_source_name_to_single_pipeline,
    parse_join_manifest_arg,
    pipeline_run_name,
    validate_asr_run,
    validate_source_name,
)
from audio_engine.core.eval_ready import inspect_eval_manifest
from audio_engine.core.transcript_reconcile import reconcile_transcripts
from audio_engine.core.training import run_training_job
from audio_engine.core.task import TaskRunner, load_task

app = typer.Typer(
    name="audio-data",
    help="Manifest-driven audio data processing engine",
    no_args_is_help=True,
)
console = Console()

import audio_engine.operators  # noqa: F401, E402 — register all operators

MANIFESTS_DIR = Path("datasets/manifests")
EXPORTS_DIR = Path("data/exports")
RUNS_DIR = Path("runs")
CATALOG_DIR = Path("data/catalog")


def _register_output(
    path: Path,
    *,
    pipeline: str,
    run_dir: Path,
    catalog_dir: Path,
    sample_count: int,
    config_digest: str | None = None,
) -> str:
    record = register_manifest_output(
        path,
        catalog_dir=catalog_dir,
        pipeline=pipeline,
        run_dir=run_dir,
        config_digest=config_digest,
        sample_count=sample_count,
    )
    console.print(f"  Artifact:   [cyan]{record.artifact_id}[/cyan]")
    return record.artifact_id


def _review_queue_id(dataset_path: Path, buckets: list[str], revision: str) -> str:
    canonical_buckets = ",".join(sorted(set(buckets)))
    raw = f"{dataset_path.resolve()}\0{canonical_buckets}\0{revision}"
    return f"review_{hashlib.sha256(raw.encode()).hexdigest()[:16]}"


_DEFAULT_REVIEW_BUCKETS = [
    "qwen_missing",
    "hardcase",
    "hallucination",
    "semantic_inversion",
    "review_queue",
]


def _apply_aggregate_manifests(cfg: PipelineConfig, manifests: list[dict]) -> None:
    """Patch ``quality.aggregate_manifests`` step params with resolved join list."""
    updated = False
    for step in cfg.steps:
        if step.operator == "quality.aggregate_manifests":
            step.params = {**step.params, "manifests": list(manifests)}
            updated = True
    if not updated:
        raise typer.BadParameter(
            "--join-manifest / aggregate source-name rewrite requires a "
            "quality.aggregate_manifests step"
        )


def _apply_asr_run_transcript_key(cfg: PipelineConfig, asr_run: str) -> None:
    """Inject ``transcript_key`` into ASR steps for a result alias."""
    alias = validate_asr_run(asr_run)
    patched = False
    for step in cfg.steps:
        if str(step.operator).startswith("asr."):
            step.params = {**step.params, "transcript_key": alias}
            patched = True
    if not patched:
        raise typer.BadParameter(
            "--asr-run / --transcript-key requires an ASR inference pipeline "
            "with an asr.* step"
        )


def _first_asr_transcript_key(cfg: PipelineConfig) -> str | None:
    for step in cfg.steps:
        if str(step.operator).startswith("asr."):
            key = step.params.get("transcript_key")
            if key:
                return str(key)
    return None


def _apply_agreement_base(cfg: PipelineConfig, agreement_base: str) -> None:
    """Patch ``quality.text_metrics`` to use ``vs_base.base`` / agreement_base."""
    alias = validate_asr_run(agreement_base)
    patched = False
    for step in cfg.steps:
        if step.operator == "quality.text_metrics":
            step.params = {**step.params, "agreement_base": alias}
            patched = True
    if not patched:
        raise typer.BadParameter(
            "--agreement-base requires a quality.text_metrics step "
            "(e.g. asr_metric_pipeline)"
        )


def _apply_eval_models(cfg: PipelineConfig, models: list[str]) -> None:
    """Restrict eval hypotheses / report columns to the given transcript keys."""
    aliases = [validate_asr_run(item) for item in models]
    if not aliases:
        raise typer.BadParameter("--eval-model requires at least one non-empty alias")
    patched_metrics = False
    patched_report = False
    for step in cfg.steps:
        if step.operator == "quality.text_metrics":
            step.params = {**step.params, "eval_hypotheses": list(aliases)}
            patched_metrics = True
        if step.operator == "quality.evaluation_report":
            step.params = {**step.params, "model_prefixes": list(aliases)}
            patched_report = True
    if not patched_metrics and not patched_report:
        raise typer.BadParameter(
            "--eval-model requires quality.text_metrics and/or "
            "quality.evaluation_report (e.g. eval_metric_pipeline)"
        )


def _resolve_dataset(name: str) -> Path:
    if name.startswith("manifest_"):
        try:
            return Path(ArtifactCatalog(CATALOG_DIR).get(name, verify=True).uri)
        except (KeyError, FileNotFoundError, ValueError) as exc:
            raise typer.BadParameter(str(exc)) from exc
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
        "kimi_asr": "asr.kimi_batch",
        "kimi": "asr.kimi_batch",
        "kimi_audio_batch": "asr.kimi_audio_batch",
        "doubao_asr": "asr.doubao_batch",
        "doubao": "asr.doubao_batch",
        "doubao_single": "asr.doubao",
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
    _register_output(
        out_path,
        pipeline=cfg.name,
        run_dir=runner.run_dir,
        catalog_dir=cfg.catalog_dir,
        sample_count=len(result),
        config_digest=cfg.digest(),
    )

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
    _register_output(
        source_out,
        pipeline=pipeline_cfg.name,
        run_dir=runner.run_dir,
        catalog_dir=pipeline_cfg.catalog_dir,
        sample_count=len(result),
        config_digest=pipeline_cfg.digest(),
    )

    m = runner.metrics.to_dict()
    console.print(f"[green]OK[/green] Operator '{op_name}' finished")
    console.print(
        f"  Processed: {m['processed']}, Cache hits: {m['cache_hits']}, Skipped: {m['skipped']}"
    )
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
    source_name: Optional[str] = typer.Option(
        None,
        "--source-name",
        help=(
            "Batch tag for unified manifest naming, e.g. mt3000 → "
            "cleaned_mt3000 / qwen_asr_mt3000 / sensevoice_asr_mt3000 / "
            "multi_asr_aggregate_mt3000 / multi_asr_metrics_mt3000"
        ),
    ),
    source_dir: Optional[Path] = typer.Option(
        None,
        "--source-dir",
        help="Raw audio directory (required with --source-name for data-cleaning pipelines)",
    ),
    join_manifest: Optional[list[str]] = typer.Option(
        None,
        "--join-manifest",
        help=(
            "For multi_asr_aggregate / eval_aggregate: result alias to join, e.g. "
            "sensevoice1 or kimi=/path/to.parquet (repeatable). With "
            "--source-name / --eval-name, bare names resolve to "
            "{alias}_asr_<name>.parquet"
        ),
    ),
    asr_run: Optional[str] = typer.Option(
        None,
        "--asr-run",
        help=(
            "Result alias for ASR inference under --source-name or --eval-name: "
            "writes {alias}_asr_<name>.parquet and sets transcript_key=<alias> "
            "(e.g. qwen1, qwen-sft-epoch10)"
        ),
    ),
    transcript_key: Optional[str] = typer.Option(
        None,
        "--transcript-key",
        help=(
            "Inject transcript_key into asr.* steps without renaming manifests "
            "(used by shard workers; prefer --asr-run with --source-name)"
        ),
    ),
    aggregate_base: Optional[str] = typer.Option(
        None,
        "--aggregate-base",
        help=(
            "For multi_asr_aggregate: base result alias whose manifest is the "
            "join left table (default: qwen). Example: qwen1"
        ),
    ),
    agreement_base: Optional[str] = typer.Option(
        None,
        "--agreement-base",
        help=(
            "For asr_metric_pipeline: pseudo-reference transcript key for "
            "vs_base agreement CER (e.g. qwen1)"
        ),
    ),
    eval_name: Optional[str] = typer.Option(
        None,
        "--eval-name",
        help=(
            "Registered evaluation-set stem under datasets/manifests/ "
            "(e.g. eval_local_test). Decouples eval from training: ASR reads "
            "the eval set, eval_aggregate joins by id, eval_metric scores vs gold."
        ),
    ),
    eval_model: Optional[list[str]] = typer.Option(
        None,
        "--eval-model",
        help=(
            "For eval_metric_pipeline / asr_eval: transcript key to score vs "
            "gold_text (repeatable). Example: --eval-model qwen "
            "--eval-model qwen-sft-epoch10"
        ),
    ),
) -> None:
    """Run a pipeline from YAML configuration.

    When the YAML defines ``stages:``, runs each stage YAML sequentially
    under one command (legacy orchestrator).

    When the YAML defines ``sharding:``, splits the input, runs shard workers
    in parallel, and merges into ``output.manifest``.

    ``--source-name`` sets unified dataset names under datasets/manifests/.
    Cleaning also needs ``--source-dir``. ASR / aggregate / metric only need
    the name (each result alias writes ``{alias}_asr_<name>.parquet``).

    ``--eval-name`` is the evaluation-set counterpart (no training dependency):
    ASR reads the registered eval Manifest; aggregate joins by id; metric
    scores vs ``gold_text``.
    """
    if source_dir is not None and source_name is None:
        raise typer.BadParameter("--source-dir requires --source-name")
    if source_name is not None and eval_name is not None:
        raise typer.BadParameter("--source-name and --eval-name are mutually exclusive")
    if asr_run is not None and source_name is None and eval_name is None:
        raise typer.BadParameter("--asr-run requires --source-name or --eval-name")
    if aggregate_base is not None and source_name is None:
        raise typer.BadParameter("--aggregate-base requires --source-name")
    if aggregate_base is not None and eval_name is not None:
        raise typer.BadParameter(
            "--aggregate-base is for multi_asr_aggregate; "
            "eval_aggregate always uses the eval set as the left table"
        )
    if asr_run is not None and transcript_key is not None and asr_run != transcript_key:
        raise typer.BadParameter("--asr-run and --transcript-key must match when both set")

    stage_paths = load_stage_paths(config_path.resolve())
    if stage_paths is not None:
        if no_sharding or skip_step or input_manifest or output_manifest:
            raise typer.BadParameter(
                "staged pipelines do not support --no-sharding / --skip-step / "
                "--input-manifest / --output-manifest; pass those on each stage YAML run"
            )
        if source_dir is not None:
            raise typer.BadParameter(
                "staged pipelines do not use --source-dir; pass --source-name only "
                "(input is cleaned_<name>.parquet/.jsonl)"
            )
        if join_manifest:
            raise typer.BadParameter(
                "staged pipelines do not support --join-manifest; "
                "run multi_asr_aggregate.yaml as a single pipeline instead"
            )
        if asr_run or aggregate_base or agreement_base or transcript_key or eval_name or eval_model:
            raise typer.BadParameter(
                "staged pipelines do not support --asr-run / --transcript-key / "
                "--aggregate-base / --agreement-base / --eval-name / --eval-model; "
                "run each YAML as a single pipeline instead"
            )
        root_data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        stage_name = str(root_data.get("name") or config_path.stem)
        if source_name is not None:
            try:
                validate_source_name(source_name)
            except ValueError as exc:
                raise typer.BadParameter(str(exc)) from exc
            stage_name = pipeline_run_name(stage_name, source_name)
        stage_runs_dir = runs_dir or Path(root_data.get("runs_dir") or RUNS_DIR)
        layout = root_data.get("source_name_layout")
        try:
            staged = run_staged_pipelines(
                stage_paths,
                name=stage_name,
                runs_dir=stage_runs_dir,
                resume=resume,
                force=force,
                mock=mock,
                execution_override=_execution_override(
                    workers, executor, max_in_flight, checkpoint_every
                ),
                source_name=source_name,
                source_name_layout=layout,
                log=lambda msg: console.print(msg),
            )
        except ShardedRunError as exc:
            console.print(f"[red]{exc}[/red]")
            console.print(f"  Logs: [cyan]{exc.run_root}[/cyan]")
            raise typer.Exit(code=1) from exc
        except (ValueError, FileNotFoundError) as exc:
            raise typer.BadParameter(str(exc)) from exc

        console.print(f"[green]OK[/green] Staged pipeline '{stage_name}' finished")
        console.print(f"  Stages: {len(staged.stage_roots)}")
        if staged.final_manifest:
            console.print(f"  Output: [cyan]{staged.final_manifest}[/cyan]")
        console.print(f"  Run root: [cyan]{staged.run_root}[/cyan]")
        return

    cfg = PipelineConfig.from_yaml(config_path)
    cfg.force = force or cfg.force
    cfg.mock = mock or cfg.mock
    cfg.execution_override = _execution_override(workers, executor, max_in_flight, checkpoint_every)
    cfg.resume = resume or cfg.resume

    cli_joins: list[dict] | None = None
    if join_manifest:
        try:
            from audio_engine.core.source_naming import resolve_existing_manifest

            name_for_join = source_name or eval_name
            cli_joins = []
            for item in join_manifest:
                parsed = parse_join_manifest_arg(item, name_for_join)
                parsed["path"] = str(resolve_existing_manifest(parsed["path"]))
                cli_joins.append(parsed)
        except (ValueError, FileNotFoundError) as exc:
            raise typer.BadParameter(str(exc)) from exc

    if eval_name is not None:
        try:
            validate_source_name(eval_name)
            overrides = apply_eval_name_to_single_pipeline(
                pipeline_name=cfg.name,
                steps=cfg.steps,
                eval_name=eval_name,
                join_manifests=cli_joins,
                asr_run=asr_run,
            )
        except (ValueError, FileNotFoundError) as exc:
            raise typer.BadParameter(str(exc)) from exc
        cfg.source_dir = overrides["source_dir"]
        cfg.input_manifest = overrides["input_manifest"]
        cfg.source_id = overrides["source_id"]
        cfg.output_manifest = overrides["output_manifest"]
        if overrides.get("aggregate_manifests") is not None:
            _apply_aggregate_manifests(cfg, overrides["aggregate_manifests"])
        if overrides.get("asr_run"):
            _apply_asr_run_transcript_key(cfg, str(overrides["asr_run"]))
            cfg.name = pipeline_run_name(
                f"{cfg.name}_{overrides['asr_run']}", eval_name
            )
        else:
            cfg.name = pipeline_run_name(cfg.name, eval_name)
        console.print(f"  Eval name:   [cyan]{eval_name}[/cyan]")
        if overrides.get("asr_run"):
            console.print(f"  ASR run:     [cyan]{overrides['asr_run']}[/cyan]")
        if cfg.input_manifest:
            console.print(f"  Input:       [cyan]{cfg.input_manifest}[/cyan]")
        console.print(f"  Output:      [cyan]{cfg.output_manifest}[/cyan]")
        console.print(f"  Run name:    [cyan]{cfg.name}[/cyan]")
        if overrides.get("aggregate_manifests"):
            for item in overrides["aggregate_manifests"]:
                console.print(f"  Join:        [cyan]{item['model']}[/cyan] ← {item['path']}")
    elif source_name is not None:
        try:
            overrides = apply_source_name_to_single_pipeline(
                pipeline_name=cfg.name,
                steps=cfg.steps,
                source_name=source_name,
                source_dir=source_dir,
                join_manifests=cli_joins,
                asr_run=asr_run,
                aggregate_base=aggregate_base,
            )
        except (ValueError, FileNotFoundError) as exc:
            raise typer.BadParameter(str(exc)) from exc
        cfg.source_dir = overrides["source_dir"]
        cfg.input_manifest = overrides["input_manifest"]
        cfg.source_id = overrides["source_id"]
        cfg.output_manifest = overrides["output_manifest"]
        if overrides.get("aggregate_manifests") is not None:
            _apply_aggregate_manifests(cfg, overrides["aggregate_manifests"])
        if overrides.get("asr_run"):
            _apply_asr_run_transcript_key(cfg, str(overrides["asr_run"]))
            cfg.name = pipeline_run_name(
                f"{cfg.name}_{overrides['asr_run']}", source_name
            )
        else:
            cfg.name = pipeline_run_name(cfg.name, source_name)
        console.print(f"  Source name: [cyan]{source_name}[/cyan]")
        if overrides.get("asr_run"):
            console.print(f"  ASR run:     [cyan]{overrides['asr_run']}[/cyan]")
        if overrides.get("aggregate_base"):
            console.print(f"  Agg base:    [cyan]{overrides['aggregate_base']}[/cyan]")
        if cfg.source_dir:
            console.print(f"  Source dir:  [cyan]{cfg.source_dir}[/cyan]")
        if cfg.input_manifest:
            console.print(f"  Input:       [cyan]{cfg.input_manifest}[/cyan]")
        console.print(f"  Output:      [cyan]{cfg.output_manifest}[/cyan]")
        console.print(f"  Run name:    [cyan]{cfg.name}[/cyan]")
        if overrides.get("aggregate_manifests"):
            for item in overrides["aggregate_manifests"]:
                console.print(f"  Join:        [cyan]{item['model']}[/cyan] ← {item['path']}")
    elif cli_joins is not None:
        _apply_aggregate_manifests(cfg, cli_joins)
        for item in cli_joins:
            console.print(f"  Join:        [cyan]{item['model']}[/cyan] ← {item['path']}")
        if asr_run is not None:
            raise typer.BadParameter("--asr-run requires --source-name or --eval-name")
        if aggregate_base is not None:
            raise typer.BadParameter("--aggregate-base requires --source-name")
    elif asr_run is not None or aggregate_base is not None:
        raise typer.BadParameter(
            "--asr-run / --aggregate-base require --source-name or --eval-name"
        )

    if agreement_base is not None:
        try:
            _apply_agreement_base(cfg, agreement_base)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
        console.print(f"  Agree base:  [cyan]{validate_asr_run(agreement_base)}[/cyan]")

    if eval_model:
        try:
            _apply_eval_models(cfg, list(eval_model))
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
        console.print(
            "  Eval models: [cyan]"
            + ", ".join(validate_asr_run(item) for item in eval_model)
            + "[/cyan]"
        )

    # Shard workers pass --transcript-key without --asr-run/--source-name.
    if transcript_key is not None and asr_run is None:
        try:
            _apply_asr_run_transcript_key(cfg, transcript_key)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
        console.print(f"  Transcript:  [cyan]{validate_asr_run(transcript_key)}[/cyan]")

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
                transcript_key=_first_asr_transcript_key(cfg),
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
        _register_output(
            out,
            pipeline=cfg.name,
            run_dir=runner.run_dir,
            catalog_dir=cfg.catalog_dir,
            sample_count=len(result),
            config_digest=cfg.digest(),
        )
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


artifact_app = typer.Typer(help="Immutable artifact catalog commands")
app.add_typer(artifact_app, name="artifact")


@artifact_app.command("register")
def artifact_register(
    path: Path = typer.Argument(..., help="Existing artifact file"),
    kind: str = typer.Option("manifest", "--kind", help="Artifact kind"),
    catalog_dir: Path = typer.Option(CATALOG_DIR, "--catalog-dir"),
) -> None:
    """Register an existing file without changing or copying its payload."""
    try:
        record = ArtifactCatalog(catalog_dir).register_file(path, kind=kind)  # type: ignore[arg-type]
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print(f"[green]OK[/green] registered [cyan]{record.artifact_id}[/cyan]")
    console.print(f"  URI: {record.uri}")
    console.print(f"  SHA256: {record.sha256}")


@artifact_app.command("list")
def artifact_list(
    kind: Optional[str] = typer.Option(None, "--kind", help="Filter by artifact kind"),
    catalog_dir: Path = typer.Option(CATALOG_DIR, "--catalog-dir"),
) -> None:
    """List catalog records newest first."""
    try:
        records = ArtifactCatalog(catalog_dir).list(kind=kind)  # type: ignore[arg-type]
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    table = Table(title="Artifact Catalog")
    table.add_column("Artifact ID", style="cyan")
    table.add_column("Kind")
    table.add_column("Created (UTC)")
    table.add_column("Pipeline")
    table.add_column("URI")
    for record in records:
        table.add_row(
            record.artifact_id,
            record.kind,
            record.created_at,
            record.producer.pipeline or "-",
            record.uri,
        )
    console.print(table)


@artifact_app.command("show")
def artifact_show(
    artifact_id: str = typer.Argument(...),
    verify: bool = typer.Option(False, "--verify", help="Re-hash payload and verify immutability"),
    catalog_dir: Path = typer.Option(CATALOG_DIR, "--catalog-dir"),
) -> None:
    """Print a record; --verify also detects missing or modified payloads."""
    try:
        record = ArtifactCatalog(catalog_dir).get(artifact_id, verify=verify)
    except (KeyError, FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print_json(data=record.model_dump(mode="json"))


@artifact_app.command("path")
def artifact_path(
    artifact_id: str = typer.Argument(...),
    catalog_dir: Path = typer.Option(CATALOG_DIR, "--catalog-dir"),
) -> None:
    """Resolve an artifact id to the immutable payload location."""
    try:
        record = ArtifactCatalog(catalog_dir).get(artifact_id, verify=True)
    except (KeyError, FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(record.uri)


release_app = typer.Typer(help="Immutable dataset release commands")
app.add_typer(release_app, name="release")


@release_app.command("create")
def release_create(
    release_id: str = typer.Option(..., "--id"),
    source: str = typer.Option(..., "--source", help="Source manifest artifact id"),
    train: str = typer.Option(..., "--train", help="Train manifest artifact id"),
    dev: str = typer.Option(..., "--dev", help="Dev manifest artifact id"),
    test: str = typer.Option(..., "--test", help="Test manifest artifact id"),
    policy_version: str = typer.Option(..., "--policy-version"),
    normalization_version: str = typer.Option(..., "--normalization-version"),
    gold_revision: str = typer.Option(..., "--gold-revision"),
    split_seed: int = typer.Option(..., "--split-seed"),
    group_key: str = typer.Option(..., "--group-key"),
    catalog_dir: Path = typer.Option(CATALOG_DIR, "--catalog-dir"),
) -> None:
    """Freeze existing manifest artifacts as one reproducible dataset release."""
    catalog = ArtifactCatalog(catalog_dir)
    outputs = {"train": train, "dev": dev, "test": test}
    try:
        counts = {
            split: int(catalog.get(artifact_id, verify=True).metadata.get("sample_count", 0))
            for split, artifact_id in outputs.items()
        }
        release = catalog.put_release(
            DatasetRelease(
                release_id=release_id,
                source_artifact_id=source,
                outputs=outputs,
                policy_version=policy_version,
                normalization_version=normalization_version,
                gold_revision=gold_revision,
                split_seed=split_seed,
                group_key=group_key,
                counts=counts,
                git_commit=current_git_commit(),
            )
        )
    except (KeyError, FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print(f"[green]OK[/green] frozen dataset release [cyan]{release.release_id}[/cyan]")


@release_app.command("show")
def release_show(
    release_id: str,
    catalog_dir: Path = typer.Option(CATALOG_DIR, "--catalog-dir"),
) -> None:
    try:
        release = ArtifactCatalog(catalog_dir).get_release(release_id)
    except KeyError as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print_json(data=release.model_dump(mode="json"))


@release_app.command("path")
def release_path(
    release_id: str,
    split: str = typer.Option("test", "--split", help="train/dev/test/holdout"),
    catalog_dir: Path = typer.Option(CATALOG_DIR, "--catalog-dir"),
) -> None:
    """Resolve one frozen release split to its verified Manifest path."""
    try:
        catalog = ArtifactCatalog(catalog_dir)
        release = catalog.get_release(release_id)
        if split not in release.outputs:
            raise ValueError(f"release {release_id} has no split: {split}")
        record = catalog.get(release.outputs[split], verify=True)
    except (KeyError, FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(record.uri)


@release_app.command("build")
def release_build(
    dataset: str = typer.Argument(..., help="Reviewed manifest path/name/artifact id"),
    release_id: str = typer.Option(..., "--id"),
    policy_version: str = typer.Option(..., "--policy-version"),
    normalization_version: str = typer.Option(..., "--normalization-version"),
    gold_revision: str = typer.Option(..., "--gold-revision"),
    group_key: str = typer.Option("speaker_id", "--group-key"),
    split_seed: int = typer.Option(42, "--split-seed"),
    train_ratio: float = typer.Option(0.8, "--train-ratio"),
    dev_ratio: float = typer.Option(0.1, "--dev-ratio"),
    test_ratio: float = typer.Option(0.1, "--test-ratio"),
    stratify_key: Optional[str] = typer.Option("classification_bucket", "--stratify-key"),
    allow_unresolved_review: bool = typer.Option(
        False,
        "--allow-unresolved-review",
        help="Skip pending review_queue/hardcase; freeze only auto_accepted/human_accepted gold",
    ),
    output_dir: Path = typer.Option(Path("data/releases"), "--output-dir"),
    catalog_dir: Path = typer.Option(CATALOG_DIR, "--catalog-dir"),
) -> None:
    """Validate reviewed Gold, split by group, register outputs and freeze a release."""
    catalog = ArtifactCatalog(catalog_dir)
    if dataset.startswith("manifest_"):
        try:
            source_record = catalog.get(dataset, verify=True)
        except (KeyError, FileNotFoundError, ValueError) as exc:
            raise typer.BadParameter(str(exc)) from exc
        source_path = Path(source_record.uri)
    else:
        source_path = _resolve_dataset(dataset)
    full_manifest = Manifest.load(source_path)
    pending_types = {
        "review_queue",
        "qwen_missing",
        "hardcase",
        "hallucination",
        "semantic_inversion",
    }
    unresolved = [
        sample.id
        for sample in full_manifest
        if sample.labels.get("annotation_state")
        not in {"human_accepted", "auto_accepted", "rejected"}
        and (
            sample.labels.get("decision") in {"model_review", "manual_review"}
            or sample.labels.get("classification_bucket") in pending_types
            or (
                sample.labels.get("classification_bucket") == "voicemail"
                and sample.labels.get("decision") != "auto_accept"
            )
        )
    ]
    if unresolved and not allow_unresolved_review:
        raise typer.BadParameter(
            f"release contains {len(unresolved)} unresolved review samples: {unresolved[:10]}"
        )
    if unresolved and allow_unresolved_review:
        console.print(
            f"[yellow]WARN[/yellow] skipping {len(unresolved)} unresolved review samples "
            "(--allow-unresolved-review)"
        )
    eligible = [
        sample
        for sample in full_manifest
        if sample.labels.get("annotation_state") in {"human_accepted", "auto_accepted"}
    ]
    invalid_gold = [
        sample.id for sample in eligible if not str(sample.labels.get("gold_text") or "").strip()
    ]
    if invalid_gold:
        raise typer.BadParameter(
            f"release contains {len(invalid_gold)} accepted samples without Gold: {invalid_gold[:10]}"
        )
    if not eligible:
        raise typer.BadParameter("release has no accepted Gold samples")
    manifest = Manifest(eligible)
    if not dataset.startswith("manifest_"):
        source_record = catalog.register_file(
            source_path, kind="manifest", metadata={"sample_count": len(full_manifest)}
        )
    ratios = {"train": train_ratio, "dev": dev_ratio, "test": test_ratio}
    try:
        split_samples = OperatorRegistry.get("quality.split_dataset").run(
            manifest.samples,
            OperatorConfig(
                params={
                    "group_key": group_key,
                    "stratify_key": stratify_key,
                    "seed": split_seed,
                    "ratios": ratios,
                }
            ),
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    release_dir = output_dir / release_id
    run_dir = RUNS_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_release_{release_id}"
    outputs: dict[str, str] = {}
    counts: dict[str, int] = {}
    for split in ("train", "dev", "test"):
        subset = Manifest([sample for sample in split_samples if sample.labels["split"] == split])
        path = release_dir / f"{split}.parquet"
        subset.save(path)
        subset.save(path.with_suffix(".jsonl"))
        record = register_manifest_output(
            path,
            catalog_dir=catalog_dir,
            pipeline="release.build",
            run_dir=run_dir / split,
            sample_count=len(subset),
        )
        outputs[split] = record.artifact_id
        counts[split] = len(subset)
    release = catalog.put_release(
        DatasetRelease(
            release_id=release_id,
            source_artifact_id=source_record.artifact_id,
            outputs=outputs,
            policy_version=policy_version,
            normalization_version=normalization_version,
            gold_revision=gold_revision,
            split_seed=split_seed,
            group_key=group_key,
            counts=counts,
            git_commit=current_git_commit(),
        )
    )
    console.print(f"[green]OK[/green] built release [cyan]{release.release_id}[/cyan]")
    console.print(f"  Counts: {counts}")


model_app = typer.Typer(help="Trained model registry commands")
app.add_typer(model_app, name="model")


@model_app.command("register")
def model_register(
    model_id: str = typer.Option(..., "--id"),
    base_model: str = typer.Option(..., "--base-model"),
    release_id: str = typer.Option(..., "--release"),
    recipe: str = typer.Option(..., "--recipe"),
    checkpoint: Path = typer.Option(..., "--checkpoint"),
    status: str = typer.Option("ready", "--status"),
    catalog_dir: Path = typer.Option(CATALOG_DIR, "--catalog-dir"),
) -> None:
    if not checkpoint.exists():
        raise typer.BadParameter(f"checkpoint does not exist: {checkpoint}")
    try:
        model = ArtifactCatalog(catalog_dir).put_model(
            ModelVersion(
                model_id=model_id,
                base_model=base_model,
                training_release_id=release_id,
                training_recipe=recipe,
                checkpoint_uri=str(checkpoint.resolve()),
                status=status,  # type: ignore[arg-type]
                git_commit=current_git_commit(),
            )
        )
    except (KeyError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print(f"[green]OK[/green] registered model [cyan]{model.model_id}[/cyan]")


@model_app.command("show")
def model_show(
    model_id: str,
    catalog_dir: Path = typer.Option(CATALOG_DIR, "--catalog-dir"),
) -> None:
    try:
        model = ArtifactCatalog(catalog_dir).get_model(model_id)
    except KeyError as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print_json(data=model.model_dump(mode="json"))


training_app = typer.Typer(help="External training framework adapter")
app.add_typer(training_app, name="training")


@training_app.command("run")
def training_run(
    release_id: str = typer.Option(..., "--release"),
    recipe: Path = typer.Option(..., "--recipe"),
    command: str = typer.Option(..., "--command", help="Trainer command, executed without a shell"),
    checkpoint: Path = typer.Option(..., "--checkpoint"),
    model_id: str = typer.Option(..., "--model-id"),
    base_model: str = typer.Option(..., "--base-model"),
    jobs_dir: Path = typer.Option(Path("runs/training"), "--jobs-dir"),
    catalog_dir: Path = typer.Option(CATALOG_DIR, "--catalog-dir"),
) -> None:
    """Run an external trainer and register its checkpoint on verified success."""
    try:
        job, model = run_training_job(
            catalog=ArtifactCatalog(catalog_dir),
            jobs_dir=jobs_dir,
            release_id=release_id,
            recipe=recipe,
            command=command,
            checkpoint=checkpoint,
            model_id=model_id,
            base_model=base_model,
        )
    except (KeyError, FileNotFoundError, RuntimeError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print(f"[green]OK[/green] training job [cyan]{job.job_id}[/cyan] succeeded")
    console.print(f"  Model: [cyan]{model.model_id}[/cyan]")
    console.print(f"  Checkpoint: {model.checkpoint_uri}")


review_app = typer.Typer(help="Human review queue import/export")
app.add_typer(review_app, name="review")


@review_app.command("export")
def review_export(
    dataset: str = typer.Argument(...),
    output: Path = typer.Option(..., "--output", "-o"),
    bucket: Optional[list[str]] = typer.Option(
        None,
        "--bucket",
        help=(
            "Bucket to review; repeatable "
            "(default: qwen_missing/hardcase/hallucination/semantic_inversion/review_queue)"
        ),
    ),
    revision: str = typer.Option(..., "--revision"),
) -> None:
    dataset_path = _resolve_dataset(dataset)
    manifest = Manifest.load(dataset_path)
    buckets = bucket or list(_DEFAULT_REVIEW_BUCKETS)
    queue_id = _review_queue_id(dataset_path, buckets, revision)
    rows = []
    for sample in manifest:
        if sample.labels.get("classification_bucket") not in buckets:
            continue
        row = {
            "sample_id": sample.id,
            "sha256": sample.sha256,
            "queue_id": queue_id,
            "review_revision": revision,
            "source_path": sample.source_path,
            "decision": "",
            "gold_text": "",
            "reason": "",
        }
        row.update({f"{key}_text": sample.get_transcript_text(key) for key in sample.transcripts})
        rows.append(row)
    output.parent.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    pd.DataFrame(rows).to_excel(output, index=False)
    console.print(f"[green]OK[/green] exported {len(rows)} review rows to [cyan]{output}[/cyan]")


@review_app.command("export-gold")
def review_export_gold(
    dataset: str = typer.Argument(..., help="Classified manifest path/name/artifact id"),
    output: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="XLSX path; includes a filled label column (= gold_text)",
    ),
    output_manifest: Optional[Path] = typer.Option(
        None,
        "--output-manifest",
        help="Optional gold-only parquet/jsonl for evaluation / release input",
    ),
    bucket: Optional[list[str]] = typer.Option(
        None,
        "--bucket",
        help="Bucket to collect; repeatable (default: auto_gold + consensus_gold)",
    ),
    catalog_dir: Path = typer.Option(CATALOG_DIR, "--catalog-dir"),
) -> None:
    """Export auto-accepted gold (auto_gold / consensus_gold) with label column."""
    dataset_path = _resolve_dataset(dataset)
    manifest = Manifest.load(dataset_path)
    buckets = bucket or ["auto_gold", "consensus_gold"]
    gold_samples = []
    rows = []
    missing_gold: list[str] = []
    for sample in manifest:
        if sample.labels.get("classification_bucket") not in buckets:
            continue
        # Only export auto-accepted (or legacy rows without decision).
        decision = str(sample.labels.get("decision") or "").strip()
        if decision and decision not in {"auto_accept", ""}:
            continue
        gold_text = str(sample.labels.get("gold_text") or sample.labels.get("label") or "").strip()
        if not gold_text:
            missing_gold.append(sample.id)
            continue
        copied = sample.model_copy(deep=True)
        copied.labels["gold_text"] = gold_text
        copied.labels["label"] = gold_text
        if not copied.labels.get("annotation_state"):
            copied.labels["annotation_state"] = "auto_accepted"
        gold_samples.append(copied)
        row = {
            "sample_id": sample.id,
            "sha256": sample.sha256,
            "source_path": sample.source_path,
            "classification_bucket": sample.labels.get("classification_bucket", ""),
            "classification_reason": ",".join(
                str(code) for code in (sample.labels.get("classification_reason_codes") or [])
            ),
            "label": gold_text,
            "gold_text": gold_text,
            "gold_source": sample.labels.get("gold_source", ""),
            "annotation_state": copied.labels.get("annotation_state", ""),
            "selection_policy_version": sample.labels.get("selection_policy_version", ""),
        }
        row.update({f"{key}_text": sample.get_transcript_text(key) for key in sample.transcripts})
        rows.append(row)
    if missing_gold:
        raise typer.BadParameter(
            f"{len(missing_gold)} samples in {buckets} missing gold/label: {missing_gold[:10]}"
        )
    if not rows:
        raise typer.BadParameter(f"no samples found in buckets {buckets}")

    output.parent.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    pd.DataFrame(rows).to_excel(output, index=False)
    console.print(f"[green]OK[/green] exported {len(rows)} gold rows to [cyan]{output}[/cyan]")

    if output_manifest is not None:
        gold_manifest = Manifest(gold_samples)
        output_manifest.parent.mkdir(parents=True, exist_ok=True)
        gold_manifest.save(output_manifest)
        gold_manifest.save(output_manifest.with_suffix(".jsonl"))
        run_dir = RUNS_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_export_gold"
        record = register_manifest_output(
            output_manifest,
            catalog_dir=catalog_dir,
            pipeline="review.export_gold",
            run_dir=run_dir,
            sample_count=len(gold_manifest),
        )
        console.print(
            f"[green]OK[/green] wrote gold Manifest [cyan]{output_manifest}[/cyan] "
            f"({len(gold_manifest)} samples)"
        )
        console.print(f"  Artifact: [cyan]{record.artifact_id}[/cyan]")


@review_app.command("import")
def review_import(
    dataset: str = typer.Argument(...),
    review_file: Path = typer.Option(..., "--input"),
    output: Path = typer.Option(..., "--output", "-o"),
    expected_revision: str = typer.Option(..., "--revision"),
    bucket: Optional[list[str]] = typer.Option(
        None, "--bucket", help="Bucket included by export; repeatable"
    ),
    catalog_dir: Path = typer.Option(CATALOG_DIR, "--catalog-dir"),
) -> None:
    import pandas as pd

    dataset_path = _resolve_dataset(dataset)
    manifest = Manifest.load(dataset_path)
    frame = pd.read_excel(review_file, dtype=str).fillna("")
    required = {
        "sample_id",
        "sha256",
        "queue_id",
        "review_revision",
        "decision",
        "gold_text",
        "reason",
    }
    missing = required - set(frame.columns)
    if missing:
        raise typer.BadParameter(f"review file missing columns: {sorted(missing)}")
    if frame["sample_id"].duplicated().any():
        raise typer.BadParameter("review file contains duplicate sample_id")
    buckets = bucket or list(_DEFAULT_REVIEW_BUCKETS)
    expected_queue_id = _review_queue_id(dataset_path, buckets, expected_revision)
    if set(frame["queue_id"]) != {expected_queue_id}:
        raise typer.BadParameter(f"review queue identity mismatch: expected {expected_queue_id}")
    indexed = {sample.id: sample.model_copy(deep=True) for sample in manifest}
    for row in frame.to_dict(orient="records"):
        sample_id = str(row["sample_id"])
        if sample_id not in indexed:
            raise typer.BadParameter(f"unknown review sample_id: {sample_id}")
        sample = indexed[sample_id]
        if str(row["sha256"]) != sample.sha256:
            raise typer.BadParameter(f"audio hash changed for review sample: {sample_id}")
        if str(row["review_revision"]) != expected_revision:
            raise typer.BadParameter(f"stale review revision for sample: {sample_id}")
        existing_revision = sample.labels.get("annotation_revision")
        existing_state = sample.labels.get("annotation_state")
        if (
            existing_state in {"human_accepted", "rejected"}
            and existing_revision != expected_revision
        ):
            raise typer.BadParameter(
                f"refusing to overwrite annotated sample {sample_id} revision {existing_revision}"
            )
        decision = str(row["decision"]).strip()
        if decision not in {"accepted", "rejected"}:
            raise typer.BadParameter(f"invalid decision for {sample_id}: {decision!r}")
        gold_text = str(row["gold_text"]).strip()
        if decision == "accepted" and not gold_text:
            raise typer.BadParameter(f"accepted sample requires gold_text: {sample_id}")
        updates = {
            "annotation_state": "human_accepted" if decision == "accepted" else "rejected",
            "gold_text": gold_text,
            "annotation_revision": expected_revision,
            "annotation_reason": str(row["reason"]).strip(),
        }
        if decision == "accepted":
            updates["label"] = gold_text
        sample.labels.update(updates)
    result = Manifest([indexed[sample.id] for sample in manifest])
    result.save(output)
    result.save(output.with_suffix(".jsonl"))
    run_dir = RUNS_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_review_import"
    record = register_manifest_output(
        output,
        catalog_dir=catalog_dir,
        pipeline="review.import",
        run_dir=run_dir,
        sample_count=len(result),
    )
    console.print(f"[green]OK[/green] imported review decisions to [cyan]{output}[/cyan]")
    console.print(f"  Artifact: [cyan]{record.artifact_id}[/cyan]")


def _write_xlsx_parts(
    rows: list[dict],
    output: Path,
    *,
    max_rows: int,
) -> list[Path]:
    """Write one xlsx, or stem-part-001.xlsx … when rows exceed max_rows."""
    import pandas as pd

    output.parent.mkdir(parents=True, exist_ok=True)
    limit = max(1, int(max_rows))
    if len(rows) <= limit:
        pd.DataFrame(rows).to_excel(output, index=False)
        return [output]
    stem = output.with_suffix("")
    written: list[Path] = []
    part = 0
    for start in range(0, len(rows), limit):
        part += 1
        path = Path(f"{stem}-part-{part:03d}.xlsx")
        pd.DataFrame(rows[start : start + limit]).to_excel(path, index=False)
        written.append(path)
    return written


@review_app.command("export-summary")
def review_export_summary(
    dataset: str = typer.Argument(
        ...,
        help="Reviewed or classified Manifest（path / name / artifact id）",
    ),
    output: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="汇总 xlsx；超过 --max-rows 时写成 stem-part-001.xlsx …",
    ),
    output_manifest: Optional[Path] = typer.Option(
        None,
        "--output-manifest",
        help="可选：写出带 labels.type 的全量 Manifest",
    ),
    max_rows: int = typer.Option(
        20000,
        "--max-rows",
        help="单个 xlsx 最大行数（默认 20000）",
    ),
    catalog_dir: Path = typer.Option(CATALOG_DIR, "--catalog-dir"),
) -> None:
    """Export full-batch summary xlsx with type=classification_bucket.

    工序金标汇总总表：type=classification_bucket（含 consensus_gold / qwen_missing 等）。
    输入一般为 review import 后的 reviewed_*.parquet；人审搁置时也可用 classified_*.
    """
    dataset_path = _resolve_dataset(dataset)
    manifest = Manifest.load(dataset_path)
    rows: list[dict] = []
    typed_samples: list = []
    missing_type: list[str] = []
    for sample in manifest:
        bucket = str(sample.labels.get("classification_bucket") or "").strip()
        if not bucket:
            missing_type.append(sample.id)
            continue
        copied = sample.model_copy(deep=True)
        copied.labels["type"] = bucket
        copied.labels["classification_bucket"] = bucket
        typed_samples.append(copied)
        row = {
            "sample_id": sample.id,
            "sha256": sample.sha256,
            "source_path": sample.source_path,
            "type": bucket,
            "classification_bucket": bucket,
            "classification_reason": ",".join(
                str(code) for code in (sample.labels.get("classification_reason_codes") or [])
            ),
            "label": str(sample.labels.get("label") or sample.labels.get("gold_text") or ""),
            "gold_text": str(sample.labels.get("gold_text") or ""),
            "gold_source": sample.labels.get("gold_source", ""),
            "annotation_state": sample.labels.get("annotation_state", ""),
            "annotation_reason": sample.labels.get("annotation_reason", ""),
            "selection_policy_version": sample.labels.get("selection_policy_version", ""),
        }
        row.update({f"{key}_text": sample.get_transcript_text(key) for key in sample.transcripts})
        rows.append(row)

    if missing_type:
        raise typer.BadParameter(
            f"{len(missing_type)} samples missing classification_bucket/type: "
            f"{missing_type[:10]}"
        )
    if not rows:
        raise typer.BadParameter("manifest is empty; nothing to export")

    written = _write_xlsx_parts(rows, output, max_rows=max_rows)
    if len(written) == 1:
        console.print(
            f"[green]OK[/green] exported {len(rows)} summary rows to [cyan]{written[0]}[/cyan]"
        )
    else:
        console.print(
            f"[green]OK[/green] exported {len(rows)} summary rows in {len(written)} files "
            f"(max_rows={max_rows})"
        )
        for path in written:
            console.print(f"  [cyan]{path}[/cyan]")

    from collections import Counter

    counts = Counter(row["type"] for row in rows)
    console.print(f"  type counts: {dict(sorted(counts.items()))}")

    if output_manifest is not None:
        typed = Manifest(typed_samples)
        output_manifest.parent.mkdir(parents=True, exist_ok=True)
        typed.save(output_manifest)
        typed.save(output_manifest.with_suffix(".jsonl"))
        run_dir = RUNS_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_export_summary"
        record = register_manifest_output(
            output_manifest,
            catalog_dir=catalog_dir,
            pipeline="review.export_summary",
            run_dir=run_dir,
            sample_count=len(typed),
        )
        console.print(
            f"[green]OK[/green] wrote typed Manifest [cyan]{output_manifest}[/cyan] "
            f"({len(typed)} samples)"
        )
        console.print(f"  Artifact: [cyan]{record.artifact_id}[/cyan]")


eval_app = typer.Typer(help="Evaluation readiness check and helpers")
app.add_typer(eval_app, name="eval")


@eval_app.command("check")
def eval_check(
    dataset: str = typer.Argument(..., help="评测输入 Manifest（path / name / artifact id）"),
    gold_field: str = typer.Option("gold_text", "--gold-field"),
    type_field: str = typer.Option("type", "--type-field"),
    min_gold_ratio: float = typer.Option(
        0.0,
        "--min-gold-ratio",
        help="有金标样本占比下限；低于该值时退出码非 0（默认 0 只报告）",
    ),
    require_audio_key: str = typer.Option(
        "resampled_16k",
        "--require-audio-key",
        help="评测推理所需音频键；空字符串表示不检查",
    ),
) -> None:
    """检查 Manifest 是否满足进入评测流水线的条件（金标覆盖、id 唯一、音频键）。"""
    path = _resolve_dataset(dataset)
    report = inspect_eval_manifest(
        path,
        gold_field=gold_field,
        type_field=type_field,
        min_gold_ratio=min_gold_ratio,
        require_audio_key=require_audio_key,
    )
    if report.total == 0:
        raise typer.BadParameter(f"manifest is empty: {path}")

    console.print(f"Manifest: [cyan]{path}[/cyan]")
    console.print(f"  total:          {report.total}")
    console.print(f"  with_gold:      {len(report.with_gold)} ({report.gold_ratio:.2%})")
    console.print(f"  without_gold:   {len(report.without_gold)}")
    if report.without_gold:
        preview = ", ".join(report.without_gold[:10])
        more = "" if len(report.without_gold) <= 10 else f" ... (+{len(report.without_gold) - 10})"
        console.print(f"  without_gold_ids: {preview}{more}")
    console.print(f"  unique_ids:     {report.unique_ids}")
    console.print(f"  duplicate_ids:  {len(report.duplicates)}")
    if report.duplicates:
        console.print(f"  duplicate_preview: {report.duplicates[:10]}")
    console.print(f"  empty_ids:      {report.empty_ids}")
    if require_audio_key:
        console.print(f"  missing_audio[{require_audio_key}]: {len(report.missing_audio)}")
        if report.missing_audio:
            console.print(f"  missing_audio_preview: {report.missing_audio[:10]}")
    console.print("  type / bucket counts:")
    for name in sorted(report.type_counts.keys(), key=lambda x: (x == "(空)", x)):
        console.print(f"    {name}: {report.type_counts[name]}")

    if report.errors:
        for item in report.errors:
            console.print(f"[red]FAIL[/red] {item}")
        raise typer.Exit(code=1)
    console.print(
        "[green]OK[/green] eval readiness passed "
        f"(missing gold allowed; scored subset will be {len(report.with_gold)}/{report.total})"
    )


@eval_app.command("register")
def eval_register(
    dataset: str = typer.Argument(
        ...,
        help="评测源 Manifest（通常为 summary_ / gold_ / reviewed_，path / name / artifact id）",
    ),
    name: str = typer.Option(
        ...,
        "--name",
        help="评测集 stem，写入 datasets/manifests/<name>.parquet（如 eval_local_test）",
    ),
    force: bool = typer.Option(False, "--force", help="覆盖已存在的同名评测集"),
    min_gold_ratio: float = typer.Option(
        0.0,
        "--min-gold-ratio",
        help="有金标样本占比下限（默认 0 只报告）",
    ),
    require_audio_key: str = typer.Option(
        "resampled_16k",
        "--require-audio-key",
        help="评测推理所需音频键；空字符串表示不检查",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        help="覆盖默认输出路径 datasets/manifests/<name>.parquet",
    ),
    catalog_dir: Path = typer.Option(CATALOG_DIR, "--catalog-dir"),
) -> None:
    """把 summary/gold Manifest 注册为评测集快照，供 --eval-name 使用。

    与训练 / Model Registry 无依赖；后续只要推理结果 id 与评测集对齐即可算字准。
    """
    try:
        eval_stem = validate_source_name(name)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    source_path = _resolve_dataset(dataset)
    dest = Path(output) if output is not None else MANIFESTS_DIR / f"{eval_stem}.parquet"
    if dest.exists() and not force:
        raise typer.BadParameter(
            f"eval set already exists: {dest} (pass --force to overwrite)"
        )

    report = inspect_eval_manifest(
        source_path,
        min_gold_ratio=min_gold_ratio,
        require_audio_key=require_audio_key,
    )
    if report.total == 0:
        raise typer.BadParameter(f"manifest is empty: {source_path}")
    if report.errors:
        for item in report.errors:
            console.print(f"[red]FAIL[/red] {item}")
        raise typer.Exit(code=1)

    manifest = Manifest.load(source_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    manifest.save(dest)
    manifest.save(dest.with_suffix(".jsonl"))
    run_dir = RUNS_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_eval_register"
    run_dir.mkdir(parents=True, exist_ok=True)
    artifact_id = _register_output(
        dest,
        pipeline="eval.register",
        run_dir=run_dir,
        catalog_dir=catalog_dir,
        sample_count=len(manifest),
    )
    console.print(f"[green]OK[/green] registered eval set [cyan]{eval_stem}[/cyan]")
    console.print(f"  Source:   [cyan]{source_path}[/cyan]")
    console.print(f"  Output:   [cyan]{dest}[/cyan]")
    console.print(f"  Samples:  {len(manifest)} (gold={len(report.with_gold)})")
    console.print(f"  Artifact: [cyan]{artifact_id}[/cyan]")
    console.print(
        f"  Next: audio-data pipeline run pipelines/qwen_asr_batch.yaml "
        f"--eval-name {eval_stem} --asr-run <alias>"
    )


task_app = typer.Typer(help="Resumable task DAG orchestration")
app.add_typer(task_app, name="task")


@task_app.command("run")
def task_run(
    config: Path = typer.Argument(..., help="Task DAG YAML"),
    runs_dir: Path = typer.Option(Path("runs/tasks"), "--runs-dir"),
) -> None:
    try:
        runner = TaskRunner(load_task(config), runs_dir=runs_dir)
        state = runner.run()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print(f"[green]OK[/green] task [cyan]{state.task_id}[/cyan] succeeded")
    console.print(f"  Run dir: {runner.run_dir}")


@task_app.command("status")
def task_status(run_dir: Path = typer.Argument(..., help="Task run directory")) -> None:
    state_path = run_dir / "state.json"
    if not state_path.is_file():
        raise typer.BadParameter(f"task state not found: {state_path}")
    console.print_json(state_path.read_text(encoding="utf-8"))


@pipeline_app.command("progress")
def pipeline_progress(
    run_root: Path = typer.Argument(
        ...,
        help="Sharded run directory (contains shards/ and shard-*.parquet / PROGRESS.log)",
    ),
    follow: bool = typer.Option(
        False,
        "--follow",
        "-f",
        help="Keep printing the latest PROGRESS.log line (Ctrl+C to stop)",
    ),
    interval: float = typer.Option(5.0, "--interval", help="Follow refresh seconds"),
) -> None:
    """Show aggregate recognition progress / throughput for a sharded run.

    Prefer ``tail -f runs/<run>/PROGRESS.log`` while a job is running; this
    command also rebuilds a snapshot from checkpoints if the log is missing.
    """
    import time as _time

    from audio_engine.core.progress import (
        PROGRESS_JSON,
        PROGRESS_LOG,
        checkpoint_consumed,
        collect_sharded_progress,
        shard_input_count,
    )

    root = run_root.resolve()
    if not root.is_dir():
        raise typer.BadParameter(f"run dir not found: {root}")

    def _print_once() -> None:
        progress_json = root / PROGRESS_JSON
        if progress_json.is_file():
            import json as _json

            data = _json.loads(progress_json.read_text(encoding="utf-8"))
            console.print(
                f"[PROGRESS] done={data.get('done')}/{data.get('total')} "
                f"({data.get('pct', 0):.1f}%) "
                f"rate={data.get('rate_overall', 0):.2f} samples/s "
                f"(window={data.get('rate_window', 0):.2f}/s) "
                f"elapsed={data.get('elapsed_s', 0):.0f}s "
                f"eta={data.get('eta_s')} "
                f"shards[run={data.get('running')} done={data.get('finished')} "
                f"queue={data.get('queued')}]"
            )
            console.print(f"  Source: [cyan]{progress_json}[/cyan]")
            console.print(f"  Log:    [cyan]{root / PROGRESS_LOG}[/cyan]")
            return

        shard_dir = root / "shards"
        shards = sorted(shard_dir.glob("shard-*.parquet")) if shard_dir.is_dir() else []
        if not shards:
            shards = sorted(root.glob("shard-*.parquet"))
        if not shards:
            raise typer.BadParameter(f"no shard-*.parquet under {root}")
        totals = {p.stem: shard_input_count(p) for p in shards}
        done = sum(min(checkpoint_consumed(root, stem), size) for stem, size in totals.items())
        total = sum(totals.values())
        finished = sum(1 for stem in totals if (root / f"{stem}.parquet").exists())
        snapshot = collect_sharded_progress(
            run_root=root,
            shard_totals=totals,
            running_stems=set(),
            finished_stems={stem for stem in totals if (root / f"{stem}.parquet").exists()},
            queued_stems=set(),
            started_at=_time.time(),
            prev_done=done,
            prev_at=_time.time(),
            config={},
        )
        # Override done/total from fresh scan (started_at unknown without progress.json).
        snapshot.done = done
        snapshot.total = total
        snapshot.finished = finished
        snapshot.pct = (100.0 * done / total) if total else 0.0
        snapshot.rate_overall = 0.0
        snapshot.rate_window = 0.0
        snapshot.eta_s = None
        console.print(snapshot.format_line())
        console.print("  (no progress.json yet — counts from checkpoints only, rate n/a)")

    try:
        _print_once()
        while follow:
            _time.sleep(max(interval, 0.5))
            console.print("---")
            _print_once()
    except KeyboardInterrupt:
        console.print("\nstopped")


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
            f"Expected {expected_shards} shard files, found {len(paths)}: {[p.name for p in paths]}"
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
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if len(mismatches):
        badcases_path = run_dir / "badcases.xlsx"
        mismatches[["id", "source_path", col_a, col_b]].to_excel(badcases_path, index=False)
        console.print(f"Badcases exported: [cyan]{badcases_path}[/cyan]")

    console.print(
        f"[green]OK[/green] Compare finished: {summary['match']}/{summary['total']} match"
    )
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
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
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
    format: str = typer.Option(
        "jsonl", "--format", "-f", help="Export format: jsonl, parquet, scp"
    ),
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


def main() -> None:
    """Console entry point for setuptools / typer."""
    load_dotenv()
    app()


if __name__ == "__main__":
    main()
