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
from audio_engine.core.pipeline import PipelineConfig, PipelineRunner, PipelineStep
from audio_engine.core.registry import OperatorRegistry

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
) -> None:
    """Run a pipeline from YAML configuration."""
    cfg = PipelineConfig.from_yaml(config_path)
    cfg.force = force or cfg.force
    cfg.mock = mock or cfg.mock

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
