"""Aggregate progress / throughput for pipeline and sharded ASR runs."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from audio_engine.core.artifacts import atomic_write_json

PROGRESS_JSON = "progress.json"
PROGRESS_LOG = "PROGRESS.log"
INGEST_STEP_PROGRESS = "ingest_step_progress.json"


def is_audio_ingest(pipeline_name: str) -> bool:
    # The CLI appends the batch name to the configured pipeline name.
    return pipeline_name == "audio_ingest" or pipeline_name.startswith("audio_ingest_")


@dataclass
class ProgressSnapshot:
    total: int
    done: int
    running: int = 0
    finished: int = 0
    queued: int = 0
    elapsed_s: float = 0.0
    rate_overall: float = 0.0
    rate_window: float = 0.0
    eta_s: float | None = None
    pct: float = 0.0
    config: dict[str, Any] = field(default_factory=dict)
    updated_at: float = 0.0
    step_done: dict[str, int] = field(default_factory=dict)
    output_rows: int | None = None
    phase: str | None = None

    def format_line(self) -> str:
        if self.phase == "preparing":
            return "[PROGRESS] pipeline=audio_ingest phase=preparing total=pending_scan"
        eta = _format_duration(self.eta_s) if self.eta_s is not None else "n/a"
        cfg = _format_config(self.config)
        return (
            f"[PROGRESS] done={self.done}/{self.total} ({self.pct:.1f}%) "
            f"rate={self.rate_overall:.2f} samples/s "
            f"(window={self.rate_window:.2f}/s) "
            f"elapsed={_format_duration(self.elapsed_s)} eta={eta} "
            f"shards[run={self.running} done={self.finished} queue={self.queued}]"
            + (
                f" unit=parents ingested_rows={self.output_rows} excluded={self.done - self.output_rows}"
                if self.output_rows is not None else ""
            )
            + (
                " steps[" + " ".join(f"{name}={done}/{self.total}" for name, done in self.step_done.items()) + "]"
                if self.step_done else ""
            )
            + (f" | {cfg}" if cfg else "")
        )


def _format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0 or seconds != seconds:  # NaN
        return "n/a"
    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _format_config(config: dict[str, Any]) -> str:
    if not config:
        return ""
    parts: list[str] = []
    for key in (
        "pipeline",
        "shards",
        "parallel",
        "gpus",
        "instances_per_gpu",
        "batch_size",
        "checkpoint_every",
        "strategy",
    ):
        if key in config and config[key] is not None:
            parts.append(f"{key}={config[key]}")
    return " ".join(parts)


def shard_input_count(shard_path: Path) -> int:
    """Fast row count for a shard parquet (fallback to Manifest).

    Returns 0 for missing/empty/unreadable files so a mid-write 0-byte parquet
    never crashes parent progress polls.
    """
    path = Path(shard_path)
    if not path.exists():
        return 0
    try:
        if path.stat().st_size == 0:
            return 0
    except OSError:
        return 0
    try:
        import pyarrow.parquet as pq

        return int(pq.ParquetFile(path).metadata.num_rows)
    except Exception:
        try:
            from audio_engine.core.manifest import Manifest

            return len(Manifest.load(path))
        except Exception:
            return 0


def checkpoint_consumed(run_root: Path, shard_stem: str) -> int:
    """How many samples a shard has already committed via checkpoints."""
    out = Path(run_root) / f"{shard_stem}.parquet"
    if out.exists():
        return shard_input_count(out)

    ckpt_root = Path(run_root) / shard_stem / "checkpoints"
    if not ckpt_root.is_dir():
        return 0

    best = 0
    for step_dir in ckpt_root.iterdir():
        if not step_dir.is_dir():
            continue
        state_path = step_dir / "_state.json"
        if not state_path.exists():
            continue
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        consumed = sum(int(part.get("count_in", 0)) for part in (state.get("parts") or []))
        if state.get("complete"):
            best = max(best, int(state.get("output_count") or consumed))
        else:
            best = max(best, consumed)
    return best


def _count_result_jsons(directory: Path) -> int:
    """Count per-source result JSONs; skip sidecar summaries like ``_batch_summary.json``."""
    if not directory.is_dir():
        return 0
    n = 0
    for path in directory.glob("*.json"):
        if path.name.startswith("_"):
            continue
        n += 1
    return n


def count_vad_source_results(run_root: Path, shard_stem: str | None = None) -> int:
    """Count FSMN VAD per-source result files already written under a run.

    Looks at ``run_root/<shard>/vad_source_results/`` when ``shard_stem`` is set,
    otherwise sums all ``shard-*/vad_source_results/`` plus a parent-level
    ``run_root/vad_source_results/`` (used when results are aggregated at the parent).
    """
    root = Path(run_root)
    if shard_stem is not None:
        return _count_result_jsons(root / shard_stem / "vad_source_results")

    total = 0
    for shard_dir in root.glob("shard-*"):
        if shard_dir.is_dir():
            total += _count_result_jsons(shard_dir / "vad_source_results")
    # Parent-level aggregate (compat with layouts that write all sources here).
    total += _count_result_jsons(root / "vad_source_results")
    return total


def shard_done_count(run_root: Path, shard_stem: str, shard_total: int) -> int:
    """Best-effort done for one shard: max(checkpoint, per-shard VAD source results)."""
    ckpt = checkpoint_consumed(run_root, shard_stem)
    vad = count_vad_source_results(run_root, shard_stem)
    return min(max(ckpt, vad), shard_total)


def collect_sharded_progress(
    *,
    run_root: Path,
    shard_totals: dict[str, int],
    running_stems: set[str],
    finished_stems: set[str],
    queued_stems: set[str],
    started_at: float,
    prev_done: int,
    prev_at: float,
    config: dict[str, Any] | None = None,
    succeeded_stems: set[str] | None = None,
) -> ProgressSnapshot:
    """Aggregate done counts across shard checkpoints / VAD source results / outputs."""
    root = Path(run_root)
    total = sum(shard_totals.values())
    steps = list((config or {}).get("ingest_steps") or [])
    step_done = dict.fromkeys(steps, 0)
    output_rows = None
    if steps:
        succeeded = succeeded_stems or set()
        done = sum(shard_totals[stem] for stem in succeeded)
        output_rows = sum(shard_input_count(root / f"{stem}.parquet") for stem in succeeded)
        for stem, size in shard_totals.items():
            if stem in succeeded:
                for step in steps:
                    step_done[step] += size
                continue
            try:
                progress = json.loads((root / stem / INGEST_STEP_PROGRESS).read_text(encoding="utf-8"))
                current = steps.index(progress["step"])
            except (OSError, ValueError, KeyError):
                continue
            for step in steps[:current]:
                step_done[step] += size
            step_done[steps[current]] += min(progress["done"], size)
    else:
        done = sum(shard_done_count(root, stem, size) for stem, size in shard_totals.items())
        # Prefer per-shard VAD results; fall back to parent-level aggregate.
        parent_vad = _count_result_jsons(root / "vad_source_results")
        if parent_vad and not any(count_vad_source_results(root, stem) for stem in shard_totals):
            done = max(done, parent_vad)
    done = min(done, total)

    now = time.time()
    elapsed = max(now - started_at, 1e-6)
    window = max(now - prev_at, 1e-6)
    rate_overall = done / elapsed
    rate_window = max(done - prev_done, 0) / window
    remaining = max(total - done, 0)
    eta = remaining / rate_overall if rate_overall > 0 and remaining else None
    pct = (100.0 * done / total) if total else 0.0
    return ProgressSnapshot(
        total=total,
        done=done,
        running=len(running_stems),
        finished=len(finished_stems),
        queued=len(queued_stems),
        elapsed_s=elapsed,
        rate_overall=rate_overall,
        rate_window=rate_window,
        eta_s=eta,
        pct=pct,
        config=dict(config or {}),
        updated_at=now,
        step_done=step_done,
        output_rows=output_rows,
    )


def write_progress(run_root: Path, snapshot: ProgressSnapshot) -> None:
    """Update progress.json and append one line to PROGRESS.log."""
    root = Path(run_root)
    root.mkdir(parents=True, exist_ok=True)
    payload = asdict(snapshot)
    atomic_write_json(root / PROGRESS_JSON, payload)
    line = snapshot.format_line()
    with (root / PROGRESS_LOG).open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def read_progress(run_root: Path) -> dict[str, Any] | None:
    try:
        return json.loads((run_root / PROGRESS_JSON).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def emit_progress(
    snapshot: ProgressSnapshot,
    *,
    run_root: Path | None = None,
    log: Any = None,
) -> None:
    line = snapshot.format_line()
    if log is not None:
        log(line)
    else:
        logger.info(line)
    if run_root is not None:
        write_progress(run_root, snapshot)


def sharding_config_summary(
    sharding: Any,
    *,
    pipeline: str | None = None,
    batch_size: int | None = None,
    checkpoint_every: int | None = None,
) -> dict[str, Any]:
    gpus = list(getattr(sharding, "gpus", ()) or ())
    return {
        "pipeline": pipeline,
        "shards": getattr(sharding, "shards", None),
        "parallel": getattr(sharding, "effective_parallel", None),
        "gpus": ",".join(str(g) for g in gpus) if gpus else None,
        "instances_per_gpu": getattr(sharding, "instances_per_gpu", None),
        "strategy": getattr(sharding, "strategy", None),
        "batch_size": batch_size,
        "checkpoint_every": checkpoint_every,
    }


def resolve_asr_batch_size(cfg: Any) -> int | None:
    """Best-effort read batch_size from the first ASR step's config yaml."""
    try:
        import yaml
    except ImportError:
        return None
    for step in getattr(cfg, "steps", []) or []:
        operator = getattr(step, "operator", "") or ""
        if not operator.startswith("asr."):
            continue
        params = getattr(step, "params", {}) or {}
        path = params.get("config_path")
        if not path:
            continue
        file_path = Path(path)
        if not file_path.is_file():
            continue
        try:
            data = yaml.safe_load(file_path.read_text(encoding="utf-8")) or {}
        except OSError:
            continue
        if data.get("batch_size") is not None:
            return int(data["batch_size"])
    return None
