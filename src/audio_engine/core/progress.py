"""Aggregate progress / throughput for pipeline and sharded ASR runs."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

PROGRESS_JSON = "progress.json"
PROGRESS_LOG = "PROGRESS.log"


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

    def format_line(self) -> str:
        eta = _format_duration(self.eta_s) if self.eta_s is not None else "n/a"
        cfg = _format_config(self.config)
        return (
            f"[PROGRESS] done={self.done}/{self.total} ({self.pct:.1f}%) "
            f"rate={self.rate_overall:.2f} samples/s "
            f"(window={self.rate_window:.2f}/s) "
            f"elapsed={_format_duration(self.elapsed_s)} eta={eta} "
            f"shards[run={self.running} done={self.finished} queue={self.queued}]"
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
    """Fast row count for a shard parquet (fallback to Manifest)."""
    path = Path(shard_path)
    if not path.exists():
        return 0
    try:
        import pyarrow.parquet as pq

        return int(pq.ParquetFile(path).metadata.num_rows)
    except Exception:
        from audio_engine.core.manifest import Manifest

        return len(Manifest.load(path))


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
) -> ProgressSnapshot:
    """Aggregate done counts across shard checkpoints / outputs."""
    done = 0
    total = 0
    for stem, size in shard_totals.items():
        total += size
        done += min(checkpoint_consumed(run_root, stem), size)

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
    )


def write_progress(run_root: Path, snapshot: ProgressSnapshot) -> None:
    """Update progress.json and append one line to PROGRESS.log."""
    root = Path(run_root)
    root.mkdir(parents=True, exist_ok=True)
    payload = asdict(snapshot)
    (root / PROGRESS_JSON).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    line = snapshot.format_line()
    with (root / PROGRESS_LOG).open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


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
