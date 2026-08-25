"""Tests for aggregate sharded progress helpers."""

from __future__ import annotations

import json
from pathlib import Path

from audio_engine.core.manifest import Manifest
from audio_engine.core.progress import (
    ProgressSnapshot,
    checkpoint_consumed,
    collect_sharded_progress,
    write_progress,
)
from audio_engine.core.sample import Sample


def test_checkpoint_consumed_and_snapshot(tmp_path: Path):
    run_root = tmp_path / "run"
    shard = "shard-000"
    ckpt = run_root / shard / "checkpoints" / "00_qwen_asr"
    ckpt.mkdir(parents=True)
    (ckpt / "_state.json").write_text(
        json.dumps(
            {
                "parts": [
                    {"file": "part-000000.parquet", "count_in": 100, "count_out": 100},
                    {"file": "part-000001.parquet", "count_in": 50, "count_out": 50},
                ],
                "complete": False,
                "output_count": 0,
            }
        ),
        encoding="utf-8",
    )
    assert checkpoint_consumed(run_root, shard) == 150

    Manifest(
        [Sample(id=f"s{i}", source_path=f"/tmp/{i}.wav", duration=1.0) for i in range(40)]
    ).save(run_root / "shard-001.parquet")

    snap = collect_sharded_progress(
        run_root=run_root,
        shard_totals={"shard-000": 200, "shard-001": 40},
        running_stems={"shard-000"},
        finished_stems={"shard-001"},
        queued_stems=set(),
        started_at=0,
        prev_done=100,
        prev_at=0,
        config={"shards": 2, "parallel": 2, "batch_size": 32},
    )
    assert snap.done == 190
    assert snap.total == 240
    assert "batch_size=32" in snap.format_line()

    write_progress(run_root, snap)
    assert (run_root / "PROGRESS.log").exists()
    assert (run_root / "progress.json").exists()


def test_progress_snapshot_format():
    line = ProgressSnapshot(
        total=1000,
        done=250,
        running=4,
        finished=1,
        queued=3,
        elapsed_s=50,
        rate_overall=5.0,
        rate_window=8.0,
        eta_s=150,
        pct=25.0,
        config={"pipeline": "qwen_asr_batch", "shards": 8},
    ).format_line()
    assert line.startswith("[PROGRESS]")
    assert "done=250/1000" in line
    assert "rate=5.00 samples/s" in line
