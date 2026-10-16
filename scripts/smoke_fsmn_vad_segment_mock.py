#!/usr/bin/env python3
"""Small-batch closed-loop smoke for 041 (mock VAD — not real-model parity).

Creates 2 synthetic parents → segments → Manifest roundtrip → grouping check.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

import audio_engine.operators  # noqa: F401
from audio_engine.core.dataset_v3.grouping import GroupingConfig, build_leakage_groups
from audio_engine.core.manifest import Manifest, file_sha256
from audio_engine.core.operator import OperatorConfig
from audio_engine.core.registry import OperatorRegistry
from audio_engine.core.sample import Sample

SR = 16000


def _tone(ms: float) -> np.ndarray:
    n = int(ms * SR / 1000)
    t = np.arange(n, dtype=np.float32) / SR
    return (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def main() -> int:
    out = _ROOT / "var" / "smoke_vad_041"
    out.mkdir(parents=True, exist_ok=True)
    parents = []
    for name, dur_ms, intervals in (
        ("src_a", 5000, [[200, 1600], [2000, 3800], [3900, 4200]]),
        ("src_b", 3000, [[100, 1400]]),
    ):
        path = out / f"{name}.wav"
        sf.write(str(path), _tone(dur_ms), SR)
        parents.append(
            Sample(
                id=name,
                source_path=str(path),
                sha256=file_sha256(path),
                sample_rate=SR,
                duration=dur_ms / 1000.0,
                audio={"resampled_16k": str(path), "raw": str(path)},
                labels={"audio_pass": True},
            )
        )

    op = OperatorRegistry.get("audio.fsmn_vad_segment")
    # Per-source intervals via separate runs (ScheduleVadModel is global to params)
    all_segs: list[Sample] = []
    for parent, intervals in zip(
        parents,
        [[[200, 1600], [2000, 3800], [3900, 4200]], [[100, 1400]]],
    ):
        cfg = OperatorConfig(
            params={
                "mock": True,
                "mock_intervals": intervals,
                "input_audio_key": "resampled_16k",
                "output_audio_key": "resampled_16k",
            },
            output_dir=out / "derived",
            cache_dir=out / "cache",
            run_dir=out / "run",
        )
        all_segs.extend(op.run([parent], cfg))

    man_path = out / "cleaned_smoke_vad_041.parquet"
    Manifest(all_segs).save(man_path)
    loaded = Manifest.load(man_path).samples
    groups = build_leakage_groups(loaded, GroupingConfig(require_source_index=False))

    report = {
        "mode": "mock_vad",
        "real_model_parity": False,
        "parents": len(parents),
        "segments": len(loaded),
        "segment_ids": [s.id for s in loaded],
        "by_parent": {},
        "grouping": {
            sid: groups.leakage_group_id.get(sid) for sid in (s.id for s in loaded)
        },
        "manifest": str(man_path),
    }
    for s in loaded:
        pid = s.labels["source_audio_id"]
        report["by_parent"].setdefault(pid, []).append(s.id)

    # Expected: src_a → 2 kept (300ms+pad <1s discarded), src_b → 1
    assert len(report["by_parent"].get("src_a", [])) == 2, report
    assert len(report["by_parent"].get("src_b", [])) == 1, report
    g_a = {groups.leakage_group_id[i] for i in report["by_parent"]["src_a"]}
    assert len(g_a) == 1

    (out / "smoke_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("OK mock closed-loop")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
