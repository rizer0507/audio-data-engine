"""Benchmark method helpers for serial vs dual-GPU stage1 ASR (no fabricated numbers).

Run on the server with the same batch / decode config:

  # Serial baseline (force single card + disable dual scheduler)
  # In configs/stage1/server.yaml set scheduler.dual_gpu: false and --gpus <one>

  # Dual-GPU
  # scheduler.dual_gpu: true and --gpus "$AUTHORIZED_GPUS" (exactly two)

Record from job artifacts (do not invent):
  - total wall time: state.created_at → job_finished
  - per-family load: scheduler_metrics.json load_timings_s
  - claim delay P95: scheduler_metrics.json claim_delay_p95_s
  - busy time per GPU: scheduler_metrics.json busy_time_s
  - idle reasons: scheduler_metrics.json idle_reasons

Expected acceptance (server-measured only):
  - claim delay P95 ≤ 10s when a compatible task is ready and VRAM/lease free
    (model load/probe timed separately; not part of claim delay)
  - one card blocked/foreign → the other continues compatible families
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from audio_engine.core.stage1.gpu_scheduler import (
    build_dual_gpu_plan,
    build_serial_baseline_plan,
)


def write_benchmark_method(path: Path, *, gpus: list[str]) -> dict[str, Any]:
    payload = {
        "schema_version": "1.0",
        "title": "stage1 serial vs dual-GPU benchmark method",
        "inputs_must_match": [
            "same batch / source",
            "same model weights digests",
            "same decode / prompt / selection rule",
        ],
        "serial_baseline": {
            "config": "scheduler.dual_gpu=false; --gpus <single authorized card>",
            "plan": build_serial_baseline_plan(
                ["qwen", "glm", "sensevoice"], gpus[0] if gpus else "?"
            ),
        },
        "dual_gpu": {
            "config": "scheduler.dual_gpu=true; --gpus <two authorized cards>",
            "plan": build_dual_gpu_plan(["qwen", "glm", "sensevoice"], gpus),
            "constraints": [
                "util==0 must not be treated as idle",
                "Qwen and GLM must not co-reside on one card",
                "SenseVoice workers constrained to leased GPU only",
                "multi-job must not steal foreign leases",
            ],
        },
        "metrics_sources": {
            "scheduler_metrics": "runs/stage1/jobs/<job_id>/scheduler_metrics.json",
            "events": "runs/stage1/jobs/<job_id>/events.jsonl",
            "gpu_binding": "runs/stage1/jobs/<job_id>/gpu_binding.json",
        },
        "results": {
            "serial_total_s": None,
            "dual_total_s": None,
            "claim_delay_p95_s": None,
            "load_timings_s": None,
            "busy_time_s": None,
            "note": "未实测不得填写；本地单测只验证计划结构",
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return payload
