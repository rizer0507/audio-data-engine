#!/usr/bin/env python3
"""Compare original ORT defaults with bounded threads using identical uncached inputs.

python scripts/benchmark_dnsmos_threads.py --model-path /path/sig_bak_ovr.onnx
Add --audio-dir /path/to/segment_wavs to use real WAVs instead of synthetic inputs.
This measures one shard. It does not certify a 128-core server's aggregate throughput.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from audio_engine.core.quality.dnsmos_p835 import (
    DNSMOS_SAMPLE_RATE,
    DnsmosP835Session,
    DnsmosRuntimeOptions,
    model_file_digest,
)


def benchmark(model: Path, inputs: list[tuple[np.ndarray, int]], workers: int,
              intra_threads: int, spinning: bool) -> dict:
    import onnxruntime as ort

    local = threading.local()
    barrier = threading.Barrier(workers, timeout=120)
    sessions = []
    lock = threading.Lock()

    def initialize():
        if intra_threads:
            session = DnsmosP835Session(model, runtime_options=DnsmosRuntimeOptions(
                intra_op_num_threads=intra_threads, allow_spinning=spinning,
            ))
        else:
            # Exact pre-fix thread defaults: auto-sized pool and default spinning.
            session = object.__new__(DnsmosP835Session)
            options = ort.SessionOptions()
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session._session = ort.InferenceSession(str(model), sess_options=options,
                                                    providers=["CPUExecutionProvider"])
            session.input_name = session._session.get_inputs()[0].name
            session.model_digest = model_file_digest(model)
        local.session = session
        session.score_array(*inputs[0])  # Same warmup for both variants.
        with lock:
            sessions.append(session)
        barrier.wait()

    def score(item):
        result = local.session.score_array(*item)
        if result.status != "success":
            raise RuntimeError(result.error)
        return [result.sig, result.bak, result.ovrl]

    with ThreadPoolExecutor(max_workers=workers, initializer=initialize) as pool:
        # Force every worker to initialize before the measured region.
        ready = threading.Barrier(workers + 1, timeout=120)
        futures = [pool.submit(ready.wait) for _ in range(workers)]
        ready.wait()
        for future in futures:
            future.result()
        cpu_start, wall_start = time.process_time(), time.perf_counter()
        scores = list(pool.map(score, inputs))
        elapsed, cpu = time.perf_counter() - wall_start, time.process_time() - cpu_start
    return {"count": len(inputs), "workers": workers, "intra_threads": intra_threads,
            "spinning": spinning, "elapsed_s": elapsed, "clips_per_s": len(inputs) / elapsed,
            "cpu_seconds": cpu, "cpu_core_equivalents": cpu / elapsed, "scores": scores}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--audio-dir", type=Path)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--intra-threads", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("var/dnsmos_thread_benchmark.json"))
    args = parser.parse_args()
    if min(args.limit, args.workers, args.intra_threads) < 1:
        parser.error("limit, workers and intra-threads must be positive")
    if args.audio_dir:
        import soundfile as sf

        paths = sorted(args.audio_dir.rglob("*.wav"))[:args.limit]
        inputs = [sf.read(str(path), always_2d=False) for path in paths]
        source = str(args.audio_dir)
    else:
        rng = np.random.default_rng(20261016)
        inputs = [(rng.normal(0, 0.03, 16000 * 10).astype(np.float32), DNSMOS_SAMPLE_RATE)
                  for _ in range(args.limit)]
        source = "synthetic seeded 10-second noise (not production throughput)"
    if not inputs:
        parser.error("No WAV inputs found")
    baseline = benchmark(args.model_path, inputs, args.workers, 0, True)
    print(f"Baseline: {baseline['clips_per_s']:.2f} clips/s", flush=True)
    optimized = benchmark(args.model_path, inputs, args.workers, args.intra_threads, False)
    max_diff = float(np.max(np.abs(np.asarray(baseline.pop("scores")) -
                                   np.asarray(optimized.pop("scores")))))
    result = {"host_logical_cpus": os.cpu_count(), "input_source": source,
              "model_path": str(args.model_path), "model_digest": model_file_digest(args.model_path),
              "baseline": baseline, "optimized": optimized, "max_score_abs_diff": max_diff,
              "throughput_ratio": optimized["clips_per_s"] / baseline["clips_per_s"],
              "score_parity_pass": max_diff <= 1e-4,
              "throughput_not_lower": optimized["clips_per_s"] >= baseline["clips_per_s"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["score_parity_pass"] and result["throughput_not_lower"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
