from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from audio_engine.core.artifacts import atomic_write_json
from audio_engine.core.operator import ManifestOperator, OperatorConfig
from audio_engine.core.registry import register_operator
from audio_engine.core.sample import Sample


def _corpus(samples: list[Sample], prefix: str) -> dict[str, float | int]:
    keys = ("substitutions", "deletions", "insertions")
    totals = {
        key: sum(int(s.quality.get(f"{prefix}_{key}", 0) or 0) for s in samples) for key in keys
    }
    reference_length = sum(
        int(s.quality.get(f"{prefix}_reference_length", 0) or 0) for s in samples
    )
    errors = sum(totals.values())
    return {
        **totals,
        "errors": errors,
        "reference_length": reference_length,
        "corpus_cer": errors / max(reference_length, 1),
        "samples": len(samples),
    }


def _bootstrap_delta(
    samples: list[Sample], baseline: str, candidate: str, *, iterations: int, seed: int
) -> dict[str, float | int | list[float]]:
    if not samples or iterations <= 0:
        return {"iterations": 0, "seed": seed, "ci95": []}
    rng = random.Random(seed)
    deltas: list[float] = []
    for _ in range(iterations):
        draw = [samples[rng.randrange(len(samples))] for _ in samples]
        delta = float(_corpus(draw, candidate)["corpus_cer"]) - float(
            _corpus(draw, baseline)["corpus_cer"]
        )
        deltas.append(delta)
    deltas.sort()
    return {
        "iterations": iterations,
        "seed": seed,
        "ci95": [
            deltas[int(0.025 * (iterations - 1))],
            deltas[int(0.975 * (iterations - 1))],
        ],
    }


@register_operator
class EvaluationReportOperator(ManifestOperator):
    """Aggregate paired sample metrics and enforce explicit candidate regression gates."""

    name = "evaluation_report"
    version = "1.0.0"
    category = "quality"

    def run(self, samples: list[Sample], config: OperatorConfig) -> list[Sample]:
        baseline = str(config.params.get("baseline_prefix", "old_model"))
        candidate = str(config.params.get("candidate_prefix", "new_model"))
        bucket_key = str(config.params.get("bucket_key", "classification_bucket"))
        missing = [
            sample.id
            for sample in samples
            if f"{baseline}_reference_length" not in sample.quality
            or f"{candidate}_reference_length" not in sample.quality
        ]
        if missing:
            raise ValueError(
                f"evaluation metrics missing for {len(missing)} samples: {missing[:10]}"
            )
        report: dict[str, Any] = {
            "baseline_prefix": baseline,
            "candidate_prefix": candidate,
            "overall": {
                baseline: _corpus(samples, baseline),
                candidate: _corpus(samples, candidate),
            },
            "buckets": {},
        }
        buckets: dict[str, list[Sample]] = defaultdict(list)
        for sample in samples:
            buckets[str(sample.labels.get(bucket_key, "unclassified"))].append(sample)
        for name, members in sorted(buckets.items()):
            report["buckets"][name] = {
                baseline: _corpus(members, baseline),
                candidate: _corpus(members, candidate),
            }
        baseline_cer = report["overall"][baseline]["corpus_cer"]
        candidate_cer = report["overall"][candidate]["corpus_cer"]
        report["delta_cer"] = candidate_cer - baseline_cer
        report["paired_bootstrap"] = _bootstrap_delta(
            samples,
            baseline,
            candidate,
            iterations=int(config.params.get("bootstrap_iterations", 1000)),
            seed=int(config.params.get("bootstrap_seed", 42)),
        )
        gates = config.params.get("gates") or []
        results = []
        for gate in gates:
            name = str(gate.get("name") or "max_cer_regression")
            max_regression = float(gate.get("max_cer_regression", 0.0))
            bucket = gate.get("bucket")
            view = report["overall"] if bucket is None else report["buckets"].get(str(bucket))
            if view is None:
                passed = False
                delta = None
            else:
                delta = view[candidate]["corpus_cer"] - view[baseline]["corpus_cer"]
                passed = delta <= max_regression
            results.append(
                {
                    "name": name,
                    "bucket": bucket,
                    "delta_cer": delta,
                    "limit": max_regression,
                    "passed": passed,
                }
            )
        report["gates"] = results
        report["passed"] = all(item["passed"] for item in results) if results else True
        if config.run_dir is None:
            raise ValueError("evaluation_report requires a pipeline run directory")
        report_path = Path(config.run_dir) / "reports" / "evaluation.json"
        atomic_write_json(report_path, report)
        if not report["passed"] and bool(config.params.get("fail_on_regression", True)):
            failed = [item["name"] for item in results if not item["passed"]]
            raise ValueError(f"evaluation regression gate failed: {failed}; report={report_path}")
        return list(samples)
