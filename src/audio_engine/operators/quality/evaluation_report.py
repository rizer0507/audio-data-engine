from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from audio_engine.core.artifacts import atomic_write_json
from audio_engine.core.operator import ManifestOperator, OperatorConfig
from audio_engine.core.registry import register_operator
from audio_engine.core.sample import Sample


def _corpus(samples: list[Sample], prefix: str) -> dict[str, float | int | None]:
    keys = ("substitutions", "deletions", "insertions")
    totals = {
        key: sum(int(s.quality.get(f"{prefix}_{key}", 0) or 0) for s in samples) for key in keys
    }
    reference_length = sum(
        int(s.quality.get(f"{prefix}_reference_length", 0) or 0) for s in samples
    )
    errors = sum(totals.values())
    corpus_cer = errors / max(reference_length, 1) if samples else 0.0
    return {
        **totals,
        "errors": errors,
        "reference_length": reference_length,
        "corpus_cer": corpus_cer,
        "corpus_char_acc": max(0.0, 1.0 - corpus_cer) if reference_length > 0 else None,
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


def _has_metric(sample: Sample, prefix: str) -> bool:
    return f"{prefix}_reference_length" in sample.quality


def _bucket_value(sample: Sample, bucket_key: str) -> str:
    labels = sample.labels or {}
    for key in (bucket_key, "type", "classification_bucket"):
        value = labels.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return "unclassified"


def _gold_text(sample: Sample) -> str:
    text = str(sample.labels.get("gold_text") or sample.labels.get("label") or "").strip()
    if text:
        return text
    return str(sample.get_transcript_text("gold") or "").strip()


def _char_acc(cer: Any) -> float | None:
    if cer is None:
        return None
    try:
        return round(max(0.0, 1.0 - float(cer)), 6)
    except (TypeError, ValueError):
        return None


def _discover_prefixes(samples: list[Sample]) -> list[str]:
    prefixes: list[str] = []
    seen: set[str] = set()
    for sample in samples:
        for key in sample.quality:
            if not key.endswith("_reference_length"):
                continue
            prefix = key[: -len("_reference_length")]
            if not prefix or prefix in seen or "_vs_" in prefix:
                continue
            seen.add(prefix)
            prefixes.append(prefix)
    return prefixes


def _resolve_prefixes(config: OperatorConfig, samples: list[Sample]) -> list[str]:
    raw_models = config.params.get("model_prefixes")
    if raw_models:
        return [str(item).strip() for item in raw_models if str(item).strip()]
    baseline = config.params.get("baseline_prefix")
    candidate = config.params.get("candidate_prefix")
    if baseline is not None or candidate is not None:
        prefixes: list[str] = []
        if baseline is not None and str(baseline).strip():
            prefixes.append(str(baseline).strip())
        if candidate is not None and str(candidate).strip() and str(candidate).strip() not in prefixes:
            prefixes.append(str(candidate).strip())
        return prefixes
    discovered = _discover_prefixes(samples)
    return discovered or ["old_model", "new_model"]


def _sample_row(
    sample: Sample,
    *,
    prefixes: list[str],
    bucket_key: str,
    scored_prefixes: set[str],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": sample.id,
        "type": _bucket_value(sample, bucket_key),
        "label": _gold_text(sample),
        "has_gold": bool(_gold_text(sample)),
        "eval_scored": bool(scored_prefixes),
    }
    for prefix in prefixes:
        row[f"{prefix}_text"] = sample.get_transcript_text(prefix)
    for prefix in prefixes:
        hyp_key = f"{prefix}_text"
        if prefix not in scored_prefixes:
            for suffix in (
                "total",
                "错字",
                "少字",
                "多字",
                "cer",
                "ref_len",
                "hyp_len",
                "字准率",
            ):
                row[f"vs_label_{hyp_key}_{suffix}"] = None
            continue
        sub = int(sample.quality.get(f"{prefix}_substitutions", 0) or 0)
        dele = int(sample.quality.get(f"{prefix}_deletions", 0) or 0)
        ins = int(sample.quality.get(f"{prefix}_insertions", 0) or 0)
        cer = sample.quality.get(f"{prefix}_cer")
        ref_len = int(sample.quality.get(f"{prefix}_reference_length", 0) or 0)
        hyp_len = len(sample.get_transcript_text(prefix) or "")
        row[f"vs_label_{hyp_key}_total"] = sub + dele + ins
        row[f"vs_label_{hyp_key}_错字"] = sub
        row[f"vs_label_{hyp_key}_少字"] = dele
        row[f"vs_label_{hyp_key}_多字"] = ins
        row[f"vs_label_{hyp_key}_cer"] = cer
        row[f"vs_label_{hyp_key}_ref_len"] = ref_len
        row[f"vs_label_{hyp_key}_hyp_len"] = hyp_len
        row[f"vs_label_{hyp_key}_字准率"] = _char_acc(cer)
    return row


def _empty_bucket() -> dict[str, float | int]:
    return {
        "n": 0,
        "n_skip": 0,
        "错字": 0,
        "少字": 0,
        "多字": 0,
        "dis": 0,
        "ref_len": 0,
        "sum_acc": 0.0,
    }


def _add_sample_metrics(bucket: dict[str, float | int], sample: Sample, prefix: str) -> None:
    sub = int(sample.quality.get(f"{prefix}_substitutions", 0) or 0)
    dele = int(sample.quality.get(f"{prefix}_deletions", 0) or 0)
    ins = int(sample.quality.get(f"{prefix}_insertions", 0) or 0)
    ref_len = int(sample.quality.get(f"{prefix}_reference_length", 0) or 0)
    cer = sample.quality.get(f"{prefix}_cer")
    acc = _char_acc(cer)
    bucket["n"] += 1
    bucket["错字"] += sub
    bucket["少字"] += dele
    bucket["多字"] += ins
    bucket["dis"] += sub + dele + ins
    bucket["ref_len"] += ref_len
    if acc is not None:
        bucket["sum_acc"] += float(acc)


def _overall_acc(bucket: dict[str, float | int]) -> float | None:
    ref_len = int(bucket["ref_len"])
    if ref_len <= 0:
        return None
    return round(max(0.0, 1.0 - int(bucket["dis"]) / ref_len), 6)


def _mean_acc(bucket: dict[str, float | int]) -> float | None:
    n = int(bucket["n"])
    if n <= 0:
        return None
    return round(float(bucket["sum_acc"]) / n, 6)


def _summary_rows(
    name: str,
    totals: dict[str, Any],
    *,
    group_by: str,
) -> list[dict[str, Any]]:
    rows = [
        {
            "对比": name,
            group_by: "总计",
            "参与行数": totals["n"],
            "跳过行数": totals["n_skip"],
            "错字": totals["错字"],
            "少字": totals["少字"],
            "多字": totals["多字"],
            "总编辑距离": totals["dis"],
            "总基准字数": totals["ref_len"],
            "总体字准率": _overall_acc(totals),
            "平均字准率": _mean_acc(totals),
        }
    ]
    by_group = totals.get("by_group") or {}
    for gname in sorted(by_group.keys(), key=lambda x: (x == "unclassified", str(x))):
        g = by_group[gname]
        rows.append(
            {
                "对比": name,
                group_by: gname,
                "参与行数": g["n"],
                "跳过行数": g["n_skip"],
                "错字": g["错字"],
                "少字": g["少字"],
                "多字": g["多字"],
                "总编辑距离": g["dis"],
                "总基准字数": g["ref_len"],
                "总体字准率": _overall_acc(g),
                "平均字准率": _mean_acc(g),
            }
        )
    return rows


def _build_xlsx_summaries(
    samples: list[Sample],
    scored_by_prefix: dict[str, set[str]],
    *,
    prefixes: list[str],
    bucket_key: str,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for prefix in prefixes:
        scored_ids = scored_by_prefix.get(prefix) or set()
        totals: dict[str, Any] = _empty_bucket()
        totals["by_group"] = {}
        totals["n_skip"] = len(samples) - len(scored_ids)
        for sample in samples:
            gname = _bucket_value(sample, bucket_key)
            group = totals["by_group"].setdefault(gname, _empty_bucket())
            if sample.id not in scored_ids:
                group["n_skip"] += 1
                continue
            _add_sample_metrics(totals, sample, prefix)
            _add_sample_metrics(group, sample, prefix)
        name = f"{prefix}_text ← label"
        summaries.extend(_summary_rows(name, totals, group_by="type"))
    return summaries


def _write_xlsx(
    path: Path,
    samples: list[Sample],
    scored_by_prefix: dict[str, set[str]],
    *,
    prefixes: list[str],
    bucket_key: str,
) -> None:
    rows = []
    for sample in samples:
        scored_prefixes = {
            prefix for prefix in prefixes if sample.id in (scored_by_prefix.get(prefix) or set())
        }
        rows.append(
            _sample_row(
                sample,
                prefixes=prefixes,
                bucket_key=bucket_key,
                scored_prefixes=scored_prefixes,
            )
        )
    summary = _build_xlsx_summaries(
        samples,
        scored_by_prefix,
        prefixes=prefixes,
        bucket_key=bucket_key,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame(rows).to_excel(writer, index=False, sheet_name="结果")
        summary_df = pd.DataFrame(summary)
        summary_df.to_excel(writer, index=False, sheet_name="统计摘要")
        summary_df.to_excel(writer, index=False, sheet_name="按type统计")


@register_operator
class EvaluationReportOperator(ManifestOperator):
    """Aggregate multi-model sample metrics vs gold; optional pairwise regression gates."""

    name = "evaluation_report"
    version = "1.2.0"
    category = "quality"

    def run(self, samples: list[Sample], config: OperatorConfig) -> list[Sample]:
        prefixes = _resolve_prefixes(config, samples)
        if not prefixes:
            raise ValueError("evaluation_report requires at least one model prefix")
        bucket_key = str(config.params.get("bucket_key", "classification_bucket"))
        allow_missing_gold = bool(config.params.get("allow_missing_gold", True))
        baseline = config.params.get("baseline_prefix")
        candidate = config.params.get("candidate_prefix")
        if baseline is not None:
            baseline = str(baseline).strip() or None
        if candidate is not None:
            candidate = str(candidate).strip() or None

        scored_by_prefix: dict[str, set[str]] = {
            prefix: {s.id for s in samples if _has_metric(s, prefix)} for prefix in prefixes
        }
        any_scored_ids = set().union(*scored_by_prefix.values()) if scored_by_prefix else set()
        missing = [sample for sample in samples if sample.id not in any_scored_ids]
        if missing and not allow_missing_gold:
            raise ValueError(
                f"evaluation metrics missing for {len(missing)} samples: "
                f"{[s.id for s in missing[:10]]}"
            )
        if not any_scored_ids:
            raise ValueError(
                "evaluation_report found 0 scored samples with gold metrics; "
                "check gold_text coverage and text_metrics config"
            )

        duplicate_ids = sorted(
            {
                sample_id
                for sample_id in [s.id for s in samples]
                if sum(1 for s in samples if s.id == sample_id) > 1
            }
        )
        if duplicate_ids:
            raise ValueError(
                f"evaluation_report requires unique sample ids; duplicates: {duplicate_ids[:10]}"
            )

        overall = {
            prefix: _corpus([s for s in samples if s.id in scored_by_prefix[prefix]], prefix)
            for prefix in prefixes
        }
        report: dict[str, Any] = {
            "model_prefixes": prefixes,
            "baseline_prefix": baseline,
            "candidate_prefix": candidate,
            "gold_coverage": {
                "total": len(samples),
                "scored": len(any_scored_ids),
                "missing_gold": len(missing),
                "missing_gold_ids": [s.id for s in missing[:200]],
                "missing_gold_truncated": bool(len(missing) > 200),
                "scored_by_model": {
                    prefix: len(scored_by_prefix[prefix]) for prefix in prefixes
                },
            },
            "overall": overall,
            "buckets": {},
        }
        buckets: dict[str, list[Sample]] = defaultdict(list)
        for sample in samples:
            if sample.id not in any_scored_ids:
                continue
            buckets[_bucket_value(sample, bucket_key)].append(sample)
        for name, members in sorted(buckets.items()):
            report["buckets"][name] = {
                prefix: _corpus(
                    [s for s in members if s.id in scored_by_prefix[prefix]],
                    prefix,
                )
                for prefix in prefixes
            }

        if (
            baseline
            and candidate
            and baseline in overall
            and candidate in overall
        ):
            pair_scored = [
                s
                for s in samples
                if s.id in scored_by_prefix.get(baseline, set())
                and s.id in scored_by_prefix.get(candidate, set())
            ]
            baseline_cer = overall[baseline]["corpus_cer"]
            candidate_cer = overall[candidate]["corpus_cer"]
            report["delta_cer"] = candidate_cer - baseline_cer
            report["delta_char_acc"] = (
                (overall[candidate]["corpus_char_acc"] or 0.0)
                - (overall[baseline]["corpus_char_acc"] or 0.0)
            )
            report["paired_bootstrap"] = _bootstrap_delta(
                pair_scored,
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
                if bucket is not None and view is None:
                    results.append(
                        {
                            "name": name,
                            "bucket": bucket,
                            "delta_cer": None,
                            "limit": max_regression,
                            "passed": True,
                            "skipped": True,
                            "reason": "bucket_absent",
                        }
                    )
                    continue
                if view is None or baseline not in view or candidate not in view:
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
        else:
            report["delta_cer"] = None
            report["delta_char_acc"] = None
            report["paired_bootstrap"] = {"iterations": 0, "seed": 0, "ci95": []}
            report["gates"] = []
            report["passed"] = True

        if config.run_dir is None:
            raise ValueError("evaluation_report requires a pipeline run directory")
        report_path = Path(config.run_dir) / "reports" / "evaluation.json"
        atomic_write_json(report_path, report)

        export_xlsx = config.params.get("export_xlsx")
        if export_xlsx is None or export_xlsx is True:
            xlsx_path = Path(config.run_dir) / "reports" / "evaluation.xlsx"
        elif export_xlsx in (False, "", "false", "0"):
            xlsx_path = None
        else:
            xlsx_path = Path(str(export_xlsx))
        if xlsx_path is not None:
            _write_xlsx(
                xlsx_path,
                samples,
                scored_by_prefix,
                prefixes=prefixes,
                bucket_key=bucket_key,
            )
            report["export_xlsx"] = str(xlsx_path)
            atomic_write_json(report_path, report)
            logger.info(
                "evaluation xlsx written path={} scored={} missing_gold={} models={}",
                xlsx_path,
                len(any_scored_ids),
                len(missing),
                prefixes,
            )

        if missing:
            logger.warning(
                "evaluation skipped {} samples without gold metrics; examples={}",
                len(missing),
                [s.id for s in missing[:10]],
            )
        if not report["passed"] and bool(config.params.get("fail_on_regression", True)):
            failed = [item["name"] for item in report["gates"] if not item["passed"]]
            raise ValueError(f"evaluation regression gate failed: {failed}; report={report_path}")
        return list(samples)
