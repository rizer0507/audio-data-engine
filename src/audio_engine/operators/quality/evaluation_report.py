from __future__ import annotations

import random
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from audio_engine.core.artifacts import atomic_write_json
from audio_engine.core.operator import ManifestOperator, OperatorConfig
from audio_engine.core.registry import register_operator
from audio_engine.core.sample import Sample
from audio_engine.core.selection_v2.types import EMPTY_GOLD_TYPES
from audio_engine.core.source_naming import evaluation_report_dir
from audio_engine.metrics.business import (
    load_business_metric_config,
    prediction_completeness,
)
from audio_engine.metrics.gate import evaluate_release_gate, load_gate_config
from audio_engine.metrics.runner import MetricRunner, load_business_risk_config

# Types whose empty reference must not pollute the primary CER table.
_EMPTY_REF_TYPES = frozenset(EMPTY_GOLD_TYPES) | {
    "noise",
    "true_silence",
    "invalid_audio",
}


def _resolve_authoritative_report_dir(config: OperatorConfig) -> Path:
    """Prefer staged ``datasets/stage3/reports/{eval_name}``; else run_dir/reports."""
    report_dir = config.params.get("report_dir")
    if report_dir is not None and str(report_dir).strip():
        return Path(str(report_dir).strip())
    eval_name = config.params.get("eval_name") or config.params.get("report_name")
    if eval_name is not None and str(eval_name).strip():
        return evaluation_report_dir(str(eval_name).strip())
    if config.run_dir is None:
        raise ValueError(
            "evaluation_report requires run_dir, or params.eval_name / params.report_dir"
        )
    return Path(config.run_dir) / "reports"


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
    for key in (bucket_key, "type", "classification_bucket", "subtype"):
        value = labels.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return "unclassified"


def _effective_bucket(sample: Sample, bucket_key: str) -> str:
    """Prefer subtype when present (v1 type + v2 subtype transition)."""
    labels = sample.labels or {}
    subtype = str(labels.get("subtype") or "").strip()
    primary = _bucket_value(sample, bucket_key)
    if subtype and subtype != primary:
        return f"{primary}/{subtype}" if bucket_key != "subtype" else subtype
    return primary


def _is_empty_ref_type(sample: Sample, bucket_key: str) -> bool:
    labels = sample.labels or {}
    for key in (bucket_key, "type", "classification_bucket", "subtype"):
        value = str(labels.get(key) or "").strip()
        if value in _EMPTY_REF_TYPES:
            return True
    return False


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
    dis = int(bucket["dis"])
    n = int(bucket["n"])
    if n <= 0:
        return None
    # 空参考（如【无声音输出】幻觉）：无基准字时，有编辑则总体字准为 0，否则为 1
    if ref_len <= 0:
        return 0.0 if dis > 0 else 1.0
    return round(max(0.0, 1.0 - dis / ref_len), 6)


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


def _acc_delta(value: Any, base: Any) -> float | None:
    if value is None or base is None:
        return None
    try:
        return round(float(value) - float(base), 6)
    except (TypeError, ValueError):
        return None


def _resolve_xlsx_base_prefix(config: OperatorConfig, prefixes: list[str]) -> str | None:
    """Base model for xlsx delta columns: base_prefix > baseline_prefix > first prefix."""
    for key in ("base_prefix", "baseline_prefix"):
        raw = config.params.get(key)
        if raw is None:
            continue
        name = str(raw).strip()
        if name and name in prefixes:
            return name
    return prefixes[0] if prefixes else None


def _annotate_relative_base_deltas(
    rows: list[dict[str, Any]],
    *,
    base_prefix: str | None,
    group_by: str = "type",
) -> list[dict[str, Any]]:
    """Append vs-base char-acc deltas for every non-base model / type row."""
    overall_key = "相对base总体字准率"
    mean_key = "相对base平均字准率"
    if not rows:
        return rows
    if not base_prefix:
        for row in rows:
            row[overall_key] = None
            row[mean_key] = None
        return rows
    base_name = f"{base_prefix}_text ← label"
    base_by_group = {
        row.get(group_by): row for row in rows if row.get("对比") == base_name
    }
    for row in rows:
        if row.get("对比") == base_name:
            row[overall_key] = None
            row[mean_key] = None
            continue
        base_row = base_by_group.get(row.get(group_by))
        if base_row is None:
            row[overall_key] = None
            row[mean_key] = None
            continue
        row[overall_key] = _acc_delta(row.get("总体字准率"), base_row.get("总体字准率"))
        row[mean_key] = _acc_delta(row.get("平均字准率"), base_row.get("平均字准率"))
    return rows


def _build_xlsx_summaries(
    samples: list[Sample],
    scored_by_prefix: dict[str, set[str]],
    *,
    prefixes: list[str],
    bucket_key: str,
    base_prefix: str | None = None,
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
    return _annotate_relative_base_deltas(
        summaries,
        base_prefix=base_prefix if base_prefix is not None else (prefixes[0] if prefixes else None),
        group_by="type",
    )


def _write_xlsx(
    path: Path,
    samples: list[Sample],
    scored_by_prefix: dict[str, set[str]],
    *,
    prefixes: list[str],
    bucket_key: str,
    base_prefix: str | None = None,
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
        base_prefix=base_prefix,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame(rows).to_excel(writer, index=False, sheet_name="结果")
        summary_df = pd.DataFrame(summary)
        summary_df.to_excel(writer, index=False, sheet_name="统计摘要")
        summary_df.to_excel(writer, index=False, sheet_name="按type统计")


def _attach_business_metrics(
    report: dict[str, Any],
    samples: list[Sample],
    prefixes: list[str],
    config: OperatorConfig,
) -> None:
    """Extend evaluation.json with business_metrics_v1 + optional release gate."""
    cfg_path = config.params.get("business_metrics_config") or config.params.get(
        "business_risk_config"
    )
    enable = config.params.get("enable_business_metrics")
    if enable is False:
        return
    if cfg_path is None and enable is not True:
        # Opt-in via explicit path or enable flag (keeps v1/v2 CER-only path stable).
        return
    raw = load_business_risk_config(cfg_path) if cfg_path else {}
    runner = MetricRunner(
        business_config_path=None,
        business_config=raw,
    )
    business_report = runner.score_corpus(samples, prefixes)
    report["business_metrics"] = business_report
    report["prediction_completeness"] = {
        prefix: prediction_completeness(samples, prefix) for prefix in prefixes
    }
    incomplete_models = [
        prefix
        for prefix, info in report["prediction_completeness"].items()
        if not info.get("is_complete")
    ]
    if incomplete_models:
        report["publish_status"] = "incomplete"
        report["publish_reasons"] = [
            f"incomplete predictions for models: {incomplete_models}"
        ]

    baseline = config.params.get("baseline_prefix") or config.params.get("base_prefix")
    candidate = config.params.get("candidate_prefix")
    if baseline is not None:
        baseline = str(baseline).strip() or None
    if candidate is not None:
        candidate = str(candidate).strip() or None
    if not baseline or not candidate or baseline not in prefixes or candidate not in prefixes:
        if "publish_status" not in report:
            report["publish_status"] = "diagnostic_only"
        return

    gate_raw = raw.get("gate") or {}
    gate_overrides = config.params.get("business_gate") or {}
    if isinstance(gate_overrides, dict):
        gate_raw = {**gate_raw, **gate_overrides}
    gate_cfg = load_gate_config(gate_raw)
    biz_cfg = load_business_metric_config(raw.get("business") or raw)
    gate_result = evaluate_release_gate(
        samples,
        baseline=baseline,
        candidate=candidate,
        gate=gate_cfg,
        business=biz_cfg,
        judge=runner.judge,
    )
    report["business_gate"] = gate_result.to_dict()
    report["publish_status"] = gate_result.status
    report["publish_reasons"] = list(gate_result.reasons)
    # Report the two fixed evaluation populations independently; a pooled
    # improvement must not hide a regression in either population.
    by_role = {}
    for role in ("eval_core", "eval_random"):
        subset = [s for s in samples if (s.labels.get("eval_role") or s.labels.get("split")
                  or s.labels.get("dataset_role")) == role]
        if subset:
            by_role[role] = evaluate_release_gate(subset, baseline=baseline, candidate=candidate,
                gate=gate_cfg, business=biz_cfg, judge=runner.judge).to_dict()
    if by_role:
        report["business_gate_by_eval_role"] = by_role
        statuses = [v["status"] for v in by_role.values()]
        for status in ("incomplete", "fail", "needs_review"):
            if status in statuses:
                report["publish_status"] = status
                report["publish_reasons"].append(f"per-eval-role gate: {status}")
                break
    if incomplete_models:
        report["publish_status"] = "incomplete"
        report["publish_reasons"].append(f"incomplete predictions for models: {incomplete_models}")


@register_operator
class EvaluationReportOperator(ManifestOperator):
    """Aggregate multi-model sample metrics vs gold; optional pairwise regression gates."""

    name = "evaluation_report"
    version = "1.6.0"
    category = "quality"

    def run(self, samples: list[Sample], config: OperatorConfig) -> list[Sample]:
        if config.params.get("require_formal_v3"):
            from audio_engine.core.annotation_v3.gold import has_formal_gold_evidence
            if any(not has_formal_gold_evidence(s, require_dual=True) for s in samples):
                raise ValueError("formal v3 evaluation requires complete independently reviewed gold")
            if any((s.labels.get("split") or s.labels.get("dataset_role")) not in {"eval_core", "eval_random"} for s in samples):
                raise ValueError("formal v3 evaluation requires a frozen eval split")
            from audio_engine.core.catalog import ArtifactCatalog
            from audio_engine.core.manifest import Manifest
            release_ids = {s.labels.get("release_id") for s in samples}
            if len(release_ids) != 1 or None in release_ids or "" in release_ids:
                raise ValueError("formal v3 evaluation requires exactly one release_id")
            catalog = ArtifactCatalog(config.params.get("catalog_dir") or "data/catalog")
            release = catalog.get_release(next(iter(release_ids)))
            roles = {s.labels.get("split") or s.labels.get("dataset_role") for s in samples}
            expected = {}
            for role in roles:
                record = catalog.get(release.outputs[role], verify=True)
                expected.update({s.id: s for s in Manifest.load(record.uri)})
            if len({s.id for s in samples}) != len(samples) or {s.id for s in samples} != set(expected):
                raise ValueError("evaluation IDs differ from frozen release membership")
            for sample in samples:
                frozen = expected[sample.id]
                if sample.sha256 != frozen.sha256 or sample.labels.get("gold_text") != frozen.labels.get("gold_text"):
                    raise ValueError(f"evaluation audio/gold differs from frozen release: {sample.id}")
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
            prefix: _corpus(
                [
                    s
                    for s in samples
                    if s.id in scored_by_prefix[prefix]
                    and not _is_empty_ref_type(s, bucket_key)
                ],
                prefix,
            )
            for prefix in prefixes
        }
        empty_ref_overall = {
            prefix: _corpus(
                [
                    s
                    for s in samples
                    if s.id in scored_by_prefix[prefix] and _is_empty_ref_type(s, bucket_key)
                ],
                prefix,
            )
            for prefix in prefixes
        }
        report: dict[str, Any] = {
            "model_prefixes": prefixes,
            "baseline_prefix": baseline,
            "candidate_prefix": candidate,
            "eval_release": str(
                config.params.get("eval_release")
                or config.params.get("eval_name")
                or ""
            ),
            "eval_trust": str(
                next(
                    (
                        s.labels.get("eval_trust")
                        for s in samples
                        if s.labels.get("eval_trust")
                    ),
                    config.params.get("eval_trust") or "unknown",
                )
            ),
            "gold_coverage": {
                "total": len(samples),
                "scored": len(any_scored_ids),
                "missing_gold": len(missing),
                "missing_gold_ids": [s.id for s in missing[:200]],
                "missing_gold_truncated": bool(len(missing) > 200),
                "scored_by_model": {
                    prefix: len(scored_by_prefix[prefix]) for prefix in prefixes
                },
                "empty_ref_excluded_from_main_cer": {
                    prefix: int(empty_ref_overall[prefix].get("samples") or 0)
                    for prefix in prefixes
                },
            },
            "overall": overall,
            "empty_ref_slice": empty_ref_overall,
            "buckets": {},
        }
        buckets: dict[str, list[Sample]] = defaultdict(list)
        for sample in samples:
            if sample.id not in any_scored_ids:
                continue
            buckets[_effective_bucket(sample, bucket_key)].append(sample)
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

        _attach_business_metrics(report, samples, prefixes, config)

        auth_dir = _resolve_authoritative_report_dir(config)
        auth_dir.mkdir(parents=True, exist_ok=True)
        run_reports = (
            Path(config.run_dir) / "reports" if config.run_dir is not None else None
        )
        report_path = auth_dir / "evaluation.json"
        report["report_dir"] = Path(auth_dir).as_posix()
        report["run_id"] = Path(config.run_dir).name if config.run_dir is not None else None
        report["generated_at"] = datetime.now(timezone.utc).isoformat()
        atomic_write_json(report_path, report)

        export_xlsx = config.params.get("export_xlsx")
        if export_xlsx is None or export_xlsx is True:
            xlsx_path = auth_dir / "evaluation.xlsx"
        elif export_xlsx in (False, "", "false", "0"):
            xlsx_path = None
        else:
            xlsx_path = Path(str(export_xlsx))
        if xlsx_path is not None:
            xlsx_base = _resolve_xlsx_base_prefix(config, prefixes)
            _write_xlsx(
                xlsx_path,
                samples,
                scored_by_prefix,
                prefixes=prefixes,
                bucket_key=bucket_key,
                base_prefix=xlsx_base,
            )
            report["export_xlsx"] = Path(xlsx_path).as_posix()
            report["xlsx_base_prefix"] = xlsx_base
            atomic_write_json(report_path, report)
            logger.info(
                "evaluation xlsx written path={} scored={} missing_gold={} models={} base={}",
                xlsx_path,
                len(any_scored_ids),
                len(missing),
                prefixes,
                xlsx_base,
            )

        # Keep a run-local copy so existing tooling / tests that look under runs/ still work.
        if run_reports is not None:
            try:
                same_dir = run_reports.resolve() == auth_dir.resolve()
            except OSError:
                same_dir = False
            if not same_dir:
                run_reports.mkdir(parents=True, exist_ok=True)
                atomic_write_json(run_reports / "evaluation.json", report)
                if xlsx_path is not None and xlsx_path.is_file():
                    shutil.copy2(xlsx_path, run_reports / "evaluation.xlsx")

        if missing:
            logger.warning(
                "evaluation skipped {} samples without gold metrics; examples={}",
                len(missing),
                [s.id for s in missing[:10]],
            )
        if not report["passed"] and bool(config.params.get("fail_on_regression", True)):
            failed = [item["name"] for item in report["gates"] if not item["passed"]]
            raise ValueError(f"evaluation regression gate failed: {failed}; report={report_path}")
        fail_on_business = bool(config.params.get("fail_on_business_gate", False))
        if fail_on_business:
            status = str(report.get("publish_status") or "")
            if status != "pass":
                raise ValueError(
                    f"business publish gate status={status}; "
                    f"reasons={report.get('publish_reasons')}; report={report_path}"
                )
        return list(samples)
