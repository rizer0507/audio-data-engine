"""Release / regression gate for business metrics (needs_review when uncalibrated)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from audio_engine.metrics.business import (
    BusinessMetricConfig,
    compute_business_metrics_for_model,
    group_bootstrap_metric_delta,
    prediction_completeness,
)
from audio_engine.metrics.semantic_judge import SemanticJudge, load_semantic_judge
from audio_engine.core.sample import Sample


GATE_PASS = "pass"
GATE_FAIL = "fail"
GATE_NEEDS_REVIEW = "needs_review"
GATE_INCOMPLETE = "incomplete"


@dataclass
class GateConfig:
    """Publish gate. Missing calibrated thresholds → needs_review (never auto-pass)."""

    primary_metric: str = "nfr"
    # Lower is better for risk metrics; higher is better for positive_retention.
    primary_higher_is_better: bool = False
    min_improvement: float | None = None
    # Non-inferiority tolerances (absolute): candidate may be worse by at most this amount.
    cer_non_inferiority: float | None = None
    positive_retention_non_inferiority: float | None = None
    nshr_non_inferiority: float | None = None
    confidence: float = 0.95
    min_denominator: int = 30
    max_semantic_unk_rate: float | None = None
    require_complete_predictions: bool = True
    bootstrap_iterations: int = 1000
    bootstrap_seed: int = 42
    # When True, missing min_improvement / tolerances force needs_review.
    require_calibrated_thresholds: bool = True


def load_gate_config(raw: dict[str, Any] | None) -> GateConfig:
    data = dict(raw or {})
    primary = str(data.get("primary_metric") or "nfr").strip().lower()
    higher = data.get("primary_higher_is_better")
    if higher is None:
        higher = primary in {"positive_retention"}
    return GateConfig(
        primary_metric=primary,
        primary_higher_is_better=bool(higher),
        min_improvement=(
            float(data["min_improvement"])
            if data.get("min_improvement") is not None
            else None
        ),
        cer_non_inferiority=(
            float(data["cer_non_inferiority"])
            if data.get("cer_non_inferiority") is not None
            else None
        ),
        positive_retention_non_inferiority=(
            float(data["positive_retention_non_inferiority"])
            if data.get("positive_retention_non_inferiority") is not None
            else None
        ),
        nshr_non_inferiority=(
            float(data["nshr_non_inferiority"])
            if data.get("nshr_non_inferiority") is not None
            else None
        ),
        confidence=float(data.get("confidence", 0.95)),
        min_denominator=int(data.get("min_denominator", 30)),
        max_semantic_unk_rate=(
            float(data["max_semantic_unk_rate"])
            if data.get("max_semantic_unk_rate") is not None
            else None
        ),
        require_complete_predictions=bool(
            data.get("require_complete_predictions", True)
        ),
        bootstrap_iterations=int(data.get("bootstrap_iterations", 1000)),
        bootstrap_seed=int(data.get("bootstrap_seed", 42)),
        require_calibrated_thresholds=bool(
            data.get("require_calibrated_thresholds", True)
        ),
    )


@dataclass
class GateResult:
    status: str
    reasons: list[str] = field(default_factory=list)
    checks: list[dict[str, Any]] = field(default_factory=list)
    bootstrap: dict[str, Any] = field(default_factory=dict)
    baseline_model: str = ""
    candidate_model: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reasons": list(self.reasons),
            "checks": list(self.checks),
            "bootstrap": dict(self.bootstrap),
            "baseline_model": self.baseline_model,
            "candidate_model": self.candidate_model,
            "passed": self.status == GATE_PASS,
        }


def _metric_value(block: dict[str, Any], name: str) -> float | None:
    payload = (block.get("metrics") or {}).get(name) or {}
    value = payload.get("value")
    return None if value is None else float(value)


def _metric_den(block: dict[str, Any], name: str) -> int:
    payload = (block.get("metrics") or {}).get(name) or {}
    return int(payload.get("denominator") or 0)


def evaluate_release_gate(
    samples: list[Sample],
    *,
    baseline: str,
    candidate: str,
    gate: GateConfig,
    business: BusinessMetricConfig | None = None,
    judge: SemanticJudge | None = None,
) -> GateResult:
    """Compare candidate vs baseline on fixed eval samples.

    Rules (012-E §3 / §7):
    - Any failed/missing prediction → incomplete (diagnostic OK, publish blocked).
    - Missing calibrated thresholds or insufficient denominator → needs_review.
    - Pass only when primary improves AND protection metrics are non-inferior.
    """
    biz = business or BusinessMetricConfig()
    judge = judge or load_semantic_judge(
        lexicon_path=biz.lexicon_path, judge_version=biz.judge_version
    )
    result = GateResult(
        status=GATE_NEEDS_REVIEW,
        baseline_model=baseline,
        candidate_model=candidate,
    )

    base_complete = prediction_completeness(samples, baseline)
    cand_complete = prediction_completeness(samples, candidate)
    if gate.require_complete_predictions and (
        not base_complete["is_complete"] or not cand_complete["is_complete"]
    ):
        result.status = GATE_INCOMPLETE
        result.reasons.append(
            "predictions incomplete on fixed eval set "
            f"(baseline_missing={base_complete['incomplete']}, "
            f"candidate_missing={cand_complete['incomplete']})"
        )
        result.checks.append(
            {
                "name": "prediction_completeness",
                "passed": False,
                "baseline": base_complete,
                "candidate": cand_complete,
            }
        )
        return result

    if gate.require_calibrated_thresholds and (
        gate.min_improvement is None
        or gate.cer_non_inferiority is None
        or gate.positive_retention_non_inferiority is None
        or gate.nshr_non_inferiority is None
        or gate.max_semantic_unk_rate is None
    ):
        result.status = GATE_NEEDS_REVIEW
        result.reasons.append(
            "gate thresholds not fully calibrated "
            "(min_improvement / cer / positive_retention / nshr); "
            "refusing auto-pass"
        )
        result.checks.append(
            {
                "name": "calibration",
                "passed": False,
                "min_improvement": gate.min_improvement,
                "cer_non_inferiority": gate.cer_non_inferiority,
                "positive_retention_non_inferiority": gate.positive_retention_non_inferiority,
                "nshr_non_inferiority": gate.nshr_non_inferiority,
            }
        )
        # Still compute diagnostics below, but status stays needs_review.
        calibrated = False
    else:
        calibrated = True
    if any(not s.labels.get("leakage_group_id") for s in samples):
        calibrated = False
        result.reasons.append("missing leakage_group_id: paired group inference unavailable")

    base_block = compute_business_metrics_for_model(
        samples, baseline, judge=judge, config=biz
    )
    cand_block = compute_business_metrics_for_model(
        samples, candidate, judge=judge, config=biz
    )
    # Strip metric_objects
    base_block = {k: v for k, v in base_block.items() if k != "metric_objects"}
    cand_block = {k: v for k, v in cand_block.items() if k != "metric_objects"}

    primary = gate.primary_metric
    base_v = _metric_value(base_block, primary)
    cand_v = _metric_value(cand_block, primary)
    primary_den = min(_metric_den(base_block, primary), _metric_den(cand_block, primary))
    if primary_den < gate.min_denominator or base_v is None or cand_v is None:
        result.status = GATE_NEEDS_REVIEW
        result.reasons.append(
            f"primary metric {primary} insufficient "
            f"(den={primary_den}, min={gate.min_denominator}, "
            f"base={base_v}, cand={cand_v})"
        )
        result.checks.append(
            {
                "name": "primary_denominator",
                "metric": primary,
                "denominator": primary_den,
                "min_denominator": gate.min_denominator,
                "passed": False,
            }
        )
        calibrated = False

    unk = _metric_value(cand_block, "semantic_unk_rate")
    if gate.max_semantic_unk_rate is not None and unk is not None:
        unk_ok = unk <= gate.max_semantic_unk_rate
        result.checks.append(
            {
                "name": "semantic_unk_rate",
                "value": unk,
                "limit": gate.max_semantic_unk_rate,
                "passed": unk_ok,
            }
        )
        if not unk_ok:
            result.status = GATE_FAIL if calibrated else GATE_NEEDS_REVIEW
            result.reasons.append(
                f"semantic_unk_rate {unk} > max {gate.max_semantic_unk_rate}"
            )

    improvement = None
    if base_v is not None and cand_v is not None:
        if gate.primary_higher_is_better:
            improvement = cand_v - base_v
        else:
            improvement = base_v - cand_v
        primary_ok = (
            gate.min_improvement is not None and improvement >= gate.min_improvement
        )
        result.checks.append(
            {
                "name": "primary_improvement",
                "metric": primary,
                "baseline": base_v,
                "candidate": cand_v,
                "improvement": improvement,
                "min_improvement": gate.min_improvement,
                "passed": bool(primary_ok) if gate.min_improvement is not None else False,
            }
        )
        if gate.min_improvement is not None and not primary_ok:
            result.reasons.append(
                f"primary {primary} improvement {improvement} "
                f"< min_improvement {gate.min_improvement}"
            )

    # Protection metrics (non-inferiority)
    def _protect(name: str, tol: float | None, *, higher_better: bool) -> None:
        nonlocal calibrated
        if tol is None:
            return
        bv = _metric_value(base_block, name)
        cv = _metric_value(cand_block, name)
        bden = _metric_den(base_block, name)
        cden = _metric_den(cand_block, name)
        if min(bden, cden) < gate.min_denominator or bv is None or cv is None:
            calibrated = False
            result.checks.append(
                {
                    "name": f"protect_{name}",
                    "passed": False,
                    "reason": "insufficient_protection_denominator",
                    "baseline": bv,
                    "candidate": cv,
                    "baseline_den": bden,
                    "candidate_den": cden,
                }
            )
            return
        if higher_better:
            # candidate may drop by at most tol
            ok = (bv - cv) <= tol
            delta = bv - cv
        else:
            # candidate may rise by at most tol
            ok = (cv - bv) <= tol
            delta = cv - bv
        result.checks.append(
            {
                "name": f"protect_{name}",
                "baseline": bv,
                "candidate": cv,
                "delta": delta,
                "tolerance": tol,
                "passed": ok,
            }
        )
        if not ok:
            result.reasons.append(
                f"protection {name} degraded beyond tolerance "
                f"(delta={delta}, tol={tol})"
            )

    _protect("cer", gate.cer_non_inferiority, higher_better=False)
    _protect(
        "positive_retention",
        gate.positive_retention_non_inferiority,
        higher_better=True,
    )
    _protect("nshr", gate.nshr_non_inferiority, higher_better=False)

    result.bootstrap = group_bootstrap_metric_delta(
        samples,
        baseline,
        candidate,
        primary,
        judge=judge,
        config=biz,
        iterations=gate.bootstrap_iterations,
        seed=gate.bootstrap_seed,
        confidence=gate.confidence,
    )
    intervals = {primary: result.bootstrap}
    for name in ("cer", "positive_retention", "nshr"):
        if name not in intervals:
            intervals[name] = group_bootstrap_metric_delta(
                samples, baseline, candidate, name, judge=judge, config=biz,
                iterations=gate.bootstrap_iterations, seed=gate.bootstrap_seed,
                confidence=gate.confidence)
    result.bootstrap = {"metrics": intervals}
    limits = {primary: -gate.min_improvement if gate.min_improvement is not None else None,
              "cer": gate.cer_non_inferiority,
              "positive_retention": gate.positive_retention_non_inferiority,
              "nshr": gate.nshr_non_inferiority}
    for name, limit in limits.items():
        interval = intervals[name]
        ci = interval.get("ci") or []
        if (limit is None or len(ci) != 2 or interval.get("group_count", 0) < 2
            or interval.get("n_valid", 0) < max(1, int(gate.bootstrap_iterations * 0.9))):
            calibrated = False
            result.checks.append({"name": f"ci_{name}", "passed": False, "reason": "insufficient_interval_evidence"})
        else:
            ok = ci[1] <= limit
            result.checks.append({"name": f"ci_{name}", "passed": ok, "ci": ci, "upper_limit": limit})
            if not ok:
                result.reasons.append(f"{name} interval does not establish improvement/non-inferiority")

    if not calibrated or result.status in {GATE_FAIL, GATE_INCOMPLETE}:
        if result.status not in {GATE_FAIL, GATE_INCOMPLETE}:
            result.status = GATE_NEEDS_REVIEW
        return result

    failed = [c for c in result.checks if c.get("passed") is False]
    if failed or result.reasons:
        result.status = GATE_FAIL
        return result

    result.status = GATE_PASS
    result.reasons.append("primary improved and protection metrics within tolerance")
    return result
