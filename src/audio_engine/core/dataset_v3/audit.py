"""Pseudo-label independent audit gate (dataset_policy / annotation_v3 stage C).

Group-balanced mislabel rate + Wilson 95% upper bound; protected layers;
critical semantic errors must be zero. Calibration samples must not self-prove quality.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from audio_engine.core.annotation_v3.config import AnnotationConfig, ProtectedLayerPolicy
from audio_engine.core.annotation_v3.types import (
    GOLD_KIND_SPEECH,
    NON_FORMAL_GOLD_KINDS,
    STATE_ADJUDICATED,
    STATE_SECOND_REVIEW,
)
from audio_engine.core.sample import Sample
from audio_engine.core.selection_v3.text import comparison_text
from audio_engine.core.selection_v3.types import (
    RISK_FALSE_AFFIRMATION,
    RISK_FILLER_AFFIRMATION,
    RISK_NEGATION_FLIP,
    RISK_REJECTION_SANITIZATION,
)


def wilson_interval(
    errors: int,
    n: int,
    *,
    z: float = 1.96,
) -> tuple[float | None, float | None, float | None]:
    """Two-sided Wilson score interval; returns (point, lower, upper).

    When n==0 returns (None, None, None).
    """
    if n <= 0:
        return None, None, None
    errors = max(0, min(int(errors), int(n)))
    n = int(n)
    phat = errors / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = phat + z2 / (2 * n)
    margin = z * math.sqrt((phat * (1.0 - phat) + z2 / (4 * n)) / n)
    lower = (center - margin) / denom
    upper = (center + margin) / denom
    return phat, max(0.0, lower), min(1.0, upper)


def _stable_unit(seed: int | str, *parts: str) -> float:
    payload = "\0".join([str(seed), *parts]).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _risk_tags(sample: Sample) -> set[str]:
    raw = sample.labels.get("risk_tags") or []
    if isinstance(raw, str):
        return {x.strip() for x in raw.split(",") if x.strip()}
    return {str(x) for x in raw}


def _group_id(sample: Sample) -> str:
    return str(
        sample.labels.get("leakage_group_id")
        or sample.labels.get("duplicate_group_id")
        or sample.id
    )


CRITICAL_SEMANTIC_TAGS = frozenset(
    {
        RISK_NEGATION_FLIP,
        RISK_FALSE_AFFIRMATION,
        RISK_FILLER_AFFIRMATION,
        RISK_REJECTION_SANITIZATION,
    }
)


@dataclass
class AuditSampleRecord:
    sample_id: str
    leakage_group_id: str
    text_error: bool
    critical_semantic_error: bool
    layer_hits: list[str] = field(default_factory=list)
    candidate_comparison: str = ""
    human_comparison: str = ""


@dataclass
class LayerAuditReport:
    name: str
    n: int
    errors: int
    critical_semantic_errors: int
    point: float | None
    lower: float | None
    upper: float | None
    upper_bound_limit: float
    min_groups: int
    seed: str
    passed: bool
    reason: str
    sample_ids: list[str] = field(default_factory=list)


@dataclass
class PseudoAuditReport:
    passed: bool
    rule_version: str
    seed: int
    overall: LayerAuditReport
    protected_layers: list[LayerAuditReport]
    critical_semantic_errors: int
    stop_publish: bool
    reasons: list[str] = field(default_factory=list)
    excluded_calibration_count: int = 0
    sampled_group_count: int = 0
    records: list[AuditSampleRecord] = field(default_factory=list)
    scope: dict[str, str] = field(default_factory=dict)
    scope_digest: str = ""
    annotation_digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        def layer_dict(layer: LayerAuditReport) -> dict[str, Any]:
            return {
                "name": layer.name,
                "n": layer.n,
                "errors": layer.errors,
                "critical_semantic_errors": layer.critical_semantic_errors,
                "point": layer.point,
                "lower": layer.lower,
                "upper": layer.upper,
                "upper_bound_limit": layer.upper_bound_limit,
                "min_groups": layer.min_groups,
                "seed": layer.seed,
                "passed": layer.passed,
                "reason": layer.reason,
                "sample_ids": list(layer.sample_ids),
            }

        payload = {
            "passed": self.passed,
            "stop_publish": self.stop_publish,
            "rule_version": self.rule_version,
            "seed": self.seed,
            "statistic_name": "group_balanced_mislabel_rate",
            "excluded_calibration_count": self.excluded_calibration_count,
            "sampled_group_count": self.sampled_group_count,
            "critical_semantic_errors": self.critical_semantic_errors,
            "reasons": list(self.reasons),
            "overall": layer_dict(self.overall),
            "protected_layers": [layer_dict(x) for x in self.protected_layers],
            "records": [
                {
                    "sample_id": r.sample_id,
                    "leakage_group_id": r.leakage_group_id,
                    "text_error": r.text_error,
                    "critical_semantic_error": r.critical_semantic_error,
                    "layer_hits": list(r.layer_hits),
                }
                for r in self.records
            ],
            "scope": self.scope,
            "scope_digest": self.scope_digest,
            "annotation_digest": self.annotation_digest,
        }
        from audio_engine.core.dataset_v3.audit_plan import digest_payload
        return {**payload, "report_digest": digest_payload(payload)}


def _is_human_verified_gold(sample: Sample) -> bool:
    from audio_engine.core.annotation_v3.gold import has_formal_gold_evidence
    if not has_formal_gold_evidence(sample, require_dual=True):
        return False
    state = str(sample.labels.get("annotation_state") or "")
    if state not in {STATE_SECOND_REVIEW, STATE_ADJUDICATED, "human_accepted"}:
        # Dual-complete preferred; allow adjudicated/second_review only for audit truth.
        if not sample.labels.get("is_human_verified"):
            return False
        if state not in {STATE_SECOND_REVIEW, STATE_ADJUDICATED}:
            return False
    kind = str(sample.labels.get("gold_kind") or "")
    if kind in NON_FORMAL_GOLD_KINDS:
        return False
    if sample.labels.get("gold_text") is None and kind != GOLD_KIND_SPEECH:
        # non_speech empty is ok when key present as ""
        if "gold_text" not in sample.labels:
            return False
    return bool(sample.labels.get("is_human_verified"))


def _candidate_text(sample: Sample) -> str:
    return str(sample.labels.get("candidate_text") or "")


def _human_text(sample: Sample) -> str | None:
    if "gold_text" not in sample.labels:
        return None
    return sample.labels.get("gold_text")


def _text_error(sample: Sample) -> bool:
    human = _human_text(sample)
    if human is None:
        return True
    left = comparison_text(_candidate_text(sample))
    right = comparison_text(human)
    return left != right


def _critical_semantic_error(sample: Sample) -> bool:
    """Critical semantic mislabel: human polarity/risk disagrees with treating candidate as gold.

    Count when human marks a semantic risk that the pseudo path would have missed,
    or verified_error_tags include critical semantic tags, or human semantic conflicts
    with candidate polarity when candidate was accepted as pseudo_high.
    """
    tags = set(_risk_tags(sample))
    verified = sample.labels.get("verified_error_tags") or []
    if isinstance(verified, str):
        verified_set = {x.strip() for x in verified.split(",") if x.strip()}
    else:
        verified_set = {str(x) for x in verified}
    if verified_set & CRITICAL_SEMANTIC_TAGS:
        return True
    # If human gold differs AND human_semantic indicates polarity flip relative to candidate risk
    if _text_error(sample):
        human_sem = str(sample.labels.get("human_semantic") or "")
        if human_sem in {"positive", "negative"} and (
            tags & CRITICAL_SEMANTIC_TAGS or sample.labels.get("type") == "pseudo_high"
        ):
            # Any text error on a protected semantic sample is critical when polarity is clear.
            if human_sem == "negative" and any(
                tok in comparison_text(_candidate_text(sample))
                for tok in ("需要", "可以", "好的", "是的", "有")
            ):
                return True
            if human_sem == "positive" and any(
                tok in comparison_text(_candidate_text(sample))
                for tok in ("不需要", "不用", "不要", "没有", "不是")
            ):
                return True
    return False


def _match_layer(sample: Sample, policy: ProtectedLayerPolicy) -> bool:
    match = policy.match or {}
    if not match:
        return False
    tags = _risk_tags(sample)
    if "risk_tags_any" in match:
        wanted = {str(x) for x in match["risk_tags_any"]}
        if tags & wanted:
            return True
    if "human_semantic" in match:
        if str(sample.labels.get("human_semantic") or "") in {
            str(x) for x in match["human_semantic"]
        }:
            return True
    if "noise_band" in match:
        quality = sample.labels.get("quality")
        band = sample.labels.get("noise_band")
        if band is None and isinstance(quality, dict):
            band = quality.get("noise_band")
        if str(band or "") in {str(x) for x in match["noise_band"]}:
            return True
    if "human_noise" in match:
        if str(sample.labels.get("human_noise") or "") in {str(x) for x in match["human_noise"]}:
            return True
    if "types" in match:
        if str(sample.labels.get("type") or "") in {str(x) for x in match["types"]}:
            return True
    return False


def sample_group_balanced(
    samples: Sequence[Sample],
    *,
    seed: int | str,
    min_groups: int,
    salt: str = "overall",
) -> list[Sample]:
    """Uniformly sample leakage groups, one random sample per group."""
    by_group: dict[str, list[Sample]] = {}
    for sample in samples:
        by_group.setdefault(_group_id(sample), []).append(sample)

    group_ids = sorted(by_group.keys())
    ranked = sorted(
        group_ids,
        key=lambda g: (_stable_unit(seed, salt, "group", g), g),
    )
    selected_groups = ranked[: min(len(ranked), max(0, int(min_groups)))]
    out: list[Sample] = []
    for gid in selected_groups:
        members = sorted(by_group[gid], key=lambda s: s.id)
        pick = sorted(
            members,
            key=lambda s: (_stable_unit(seed, salt, "member", gid, s.id), s.id),
        )[0]
        out.append(pick)
    return out


def evaluate_pseudo_audit(
    candidates: Iterable[Sample],
    *,
    config: AnnotationConfig,
    rule_version: str = "selection_v3.0",
    plan: dict[str, Any] | None = None,
) -> PseudoAuditReport:
    """Evaluate independent audit gate on human-verified audit samples.

    ``candidates`` should be the audited subset already labeled by dual review
    (or equivalent), drawn from proposed auto-accept training pool — not from
    the calibration threshold-tuning pool.
    """
    audit_cfg = config.pseudo_audit
    all_samples = list(candidates)
    from audio_engine.core.dataset_v3.audit_plan import validate_audit_plan, candidate_signature
    plan_errors = []
    if plan is None:
        plan_errors.append("missing_frozen_audit_plan")
    else:
        validate_audit_plan(plan, config)
        indexed = {s.id: s for s in all_samples}
        if len(indexed) != len(all_samples):
            raise ValueError("duplicate audit sample ids")
        drawn = {sid for ids in plan["draws"].values() for sid in ids}
        for sid in drawn:
            sample = indexed.get(sid)
            if sample is None or not _is_human_verified_gold(sample):
                plan_errors.append(f"incomplete_frozen_draw:{sid}")
            elif candidate_signature(sample) != plan["scope"].get(sid):
                plan_errors.append(f"changed_audit_candidate:{sid}")
    excluded_cal = 0
    usable: list[Sample] = []
    for sample in all_samples:
        role = str(sample.labels.get("reservation_role") or sample.labels.get("dataset_role") or "")
        if audit_cfg.forbid_calibration_self_proof and role == "calibration":
            excluded_cal += 1
            continue
        usable.append(sample)

    # Only samples with human truth participate in error rate.
    labeled = [s for s in usable if _is_human_verified_gold(s)]
    overall_sample = sample_group_balanced(
        labeled,
        seed=audit_cfg.seed,
        min_groups=audit_cfg.min_groups_overall,
        salt="overall",
    )
    if plan is not None:
        overall_sample = [indexed[sid] for sid in plan["draws"]["overall"]
                          if sid in indexed and _is_human_verified_gold(indexed[sid])]

    records: list[AuditSampleRecord] = []
    for sample in overall_sample:
        text_err = _text_error(sample)
        crit = _critical_semantic_error(sample)
        hits = [
            name
            for name, policy in audit_cfg.protected_layers.items()
            if _match_layer(sample, policy)
        ]
        records.append(
            AuditSampleRecord(
                sample_id=sample.id,
                leakage_group_id=_group_id(sample),
                text_error=text_err,
                critical_semantic_error=crit,
                layer_hits=hits,
                candidate_comparison=comparison_text(_candidate_text(sample)),
                human_comparison=comparison_text(_human_text(sample) or ""),
            )
        )

    overall_errors = sum(1 for r in records if r.text_error)
    overall_crit = sum(1 for r in records if r.critical_semantic_error)
    point, lower, upper = wilson_interval(
        overall_errors, len(records), z=audit_cfg.wilson_z
    )
    reasons: list[str] = list(plan_errors)
    overall_ok = True
    overall_reason = "ok"
    if len(records) < audit_cfg.min_groups_overall:
        # Insufficient sample: full human confirm or hold — do not pass.
        if len(labeled) < audit_cfg.min_groups_overall:
            overall_ok = False
            overall_reason = (
                f"insufficient_groups: have {len(records)} < required {audit_cfg.min_groups_overall}"
            )
            reasons.append(overall_reason)
        else:
            # Used all available groups after sampling cap
            overall_ok = False
            overall_reason = (
                f"insufficient_audited_groups: have {len(records)} < required "
                f"{audit_cfg.min_groups_overall}"
            )
            reasons.append(overall_reason)
    if upper is not None and upper > audit_cfg.overall_upper_bound:
        overall_ok = False
        overall_reason = (
            f"overall_wilson_upper={upper:.6f} > {audit_cfg.overall_upper_bound}"
        )
        reasons.append(overall_reason)
    if overall_crit > audit_cfg.critical_semantic_errors_max:
        overall_ok = False
        reasons.append(
            f"critical_semantic_errors={overall_crit} > {audit_cfg.critical_semantic_errors_max}"
        )

    overall_report = LayerAuditReport(
        name="overall_group_balanced",
        n=len(records),
        errors=overall_errors,
        critical_semantic_errors=overall_crit,
        point=point,
        lower=lower,
        upper=upper,
        upper_bound_limit=audit_cfg.overall_upper_bound,
        min_groups=audit_cfg.min_groups_overall,
        seed=str(audit_cfg.seed),
        passed=overall_ok,
        reason=overall_reason if not overall_ok else "ok",
        sample_ids=[r.sample_id for r in records],
    )

    # Protected layers: separately sampled; do NOT mix into overall rate.
    layer_reports: list[LayerAuditReport] = []
    for name, policy in audit_cfg.protected_layers.items():
        layer_pool = [s for s in labeled if _match_layer(s, policy)]
        layer_sample = sample_group_balanced(
            layer_pool,
            seed=audit_cfg.seed,
            min_groups=policy.min_groups,
            salt=f"layer:{name}",
        )
        if plan is not None:
            layer_sample = [indexed[sid] for sid in plan["draws"].get(name, [])
                            if sid in indexed and _is_human_verified_gold(indexed[sid])]
        layer_errors = sum(1 for s in layer_sample if _text_error(s))
        layer_crit = sum(1 for s in layer_sample if _critical_semantic_error(s))
        lp, ll, lu = wilson_interval(layer_errors, len(layer_sample), z=audit_cfg.wilson_z)
        layer_ok = True
        layer_reason = "ok"
        if len(layer_sample) < policy.min_groups:
            layer_ok = False
            layer_reason = (
                f"insufficient_groups: have {len(layer_sample)} < required {policy.min_groups}; "
                "full human confirm or hold"
            )
            reasons.append(f"{name}: {layer_reason}")
        if lu is not None and lu > policy.upper_bound:
            layer_ok = False
            layer_reason = f"wilson_upper={lu:.6f} > {policy.upper_bound}"
            reasons.append(f"{name}: {layer_reason}")
        if layer_crit > audit_cfg.critical_semantic_errors_max:
            layer_ok = False
            layer_reason = f"critical_semantic_errors={layer_crit}"
            reasons.append(f"{name}: {layer_reason}")
        layer_reports.append(
            LayerAuditReport(
                name=name,
                n=len(layer_sample),
                errors=layer_errors,
                critical_semantic_errors=layer_crit,
                point=lp,
                lower=ll,
                upper=lu,
                upper_bound_limit=policy.upper_bound,
                min_groups=policy.min_groups,
                seed=f"{audit_cfg.seed}|layer:{name}",
                passed=layer_ok,
                reason=layer_reason if not layer_ok else "ok",
                sample_ids=[s.id for s in layer_sample],
            )
        )

    stop = overall_crit > audit_cfg.critical_semantic_errors_max or any(
        lr.critical_semantic_errors > audit_cfg.critical_semantic_errors_max
        for lr in layer_reports
    )
    if stop:
        reasons.append(
            "critical semantic mislabel observed — stop related pseudo-label publish; "
            "revise rules and re-audit independently; do not rewrite frozen Release"
        )

    passed = overall_ok and all(lr.passed for lr in layer_reports) and not stop and not plan_errors
    from audio_engine.core.dataset_v3.audit_plan import digest_payload
    drawn_ids = {r.sample_id for r in records} | {sid for layer in layer_reports for sid in layer.sample_ids}
    annotation_digest = digest_payload({s.id: {key: s.labels.get(key) for key in (
        "gold_kind", "gold_text", "speech_scope", "human_semantic", "human_noise", "human_crosstalk",
        "audio_event_tags", "verified_error_tags", "annotator_id", "reviewer_id", "adjudicator_id",
        "annotation_revision", "annotation_version")} for s in labeled if s.id in drawn_ids})
    return PseudoAuditReport(
        passed=passed,
        rule_version=rule_version,
        seed=audit_cfg.seed,
        overall=overall_report,
        protected_layers=layer_reports,
        critical_semantic_errors=overall_crit
        + sum(lr.critical_semantic_errors for lr in layer_reports),
        stop_publish=stop or not passed,
        reasons=reasons,
        excluded_calibration_count=excluded_cal,
        sampled_group_count=len(records),
        records=records,
        scope=dict(plan["scope"]) if plan else {},
        scope_digest=str(plan["digest"]) if plan else "",
        annotation_digest=annotation_digest,
    )


def mark_pseudo_audit_outcome(
    samples: Iterable[Sample],
    report: PseudoAuditReport,
) -> list[Sample]:
    """Stamp audit outcome on samples; never rewrite frozen release artifacts here."""
    out: list[Sample] = []
    audited_ids = {r.sample_id for r in report.records}
    for layer in report.protected_layers:
        audited_ids.update(layer.sample_ids)
    report_digest = report.to_dict()["report_digest"]
    for sample in samples:
        copied = sample.model_copy(deep=True)
        from audio_engine.core.dataset_v3.audit_plan import candidate_signature
        if report.scope.get(sample.id) != candidate_signature(sample):
            out.append(copied)
            continue
        if copied.id in audited_ids:
            copied.labels["pseudo_audit_sampled"] = True
        copied.labels["pseudo_audit_passed"] = bool(report.passed)
        copied.labels["pseudo_audit_stop_publish"] = bool(report.stop_publish)
        copied.labels["pseudo_audit_rule_version"] = report.rule_version
        copied.labels["pseudo_audit_report_digest"] = report_digest
        copied.labels["pseudo_audit_scope_digest"] = report.scope_digest
        if (
            str(copied.labels.get("type") or "") == "pseudo_high"
            and report.passed
            and not report.stop_publish
        ):
            # Gate open for train subset consumption (012-D); still not human gold.
            if str(copied.labels.get("annotation_state") or "") == "audit_pending":
                copied.labels["annotation_state"] = "auto_accept"
                copied.labels["decision"] = "auto_accept"
        elif str(copied.labels.get("type") or "") == "pseudo_high" and not report.passed:
            copied.labels["annotation_state"] = "audit_pending"
            copied.labels["pseudo_audit_block_reason"] = "; ".join(report.reasons[:5])
        out.append(copied)
    return out
