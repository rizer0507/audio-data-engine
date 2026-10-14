"""On-demand acoustic dependency union planning (040 §5.2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from audio_engine.core.tagging.conditions import (
    EvalContext,
    collect_acoustic_deps,
    evaluate_condition,
)
from audio_engine.core.tagging.config import RuleDef, TaggingConfig
from audio_engine.core.tagging.evidence.acoustic import AcousticEvidenceView
from audio_engine.core.tagging.evidence.asr_text import AsrTextEvidence
from audio_engine.core.tagging.types import Ternary

ACOUSTIC_CAPS = frozenset({"vad", "energy", "dnsmos"})


@dataclass
class SampleAcousticPlan:
    sample_id: str
    required: set[str] = field(default_factory=set)
    rule_deps: dict[str, list[str]] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "required": sorted(self.required),
            "rule_deps": {k: list(v) for k, v in self.rule_deps.items()},
            "reasons": list(self.reasons),
        }


@dataclass
class BatchAcousticPlan:
    by_sample: dict[str, SampleAcousticPlan] = field(default_factory=dict)
    totals: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "by_sample": {k: v.as_dict() for k, v in self.by_sample.items()},
            "totals": dict(self.totals),
        }


def _cheap_may_still_match(rule: RuleDef, ctx: EvalContext) -> bool:
    """
    Evaluate text-only projection: replace acoustic leaves with UNKNOWN and see
    if AND already false / OR already impossible.
    """
    deps = collect_acoustic_deps(rule.condition, fragments=ctx.config.fragments)
    if not deps:
        # Pure text — no acoustic needed regardless of outcome.
        return False
    # Build a shadow condition eval with acoustic forced unknown is already default
    # when evidence not present. If ternary is NOT_MATCHED, skip acoustics.
    result = evaluate_condition(rule.condition, ctx)
    if result.ternary == Ternary.NOT_MATCHED:
        return False
    # matched without acoustics shouldn't happen if deps exist, but if it does, skip.
    if result.ternary == Ternary.MATCHED and not result.missing_deps:
        return False
    return True


def plan_sample_acoustics(
    sample_id: str,
    asr: AsrTextEvidence,
    acoustic: AcousticEvidenceView,
    config: TaggingConfig,
    *,
    rules: list[RuleDef] | None = None,
) -> SampleAcousticPlan:
    """Plan acoustic deps for rules that are still possibly matched."""
    plan = SampleAcousticPlan(sample_id=sample_id)
    if config.evidence_mode != "compute_acoustic":
        return plan
    active = rules if rules is not None else config.enabled_rules()
    ctx = EvalContext(
        asr=asr,
        acoustic=acoustic,
        config=config,
        evidence_mode=config.evidence_mode,
    )
    for rule in active:
        if rule.text_binding:
            from audio_engine.core.tagging.conditions import _materialize_text_binding

            bind_t = _materialize_text_binding(rule.text_binding, ctx)
            if bind_t == Ternary.UNKNOWN:
                # Binding failed (e.g. family not stable) → rule cannot match via acoustics alone
                # if binding is required; still may need acoustics only if condition has them
                # and binding not required for those. Conservative: if binding unknown and
                # AND needs bound text, cheap eval will NOT_MATCH or UNKNOWN.
                pass
        deps = collect_acoustic_deps(rule.condition, fragments=config.fragments)
        if not deps:
            continue
        # Reset binding for isolation? Keep per-rule ctx copy.
        rule_ctx = EvalContext(
            asr=asr,
            acoustic=acoustic,
            config=config,
            evidence_mode=config.evidence_mode,
        )
        if rule.text_binding:
            from audio_engine.core.tagging.conditions import _materialize_text_binding

            _materialize_text_binding(rule.text_binding, rule_ctx)
        if not _cheap_may_still_match(rule, rule_ctx):
            continue
        # Only request deps not already successfully present.
        needed: list[str] = []
        for dep in sorted(deps):
            if dep == "vad" and acoustic.vad_compute_status().value == "success":
                continue
            if dep == "energy" and acoustic.energy_compute_status().value == "success":
                continue
            if dep == "dnsmos" and acoustic.dnsmos_compute_status().value == "success":
                continue
            needed.append(dep)
        if needed:
            plan.rule_deps[rule.rule_id] = needed
            plan.required.update(needed)
            plan.reasons.append(f"{rule.rule_id}:{','.join(needed)}")
    return plan


def plan_batch_acoustics(
    plans: list[SampleAcousticPlan],
) -> BatchAcousticPlan:
    batch = BatchAcousticPlan()
    totals = {"vad": 0, "energy": 0, "dnsmos": 0}
    for p in plans:
        batch.by_sample[p.sample_id] = p
        for dep in p.required:
            if dep in totals:
                totals[dep] += 1
    batch.totals = totals
    return batch
