"""Multi-tag rule engine: independent rules, shared evidence (040)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from audio_engine.core.sample import Sample
from audio_engine.core.tagging.adapters import AcousticAdapter, apply_view_to_sample
from audio_engine.core.tagging.conditions import (
    EvalContext,
    _materialize_text_binding,
    collect_acoustic_deps,
    evaluate_condition,
)
from audio_engine.core.tagging.config import TaggingConfig
from audio_engine.core.tagging.evidence.acoustic import acoustic_from_quality
from audio_engine.core.tagging.evidence.asr_text import AsrTextEvidence, build_asr_text_evidence
from audio_engine.core.tagging.planner import SampleAcousticPlan, plan_sample_acoustics
from audio_engine.core.tagging.state import (
    SOURCE_AUTO,
    SampleTagState,
    effective_tags,
    reconcile_auto_sources,
    sample_eval_status,
)
from audio_engine.core.tagging.types import RuleResult, Ternary


def build_voicemail_pattern(config: TaggingConfig) -> Any:
    """Compile voicemail patterns from tagging acoustic override or selection config."""
    import re
    from pathlib import Path

    import yaml

    vm_path = None
    if isinstance(config.acoustic, dict):
        vm_path = config.acoustic.get("voicemail_patterns_path")
    if not vm_path:
        vm_path = getattr(config.selection, "voicemail_patterns_path", None) or None
    if not vm_path:
        return None
    try:
        raw = yaml.safe_load(Path(str(vm_path)).read_text(encoding="utf-8")) or {}
        patterns = [str(p).strip() for p in (raw.get("patterns") or []) if str(p).strip()]
        if not patterns:
            return None
        flags_name = str(raw.get("flags") or "IGNORECASE").upper()
        flags = re.IGNORECASE if "IGNORECASE" in flags_name else 0
        return re.compile("|".join(f"(?:{p})" for p in patterns), flags)
    except Exception:
        return None


@dataclass
class SampleTaggingResult:
    sample_id: str
    rule_results: list[RuleResult]
    effective_tags: list[str]
    eval_status: str
    asr_fingerprint: str
    acoustic_plan: SampleAcousticPlan | None = None
    sources: list[dict[str, Any]] = field(default_factory=list)
    untagged_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "rule_results": [r.as_dict() for r in self.rule_results],
            "effective_tags": list(self.effective_tags),
            "eval_status": self.eval_status,
            "asr_fingerprint": self.asr_fingerprint,
            "acoustic_plan": self.acoustic_plan.as_dict() if self.acoustic_plan else None,
            "sources": list(self.sources),
            "untagged_reason": self.untagged_reason,
        }


@dataclass
class EngineStats:
    planned_calls: dict[str, int] = field(default_factory=lambda: {"vad": 0, "energy": 0, "dnsmos": 0})
    actual_calls: dict[str, int] = field(default_factory=lambda: {"vad": 0, "energy": 0, "dnsmos": 0})
    reused: dict[str, int] = field(default_factory=lambda: {"vad": 0, "energy": 0, "dnsmos": 0})
    failed: dict[str, int] = field(default_factory=lambda: {"vad": 0, "energy": 0, "dnsmos": 0})

    def as_dict(self) -> dict[str, Any]:
        return {
            "planned_calls": dict(self.planned_calls),
            "actual_calls": dict(self.actual_calls),
            "reused": dict(self.reused),
            "failed": dict(self.failed),
        }


class MultiTagEngine:
    def __init__(
        self,
        config: TaggingConfig,
        *,
        acoustic_adapter: AcousticAdapter | None = None,
        voicemail_pattern: Any | None = None,
    ):
        self.config = config
        self.acoustic_adapter = acoustic_adapter
        self.voicemail_pattern = voicemail_pattern if voicemail_pattern is not None else build_voicemail_pattern(config)
        self.stats = EngineStats()

    def build_asr(self, sample: Sample) -> AsrTextEvidence:
        return build_asr_text_evidence(
            sample,
            self.config.selection,
            voicemail_pattern=self.voicemail_pattern,
        )

    def evaluate_sample(
        self,
        sample: Sample,
        *,
        prior_state: SampleTagState | None = None,
        snapshot_id: str = "",
        rule_ids: set[str] | None = None,
        dry_run: bool = False,
        compute_acoustics: bool | None = None,
    ) -> SampleTaggingResult:
        asr = self.build_asr(sample)
        acoustic = acoustic_from_quality(sample.quality)
        rules = self.config.enabled_rules()
        if rule_ids is not None:
            rules = [r for r in rules if r.rule_id in rule_ids]

        do_compute = (
            self.config.evidence_mode == "compute_acoustic"
            if compute_acoustics is None
            else compute_acoustics
        )

        plan = plan_sample_acoustics(
            sample.id, asr, acoustic, self.config, rules=rules
        )
        for dep in plan.required:
            if dep in self.stats.planned_calls:
                self.stats.planned_calls[dep] += 1

        if do_compute and plan.required and self.acoustic_adapter is not None:
            # Dedup: single adapter call per sample with union deps.
            needed = set(plan.required)
            # Count reuse for already-success deps.
            for dep in list(needed):
                if dep == "vad" and acoustic.vad_compute_status().value == "success":
                    self.stats.reused["vad"] += 1
                    needed.discard("vad")
                elif dep == "energy" and acoustic.energy_compute_status().value == "success":
                    self.stats.reused["energy"] += 1
                    needed.discard("energy")
                elif dep == "dnsmos" and acoustic.dnsmos_compute_status().value == "success":
                    self.stats.reused["dnsmos"] += 1
                    needed.discard("dnsmos")
            if needed:
                view = self.acoustic_adapter.compute(
                    sample, deps=needed, params=dict(self.config.acoustic)
                )
                apply_view_to_sample(sample, view)
                acoustic = acoustic_from_quality(sample.quality)
                for dep in needed:
                    if dep in self.stats.actual_calls:
                        self.stats.actual_calls[dep] += 1
                    # Failure accounting
                    if dep == "vad" and acoustic.vad_compute_status().value == "failed":
                        self.stats.failed["vad"] += 1
                    if dep == "energy" and acoustic.energy_compute_status().value == "failed":
                        self.stats.failed["energy"] += 1
                    if dep == "dnsmos" and acoustic.dnsmos_compute_status().value == "failed":
                        self.stats.failed["dnsmos"] += 1

        rule_results: list[RuleResult] = []
        matched: list[tuple[str, str, str]] = []
        ternaries: dict[str, Ternary] = {}

        for rule in rules:
            ctx = EvalContext(
                asr=asr,
                acoustic=acoustic,
                config=self.config,
                evidence_mode=self.config.evidence_mode,
                planned_acoustic=set(plan.required),
            )
            bind_t = _materialize_text_binding(rule.text_binding, ctx)
            if bind_t == Ternary.UNKNOWN and rule.text_binding:
                # Binding failure → UNKNOWN for whole rule (cannot silently skip)
                from audio_engine.core.tagging.types import (
                    ConditionResult,
                    EvidenceResult,
                    EvidenceStatus,
                )

                cond = ConditionResult(
                    ternary=Ternary.UNKNOWN,
                    evidence=[
                        EvidenceResult(
                            capability="text_binding",
                            status=EvidenceStatus.MISSING,
                            error="text_binding_failed",
                        )
                    ],
                    missing_deps=["text_binding"],
                    explanation={"text_binding": "failed"},
                )
            else:
                cond = evaluate_condition(rule.condition, ctx)
            rr = RuleResult(
                rule_id=rule.rule_id,
                rule_version=rule.version,
                tag_id=rule.tag_id,
                ternary=cond.ternary,
                condition=cond,
                acoustic_plan=sorted(collect_acoustic_deps(rule.condition, fragments=self.config.fragments)),
            )
            rule_results.append(rr)
            ternaries[rule.rule_id] = cond.ternary
            if cond.ternary == Ternary.MATCHED:
                matched.append((rule.rule_id, rule.version, rule.tag_id))

        state = prior_state or SampleTagState(
            sample_id=sample.id, audio_sha256=str(sample.sha256 or "")
        )
        if not state.audio_sha256 and sample.sha256:
            state.audio_sha256 = sample.sha256

        executed_ids = {r.rule_id for r in rules}
        if not dry_run:
            state = reconcile_auto_sources(
                state,
                matched_rules=matched,
                snapshot_id=snapshot_id,
                scope_rule_ids=executed_ids if rule_ids is not None else None,
                executed_rule_ids=executed_ids,
            )

        enabled_tags = {
            tid for tid, t in self.config.tags.items() if t.enabled and not t.archived
        }
        eff = effective_tags(state, enabled_tag_ids=enabled_tags)
        has_stale = any(s.stale and s.source_type == SOURCE_AUTO for s in state.sources)
        status = sample_eval_status(
            effective=eff,
            rule_ternaries=ternaries,
            has_enabled_rules=bool(rules),
            has_stale=has_stale,
        )
        untagged_reason = None
        if status.value == "untagged":
            if any(
                s.source_type == "manual_block" and not s.stale for s in state.sources
            ) and matched:
                untagged_reason = "matched_but_manually_blocked"
            else:
                untagged_reason = "all_rules_not_matched"

        return SampleTaggingResult(
            sample_id=sample.id,
            rule_results=rule_results,
            effective_tags=eff,
            eval_status=status.value,
            asr_fingerprint=asr.input_fingerprint,
            acoustic_plan=plan,
            sources=[s.as_dict() for s in state.sources],
            untagged_reason=untagged_reason,
        )


def evaluate_rules_permutation_stable(
    sample: Sample,
    config: TaggingConfig,
    *,
    order_a: list[str],
    order_b: list[str],
) -> tuple[set[str], set[str]]:
    """Helper for acceptance: tag sets must match under rule order shuffle."""
    engine = MultiTagEngine(config)

    def run(order: list[str]) -> set[str]:
        # Temporarily reorder enabled rules.
        by_id = {r.rule_id: r for r in config.rules}
        config.rules = [by_id[i] for i in order if i in by_id] + [
            r for r in config.rules if r.rule_id not in order
        ]
        result = engine.evaluate_sample(sample, dry_run=True)
        return set(result.effective_tags) | {
            rr.tag_id for rr in result.rule_results if rr.ternary == Ternary.MATCHED
        }

    # Save original
    original = list(config.rules)
    try:
        a = run(order_a)
        b = run(order_b)
    finally:
        config.rules = original
    return a, b
