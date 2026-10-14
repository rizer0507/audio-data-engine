"""Condition evaluators: nested AND/OR/NOT + family quantifiers (040 §4)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from audio_engine.core.selection_v3.five_class_v2 import (
    _harmless_equivalent,
    _is_critical_short_response,
    _semantic_risk_v2,
)
from audio_engine.core.selection_v3.semantic_risk_strict import han_char_count
from audio_engine.core.selection_v3.types import (
    FAMILY_STATE_STABLE_TEXT,
    RISK_TAG_CRITICAL_POLARITY_RESPONSE,
    RISK_TAG_POLARITY_CONFLICT,
    SUBTYPE_SEMANTIC_REVERSAL,
    SUBTYPE_SHORT_POLARITY_AMBIGUITY,
)
from audio_engine.core.tagging.config import LexiconSpec, TaggingConfig
from audio_engine.core.tagging.evidence.acoustic import (
    AcousticEvidenceView,
    sig_is_low,
    strict_zero,
)
from audio_engine.core.tagging.evidence.asr_text import AsrTextEvidence
from audio_engine.core.tagging.ternary import at_least_n, ternary_and, ternary_not, ternary_or
from audio_engine.core.tagging.types import (
    ConditionResult,
    EvidenceResult,
    EvidenceStatus,
    Ternary,
)


@dataclass
class EvalContext:
    asr: AsrTextEvidence
    acoustic: AcousticEvidenceView
    config: TaggingConfig
    text_binding: dict[str, str] = field(default_factory=dict)
    # When evidence_mode=existing, missing acoustic → UNKNOWN (not false).
    evidence_mode: str = "existing"
    planned_acoustic: set[str] = field(default_factory=set)


def _bool_ternary(value: bool) -> Ternary:
    return Ternary.MATCHED if value else Ternary.NOT_MATCHED


def _status_to_unknown(status: EvidenceStatus) -> Ternary:
    return Ternary.UNKNOWN


def resolve_text_ref(ref: Any, ctx: EvalContext) -> tuple[str | None, EvidenceResult]:
    """Resolve a text source. None value with success=False means unavailable → UNKNOWN upstream."""
    if isinstance(ref, str):
        # Named binding from rule text_binding.
        if ref in ctx.text_binding:
            text = ctx.text_binding[ref]
            return text, EvidenceResult(
                capability="text_ref",
                value=text,
                status=EvidenceStatus.SUCCESS,
                details={"binding": ref},
            )
        return ref, EvidenceResult(
            capability="text_literal",
            value=ref,
            status=EvidenceStatus.SUCCESS,
        )
    if not isinstance(ref, dict):
        return None, EvidenceResult(
            capability="text_ref",
            status=EvidenceStatus.FAILED,
            error="invalid_text_ref",
        )
    # Shared binding key.
    if "binding" in ref:
        key = str(ref["binding"])
        if key not in ctx.text_binding:
            return None, EvidenceResult(
                capability="text_ref",
                status=EvidenceStatus.MISSING,
                error=f"unbound_text_binding:{key}",
            )
        text = ctx.text_binding[key]
        return text, EvidenceResult(
            capability="text_ref",
            value=text,
            status=EvidenceStatus.SUCCESS,
            details={"binding": key},
        )
    source = str(ref.get("source") or "").strip()
    if source == "family_representative":
        family = str(ref.get("family") or "").strip()
        state = ctx.asr.family_state_by_name.get(family)
        if state is None:
            return None, EvidenceResult(
                capability="family_representative",
                status=EvidenceStatus.MISSING,
                error=f"unknown_family:{family}",
            )
        if state != FAMILY_STATE_STABLE_TEXT:
            return None, EvidenceResult(
                capability="family_representative",
                value=None,
                status=EvidenceStatus.MISSING,
                error=f"family_not_stable_text:{family}:{state}",
                details={"family": family, "state": state},
            )
        text = ctx.asr.representative_text.get(family)
        if text is None:
            return None, EvidenceResult(
                capability="family_representative",
                status=EvidenceStatus.MISSING,
                error=f"missing_representative:{family}",
            )
        return str(text), EvidenceResult(
            capability="family_representative",
            value=text,
            status=EvidenceStatus.SUCCESS,
            details={"family": family, "state": state},
        )
    if source == "route_raw":
        run_id = str(ref.get("run_id") or "").strip()
        for route in ctx.asr.routes:
            if route.run_id == run_id:
                return str(route.raw_text or ""), EvidenceResult(
                    capability="route_raw",
                    value=route.raw_text,
                    status=EvidenceStatus.SUCCESS,
                    details={"run_id": run_id},
                )
        return None, EvidenceResult(
            capability="route_raw",
            status=EvidenceStatus.MISSING,
            error=f"unknown_run:{run_id}",
        )
    if source == "route_classify":
        run_id = str(ref.get("run_id") or "").strip()
        for route in ctx.asr.routes:
            if route.run_id == run_id:
                return str(route.classify_text or ""), EvidenceResult(
                    capability="route_classify",
                    value=route.classify_text,
                    status=EvidenceStatus.SUCCESS,
                    details={"run_id": run_id},
                )
        return None, EvidenceResult(
            capability="route_classify",
            status=EvidenceStatus.MISSING,
            error=f"unknown_run:{run_id}",
        )
    if source == "literal":
        text = str(ref.get("text") or "")
        return text, EvidenceResult(
            capability="text_literal",
            value=text,
            status=EvidenceStatus.SUCCESS,
        )
    return None, EvidenceResult(
        capability="text_ref",
        status=EvidenceStatus.FAILED,
        error=f"unknown_text_source:{source}",
    )


def _materialize_text_binding(rule_binding: dict[str, Any] | None, ctx: EvalContext) -> Ternary:
    """Populate ctx.text_binding from rule-level binding specs. Failure → UNKNOWN."""
    if not rule_binding:
        return Ternary.MATCHED
    for key, spec in rule_binding.items():
        text, ev = resolve_text_ref(spec, ctx)
        if ev.status != EvidenceStatus.SUCCESS or text is None:
            return Ternary.UNKNOWN
        ctx.text_binding[str(key)] = text
    return Ternary.MATCHED


def _lexicon_match(
    lexicon: LexiconSpec,
    text: str,
    *,
    match_any: bool = True,
) -> tuple[Ternary, EvidenceResult]:
    hits: list[dict[str, Any]] = []
    for entry in lexicon.entries:
        if entry.compiled is None:
            return Ternary.UNKNOWN, EvidenceResult(
                capability="lexicon_match",
                status=EvidenceStatus.FAILED,
                error=f"uncompiled_entry:{entry.entry_id}",
            )
        try:
            if entry.match_mode == "fullmatch":
                m = entry.compiled.fullmatch(text)
            else:
                m = entry.compiled.search(text)
        except Exception as exc:  # noqa: BLE001 — retain reason, never silent miss
            return Ternary.UNKNOWN, EvidenceResult(
                capability="lexicon_match",
                status=EvidenceStatus.FAILED,
                error=f"regex_error:{entry.entry_id}:{exc}",
            )
        if m is not None:
            hits.append(
                {
                    "entry_id": entry.entry_id,
                    "match": m.group(0),
                    "span": [m.start(), m.end()],
                    "lexicon_id": lexicon.lexicon_id,
                    "lexicon_version": lexicon.version,
                    "content_hash": lexicon.content_hash,
                }
            )
            if match_any:
                break
    matched = bool(hits) if match_any else (len(hits) == len(lexicon.entries) and bool(lexicon.entries))
    # all-match with empty lexicon is false
    if not match_any and not lexicon.entries:
        matched = False
    return _bool_ternary(matched), EvidenceResult(
        capability="lexicon_match",
        value={"hits": hits, "matched": matched},
        status=EvidenceStatus.SUCCESS,
        details={
            "lexicon_id": lexicon.lexicon_id,
            "lexicon_version": lexicon.version,
            "content_hash": lexicon.content_hash,
            "text": text,
            "match_any": match_any,
        },
    )


def _map_risk_tag(subtype: str | None) -> str | None:
    text = str(subtype or "").strip()
    if text in {
        RISK_TAG_POLARITY_CONFLICT,
        SUBTYPE_SEMANTIC_REVERSAL,
        "semantic_reversal",
    }:
        return RISK_TAG_POLARITY_CONFLICT
    if text in {
        RISK_TAG_CRITICAL_POLARITY_RESPONSE,
        SUBTYPE_SHORT_POLARITY_AMBIGUITY,
        "short_polarity_ambiguity",
    }:
        return RISK_TAG_CRITICAL_POLARITY_RESPONSE
    return text or None


def _eval_capability(node: dict[str, Any], ctx: EvalContext) -> ConditionResult:
    cap = str(node.get("capability") or "")
    evidence: list[EvidenceResult] = []

    if cap == "true":
        return ConditionResult(ternary=Ternary.MATCHED, explanation={"capability": cap})
    if cap == "false":
        return ConditionResult(ternary=Ternary.NOT_MATCHED, explanation={"capability": cap})

    if cap == "family_all_dual_success":
        val = bool(ctx.asr.all_families_dual_success)
        evidence.append(
            EvidenceResult(
                capability=cap,
                value=val,
                status=EvidenceStatus.SUCCESS,
                input_fingerprint=ctx.asr.input_fingerprint,
                config_digest=ctx.asr.config_digest,
                details={"by_family": dict(ctx.asr.dual_success_by_family)},
            )
        )
        return ConditionResult(ternary=_bool_ternary(val), evidence=evidence, explanation={"capability": cap})

    if cap == "family_dual_success":
        family = str(node.get("family") or "")
        if family not in ctx.asr.dual_success_by_family:
            return ConditionResult(
                ternary=Ternary.UNKNOWN,
                evidence=[
                    EvidenceResult(
                        capability=cap,
                        status=EvidenceStatus.MISSING,
                        error=f"unknown_family:{family}",
                    )
                ],
                missing_deps=[f"family:{family}"],
            )
        val = bool(ctx.asr.dual_success_by_family[family])
        evidence.append(
            EvidenceResult(capability=cap, value=val, status=EvidenceStatus.SUCCESS, details={"family": family})
        )
        return ConditionResult(ternary=_bool_ternary(val), evidence=evidence)

    if cap in {"family_state", "family_states_quantified"}:
        families = node.get("families")
        if families is None and node.get("family") is not None:
            families = [node.get("family")]
        if not isinstance(families, list) or not families:
            return ConditionResult(
                ternary=Ternary.UNKNOWN,
                evidence=[EvidenceResult(capability=cap, status=EvidenceStatus.FAILED, error="families_required")],
            )
        expected_state = str(node.get("state") or FAMILY_STATE_STABLE_TEXT)
        quant = str(node.get("quantifier") or "all").lower()
        per: list[Ternary] = []
        for fam in families:
            fam_s = str(fam)
            state = ctx.asr.family_state_by_name.get(fam_s)
            if state is None:
                per.append(Ternary.UNKNOWN)
                evidence.append(
                    EvidenceResult(
                        capability=cap,
                        status=EvidenceStatus.MISSING,
                        error=f"unknown_family:{fam_s}",
                    )
                )
            else:
                ok = state == expected_state
                per.append(_bool_ternary(ok))
                evidence.append(
                    EvidenceResult(
                        capability=cap,
                        value={"family": fam_s, "state": state, "expected": expected_state},
                        status=EvidenceStatus.SUCCESS,
                    )
                )
        if quant == "all":
            t = ternary_and(per)
        elif quant == "any":
            t = ternary_or(per)
        elif quant.startswith("at_least"):
            # at_least or at_least_n via node.n
            n = int(node.get("n") or 1)
            t = at_least_n(per, n)
        else:
            return ConditionResult(
                ternary=Ternary.UNKNOWN,
                evidence=[EvidenceResult(capability=cap, status=EvidenceStatus.FAILED, error=f"bad_quantifier:{quant}")],
            )
        return ConditionResult(ternary=t, evidence=evidence, explanation={"quantifier": quant})

    if cap in {"texts_equivalent", "texts_not_equivalent"}:
        left_text, left_ev = resolve_text_ref(node.get("left"), ctx)
        right_text, right_ev = resolve_text_ref(node.get("right"), ctx)
        evidence.extend([left_ev, right_ev])
        if left_text is None or right_text is None:
            return ConditionResult(ternary=Ternary.UNKNOWN, evidence=evidence, missing_deps=["text"])
        eq = _harmless_equivalent(left_text, right_text)
        val = eq if cap == "texts_equivalent" else (not eq)
        evidence.append(
            EvidenceResult(
                capability=cap,
                value={"equal": eq, "left": left_text, "right": right_text},
                status=EvidenceStatus.SUCCESS,
            )
        )
        return ConditionResult(ternary=_bool_ternary(val), evidence=evidence)

    if cap == "lexicon_match":
        lex_id = str(node.get("lexicon") or "")
        lexicon = ctx.config.lexicons.get(lex_id)
        if lexicon is None:
            return ConditionResult(
                ternary=Ternary.UNKNOWN,
                evidence=[EvidenceResult(capability=cap, status=EvidenceStatus.FAILED, error=f"unknown_lexicon:{lex_id}")],
            )
        text_ref = node.get("text")
        text, text_ev = resolve_text_ref(text_ref, ctx)
        evidence.append(text_ev)
        if text is None:
            return ConditionResult(ternary=Ternary.UNKNOWN, evidence=evidence, missing_deps=["text"])
        match_any = bool(node.get("match_any", True))
        t, ev = _lexicon_match(lexicon, text, match_any=match_any)
        evidence.append(ev)
        return ConditionResult(ternary=t, evidence=evidence)

    if cap == "single_han_char":
        text, text_ev = resolve_text_ref(node.get("text"), ctx)
        evidence.append(text_ev)
        if text is None:
            return ConditionResult(ternary=Ternary.UNKNOWN, evidence=evidence, missing_deps=["text"])
        # Entire cleaned string is exactly one CJK ideograph; no digits/other chars.
        ok = len(text) == 1 and han_char_count(text) == 1
        evidence.append(
            EvidenceResult(
                capability=cap,
                value={"text": text, "han_count": han_char_count(text), "len": len(text)},
                status=EvidenceStatus.SUCCESS,
            )
        )
        return ConditionResult(ternary=_bool_ternary(ok), evidence=evidence)

    if cap == "han_char_count":
        text, text_ev = resolve_text_ref(node.get("text"), ctx)
        evidence.append(text_ev)
        if text is None:
            return ConditionResult(ternary=Ternary.UNKNOWN, evidence=evidence, missing_deps=["text"])
        count = han_char_count(text)
        op = str(node.get("op") or "eq")
        n = int(node.get("n") or 0)
        ok = {
            "eq": count == n,
            "le": count <= n,
            "ge": count >= n,
            "lt": count < n,
            "gt": count > n,
        }.get(op)
        if ok is None:
            return ConditionResult(
                ternary=Ternary.UNKNOWN,
                evidence=[EvidenceResult(capability=cap, status=EvidenceStatus.FAILED, error=f"bad_op:{op}")],
            )
        evidence.append(EvidenceResult(capability=cap, value={"count": count, "op": op, "n": n}, status=EvidenceStatus.SUCCESS))
        return ConditionResult(ternary=_bool_ternary(ok), evidence=evidence)

    if cap == "text_length":
        text, text_ev = resolve_text_ref(node.get("text"), ctx)
        evidence.append(text_ev)
        if text is None:
            return ConditionResult(ternary=Ternary.UNKNOWN, evidence=evidence, missing_deps=["text"])
        length = len(text)
        op = str(node.get("op") or "eq")
        n = int(node.get("n") or 0)
        ok = {
            "eq": length == n,
            "le": length <= n,
            "ge": length >= n,
            "lt": length < n,
            "gt": length > n,
        }.get(op)
        if ok is None:
            return ConditionResult(
                ternary=Ternary.UNKNOWN,
                evidence=[EvidenceResult(capability=cap, status=EvidenceStatus.FAILED, error=f"bad_op:{op}")],
            )
        evidence.append(EvidenceResult(capability=cap, value={"length": length, "op": op, "n": n}, status=EvidenceStatus.SUCCESS))
        return ConditionResult(ternary=_bool_ternary(ok), evidence=evidence)

    if cap == "is_critical_short_response":
        text, text_ev = resolve_text_ref(node.get("text"), ctx)
        evidence.append(text_ev)
        if text is None:
            return ConditionResult(ternary=Ternary.UNKNOWN, evidence=evidence, missing_deps=["text"])
        ok = _is_critical_short_response(text, ctx.config.selection)
        evidence.append(EvidenceResult(capability=cap, value=ok, status=EvidenceStatus.SUCCESS))
        return ConditionResult(ternary=_bool_ternary(ok), evidence=evidence)

    if cap in {"polarity_conflict", "critical_short_ambiguity"}:
        risk = _semantic_risk_v2(ctx.asr.bundles, ctx.config.selection)
        mapped = _map_risk_tag(getattr(risk, "subtype", None) if risk else None)
        want = (
            RISK_TAG_POLARITY_CONFLICT
            if cap == "polarity_conflict"
            else RISK_TAG_CRITICAL_POLARITY_RESPONSE
        )
        ok = mapped == want
        evidence.append(
            EvidenceResult(
                capability=cap,
                value={"mapped": mapped, "hit": ok},
                status=EvidenceStatus.SUCCESS,
                details={"risk": getattr(risk, "evidence", None) if risk else None},
            )
        )
        return ConditionResult(ternary=_bool_ternary(ok), evidence=evidence)

    if cap == "voicemail_raw_hit":
        hit = ctx.asr.voicemail_hit is not None
        evidence.append(
            EvidenceResult(
                capability=cap,
                value=ctx.asr.voicemail_hit,
                status=EvidenceStatus.SUCCESS,
            )
        )
        return ConditionResult(ternary=_bool_ternary(hit), evidence=evidence)

    # --- Acoustic capabilities ---
    if cap in {"vad_speech_present", "vad_speech_absent"}:
        status = ctx.acoustic.vad_compute_status()
        if status in {EvidenceStatus.NOT_COMPUTED, EvidenceStatus.MISSING, EvidenceStatus.FAILED, EvidenceStatus.NOT_REQUIRED}:
            # NOT_REQUIRED means we didn't plan it; treat as unknown for rule that needs it.
            evidence.append(
                EvidenceResult(capability=cap, status=status, error=f"vad_{status.value}")
            )
            return ConditionResult(
                ternary=_status_to_unknown(status),
                evidence=evidence,
                missing_deps=["vad"],
            )
        state = str(ctx.acoustic.vad_speech_state or "").strip().lower()
        want = "present" if cap == "vad_speech_present" else "absent"
        ok = state == want
        evidence.append(
            EvidenceResult(capability=cap, value={"state": state}, status=EvidenceStatus.SUCCESS)
        )
        return ConditionResult(ternary=_bool_ternary(ok), evidence=evidence)

    if cap == "strict_all_channels_zero":
        status = ctx.acoustic.energy_compute_status()
        if status in {EvidenceStatus.NOT_COMPUTED, EvidenceStatus.MISSING, EvidenceStatus.FAILED, EvidenceStatus.NOT_REQUIRED}:
            evidence.append(
                EvidenceResult(capability=cap, status=status, error=f"energy_{status.value}")
            )
            return ConditionResult(ternary=Ternary.UNKNOWN, evidence=evidence, missing_deps=["energy"])
        flag, reason = strict_zero(ctx.acoustic)
        if flag is None:
            evidence.append(
                EvidenceResult(capability=cap, status=EvidenceStatus.FAILED, error=reason)
            )
            return ConditionResult(ternary=Ternary.UNKNOWN, evidence=evidence, missing_deps=["energy"])
        evidence.append(
            EvidenceResult(capability=cap, value=flag, status=EvidenceStatus.SUCCESS, details={"reason": reason})
        )
        return ConditionResult(ternary=_bool_ternary(bool(flag)), evidence=evidence)

    if cap == "dnsmos_sig_low":
        status = ctx.acoustic.dnsmos_compute_status()
        if status in {EvidenceStatus.NOT_COMPUTED, EvidenceStatus.MISSING, EvidenceStatus.FAILED, EvidenceStatus.NOT_REQUIRED}:
            evidence.append(
                EvidenceResult(capability=cap, status=status, error=f"dnsmos_{status.value}")
            )
            return ConditionResult(ternary=Ternary.UNKNOWN, evidence=evidence, missing_deps=["dnsmos"])
        is_low, reason = sig_is_low(ctx.acoustic, ctx.config.selection)
        if is_low is None:
            evidence.append(
                EvidenceResult(capability=cap, status=EvidenceStatus.FAILED, error=reason)
            )
            return ConditionResult(ternary=Ternary.UNKNOWN, evidence=evidence, missing_deps=["dnsmos"])
        evidence.append(
            EvidenceResult(
                capability=cap,
                value={"is_low": is_low, "sig": ctx.acoustic.dnsmos_sig},
                status=EvidenceStatus.SUCCESS,
                details={"reason": reason},
            )
        )
        return ConditionResult(ternary=_bool_ternary(bool(is_low)), evidence=evidence)

    return ConditionResult(
        ternary=Ternary.UNKNOWN,
        evidence=[EvidenceResult(capability=cap, status=EvidenceStatus.FAILED, error="unhandled_capability")],
    )


def evaluate_condition(node: dict[str, Any], ctx: EvalContext) -> ConditionResult:
    if not isinstance(node, dict):
        return ConditionResult(
            ternary=Ternary.UNKNOWN,
            evidence=[EvidenceResult(capability="condition", status=EvidenceStatus.FAILED, error="not_mapping")],
        )

    if "all" in node:
        children = [evaluate_condition(c, ctx) for c in node["all"]]
        # Short-circuit friendly collect: still evaluate all for audit unless AND already false?
        # Spec allows cheap short-circuit inside AND; for determinism in tests we evaluate all
        # but ternary_and still correct. Keep full evaluation for explanations.
        t = ternary_and(c.ternary for c in children)
        evidence: list[EvidenceResult] = []
        missing: list[str] = []
        for c in children:
            evidence.extend(c.evidence)
            missing.extend(c.missing_deps)
        return ConditionResult(
            ternary=t,
            evidence=evidence,
            explanation={"op": "all", "children": [c.ternary.value for c in children]},
            missing_deps=missing,
        )

    if "any" in node:
        children = [evaluate_condition(c, ctx) for c in node["any"]]
        t = ternary_or(c.ternary for c in children)
        evidence = []
        missing = []
        for c in children:
            evidence.extend(c.evidence)
            missing.extend(c.missing_deps)
        return ConditionResult(
            ternary=t,
            evidence=evidence,
            explanation={"op": "any", "children": [c.ternary.value for c in children]},
            missing_deps=missing,
        )

    if "not" in node:
        child = evaluate_condition(node["not"], ctx)
        return ConditionResult(
            ternary=ternary_not(child.ternary),
            evidence=list(child.evidence),
            explanation={"op": "not", "child": child.ternary.value},
            missing_deps=list(child.missing_deps),
        )

    if "at_least" in node:
        body = node["at_least"]
        n = int(body.get("n") or 1)
        children = [evaluate_condition(c, ctx) for c in (body.get("of") or [])]
        t = at_least_n((c.ternary for c in children), n)
        evidence = []
        missing = []
        for c in children:
            evidence.extend(c.evidence)
            missing.extend(c.missing_deps)
        return ConditionResult(
            ternary=t,
            evidence=evidence,
            explanation={"op": "at_least", "n": n, "children": [c.ternary.value for c in children]},
            missing_deps=missing,
        )

    if "fragment" in node:
        fid = str(node["fragment"])
        frag = ctx.config.fragments.get(fid)
        if frag is None:
            return ConditionResult(
                ternary=Ternary.UNKNOWN,
                evidence=[EvidenceResult(capability="fragment", status=EvidenceStatus.FAILED, error=f"unknown:{fid}")],
            )
        child = evaluate_condition(frag.condition, ctx)
        return ConditionResult(
            ternary=child.ternary,
            evidence=list(child.evidence),
            explanation={"op": "fragment", "fragment_id": fid, "version": frag.version},
            missing_deps=list(child.missing_deps),
        )

    if "capability" in node:
        return _eval_capability(node, ctx)

    return ConditionResult(
        ternary=Ternary.UNKNOWN,
        evidence=[EvidenceResult(capability="condition", status=EvidenceStatus.FAILED, error="unknown_node")],
    )


def collect_acoustic_deps(node: Any, *, fragments: dict[str, Any] | None = None) -> set[str]:
    """Walk condition tree and collect acoustic capability dependencies."""
    deps: set[str] = set()
    acoustic_caps = {
        "vad_speech_present": "vad",
        "vad_speech_absent": "vad",
        "strict_all_channels_zero": "energy",
        "dnsmos_sig_low": "dnsmos",
    }

    def walk(n: Any) -> None:
        if isinstance(n, dict):
            if "capability" in n:
                cap = str(n["capability"])
                if cap in acoustic_caps:
                    deps.add(acoustic_caps[cap])
            if "fragment" in n and fragments:
                fid = str(n["fragment"])
                frag = fragments.get(fid)
                if frag is not None:
                    cond = frag.condition if hasattr(frag, "condition") else frag.get("condition")
                    walk(cond)
            for key in ("all", "any"):
                if key in n and isinstance(n[key], list):
                    for c in n[key]:
                        walk(c)
            if "not" in n:
                walk(n["not"])
            if "at_least" in n and isinstance(n["at_least"], dict):
                walk(n["at_least"].get("of"))
            for k, v in n.items():
                if k in {"all", "any", "not", "at_least", "fragment", "capability"}:
                    continue
                walk(v)
        elif isinstance(n, list):
            for c in n:
                walk(c)

    walk(node)
    return deps
