"""022 semantic-tolerant classifier. Opt-in via rule_version; old path unchanged.

024 does not rewrite this path. Punctuation-only empty routing, verifier wiring
through family/clique/dissent, and integer 2/3 live in ``business_semantic.py``
under ``selection_business_semantic_v4``. Re-running this module must keep the
published 022 shadow comparable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

from audio_engine.core.sample import Sample
from audio_engine.core.selection_v3.acoustic_evidence import collect_acoustic_evidence
from audio_engine.core.selection_v3.config import SelectionV3Config
from audio_engine.core.selection_v3.gold_select import (
    FamilyRep,
    select_representative_text,
)
from audio_engine.core.selection_v3.input_contract import classify_run_status, is_physically_invalid
from audio_engine.core.selection_v3.legacy_map import map_legacy
from audio_engine.core.selection_v3.result import ClassificationResultV3
from audio_engine.core.selection_v3.semantic_verify import (
    SemanticEvidence,
    VerifyRequest,
    build_verifier,
    local_polarity_conflict,
)
from audio_engine.core.selection_v3.speech_rate import assess_speech_rate
from audio_engine.core.selection_v3.classify_text import apply_route_audit
from audio_engine.core.selection_v3.text import raw_transcript_text
from audio_engine.core.selection_v3.text_tolerance import (
    TextLayers,
    build_layers,
    contains_control_tag,
    is_short_pair,
    tolerant_distance,
)
from audio_engine.core.selection_v3.types import (
    CATEGORY_GOLD,
    CATEGORY_HARDCASE,
    CATEGORY_NOISE,
    CATEGORY_SEMANTIC_RISK,
    CATEGORY_VOICEMAIL,
    FAMILY_INCOMPLETE,
    FAMILY_STABLE_EMPTY,
    FAMILY_STABLE_TEXT,
    FAMILY_UNSTABLE_PRESENCE,
    FAMILY_UNSTABLE_SEMANTIC,
    FAMILY_UNSTABLE_TEXT,
    LABEL_SOURCE_NONE,
    RUN_STATUS_FAILED,
    RUN_STATUS_MISSING,
    RUN_STATUS_SUCCESS_EMPTY,
    RUN_STATUS_SUCCESS_TEXT,
    STATUS_CANDIDATE,
    STATUS_EXCLUDED,
    STATUS_HOLD,
    STATUS_MANUAL_REVIEW,
    STATUS_RETRY,
    TOLERANCE_VERSION,
)
from audio_engine.core.selection_v3.voicemail_strong import (
    VoicemailLibrary,
    agreed_scene,
    load_voicemail_library,
)

_LIBRARY_CACHE: dict[str, VoicemailLibrary] = {}
_AFFIRM = ("办理", "同意", "需要", "可以", "好的", "好")
_FILLER = ("嗯", "啊", "哦", "呃", "唔")


@dataclass
class RouteRec:
    run_id: str
    family: str
    status: str
    layers: TextLayers
    quarantined: bool = False


@dataclass
class FamilyRec:
    family: str
    routes: list[RouteRec]
    status: str
    representative: RouteRec | None = None
    chinese_vote: bool = False
    exclusion_reason: str | None = None
    language: str = ""


@dataclass
class Outcome:
    category: str | None
    status: str
    reason: str
    reason_codes: list[str] = field(default_factory=list)
    subtype: str | None = None
    auxiliary: list[str] = field(default_factory=list)
    candidate: str | None = None
    selection: Any = None
    semantic: list[SemanticEvidence] = field(default_factory=list)


def _library(config: SelectionV3Config) -> VoicemailLibrary:
    path = str(getattr(config, "voicemail_strong_path", "") or "")
    if path not in _LIBRARY_CACHE:
        _LIBRARY_CACHE[path] = load_voicemail_library(path or None)
    return _LIBRARY_CACHE[path]


def _homophones(config: SelectionV3Config) -> tuple[tuple[str, str], ...]:
    raw = getattr(config, "homophone_pairs", None) or (
        ("先声", "先生"),
        ("嘀声", "滴声"),
        ("嘀一声", "滴一声"),
    )
    pairs = []
    for item in raw:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            pairs.append((str(item[0]), str(item[1])))
    return tuple(pairs)


def _collect_routes(
    sample: Sample,
    config: SelectionV3Config,
    blocked: set[str],
    *,
    family_order: list[str],
) -> list[RouteRec]:
    routes: list[RouteRec] = []
    punct = config.punctuation_to_strip
    pairs = _homophones(config)
    for family in family_order:
        for key in config.model_families.get(family, []):
            status = classify_run_status(sample, key, config)
            entry = sample.transcripts.get(key)
            raw = raw_transcript_text(entry) if entry is not None else ""
            layers = build_layers(
                raw,
                punctuation_to_strip=punct,
                homophone_pairs=pairs,
                classify_text_policy=config.classify_text_policy,
                echo=config.echo_table_for(family),
                keep_digits=config.classify_text_keep_digits,
            )
            quarantined = str(key) in blocked
            if quarantined:
                status = RUN_STATUS_FAILED
            elif status in {RUN_STATUS_FAILED, RUN_STATUS_MISSING}:
                pass
            elif config.uses_chinese_only_text():
                status = (
                    RUN_STATUS_SUCCESS_TEXT
                    if layers.classify_text
                    else RUN_STATUS_SUCCESS_EMPTY
                )
            elif status == RUN_STATUS_SUCCESS_TEXT and not layers.transcript_text:
                status = RUN_STATUS_SUCCESS_EMPTY
            routes.append(
                RouteRec(
                    run_id=str(key),
                    family=family,
                    status=status,
                    layers=layers,
                    quarantined=quarantined,
                )
            )
    return routes


def _run_order(config: SelectionV3Config, family: str) -> list[str]:
    return [str(key) for key in config.model_families.get(family, [])]


def _analyze_family(family: str, routes: list[RouteRec], config: SelectionV3Config) -> FamilyRec:
    recall = float(getattr(config, "recall_max_distance", 0.10))
    short_max = int(getattr(config, "short_pair_max_chars", config.short_text_chars))
    members = [r for r in routes if r.family == family]
    if any(r.quarantined for r in members):
        return FamilyRec(
            family=family,
            routes=members,
            status=FAMILY_INCOMPLETE,
            exclusion_reason="speech_rate_quarantine",
        )
    if any(r.status in {RUN_STATUS_FAILED, RUN_STATUS_MISSING} for r in members):
        return FamilyRec(
            family=family,
            routes=members,
            status=FAMILY_INCOMPLETE,
            exclusion_reason="run_failed_or_missing",
        )
    text_routes = [r for r in members if r.status == RUN_STATUS_SUCCESS_TEXT and r.layers.transcript_text]
    empty = [r for r in members if r.status == RUN_STATUS_SUCCESS_EMPTY or not r.layers.transcript_text]
    if members and not text_routes:
        return FamilyRec(
            family=family,
            routes=members,
            status=FAMILY_STABLE_EMPTY,
            exclusion_reason="success_empty",
            language="empty",
        )
    if text_routes and empty:
        return FamilyRec(
            family=family,
            routes=members,
            status=FAMILY_UNSTABLE_PRESENCE,
            exclusion_reason="presence_unstable",
        )
    langs = {r.layers.language for r in text_routes}
    if langs == {"en"} or (langs and "zh" not in langs and "mixed" not in langs and langs != {"zh"}):
        if langs <= {"en"}:
            return FamilyRec(
                family=family,
                routes=members,
                status=FAMILY_STABLE_TEXT,
                exclusion_reason="language_abstain",
                language="en",
            )
    if "mixed" in langs or (len(langs) > 1 and "zh" in langs):
        return FamilyRec(
            family=family,
            routes=members,
            status=FAMILY_UNSTABLE_TEXT,
            exclusion_reason="language_incomparable",
            language="mixed",
        )
    if not all(r.layers.language == "zh" for r in text_routes):
        return FamilyRec(
            family=family,
            routes=members,
            status=FAMILY_UNSTABLE_TEXT,
            exclusion_reason="language_incomparable",
            language="mixed" if "mixed" in langs else "unknown",
        )
    if len(text_routes) < 2:
        return FamilyRec(
            family=family,
            routes=members,
            status=FAMILY_INCOMPLETE,
            exclusion_reason="single_chinese_route",
        )
    left, right = text_routes[0], text_routes[1]
    relation = _pair_relation(left.layers, right.layers, recall=recall, short_max=short_max)
    if relation == "conflict":
        return FamilyRec(
            family=family,
            routes=members,
            status=FAMILY_UNSTABLE_SEMANTIC,
            exclusion_reason="intra_family_conflict",
            language="zh",
        )
    if relation != "equivalent":
        return FamilyRec(
            family=family,
            routes=members,
            status=FAMILY_UNSTABLE_TEXT,
            exclusion_reason=f"intra_family_{relation}",
            language="zh",
        )
    order = _run_order(config, family)
    chosen = sorted(text_routes, key=lambda r: order.index(r.run_id) if r.run_id in order else 99)[0]
    return FamilyRec(
        family=family,
        routes=members,
        status=FAMILY_STABLE_TEXT,
        representative=chosen,
        chinese_vote=True,
        language="zh",
    )


def _pair_relation(
    left: TextLayers,
    right: TextLayers,
    *,
    recall: float,
    short_max: int,
    verifier: Any | None = None,
    rule_version: str = "",
) -> str:
    if left.language == "zh" and right.language == "zh":
        if left.comparison_text == right.comparison_text or left.tolerant_key == right.tolerant_key:
            # Identical comparison is not a model conflict even if both contain 不.
            if left.comparison_text == right.comparison_text:
                return "equivalent"
            if left.tolerant_key == right.tolerant_key:
                return "equivalent"
        hit = local_polarity_conflict(left.comparison_text, right.comparison_text)
        if hit is not None:
            return "conflict"
        dist = tolerant_distance(left.tolerant_key, right.tolerant_key)
        short = is_short_pair(left.comparison_text, right.comparison_text, short_max_chars=short_max)
        if short:
            return "unclear" if (dist is None or dist < 0.25) else "divergent"
        if dist is None:
            return "unclear"
        if dist >= 0.25:
            return "divergent"
        if verifier is not None:
            result = verifier.verify(
                VerifyRequest(
                    left_raw=left.raw_text,
                    right_raw=right.raw_text,
                    left_transcript=left.transcript_text,
                    right_transcript=right.transcript_text,
                    left_language=left.language,
                    right_language=right.language,
                    rule_version=rule_version,
                )
            )
            if result.verdict == "equivalent" and dist <= recall:
                return "equivalent"
            if result.verdict == "conflict":
                return "conflict"
        return "unclear"
    if {left.language, right.language} == {"zh", "en"} and verifier is not None:
        result = verifier.verify(
            VerifyRequest(
                left_raw=left.raw_text,
                right_raw=right.raw_text,
                left_transcript=left.transcript_text,
                right_transcript=right.transcript_text,
                left_language=left.language,
                right_language=right.language,
                rule_version=rule_version,
            )
        )
        if result.verdict == "conflict":
            return "conflict"
        if result.verdict == "equivalent":
            return "cross_equivalent"
        return "cross_unknown"
    if left.language != right.language:
        return "incomparable"
    return "unclear"


def _largest_cliques(reps: list[FamilyRec], config: SelectionV3Config) -> list[list[FamilyRec]]:
    recall = float(getattr(config, "recall_max_distance", 0.10))
    short_max = int(getattr(config, "short_pair_max_chars", config.short_text_chars))
    n = len(reps)
    if n < 2:
        return []
    cliques: list[list[FamilyRec]] = []
    for size in range(n, 1, -1):
        for combo in combinations(range(n), size):
            group = [reps[i] for i in combo]
            if _clique_ok(group, recall=recall, short_max=short_max):
                cliques.append(group)
        if cliques:
            break
    return cliques


def _clique_ok(group: list[FamilyRec], *, recall: float, short_max: int) -> bool:
    for i, left in enumerate(group):
        for right in group[i + 1 :]:
            if left.representative is None or right.representative is None:
                return False
            relation = _pair_relation(
                left.representative.layers,
                right.representative.layers,
                recall=recall,
                short_max=short_max,
            )
            if relation != "equivalent":
                return False
    return True


def _pick_clique(cliques: list[list[FamilyRec]], family_order: list[str]) -> list[FamilyRec]:
    if not cliques:
        return []

    def key(group: list[FamilyRec]) -> tuple:
        names = tuple(
            sorted(
                (member.family for member in group),
                key=lambda name: family_order.index(name) if name in family_order else 99,
            )
        )
        return (names,)

    # Same size already. Prefer the lexicographically earliest configured family tuple.
    return sorted(cliques, key=key)[0]


def _is_affirmation(text: str) -> bool:
    value = text.strip()
    if not value:
        return False
    if local_polarity_conflict(value, "不" + value):
        return False
    if any(token in value for token in ("不", "没", "别")):
        return False
    return any(token in value for token in _AFFIRM) or value in _FILLER


def _is_filler(text: str) -> bool:
    return text.strip() in _FILLER or text.strip() in {"嗯嗯", "啊啊"}


def _to_rep(family: FamilyRec) -> FamilyRep | None:
    route = family.representative
    if route is None:
        return None
    return FamilyRep(
        family=family.family,
        run_id=route.run_id,
        transcript_text=route_body(route),
        raw_text=route.layers.raw_text,
        tolerant_key=route.layers.tolerant_key,
        comparison_text=route.layers.comparison_text,
    )


def route_body(route: RouteRec) -> str:
    body = route.layers.transcript_text
    if contains_control_tag(body):
        return ""
    return body


def classify_semantic_tolerant(
    sample: Sample,
    config: SelectionV3Config,
    *,
    voicemail_pattern: Any = None,
    exclude_families: set[str] | None = None,
) -> ClassificationResultV3:
    del voicemail_pattern  # weak legacy regex must not assign automatic V
    from audio_engine.core.selection_v3.noise_trigger import (
        ensure_trigger_record,
        uses_asr_anomaly_noise,
    )

    quality = sample.quality if isinstance(sample.quality, dict) else {}
    diagnosis = ensure_trigger_record(sample, config) if uses_asr_anomaly_noise(config) else None
    if diagnosis and diagnosis.get("status") == "not_required":
        quality = sample.quality if isinstance(sample.quality, dict) else {}
    excluded = set(exclude_families or ())
    # Evaluation isolation drops the target family before votes, distances, and ties.
    # It must not look like a failed run, or the remaining consensus would become a retry.
    family_order = [name for name in config.ordered_families() if name not in excluded]
    verifier = build_verifier(
        str(getattr(config, "semantic_verifier_mode", "local") or "local"),
        endpoint=str(getattr(config, "semantic_verifier_endpoint", "") or ""),
    )

    if is_physically_invalid(sample):
        return _finish(
            sample,
            config,
            Outcome(None, STATUS_EXCLUDED, "broken_or_invalid_audio", ["invalid_audio"]),
            families={},
            routes=[],
            acoustic=collect_acoustic_evidence(quality, sample.labels),
            languages={},
        )

    duration = float(sample.duration) if sample.duration is not None else None
    probe_routes = _collect_routes(sample, config, set(), family_order=family_order)
    rate = assess_speech_rate(
        _rate_views(probe_routes),
        duration_sec=duration,
        max_chars_per_sec=config.max_chars_per_sec,
        min_text_chars=config.speech_rate_min_text_chars,
    )
    blocked = set(rate.implausible_routes) if rate.triggered else set()
    routes = (
        _collect_routes(sample, config, blocked, family_order=family_order) if blocked else probe_routes
    )
    families = {
        family: _analyze_family(family, routes, config) for family in family_order
    }
    acoustic = collect_acoustic_evidence(quality, sample.labels)
    languages = {route.run_id: route.layers.language for route in routes}
    semantic = _semantic_evidence(families, verifier, config.rule_version)
    gaps = _unresolved_gaps(families, sample)
    library = _library(config)
    voicemail = _voicemail(families, library)
    votes = [families[name] for name in family_order if families[name].chinese_vote]
    m = len(votes)
    cliques = _largest_cliques(votes, config)
    clique = _pick_clique(cliques, family_order)
    if config.uses_chinese_only_text():
        cross = "none"
    else:
        cross = _cross_language(families, verifier, config.rule_version)
    outcome = _decide(
        families=families,
        clique=clique,
        m=m,
        config=config,
        semantic=semantic,
        acoustic=acoustic,
        voicemail=voicemail,
        gaps=gaps,
        cross=cross,
        family_order=family_order,
    )
    return _finish(
        sample,
        config,
        outcome,
        families=families,
        routes=routes,
        acoustic=acoustic,
        languages=languages,
        clique=clique,
        m=m,
        rate_routes=sorted(blocked),
        rate_max=rate.max_chars_per_sec_observed if blocked else None,
        library_version=library.version,
        considered_family_count=len(family_order),
    )


def _rate_views(routes: list[RouteRec]):
    from audio_engine.core.selection_v3.family_evidence import RouteView

    return [
        RouteView(
            run_id=route.run_id,
            family=route.family,
            status=route.status,
            raw_text=route.layers.raw_text,
            comparison_text=route.layers.comparison_text,
        )
        for route in routes
    ]


def _unresolved_gaps(families: dict[str, FamilyRec], sample: Sample) -> list[str]:
    tried = int(sample.labels.get("route_retry_count") or 0)
    reasons = []
    for family, state in families.items():
        if state.exclusion_reason in {"run_failed_or_missing", "speech_rate_quarantine", "single_chinese_route"}:
            reasons.append(f"{family}:{state.exclusion_reason}:retry_count={tried}")
    return reasons


def _semantic_evidence(families: dict[str, FamilyRec], verifier: Any, rule_version: str) -> list[SemanticEvidence]:
    evidence: list[SemanticEvidence] = []
    text_families = [
        state
        for state in families.values()
        if state.representative is not None and state.language == "zh"
    ]
    # Intra-family conflicts already excluded from votes; still surface them.
    for state in families.values():
        if state.exclusion_reason != "intra_family_conflict":
            continue
        bodies = [r.layers.comparison_text for r in state.routes if r.layers.comparison_text]
        if len(bodies) >= 2:
            hit = local_polarity_conflict(bodies[0], bodies[1])
            if hit is not None:
                hit.families = [state.family]
                evidence.append(hit)
    for i, left in enumerate(text_families):
        for right in text_families[i + 1 :]:
            assert left.representative and right.representative
            hit = local_polarity_conflict(
                left.representative.layers.comparison_text,
                right.representative.layers.comparison_text,
            )
            if hit is not None:
                hit.families = [left.family, right.family]
                evidence.append(hit)
                continue
            relation = _pair_relation(
                left.representative.layers,
                right.representative.layers,
                recall=0.10,
                short_max=6,
                verifier=verifier,
                rule_version=rule_version,
            )
            if relation == "conflict":
                evidence.append(
                    SemanticEvidence(
                        kind="verifier_conflict",
                        verdict="conflict",
                        families=[left.family, right.family],
                        citations=[
                            left.representative.layers.transcript_text,
                            right.representative.layers.transcript_text,
                        ],
                        version=getattr(verifier, "version", ""),
                        conflict_type="verifier_conflict",
                    )
                )
    return evidence


def _cross_language(families: dict[str, FamilyRec], verifier: Any, rule_version: str) -> str:
    zh = [
        state.representative
        for state in families.values()
        if state.chinese_vote and state.representative is not None
    ]
    foreign = []
    for state in families.values():
        for route in state.routes:
            if route.status == RUN_STATUS_SUCCESS_TEXT and route.layers.language in {
                "en",
                "mixed",
                "unknown",
            }:
                foreign.append(route)
    if not foreign:
        return "none"
    if not zh:
        return "unknown"
    verdicts = []
    for route in foreign:
        for rep in zh:
            result = verifier.verify(
                VerifyRequest(
                    left_raw=rep.layers.raw_text,
                    right_raw=route.layers.raw_text,
                    left_transcript=rep.layers.transcript_text,
                    right_transcript=route.layers.transcript_text,
                    left_language="zh",
                    right_language=route.layers.language,
                    rule_version=rule_version,
                )
            )
            verdicts.append(result.verdict)
    if "conflict" in verdicts:
        return "conflict"
    if verdicts and all(v == "equivalent" for v in verdicts):
        return "equivalent"
    return "unknown"


def _voicemail(families: dict[str, FamilyRec], library: VoicemailLibrary):
    texts = {}
    for name, state in families.items():
        if state.representative is None:
            continue
        texts[name] = state.representative.layers.transcript_text
    return agreed_scene(texts, library)


def _decide(
    *,
    families: dict[str, FamilyRec],
    clique: list[FamilyRec],
    m: int,
    config: SelectionV3Config,
    semantic: list[SemanticEvidence],
    acoustic,
    voicemail,
    gaps: list[str],
    cross: str,
    family_order: list[str],
) -> Outcome:
    auxiliary: list[str] = []
    if acoustic.state == "environment_confirmed":
        auxiliary.append("noise")
    if acoustic.state == "crosstalk_confirmed":
        auxiliary.append("crosstalk")
    if acoustic.background_only:
        auxiliary.append("background")
    if voicemail is not None:
        auxiliary.append("voicemail")

    affirm = _affirmation_against_empty(families)
    if semantic or (affirm and acoustic.vad_calibrated and acoustic.vad_speech_present is False):
        codes = ["semantic_risk"] + [item.kind for item in semantic]
        if affirm and acoustic.vad_calibrated and acoustic.vad_speech_present is False:
            codes.append("hallucination_affirmation")
        if gaps:
            codes.append("technical_gap_retained")
        return Outcome(
            CATEGORY_SEMANTIC_RISK,
            STATUS_MANUAL_REVIEW,
            "semantic_conflict_requires_review",
            codes,
            auxiliary=auxiliary,
            semantic=semantic,
        )
    if cross == "conflict":
        return Outcome(
            CATEGORY_SEMANTIC_RISK,
            STATUS_MANUAL_REVIEW,
            "cross_language_conflict",
            ["cross_language_conflict"],
            auxiliary=auxiliary,
        )
    if affirm and acoustic.state in {"environment_confirmed", "crosstalk_confirmed"}:
        return Outcome(
            CATEGORY_SEMANTIC_RISK,
            STATUS_MANUAL_REVIEW,
            "noise_with_affirmation_hallucination_risk",
            ["hallucination_affirmation", "noise_overlap"],
            auxiliary=["noise", *auxiliary],
        )
    if affirm and not (acoustic.vad_calibrated and acoustic.vad_speech_present is False):
        return Outcome(
            None,
            STATUS_HOLD,
            "affirmation_without_audio_evidence",
            ["hallucination_unconfirmed", "acoustic_evidence_missing"],
            auxiliary=auxiliary,
        )

    if _confirmed_noise(families, acoustic) and not semantic:
        subtype = "crosstalk" if acoustic.state == "crosstalk_confirmed" else "environment"
        return Outcome(
            CATEGORY_NOISE,
            STATUS_CANDIDATE,
            "acoustic_noise_confirmed",
            ["noise_confirmed"],
            subtype=subtype,
        )

    if gaps:
        selection = _select(clique, config, family_order) if clique else None
        return Outcome(
            None,
            STATUS_RETRY if _retry_open(gaps) else STATUS_HOLD,
            "unresolved_route_gap",
            ["route_gap", *gaps],
            candidate=selection.transcript_text if selection is not None else None,
            selection=selection,
        )

    if cross == "unknown":
        return Outcome(
            None,
            STATUS_HOLD,
            "cross_language_unverified",
            ["language_unverified"],
            auxiliary=auxiliary,
        )

    min_k = int(getattr(config, "min_support_families", 2))
    min_ratio = float(getattr(config, "min_support_ratio", 2 / 3))
    k = len(clique)
    ratio = (k / m) if m else None
    eligible = k >= min_k and m >= min_k and ratio is not None and ratio + 1e-9 >= min_ratio
    remaining = [state for state in families.values() if state.chinese_vote and state not in clique]
    if remaining:
        # A leftover Chinese vote was not pairwise compatible with the clique.
        relations = []
        for state in remaining:
            for member in clique or []:
                if member.representative and state.representative:
                    relations.append(
                        _pair_relation(
                            member.representative.layers,
                            state.representative.layers,
                            recall=float(getattr(config, "recall_max_distance", 0.10)),
                            short_max=int(getattr(config, "short_pair_max_chars", config.short_text_chars)),
                        )
                    )
        if any(rel == "conflict" for rel in relations):
            return Outcome(
                CATEGORY_SEMANTIC_RISK,
                STATUS_MANUAL_REVIEW,
                "remaining_family_semantic_conflict",
                ["semantic_risk"],
            )
        if any(rel == "divergent" for rel in relations) or (not clique and _has_divergence(families, config)):
            return Outcome(
                CATEGORY_HARDCASE,
                STATUS_MANUAL_REVIEW,
                "substantive_family_divergence",
                ["hardcase"],
            )
        return Outcome(
            None,
            STATUS_HOLD,
            "residual_difference_unverified",
            ["semantic_unclear"],
        )

    if not eligible:
        if _has_divergence(families, config):
            return Outcome(
                CATEGORY_HARDCASE,
                STATUS_MANUAL_REVIEW,
                "substantive_family_divergence",
                ["hardcase"],
            )
        if _all_success_empty(families):
            return Outcome(
                None,
                STATUS_HOLD,
                "all_empty_acoustic_evidence_missing",
                ["acoustic_evidence_missing"],
            )
        return Outcome(
            None,
            STATUS_HOLD,
            "insufficient_chinese_consensus",
            ["chinese_support_insufficient"],
        )

    if _presence_mixed(families):
        return Outcome(
            None,
            STATUS_HOLD,
            "presence_disagreement_unconfirmed",
            ["presence_unconfirmed"],
        )

    selection = _select(clique, config, family_order)
    if selection is None or contains_control_tag(selection.transcript_text):
        return Outcome(None, STATUS_HOLD, "selection_failed", ["selection_failed"])

    if voicemail is not None:
        scene, subtype, hit_families, evidence = voicemail
        del evidence
        if len(hit_families) >= 2:
            return Outcome(
                CATEGORY_VOICEMAIL,
                STATUS_CANDIDATE,
                "strong_voicemail_template_consensus",
                ["voicemail_strong", f"scene:{scene}"],
                subtype=subtype,
                auxiliary=["voicemail"],
                candidate=selection.transcript_text,
                selection=selection,
            )

    codes = ["gold_candidate"]
    if any(item.layers.script_converted for item in families.values() for item in item.routes):
        codes.append("script_conversion")
    if any("你" in (item.representative.layers.tolerant_key if item.representative else "") for item in clique):
        codes.append("tolerance")
    if acoustic.background_only:
        codes.append("background_present")
    return Outcome(
        CATEGORY_GOLD,
        STATUS_CANDIDATE,
        "chinese_tolerant_consensus",
        codes,
        auxiliary=auxiliary,
        candidate=selection.transcript_text,
        selection=selection,
    )


def _affirmation_against_empty(families: dict[str, FamilyRec]) -> bool:
    has_empty = any(state.status == FAMILY_STABLE_EMPTY for state in families.values())
    if not has_empty:
        return False
    for state in families.values():
        if state.representative is None:
            continue
        text = state.representative.layers.comparison_text
        if any(token in text for token in ("办理", "同意", "需要")) and "不" not in text:
            return True
    return False


def _confirmed_noise(families: dict[str, FamilyRec], acoustic) -> bool:
    empty_families = [
        state for state in families.values() if state.status == FAMILY_STABLE_EMPTY
    ]
    if acoustic.state == "crosstalk_confirmed":
        return True
    if acoustic.state == "environment_confirmed" and len(empty_families) >= 2:
        return True
    return False


def _retry_open(gaps: list[str]) -> bool:
    return any("retry_count=0" in gap for gap in gaps)


def _has_divergence(families: dict[str, FamilyRec], config: SelectionV3Config) -> bool:
    votes = [state for state in families.values() if state.chinese_vote and state.representative]
    if len(votes) < 2:
        return False
    recall = float(getattr(config, "divergence_min_distance", 0.25))
    short_max = int(getattr(config, "short_pair_max_chars", config.short_text_chars))
    divergent_pairs = 0
    for i, left in enumerate(votes):
        for right in votes[i + 1 :]:
            relation = _pair_relation(
                left.representative.layers,
                right.representative.layers,
                recall=float(getattr(config, "recall_max_distance", 0.10)),
                short_max=short_max,
            )
            if relation == "divergent":
                divergent_pairs += 1
            dist = tolerant_distance(
                left.representative.layers.tolerant_key,
                right.representative.layers.tolerant_key,
            )
            if dist is not None and dist >= recall and relation != "equivalent":
                divergent_pairs += 1
    return divergent_pairs >= 1 and len(votes) >= 2


def _all_success_empty(families: dict[str, FamilyRec]) -> bool:
    return bool(families) and all(state.status == FAMILY_STABLE_EMPTY for state in families.values())


def _presence_mixed(families: dict[str, FamilyRec]) -> bool:
    has_text = any(state.chinese_vote for state in families.values())
    has_empty = any(state.status == FAMILY_STABLE_EMPTY for state in families.values())
    return has_text and has_empty


def _select(clique: list[FamilyRec], config: SelectionV3Config, family_order: list[str]):
    reps = [rep for item in clique if (rep := _to_rep(item)) is not None]
    run_order = []
    for family in family_order:
        run_order.extend(_run_order(config, family))
    return select_representative_text(
        reps,
        family_order=family_order,
        run_order=run_order,
        tolerance_version=str(getattr(config, "tolerance_version", TOLERANCE_VERSION)),
    )


def eval_reference_excluding(
    sample: Sample,
    config: SelectionV3Config,
    family: str,
) -> dict[str, Any]:
    """Recompute consensus and selection after dropping the target family.

    Support votes, family representatives, distances, exact-text support, and
    tie-break all run again without that family. Filtering an already chosen
    clique is not enough: a leftover disagreement may have blocked gold.
    """
    if family not in set(config.ordered_families()):
        return {"eligible": False, "reason": "unknown_family", "candidate_text": None}
    result = classify_semantic_tolerant(sample, config, exclude_families={family})
    enough = (result.chinese_available_family_count or 0) >= 2 and (result.support_family_count or 0) >= 2
    if (
        result.category not in {CATEGORY_GOLD, CATEGORY_VOICEMAIL}
        or result.status != STATUS_CANDIDATE
        or not enough
        or not result.candidate_text
        or contains_control_tag(result.candidate_text)
    ):
        return {
            "eligible": False,
            "reason": "fewer_than_two_chinese_families"
            if not enough
            else (result.reason or "not_candidate_after_exclusion"),
            "candidate_text": None,
            "category": result.category,
            "status": result.status,
        }
    return {
        "eligible": True,
        "candidate_text": result.candidate_text,
        "selected_family": result.selected_family,
        "selected_run_id": result.selected_run_id,
        "reason": "excluded_target_family",
    }


def _quality_state_not_gating(acoustic, diagnosis: dict | None = None) -> str:
    """Record evidence without reintroducing the old uncalibrated gold gate."""
    from audio_engine.core.selection_v3.noise_trigger import quality_state_for_diagnosis
    from audio_engine.core.selection_v3.types import DNSMOS_STATUS_NOT_REQUIRED

    if diagnosis and (
        diagnosis.get("status") == DNSMOS_STATUS_NOT_REQUIRED or diagnosis.get("policy")
    ):
        if diagnosis.get("status") == DNSMOS_STATUS_NOT_REQUIRED:
            return "not_required"
        return quality_state_for_diagnosis(diagnosis)
    if acoustic.state in {"environment_confirmed", "crosstalk_confirmed"}:
        return "evidence_confirmed"
    if acoustic.scores.get("calibrated") is True:
        return "scored_not_gating"
    return "not_gating"


def _finish(
    sample: Sample,
    config: SelectionV3Config,
    outcome: Outcome,
    *,
    families: dict[str, FamilyRec],
    routes: list[RouteRec],
    acoustic,
    languages: dict[str, str],
    clique: list[FamilyRec] | None = None,
    m: int = 0,
    rate_routes: list[str] | None = None,
    rate_max: float | None = None,
    library_version: str = "",
    considered_family_count: int | None = None,
) -> ClassificationResultV3:
    diagnosis = sample.labels.get("noise_diagnosis") if isinstance(sample.labels.get("noise_diagnosis"), dict) else None
    del sample
    legacy = map_legacy(category=outcome.category, status=outcome.status, reason=outcome.reason)
    k = len(clique or [])
    n = considered_family_count if considered_family_count is not None else len(config.model_families)
    selection = outcome.selection
    candidate = outcome.candidate if outcome.candidate is not None else (
        selection.transcript_text if selection is not None else None
    )
    if candidate and contains_control_tag(candidate):
        candidate = ""
        outcome.status = STATUS_HOLD
        legacy = map_legacy(category=None, status=STATUS_HOLD, reason="control_tag_in_candidate")
    abstain = {
        name: state.exclusion_reason
        for name, state in families.items()
        if state.exclusion_reason
    }
    char_comparable = None
    if languages:
        zh = [lang == "zh" for lang in languages.values() if lang not in {"empty"}]
        char_comparable = bool(zh) and all(lang == "zh" for lang in languages.values() if lang not in {"empty"})
    trace: dict[str, Any] = {}
    if selection is not None:
        trace = {
            "selected_family": selection.family,
            "selected_run_id": selection.run_id,
            "distance": selection.distance,
            "transcript_support_count": selection.transcript_support_count,
            "tie_break_reason": selection.tie_break_reason,
            "representatives": selection.representatives,
            "family_order": selection.family_order,
            "run_order": selection.run_order,
            "tolerance_version": selection.tolerance_version,
            "selected_raw_text": selection.raw_text,
        }
    return apply_route_audit(
        ClassificationResultV3(
        type=legacy.type,
        decision=legacy.decision,
        reason=outcome.reason,
        review_priority=legacy.review_priority,
        review_queue=legacy.review_queue,
        candidate_text=candidate,
        label_source=legacy.label_source if candidate else LABEL_SOURCE_NONE,
        label_tier=legacy.label_tier,
        is_human_verified=False,
        risk_tags=list(dict.fromkeys(outcome.reason_codes + outcome.auxiliary)),
        family_status={name: state.status for name, state in families.items()},
        support_family_count=k,
        support_ratio_of_4=(k / n) if n else None,
        configured_family_count=n,
        selected_run_id=selection.run_id if selection is not None else None,
        support_run_ids=[item.representative.run_id for item in (clique or []) if item.representative],
        rule_version=config.rule_version,
        review_reason=outcome.reason,
        # DNSMOS is not a gold gate on this path. not_required is not uncalibrated.
        quality_state=_quality_state_not_gating(acoustic, diagnosis),
        noise_diagnosis=dict(diagnosis or {}),
        disposition=legacy.annotation_state,
        max_chars_per_sec=rate_max,
        implausible_routes=list(rate_routes or []),
        category=outcome.category,
        subtype=outcome.subtype,
        status=outcome.status,
        reason_codes=list(outcome.reason_codes),
        chinese_available_family_count=m,
        support_ratio_of_chinese=(k / m) if m else None,
        char_comparable=char_comparable,
        selected_family=selection.family if selection is not None else None,
        selected_raw_text=selection.raw_text if selection is not None else None,
        selection_distance=selection.distance if selection is not None else None,
        transcript_support_count=selection.transcript_support_count if selection is not None else None,
        tie_break_reason=selection.tie_break_reason if selection is not None else None,
        tolerance_version=str(getattr(config, "tolerance_version", TOLERANCE_VERSION)),
        auxiliary_tags=list(outcome.auxiliary),
        semantic_evidence=[
            {
                "kind": item.kind,
                "verdict": item.verdict,
                "families": list(item.families),
                "citations": list(item.citations),
                "version": item.version,
                "conflict_type": item.conflict_type,
            }
            for item in outcome.semantic
        ],
        acoustic_evidence=acoustic.as_dict(),
        language_by_run=dict(languages),
        abstain_reasons=abstain,
        selection_trace=trace,
        annotation_state=legacy.annotation_state,
        voicemail_library_version=library_version,
        ),
        config,
        routes,
    )
