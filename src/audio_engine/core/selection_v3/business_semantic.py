"""024 business-semantic classifier. Opt-in via rule_version.

Business agreement decides the class. Character distance only ranks how much
wording evidence is missing; it does not skip verification or mint gold.
Automatic rows stay ``auto_classified`` and ``is_human_verified=false``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

from audio_engine.core.sample import Sample
from audio_engine.core.selection_v3.acoustic_evidence import collect_acoustic_evidence
from audio_engine.core.selection_v3.config import SelectionV3Config
from audio_engine.core.selection_v3.gold_select import FamilyRep, select_representative_text
from audio_engine.core.selection_v3.input_contract import classify_run_status, is_physically_invalid
from audio_engine.core.selection_v3.legacy_map import map_business_v4
from audio_engine.core.selection_v3.result import ClassificationResultV3
from audio_engine.core.selection_v3.semantic_verify import (
    VerifyRequest,
    build_callable_verifier,
    extract_business_fields,
)
from audio_engine.core.selection_v3.classify_text import (
    EMPTY_HOTWORD_ECHO,
    EMPTY_NON_CHINESE,
    EMPTY_PROMPT_ECHO,
    apply_route_audit,
)
from audio_engine.core.selection_v3.speech_rate import assess_speech_rate
from audio_engine.core.selection_v3.text import raw_transcript_text
from audio_engine.core.selection_v3.text_tolerance import (
    TextLayers,
    build_layers,
    contains_control_tag,
    content_language,
    has_lexical_content,
    tolerant_distance,
)
from audio_engine.core.selection_v3.types import (
    CATEGORY_BUSINESS_CONSISTENT,
    CATEGORY_HARDCASE,
    CATEGORY_NON_SPEECH,
    CATEGORY_SEMANTIC_RISK,
    CATEGORY_VOICEMAIL,
    COVERAGE_AUTO,
    COVERAGE_HUMAN,
    COVERAGE_UNRESOLVED,
    FAMILY_INCOMPLETE,
    FAMILY_STABLE_EMPTY,
    FAMILY_STABLE_TEXT,
    FAMILY_UNSTABLE_PRESENCE,
    FAMILY_UNSTABLE_SEMANTIC,
    FAMILY_UNSTABLE_TEXT,
    LABEL_SOURCE_NONE,
    LABEL_TIER_NONE,
    LABEL_TIER_NO_TRANSCRIPT,
    LABEL_TIER_PSEUDO_VERBATIM_HIGH,
    LABEL_TIER_SEMANTIC_ONLY,
    RUN_STATUS_FAILED,
    RUN_STATUS_MISSING,
    RUN_STATUS_SUCCESS_EMPTY,
    RUN_STATUS_SUCCESS_TEXT,
    STATUS_AUTO_CLASSIFIED,
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
_ABSENT_EVENTS = frozenset({"silence", "music", "tone", "environment"})
_TECHNICAL_GAPS = frozenset(
    {"run_failed_or_missing", "speech_rate_quarantine", "single_chinese_route"}
)
_PROTOCOL_NEEDLES = (
    "i'm sorry",
    "i am sorry",
    "i cannot transcribe",
    "i can't transcribe",
    "the user has not provided",
    "the following is a transcription",
)


def support_ratio_met(k: int, m: int, *, min_families: int = 2) -> bool:
    """Integer 2/3: ``3*k >= 2*m``. Does not use a truncated decimal threshold.

    ``2/3 + 1e-9`` is still below a config value of ``0.6666667``. That float
    check is not used here. Passing this check is not enough when a leftover
    family still has an unresolved business objection.
    """
    if k < min_families or m < min_families or m <= 0:
        return False
    return 3 * k >= 2 * m


def legacy_float_ratio_accepts(k: int, m: int, *, configured: float = 0.6666667, eps: float = 1e-9) -> bool:
    """The 022 comparison. Exposed so the 2/3 boundary can be regression-tested."""
    if m <= 0:
        return False
    return (k / m) + eps >= configured


@dataclass
class RouteRec:
    run_id: str
    family: str
    status: str
    layers: TextLayers
    content_language: str
    lexical: str
    quarantined: bool = False


@dataclass
class FamilyRec:
    family: str
    routes: list[RouteRec]
    status: str
    representative: RouteRec | None = None
    business_vote: bool = False
    verbatim_stable: bool = False
    business_conflict: bool = False
    exclusion_reason: str | None = None
    language: str = ""
    fields: Any = None


@dataclass
class Outcome:
    category: str | None
    status: str
    reason: str
    coverage: str
    reason_codes: list[str] = field(default_factory=list)
    subtype: str | None = None
    auxiliary: list[str] = field(default_factory=list)
    candidate: str | None = None
    selection: Any = None
    label_grade: str = LABEL_TIER_NONE
    commitment: str = "unknown"
    usage_blocks: list[str] = field(default_factory=list)
    semantic: list[dict[str, Any]] = field(default_factory=list)


def _library(config: SelectionV3Config) -> VoicemailLibrary:
    path = str(getattr(config, "voicemail_strong_path", "") or "")
    if path not in _LIBRARY_CACHE:
        _LIBRARY_CACHE[path] = load_voicemail_library(path or None)
    return _LIBRARY_CACHE[path]


def _homophones(config: SelectionV3Config) -> tuple[tuple[str, str], ...]:
    raw = getattr(config, "homophone_pairs", None) or (("先声", "先生"), ("嘀声", "滴声"), ("嘀一声", "滴一声"))
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
            if config.uses_chinese_only_text():
                lexical = layers.classify_text
                language = layers.language
            else:
                lexical = layers.comparison_text if has_lexical_content(raw) else ""
                language = content_language(raw)
            quarantined = str(key) in blocked
            if quarantined:
                status = RUN_STATUS_FAILED
            elif status in {RUN_STATUS_FAILED, RUN_STATUS_MISSING}:
                pass
            elif not lexical:
                status = RUN_STATUS_SUCCESS_EMPTY
            else:
                status = RUN_STATUS_SUCCESS_TEXT
            routes.append(
                RouteRec(
                    run_id=str(key),
                    family=family,
                    status=status,
                    layers=layers,
                    content_language=language,
                    lexical=lexical,
                    quarantined=quarantined,
                )
            )
    return routes


def _run_order(config: SelectionV3Config, family: str) -> list[str]:
    return [str(key) for key in config.model_families.get(family, [])]


def _request(left: RouteRec, right: RouteRec, rule_version: str) -> VerifyRequest:
    return VerifyRequest(
        left_raw=left.layers.raw_text,
        right_raw=right.layers.raw_text,
        left_transcript=left.layers.transcript_text,
        right_transcript=right.layers.transcript_text,
        left_language=left.content_language,
        right_language=right.content_language,
        rule_version=rule_version,
    )


def _relation(left: RouteRec, right: RouteRec, verifier: Any, rule_version: str) -> str:
    """Always ask the verifier. Distance is not a reason to skip."""
    if not left.lexical or not right.lexical:
        if not left.lexical and not right.lexical:
            return "insufficient"
        return "conflict"
    result = verifier.verify(_request(left, right, rule_version))
    if result.verdict == "equivalent":
        return "equivalent"
    if result.verdict == "conflict":
        return "conflict"
    return "insufficient"


def _analyze_family(
    family: str,
    routes: list[RouteRec],
    config: SelectionV3Config,
    verifier: Any,
) -> FamilyRec:
    members = [route for route in routes if route.family == family]
    if any(route.quarantined for route in members):
        return FamilyRec(
            family=family,
            routes=members,
            status=FAMILY_INCOMPLETE,
            exclusion_reason="speech_rate_quarantine",
        )
    if any(_protocol_error(route.layers.transcript_text) for route in members if route.lexical):
        return FamilyRec(
            family=family,
            routes=members,
            status=FAMILY_INCOMPLETE,
            exclusion_reason="protocol_error",
        )
    if any(route.status in {RUN_STATUS_FAILED, RUN_STATUS_MISSING} for route in members):
        return FamilyRec(
            family=family,
            routes=members,
            status=FAMILY_INCOMPLETE,
            exclusion_reason="run_failed_or_missing",
        )
    text_routes = [route for route in members if route.status == RUN_STATUS_SUCCESS_TEXT and route.lexical]
    empty = [route for route in members if route.status == RUN_STATUS_SUCCESS_EMPTY or not route.lexical]
    if members and not text_routes:
        discarded = bool(empty) and all(
            set(route.layers.empty_reason_codes)
            & {EMPTY_NON_CHINESE, EMPTY_PROMPT_ECHO, EMPTY_HOTWORD_ECHO}
            for route in empty
        )
        return FamilyRec(
            family=family,
            routes=members,
            status=FAMILY_STABLE_EMPTY,
            exclusion_reason="classify_text_discarded" if discarded else "success_empty",
            language="empty",
        )
    if text_routes and empty:
        return FamilyRec(
            family=family,
            routes=members,
            status=FAMILY_UNSTABLE_PRESENCE,
            exclusion_reason="presence_unstable",
        )
    if len(text_routes) < 2:
        return FamilyRec(
            family=family,
            routes=members,
            status=FAMILY_INCOMPLETE,
            exclusion_reason="single_route",
            language=text_routes[0].content_language if text_routes else "",
        )
    left, right = text_routes[0], text_routes[1]
    relation = _relation(left, right, verifier, config.rule_version)
    if relation == "conflict":
        return FamilyRec(
            family=family,
            routes=members,
            status=FAMILY_UNSTABLE_SEMANTIC,
            business_conflict=True,
            exclusion_reason="intra_family_conflict",
            language=left.content_language,
            fields=extract_business_fields(left.layers.transcript_text),
        )
    if relation != "equivalent":
        return FamilyRec(
            family=family,
            routes=members,
            status=FAMILY_UNSTABLE_TEXT,
            exclusion_reason="business_unverified",
            language=left.content_language,
        )
    order = _run_order(config, family)
    chosen = sorted(text_routes, key=lambda route: order.index(route.run_id) if route.run_id in order else 99)[0]
    dist = tolerant_distance(left.layers.tolerant_key, right.layers.tolerant_key)
    verbatim = left.layers.comparison_text == right.layers.comparison_text or (
        dist is not None and dist <= float(getattr(config, "recall_max_distance", 0.10))
    )
    fields = extract_business_fields(chosen.layers.transcript_text)
    chinese = chosen.content_language == "zh"
    return FamilyRec(
        family=family,
        routes=members,
        status=FAMILY_STABLE_TEXT,
        representative=chosen,
        business_vote=chinese,
        verbatim_stable=bool(verbatim and chinese),
        language=chosen.content_language,
        fields=fields,
    )


def _largest_cliques(votes: list[FamilyRec], config: SelectionV3Config, verifier: Any) -> list[list[FamilyRec]]:
    n = len(votes)
    if n < 2:
        return []
    cliques: list[list[FamilyRec]] = []
    for size in range(n, 1, -1):
        for combo in combinations(range(n), size):
            group = [votes[i] for i in combo]
            if _clique_ok(group, config, verifier):
                cliques.append(group)
        if cliques:
            break
    return cliques


def _clique_ok(group: list[FamilyRec], config: SelectionV3Config, verifier: Any) -> bool:
    for i, left in enumerate(group):
        for right in group[i + 1 :]:
            if left.representative is None or right.representative is None:
                return False
            relation = _relation(left.representative, right.representative, verifier, config.rule_version)
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

    return sorted(cliques, key=key)[0]


def _speech_presence(quality: dict[str, Any], labels: dict[str, Any], *, all_empty: bool) -> dict[str, Any]:
    raw = labels.get("speech_presence")
    if not isinstance(raw, dict):
        raw = quality.get("speech_presence")
    if not isinstance(raw, dict):
        return {
            "status": "capability_missing",
            "confirmed_absent": False,
            "event": None,
            "reason": "speech_presence_not_deployed",
        }
    version = str(raw.get("model_version") or raw.get("version") or "").strip()
    if raw.get("placeholder") is True or raw.get("deployed") is not True or raw.get("calibrated") is not True or not version:
        return {
            "status": "capability_missing",
            "confirmed_absent": False,
            "event": None,
            "reason": "speech_presence_not_deployed",
        }
    sources = [str(item) for item in (raw.get("sources") or [])]
    if sources in (["vad"], ["dnsmos"]):
        return {
            "status": "insufficient",
            "confirmed_absent": False,
            "event": None,
            "reason": "single_score_not_sufficient",
        }
    event = str(raw.get("event") or "")
    present = raw.get("speech_present")
    if present is False and event in _ABSENT_EVENTS and all_empty:
        return {
            "status": "confirmed_absent",
            "confirmed_absent": True,
            "event": event,
            "reason": "calibrated_event_and_cross_family_empty",
            "version": version,
        }
    if present is True:
        return {
            "status": "speech_present",
            "confirmed_absent": False,
            "event": event or "speech",
            "reason": "speech_heard",
            "version": version,
        }
    return {
        "status": "insufficient",
        "confirmed_absent": False,
        "event": event or None,
        "reason": "speech_presence_unconfirmed",
    }


def classify_business_semantic(
    sample: Sample,
    config: SelectionV3Config,
    *,
    voicemail_pattern: Any = None,
    exclude_families: set[str] | None = None,
    verifier: Any | None = None,
) -> ClassificationResultV3:
    del voicemail_pattern
    from audio_engine.core.selection_v3.noise_trigger import ensure_trigger_record, uses_asr_anomaly_noise

    if uses_asr_anomaly_noise(config):
        ensure_trigger_record(sample, config)
    diagnosis = sample.labels.get("noise_diagnosis") if isinstance(sample.labels.get("noise_diagnosis"), dict) else None
    quality = sample.quality if isinstance(sample.quality, dict) else {}
    excluded = set(exclude_families or ())
    family_order = [name for name in config.ordered_families() if name not in excluded]
    active = verifier or build_callable_verifier(
        str(getattr(config, "semantic_verifier_mode", "business_local") or "business_local"),
        endpoint=str(getattr(config, "semantic_verifier_endpoint", "") or ""),
        timeout_sec=float(getattr(config, "semantic_verifier_timeout_sec", 5.0) or 5.0),
        max_retries=int(getattr(config, "max_route_retries", 1) or 0),
        protocol=str(getattr(config, "semantic_verifier_protocol", "auto") or "auto"),
        model_version=str(getattr(config, "semantic_verifier_model", "") or ""),
    )

    if is_physically_invalid(sample):
        return _finish(
            sample,
            config,
            Outcome(
                None,
                STATUS_EXCLUDED,
                "broken_or_invalid_audio",
                COVERAGE_UNRESOLVED,
                ["invalid_audio"],
                usage_blocks=["invalid_audio"],
            ),
            families={},
            routes=[],
            languages={},
            acoustic=collect_acoustic_evidence(quality, sample.labels),
            speech_presence={"status": "not_applicable"},
            verifier_version=getattr(active, "version", ""),
        )

    duration = float(sample.duration) if sample.duration is not None else None
    probe = _collect_routes(sample, config, set(), family_order=family_order)
    rate = assess_speech_rate(
        _rate_views(probe),
        duration_sec=duration,
        max_chars_per_sec=config.max_chars_per_sec,
        min_text_chars=config.speech_rate_min_text_chars,
    )
    blocked = set(rate.implausible_routes) if rate.triggered else set()
    routes = _collect_routes(sample, config, blocked, family_order=family_order) if blocked else probe
    families = {
        family: _analyze_family(family, routes, config, active) for family in family_order
    }
    acoustic = collect_acoustic_evidence(quality, sample.labels)
    languages = {route.run_id: route.content_language for route in routes}
    presence = _speech_presence(
        quality,
        sample.labels,
        all_empty=_all_success_empty(families),
    )
    library = _library(config)
    voicemail = _voicemail(families, library)
    votes = [families[name] for name in family_order if families[name].business_vote]
    cliques = _largest_cliques(votes, config, active)
    clique = _pick_clique(cliques, family_order)
    evidence = _semantic_rows(families, clique, active, config.rule_version)
    if not config.uses_chinese_only_text():
        evidence.extend(_cross_language_rows(families, active, config.rule_version))
    outcome = _decide(
        sample=sample,
        families=families,
        clique=clique,
        votes=votes,
        config=config,
        voicemail=voicemail,
        presence=presence,
        evidence=evidence,
        family_order=family_order,
        non_chinese_skipped=_non_chinese_long(probe, blocked),
    )
    return _finish(
        sample,
        config,
        outcome,
        families=families,
        routes=routes,
        languages=languages,
        acoustic=acoustic,
        speech_presence=presence,
        clique=clique,
        vote_count=len(votes),
        rate_routes=sorted(blocked),
        rate_max=rate.max_chars_per_sec_observed if blocked else None,
        library_version=library.version,
        diagnosis=diagnosis,
        verifier_version=getattr(active, "version", ""),
        verifier_remote_calls=int(getattr(active, "remote_calls", 0) or 0),
    )


def _rate_views(routes: list[RouteRec]):
    """Chinese character rate only. A long non-Chinese string is not audio failure."""
    from audio_engine.core.selection_v3.family_evidence import RouteView

    views = []
    for route in routes:
        if route.content_language != "zh":
            continue
        views.append(
            RouteView(
                run_id=route.run_id,
                family=route.family,
                status=route.status,
                raw_text=route.layers.raw_text,
                comparison_text=route.layers.comparison_text,
            )
        )
    return views


def _non_chinese_long(routes: list[RouteRec], blocked: set[str]) -> bool:
    return any(route.content_language not in {"zh", "empty"} and route.lexical and route.run_id not in blocked for route in routes)


def _voicemail(families: dict[str, FamilyRec], library: VoicemailLibrary):
    texts = {}
    for name, state in families.items():
        if state.representative is None:
            continue
        texts[name] = state.representative.layers.transcript_text
    return agreed_scene(texts, library)


def _semantic_rows(
    families: dict[str, FamilyRec],
    clique: list[FamilyRec],
    verifier: Any,
    rule_version: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for state in families.values():
        if not state.business_conflict:
            continue
        bodies = [route.layers.transcript_text for route in state.routes if route.lexical]
        rows.append(
            {
                "kind": "intra_family_conflict",
                "verdict": "conflict",
                "families": [state.family],
                "citations": bodies[:2],
                "version": getattr(verifier, "version", ""),
                "conflict_type": "intra_family_conflict",
            }
        )
    compared = [state for state in families.values() if state.representative is not None and state.business_vote]
    for i, left in enumerate(compared):
        for right in compared[i + 1 :]:
            assert left.representative and right.representative
            relation = _relation(left.representative, right.representative, verifier, rule_version)
            if relation != "conflict":
                continue
            rows.append(
                {
                    "kind": "business_conflict",
                    "verdict": "conflict",
                    "families": [left.family, right.family],
                    "citations": [
                        left.representative.layers.transcript_text,
                        right.representative.layers.transcript_text,
                    ],
                    "version": getattr(verifier, "version", ""),
                    "conflict_type": "business_conflict",
                }
            )
    if clique:
        rows.append(
            {
                "kind": "business_clique",
                "verdict": "equivalent",
                "families": [member.family for member in clique],
                "citations": [
                    member.representative.layers.transcript_text
                    for member in clique
                    if member.representative is not None
                ],
                "version": getattr(verifier, "version", ""),
                "conflict_type": None,
            }
        )
    return rows


def _cross_language_rows(families: dict[str, FamilyRec], verifier: Any, rule_version: str) -> list[dict[str, Any]]:
    """Verified foreign text can support or conflict. It never adds a Chinese verbatim vote."""
    chinese = [state for state in families.values() if state.business_vote and state.representative is not None]
    foreign = [
        state
        for state in families.values()
        if state.representative is not None and state.language in {"en", "mixed", "unknown"}
    ]
    rows: list[dict[str, Any]] = []
    for other in foreign:
        for rep in chinese:
            assert other.representative and rep.representative
            relation = _relation(rep.representative, other.representative, verifier, rule_version)
            if relation == "insufficient":
                continue
            rows.append(
                {
                    "kind": "cross_language",
                    "verdict": relation,
                    "families": [rep.family, other.family],
                    "citations": [
                        rep.representative.layers.transcript_text,
                        other.representative.layers.transcript_text,
                    ],
                    "version": getattr(verifier, "version", ""),
                    "conflict_type": "cross_language_polarity" if relation == "conflict" else None,
                    "verbatim_vote": False,
                }
            )
    return rows


def _protocol_error(text: str) -> bool:
    low = str(text or "").lower()
    return any(needle in low for needle in _PROTOCOL_NEEDLES)


def _mixed_presence(families: dict[str, FamilyRec]) -> bool:
    emptyish = {
        FAMILY_STABLE_EMPTY,
        FAMILY_UNSTABLE_PRESENCE,
    }
    has_empty = any(
        state.status in emptyish and state.exclusion_reason != "classify_text_discarded"
        for state in families.values()
    )
    has_text = any(state.business_vote or state.status == FAMILY_STABLE_TEXT for state in families.values())
    return has_empty and has_text


def _retry_open(sample: Sample, config: SelectionV3Config) -> bool:
    tried = int(sample.labels.get("route_retry_count") or 0)
    return tried < int(getattr(config, "max_route_retries", 1) or 0)


def _conflicts(families: dict[str, FamilyRec], evidence: list[dict[str, Any]]) -> bool:
    if any(state.business_conflict for state in families.values()):
        return True
    return any(item.get("verdict") == "conflict" for item in evidence)


def _voicemail_blocked_by_human(families: dict[str, FamilyRec]) -> bool:
    acts = set()
    for state in families.values():
        fields = state.fields
        if fields is None and state.representative is not None:
            fields = extract_business_fields(state.representative.layers.transcript_text)
        if fields is None:
            continue
        acts.add(fields.speech_act)
    if "voicemail" in acts and acts & {"refusal", "acceptance", "absence", "continue_listening"}:
        return True
    return False


def _decide(
    *,
    sample: Sample,
    families: dict[str, FamilyRec],
    clique: list[FamilyRec],
    votes: list[FamilyRec],
    config: SelectionV3Config,
    voicemail,
    presence: dict[str, Any],
    evidence: list[dict[str, Any]],
    family_order: list[str],
    non_chinese_skipped: bool,
) -> Outcome:
    auxiliary: list[str] = []
    if non_chinese_skipped:
        auxiliary.append("non_chinese_rate_not_applied")
    blocks = ["not_human_accepted", "not_formal_eval"]
    if _conflicts(families, evidence):
        return Outcome(
            CATEGORY_SEMANTIC_RISK,
            STATUS_MANUAL_REVIEW,
            "business_conflict_requires_review",
            COVERAGE_HUMAN,
            ["semantic_risk", *[str(item.get("kind")) for item in evidence if item.get("verdict") == "conflict"]],
            auxiliary=auxiliary,
            semantic=evidence,
            usage_blocks=["not_human_accepted", "risk_unresolved"],
        )

    if voicemail is not None and not _voicemail_blocked_by_human(families):
        scene, subtype, hit_families, _evidence = voicemail
        if len(hit_families) >= 2:
            selection = _select(clique or _voicemail_clique(families, hit_families), config, family_order)
            grade = _grade(selection, verbatim=False, category=CATEGORY_VOICEMAIL)
            return Outcome(
                CATEGORY_VOICEMAIL,
                STATUS_AUTO_CLASSIFIED,
                "voicemail_scene_consensus",
                COVERAGE_AUTO,
                ["voicemail_scene", f"scene:{scene}"],
                subtype=subtype,
                auxiliary=["voicemail", *auxiliary],
                candidate=None if grade == LABEL_TIER_NO_TRANSCRIPT else (selection.transcript_text if selection else None),
                selection=selection,
                label_grade=grade,
                commitment="none",
                usage_blocks=[*blocks, "not_verbatim_sft"],
                semantic=evidence,
            )

    if _all_success_empty(families):
        return _empty_outcome(presence, auxiliary)

    if _mixed_presence(families):
        if presence.get("confirmed_absent") is True:
            return Outcome(
                CATEGORY_SEMANTIC_RISK,
                STATUS_MANUAL_REVIEW,
                "text_on_confirmed_absent_audio",
                COVERAGE_HUMAN,
                ["semantic_risk", "hallucination_unconfirmed", "presence_conflict"],
                auxiliary=auxiliary,
                semantic=evidence,
                usage_blocks=["not_human_accepted", "risk_unresolved"],
            )
        if presence.get("status") != "speech_present":
            return Outcome(
                None,
                STATUS_HOLD,
                "empty_and_text_presence_unconfirmed",
                COVERAGE_UNRESOLVED,
                ["presence_unconfirmed", str(presence.get("reason") or "speech_presence_capability_missing")],
                auxiliary=auxiliary,
                usage_blocks=["not_human_accepted", "not_non_speech", "presence_unconfirmed"],
            )

    gaps = [
        f"{name}:{state.exclusion_reason}"
        for name, state in families.items()
        if state.exclusion_reason in _TECHNICAL_GAPS
    ]
    min_k = int(getattr(config, "min_support_families", 2))
    k = len(clique)
    m = len(votes)
    eligible = support_ratio_met(k, m, min_families=min_k)
    # A leftover vote is not in the clique, so it is not equivalent. Confirmed
    # conflicts already returned above. An unknown leftover is still an
    # unresolved objection: 2/3 passing does not auto-clear it.
    remaining = [state for state in votes if state not in clique]
    if clique and remaining:
        return Outcome(
            None,
            STATUS_HOLD,
            "residual_business_objection",
            COVERAGE_UNRESOLVED,
            ["business_unverified", "support_ratio_not_sufficient_alone"],
            auxiliary=auxiliary,
            semantic=evidence,
            usage_blocks=[*blocks, "objection_unresolved"],
        )

    if not eligible:
        if gaps and _retry_open(sample, config):
            return Outcome(
                None,
                STATUS_RETRY,
                "route_gap_retry_open",
                COVERAGE_UNRESOLVED,
                ["route_gap", *gaps],
                auxiliary=auxiliary,
                usage_blocks=[*blocks, "route_retry_open"],
            )
        if gaps:
            return Outcome(
                CATEGORY_HARDCASE,
                STATUS_MANUAL_REVIEW,
                "independent_evidence_insufficient",
                COVERAGE_HUMAN,
                ["hardcase", "independent_evidence_insufficient", *gaps],
                auxiliary=auxiliary,
                usage_blocks=[*blocks, "evidence_insufficient"],
            )
        if any(state.language not in {"", "zh", "empty"} for state in families.values() if state.status == FAMILY_STABLE_TEXT):
            return Outcome(
                None,
                STATUS_HOLD,
                "language_unresolved",
                COVERAGE_UNRESOLVED,
                ["language_unresolved"],
                auxiliary=auxiliary,
                usage_blocks=[*blocks, "not_chinese_verbatim"],
            )
        return Outcome(
            None,
            STATUS_HOLD,
            "business_unverified",
            COVERAGE_UNRESOLVED,
            ["business_unverified"],
            auxiliary=auxiliary,
            usage_blocks=[*blocks, "verifier_residual"],
        )

    selection = _select(clique, config, family_order)
    if selection is None or contains_control_tag(selection.transcript_text or ""):
        return Outcome(
            None,
            STATUS_HOLD,
            "selection_failed",
            COVERAGE_UNRESOLVED,
            ["selection_failed"],
            usage_blocks=blocks,
        )
    fields = extract_business_fields(selection.transcript_text or "")
    verbatim = all(member.verbatim_stable for member in clique) and _same_wording(clique)
    grade = _grade(selection, verbatim=verbatim, category=CATEGORY_BUSINESS_CONSISTENT)
    usage = list(blocks)
    if grade != LABEL_TIER_PSEUDO_VERBATIM_HIGH:
        usage.append("not_verbatim_sft")
    else:
        usage.append("pending_spot_audit")
    if fields.commitment == "undetermined":
        usage.append("not_authorization")
    if gaps:
        auxiliary = [*auxiliary, "route_gap_not_blocking"]
    return Outcome(
        CATEGORY_BUSINESS_CONSISTENT,
        STATUS_AUTO_CLASSIFIED,
        "business_field_consensus",
        COVERAGE_AUTO,
        ["business_consistent", *([f"route_gap:{gap}" for gap in gaps])],
        subtype=fields.speech_act,
        auxiliary=auxiliary,
        candidate=selection.transcript_text,
        selection=selection,
        label_grade=grade,
        commitment=fields.commitment,
        usage_blocks=usage,
        semantic=evidence,
    )


def _empty_outcome(presence: dict[str, Any], auxiliary: list[str]) -> Outcome:
    if presence.get("confirmed_absent") is True:
        event = str(presence.get("event") or "environment")
        return Outcome(
            CATEGORY_NON_SPEECH,
            STATUS_AUTO_CLASSIFIED,
            "speech_presence_confirmed_absent",
            COVERAGE_AUTO,
            ["non_speech", f"event:{event}"],
            subtype=event,
            auxiliary=auxiliary,
            label_grade=LABEL_TIER_NO_TRANSCRIPT,
            commitment="none",
            usage_blocks=["not_human_accepted", "not_asr_train", "no_transcript"],
        )
    if presence.get("status") == "speech_present":
        return Outcome(
            CATEGORY_HARDCASE,
            STATUS_MANUAL_REVIEW,
            "empty_text_but_speech_present",
            COVERAGE_HUMAN,
            ["hardcase", "speech_present_no_transcript"],
            auxiliary=auxiliary,
            usage_blocks=["not_human_accepted", "speech_without_text"],
        )
    return Outcome(
        None,
        STATUS_HOLD,
        "empty_output_presence_unconfirmed",
        COVERAGE_UNRESOLVED,
        ["presence_unconfirmed", str(presence.get("reason") or "speech_presence_capability_missing")],
        auxiliary=auxiliary,
        usage_blocks=["not_human_accepted", "not_non_speech", "presence_unconfirmed"],
    )


def _voicemail_clique(families: dict[str, FamilyRec], hit_families: list[str]) -> list[FamilyRec]:
    return [families[name] for name in hit_families if name in families and families[name].representative is not None]


def _same_wording(clique: list[FamilyRec]) -> bool:
    keys = {
        member.representative.layers.tolerant_key
        for member in clique
        if member.representative is not None
    }
    return len(keys) == 1


def _grade(selection: Any, *, verbatim: bool, category: str) -> str:
    if category == CATEGORY_NON_SPEECH or selection is None or not getattr(selection, "transcript_text", None):
        return LABEL_TIER_NO_TRANSCRIPT
    if verbatim:
        return LABEL_TIER_PSEUDO_VERBATIM_HIGH
    return LABEL_TIER_SEMANTIC_ONLY


def _select(clique: list[FamilyRec], config: SelectionV3Config, family_order: list[str]):
    reps = []
    for item in clique:
        route = item.representative
        if route is None or contains_control_tag(route.layers.transcript_text):
            continue
        reps.append(
            FamilyRep(
                family=item.family,
                run_id=route.run_id,
                transcript_text=route.layers.transcript_text,
                raw_text=route.layers.raw_text,
                tolerant_key=route.layers.tolerant_key,
                comparison_text=route.layers.comparison_text,
            )
        )
    if not reps:
        return None
    run_order = []
    for family in family_order:
        run_order.extend(_run_order(config, family))
    return select_representative_text(
        reps,
        family_order=family_order,
        run_order=run_order,
        tolerance_version=str(getattr(config, "tolerance_version", TOLERANCE_VERSION)),
    )


def _all_success_empty(families: dict[str, FamilyRec]) -> bool:
    return bool(families) and all(state.status == FAMILY_STABLE_EMPTY for state in families.values())


def _quality_state(acoustic, diagnosis: dict | None) -> str:
    from audio_engine.core.selection_v3.noise_trigger import quality_state_for_diagnosis
    from audio_engine.core.selection_v3.types import DNSMOS_STATUS_NOT_REQUIRED

    if diagnosis and diagnosis.get("status") == DNSMOS_STATUS_NOT_REQUIRED:
        return "not_required"
    if diagnosis and diagnosis.get("policy"):
        return quality_state_for_diagnosis(diagnosis)
    return "not_gating"


def _finish(
    sample: Sample,
    config: SelectionV3Config,
    outcome: Outcome,
    *,
    families: dict[str, FamilyRec],
    routes: list[RouteRec],
    languages: dict[str, str],
    acoustic,
    speech_presence: dict[str, Any],
    clique: list[FamilyRec] | None = None,
    vote_count: int = 0,
    rate_routes: list[str] | None = None,
    rate_max: float | None = None,
    library_version: str = "",
    diagnosis: dict | None = None,
    verifier_version: str = "",
    verifier_remote_calls: int = 0,
) -> ClassificationResultV3:
    del sample
    legacy = map_business_v4(
        category=outcome.category,
        status=outcome.status,
        label_grade=outcome.label_grade,
        reason=outcome.reason,
    )
    k = len(clique or [])
    n = len(config.model_families)
    selection = outcome.selection
    candidate = outcome.candidate
    if candidate and contains_control_tag(candidate):
        candidate = None
        outcome.status = STATUS_HOLD
        outcome.coverage = COVERAGE_UNRESOLVED
        outcome.category = None
        legacy = map_business_v4(category=None, status=STATUS_HOLD, label_grade=LABEL_TIER_NO_TRANSCRIPT, reason="control_tag_in_candidate")
    abstain = {name: state.exclusion_reason for name, state in families.items() if state.exclusion_reason}
    trace = {
        "coverage_bucket": outcome.coverage,
        "label_grade": outcome.label_grade,
        "commitment": outcome.commitment,
        "usage_blocks": list(outcome.usage_blocks),
        "verifier_version": verifier_version,
        "verifier_remote_calls": verifier_remote_calls,
        "support_integer_two_thirds": support_ratio_met(k, vote_count) if vote_count else False,
        "prior_information_used": bool(getattr(config, "prior_information_used", False)),
        "speech_presence": dict(speech_presence),
    }
    if selection is not None:
        trace.update(
            {
                "selected_family": selection.family,
                "selected_run_id": selection.run_id,
                "distance": selection.distance,
                "transcript_support_count": selection.transcript_support_count,
                "tie_break_reason": selection.tie_break_reason,
                "tolerance_version": selection.tolerance_version,
            }
        )
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
        quality_state=_quality_state(acoustic, diagnosis),
        noise_diagnosis=dict(diagnosis or {}),
        disposition=legacy.annotation_state,
        max_chars_per_sec=rate_max,
        implausible_routes=list(rate_routes or []),
        category=outcome.category,
        subtype=outcome.subtype,
        status=outcome.status,
        reason_codes=list(outcome.reason_codes),
        chinese_available_family_count=vote_count,
        support_ratio_of_chinese=(k / vote_count) if vote_count else None,
        char_comparable=all(lang == "zh" for lang in languages.values() if lang not in {"empty"}) if languages else None,
        selected_family=selection.family if selection is not None else None,
        selected_raw_text=selection.raw_text if selection is not None else None,
        selection_distance=selection.distance if selection is not None else None,
        transcript_support_count=selection.transcript_support_count if selection is not None else None,
        tie_break_reason=selection.tie_break_reason if selection is not None else None,
        tolerance_version=str(getattr(config, "tolerance_version", TOLERANCE_VERSION)),
        auxiliary_tags=list(outcome.auxiliary),
        semantic_evidence=list(outcome.semantic),
        acoustic_evidence=acoustic.as_dict(),
        language_by_run=dict(languages),
        abstain_reasons=abstain,
        selection_trace=trace,
        annotation_state=legacy.annotation_state,
        voicemail_library_version=library_version,
        coverage_bucket=outcome.coverage,
        label_grade=outcome.label_grade,
        commitment=outcome.commitment,
        speech_presence=dict(speech_presence),
        usage_blocks=list(outcome.usage_blocks),
        ),
        config,
        routes,
    )


def governance_release_status(labels: dict[str, Any]) -> dict[str, Any]:
    """Source isolation is separate from business classification.

    A missing group flag must not be rewritten to false. Classification can
    finish; a leak-free train/eval release cannot be claimed from that flag.
    """
    missing = labels.get("missing_group_metadata") is True
    role = str(labels.get("dataset_role") or labels.get("reservation_role") or "")
    blocked = missing or role == "governance_hold"
    return {
        "classification_allowed": True,
        "train_eval_release": "blocked" if blocked else "not_claimed",
        "missing_group_metadata": labels.get("missing_group_metadata"),
        "cleared_missing_flag": False,
        "reason": "missing_group_metadata" if missing else ("governance_hold" if role == "governance_hold" else ""),
    }


def partition_coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Mutually exclusive A/H/U. Does not claim the 3,000 human budget is met."""
    counts = {COVERAGE_AUTO: 0, COVERAGE_HUMAN: 0, COVERAGE_UNRESOLVED: 0}
    invalid = 0
    for row in rows:
        bucket = str(row.get("coverage_bucket") or "")
        if bucket not in counts:
            invalid += 1
            counts[COVERAGE_UNRESOLVED] += 1
            continue
        counts[bucket] += 1
    total = len(rows)
    a, h, u = counts[COVERAGE_AUTO], counts[COVERAGE_HUMAN], counts[COVERAGE_UNRESOLVED]
    return {
        "n": total,
        "A": a,
        "H": h,
        "U": u,
        "invalid_bucket": invalid,
        "conserved": a + h + u == total,
        "arithmetic_target_for_30000": {
            "A_at_least": 27000,
            "H_at_most": 3000,
            "U": 0,
            "met": total == 30000 and a >= 27000 and h <= 3000 and u == 0,
        },
        "acceptance_claimed": False,
        "note": "Classification counts are not a listening acceptance. U must be reported, not renamed into A or hardcase.",
    }
