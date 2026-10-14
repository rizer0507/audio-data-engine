"""ASR / text evidence (capability group A) — reuses selection_v3 semantics."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from audio_engine.core.sample import Sample
from audio_engine.core.selection_v3.config import SelectionV3Config
from audio_engine.core.selection_v3.family_evidence import RouteView, collect_route_views
from audio_engine.core.selection_v3.five_class_v2 import FamilyStateBundle, build_family_states
from audio_engine.core.selection_v3.five_class_v2_4_on_demand_acoustic import (
    _family_dual_run_success,
    _voicemail_hit_raw,
)
from audio_engine.core.selection_v3.types import (
    FAMILY_STATE_STABLE_EMPTY,
    FAMILY_STATE_STABLE_TEXT,
    FAMILY_STATE_UNAVAILABLE,
    FAMILY_STATE_UNSTABLE,
)


@dataclass
class AsrTextEvidence:
    """Immutable per-sample ASR/text snapshot for condition evaluation."""

    routes: list[RouteView]
    bundles: list[FamilyStateBundle]
    family_state_by_name: dict[str, str]
    representative_text: dict[str, str | None]
    all_families_dual_success: bool
    dual_success_by_family: dict[str, bool]
    voicemail_hit: dict[str, Any] | None
    input_fingerprint: str
    config_digest: str
    details: dict[str, Any] = field(default_factory=dict)

    def bundle(self, family: str) -> FamilyStateBundle | None:
        for b in self.bundles:
            if b.family == family:
                return b
        return None

    def as_dict(self) -> dict[str, Any]:
        return {
            "family_state_by_name": dict(self.family_state_by_name),
            "representative_text": dict(self.representative_text),
            "all_families_dual_success": self.all_families_dual_success,
            "dual_success_by_family": dict(self.dual_success_by_family),
            "voicemail_hit": self.voicemail_hit,
            "input_fingerprint": self.input_fingerprint,
            "config_digest": self.config_digest,
            "details": dict(self.details),
        }


def _config_digest(config: SelectionV3Config) -> str:
    payload = {
        "families": config.ordered_families(),
        "expected_runs_per_family": config.expected_runs_per_family,
        "classify_text_policy": getattr(config, "classify_text_policy", None),
        "critical_short_responses": list(config.critical_short_responses or []),
        "rule_version": getattr(config, "rule_version", None),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode()
    ).hexdigest()[:16]


def _input_fingerprint(sample: Sample, routes: list[RouteView]) -> str:
    payload = {
        "id": sample.id,
        "sha256": sample.sha256,
        "routes": [
            {
                "run_id": r.run_id,
                "family": r.family,
                "status": r.status,
                "raw_text": r.raw_text,
                "classify_text": r.classify_text,
                "disposition": r.route_disposition,
            }
            for r in routes
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode()
    ).hexdigest()


def build_asr_text_evidence(
    sample: Sample,
    config: SelectionV3Config,
    *,
    voicemail_pattern: Any | None = None,
) -> AsrTextEvidence:
    routes = collect_route_views(sample, config)
    bundles = build_family_states(routes, config)
    expected = int(config.expected_runs_per_family or 2)
    families = list(config.ordered_families())
    dual = {f: _family_dual_run_success(routes, f, expected) for f in families}
    states = {b.family: b.state for b in bundles}
    reps = {
        b.family: (b.representative_text if b.state == FAMILY_STATE_STABLE_TEXT else None)
        for b in bundles
    }
    # Keep stable_empty / unstable / unavailable explicit for diagnostics.
    for b in bundles:
        if b.state == FAMILY_STATE_STABLE_EMPTY:
            reps[b.family] = ""
        elif b.state in {FAMILY_STATE_UNSTABLE, FAMILY_STATE_UNAVAILABLE}:
            reps.setdefault(b.family, None)

    vm = _voicemail_hit_raw(routes, voicemail_pattern)
    digest = _config_digest(config)
    fp = _input_fingerprint(sample, routes)
    return AsrTextEvidence(
        routes=routes,
        bundles=bundles,
        family_state_by_name=states,
        representative_text=reps,
        all_families_dual_success=all(dual.values()) if dual else False,
        dual_success_by_family=dual,
        voicemail_hit=vm,
        input_fingerprint=fp,
        config_digest=digest,
        details={
            "configured_families": families,
            "route_statuses": {r.run_id: r.status for r in routes},
        },
    )
