"""Freeze audit draws before annotation and bind acceptance to the candidate corpus."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Any, Sequence

from audio_engine.core.sample import Sample
from audio_engine.core.annotation_v3.config import AnnotationConfig


def digest_payload(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), default=str).encode()).hexdigest()


def candidate_signature(sample: Sample) -> str:
    labels = sample.labels
    return digest_payload({
        "id": sample.id, "sha256": labels.get("original_audio_sha256") or sample.sha256,
        "group": labels.get("leakage_group_id"), "candidate": labels.get("candidate_text"),
        "rule": labels.get("rule_version"), "type": labels.get("type"),
        "reservation": labels.get("reservation_digest"),
        "role": labels.get("reservation_role") or labels.get("dataset_role"),
        "quality": sample.quality, "transcripts": sample.transcripts,
        "run_identities_digest": labels.get("run_identities_digest"),
        "run_identities_verified": labels.get("run_identities_verified"),
    })


def freeze_audit_plan(samples: Sequence[Sample], config: AnnotationConfig) -> dict[str, Any]:
    from audio_engine.core.dataset_v3.audit import _match_layer, sample_group_balanced
    pool = [s for s in samples if s.labels.get("type") == "pseudo_high"
            and (s.labels.get("reservation_role") or s.labels.get("dataset_role")) == "train_pool"]
    if not pool or len({s.id for s in pool}) != len(pool):
        raise ValueError("audit requires a nonempty unique pseudo_high train pool")
    for sample in pool:
        if sample.labels.get("is_human_verified") or sample.labels.get("annotator_id"):
            raise ValueError("freeze audit plan before annotation")
        if not sample.labels.get("leakage_group_id") or not sample.sha256:
            raise ValueError("audit candidate requires audio hash and leakage group")
        if not sample.labels.get("run_identities_digest") or sample.labels.get("run_identities_verified") is not True:
            raise ValueError("audit candidate requires verified eight-run identities from prepare")
        if not all(sample.quality.get(k) for k in ("dnsmos_model_digest", "dnsmos_preprocess_version", "quality_policy_version")):
            raise ValueError("audit candidate requires versioned DNSMOS evidence")
        if sample.quality.get("noise_band") not in {"clean", "moderate"} or sample.quality.get("noise_risk") is not False:
            raise ValueError("audit candidate requires calibrated clean/moderate quality")
    # Protected strata are predicted strata, never selected using audit outcomes.
    views = []
    for sample in pool:
        view = sample.model_copy(deep=True)
        view.labels["human_semantic"] = view.labels.get("polarity")
        view.labels["human_noise"] = view.quality.get("noise_band")
        view.labels["quality"] = dict(view.quality)
        views.append(view)
    policy = config.pseudo_audit
    draws = {"overall": [s.id for s in sample_group_balanced(
        pool, seed=policy.seed, min_groups=policy.min_groups_overall, salt="overall")]}
    for name, layer in policy.protected_layers.items():
        draws[name] = [s.id for s in sample_group_balanced(
            [s for s in views if _match_layer(s, layer)], seed=policy.seed,
            min_groups=layer.min_groups, salt=f"layer:{name}")]
    payload = {"schema_version": "pseudo_audit_plan_v1", "policy_digest": digest_payload(asdict(config)),
               "scope": {s.id: candidate_signature(s) for s in sorted(pool, key=lambda s: s.id)},
               "draws": draws}
    return {**payload, "digest": digest_payload(payload)}


def validate_audit_plan(plan: dict[str, Any], config: AnnotationConfig) -> None:
    payload = {k: v for k, v in plan.items() if k != "digest"}
    if plan.get("digest") != digest_payload(payload):
        raise ValueError("audit plan digest mismatch")
    if plan.get("policy_digest") != digest_payload(asdict(config)):
        raise ValueError("audit policy changed after plan freeze")


def stamp_audit_draw(samples: Sequence[Sample], plan: dict[str, Any]) -> list[Sample]:
    selected = {sid for ids in plan["draws"].values() for sid in ids}
    out = []
    for source in samples:
        sample = source.model_copy(deep=True)
        if sample.id in plan["scope"]:
            sample.labels["review_queue"] = "pseudo_audit_pool"
            sample.labels["pseudo_audit_sampled"] = False
        if sample.id in selected:
            sample.labels.update(pseudo_audit_sampled=True, review_queue="pseudo_audit",
                                 annotation_state="pending", pseudo_audit_plan_digest=plan["digest"])
        out.append(sample)
    return out


def validate_publish_audit(samples: Sequence[Sample], report: dict[str, Any] | None) -> None:
    """Validate evidence at consumption time, independently of auto_accept stamps."""
    if not report or not report.get("passed") or report.get("stop_publish"):
        raise ValueError("passing pseudo audit report required")
    raw = {k: v for k, v in report.items() if k != "report_digest"}
    if report.get("report_digest") != digest_payload(raw) or not report.get("scope_digest"):
        raise ValueError("pseudo audit report digest/scope missing or invalid")
    for sample in samples:
        if report.get("scope", {}).get(sample.id) != candidate_signature(sample):
            raise ValueError(f"pseudo audit scope mismatch: {sample.id}")
        if sample.labels.get("pseudo_audit_report_digest") != report["report_digest"]:
            raise ValueError(f"pseudo audit stamp mismatch: {sample.id}")
