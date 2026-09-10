"""Load annotation_v3 policy from YAML."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from audio_engine.core.annotation_v3.types import ANNOTATION_VERSION


@dataclass
class DualReviewPolicy:
    """Which sample classes require independent dual review."""

    eval_formal: bool = True
    priority_p0: bool = True
    empty_gold_candidates: bool = True
    pseudo_audit_samples: bool = True
    reservation_roles: tuple[str, ...] = (
        "eval_random",
        "eval_core_reserve",
        "dev",
    )


@dataclass
class SpotCheckPolicy:
    single_review_min_rate: float = 0.10
    expand_on_critical_error: bool = True


@dataclass
class ProtectedLayerPolicy:
    min_groups: int = 200
    upper_bound: float = 0.02
    # How to select the layer from candidate / human fields
    match: dict[str, Any] = field(default_factory=dict)


@dataclass
class PseudoAuditPolicy:
    min_groups_overall: int = 500
    wilson_z: float = 1.96
    overall_upper_bound: float = 0.01
    critical_semantic_errors_max: int = 0
    seed: int = 42
    # Calibration samples must never self-prove audit quality.
    forbid_calibration_self_proof: bool = True
    protected_layers: dict[str, ProtectedLayerPolicy] = field(default_factory=dict)


@dataclass
class BudgetPolicy:
    calibration_target: int = 3000
    calibration_min: int = 2000
    calibration_max: int = 5000
    p0_all_to_queue: bool = True
    p1_priority_tags: tuple[str, ...] = (
        "false_affirmation_candidate",
        "qwen_correction_candidate",
        "short_utterance",
        "all_empty_unverified",
        "noisy_audio",
        "crosstalk_suspected",
    )


@dataclass
class AnnotationConfig:
    annotation_version: str = ANNOTATION_VERSION
    dual_review: DualReviewPolicy = field(default_factory=DualReviewPolicy)
    spot_check: SpotCheckPolicy = field(default_factory=SpotCheckPolicy)
    pseudo_audit: PseudoAuditPolicy = field(default_factory=PseudoAuditPolicy)
    budget: BudgetPolicy = field(default_factory=BudgetPolicy)
    blind_seed_salt: str = "annotation_v3_blind"
    candidate_seed_salt: str = "annotation_v3_candidate"
    audio_event_tags_required_for_non_speech: bool = True

    @classmethod
    def from_params(cls, params: dict[str, Any] | None) -> AnnotationConfig:
        raw = dict(params or {})
        dual_raw = dict(raw.get("dual_review") or {})
        spot_raw = dict(raw.get("spot_check") or {})
        audit_raw = dict(raw.get("pseudo_audit") or {})
        budget_raw = dict(raw.get("budget") or {})

        layers: dict[str, ProtectedLayerPolicy] = {}
        for name, layer in dict(audit_raw.get("protected_layers") or {}).items():
            layer = dict(layer or {})
            layers[str(name)] = ProtectedLayerPolicy(
                min_groups=int(layer.get("min_groups", 200)),
                upper_bound=float(layer.get("upper_bound", 0.02)),
                match=dict(layer.get("match") or {}),
            )
        if not layers:
            layers = {
                "short_utterance": ProtectedLayerPolicy(
                    match={"risk_tags_any": ["short_utterance"]}
                ),
                "negation": ProtectedLayerPolicy(
                    match={"human_semantic": ["negative"], "risk_tags_any": ["negation_flip"]}
                ),
                "affirmation": ProtectedLayerPolicy(
                    match={
                        "human_semantic": ["positive"],
                        "risk_tags_any": ["false_affirmation_candidate"],
                    }
                ),
                "moderate_quality": ProtectedLayerPolicy(
                    match={"noise_band": ["moderate"], "human_noise": ["moderate"]}
                ),
            }

        return cls(
            annotation_version=str(raw.get("annotation_version") or ANNOTATION_VERSION),
            dual_review=DualReviewPolicy(
                eval_formal=bool(dual_raw.get("eval_formal", True)),
                priority_p0=bool(dual_raw.get("priority_p0", True)),
                empty_gold_candidates=bool(dual_raw.get("empty_gold_candidates", True)),
                pseudo_audit_samples=bool(dual_raw.get("pseudo_audit_samples", True)),
                reservation_roles=tuple(
                    str(x)
                    for x in (
                        dual_raw.get("reservation_roles")
                        or ["eval_random", "eval_core_reserve", "dev"]
                    )
                ),
            ),
            spot_check=SpotCheckPolicy(
                single_review_min_rate=float(spot_raw.get("single_review_min_rate", 0.10)),
                expand_on_critical_error=bool(spot_raw.get("expand_on_critical_error", True)),
            ),
            pseudo_audit=PseudoAuditPolicy(
                min_groups_overall=int(audit_raw.get("min_groups_overall", 500)),
                wilson_z=float(audit_raw.get("wilson_z", 1.96)),
                overall_upper_bound=float(audit_raw.get("overall_upper_bound", 0.01)),
                critical_semantic_errors_max=int(
                    audit_raw.get("critical_semantic_errors_max", 0)
                ),
                seed=int(audit_raw.get("seed", 42)),
                forbid_calibration_self_proof=bool(
                    audit_raw.get("forbid_calibration_self_proof", True)
                ),
                protected_layers=layers,
            ),
            budget=BudgetPolicy(
                calibration_target=int(budget_raw.get("calibration_target", 3000)),
                calibration_min=int(budget_raw.get("calibration_min", 2000)),
                calibration_max=int(budget_raw.get("calibration_max", 5000)),
                p0_all_to_queue=bool(budget_raw.get("p0_all_to_queue", True)),
                p1_priority_tags=tuple(
                    str(x)
                    for x in (
                        budget_raw.get("p1_priority_tags")
                        or [
                            "false_affirmation_candidate",
                            "qwen_correction_candidate",
                            "short_utterance",
                            "all_empty_unverified",
                            "noisy_audio",
                            "crosstalk_suspected",
                        ]
                    )
                ),
            ),
            blind_seed_salt=str(raw.get("blind_seed_salt") or "annotation_v3_blind"),
            candidate_seed_salt=str(
                raw.get("candidate_seed_salt") or "annotation_v3_candidate"
            ),
            audio_event_tags_required_for_non_speech=bool(
                raw.get("audio_event_tags_required_for_non_speech", True)
            ),
        )

    @classmethod
    def load(cls, path: str | Path) -> AnnotationConfig:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError(f"annotation config must be a mapping: {path}")
        return cls.from_params(data)


def default_annotation_config_path() -> Path:
    return Path("configs/annotation/zh_asr_v3.yaml")
