"""dataset_v3 — grouping, reservation (A), audit (C), sampling/release (D)."""

from audio_engine.core.dataset_v3.audit import (
    PseudoAuditReport,
    evaluate_pseudo_audit,
    mark_pseudo_audit_outcome,
    sample_group_balanced,
    wilson_interval,
)
from audio_engine.core.dataset_v3.grouping import (
    GroupingConfig,
    GroupingResult,
    apply_grouping_to_samples,
    build_leakage_groups,
)
from audio_engine.core.dataset_v3.release import (
    LeakageReport,
    ReleaseBuildError,
    ReleasePublishResult,
    publish_release_v3,
    validate_cross_split_leakage,
)
from audio_engine.core.dataset_v3.reservation import (
    ReservationArtifact,
    ReservationConfig,
    apply_reservation_to_samples,
    build_reservation,
)
from audio_engine.core.dataset_v3.sampling import (
    SamplingConfig,
    SamplingPlan,
    Shortfall,
    apply_sampling_plan_to_samples,
    build_sampling_plan,
)

__all__ = [
    "GroupingConfig",
    "GroupingResult",
    "LeakageReport",
    "PseudoAuditReport",
    "ReleaseBuildError",
    "ReleasePublishResult",
    "ReservationArtifact",
    "ReservationConfig",
    "SamplingConfig",
    "SamplingPlan",
    "Shortfall",
    "apply_grouping_to_samples",
    "apply_reservation_to_samples",
    "apply_sampling_plan_to_samples",
    "build_leakage_groups",
    "build_reservation",
    "build_sampling_plan",
    "evaluate_pseudo_audit",
    "mark_pseudo_audit_outcome",
    "publish_release_v3",
    "sample_group_balanced",
    "validate_cross_split_leakage",
    "wilson_interval",
]
