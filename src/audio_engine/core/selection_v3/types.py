"""Constants and type names for selection_v3.0."""

from __future__ import annotations

RULE_VERSION = "selection_v3.0"
# Explicit opt-in. Production default stays RULE_VERSION until business calibration.
RULE_VERSION_SEMANTIC_TOLERANT = "selection_v3_semantic_tolerant_20260911"
# 024 opt-in. Does not replace 022 or the production default.
RULE_VERSION_BUSINESS_SEMANTIC = "selection_business_semantic_v4"
TOLERANCE_VERSION = "text_tolerance_v1"
# 025 opt-in text pre-layer. Production default stays legacy until an explicit switch.
CLASSIFY_TEXT_VERSION = "classify_text_zh_only_v1"
CLASSIFY_TEXT_POLICY_LEGACY = "legacy"
CLASSIFY_TEXT_POLICY_CHINESE_ONLY = "chinese_only_v1"
VERIFIER_VERSION_LOCAL = "local_semantic_verifier_v1"
ANNOTATION_VERSION = "annotation_v3.0"
DATASET_POLICY_VERSION = "dataset_policy_v3.0"
BUSINESS_METRICS_VERSION = "business_metrics_v1.0"
QUALITY_POLICY_VERSION = "quality_policy_v3.0"

# Per-run inference outcome — must not be confused with text content.
RUN_STATUS_SUCCESS_TEXT = "success_text"
RUN_STATUS_SUCCESS_EMPTY = "success_empty"
RUN_STATUS_FAILED = "failed"
RUN_STATUS_MISSING = "missing"

RUN_STATUSES = frozenset(
    {
        RUN_STATUS_SUCCESS_TEXT,
        RUN_STATUS_SUCCESS_EMPTY,
        RUN_STATUS_FAILED,
        RUN_STATUS_MISSING,
    }
)

# Sample-level readiness after configured multi-run contract check (N≥3 families × 2 runs).
SAMPLE_CLASSIFIABLE = "classifiable"
SAMPLE_INFERENCE_INCOMPLETE = "inference_incomplete"
SAMPLE_INVALID_AUDIO = "invalid_audio"

# Reservation / allocation roles (immutable once frozen).
RESERVATION_EVAL_RANDOM = "eval_random"
RESERVATION_EVAL_CORE_RESERVE = "eval_core_reserve"
RESERVATION_CALIBRATION = "calibration"
RESERVATION_DEV = "dev"
RESERVATION_TRAIN_POOL = "train_pool"
RESERVATION_GOVERNANCE_HOLD = "governance_hold"

RESERVATION_ROLES = frozenset(
    {
        RESERVATION_EVAL_RANDOM,
        RESERVATION_EVAL_CORE_RESERVE,
        RESERVATION_CALIBRATION,
        RESERVATION_DEV,
        RESERVATION_TRAIN_POOL,
        RESERVATION_GOVERNANCE_HOLD,
    }
)

# Governance flags
GOVERNANCE_MISSING_GROUP_META = "missing_group_metadata"
GOVERNANCE_NEAR_DUP_UNCERTAIN = "near_duplicate_uncertain"

DEFAULT_TEACHER_FAMILIES = ("kimi", "glm", "sensevoice")
DEFAULT_TARGET_FAMILY = "qwen"
DEFAULT_EXPECTED_RUNS_PER_FAMILY = 2
# consensus_v3: at least three families; four remains a recommended (not required) config.
MIN_MODEL_FAMILIES = 3

# Family dual-run status (exactly one per family, priority order).
FAMILY_INCOMPLETE = "incomplete"
FAMILY_STABLE_EMPTY = "stable_empty"
FAMILY_UNSTABLE_PRESENCE = "unstable_presence"
FAMILY_UNSTABLE_SEMANTIC = "unstable_semantic"
FAMILY_STABLE_TEXT = "stable_text"
FAMILY_UNSTABLE_TEXT = "unstable_text"

FAMILY_STATUSES = frozenset(
    {
        FAMILY_INCOMPLETE,
        FAMILY_STABLE_EMPTY,
        FAMILY_UNSTABLE_PRESENCE,
        FAMILY_UNSTABLE_SEMANTIC,
        FAMILY_STABLE_TEXT,
        FAMILY_UNSTABLE_TEXT,
    }
)

# Business categories (022). category is null when unclassified.
CATEGORY_GOLD = "gold"
CATEGORY_SEMANTIC_RISK = "semantic_risk"
CATEGORY_NOISE = "noise"
CATEGORY_HARDCASE = "hardcase"
CATEGORY_VOICEMAIL = "voicemail"
# 024 business-first classes. Not a silent rename of the 022 five-class schema.
CATEGORY_BUSINESS_CONSISTENT = "business_consistent"
CATEGORY_NON_SPEECH = "non_speech"

BUSINESS_CATEGORIES = frozenset(
    {
        CATEGORY_GOLD,
        CATEGORY_SEMANTIC_RISK,
        CATEGORY_NOISE,
        CATEGORY_HARDCASE,
        CATEGORY_VOICEMAIL,
    }
)
BUSINESS_CATEGORIES_V4 = frozenset(
    {
        CATEGORY_BUSINESS_CONSISTENT,
        CATEGORY_SEMANTIC_RISK,
        CATEGORY_NON_SPEECH,
        CATEGORY_HARDCASE,
        CATEGORY_VOICEMAIL,
    }
)

# Processing lifecycle. Classification never writes accepted.
STATUS_CANDIDATE = "candidate"
STATUS_MANUAL_REVIEW = "manual_review"
STATUS_RETRY = "retry"
STATUS_HOLD = "hold"
STATUS_ACCEPTED = "accepted"
STATUS_EXCLUDED = "excluded"
# 024 automatic business label. Never means a human accepted the row.
STATUS_AUTO_CLASSIFIED = "auto_classified"

COVERAGE_AUTO = "A"
COVERAGE_HUMAN = "H"
COVERAGE_UNRESOLVED = "U"

LABEL_TIER_HUMAN_VERBATIM = "human_verbatim"
LABEL_TIER_TRUSTED_EXTERNAL_VERBATIM = "trusted_external_verbatim"
LABEL_TIER_PSEUDO_VERBATIM_HIGH = "pseudo_verbatim_high"
LABEL_TIER_SEMANTIC_ONLY = "semantic_only"
LABEL_TIER_NO_TRANSCRIPT = "no_transcript"

# Decisions (stage B outputs candidates only — no auto train publish).
DECISION_EXCLUDE = "exclude"
DECISION_RETRY = "retry"
DECISION_MANUAL_REVIEW = "manual_review"
DECISION_AUDIT_PENDING = "audit_pending"
# hold = batch/calibration/governance staging; not a transcription review job.
DECISION_HOLD = "hold"

# Primary classification types (ordered decision chain).
TYPE_INVALID_AUDIO = "invalid_audio"
TYPE_INFERENCE_INCOMPLETE = "inference_incomplete"
TYPE_IMPLAUSIBLE_SPEECH_RATE = "implausible_speech_rate"
TYPE_ROUTE_QUARANTINE = "route_quarantine"
TYPE_SEMANTIC_RISK = "semantic_risk"
TYPE_CRITICAL_CONTENT_RISK = "critical_content_risk"
TYPE_ALL_EMPTY_UNVERIFIED = "all_empty_unverified"
TYPE_SPEECH_PRESENCE_DISAGREEMENT = "speech_presence_disagreement"
TYPE_VOICEMAIL_CANDIDATE = "voicemail_candidate"
TYPE_QWEN_CORRECTION_CANDIDATE = "qwen_correction_candidate"
TYPE_FAMILY_UNSTABLE = "family_unstable"
TYPE_AUDIO_QUALITY_RISK = "audio_quality_risk"
TYPE_QUALITY_UNCALIBRATED = "quality_uncalibrated"
TYPE_CONTENT_COMPLEXITY = "content_complexity"
TYPE_PSEUDO_HIGH = "pseudo_high"
TYPE_PSEUDO_MEDIUM = "pseudo_medium"
TYPE_HARDCASE = "hardcase"

# Risk tags (multi-select).
RISK_NEGATION_FLIP = "negation_flip"
RISK_FALSE_AFFIRMATION = "false_affirmation_candidate"
RISK_FILLER_AFFIRMATION = "filler_affirmation"
RISK_REJECTION_SANITIZATION = "rejection_sanitization"
RISK_CRITICAL_TOKEN_CONFLICT = "critical_token_conflict"
RISK_CONTENT_COMPLEXITY = "content_complexity"
RISK_PRESENCE_CONFLICT = "presence_conflict"
RISK_SHORT_UTTERANCE = "short_utterance"
RISK_FAMILY_INSTABILITY = "family_instability"
RISK_NOISY_AUDIO = "noisy_audio"
RISK_QUALITY_UNKNOWN = "quality_unknown"
RISK_CROSSTALK_SUSPECTED = "crosstalk_suspected"
RISK_CONSENSUS_AMBIGUOUS = "consensus_ambiguous"
RISK_IMPLAUSIBLE_SPEECH_RATE = "implausible_speech_rate"

# Quality four-state (020): do not collapse into a single audio_quality_risk job.
QUALITY_STATE_NOT_REQUIRED = "not_required"
QUALITY_STATE_UNCALIBRATED = "uncalibrated"
QUALITY_STATE_SCORED_OK = "scored_ok"
QUALITY_STATE_SCORED_NOISY = "scored_noisy"
QUALITY_STATE_FAILED = "failed"
QUALITY_STATE_UNSUPPORTED = "unsupported"

# 023. Historical full-batch gate stays behind an explicit rollback id.
NOISE_POLICY_ASR_ANOMALY = "asr_anomaly_noise_v1"
NOISE_POLICY_LEGACY = "legacy_full_quality_gate"
TRIGGER_ALL_FAMILIES_NO_VALID = "all_families_no_valid_transcript"
TRIGGER_NON_CHINESE = "non_chinese_transcript"

# Mutual disposition (exactly one per sample under 020).
DISPOSITION_MACHINE_RETRY = "machine_retry"
DISPOSITION_ROUTE_QUARANTINE = "route_quarantine"
DISPOSITION_CALIBRATION_HOLD = "calibration_hold"
DISPOSITION_GOVERNANCE_HOLD = "governance_hold"
DISPOSITION_AUDIO_EXCLUDE = "audio_exclude"
DISPOSITION_PRESENCE_CONFIRM = "presence_confirm"
DISPOSITION_VOICEMAIL_ISOLATION = "voicemail_isolation"
DISPOSITION_PSEUDO_PENDING_AUDIT = "pseudo_pending_audit"
DISPOSITION_TRAIN_ASSISTED = "train_assisted_correction"
DISPOSITION_HUMAN_BLIND = "human_blind_label"
DISPOSITION_CONTENT_COMPLEXITY = "content_complexity_sample"
DISPOSITION_BACKLOG = "backlog"

SEMANTIC_RISK_TAGS = frozenset(
    {
        RISK_NEGATION_FLIP,
        RISK_FALSE_AFFIRMATION,
        RISK_FILLER_AFFIRMATION,
        RISK_REJECTION_SANITIZATION,
    }
)

# Polarity
POLARITY_POSITIVE = "positive"
POLARITY_NEGATIVE = "negative"
POLARITY_NEUTRAL = "neutral"
POLARITY_MIXED = "mixed"
POLARITY_UNKNOWN = "unknown"

# Label provenance (candidate only in stage B)
LABEL_SOURCE_MODEL = "model_consensus"
LABEL_SOURCE_NONE = "none"
LABEL_TIER_PSEUDO_HIGH = "pseudo_high"
LABEL_TIER_PSEUDO_MEDIUM = "pseudo_medium"
LABEL_TIER_NONE = "none"

# Review priority
PRIORITY_P0 = "P0"
PRIORITY_P1 = "P1"
PRIORITY_P2 = "P2"

# DNSMOS / quality execution status. Calibration is a separate flag.
DNSMOS_STATUS_NOT_REQUIRED = "not_required"
DNSMOS_STATUS_PENDING = "pending"
DNSMOS_STATUS_SUCCESS = "success"
DNSMOS_STATUS_FAILED = "failed"
DNSMOS_STATUS_UNSUPPORTED = "unsupported"
NOISE_BAND_CLEAN = "clean"
NOISE_BAND_MODERATE = "moderate"
NOISE_BAND_NOISY = "noisy"
NOISE_BAND_UNKNOWN = "unknown"


def is_semantic_tolerant_rule(rule_version: str | None) -> bool:
    """True only for the explicit 022 rule id. Old ``selection_v3.0`` stays off."""
    text = str(rule_version or "").strip()
    return text == RULE_VERSION_SEMANTIC_TOLERANT or text.startswith(
        "selection_v3_semantic_tolerant_"
    )


def is_business_semantic_rule(rule_version: str | None) -> bool:
    """True only for the explicit 024 rule id. 022 and production v3 stay off."""
    text = str(rule_version or "").strip()
    return text == RULE_VERSION_BUSINESS_SEMANTIC or text.startswith(
        "selection_business_semantic_v4"
    )
