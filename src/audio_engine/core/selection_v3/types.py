"""Constants and type names for selection_v3.0."""

from __future__ import annotations

RULE_VERSION = "selection_v3.0"
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

# Decisions (stage B outputs candidates only — no auto train publish).
DECISION_EXCLUDE = "exclude"
DECISION_RETRY = "retry"
DECISION_MANUAL_REVIEW = "manual_review"
DECISION_AUDIT_PENDING = "audit_pending"

# Primary classification types (ordered decision chain).
TYPE_INVALID_AUDIO = "invalid_audio"
TYPE_INFERENCE_INCOMPLETE = "inference_incomplete"
TYPE_SEMANTIC_RISK = "semantic_risk"
TYPE_CRITICAL_CONTENT_RISK = "critical_content_risk"
TYPE_ALL_EMPTY_UNVERIFIED = "all_empty_unverified"
TYPE_SPEECH_PRESENCE_DISAGREEMENT = "speech_presence_disagreement"
TYPE_VOICEMAIL_CANDIDATE = "voicemail_candidate"
TYPE_QWEN_CORRECTION_CANDIDATE = "qwen_correction_candidate"
TYPE_FAMILY_UNSTABLE = "family_unstable"
TYPE_AUDIO_QUALITY_RISK = "audio_quality_risk"
TYPE_PSEUDO_HIGH = "pseudo_high"
TYPE_PSEUDO_MEDIUM = "pseudo_medium"
TYPE_HARDCASE = "hardcase"

# Risk tags (multi-select).
RISK_NEGATION_FLIP = "negation_flip"
RISK_FALSE_AFFIRMATION = "false_affirmation_candidate"
RISK_FILLER_AFFIRMATION = "filler_affirmation"
RISK_REJECTION_SANITIZATION = "rejection_sanitization"
RISK_CRITICAL_TOKEN_CONFLICT = "critical_token_conflict"
RISK_PRESENCE_CONFLICT = "presence_conflict"
RISK_SHORT_UTTERANCE = "short_utterance"
RISK_FAMILY_INSTABILITY = "family_instability"
RISK_NOISY_AUDIO = "noisy_audio"
RISK_QUALITY_UNKNOWN = "quality_unknown"
RISK_CROSSTALK_SUSPECTED = "crosstalk_suspected"
RISK_CONSENSUS_AMBIGUOUS = "consensus_ambiguous"

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

# DNSMOS / quality
DNSMOS_STATUS_SUCCESS = "success"
DNSMOS_STATUS_FAILED = "failed"
DNSMOS_STATUS_UNSUPPORTED = "unsupported"
NOISE_BAND_CLEAN = "clean"
NOISE_BAND_MODERATE = "moderate"
NOISE_BAND_NOISY = "noisy"
NOISE_BAND_UNKNOWN = "unknown"
