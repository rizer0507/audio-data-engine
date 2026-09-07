"""Constants and type names for selection_v2.0."""

from __future__ import annotations

RULE_VERSION = "selection_v2.0"

DECISION_AUTO_ACCEPT = "auto_accept"
DECISION_AUTO_EMPTY = "auto_empty"
DECISION_MODEL_REVIEW = "model_review"
DECISION_MANUAL_REVIEW = "manual_review"
DECISION_EXCLUDE = "exclude"

# v2 primary buckets
TYPE_INVALID_AUDIO = "invalid_audio"
TYPE_TRUE_SILENCE = "true_silence"
TYPE_POSSIBLE_VAD_MISS = "possible_vad_miss"
TYPE_OVERLAP_CROSSTALK = "overlap_crosstalk"
TYPE_DUPLICATE = "duplicate"
TYPE_VOICEMAIL = "voicemail"
TYPE_FAMILY_INTERNAL_CONFLICT = "family_internal_conflict"
TYPE_SEMANTIC_INVERSION = "semantic_inversion"
TYPE_CRITICAL_TOKEN_CONFLICT = "critical_token_conflict"
TYPE_SEMANTIC_SANITIZATION = "semantic_sanitization"
TYPE_SHORT_UTTERANCE_RISK = "short_utterance_risk"
TYPE_HALLUCINATION = "hallucination"
TYPE_MODEL_MISSING = "model_missing"
TYPE_PSEUDO_GOLD_HIGH = "pseudo_gold_high"
TYPE_PSEUDO_GOLD_MEDIUM = "pseudo_gold_medium"
TYPE_HARDCASE = "hardcase"

LABEL_SOURCE_MODEL = "model_consensus"
LABEL_SOURCE_HUMAN = "human"
LABEL_SOURCE_EXTERNAL = "external"
LABEL_SOURCE_NONE = "none"

LABEL_TIER_GOLD = "gold"
LABEL_TIER_PSEUDO_HIGH = "pseudo_high"
LABEL_TIER_PSEUDO_MEDIUM = "pseudo_medium"
LABEL_TIER_NONE = "none"

DATASET_ROLE_TRAIN = "train_candidate"
DATASET_ROLE_EVAL = "eval_candidate"
DATASET_ROLE_EXCLUDE = "exclude"

SEMANTIC_POSITIVE = "positive"
SEMANTIC_NEGATIVE = "negative"
SEMANTIC_NEUTRAL = "neutral"
SEMANTIC_UNKNOWN = "unknown"
SEMANTIC_NONE = "none"
SEMANTIC_CONFLICT = "conflict"

# v1 type aliases for export/review defaults
V1_COMPAT_TYPE_MAP: dict[str, str] = {
    TYPE_INVALID_AUDIO: "noise",
    TYPE_TRUE_SILENCE: "noise",
    TYPE_POSSIBLE_VAD_MISS: "noise",
    TYPE_PSEUDO_GOLD_HIGH: "auto_gold",
    TYPE_PSEUDO_GOLD_MEDIUM: "consensus_gold",
    TYPE_MODEL_MISSING: "qwen_missing",
}

PSEUDO_GOLD_TYPES = frozenset({TYPE_PSEUDO_GOLD_HIGH, TYPE_PSEUDO_GOLD_MEDIUM})
EMPTY_GOLD_TYPES = frozenset({TYPE_TRUE_SILENCE, TYPE_INVALID_AUDIO})
