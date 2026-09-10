"""annotation_v3.0 constants and enums."""

from __future__ import annotations

ANNOTATION_VERSION = "annotation_v3.0"

# Workflow states (012-C §3.1)
STATE_PENDING = "pending"
STATE_ANNOTATED = "annotated"
STATE_SECOND_REVIEW = "second_review"
STATE_CONFLICT = "conflict"
STATE_ADJUDICATED = "adjudicated"
STATE_REJECTED = "rejected"

ANNOTATION_STATES = frozenset(
    {
        STATE_PENDING,
        STATE_ANNOTATED,
        STATE_SECOND_REVIEW,
        STATE_CONFLICT,
        STATE_ADJUDICATED,
        STATE_REJECTED,
    }
)

# Terminal states that may carry formal human gold (when other gates pass).
GOLD_ELIGIBLE_STATES = frozenset({STATE_SECOND_REVIEW, STATE_ADJUDICATED})

# Single-review path may also become gold after first pass when dual is not required.
GOLD_ELIGIBLE_SINGLE = frozenset({STATE_ANNOTATED})

GOLD_KIND_SPEECH = "speech"
GOLD_KIND_NON_SPEECH = "non_speech"
GOLD_KIND_UNINTELLIGIBLE = "unintelligible"
GOLD_KIND_AMBIGUOUS_TARGET = "ambiguous_target"
GOLD_KIND_INVALID = "invalid"

GOLD_KINDS = frozenset(
    {
        GOLD_KIND_SPEECH,
        GOLD_KIND_NON_SPEECH,
        GOLD_KIND_UNINTELLIGIBLE,
        GOLD_KIND_AMBIGUOUS_TARGET,
        GOLD_KIND_INVALID,
    }
)

# Kinds that must never pass formal train/eval gold paths.
NON_FORMAL_GOLD_KINDS = frozenset(
    {
        GOLD_KIND_UNINTELLIGIBLE,
        GOLD_KIND_AMBIGUOUS_TARGET,
        GOLD_KIND_INVALID,
    }
)

SPEECH_SCOPE_TARGET = "target"
SPEECH_SCOPE_BACKGROUND_ONLY = "background_only"
SPEECH_SCOPE_MIXED = "mixed"
SPEECH_SCOPE_NONE = "none"
SPEECH_SCOPE_UNKNOWN = "unknown"

SPEECH_SCOPES = frozenset(
    {
        SPEECH_SCOPE_TARGET,
        SPEECH_SCOPE_BACKGROUND_ONLY,
        SPEECH_SCOPE_MIXED,
        SPEECH_SCOPE_NONE,
        SPEECH_SCOPE_UNKNOWN,
    }
)

HUMAN_SEMANTIC_VALUES = frozenset(
    {"positive", "negative", "neutral", "mixed", "unknown", "not_applicable"}
)
HUMAN_NOISE_VALUES = frozenset({"clean", "moderate", "noisy", "unknown"})
HUMAN_CROSSTALK_VALUES = frozenset({"true", "false", "unknown"})

# Export / import pass roles
PASS_FIRST = "first"
PASS_SECOND = "second"
PASS_ADJUDICATION = "adjudication"
PASS_SPOT_CHECK = "spot_check"

PASSES = frozenset({PASS_FIRST, PASS_SECOND, PASS_ADJUDICATION, PASS_SPOT_CHECK})

# Package views
VIEW_BLIND = "blind"
VIEW_CANDIDATE_CHECK = "candidate_check"
VIEW_SECOND = "second_review"
VIEW_ADJUDICATION = "adjudication"
VIEW_SPOT_CHECK = "spot_check"

VIEWS = frozenset(
    {VIEW_BLIND, VIEW_CANDIDATE_CHECK, VIEW_SECOND, VIEW_ADJUDICATION, VIEW_SPOT_CHECK}
)

# Sentinel in XLSX for confirmed empty string (distinct from blank=null).
EMPTY_GOLD_SENTINEL = "__EMPTY__"
NULL_GOLD_SENTINEL = "__NULL__"

LABEL_SOURCE_HUMAN = "human"
LABEL_SOURCE_TRUSTED_EXTERNAL = "trusted_external"
LABEL_TIER_GOLD = "gold"

# Immutable identity / provenance columns (reject edits on import).
IMMUTABLE_EXPORT_COLUMNS = frozenset(
    {
        "sample_id",
        "original_audio_sha256",
        "queue_id",
        "queue_revision",
        "source_path",
        "type",
        "risk_tags",
        "review_priority",
        "review_queue",
        "candidate_text",
        "requires_dual_review",
        "leakage_group_id",
        "reservation_role",
    }
)
