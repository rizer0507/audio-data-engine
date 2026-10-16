"""Neutral evidence constants (044 phase B).

These names are mirrored from the historical ``selection_v3.types`` module so that
the evidence layer does not import selection business modules. Values MUST stay
identical; ``tests/test_tagging_decoupling_044.py`` asserts parity.

Fact states
-----------
``success`` / ``failed`` / ``missing`` / ``not_computed`` describe a *fact* about
computed evidence. ``not_required`` is a **consumer scheduling decision** (some
consumer decided it does not need the evidence this run). It is not a shared fact
state and must never overwrite an existing ``success`` fact.
"""

from __future__ import annotations

# --- Fact states (shared evidence) -------------------------------------------------
FACT_SUCCESS = "success"
FACT_FAILED = "failed"
FACT_MISSING = "missing"
FACT_NOT_COMPUTED = "not_computed"
FACT_STATES = frozenset({FACT_SUCCESS, FACT_FAILED, FACT_MISSING, FACT_NOT_COMPUTED})

# Schedule-only marker. Written by consumers (old queues), never a fact.
SCHEDULE_NOT_REQUIRED = "not_required"
SCHEDULE_ONLY_STATES = frozenset({SCHEDULE_NOT_REQUIRED})

ACOUSTIC_STATUS_NOT_REQUIRED = SCHEDULE_NOT_REQUIRED
ACOUSTIC_STATUS_PENDING = "pending"

ACOUSTIC_DEPS = ("vad", "energy", "dnsmos")


def normalize_fact_status(raw: object) -> str:
    """Map a stored status string to a fact state.

    ``not_required`` / ``pending`` / empty are read as *no valid evidence provided*
    (``not_computed``); they are never reinterpreted as a computed result.
    """
    text = str(raw or "").strip().lower()
    if text == FACT_SUCCESS:
        return FACT_SUCCESS
    if text in {FACT_FAILED, "error", "fail"}:
        return FACT_FAILED
    if text == FACT_MISSING:
        return FACT_MISSING
    # not_required / pending / not_computed / unknown strings
    return FACT_NOT_COMPUTED


# --- Run status (per ASR route) ----------------------------------------------------
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

# --- Route disposition -------------------------------------------------------------
ROUTE_ELIGIBLE = "eligible"
ROUTE_EXCLUDED = "excluded"
ROUTE_FAILED = "failed"
ROUTE_MISSING = "missing"
ROUTE_DISPOSITIONS = frozenset(
    {ROUTE_ELIGIBLE, ROUTE_EXCLUDED, ROUTE_FAILED, ROUTE_MISSING}
)

EXCLUSION_FOREIGN = "foreign_transcript"
EXCLUSION_HOTWORD_ECHO = "hotword_echo"
EXCLUSION_PROMPT_ECHO = "prompt_echo"
EXCLUSION_REASONS = frozenset(
    {EXCLUSION_FOREIGN, EXCLUSION_HOTWORD_ECHO, EXCLUSION_PROMPT_ECHO}
)

# --- Family dual-run states --------------------------------------------------------
FAMILY_STATE_STABLE_TEXT = "stable_text"
FAMILY_STATE_STABLE_EMPTY = "stable_empty"
FAMILY_STATE_UNSTABLE = "unstable"
FAMILY_STATE_UNAVAILABLE = "unavailable"
FAMILY_STATES_V2 = frozenset(
    {
        FAMILY_STATE_STABLE_TEXT,
        FAMILY_STATE_STABLE_EMPTY,
        FAMILY_STATE_UNSTABLE,
        FAMILY_STATE_UNAVAILABLE,
    }
)

# --- Text policy / versions --------------------------------------------------------
TOLERANCE_VERSION = "text_tolerance_v1"
CLASSIFY_TEXT_VERSION = "classify_text_zh_only_v1"
CLASSIFY_TEXT_VERSION_FIVE_CLASS = "classify_text_five_class_v1"
CLASSIFY_TEXT_POLICY_LEGACY = "legacy"
CLASSIFY_TEXT_POLICY_CHINESE_ONLY = "chinese_only_v1"
CLASSIFY_TEXT_POLICY_FIVE_CLASS = "five_class_v1"

# --- Polarity / conflict tags (text-level facts, no business routing) --------------
POLARITY_POSITIVE = "positive"
POLARITY_NEGATIVE = "negative"
POLARITY_NEUTRAL = "neutral"
POLARITY_MIXED = "mixed"
POLARITY_UNKNOWN = "unknown"

RISK_NEGATION_FLIP = "negation_flip"
RISK_FALSE_AFFIRMATION = "false_affirmation_candidate"
RISK_FILLER_AFFIRMATION = "filler_affirmation"
RISK_REJECTION_SANITIZATION = "rejection_sanitization"
RISK_CRITICAL_TOKEN_CONFLICT = "critical_token_conflict"
RISK_CONTENT_COMPLEXITY = "content_complexity"

SUBTYPE_SEMANTIC_REVERSAL = "semantic_reversal"
SUBTYPE_HALLUCINATED_ASSERTION = "hallucinated_assertion"
SUBTYPE_SHORT_POLARITY_AMBIGUITY = "short_polarity_ambiguity"

RISK_TAG_POLARITY_CONFLICT = "polarity_conflict"
RISK_TAG_CRITICAL_POLARITY_RESPONSE = "critical_polarity_response"

# --- Acoustic scoring statuses -----------------------------------------------------
DNSMOS_STATUS_SUCCESS = "success"
DNSMOS_STATUS_FAILED = "failed"
DNSMOS_STATUS_UNSUPPORTED = "unsupported"

QUALITY_POLICY_VERSION = "quality_policy_v3.0"
NOISE_BAND_CLEAN = "clean"
NOISE_BAND_MODERATE = "moderate"
NOISE_BAND_NOISY = "noisy"
NOISE_BAND_UNKNOWN = "unknown"

ENERGY_STATE_TOO_SHORT = "too_short"
ENERGY_STATE_INAUDIBLE = "inaudible"
ENERGY_STATE_AUDIBLE = "audible"
ENERGY_STATE_BORDERLINE = "borderline"
ENERGY_STATE_FAILED = "failed"
