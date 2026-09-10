"""Annotation field contract: null vs empty gold, gold_kind rules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from audio_engine.core.annotation_v3.types import (
    EMPTY_GOLD_SENTINEL,
    GOLD_KIND_NON_SPEECH,
    GOLD_KIND_SPEECH,
    GOLD_KINDS,
    HUMAN_CROSSTALK_VALUES,
    HUMAN_NOISE_VALUES,
    HUMAN_SEMANTIC_VALUES,
    NON_FORMAL_GOLD_KINDS,
    NULL_GOLD_SENTINEL,
    SPEECH_SCOPES,
)


@dataclass
class GoldTextValue:
    """Distinguish Python None (incomplete) from confirmed empty string."""

    value: str | None

    @property
    def is_null(self) -> bool:
        return self.value is None

    @property
    def is_confirmed_empty(self) -> bool:
        return self.value == ""

    @property
    def is_nonempty(self) -> bool:
        return self.value is not None and self.value != ""


def encode_gold_text_for_tabular(value: str | None) -> str:
    """Encode null/empty for XLSX cells (blank alone cannot round-trip)."""
    if value is None:
        return NULL_GOLD_SENTINEL
    if value == "":
        return EMPTY_GOLD_SENTINEL
    return value


def decode_gold_text_from_tabular(raw: Any) -> GoldTextValue:
    """Decode tabular / JSON cell into GoldTextValue.

    - ``None`` / NaN / ``__NULL__`` → incomplete null
    - ``__EMPTY__`` / ``""`` → confirmed empty string (non_speech)
    - other text → as-is

    XLSX blank cells should be loaded as None (see ``load_review_rows``), not as
    empty strings, so JSONL ``""`` can round-trip as confirmed empty.
    """
    if raw is None:
        return GoldTextValue(None)
    if isinstance(raw, float) and raw != raw:  # NaN
        return GoldTextValue(None)
    text = str(raw)
    if text == NULL_GOLD_SENTINEL:
        return GoldTextValue(None)
    if text == EMPTY_GOLD_SENTINEL or text == "":
        return GoldTextValue("")
    return GoldTextValue(text)


def parse_boolish_crosstalk(raw: Any) -> str | None:
    if raw is None:
        return None
    text = str(raw).strip().lower()
    if text in {"", "null", "none", NULL_GOLD_SENTINEL}:
        return None
    if text in {"true", "1", "yes"}:
        return "true"
    if text in {"false", "0", "no"}:
        return "false"
    if text == "unknown":
        return "unknown"
    return text


@dataclass
class AnnotationDraft:
    """One annotator submission for a sample."""

    sample_id: str
    gold_kind: str | None = None
    gold_text: str | None = None
    speech_scope: str | None = None
    human_semantic: str | None = None
    human_noise: str | None = None
    human_crosstalk: str | None = None
    verified_error_tags: list[str] = field(default_factory=list)
    audio_event_tags: list[str] = field(default_factory=list)
    reason: str = ""
    decision: str = ""  # accept | reject | "" (incomplete)
    annotator_id: str = ""
    timestamp: str = ""

    @property
    def is_complete(self) -> bool:
        """Completed means decision filled and gold fields consistent (or reject)."""
        decision = (self.decision or "").strip().lower()
        if decision in {"reject", "rejected"}:
            return True
        if decision not in {"accept", "accepted", "annotate"}:
            return False
        if not self.gold_kind:
            return False
        if self.gold_kind == GOLD_KIND_SPEECH and not (self.gold_text or "").strip():
            return False
        if self.gold_kind == GOLD_KIND_NON_SPEECH and self.gold_text is None:
            return False
        if self.gold_kind == GOLD_KIND_NON_SPEECH and self.gold_text not in {"", None}:
            # non_speech requires confirmed empty string, not free text
            if self.gold_text != "":
                return False
        return True


@dataclass
class ContractViolation:
    sample_id: str
    code: str
    message: str


def validate_annotation_draft(
    draft: AnnotationDraft,
    *,
    require_audio_event_for_non_speech: bool = True,
) -> list[ContractViolation]:
    """Validate a completed draft against annotation_v3 field contract."""
    errors: list[ContractViolation] = []
    sid = draft.sample_id
    decision = (draft.decision or "").strip().lower()
    if decision in {"", "pending"}:
        return errors  # incomplete rows are allowed on partial import
    if decision in {"reject", "rejected"}:
        return errors
    if decision not in {"accept", "accepted", "annotate"}:
        errors.append(
            ContractViolation(sid, "invalid_decision", f"invalid decision {draft.decision!r}")
        )
        return errors

    kind = (draft.gold_kind or "").strip()
    if kind not in GOLD_KINDS:
        errors.append(
            ContractViolation(sid, "invalid_gold_kind", f"gold_kind must be one of {sorted(GOLD_KINDS)}")
        )
        return errors

    if kind in {GOLD_KIND_SPEECH, GOLD_KIND_NON_SPEECH}:
        for name in ("speech_scope", "human_semantic", "human_noise", "human_crosstalk"):
            if not getattr(draft, name):
                errors.append(ContractViolation(sid, "missing_human_field", f"{name} is required for completed gold"))
        if kind == GOLD_KIND_SPEECH and draft.speech_scope in {"none", "background_only", "mixed", "unknown"}:
            errors.append(ContractViolation(sid, "invalid_target_scope", "speech gold requires target scope; unresolved speaker must be ambiguous_target"))
        if kind == GOLD_KIND_NON_SPEECH and draft.speech_scope not in {None, "none"}:
            errors.append(ContractViolation(sid, "invalid_target_scope", "non_speech gold requires none scope"))
        if kind == GOLD_KIND_NON_SPEECH and draft.human_semantic not in {None, "not_applicable"}:
            errors.append(ContractViolation(sid, "invalid_gold_semantic", "non_speech semantic must be not_applicable"))
        if kind == GOLD_KIND_SPEECH and draft.human_semantic == "not_applicable":
            errors.append(ContractViolation(sid, "invalid_gold_semantic", "speech semantic cannot be not_applicable"))

    if kind == GOLD_KIND_SPEECH:
        if draft.gold_text is None:
            errors.append(
                ContractViolation(
                    sid, "speech_null_gold", "speech requires non-empty gold_text, not null"
                )
            )
        elif draft.gold_text == "":
            errors.append(
                ContractViolation(
                    sid,
                    "speech_empty_gold",
                    "speech requires non-empty verbatim transcript; use non_speech for confirmed empty",
                )
            )
        elif not str(draft.gold_text).strip():
            errors.append(
                ContractViolation(
                    sid, "speech_whitespace_only", "speech gold_text must be non-empty after strip"
                )
            )

    if kind == GOLD_KIND_NON_SPEECH:
        if draft.gold_text is None:
            errors.append(
                ContractViolation(
                    sid,
                    "non_speech_null",
                    "non_speech requires gold_text=\"\" (confirmed empty), not null",
                )
            )
        elif draft.gold_text != "":
            errors.append(
                ContractViolation(
                    sid,
                    "non_speech_nonempty",
                    "non_speech requires gold_text=\"\"; do not write noise words as target text",
                )
            )
        if require_audio_event_for_non_speech and not draft.audio_event_tags:
            errors.append(
                ContractViolation(
                    sid,
                    "non_speech_missing_event",
                    "non_speech requires explicit audio_event_tags (silence/env/music/busy…)",
                )
            )

    if kind in NON_FORMAL_GOLD_KINDS:
        # Allowed as annotated outcomes, but must not be treated as formal empty gold.
        if draft.gold_text == "":
            errors.append(
                ContractViolation(
                    sid,
                    "unintelligible_as_empty",
                    f"{kind} must not be marked as confirmed empty gold_text",
                )
            )

    if draft.speech_scope and draft.speech_scope not in SPEECH_SCOPES:
        errors.append(
            ContractViolation(
                sid, "invalid_speech_scope", f"speech_scope invalid: {draft.speech_scope!r}"
            )
        )
    if draft.human_semantic and draft.human_semantic not in HUMAN_SEMANTIC_VALUES:
        errors.append(
            ContractViolation(
                sid, "invalid_human_semantic", f"human_semantic invalid: {draft.human_semantic!r}"
            )
        )
    if draft.human_noise and draft.human_noise not in HUMAN_NOISE_VALUES:
        errors.append(
            ContractViolation(
                sid, "invalid_human_noise", f"human_noise invalid: {draft.human_noise!r}"
            )
        )
    if draft.human_crosstalk and draft.human_crosstalk not in HUMAN_CROSSTALK_VALUES:
        errors.append(
            ContractViolation(
                sid,
                "invalid_human_crosstalk",
                f"human_crosstalk invalid: {draft.human_crosstalk!r}",
            )
        )
    return errors


def drafts_conflict(a: AnnotationDraft, b: AnnotationDraft) -> bool:
    """True when two independent reviews disagree on material fields."""
    if (a.decision or "").lower().startswith("reject") != (b.decision or "").lower().startswith(
        "reject"
    ):
        return True
    if (a.gold_kind or "") != (b.gold_kind or ""):
        return True
    if (a.gold_text if a.gold_text is not None else None) != (
        b.gold_text if b.gold_text is not None else None
    ):
        return True
    if (a.speech_scope or "") != (b.speech_scope or ""):
        return True
    if (a.human_semantic or "") != (b.human_semantic or ""):
        return True
    if (a.human_noise, a.human_crosstalk) != (b.human_noise, b.human_crosstalk):
        return True
    if set(a.audio_event_tags) != set(b.audio_event_tags):
        return True
    if set(a.verified_error_tags) != set(b.verified_error_tags):
        return True
    return False


def may_pass_formal_gold(gold_kind: str | None, gold_text: str | None, state: str) -> bool:
    """Whether a sample may enter formal train/eval gold paths."""
    from audio_engine.core.annotation_v3.types import (
        GOLD_ELIGIBLE_SINGLE,
        GOLD_ELIGIBLE_STATES,
        STATE_ANNOTATED,
    )

    if gold_kind not in {GOLD_KIND_SPEECH, GOLD_KIND_NON_SPEECH}:
        return False
    if gold_text is None:
        return False
    if gold_kind == GOLD_KIND_SPEECH and not str(gold_text).strip():
        return False
    if gold_kind == GOLD_KIND_NON_SPEECH and gold_text != "":
        return False
    if state in GOLD_ELIGIBLE_STATES:
        return True
    if state in GOLD_ELIGIBLE_SINGLE or state == STATE_ANNOTATED:
        # Caller must also confirm dual was not required.
        return gold_kind in {GOLD_KIND_SPEECH, GOLD_KIND_NON_SPEECH}
    return False
