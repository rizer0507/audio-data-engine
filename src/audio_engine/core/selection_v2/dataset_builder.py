"""Dataset role assignment for selection_v2.0 (lightweight Phase A)."""

from __future__ import annotations

from audio_engine.core.selection_v2.types import (
    DATASET_ROLE_EVAL,
    DATASET_ROLE_EXCLUDE,
    DATASET_ROLE_TRAIN,
    DECISION_AUTO_ACCEPT,
    DECISION_EXCLUDE,
    TYPE_HARDCASE,
    TYPE_INVALID_AUDIO,
    TYPE_PSEUDO_GOLD_HIGH,
    TYPE_PSEUDO_GOLD_MEDIUM,
    TYPE_TRUE_SILENCE,
    TYPE_VOICEMAIL,
)


def assign_dataset_role(type_: str, decision: str) -> str:
    if decision == DECISION_EXCLUDE or type_ in {TYPE_INVALID_AUDIO}:
        return DATASET_ROLE_EXCLUDE
    if type_ == TYPE_VOICEMAIL:
        return DATASET_ROLE_EXCLUDE
    if type_ == TYPE_PSEUDO_GOLD_HIGH and decision == DECISION_AUTO_ACCEPT:
        return DATASET_ROLE_TRAIN
    if type_ == TYPE_PSEUDO_GOLD_MEDIUM:
        return DATASET_ROLE_TRAIN
    if type_ in {TYPE_HARDCASE, TYPE_TRUE_SILENCE}:
        # hardcase = high-value pool after human fix; silence optional for train
        return DATASET_ROLE_EVAL if type_ == TYPE_HARDCASE else DATASET_ROLE_EXCLUDE
    # Risk / review buckets: prefer eval_candidate after human gold upgrade
    return DATASET_ROLE_EVAL
