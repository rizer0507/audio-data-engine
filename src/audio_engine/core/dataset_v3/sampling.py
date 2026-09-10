"""Deterministic quota sampling for dataset_policy_v3 (stage D).

Consumes immutable reservation + reviewed annotations; never randomly remaps
accepted gold into train/dev/test. Shortfalls fail formal builds when configured.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Sequence

from pydantic import BaseModel, ConfigDict

from audio_engine.core.annotation_v3.contract import may_pass_formal_gold
from audio_engine.core.annotation_v3.types import (
    GOLD_KIND_NON_SPEECH,
    GOLD_KIND_SPEECH,
    LABEL_SOURCE_HUMAN,
    LABEL_SOURCE_TRUSTED_EXTERNAL,
    NON_FORMAL_GOLD_KINDS,
    STATE_ADJUDICATED,
    STATE_SECOND_REVIEW,
)
from audio_engine.core.dataset_v3.reservation import ReservationArtifact
from audio_engine.core.sample import Sample
from audio_engine.core.selection_v3.text import comparison_text
from audio_engine.core.selection_v3.types import (
    DATASET_POLICY_VERSION,
    RESERVATION_CALIBRATION,
    RESERVATION_DEV,
    RESERVATION_EVAL_CORE_RESERVE,
    RESERVATION_EVAL_RANDOM,
    RESERVATION_GOVERNANCE_HOLD,
    RESERVATION_TRAIN_POOL,
    RISK_CROSSTALK_SUSPECTED,
    RISK_NOISY_AUDIO,
    RISK_SHORT_UTTERANCE,
    TYPE_PSEUDO_HIGH,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class EvalCoreStrataConfig:
    negative_reject: int = 500
    filler_neutral: int = 300
    confirmed_non_speech: int = 300
    noisy_crosstalk_speech: int = 300
    positive: int = 300
    other: int = 300

    def as_ordered(self) -> list[tuple[str, int]]:
        return [
            ("negative_reject", self.negative_reject),
            ("filler_neutral", self.filler_neutral),
            ("confirmed_non_speech", self.confirmed_non_speech),
            ("noisy_crosstalk_speech", self.noisy_crosstalk_speech),
            ("positive", self.positive),
            ("other", self.other),
        ]

    @property
    def total(self) -> int:
        return sum(v for _, v in self.as_ordered())

    @classmethod
    def from_params(cls, params: dict[str, Any] | None) -> EvalCoreStrataConfig:
        raw = params or {}
        return cls(
            negative_reject=int(raw.get("negative_reject", 500)),
            filler_neutral=int(raw.get("filler_neutral", 300)),
            confirmed_non_speech=int(raw.get("confirmed_non_speech", 300)),
            noisy_crosstalk_speech=int(raw.get("noisy_crosstalk_speech", 300)),
            positive=int(raw.get("positive", 300)),
            other=int(raw.get("other", 300)),
        )


@dataclass
class TrainPoolRatios:
    qwen_error_fix: float = 0.40
    human_high_risk: float = 0.20
    human_ordinary: float = 0.20
    pseudo_high_audited: float = 0.20

    def as_ordered(self) -> list[tuple[str, float]]:
        return [
            ("qwen_error_fix", self.qwen_error_fix),
            ("human_high_risk", self.human_high_risk),
            ("human_ordinary", self.human_ordinary),
            ("pseudo_high_audited", self.pseudo_high_audited),
        ]

    @classmethod
    def from_params(cls, params: dict[str, Any] | None) -> TrainPoolRatios:
        raw = params or {}
        return cls(
            qwen_error_fix=float(raw.get("qwen_error_fix", 0.40)),
            human_high_risk=float(raw.get("human_high_risk", 0.20)),
            human_ordinary=float(raw.get("human_ordinary", 0.20)),
            pseudo_high_audited=float(raw.get("pseudo_high_audited", 0.20)),
        )


@dataclass
class TrainCrossConstraints:
    min_positive_ratio: float = 0.15
    min_negative_ratio: float = 0.15
    min_noisy_crosstalk_ratio: float = 0.10
    max_non_speech_ratio: float = 0.10
    max_pseudo_high_ratio: float = 0.20

    @classmethod
    def from_params(cls, params: dict[str, Any] | None) -> TrainCrossConstraints:
        raw = params or {}
        return cls(
            min_positive_ratio=float(raw.get("min_positive_ratio", 0.15)),
            min_negative_ratio=float(raw.get("min_negative_ratio", 0.15)),
            min_noisy_crosstalk_ratio=float(raw.get("min_noisy_crosstalk_ratio", 0.10)),
            max_non_speech_ratio=float(raw.get("max_non_speech_ratio", 0.10)),
            max_pseudo_high_ratio=float(raw.get("max_pseudo_high_ratio", 0.20)),
        )


@dataclass
class SamplingConfig:
    """Build-time sampling / gate policy (dataset_policy_v3 stage D)."""

    release_id: str = ""
    train_size: int = 10000
    sampling_seed: int = 42
    policy_version: str = DATASET_POLICY_VERSION
    normalization_version: str = "zh_v1"
    gold_revision: str = "annotation_v3.0"
    group_key: str = "leakage_group_id"
    allow_non_speech_train: bool = False
    max_per_call: int = 3
    exact_duplicate_fields: list[str] = field(
        default_factory=lambda: [
            "duplicate_group_id",
            "original_audio_sha256",
            "pcm_sha256",
        ]
    )
    qwen_run_keys: list[str] = field(default_factory=lambda: ["qwen_1", "qwen_2"])
    eval_random_target: int = 2000
    eval_core: EvalCoreStrataConfig = field(default_factory=EvalCoreStrataConfig)
    train_pools: TrainPoolRatios = field(default_factory=TrainPoolRatios)
    train_cross: TrainCrossConstraints = field(default_factory=TrainCrossConstraints)
    fail_on_shortfall: bool = True
    require_dual_review_for_eval: bool = True
    require_pseudo_audit_for_pseudo_train: bool = True
    require_reservation: bool = True

    @classmethod
    def from_params(cls, params: dict[str, Any] | None) -> SamplingConfig:
        raw = dict(params or {})
        build = dict(raw.get("build") or {})
        # Flat overrides win over nested build block.
        for key in (
            "release_id",
            "train_size",
            "sampling_seed",
            "normalization_version",
            "gold_revision",
            "group_key",
            "allow_non_speech_train",
            "max_per_call",
            "fail_on_shortfall",
            "require_dual_review_for_eval",
            "require_pseudo_audit_for_pseudo_train",
            "require_reservation",
        ):
            if key in raw:
                build[key] = raw[key]
        eval_random = dict(raw.get("eval_random") or {})
        return cls(
            release_id=str(build.get("release_id") or ""),
            train_size=int(build.get("train_size", 10000)),
            sampling_seed=int(
                build.get("sampling_seed", (raw.get("reservation") or {}).get("seed", 42))
            ),
            policy_version=str(
                raw.get("dataset_policy_version")
                or build.get("policy_version")
                or DATASET_POLICY_VERSION
            ),
            normalization_version=str(build.get("normalization_version") or "zh_v1"),
            gold_revision=str(build.get("gold_revision") or "annotation_v3.0"),
            group_key=str(build.get("group_key") or "leakage_group_id"),
            allow_non_speech_train=bool(build.get("allow_non_speech_train", False)),
            max_per_call=int(build.get("max_per_call", 3)),
            exact_duplicate_fields=[
                str(x)
                for x in (
                    build.get("exact_duplicate_fields")
                    or [
                        "duplicate_group_id",
                        "original_audio_sha256",
                        "pcm_sha256",
                    ]
                )
            ],
            qwen_run_keys=[
                str(x)
                for x in (
                    build.get("qwen_run_keys")
                    or (raw.get("model_families") or {}).get("qwen")
                    or ["qwen_1", "qwen_2"]
                )
            ],
            eval_random_target=int(
                eval_random.get("target", (raw.get("reservation") or {}).get("eval_random_target", 2000))
            ),
            eval_core=EvalCoreStrataConfig.from_params(
                (raw.get("eval_core") or {}).get("strata") or raw.get("eval_core")
            ),
            train_pools=TrainPoolRatios.from_params(raw.get("train_pools")),
            train_cross=TrainCrossConstraints.from_params(
                raw.get("train_cross_constraints") or raw.get("train_cross")
            ),
            fail_on_shortfall=bool(build.get("fail_on_shortfall", True)),
            require_dual_review_for_eval=bool(
                build.get("require_dual_review_for_eval", True)
            ),
            require_pseudo_audit_for_pseudo_train=bool(
                build.get("require_pseudo_audit_for_pseudo_train", True)
            ),
            require_reservation=bool(build.get("require_reservation", True)),
        )


# ---------------------------------------------------------------------------
# Allocation contract (Pydantic; dimensions stay separate)
# ---------------------------------------------------------------------------


class AllocationFields(BaseModel):
    model_config = ConfigDict(extra="forbid")

    leakage_group_id: str
    duplicate_group_id: str | None = None
    reservation: str
    dataset_role: str
    split: str
    sampling_stratum: str | None = None
    sampling_probability: float | None = None
    train_pool: str | None = None


class AnnotationExportFields(BaseModel):
    """Subset used for train/eval target text — never conflate with classification type."""

    model_config = ConfigDict(extra="allow")

    candidate_text: str | None = None
    gold_text: str | None = None
    gold_kind: str | None = None
    label_tier: str | None = None
    label_source: str | None = None
    annotation_state: str | None = None
    is_human_verified: bool | None = None


# ---------------------------------------------------------------------------
# Plan / shortfall
# ---------------------------------------------------------------------------


@dataclass
class Shortfall:
    split: str
    stratum: str
    requested: int
    available: int
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "stratum": self.stratum,
            "requested": self.requested,
            "available": self.available,
            "deficit": max(0, self.requested - self.available),
            "reason": self.reason,
        }


@dataclass
class SamplingPlan:
    policy_version: str
    sampling_seed: int
    train_size: int
    eval_random_ids: list[str] = field(default_factory=list)
    eval_core_ids: list[str] = field(default_factory=list)
    eval_core_strata: dict[str, list[str]] = field(default_factory=dict)
    train_ids: list[str] = field(default_factory=list)
    train_pool_assignment: dict[str, str] = field(default_factory=dict)
    train_stratum: dict[str, str] = field(default_factory=dict)
    dev_ids: list[str] = field(default_factory=list)
    excluded: list[dict[str, str]] = field(default_factory=list)
    abstain_eval: list[dict[str, str]] = field(default_factory=list)
    shortfalls: list[Shortfall] = field(default_factory=list)
    pending_non_speech_train: list[str] = field(default_factory=list)
    sampling_digest: str = ""
    notes: list[str] = field(default_factory=list)

    def has_blocking_shortfall(self) -> bool:
        return bool(self.shortfalls)  # Includes maximum limits exceeded (available > requested).

    def compute_digest(self) -> str:
        payload = {
            "policy_version": self.policy_version,
            "sampling_seed": self.sampling_seed,
            "train_size": self.train_size,
            "eval_random_ids": self.eval_random_ids,
            "eval_core_ids": self.eval_core_ids,
            "eval_core_strata": self.eval_core_strata,
            "train_ids": self.train_ids,
            "train_pool_assignment": self.train_pool_assignment,
            "dev_ids": self.dev_ids,
            "excluded": self.excluded,
            "abstain_eval": self.abstain_eval,
            "shortfalls": [s.to_dict() for s in self.shortfalls],
            "pending_non_speech_train": self.pending_non_speech_train,
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def freeze(self) -> SamplingPlan:
        self.sampling_digest = self.compute_digest()
        return self

    def to_dict(self) -> dict[str, Any]:
        if not self.sampling_digest:
            self.freeze()
        return {
            "policy_version": self.policy_version,
            "sampling_seed": self.sampling_seed,
            "train_size": self.train_size,
            "sampling_digest": self.sampling_digest,
            "counts": {
                "eval_random": len(self.eval_random_ids),
                "eval_core": len(self.eval_core_ids),
                "train": len(self.train_ids),
                "dev": len(self.dev_ids),
                "excluded": len(self.excluded),
                "abstain_eval": len(self.abstain_eval),
                "shortfalls": len(self.shortfalls),
                "pending_non_speech_train": len(self.pending_non_speech_train),
            },
            "eval_random_ids": list(self.eval_random_ids),
            "eval_core_ids": list(self.eval_core_ids),
            "eval_core_strata": {k: list(v) for k, v in self.eval_core_strata.items()},
            "train_ids": list(self.train_ids),
            "train_pool_assignment": dict(self.train_pool_assignment),
            "train_stratum": dict(self.train_stratum),
            "dev_ids": list(self.dev_ids),
            "excluded": list(self.excluded),
            "abstain_eval": list(self.abstain_eval),
            "shortfalls": [s.to_dict() for s in self.shortfalls],
            "pending_non_speech_train": list(self.pending_non_speech_train),
            "notes": list(self.notes),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stable_unit(seed: int | str, *parts: str) -> float:
    payload = "\0".join([str(seed), *parts]).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _rank_key(seed: int | str, *parts: str) -> tuple[float, str]:
    return (_stable_unit(seed, *parts), "|".join(parts))


def _label(sample: Sample, *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in sample.labels and sample.labels[key] is not None:
            return sample.labels[key]
    return default


def _reservation_role(sample: Sample, reservation: ReservationArtifact | None) -> str:
    if reservation is not None:
        role = reservation.sample_role.get(sample.id)
        if role:
            return role
    return str(
        _label(sample, "reservation", "dataset_role", "reservation_role", default="") or ""
    )


def _group_id(sample: Sample, reservation: ReservationArtifact | None) -> str:
    if reservation is not None and sample.id in reservation.group_mapping:
        return reservation.group_mapping[sample.id]
    return str(
        _label(sample, "leakage_group_id", "duplicate_group_id", default=sample.id) or sample.id
    )


def _call_key(sample: Sample) -> str | None:
    for key in ("call_id", "conversation_id"):
        value = sample.labels.get(key)
        if value:
            return f"{key}:{value}"
    return None


def _exact_dup_key(sample: Sample, fields: Sequence[str]) -> str | None:
    for field_name in fields:
        value = sample.labels.get(field_name)
        if value:
            return f"{field_name}:{value}"
    if sample.sha256:
        return f"sha256:{sample.sha256}"
    return None


def _risk_tags(sample: Sample) -> set[str]:
    raw = sample.labels.get("risk_tags") or []
    if isinstance(raw, str):
        return {x.strip() for x in raw.split(",") if x.strip()}
    return {str(x) for x in raw}


def _verified_error_tags(sample: Sample) -> set[str]:
    raw = sample.labels.get("verified_error_tags") or []
    if isinstance(raw, str):
        return {x.strip() for x in raw.split(",") if x.strip()}
    return {str(x) for x in raw}


def _is_trusted_or_human(sample: Sample) -> bool:
    source = str(sample.labels.get("label_source") or "")
    if source in {LABEL_SOURCE_HUMAN, LABEL_SOURCE_TRUSTED_EXTERNAL, "trusted_external"}:
        return True
    return bool(sample.labels.get("is_human_verified"))


def is_formal_eval_gold(sample: Sample, *, require_dual: bool = True) -> bool:
    """Speech / confirmed non_speech empty after dual/adjudication (or trusted external)."""
    from audio_engine.core.annotation_v3.gold import has_formal_gold_evidence
    if not has_formal_gold_evidence(sample, require_dual=require_dual):
        return False
    kind = sample.labels.get("gold_kind")
    text = sample.labels.get("gold_text") if "gold_text" in sample.labels else None
    state = str(sample.labels.get("annotation_state") or "")
    source = str(sample.labels.get("label_source") or "")
    if kind in NON_FORMAL_GOLD_KINDS:
        return False
    if text is None:
        return False
    if kind == GOLD_KIND_SPEECH and not str(text).strip():
        return False
    if kind == GOLD_KIND_NON_SPEECH and text != "":
        return False
    if source == LABEL_SOURCE_TRUSTED_EXTERNAL or source == "trusted_external":
        return kind in {GOLD_KIND_SPEECH, GOLD_KIND_NON_SPEECH}
    if require_dual:
        if state not in {STATE_SECOND_REVIEW, STATE_ADJUDICATED}:
            return False
        return bool(sample.labels.get("is_human_verified")) and may_pass_formal_gold(
            kind, text, state
        )
    return bool(sample.labels.get("is_human_verified")) and may_pass_formal_gold(
        kind, text, state
    )


def is_abstain_eval(sample: Sample) -> bool:
    kind = str(sample.labels.get("gold_kind") or "")
    return kind in NON_FORMAL_GOLD_KINDS


def is_human_train_gold(
    sample: Sample,
    *,
    allow_non_speech: bool,
) -> bool:
    from audio_engine.core.annotation_v3.gold import has_formal_gold_evidence
    if not has_formal_gold_evidence(sample, require_dual=False):
        return False
    if not _is_trusted_or_human(sample):
        return False
    kind = str(sample.labels.get("gold_kind") or "")
    if kind == GOLD_KIND_SPEECH:
        text = sample.labels.get("gold_text")
        return bool(text) and bool(str(text).strip()) and bool(
            sample.labels.get("is_human_verified")
        )
    if kind == GOLD_KIND_NON_SPEECH:
        if not allow_non_speech:
            return False
        return sample.labels.get("gold_text") == "" and bool(
            sample.labels.get("is_human_verified")
        )
    return False


def is_audited_pseudo_high(sample: Sample, *, require_audit: bool) -> bool:
    type_ = str(sample.labels.get("type") or "")
    tier = str(sample.labels.get("label_tier") or "")
    if type_ != TYPE_PSEUDO_HIGH and tier != TYPE_PSEUDO_HIGH:
        return False
    if not str(sample.labels.get("candidate_text") or "").strip():
        return False
    decision = str(sample.labels.get("decision") or "")
    state = str(sample.labels.get("annotation_state") or "")
    if (sample.labels.get("pseudo_audit_passed") is True
        and sample.labels.get("pseudo_audit_stop_publish") is not True
        and sample.labels.get("pseudo_audit_report_digest")
        and sample.labels.get("pseudo_audit_scope_digest")):
        return True
    if not require_audit and decision == "auto_accept":
        return True
    return False


def qwen_has_confirmed_error(sample: Sample, qwen_keys: Sequence[str]) -> bool:
    for key in ("qwen_verified_error_1", "qwen_verified_error_2", "qwen_correction_candidate"):
        val = sample.labels.get(key)
        if val is True or str(val).lower() in {"true", "1", "yes"}:
            # correction_candidate alone is not enough without human gold mismatch
            if key == "qwen_correction_candidate":
                continue
            return True
    gold = sample.labels.get("gold_text")
    if gold is None or not sample.labels.get("is_human_verified"):
        return False
    gold_cmp = comparison_text(str(gold))
    for run in qwen_keys:
        text = sample.get_transcript_text(run)
        if comparison_text(text) != gold_cmp:
            return True
    return False


def is_human_high_risk(sample: Sample) -> bool:
    tags = _risk_tags(sample)
    if tags & {
        RISK_SHORT_UTTERANCE,
        RISK_NOISY_AUDIO,
        RISK_CROSSTALK_SUSPECTED,
        "negation_flip",
        "false_affirmation_candidate",
        "filler_affirmation",
        "rejection_sanitization",
        "critical_token_conflict",
        "presence_conflict",
    }:
        return True
    semantic = str(sample.labels.get("human_semantic") or "")
    if semantic in {"negative", "mixed"}:
        return True
    noise = str(sample.labels.get("human_noise") or "")
    if noise == "noisy":
        return True
    crosstalk = str(sample.labels.get("human_crosstalk") or "").lower()
    if crosstalk in {"true", "1", "yes"}:
        return True
    duration = sample.duration
    gold = str(sample.labels.get("gold_text") or "")
    if duration is not None and duration <= 2.0:
        return True
    if 0 < len(gold) <= 6:
        return True
    return False


def assign_eval_core_stratum(sample: Sample) -> str:
    """Mutually exclusive primary layer by ordered rules (012-D §4.2)."""
    kind = str(sample.labels.get("gold_kind") or "")
    semantic = str(sample.labels.get("human_semantic") or "")
    noise = str(sample.labels.get("human_noise") or "")
    crosstalk = str(sample.labels.get("human_crosstalk") or "").lower()
    gold = str(sample.labels.get("gold_text") or "")
    duration = sample.duration if sample.duration is not None else 99.0

    if kind == GOLD_KIND_NON_SPEECH:
        return "confirmed_non_speech"
    if semantic == "negative":
        return "negative_reject"
    # Pure filler / neutral short feedback (not a full negative/positive sentence).
    if semantic == "neutral" and (
        duration <= 2.0 or len(gold) <= 6 or RISK_SHORT_UTTERANCE in _risk_tags(sample)
    ):
        return "filler_neutral"
    if kind == GOLD_KIND_NON_SPEECH:
        return "confirmed_non_speech"
    if kind == GOLD_KIND_SPEECH and (
        noise == "noisy" or crosstalk in {"true", "1", "yes"}
    ):
        return "noisy_crosstalk_speech"
    if semantic == "positive":
        return "positive"
    return "other"


def assign_train_pool(sample: Sample, cfg: SamplingConfig) -> str | None:
    if is_human_train_gold(sample, allow_non_speech=cfg.allow_non_speech_train):
        if qwen_has_confirmed_error(sample, cfg.qwen_run_keys):
            return "qwen_error_fix"
        if is_human_high_risk(sample):
            return "human_high_risk"
        return "human_ordinary"
    if is_audited_pseudo_high(
        sample, require_audit=cfg.require_pseudo_audit_for_pseudo_train
    ):
        return "pseudo_high_audited"
    return None


def _blocked_groups_for_train(
    reservation: ReservationArtifact,
) -> set[str]:
    """All eval/calibration/governance reserved groups stay blocked from train/dev."""
    blocked: set[str] = set()
    for gid, role in reservation.group_role.items():
        if role in {
            RESERVATION_EVAL_RANDOM,
            RESERVATION_EVAL_CORE_RESERVE,
            RESERVATION_CALIBRATION,
            RESERVATION_GOVERNANCE_HOLD,
        }:
            blocked.add(gid)
    return blocked


def _pick_quota(
    candidates: list[Sample],
    *,
    n: int,
    seed: int,
    salt: str,
    used_ids: set[str],
    used_dups: set[str],
    call_counts: dict[str, int],
    max_per_call: int,
    dup_fields: Sequence[str],
    blocked_groups: set[str],
    group_of: dict[str, str],
) -> list[Sample]:
    ordered = sorted(
        candidates,
        key=lambda s: _rank_key(seed, salt, s.id),
    )
    picked: list[Sample] = []
    for sample in ordered:
        if len(picked) >= n:
            break
        if sample.id in used_ids:
            continue
        gid = group_of.get(sample.id, _group_id(sample, None))
        if gid in blocked_groups:
            continue
        dup = _exact_dup_key(sample, dup_fields)
        if dup and dup in used_dups:
            continue
        call = _call_key(sample)
        if call and call_counts.get(call, 0) >= max_per_call:
            continue
        picked.append(sample)
        used_ids.add(sample.id)
        if dup:
            used_dups.add(dup)
        if call:
            call_counts[call] = call_counts.get(call, 0) + 1
    return picked


def _cross_constraint_shortfalls(
    selected: list[Sample],
    *,
    train_size: int,
    cross: TrainCrossConstraints,
    pool_assignment: dict[str, str],
) -> list[Shortfall]:
    if train_size <= 0 or not selected:
        return []
    n = len(selected)
    shortfalls: list[Shortfall] = []

    def ratio(count: int) -> float:
        return count / float(train_size)

    positives = sum(
        1 for s in selected if str(s.labels.get("human_semantic") or "") == "positive"
    )
    negatives = sum(
        1 for s in selected if str(s.labels.get("human_semantic") or "") == "negative"
    )
    noisy = sum(
        1
        for s in selected
        if s.labels.get("gold_kind") == GOLD_KIND_SPEECH and (
            str(s.labels.get("human_noise") or "") == "noisy"
            or str(s.labels.get("human_crosstalk") or "").lower() in {"true", "1", "yes"})
    )
    non_speech = sum(
        1 for s in selected if str(s.labels.get("gold_kind") or "") == GOLD_KIND_NON_SPEECH
    )
    pseudo = sum(
        1 for s in selected if pool_assignment.get(s.id) == "pseudo_high_audited"
    )

    checks = [
        ("positive", cross.min_positive_ratio, positives, True),
        ("negative", cross.min_negative_ratio, negatives, True),
        ("noisy_crosstalk", cross.min_noisy_crosstalk_ratio, noisy, True),
        ("non_speech", cross.max_non_speech_ratio, non_speech, False),
        ("pseudo_high", cross.max_pseudo_high_ratio, pseudo, False),
    ]
    for name, limit, count, is_min in checks:
        if is_min:
            need = int(train_size * limit + 0.999999)  # ceil without import
            if count < need:
                shortfalls.append(
                    Shortfall(
                        split="train",
                        stratum=f"cross:{name}",
                        requested=need,
                        available=count,
                        reason=f"cross_constraint_min_{name}",
                    )
                )
        else:
            max_allowed = int(train_size * limit)
            if count > max_allowed:
                shortfalls.append(
                    Shortfall(
                        split="train",
                        stratum=f"cross:{name}",
                        requested=max_allowed,
                        available=count,
                        reason=f"cross_constraint_max_{name}_exceeded",
                    )
                )
    # Also flag if final n < train_size (pool fill shortfall already recorded).
    _ = ratio  # keep helper for clarity / future reports
    _ = n
    return shortfalls


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------


def build_sampling_plan(
    samples: Sequence[Sample],
    reservation: ReservationArtifact | None,
    config: SamplingConfig,
) -> SamplingPlan:
    """Build a deterministic sampling plan. Does not mutate samples."""
    samples = sorted(samples, key=lambda s: s.id)
    if config.require_reservation and reservation is None:
        raise ValueError(
            "dataset_policy_v3 requires an immutable reservation artifact; "
            "missing group metadata must not silently fall back to ID-only split"
        )
    if config.train_size <= 0:
        raise ValueError("train_size is required and must be > 0")
    import math
    ratios = [ratio for _, ratio in config.train_pools.as_ordered()]
    if any(not math.isfinite(r) or not 0 <= r <= 1 for r in ratios) or not math.isclose(sum(ratios), 1.0):
        raise ValueError("train pool ratios must be finite nonnegative values summing to 1")

    by_id = {s.id: s for s in samples}
    if len(by_id) != len(samples):
        raise ValueError("duplicate sample ids in dataset build")
    plan = SamplingPlan(
        policy_version=config.policy_version,
        sampling_seed=config.sampling_seed,
        train_size=config.train_size,
    )
    notes = plan.notes

    if reservation is not None and not reservation.verify_digest():
        raise ValueError("reservation content_digest mismatch — refusing to sample")
    if reservation is not None:
        unknown = set(by_id) - set(reservation.input_snapshot_ids)
        if unknown:
            raise ValueError(f"samples absent from frozen reservation: {sorted(unknown)[:10]}")
        for s in samples:
            audio_hash = str(s.labels.get("original_audio_sha256") or s.sha256 or "")
            if not audio_hash or reservation.audio_hashes.get(s.id) != audio_hash:
                raise ValueError(f"audio hash differs from frozen reservation: {s.id}")

    group_of: dict[str, str] = {}
    for sample in samples:
        group_of[sample.id] = _group_id(sample, reservation)

    blocked_train_groups = (
        _blocked_groups_for_train(reservation) if reservation is not None else set()
    )

    # ---- eval_random from frozen reservation IDs (+ optional top-up) ----
    primary_ids = list(reservation.eval_random_ids) if reservation else []
    if not primary_ids:
        # Fallback only when reservation absent was already allowed.
        primary_ids = [
            s.id
            for s in samples
            if _reservation_role(s, reservation) == RESERVATION_EVAL_RANDOM
        ]
        primary_ids = sorted(
            primary_ids, key=lambda sid: _rank_key(config.sampling_seed, "eval_random", sid)
        )[: config.eval_random_target]

    evaluable_random: list[str] = []
    locked_eval_random_groups: set[str] = set()
    for sid in primary_ids:
        sample = by_id.get(sid)
        if sample is None:
            plan.excluded.append({"id": sid, "reason": "eval_random_missing_from_manifest"})
            continue
        gid = group_of[sid]
        locked_eval_random_groups.add(gid)
        if is_abstain_eval(sample):
            plan.abstain_eval.append(
                {
                    "id": sid,
                    "reason": f"abstain:{sample.labels.get('gold_kind')}",
                    "split": "eval_random",
                }
            )
            continue
        if is_formal_eval_gold(sample, require_dual=config.require_dual_review_for_eval):
            evaluable_random.append(sid)
        else:
            plan.excluded.append(
                {
                    "id": sid,
                    "reason": "eval_random_not_dual_reviewed_or_incomplete_gold",
                    "detail": str(sample.labels.get("annotation_state") or "pending"),
                }
            )

    # Top-up: (1) other members of already-locked eval_random groups;
    # (2) frozen alternate sequence ONLY when that sample's group is still
    # role=eval_random (never invade core/dev/train/calibration).
    if reservation is not None and len(evaluable_random) < config.eval_random_target:
        topup_order: list[str] = []
        for sid in reservation.eval_random_candidate_order:
            if reservation.sample_role.get(sid) == RESERVATION_EVAL_RANDOM:
                topup_order.append(sid)
        # Preserve coverage stats of the initial queue for reporting.
        plan.notes.append(
            "eval_random_topup_uses_locked_group_members_only;"
            f"frozen_alternate_len={len(reservation.eval_random_candidate_order)}"
        )
        seen_topup: set[str] = set()
        for sid in topup_order:
            if len(evaluable_random) >= config.eval_random_target:
                break
            if sid in seen_topup or sid in evaluable_random:
                continue
            seen_topup.add(sid)
            sample = by_id.get(sid)
            if sample is None:
                continue
            if is_abstain_eval(sample):
                if not any(row["id"] == sid for row in plan.abstain_eval):
                    plan.abstain_eval.append(
                        {
                            "id": sid,
                            "reason": f"abstain:{sample.labels.get('gold_kind')}",
                            "split": "eval_random",
                        }
                    )
                continue
            if is_formal_eval_gold(sample, require_dual=config.require_dual_review_for_eval):
                evaluable_random.append(sid)
                locked_eval_random_groups.add(group_of[sid])

    plan.eval_random_ids = evaluable_random[: config.eval_random_target]
    if len(plan.eval_random_ids) < config.eval_random_target:
        plan.shortfalls.append(
            Shortfall(
                split="eval_random",
                stratum="overall",
                requested=config.eval_random_target,
                available=len(plan.eval_random_ids),
                reason="insufficient_dual_reviewed_evaluable_samples",
            )
        )
    notes.append(
        "eval_random is a frozen original-distribution draw; metrics apply to the "
        "evaluable subset only and must not be claimed as the full raw distribution"
    )

    selected_eval_groups = {group_of[sid] for sid in plan.eval_random_ids}

    # ---- eval_core stratified from eval_core_reserve ----
    core_candidates = [
        s
        for s in samples
        if _reservation_role(s, reservation) == RESERVATION_EVAL_CORE_RESERVE
        and group_of[s.id] not in selected_eval_groups
    ]
    strata_buckets: dict[str, list[Sample]] = defaultdict(list)
    for sample in core_candidates:
        if is_abstain_eval(sample):
            plan.abstain_eval.append(
                {
                    "id": sample.id,
                    "reason": f"abstain:{sample.labels.get('gold_kind')}",
                    "split": "eval_core",
                }
            )
            continue
        if not is_formal_eval_gold(sample, require_dual=config.require_dual_review_for_eval):
            plan.excluded.append(
                {
                    "id": sample.id,
                    "reason": "eval_core_not_ready",
                    "detail": str(sample.labels.get("annotation_state") or "pending"),
                }
            )
            continue
        strata_buckets[assign_eval_core_stratum(sample)].append(sample)

    used_ids: set[str] = set(plan.eval_random_ids)
    used_dups: set[str] = set()
    call_counts: dict[str, int] = {}
    # Mark dups already used by eval_random
    for sid in plan.eval_random_ids:
        sample = by_id[sid]
        dup = _exact_dup_key(sample, config.exact_duplicate_fields)
        if dup:
            used_dups.add(dup)

    eval_core_ids: list[str] = []
    eval_core_strata: dict[str, list[str]] = {}
    for stratum, quota in config.eval_core.as_ordered():
        picked = _pick_quota(
            strata_buckets.get(stratum, []),
            n=quota,
            seed=config.sampling_seed,
            salt=f"eval_core:{stratum}",
            used_ids=used_ids,
            used_dups=used_dups,
            call_counts=call_counts,
            max_per_call=config.max_per_call,
            dup_fields=config.exact_duplicate_fields,
            blocked_groups=selected_eval_groups,
            group_of=group_of,
        )
        ids = [s.id for s in picked]
        eval_core_strata[stratum] = ids
        eval_core_ids.extend(ids)
        if len(ids) < quota:
            plan.shortfalls.append(
                Shortfall(
                    split="eval_core",
                    stratum=stratum,
                    requested=quota,
                    available=len(ids),
                    reason="stratum_quota_shortfall",
                )
            )
    plan.eval_core_ids = eval_core_ids
    plan.eval_core_strata = eval_core_strata
    notes.append(
        "eval_core is a non-natural stratified specialty set; do not average with "
        "eval_random as an online metric"
    )

    selected_core_groups = {group_of[sid] for sid in plan.eval_core_ids}
    # Entire eval reserves stay blocked (selected or not).
    if reservation is not None:
        for gid, role in reservation.group_role.items():
            if role in {RESERVATION_EVAL_RANDOM, RESERVATION_EVAL_CORE_RESERVE}:
                blocked_train_groups.add(gid)
    blocked_train_groups |= selected_eval_groups | selected_core_groups

    # ---- dev: human/trusted gold in DEV reservation only ----
    dev_candidates = [
        s
        for s in samples
        if _reservation_role(s, reservation) == RESERVATION_DEV
        and group_of[s.id] not in blocked_train_groups
        and is_formal_eval_gold(s, require_dual=False)  # single review OK for train/dev gold
        and _is_trusted_or_human(s)
    ]
    # Prefer dual-complete when available.
    dev_candidates = sorted(
        dev_candidates,
        key=lambda s: (
            0
            if str(s.labels.get("annotation_state") or "")
            in {STATE_SECOND_REVIEW, STATE_ADJUDICATED}
            else 1,
            _rank_key(config.sampling_seed, "dev", s.id),
        ),
    )
    # Take all eligible dev gold (quota not fixed by policy; reservation sized the pool).
    plan.dev_ids = []
    for sample in dev_candidates:
        if sample.id in used_ids:
            continue
        gid = group_of[sample.id]
        if gid in blocked_train_groups:
            continue
        dup = _exact_dup_key(sample, config.exact_duplicate_fields)
        if dup and dup in used_dups:
            continue
        plan.dev_ids.append(sample.id)
        used_ids.add(sample.id)
        if dup:
            used_dups.add(dup)

    # ---- train four pools ----
    train_pool_samples: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        role = _reservation_role(sample, reservation)
        if role != RESERVATION_TRAIN_POOL:
            continue
        if group_of[sample.id] in blocked_train_groups:
            continue
        if sample.id in used_ids:
            continue
        # Pending non_speech when adapter does not support empty targets
        kind = str(sample.labels.get("gold_kind") or "")
        if (
            kind == GOLD_KIND_NON_SPEECH
            and sample.labels.get("is_human_verified")
            and sample.labels.get("gold_text") == ""
            and not config.allow_non_speech_train
        ):
            plan.pending_non_speech_train.append(sample.id)
            plan.excluded.append(
                {
                    "id": sample.id,
                    "reason": "non_speech_pending_adapter_support",
                    "detail": "do_not_rewrite_as_noise_or_drop_silently",
                }
            )
            continue
        pool = assign_train_pool(sample, config)
        if pool is None:
            continue
        train_pool_samples[pool].append(sample)

    pool_targets: dict[str, int] = {}
    assigned = 0
    ordered_pools = config.train_pools.as_ordered()
    for i, (name, ratio) in enumerate(ordered_pools):
        if i == len(ordered_pools) - 1:
            pool_targets[name] = max(0, config.train_size - assigned)
        else:
            n = int(round(config.train_size * ratio))
            pool_targets[name] = n
            assigned += n

    train_ids: list[str] = []
    pre_train_used_ids, pre_train_dups = set(used_ids), set(used_dups)
    train_pool_assignment: dict[str, str] = {}
    train_call_counts: dict[str, int] = dict(call_counts)
    for name, _ratio in ordered_pools:
        target = pool_targets[name]
        picked = _pick_quota(
            train_pool_samples.get(name, []),
            n=target,
            seed=config.sampling_seed,
            salt=f"train:{name}",
            used_ids=used_ids,
            used_dups=used_dups,
            call_counts=train_call_counts,
            max_per_call=config.max_per_call,
            dup_fields=config.exact_duplicate_fields,
            blocked_groups=blocked_train_groups,
            group_of=group_of,
        )
        for sample in picked:
            train_ids.append(sample.id)
            train_pool_assignment[sample.id] = name
            plan.train_stratum[sample.id] = name
        if len(picked) < target:
            plan.shortfalls.append(
                Shortfall(
                    split="train",
                    stratum=name,
                    requested=target,
                    available=len(picked),
                    reason="train_pool_quota_shortfall",
                )
            )

    if len(train_ids) != config.train_size or _cross_constraint_shortfalls(
        [by_id[sid] for sid in train_ids], train_size=config.train_size,
        cross=config.train_cross, pool_assignment=train_pool_assignment):
        from audio_engine.core.dataset_v3.quota_solver import solve_train_quotas
        solved = solve_train_quotas(train_pool_samples, pool_targets, config,
            used_ids=pre_train_used_ids, used_dups=pre_train_dups, call_counts=call_counts)
        if solved is not None:
            train_ids = [s.id for s, _ in solved]
            train_pool_assignment = {s.id: pool for s, pool in solved}
            plan.train_stratum = dict(train_pool_assignment)
            plan.shortfalls = [sf for sf in plan.shortfalls if sf.split != "train"]
            used_ids = pre_train_used_ids | set(train_ids)
            notes.append("joint_integer_quota_solver: feasible solution found without relaxing quotas")
    plan.train_ids = train_ids
    plan.train_pool_assignment = train_pool_assignment

    selected_train = [by_id[sid] for sid in train_ids]
    plan.shortfalls.extend(
        _cross_constraint_shortfalls(
            selected_train,
            train_size=config.train_size,
            cross=config.train_cross,
            pool_assignment=train_pool_assignment,
        )
    )
    if len(train_ids) < config.train_size:
        # Overall fill shortfall if not already covered by pool shortfalls
        if not any(
            s.split == "train" and s.stratum != "overall" and s.requested > s.available
            for s in plan.shortfalls
        ):
            pass
        if not any(s.split == "train" and s.stratum == "overall" for s in plan.shortfalls):
            plan.shortfalls.append(
                Shortfall(
                    split="train",
                    stratum="overall",
                    requested=config.train_size,
                    available=len(train_ids),
                    reason="train_size_not_met",
                )
            )

    # Governance hold never enters formal splits
    for sample in samples:
        if _reservation_role(sample, reservation) == RESERVATION_GOVERNANCE_HOLD:
            if sample.id not in used_ids and not any(
                row.get("id") == sample.id for row in plan.excluded
            ):
                plan.excluded.append(
                    {
                        "id": sample.id,
                        "reason": "governance_hold",
                        "detail": "missing_group_metadata_or_isolation",
                    }
                )

    return plan.freeze()


def apply_sampling_plan_to_samples(
    samples: Sequence[Sample],
    plan: SamplingPlan,
    reservation: ReservationArtifact | None = None,
) -> list[Sample]:
    """Stamp split / stratum / train_pool onto labels; set training target text."""
    split_of: dict[str, str] = {}
    stratum_of: dict[str, str] = {}
    for sid in plan.eval_random_ids:
        split_of[sid] = "eval_random"
        stratum_of[sid] = "eval_random"
    for stratum, ids in plan.eval_core_strata.items():
        for sid in ids:
            split_of[sid] = "eval_core"
            stratum_of[sid] = stratum
    for sid in plan.dev_ids:
        split_of[sid] = "dev"
        stratum_of[sid] = "dev"
    for sid in plan.train_ids:
        split_of[sid] = "train"
        stratum_of[sid] = plan.train_stratum.get(sid) or plan.train_pool_assignment.get(
            sid, "train"
        )

    excluded_ids = {row["id"] for row in plan.excluded}
    abstain_ids = {row["id"] for row in plan.abstain_eval}

    updated: list[Sample] = []
    for source in samples:
        sample = source.model_copy(deep=True)
        sample.labels["split"] = "excluded"
        sample.labels["exclude_reason"] = "not_selected"
        role = _reservation_role(sample, reservation)
        gid = _group_id(sample, reservation)
        split = split_of.get(sample.id)
        if split:
            sample.labels.pop("exclude_reason", None)
            sample.labels["split"] = split
            sample.labels["dataset_role"] = split
            sample.labels["sampling_stratum"] = stratum_of.get(sample.id)
            sample.labels["sampling_digest"] = plan.sampling_digest
            if split == "train":
                pool = plan.train_pool_assignment.get(sample.id)
                sample.labels["train_pool"] = pool
                # Target text: human gold vs audited pseudo candidate
                if pool == "pseudo_high_audited":
                    sample.labels["train_target_text"] = sample.labels.get("candidate_text")
                    sample.labels["train_target_source"] = "pseudo_high_candidate"
                    # Never write pseudo into human gold fields
                else:
                    sample.labels["train_target_text"] = sample.labels.get("gold_text")
                    sample.labels["train_target_source"] = str(
                        sample.labels.get("label_source") or "human"
                    )
            elif split in {"eval_core", "eval_random", "dev"}:
                sample.labels["eval_target_text"] = sample.labels.get("gold_text")
                sample.labels["eval_gold_kind"] = sample.labels.get("gold_kind")
        elif sample.id in abstain_ids:
            sample.labels["split"] = "excluded"
            sample.labels["dataset_role"] = "abstain_eval"
            sample.labels["exclude_reason"] = "abstain_eval"
        elif sample.id in excluded_ids:
            sample.labels["split"] = "excluded"
            sample.labels["dataset_role"] = "excluded"
            reason = next(
                (row.get("reason") for row in plan.excluded if row.get("id") == sample.id),
                "excluded",
            )
            sample.labels["exclude_reason"] = reason
        else:
            # Unselected but reserved — keep reservation role, mark holdout unused
            if role in {
                RESERVATION_EVAL_RANDOM,
                RESERVATION_EVAL_CORE_RESERVE,
            }:
                sample.labels["split"] = "excluded"
                sample.labels["dataset_role"] = "eval_reserve_unused"
                sample.labels["exclude_reason"] = "eval_reserve_not_selected_still_blocked"
            elif role == RESERVATION_CALIBRATION:
                sample.labels["split"] = "excluded"
                sample.labels["dataset_role"] = RESERVATION_CALIBRATION
                sample.labels["exclude_reason"] = "calibration_not_in_formal_release"
            elif role == RESERVATION_GOVERNANCE_HOLD:
                sample.labels["split"] = "excluded"
                sample.labels["dataset_role"] = RESERVATION_GOVERNANCE_HOLD
                sample.labels["exclude_reason"] = "governance_hold"

        sample.labels["leakage_group_id"] = gid
        if role:
            sample.labels["reservation"] = role
        # Allocation contract check (best-effort; do not fail stamp on optional fields)
        try:
            AllocationFields(
                leakage_group_id=gid,
                duplicate_group_id=(
                    str(sample.labels["duplicate_group_id"])
                    if sample.labels.get("duplicate_group_id")
                    else None
                ),
                reservation=str(sample.labels.get("reservation") or role or ""),
                dataset_role=str(sample.labels.get("dataset_role") or ""),
                split=str(sample.labels.get("split") or ""),
                sampling_stratum=(
                    str(sample.labels["sampling_stratum"])
                    if sample.labels.get("sampling_stratum")
                    else None
                ),
                sampling_probability=None,
                train_pool=(
                    str(sample.labels["train_pool"])
                    if sample.labels.get("train_pool")
                    else None
                ),
            )
        except Exception:
            pass
        updated.append(sample)
    return updated
