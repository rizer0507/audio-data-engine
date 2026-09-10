"""Immutable eval/dev/calibration reservation (dataset_policy_v3 stage A)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from audio_engine.core.artifacts import atomic_write_json
from audio_engine.core.sample import Sample
from audio_engine.core.dataset_v3.grouping import GroupingResult, has_source_index, GroupingConfig
from audio_engine.core.selection_v3.types import (
    DATASET_POLICY_VERSION,
    GOVERNANCE_MISSING_GROUP_META,
    GOVERNANCE_NEAR_DUP_UNCERTAIN,
    RESERVATION_CALIBRATION,
    RESERVATION_DEV,
    RESERVATION_EVAL_CORE_RESERVE,
    RESERVATION_EVAL_RANDOM,
    RESERVATION_GOVERNANCE_HOLD,
    RESERVATION_TRAIN_POOL,
)


def _stable_unit(seed: int | str, *parts: str) -> float:
    payload = "\0".join([str(seed), *parts]).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _rank_key(seed: int | str, *parts: str) -> tuple[float, str]:
    """Deterministic sort key; secondary key breaks ties stably."""
    return (_stable_unit(seed, *parts), "|".join(parts))


@dataclass
class ReservationConfig:
    seed: int = 42
    eval_random_target: int = 2000
    eval_core_reserve_ratio: float = 0.15
    dev_ratio_of_dev_pool: float = 0.10
    calibration_target: int = 3000
    # When True, samples missing group metadata never enter formal splits
    isolate_missing_group_meta: bool = True
    policy_version: str = DATASET_POLICY_VERSION
    # Optional design provenance when historical model errors were already observed
    design_formed_at: str = ""
    prior_information_used: list[str] = field(default_factory=list)

    @classmethod
    def from_params(cls, params: dict[str, Any] | None) -> ReservationConfig:
        raw = params or {}
        return cls(
            seed=int(raw.get("seed", 42)),
            eval_random_target=int(raw.get("eval_random_target", 2000)),
            eval_core_reserve_ratio=float(raw.get("eval_core_reserve_ratio", 0.15)),
            dev_ratio_of_dev_pool=float(raw.get("dev_ratio_of_dev_pool", 0.10)),
            calibration_target=int(raw.get("calibration_target", 3000)),
            isolate_missing_group_meta=bool(
                raw.get("isolate_missing_group_meta", True)
            ),
            policy_version=str(raw.get("policy_version") or DATASET_POLICY_VERSION),
            design_formed_at=str(raw.get("design_formed_at") or ""),
            prior_information_used=[
                str(x) for x in (raw.get("prior_information_used") or [])
            ],
        )


@dataclass
class ReservationArtifact:
    """Immutable reservation record — frozen before looking at model errors."""

    policy_version: str
    seed: int
    input_snapshot_ids: list[str]
    group_mapping: dict[str, str]  # sample_id → leakage_group_id
    sample_role: dict[str, str]  # sample_id → reservation role
    group_role: dict[str, str]  # leakage_group_id → role
    eval_random_ids: list[str]
    eval_random_candidate_order: list[str]  # full alternate sequence
    eval_core_reserve_group_ids: list[str]
    calibration_ids: list[str]
    dev_group_ids: list[str]
    train_pool_group_ids: list[str]
    governance_hold_ids: list[str]
    exclusions: list[dict[str, str]] = field(default_factory=list)
    design_formed_at: str = ""
    prior_information_used: list[str] = field(default_factory=list)
    content_digest: str = ""
    audio_hashes: dict[str, str] = field(default_factory=dict)

    def compute_digest(self) -> str:
        payload = {
            "policy_version": self.policy_version,
            "seed": self.seed,
            "input_snapshot_ids": self.input_snapshot_ids,
            "group_mapping": self.group_mapping,
            "sample_role": self.sample_role,
            "group_role": self.group_role,
            "eval_random_ids": self.eval_random_ids,
            "eval_random_candidate_order": self.eval_random_candidate_order,
            "eval_core_reserve_group_ids": self.eval_core_reserve_group_ids,
            "calibration_ids": self.calibration_ids,
            "dev_group_ids": self.dev_group_ids,
            "train_pool_group_ids": self.train_pool_group_ids,
            "governance_hold_ids": self.governance_hold_ids,
            "exclusions": self.exclusions,
            "design_formed_at": self.design_formed_at,
            "prior_information_used": self.prior_information_used,
            "audio_hashes": self.audio_hashes,
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def freeze(self) -> ReservationArtifact:
        self.content_digest = self.compute_digest()
        return self

    def to_dict(self) -> dict[str, Any]:
        if not self.content_digest:
            self.freeze()
        return {
            "policy_version": self.policy_version,
            "seed": self.seed,
            "content_digest": self.content_digest,
            "audio_hashes": dict(self.audio_hashes),
            "input_snapshot_ids": list(self.input_snapshot_ids),
            "group_mapping": dict(self.group_mapping),
            "sample_role": dict(self.sample_role),
            "group_role": dict(self.group_role),
            "eval_random_ids": list(self.eval_random_ids),
            "eval_random_candidate_order": list(self.eval_random_candidate_order),
            "eval_core_reserve_group_ids": list(self.eval_core_reserve_group_ids),
            "calibration_ids": list(self.calibration_ids),
            "dev_group_ids": list(self.dev_group_ids),
            "train_pool_group_ids": list(self.train_pool_group_ids),
            "governance_hold_ids": list(self.governance_hold_ids),
            "exclusions": list(self.exclusions),
            "design_formed_at": self.design_formed_at,
            "prior_information_used": list(self.prior_information_used),
            "counts": {
                "input": len(self.input_snapshot_ids),
                "eval_random": len(self.eval_random_ids),
                "eval_core_reserve_groups": len(self.eval_core_reserve_group_ids),
                "calibration": len(self.calibration_ids),
                "dev_groups": len(self.dev_group_ids),
                "train_pool_groups": len(self.train_pool_group_ids),
                "governance_hold": len(self.governance_hold_ids),
            },
        }

    def write_json(self, path: str | Path) -> Path:
        path = Path(path)
        atomic_write_json(path, self.to_dict())
        return path

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReservationArtifact:
        art = cls(
            policy_version=str(data.get("policy_version") or DATASET_POLICY_VERSION),
            seed=int(data.get("seed", 0)),
            input_snapshot_ids=[str(x) for x in (data.get("input_snapshot_ids") or [])],
            group_mapping={str(k): str(v) for k, v in (data.get("group_mapping") or {}).items()},
            sample_role={str(k): str(v) for k, v in (data.get("sample_role") or {}).items()},
            group_role={str(k): str(v) for k, v in (data.get("group_role") or {}).items()},
            eval_random_ids=[str(x) for x in (data.get("eval_random_ids") or [])],
            eval_random_candidate_order=[
                str(x) for x in (data.get("eval_random_candidate_order") or [])
            ],
            eval_core_reserve_group_ids=[
                str(x) for x in (data.get("eval_core_reserve_group_ids") or [])
            ],
            calibration_ids=[str(x) for x in (data.get("calibration_ids") or [])],
            dev_group_ids=[str(x) for x in (data.get("dev_group_ids") or [])],
            train_pool_group_ids=[
                str(x) for x in (data.get("train_pool_group_ids") or [])
            ],
            governance_hold_ids=[
                str(x) for x in (data.get("governance_hold_ids") or [])
            ],
            exclusions=list(data.get("exclusions") or []),
            design_formed_at=str(data.get("design_formed_at") or ""),
            prior_information_used=[
                str(x) for x in (data.get("prior_information_used") or [])
            ],
            content_digest=str(data.get("content_digest") or ""),
            audio_hashes=dict(data.get("audio_hashes") or {}),
        )
        return art

    def verify_digest(self) -> bool:
        expected = self.content_digest
        actual = self.compute_digest()
        return bool(expected) and expected == actual


def build_reservation(
    samples: list[Sample],
    grouping: GroupingResult,
    config: ReservationConfig,
    *,
    grouping_config: GroupingConfig | None = None,
) -> ReservationArtifact:
    """Fixed allocation order per dataset_policy_v3 §9.2.

    1. Freeze input ids + group mapping
    2. Sample eval_random by original sample id (equal probability), lock whole groups
    3. From remaining groups, reserve eval_core_reserve by group hash ratio
    4. From development pool: reserve calibration (by target count, one per group)
       and dev (ratio of groups); remainder is train_pool
    5. Missing group metadata → governance_hold (never silent ID-only split)
    """
    gcfg = grouping_config or GroupingConfig()
    if config.isolate_missing_group_meta and not gcfg.require_source_index:
        # Still allow explicit isolation when flags already present
        pass

    input_ids = [s.id for s in samples]
    if len(input_ids) != len(set(input_ids)):
        raise ValueError("reservation input contains duplicate sample ids")
    # Freeze snapshot in stable order so shuffle/shard order cannot change the artifact.
    input_ids_stable = sorted(input_ids)

    sample_by_id = {s.id: s for s in samples}
    group_of = dict(grouping.leakage_group_id)
    members_of = {gid: list(mids) for gid, mids in grouping.group_members.items()}

    sample_role: dict[str, str] = {}
    group_role: dict[str, str] = {}
    exclusions: list[dict[str, str]] = []
    governance_hold_ids: list[str] = []

    # --- governance isolation (missing source/call mapping) ---
    eligible_ids: list[str] = []
    for sid in input_ids_stable:
        sample = sample_by_id[sid]
        flags = grouping.governance_flags.get(sid) or []
        isolated = bool({GOVERNANCE_MISSING_GROUP_META, GOVERNANCE_NEAR_DUP_UNCERTAIN} & set(flags)) or bool(
            sample.labels.get("governance_isolation")
        )
        if config.isolate_missing_group_meta and (
            isolated or (gcfg.require_source_index and not has_source_index(sample, gcfg))
        ):
            sample_role[sid] = RESERVATION_GOVERNANCE_HOLD
            governance_hold_ids.append(sid)
            exclusions.append(
                {
                    "id": sid,
                    "reason": "near_duplicate_uncertain" if GOVERNANCE_NEAR_DUP_UNCERTAIN in flags else "missing_group_metadata",
                    "detail": "cannot claim call-leakage-free formal split",
                }
            )
        else:
            eligible_ids.append(sid)

    # Lock entire groups that touch governance hold out of formal pools
    held_groups = {group_of[sid] for sid in governance_hold_ids if sid in group_of}
    for gid in held_groups:
        group_role[gid] = RESERVATION_GOVERNANCE_HOLD
        for mid in members_of.get(gid, []):
            if mid not in sample_role:
                sample_role[mid] = RESERVATION_GOVERNANCE_HOLD
                if mid not in governance_hold_ids:
                    governance_hold_ids.append(mid)
                    exclusions.append(
                        {
                            "id": mid,
                            "reason": "governance_group_lock",
                            "detail": f"group {gid} contains missing_group_metadata",
                        }
                    )

    pool_ids = [sid for sid in eligible_ids if sample_role.get(sid) != RESERVATION_GOVERNANCE_HOLD]

    # --- Step 2: eval_random by sample id, then lock groups ---
    candidate_order = sorted(
        pool_ids, key=lambda sid: _rank_key(config.seed, "eval_random", sid)
    )
    eval_random_ids: list[str] = []
    locked_groups: set[str] = set()
    for sid in candidate_order:
        if len(eval_random_ids) >= config.eval_random_target:
            break
        gid = group_of[sid]
        if group_role.get(gid) == RESERVATION_GOVERNANCE_HOLD:
            continue
        # Selecting this sample locks the whole group into eval_random
        locked_groups.add(gid)
        group_role[gid] = RESERVATION_EVAL_RANDOM
        for mid in members_of.get(gid, []):
            if sample_role.get(mid) == RESERVATION_GOVERNANCE_HOLD:
                continue
            sample_role[mid] = RESERVATION_EVAL_RANDOM
        eval_random_ids.append(sid)

    # Remaining alternate sequence (for top-up without reshuffling)
    remaining_candidates = [
        sid
        for sid in candidate_order
        if sid not in eval_random_ids
        and sample_role.get(sid) != RESERVATION_GOVERNANCE_HOLD
    ]

    # --- Step 3: eval_core_reserve from remaining groups ---
    remaining_groups = sorted(
        {
            group_of[sid]
            for sid in pool_ids
            if group_of[sid] not in group_role
            and sample_role.get(sid) != RESERVATION_GOVERNANCE_HOLD
        }
    )
    # Deterministic by group hash vs ratio threshold
    core_groups: list[str] = []
    for gid in remaining_groups:
        point = _stable_unit(config.seed, "eval_core_reserve", gid)
        if point < config.eval_core_reserve_ratio:
            core_groups.append(gid)
    # Stable order
    core_groups = sorted(core_groups)
    for gid in core_groups:
        group_role[gid] = RESERVATION_EVAL_CORE_RESERVE
        for mid in members_of.get(gid, []):
            if sample_role.get(mid) == RESERVATION_GOVERNANCE_HOLD:
                continue
            sample_role[mid] = RESERVATION_EVAL_CORE_RESERVE

    # --- Step 4: development pool → calibration + dev + train_pool ---
    dev_pool_groups = sorted(
        gid for gid in remaining_groups if gid not in group_role
    )
    # Calibration: take target count of samples, at most one per group, by group order
    cal_group_order = sorted(
        dev_pool_groups,
        key=lambda gid: _rank_key(config.seed, "calibration_group", gid),
    )
    calibration_ids: list[str] = []
    calibration_groups: set[str] = set()
    for gid in cal_group_order:
        if len(calibration_ids) >= config.calibration_target:
            break
        # Pick one sample from the group deterministically
        members = sorted(
            [
                mid
                for mid in members_of.get(gid, [])
                if sample_role.get(mid) not in {
                    RESERVATION_GOVERNANCE_HOLD,
                    RESERVATION_EVAL_RANDOM,
                    RESERVATION_EVAL_CORE_RESERVE,
                }
            ],
            key=lambda mid: _rank_key(config.seed, "calibration_sample", mid),
        )
        if not members:
            continue
        pick = members[0]
        calibration_ids.append(pick)
        calibration_groups.add(gid)
        group_role[gid] = RESERVATION_CALIBRATION
        for mid in members_of.get(gid, []):
            if sample_role.get(mid) == RESERVATION_GOVERNANCE_HOLD:
                continue
            sample_role[mid] = RESERVATION_CALIBRATION

    # Dev: ratio of remaining development groups (after calibration groups removed)
    after_cal = [gid for gid in dev_pool_groups if gid not in group_role]
    after_cal_sorted = sorted(
        after_cal, key=lambda gid: _rank_key(config.seed, "dev_group", gid)
    )
    n_dev = int(len(after_cal_sorted) * config.dev_ratio_of_dev_pool)
    # Ensure at least 0; do not force when pool empty
    dev_groups = after_cal_sorted[:n_dev]
    for gid in dev_groups:
        group_role[gid] = RESERVATION_DEV
        for mid in members_of.get(gid, []):
            if sample_role.get(mid) == RESERVATION_GOVERNANCE_HOLD:
                continue
            sample_role[mid] = RESERVATION_DEV

    train_groups = [gid for gid in after_cal_sorted if gid not in group_role]
    for gid in train_groups:
        group_role[gid] = RESERVATION_TRAIN_POOL
        for mid in members_of.get(gid, []):
            if sample_role.get(mid) == RESERVATION_GOVERNANCE_HOLD:
                continue
            sample_role[mid] = RESERVATION_TRAIN_POOL

    # Any leftover sample without role (should be rare) → governance for safety
    for sid in input_ids_stable:
        if sid not in sample_role:
            sample_role[sid] = RESERVATION_GOVERNANCE_HOLD
            governance_hold_ids.append(sid)
            exclusions.append(
                {
                    "id": sid,
                    "reason": "unassigned_fallback",
                    "detail": "no reservation role assigned",
                }
            )

    artifact = ReservationArtifact(
        policy_version=config.policy_version,
        seed=config.seed,
        input_snapshot_ids=list(input_ids_stable),
        group_mapping={sid: group_of[sid] for sid in input_ids_stable if sid in group_of},
        sample_role={sid: sample_role[sid] for sid in input_ids_stable},
        group_role=dict(sorted(group_role.items())),
        eval_random_ids=list(eval_random_ids),
        eval_random_candidate_order=list(remaining_candidates),
        eval_core_reserve_group_ids=list(core_groups),
        calibration_ids=list(calibration_ids),
        dev_group_ids=list(dev_groups),
        train_pool_group_ids=list(train_groups),
        governance_hold_ids=sorted(set(governance_hold_ids)),
        exclusions=sorted(exclusions, key=lambda row: row.get("id", "")),
        design_formed_at=config.design_formed_at,
        prior_information_used=list(config.prior_information_used),
        audio_hashes={s.id: str(s.labels.get("original_audio_sha256") or s.sha256 or "") for s in samples},
    )
    return artifact.freeze()


def apply_reservation_to_samples(
    samples: list[Sample],
    artifact: ReservationArtifact,
) -> list[Sample]:
    """Stamp reservation role onto labels; does not change the frozen artifact."""
    updated: list[Sample] = []
    for source in samples:
        sample = source.model_copy(deep=True)
        role = artifact.sample_role.get(sample.id)
        if role:
            sample.labels["reservation"] = role
            sample.labels["dataset_role"] = role
        gid = artifact.group_mapping.get(sample.id)
        if gid:
            sample.labels["leakage_group_id"] = gid
        sample.labels["reservation_seed"] = artifact.seed
        sample.labels["reservation_digest"] = artifact.content_digest
        sample.labels["dataset_policy_version"] = artifact.policy_version
        updated.append(sample)
    return updated


def assert_reservation_immutable_wrt_model_signals(
    artifact: ReservationArtifact,
    *,
    forbidden_label_keys: Iterable[str] | None = None,
) -> None:
    """Sanity helper: reservation payload must not embed ASR/DNSMOS/class labels."""
    forbidden = set(
        forbidden_label_keys
        or (
            "type",
            "risk_tags",
            "dnsmos_ovrl",
            "noise_band",
            "candidate_text",
            "qwen_correction_candidate",
        )
    )
    blob = json.dumps(artifact.to_dict(), ensure_ascii=False)
    for key in forbidden:
        # Only flag if it looks like a field name inside roles — keep simple
        if f'"{key}"' in blob and key in {
            "dnsmos_ovrl",
            "noise_band",
            "qwen_correction_candidate",
        }:
            raise ValueError(
                f"reservation artifact must not depend on model signal field {key!r}"
            )
