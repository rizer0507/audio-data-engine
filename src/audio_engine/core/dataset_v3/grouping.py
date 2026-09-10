"""Leakage-group construction for dataset_policy_v3 (stage A)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Iterable

from audio_engine.core.sample import Sample
from audio_engine.core.selection_v3.types import (
    GOVERNANCE_MISSING_GROUP_META,
    GOVERNANCE_NEAR_DUP_UNCERTAIN,
)


@dataclass
class GroupingConfig:
    """How to resolve grouping fields from a source index (never guess filename prefixes)."""

    call_id_fields: list[str] = field(
        default_factory=lambda: ["call_id", "conversation_id"]
    )
    speaker_id_fields: list[str] = field(default_factory=list)
    source_audio_id_fields: list[str] = field(
        default_factory=lambda: ["source_audio_id"]
    )
    file_hash_fields: list[str] = field(
        default_factory=lambda: ["original_audio_sha256", "sha256"]
    )
    pcm_hash_fields: list[str] = field(
        default_factory=lambda: ["pcm_sha256", "normalized_pcm_sha256"]
    )
    confirmed_near_duplicate_field: str = "near_duplicate_group_id"
    uncertain_near_duplicate_field: str = "near_duplicate_uncertain_id"
    source_index_fields: list[str] = field(
        default_factory=lambda: ["source_snapshot_id", "source_audio_id", "call_id", "conversation_id"]
    )
    require_source_index: bool = True
    # When True, speaker links only apply if labels.speaker_id_reliable is true
    require_reliable_speaker: bool = True

    @classmethod
    def from_params(cls, params: dict[str, Any] | None) -> GroupingConfig:
        raw = params or {}
        return cls(
            call_id_fields=[
                str(x) for x in (raw.get("call_id_fields") or ["call_id", "conversation_id"])
            ],
            speaker_id_fields=[str(x) for x in (raw.get("speaker_id_fields") or [])],
            source_audio_id_fields=[
                str(x)
                for x in (raw.get("source_audio_id_fields") or ["source_audio_id"])
            ],
            file_hash_fields=[
                str(x)
                for x in (
                    raw.get("file_hash_fields")
                    or ["original_audio_sha256", "sha256"]
                )
            ],
            pcm_hash_fields=[
                str(x)
                for x in (
                    raw.get("pcm_hash_fields") or ["pcm_sha256", "normalized_pcm_sha256"]
                )
            ],
            confirmed_near_duplicate_field=str(
                raw.get("confirmed_near_duplicate_field") or "near_duplicate_group_id"
            ),
            uncertain_near_duplicate_field=str(
                raw.get("uncertain_near_duplicate_field")
                or "near_duplicate_uncertain_id"
            ),
            source_index_fields=[
                str(x)
                for x in (
                    raw.get("source_index_fields")
                    or [
                        "source_snapshot_id",
                        "source_audio_id",
                        "call_id",
                        "conversation_id",
                    ]
                )
            ],
            require_source_index=bool(raw.get("require_source_index", True)),
            require_reliable_speaker=bool(raw.get("require_reliable_speaker", True)),
        )


def _label_or_quality(sample: Sample, key: str) -> Any:
    if key in sample.labels and sample.labels[key] not in (None, ""):
        return sample.labels[key]
    if key in sample.quality and sample.quality[key] not in (None, ""):
        return sample.quality[key]
    if key == "sha256" and sample.sha256:
        return sample.sha256
    return None


def _first_value(sample: Sample, fields: list[str]) -> str | None:
    for key in fields:
        value = _label_or_quality(sample, key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def has_source_index(sample: Sample, config: GroupingConfig) -> bool:
    """A snapshot name alone is not evidence of a source/call relationship."""
    for key in set(config.call_id_fields + config.source_audio_id_fields):
        value = _label_or_quality(sample, key)
        if value is not None and str(value).strip():
            return True
    return False


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.rank: dict[str, int] = {}

    def add(self, item: str) -> None:
        if item not in self.parent:
            self.parent[item] = item
            self.rank[item] = 0

    def find(self, item: str) -> str:
        self.add(item)
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a == b:
            return
        if self.rank[a] < self.rank[b]:
            self.parent[a] = b
        elif self.rank[a] > self.rank[b]:
            self.parent[b] = a
        else:
            self.parent[b] = a
            self.rank[a] += 1

    def components(self) -> dict[str, list[str]]:
        buckets: dict[str, list[str]] = {}
        for item in self.parent:
            root = self.find(item)
            buckets.setdefault(root, []).append(item)
        return buckets


@dataclass
class GroupingResult:
    leakage_group_id: dict[str, str]  # sample_id → group id
    group_members: dict[str, list[str]]  # group id → sample ids
    governance_flags: dict[str, list[str]]  # sample_id → flags
    link_evidence: dict[str, list[dict[str, str]]]  # sample_id → evidence rows
    missing_group_meta_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_count": len(self.group_members),
            "sample_count": len(self.leakage_group_id),
            "missing_group_meta_count": len(self.missing_group_meta_ids),
            "missing_group_meta_ids_preview": self.missing_group_meta_ids[:50],
            "group_size_histogram": _histogram(
                [len(v) for v in self.group_members.values()]
            ),
        }


def _histogram(sizes: list[int]) -> dict[str, int]:
    hist: dict[str, int] = {}
    for size in sizes:
        key = str(size)
        hist[key] = hist.get(key, 0) + 1
    return dict(sorted(hist.items(), key=lambda kv: int(kv[0])))


def _stable_group_id(member_ids: list[str]) -> str:
    ordered = sorted(member_ids)
    digest_seed = "|".join(ordered)
    # Short stable id; full member list is stored separately.
    short = hashlib.sha256(digest_seed.encode("utf-8")).hexdigest()[:16]
    return f"lg_{short}"


def build_leakage_groups(
    samples: Iterable[Sample],
    config: GroupingConfig | None = None,
) -> GroupingResult:
    """Build undirected leakage graph and assign connected-component ids.

    Edges:
    - same call_id / conversation_id
    - same reliable speaker/customer id (when configured)
    - same source_audio_id
    - same file hash / PCM hash
    - confirmed near-duplicate group id

    Uncertain near-duplicates are flagged for isolation, not auto-linked.
    Missing source/call mapping → governance_hold candidates.
    """
    cfg = config or GroupingConfig()
    sample_list = list(samples)
    uf = _UnionFind()
    evidence: dict[str, list[dict[str, str]]] = {s.id: [] for s in sample_list}
    flags: dict[str, list[str]] = {s.id: [] for s in sample_list}
    missing_meta: list[str] = []

    # Index buckets for linking
    buckets: dict[str, dict[str, list[str]]] = {
        "call": {},
        "speaker": {},
        "source_audio": {},
        "file_hash": {},
        "pcm_hash": {},
        "near_dup": {},
    }

    for sample in sample_list:
        uf.add(sample.id)
        if cfg.require_source_index and not has_source_index(sample, cfg):
            flags[sample.id].append(GOVERNANCE_MISSING_GROUP_META)
            missing_meta.append(sample.id)

        call = _first_value(sample, cfg.call_id_fields)
        if call:
            buckets["call"].setdefault(f"call:{call}", []).append(sample.id)
            evidence[sample.id].append({"relation": "call_id", "value": call})

        if cfg.speaker_id_fields:
            reliable = True
            if cfg.require_reliable_speaker:
                raw = _label_or_quality(sample, "speaker_id_reliable")
                reliable = raw is True or str(raw).strip().lower() in {
                    "1",
                    "true",
                    "yes",
                    "y",
                }
            if reliable:
                speaker = _first_value(sample, cfg.speaker_id_fields)
                if speaker:
                    buckets["speaker"].setdefault(f"spk:{speaker}", []).append(sample.id)
                    evidence[sample.id].append(
                        {"relation": "speaker_id", "value": speaker}
                    )

        source_audio = _first_value(sample, cfg.source_audio_id_fields)
        if source_audio:
            buckets["source_audio"].setdefault(f"src:{source_audio}", []).append(
                sample.id
            )
            evidence[sample.id].append(
                {"relation": "source_audio_id", "value": source_audio}
            )

        file_hash = _first_value(sample, cfg.file_hash_fields)
        if file_hash:
            buckets["file_hash"].setdefault(f"file:{file_hash}", []).append(sample.id)
            evidence[sample.id].append({"relation": "file_hash", "value": file_hash})

        pcm_hash = _first_value(sample, cfg.pcm_hash_fields)
        if pcm_hash:
            buckets["pcm_hash"].setdefault(f"pcm:{pcm_hash}", []).append(sample.id)
            evidence[sample.id].append({"relation": "pcm_hash", "value": pcm_hash})

        near = _label_or_quality(sample, cfg.confirmed_near_duplicate_field)
        if near is not None and str(near).strip():
            key = f"nd:{str(near).strip()}"
            buckets["near_dup"].setdefault(key, []).append(sample.id)
            evidence[sample.id].append(
                {
                    "relation": "confirmed_near_duplicate",
                    "value": str(near).strip(),
                    "algorithm": str(
                        _label_or_quality(sample, "near_duplicate_algorithm") or ""
                    ),
                    "threshold": str(
                        _label_or_quality(sample, "near_duplicate_threshold") or ""
                    ),
                }
            )

        uncertain = _label_or_quality(sample, cfg.uncertain_near_duplicate_field)
        if uncertain is not None and str(uncertain).strip():
            flags[sample.id].append(GOVERNANCE_NEAR_DUP_UNCERTAIN)

    # Every supplied identity is an edge, not only the first alias. This is
    # necessary for transitive links across mixed source schemas.
    for sample in sample_list:
        for kind, fields in (("call", cfg.call_id_fields), ("source_audio", cfg.source_audio_id_fields),
                             ("file_hash", cfg.file_hash_fields), ("pcm_hash", cfg.pcm_hash_fields)):
            prefix = {"call": "call", "source_audio": "src", "file_hash": "file", "pcm_hash": "pcm"}[kind]
            for name in fields:
                value = _label_or_quality(sample, name)
                if value is not None and str(value).strip():
                    buckets[kind].setdefault(f"{prefix}:{str(value).strip()}", []).append(sample.id)
    for kind_buckets in buckets.values():
        for members in kind_buckets.values():
            if len(members) < 2:
                continue
            head = members[0]
            for other in members[1:]:
                uf.union(head, other)

    components = uf.components()
    # Remap to stable leakage_group_id
    leakage_group_id: dict[str, str] = {}
    group_members: dict[str, list[str]] = {}
    for members in components.values():
        gid = _stable_group_id(members)
        group_members[gid] = sorted(members)
        for sid in members:
            leakage_group_id[sid] = gid

    # Singleton samples not in uf somehow — should not happen
    for sample in sample_list:
        if sample.id not in leakage_group_id:
            gid = _stable_group_id([sample.id])
            leakage_group_id[sample.id] = gid
            group_members[gid] = [sample.id]

    return GroupingResult(
        leakage_group_id=leakage_group_id,
        group_members=group_members,
        governance_flags=flags,
        link_evidence=evidence,
        missing_group_meta_ids=sorted(set(missing_meta)),
    )


def apply_grouping_to_samples(
    samples: list[Sample],
    result: GroupingResult,
) -> list[Sample]:
    """Write leakage_group_id and governance flags onto sample labels."""
    updated: list[Sample] = []
    for source in samples:
        sample = source.model_copy(deep=True)
        gid = result.leakage_group_id.get(sample.id)
        if gid:
            sample.labels["leakage_group_id"] = gid
        flags = result.governance_flags.get(sample.id) or []
        if flags:
            existing = sample.labels.get("governance_flags") or []
            if not isinstance(existing, list):
                existing = [existing]
            merged = list(dict.fromkeys([*existing, *flags]))
            sample.labels["governance_flags"] = merged
            if GOVERNANCE_MISSING_GROUP_META in merged:
                sample.labels["governance_isolation"] = True
            if GOVERNANCE_NEAR_DUP_UNCERTAIN in merged:
                sample.labels["governance_isolation"] = True
        evidence = result.link_evidence.get(sample.id) or []
        if evidence:
            sample.labels["leakage_link_evidence"] = evidence
        # Keep duplicate_group_id aligned when only hash-based exact dup exists
        if not sample.labels.get("duplicate_group_id"):
            file_hash = _first_value(
                sample, ["original_audio_sha256", "sha256"]
            )
            if file_hash:
                sample.labels["duplicate_group_id"] = f"dup_{file_hash[:16]}"
        updated.append(sample)
    return updated
