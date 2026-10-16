"""Tagging-mode downstream consumption strategies (044 §9)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from audio_engine.core.sample import Sample


@dataclass
class TagConsumePolicy:
    policy_version: str
    routing_mode: str = "tagging"
    include_any: list[str] = field(default_factory=list)
    include_all: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)
    unknown_policy: str = "keep_with_flag"
    not_evaluated_policy: str = "exclude"
    priority: list[dict[str, Any]] = field(default_factory=list)
    dual_review_trigger_any: list[str] = field(default_factory=list)
    source_path: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy_version": self.policy_version,
            "routing_mode": self.routing_mode,
            "include_any": list(self.include_any),
            "include_all": list(self.include_all),
            "exclude": list(self.exclude),
            "unknown_policy": self.unknown_policy,
            "not_evaluated_policy": self.not_evaluated_policy,
            "priority": list(self.priority),
            "dual_review_trigger_any": list(self.dual_review_trigger_any),
            "source_path": self.source_path,
        }


def load_tag_consume_policy(path: str | Path) -> TagConsumePolicy:
    p = Path(path)
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"consume policy must be mapping: {p}")
    if str(raw.get("routing_mode") or "tagging") != "tagging":
        raise ValueError(f"{p}: routing_mode must be tagging for tag consume policies")
    cand = raw.get("candidate") if isinstance(raw.get("candidate"), dict) else {}
    dual = raw.get("dual_review") if isinstance(raw.get("dual_review"), dict) else {}
    return TagConsumePolicy(
        policy_version=str(raw.get("policy_version") or p.stem),
        routing_mode="tagging",
        include_any=[str(x) for x in (cand.get("include_any") or [])],
        include_all=[str(x) for x in (cand.get("include_all") or [])],
        exclude=[str(x) for x in (cand.get("exclude") or [])],
        unknown_policy=str(cand.get("unknown_policy") or "keep_with_flag"),
        not_evaluated_policy=str(cand.get("not_evaluated_policy") or "exclude"),
        priority=list(raw.get("priority") or []),
        dual_review_trigger_any=[str(x) for x in (dual.get("trigger_any") or [])],
        source_path=str(p.as_posix()),
    )


def sample_tags(sample: Sample) -> list[str]:
    labels = sample.labels if isinstance(sample.labels, dict) else {}
    tags = labels.get("tags") or []
    if isinstance(tags, str):
        return [tags] if tags.strip() else []
    return [str(t) for t in tags if str(t).strip()]


def sample_eval_status(sample: Sample) -> str:
    labels = sample.labels if isinstance(sample.labels, dict) else {}
    tagging = labels.get("tagging") if isinstance(labels.get("tagging"), dict) else {}
    return str(tagging.get("eval_status") or "").strip().lower()


def matches_consume_policy(sample: Sample, policy: TagConsumePolicy) -> tuple[bool, dict[str, Any]]:
    """Decide whether a sample enters the tagging-mode candidate pool.

    Legacy auto fields (category/queue/...) are intentionally ignored.
    """
    tags = set(sample_tags(sample))
    status = sample_eval_status(sample)
    explanation: dict[str, Any] = {
        "tags": sorted(tags),
        "eval_status": status,
        "policy_version": policy.policy_version,
        "routing_mode": "tagging",
    }
    if status in {"not_evaluated", ""} and policy.not_evaluated_policy == "exclude":
        explanation["reason"] = "not_evaluated"
        return False, explanation
    if status == "unresolved" and policy.unknown_policy == "exclude":
        explanation["reason"] = "unknown_excluded"
        return False, explanation
    if policy.exclude and tags.intersection(policy.exclude):
        explanation["reason"] = "excluded_tag"
        return False, explanation
    if policy.include_all:
        if not set(policy.include_all).issubset(tags):
            explanation["reason"] = "include_all_miss"
            return False, explanation
    if policy.include_any:
        hit = sorted(tags.intersection(policy.include_any))
        if not hit:
            explanation["reason"] = "include_any_miss"
            return False, explanation
        explanation["matched_include_any"] = hit
    explanation["reason"] = "matched"
    return True, explanation


def review_priority(sample: Sample, policy: TagConsumePolicy) -> int:
    tags = set(sample_tags(sample))
    best = 0
    for item in policy.priority:
        tid = str(item.get("tag_id") or "")
        if tid and tid in tags:
            best = max(best, int(item.get("weight") or 0))
    return best


def needs_dual_review(sample: Sample, policy: TagConsumePolicy) -> bool:
    tags = set(sample_tags(sample))
    return bool(tags.intersection(policy.dual_review_trigger_any))


def filter_tagging_candidates(
    samples: list[Sample],
    policy: TagConsumePolicy,
) -> list[tuple[Sample, dict[str, Any]]]:
    out: list[tuple[Sample, dict[str, Any]]] = []
    for sample in samples:
        ok, expl = matches_consume_policy(sample, policy)
        if not ok:
            continue
        expl["priority"] = review_priority(sample, policy)
        expl["dual_review"] = needs_dual_review(sample, policy)
        # Explicitly ignore legacy auto fields for isolation.
        labels = sample.labels if isinstance(sample.labels, dict) else {}
        expl["ignored_legacy_auto"] = {
            k: labels.get(k)
            for k in (
                "category",
                "type",
                "classification_bucket",
                "risk_tag",
                "risk_tags",
                "status",
                "label_tier",
                "review_queue",
                "review_priority",
            )
            if k in labels
        }
        out.append((sample, expl))
    out.sort(key=lambda item: (-int(item[1].get("priority") or 0), item[0].id))
    return out
