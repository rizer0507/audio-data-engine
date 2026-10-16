"""044: tagging consume policy ignores legacy auto fields."""

from __future__ import annotations

from pathlib import Path

from audio_engine.core.sample import Sample
from audio_engine.core.tagging.consume import (
    filter_tagging_candidates,
    load_tag_consume_policy,
    matches_consume_policy,
)

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "configs/tagging/consume/free_tags_v2_review.yaml"


def _sample(tags: list[str], *, category: str | None = "human_noise", eval_status: str = "tagged") -> Sample:
    s = Sample(id="s1", source_path="a.wav", sha256="h1")
    labels = {
        "tags": list(tags),
        "tagging": {"eval_status": eval_status, "policy_version": "free_tags_v2"},
    }
    if category is not None:
        labels["category"] = category
        labels["review_queue"] = "B"
        labels["type"] = "legacy_auto"
    s.labels = labels
    return s


def test_consume_policy_loads():
    policy = load_tag_consume_policy(POLICY)
    assert policy.routing_mode == "tagging"
    assert "semantic_risk_text_disagreement" in policy.include_any


def test_legacy_auto_fields_do_not_change_candidate_set():
    policy = load_tag_consume_policy(POLICY)
    with_legacy = _sample(["semantic_risk_text_disagreement"], category="human_noise")
    without_legacy = _sample(["semantic_risk_text_disagreement"], category=None)
    # drop all legacy keys
    without_legacy.labels = {
        "tags": ["semantic_risk_text_disagreement"],
        "tagging": {"eval_status": "tagged"},
    }
    ok1, e1 = matches_consume_policy(with_legacy, policy)
    ok2, e2 = matches_consume_policy(without_legacy, policy)
    assert ok1 and ok2
    assert e1["reason"] == e2["reason"] == "matched"
    cands1 = filter_tagging_candidates([with_legacy], policy)
    cands2 = filter_tagging_candidates([without_legacy], policy)
    assert len(cands1) == len(cands2) == 1
    assert cands1[0][1]["priority"] == cands2[0][1]["priority"]
    assert cands1[0][1]["dual_review"] is True
