"""Shared formal-gold validation used at audit, dataset and eval boundaries."""
from audio_engine.core.annotation_v3.contract import may_pass_formal_gold
from audio_engine.core.sample import Sample


def has_formal_gold_evidence(sample: Sample, *, require_dual: bool) -> bool:
    labels = sample.labels
    if labels.get("review_validation_failed") or labels.get("spot_check_layer_blocked"):
        return False
    kind, text = labels.get("gold_kind"), labels.get("gold_text")
    state = str(labels.get("annotation_state") or "")
    external = labels.get("label_source") == "trusted_external"
    if not may_pass_formal_gold(kind, text, "second_review" if external else state):
        return False
    if labels.get("speech_scope") not in {"target", "none"}:
        return False
    if kind == "speech" and labels["speech_scope"] != "target":
        return False
    if kind == "non_speech" and (labels["speech_scope"] != "none" or not labels.get("audio_event_tags")):
        return False
    if labels.get("human_noise") not in {"clean", "moderate", "noisy", "unknown"}:
        return False
    if str(labels.get("human_crosstalk")).lower() not in {"true", "false", "unknown"}:
        return False
    if labels.get("human_semantic") not in {"positive", "negative", "neutral", "mixed", "unknown", "not_applicable"}:
        return False
    if kind == "speech" and labels["human_semantic"] == "not_applicable":
        return False
    if kind == "non_speech" and labels["human_semantic"] != "not_applicable":
        return False
    if labels.get("label_source") == "trusted_external":
        return bool(labels.get("external_gold_artifact_id"))
    if labels.get("is_human_verified") is not True or not labels.get("annotator_id"):
        return False
    dual = require_dual or kind == "non_speech" or labels.get("review_priority") == "P0"
    if dual and state not in {"second_review", "adjudicated"}:
        return False
    if state in {"second_review", "adjudicated"}:
        if not labels.get("reviewer_id") or labels["reviewer_id"] == labels["annotator_id"]:
            return False
    if state == "adjudicated":
        return bool(labels.get("adjudicator_id")) and labels["adjudicator_id"] not in {labels["annotator_id"], labels["reviewer_id"]}
    return True
