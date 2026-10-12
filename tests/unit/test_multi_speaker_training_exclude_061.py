"""061 — multi_speaker merge null preservation and hard training exclusion."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from audio_engine.core.sample import Sample  # noqa: E402


def _load_module(name: str, relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    # dataclasses need the module registered before exec
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


annotate = _load_module("annotate_sr_061", "scripts/annotate_semantic_risk_transcripts.py")
assemble = _load_module("assemble_tp_061", "scripts/assemble_training_parquet.py")


def _sample(sid: str = "s1", *, category: str = "semantic_risk", label: str | None = "旧label") -> Sample:
    labels: dict = {"category": category, "type": category}
    if label is not None:
        labels["label"] = label
    return Sample(
        id=sid,
        source_path=f"/tmp/{sid}.wav",
        duration=1.0,
        labels=labels,
        audio={"pcm_wav": f"/tmp/{sid}.wav"},
    )


def test_v2_multi_speaker_preserves_null_gold_and_clears_label() -> None:
    sample = _sample("ms1", label="旧转写不应回退")
    ok = annotate._apply_annotation_to_sample(
        sample,
        {
            "status": "annotated",
            "decision": "multi_speaker",
            "annotation_type": "multi_speaker",
            "gold_text": None,
            "gold_kind": "ambiguous_target",
            "category": "multi_speaker",
            "source_category": "semantic_risk",
            "reviewed_category": "multi_speaker",
            "exclude_from_training": True,
            "training_exclusion_reason": "multi_speaker",
            "label_source": "human_transcript_web",
            "transcript_annotation_protocol": "semantic_risk_web_v2",
            "annotator_id": "a1",
            "annotated_at": "2026-10-09T00:00:00Z",
            "original_audio_sha256": "",
        },
    )
    assert ok is True
    assert sample.labels["gold_text"] is None
    assert sample.labels["category"] == "multi_speaker"
    assert sample.labels["source_category"] == "semantic_risk"
    assert sample.labels["exclude_from_training"] is True
    assert sample.labels.get("label") is None
    # label-master 导出列名是 transcript_annotation_protocol（无 protocol 列）→ 必须识别为 v2
    assert sample.labels["transcript_annotation_protocol"] == "semantic_risk_web_v2"
    assert sample.labels["protocol"] == "semantic_risk_web_v2"


def test_v2_noise_keeps_empty_string() -> None:
    sample = _sample("n1")
    ok = annotate._apply_annotation_to_sample(
        sample,
        {
            "status": "empty",
            "decision": "empty",
            "annotation_type": "human_noise",
            "gold_text": "",
            "gold_kind": "non_speech",
            "category": "environment_noise",
            "source_category": "semantic_risk",
            "reviewed_category": "environment_noise",
            "noise_kind": "human_noise",
            "exclude_from_training": False,
            "label_source": "human_transcript_web",
            "transcript_annotation_protocol": "semantic_risk_web_v2",
            "original_audio_sha256": "",
        },
    )
    assert ok is True
    assert sample.labels["gold_text"] == ""
    assert sample.labels["noise_kind"] == "human_noise"
    assert sample.labels["category"] == "environment_noise"
    assert sample.labels["exclude_from_training"] is False


def test_v1_legacy_empty_coerces_null_to_empty_and_does_not_guess_silence() -> None:
    sample = _sample("legacy1")
    ok = annotate._apply_annotation_to_sample(
        sample,
        {
            "status": "empty",
            "decision": "empty",
            "gold_text": None,
            "gold_kind": "non_speech",
            "label_source": "human_transcript_web",
            "transcript_annotation_protocol": "semantic_risk_web_v1",
            "original_audio_sha256": "",
        },
    )
    assert ok is True
    assert sample.labels["gold_text"] == ""
    assert sample.labels.get("annotation_type") in (None, "")
    assert sample.labels.get("noise_kind") in (None, "")
    assert sample.labels["category"] == "semantic_risk"


def test_hard_exclude_detects_multi_speaker_and_flag() -> None:
    s = _sample("x")
    s.labels["annotation_type"] = "multi_speaker"
    s.labels["category"] = "multi_speaker"
    s.labels["exclude_from_training"] = True
    hard, reason = assemble.is_hard_training_excluded(s)
    assert hard is True
    assert "multi_speaker" in reason


def test_accept_clears_prior_exclusion_on_remerge() -> None:
    sample = _sample("re1")
    sample.labels["exclude_from_training"] = True
    sample.labels["annotation_type"] = "multi_speaker"
    sample.labels["category"] = "multi_speaker"
    ok = annotate._apply_annotation_to_sample(
        sample,
        {
            "status": "annotated",
            "decision": "accept",
            "gold_text": "改回普通听写",
            "gold_kind": "speech",
            "category": "semantic_risk",
            "source_category": "semantic_risk",
            "label_source": "human_transcript_web",
            "transcript_annotation_protocol": "semantic_risk_web_v2",
            "original_audio_sha256": "",
        },
    )
    assert ok is True
    assert sample.labels["gold_text"] == "改回普通听写"
    assert sample.labels["exclude_from_training"] is False
    assert sample.labels["annotation_type"] is None
    assert sample.labels["label"] == "改回普通听写"


def test_category_before_clean_ignores_source_category_override() -> None:
    s = _sample("c1")
    s.labels["source_category"] = "semantic_risk"
    s.labels["reviewed_category"] = "multi_speaker"
    s.labels["category"] = "environment_noise"
    assert assemble.category_before_clean(s) == "multi_speaker"
