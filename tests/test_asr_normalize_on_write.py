"""048: ASR batch pipelines normalize transcripts on write."""

from __future__ import annotations

from pathlib import Path

import yaml

from audio_engine.core.operator import OperatorConfig
from audio_engine.core.pipeline import PipelineConfig
from audio_engine.core.registry import OperatorRegistry
from audio_engine.core.sample import Sample
from audio_engine.core.transcript_reconcile import blank_applies_to_model
import audio_engine.operators  # noqa: F401


_HOTWORD_DUMP = "没有，暂时不用，不需要谢谢，不可以，啊不用，我不需要，不要，不用，不需要。"


def _normalize(sample: Sample, params: dict, *, cache_dir: Path | None = None) -> Sample:
    operator = OperatorRegistry.get("quality.normalize_transcripts")
    config = OperatorConfig(
        params=params,
        cache_dir=cache_dir or Path("data/cache"),
        force=True,
    )
    return operator.process(sample, config).sample


def test_blank_applies_to_family_aliases():
    models = {"qwen", "qwen1", "qwen2"}
    assert blank_applies_to_model("qwen", models)
    assert blank_applies_to_model("qwen1", models)
    assert blank_applies_to_model("qwen3", models)
    assert blank_applies_to_model("qwen-sft-epoch10", models)
    assert not blank_applies_to_model("glm1", models)
    assert not blank_applies_to_model("qwenthing", models)


def test_qwen_on_write_strips_punct_blanks_dump_keeps_short(tmp_path):
    params = {
        "models": ["qwen1"],
        "keep_raw": True,
        "normalization_path": "configs/normalization/zh_asr_v1.yaml",
        "blank_exact_hotwords_path": "configs/normalization/blank_exact_qwen_v1.yaml",
    }
    blanked = _normalize(
        Sample(
            id="hot",
            source_path="/tmp/a.wav",
            sha256="a" * 64,
            transcripts={"qwen1": {"text": _HOTWORD_DUMP, "model": "qwen3-asr"}},
        ),
        params,
        cache_dir=tmp_path / "c1",
    )
    assert blanked.transcripts["qwen1"]["text"] == ""
    assert blanked.transcripts["qwen1"]["extra"]["raw_text"] == _HOTWORD_DUMP
    assert blanked.transcripts["qwen1"]["extra"]["blanked_exact_hotword"] is True

    kept = _normalize(
        Sample(
            id="short",
            source_path="/tmp/b.wav",
            sha256="b" * 64,
            transcripts={"qwen1": {"text": "不需要！"}},
        ),
        params,
        cache_dir=tmp_path / "c2",
    )
    assert kept.transcripts["qwen1"]["text"] == "不需要"
    assert kept.transcripts["qwen1"]["extra"]["raw_text"] == "不需要！"


def test_sensevoice_on_write_strips_all_control_tags(tmp_path):
    sample = Sample(
        id="sv",
        source_path="/tmp/s.wav",
        sha256="c" * 64,
        transcripts={
            "sensevoice": {
                "text": "<|zh|><|NEUTRAL|><|Speech|><|withitn|>您好，世界！",
                "model": "sensevoice-small",
                "extra": {"emotion": "NEUTRAL"},
            }
        },
    )
    out = _normalize(
        sample,
        {
            "models": ["sensevoice"],
            "keep_raw": True,
            "normalization_path": "configs/normalization/zh_asr_v1.yaml",
        },
        cache_dir=tmp_path / "c",
    )
    entry = out.transcripts["sensevoice"]
    assert entry["text"] == "您好世界"
    assert "<|" not in entry["text"]
    assert entry["extra"]["raw_text"] == (
        "<|zh|><|NEUTRAL|><|Speech|><|withitn|>您好，世界！"
    )
    assert entry["extra"]["emotion"] == "NEUTRAL"


def test_glm_on_write_normalization_idempotent(tmp_path):
    params = {
        "models": ["glm1"],
        "keep_raw": True,
        "normalization_path": "configs/normalization/zh_asr_v1.yaml",
    }
    first = _normalize(
        Sample(
            id="g",
            source_path="/tmp/g.wav",
            sha256="d" * 64,
            transcripts={"glm1": {"text": "你好，世界！"}},
        ),
        params,
        cache_dir=tmp_path / "c1",
    )
    assert first.transcripts["glm1"]["text"] == "你好世界"
    raw = first.transcripts["glm1"]["extra"]["raw_text"]
    second = _normalize(first, params, cache_dir=tmp_path / "c2")
    assert second.transcripts["glm1"]["text"] == "你好世界"
    assert second.transcripts["glm1"]["extra"]["raw_text"] == raw


def test_normalize_cache_invalidates_when_blank_yaml_changes(tmp_path):
    blank_path = tmp_path / "blank.yaml"
    blank_path.write_text(
        yaml.safe_dump(
            {
                "blank_exact_hotwords": {
                    "models": ["qwen"],
                    "hotwords": ["你好世界"],
                }
            }
        ),
        encoding="utf-8",
    )
    operator = OperatorRegistry.get("quality.normalize_transcripts")
    sample = Sample(
        id="cache",
        source_path="/tmp/x.wav",
        sha256="e" * 64,
        transcripts={"qwen": {"text": "你好，世界！"}},
    )
    params = {
        "models": ["qwen"],
        "keep_raw": True,
        "blank_exact_hotwords_path": str(blank_path),
    }
    config = OperatorConfig(params=params, cache_dir=tmp_path / "cache", force=False)
    key1 = operator.compute_cache_key(sample, config)
    blank_path.write_text(
        yaml.safe_dump(
            {
                "blank_exact_hotwords": {
                    "models": ["qwen"],
                    "hotwords": ["完全不同的短语"],
                }
            }
        ),
        encoding="utf-8",
    )
    key2 = operator.compute_cache_key(sample, config)
    assert key1 != key2


def test_asr_batch_yaml_declares_normalize_step():
    for path, model, blank in (
        ("pipelines/qwen_asr_batch.yaml", "qwen", True),
        ("pipelines/sensevoice_asr_batch.yaml", "sensevoice", False),
        ("pipelines/glm_asr_batch.yaml", "glm", False),
    ):
        cfg = PipelineConfig.from_yaml(Path(path))
        ops = [step.operator for step in cfg.steps]
        assert "quality.normalize_transcripts" in ops
        asr_idx = next(i for i, op in enumerate(ops) if str(op).startswith("asr."))
        norm_idx = ops.index("quality.normalize_transcripts")
        assert norm_idx > asr_idx
        norm = cfg.steps[norm_idx]
        assert norm.params.get("keep_raw") is True
        assert norm.params.get("models") == [model]
        assert "zh_asr_v1.yaml" in str(norm.params.get("normalization_path", ""))
        has_blank = bool(norm.params.get("blank_exact_hotwords_path"))
        assert has_blank is blank


def test_apply_asr_run_scopes_normalize_models():
    from audio_engine.cli.main import _apply_asr_run_transcript_key

    cfg = PipelineConfig.from_yaml(Path("pipelines/qwen_asr_batch.yaml"))
    _apply_asr_run_transcript_key(cfg, "qwen1")
    asr = next(s for s in cfg.steps if str(s.operator).startswith("asr."))
    norm = next(s for s in cfg.steps if s.operator == "quality.normalize_transcripts")
    assert asr.params.get("transcript_key") == "qwen1"
    assert norm.params.get("models") == ["qwen1"]


def test_failed_asr_row_without_transcript_is_not_forged(tmp_path):
    """ASR failure leaves no transcript; normalize must not invent success text."""
    sample = Sample(
        id="fail",
        source_path="/tmp/f.wav",
        sha256="f" * 64,
        transcripts={},
    )
    sample.mark_failed("asr.qwen_batch", "boom")
    out = _normalize(
        sample,
        {
            "models": ["qwen"],
            "keep_raw": True,
            "blank_exact_hotwords_path": "configs/normalization/blank_exact_qwen_v1.yaml",
        },
        cache_dir=tmp_path / "c",
    )
    assert "qwen" not in out.transcripts
    assert out.status.get("asr.qwen_batch") == "failed"
