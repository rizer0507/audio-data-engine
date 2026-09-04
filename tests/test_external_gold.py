"""Tests for external-gold inject + classify (process-1 split)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import audio_engine.operators  # noqa: F401
from audio_engine.core.operator import OperatorConfig
from audio_engine.core.registry import OperatorRegistry
from audio_engine.core.sample import Sample
from audio_engine.operators.quality.external_gold import (
    EMPTY_GOLD_MARKER,
    classify_external_sample,
)


def _write_gold_xlsx(path: Path, rows: list[dict]) -> Path:
    pd.DataFrame(rows).to_excel(path, index=False)
    return path


def test_inject_external_gold_intersection(tmp_path: Path):
    xlsx = _write_gold_xlsx(
        tmp_path / "gold.xlsx",
        [
            {"id": "a", "label_text_raw": "不需要"},
            {"id": "b", "label_text_raw": "需要贷款"},
            {"id": "orphan", "label_text_raw": "多余"},
        ],
    )
    samples = [
        Sample(
            id="a",
            source_path="a.wav",
            audio={"resampled_16k": "a.wav"},
            transcripts={"qwen": {"text": "不需要"}},
        ),
        Sample(
            id="b",
            source_path="b.wav",
            audio={"resampled_16k": "b.wav"},
            transcripts={"qwen": {"text": "随便说"}},
        ),
        Sample(
            id="c",
            source_path="c.wav",
            audio={"resampled_16k": "c.wav"},
            transcripts={"qwen": {"text": "无金标"}},
        ),
    ]
    result = OperatorRegistry.get("quality.inject_external_gold").run(
        samples,
        OperatorConfig(
            params={"xlsx_path": str(xlsx), "label_col": "label_text_raw"},
            run_dir=tmp_path / "run",
            step_name="inject",
        ),
    )
    assert [s.id for s in result] == ["a", "b"]
    assert result[0].labels["gold_text"] == "不需要"
    assert result[0].labels["gold_source"] == "external"
    assert result[1].labels["label_text_raw"] == "需要贷款"


def test_inject_rejects_duplicate_xlsx_id(tmp_path: Path):
    xlsx = _write_gold_xlsx(
        tmp_path / "dup.xlsx",
        [
            {"id": "a", "label_text_raw": "一"},
            {"id": "a", "label_text_raw": "二"},
        ],
    )
    with pytest.raises(ValueError, match="duplicate id"):
        OperatorRegistry.get("quality.inject_external_gold").run(
            [Sample(id="a", source_path="a.wav")],
            OperatorConfig(params={"xlsx_path": str(xlsx)}),
        )


def test_classify_external_buckets_and_preserves_gold():
    samples = [
        Sample(
            id="match",
            source_path="m.wav",
            labels={"gold_text": "不需要", "label": "不需要", "gold_mode": "external"},
            transcripts={"qwen3-asr": {"text": "不需要"}},
        ),
        Sample(
            id="mismatch",
            source_path="h.wav",
            labels={"gold_text": "不需要", "label": "不需要", "gold_mode": "external"},
            transcripts={"qwen3-asr": {"text": "需要"}},
        ),
        Sample(
            id="both_empty",
            source_path="n.wav",
            labels={"gold_text": "", "label": "", "gold_mode": "external"},
            transcripts={"qwen3-asr": {"text": ""}},
        ),
    ]
    result = OperatorRegistry.get("quality.classify").run(
        samples,
        OperatorConfig(
            params={
                "config_path": "configs/selection/external_gold_v1.yaml",
                "compare_model": "qwen3-asr",
                "hotwords_path": "",
            }
        ),
    )
    by_id = {s.id: s for s in result}
    assert by_id["match"].labels["type"] == "auto_gold"
    assert by_id["match"].labels["gold_text"] == "不需要"
    assert by_id["match"].labels["gold_source"] == "external"
    assert by_id["mismatch"].labels["type"] == "hardcase"
    assert by_id["mismatch"].labels["gold_text"] == "不需要"
    assert by_id["both_empty"].labels["type"] == "noise"
    assert by_id["both_empty"].labels["gold_text"] == EMPTY_GOLD_MARKER


def test_consensus_path_unchanged_without_external_mode():
    """Internal gold path must still write consensus gold_text."""
    sample = Sample(
        id="agree",
        source_path="a.wav",
        transcripts={
            "qwen": {"text": "你好世界"},
            "sensevoice": {"text": "你好世界"},
        },
    )
    result = OperatorRegistry.get("quality.classify").run(
        [sample],
        OperatorConfig(params={"config_path": "configs/selection/zh_asr_v1.yaml"}),
    )
    assert result[0].labels["type"] in {"auto_gold", "consensus_gold"}
    assert result[0].labels["gold_text"] == "你好世界"
    assert result[0].labels.get("gold_mode") != "external"


def test_classify_external_sample_helper_voicemail():
    sample = Sample(
        id="vm",
        source_path="vm.wav",
        labels={"gold_text": "您好这里是语音信箱", "gold_mode": "external"},
        transcripts={"qwen": {"text": "您好这里是语音信箱请留言"}},
    )
    import re

    pattern = re.compile(r"语音信箱")
    out = classify_external_sample(
        sample,
        compare_model="qwen",
        hotwords=frozenset(),
        voicemail_pattern=pattern,
    )
    assert out.type == "voicemail"
    assert out.label  # non-empty preserved


def test_missing_asr_transcript_treated_as_empty_hyp():
    """No ASR text is normal (e.g. noise); do not raise."""
    noise = Sample(
        id="n",
        source_path="n.wav",
        labels={"gold_text": "", "label": "", "gold_mode": "external"},
        transcripts={},
    )
    out = classify_external_sample(
        noise,
        compare_model="qwen3-asr",
        hotwords=frozenset(),
        voicemail_pattern=None,
    )
    assert out.type == "noise"
    assert out.reason == "missing_asr_transcript"
    assert out.label == EMPTY_GOLD_MARKER

    hard = Sample(
        id="h",
        source_path="h.wav",
        labels={"gold_text": "不需要", "label": "不需要", "gold_mode": "external"},
        transcripts={},
    )
    out2 = classify_external_sample(
        hard,
        compare_model="qwen3-asr",
        hotwords=frozenset(),
        voicemail_pattern=None,
    )
    assert out2.type == "hardcase"
    assert out2.reason == "gold_present_asr_missing"
    assert out2.label == "不需要"


def test_external_gold_e2e_to_eval_metric(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """xlsx → inject → classify → eval register/check → aggregate → text_metrics."""
    from typer.testing import CliRunner

    from audio_engine.cli.main import app
    from audio_engine.core.eval_ready import inspect_eval_manifest
    from audio_engine.core.manifest import Manifest
    from audio_engine.core.source_naming import apply_source_name_to_single_pipeline

    repo_root = Path(__file__).resolve().parents[1]
    external_cfg = str(repo_root / "configs/selection/external_gold_v1.yaml")

    monkeypatch.chdir(tmp_path)
    manifests = tmp_path / "datasets" / "manifests"
    manifests.mkdir(parents=True)
    (tmp_path / "data" / "catalog").mkdir(parents=True)
    (tmp_path / "runs").mkdir()

    class _Step:
        def __init__(self, operator: str, params: dict | None = None):
            self.operator = operator
            self.params = params or {}

    # source-name rewrite for classify_external_gold
    asr_path = manifests / "qwen3-asr_asr_mt3000.parquet"
    Manifest(
        [
            Sample(
                id="a",
                source_path="a.wav",
                sha256="a" * 64,
                audio={"resampled_16k": "a.wav"},
                transcripts={"qwen3-asr": {"text": "不需要"}},
            ),
            Sample(
                id="b",
                source_path="b.wav",
                sha256="b" * 64,
                audio={"resampled_16k": "b.wav"},
                transcripts={"qwen3-asr": {"text": "需要"}},
            ),
        ]
    ).save(asr_path)

    overrides = apply_source_name_to_single_pipeline(
        pipeline_name="classify_external_gold",
        steps=[
            _Step("quality.inject_external_gold"),
            _Step("quality.classify"),
        ],
        source_name="mt3000",
        aggregate_base="qwen3-asr",
    )
    assert overrides["input_manifest"].endswith("qwen3-asr_asr_mt3000.parquet")
    assert overrides["output_manifest"].endswith("classified_mt3000.parquet")
    assert overrides["aggregate_base"] == "qwen3-asr"

    xlsx = _write_gold_xlsx(
        tmp_path / "gold.xlsx",
        [
            {"id": "a", "label_text_raw": "不需要"},
            {"id": "b", "label_text_raw": "不需要"},
        ],
    )
    base_samples = list(Manifest.load(asr_path))
    injected = OperatorRegistry.get("quality.inject_external_gold").run(
        base_samples,
        OperatorConfig(
            params={"xlsx_path": str(xlsx), "label_col": "label_text_raw"},
            run_dir=tmp_path / "run_inj",
            step_name="inject",
        ),
    )
    classified = OperatorRegistry.get("quality.classify").run(
        injected,
        OperatorConfig(
            params={
                "config_path": external_cfg,
                "compare_model": "qwen3-asr",
                "hotwords_path": "",
                "voicemail_patterns_path": str(
                    repo_root / "configs/selection/voicemail_patterns_v1.yaml"
                ),
            }
        ),
    )
    assert {s.id: s.labels["type"] for s in classified} == {
        "a": "auto_gold",
        "b": "hardcase",
    }
    assert all(s.labels["gold_source"] == "external" for s in classified)
    assert all("resampled_16k" in s.audio for s in classified)

    classified_path = manifests / "classified_mt3000.parquet"
    Manifest(classified).save(classified_path)
    ready = inspect_eval_manifest(classified_path)
    assert not ready.errors
    assert ready.total == 2
    assert len(ready.with_gold) == 2

    runner = CliRunner()
    reg = runner.invoke(
        app,
        ["eval", "register", str(classified_path), "--name", "eval_mt3000"],
    )
    assert reg.exit_code == 0, reg.output
    check = runner.invoke(app, ["eval", "check", "eval_mt3000"])
    assert check.exit_code == 0, check.output

    # SFT ASR dump (extra id allowed on join side)
    sft_path = manifests / "qwen-sft-e10_asr_eval_mt3000.parquet"
    Manifest(
        [
            Sample(
                id="a",
                source_path="a.wav",
                sha256="a" * 64,
                transcripts={"qwen-sft-e10": {"text": "不需要"}},
            ),
            Sample(
                id="b",
                source_path="b.wav",
                sha256="b" * 64,
                transcripts={"qwen-sft-e10": {"text": "不需要"}},
            ),
            Sample(
                id="extra",
                source_path="extra.wav",
                sha256="e" * 64,
                transcripts={"qwen-sft-e10": {"text": "多余"}},
            ),
        ]
    ).save(sft_path)

    eval_samples = list(Manifest.load(manifests / "eval_mt3000.parquet"))
    aggregated = OperatorRegistry.get("quality.aggregate_manifests").run(
        eval_samples,
        OperatorConfig(
            params={
                "id_policy": "left",
                "manifests": [{"model": "qwen-sft-e10", "path": str(sft_path)}],
            },
            run_dir=tmp_path / "run_agg",
            step_name="aggregate",
        ),
    )
    assert aggregated[0].get_transcript_text("qwen-sft-e10") == "不需要"
    assert aggregated[1].labels["gold_text"] == "不需要"

    config_path = tmp_path / "model_eval.yaml"
    config_path.write_text(
        "vs_gold:\n  purpose: model_evaluation\n  metrics: [cer]\n",
        encoding="utf-8",
    )
    norm = tmp_path / "norm.yaml"
    norm.write_text("name: test\n", encoding="utf-8")
    scored = []
    for sample in aggregated:
        out = OperatorRegistry.get("quality.text_metrics").process(
            sample,
            OperatorConfig(
                params={
                    "config_path": str(config_path),
                    "normalization_path": str(norm),
                    "eval_hypotheses": ["qwen3-asr", "qwen-sft-e10"],
                }
            ),
        )
        scored.append(out.sample)
    assert scored[0].quality["qwen-sft-e10_cer"] == 0.0
    assert scored[1].quality["qwen-sft-e10_cer"] == 0.0
    assert "qwen3-asr_cer" in scored[1].quality
