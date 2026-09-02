import pytest
import pandas as pd
from typer.testing import CliRunner

import audio_engine.operators  # noqa: F401
from audio_engine.cli.main import app
from audio_engine.core.manifest import Manifest
from audio_engine.core.operator import OperatorConfig
from audio_engine.core.registry import OperatorRegistry
from audio_engine.core.sample import Sample


def test_classify_uses_ordered_rules_and_auditable_default():
    samples = [
        Sample(
            id="good",
            source_path="g.wav",
            quality={"cer": 0.05},
            transcripts={"qwen": {"text": "金标"}},
        ),
        Sample(id="hard", source_path="h.wav", quality={"cer": 0.6}),
        Sample(id="middle", source_path="m.wav", quality={"cer": 0.2}),
    ]
    result = OperatorRegistry.get("quality.classify").run(
        samples,
        OperatorConfig(
            params={
                "policy_version": "selection_v1",
                "gold_source_model": "qwen",
                "rules": [
                    {"expr": "quality_cer <= 0.1", "bucket": "auto_gold", "reason": "agree"},
                    {"expr": "quality_cer >= 0.5", "bucket": "hardcase", "reason": "disagree"},
                ],
            }
        ),
    )
    assert [sample.labels["classification_bucket"] for sample in result] == [
        "auto_gold",
        "hardcase",
        "review_queue",
    ]
    assert result[2].labels["classification_reason_codes"] == ["default"]
    assert result[0].labels["annotation_state"] == "auto_accepted"
    assert result[0].labels["gold_text"] == "金标"


def test_group_split_is_deterministic_and_never_leaks_groups():
    samples = [
        Sample(id=f"s{i}", source_path=f"{i}.wav", labels={"speaker_id": f"p{i // 2}"})
        for i in range(20)
    ]
    operator = OperatorRegistry.get("quality.split_dataset")
    config = OperatorConfig(
        params={"group_key": "speaker_id", "seed": 42, "ratios": {"train": 0.7, "test": 0.3}}
    )
    first = operator.run(samples, config)
    second = operator.run(samples, config)
    assert [s.labels["split"] for s in first] == [s.labels["split"] for s in second]
    group_splits = {}
    for sample in first:
        group = sample.labels["speaker_id"]
        group_splits.setdefault(group, set()).add(sample.labels["split"])
    assert all(len(splits) == 1 for splits in group_splits.values())


def test_split_rejects_missing_group():
    with pytest.raises(ValueError, match="missing split group"):
        OperatorRegistry.get("quality.split_dataset").run(
            [Sample(id="a", source_path="a.wav")], OperatorConfig(params={})
        )


def test_review_import_and_release_build_end_to_end(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "classified.parquet"
    Manifest(
        [
            Sample(
                id=f"s{i}",
                source_path=f"{i}.wav",
                sha256=f"{i:064x}",
                labels={
                    "classification_bucket": "review_queue" if i % 2 else "hardcase",
                    "speaker_id": f"p{i}",
                },
                transcripts={"qwen": {"text": f"文本{i}"}},
            )
            for i in range(10)
        ]
        + [
            Sample(
                id="noise",
                source_path="noise.wav",
                sha256="f" * 64,
                labels={"classification_bucket": "noise", "speaker_id": "noise"},
            )
        ]
    ).save(source)
    runner = CliRunner()
    review = tmp_path / "review.xlsx"
    exported = runner.invoke(
        app,
        [
            "review",
            "export",
            str(source),
            "--output",
            str(review),
            "--revision",
            "r1",
            "--bucket",
            "review_queue",
            "--bucket",
            "hardcase",
        ],
    )
    assert exported.exit_code == 0, exported.output
    frame = pd.read_excel(review, dtype=str).fillna("")
    frame["decision"] = "accepted"
    frame["gold_text"] = frame["qwen_text"]
    frame.to_excel(review, index=False)
    reviewed = tmp_path / "reviewed.parquet"
    imported = runner.invoke(
        app,
        [
            "review",
            "import",
            str(source),
            "--input",
            str(review),
            "--output",
            str(reviewed),
            "--revision",
            "r1",
            "--bucket",
            "review_queue",
            "--bucket",
            "hardcase",
        ],
    )
    assert imported.exit_code == 0, imported.output
    reviewed_samples = Manifest.load(reviewed).samples
    assert sum(s.labels.get("annotation_state") == "human_accepted" for s in reviewed_samples) == 10
    assert (
        next(s for s in reviewed_samples if s.id == "noise").labels.get("annotation_state") is None
    )

    built = runner.invoke(
        app,
        [
            "release",
            "build",
            str(reviewed),
            "--id",
            "ds_v1",
            "--policy-version",
            "p1",
            "--normalization-version",
            "n1",
            "--gold-revision",
            "r1",
            "--group-key",
            "speaker_id",
        ],
    )
    assert built.exit_code == 0, built.output
    assert (tmp_path / "data/catalog/releases/ds_v1.json").exists()
    assert (
        sum(
            len(Manifest.load(tmp_path / f"data/releases/ds_v1/{s}.parquet"))
            for s in ("train", "dev", "test")
        )
        == 10
    )
    resolved = runner.invoke(app, ["release", "path", "ds_v1", "--split", "test"])
    assert resolved.exit_code == 0, resolved.output
    assert resolved.output.strip().endswith("data/releases/ds_v1/test.parquet")
