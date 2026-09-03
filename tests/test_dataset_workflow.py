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
    assert result[0].labels["label"] == "金标"
    assert result[0].labels["type"] == "auto_gold"
    assert result[1].labels["type"] == "hardcase"
    assert result[2].labels["type"] == "review_queue"


def test_classify_zh_asr_v1_excludes_empty_and_keeps_plain_text():
    samples = [
        Sample(
            id="both_empty",
            source_path="empty.wav",
            duration=8.9,
            transcripts={
                "qwen": {"text": ""},
                "sensevoice": {"text": "<|withitn|>。"},
                "doubao": {"text": ""},
            },
            quality={
                "max_vs_base_agreement_cer": 0.0,
                "vs_base_hypothesis_count": 2,
            },
        ),
        Sample(
            id="voicemail",
            source_path="vm.wav",
            duration=4.0,
            transcripts={
                "qwen": {"text": "您好，这里是语音信箱，请留言"},
                "sensevoice": {"text": "您好这里是语音信箱请留言"},
            },
            quality={
                "max_vs_base_agreement_cer": 0.0,
                "vs_base_hypothesis_count": 1,
            },
        ),
        Sample(
            id="gold_empty",
            source_path="one_side.wav",
            duration=3.0,
            transcripts={
                "qwen": {"text": ""},
                "sensevoice": {"text": "<|zh|>还有字"},
            },
            quality={
                "max_vs_base_agreement_cer": 2.0,
                "vs_base_hypothesis_count": 1,
            },
        ),
        Sample(
            id="agree",
            source_path="good.wav",
            duration=2.0,
            transcripts={
                "qwen": {"text": "你好，世界！"},
                "sensevoice": {"text": "<|zh|><|NEUTRAL|>你好世界。"},
                "doubao": {"text": "你好世界"},
            },
            quality={
                "max_vs_base_agreement_cer": 0.0,
                "vs_base_hypothesis_count": 2,
            },
        ),
        Sample(
            id="near_agree",
            source_path="near.wav",
            duration=2.0,
            transcripts={
                "qwen": {"text": "不需要贷款"},
                "sensevoice": {"text": "不需要贷"},
            },
            quality={
                "max_vs_base_agreement_cer": 0.05,
                "vs_base_hypothesis_count": 1,
            },
        ),
        Sample(
            id="one_char_diff",
            source_path="almost.wav",
            duration=2.0,
            transcripts={
                "qwen": {"text": "不需要"},
                "sensevoice": {"text": "需要"},
            },
            quality={
                "max_vs_base_agreement_cer": 1.0 / 3.0,
                "vs_base_hypothesis_count": 1,
            },
        ),
        Sample(
            id="hard",
            source_path="hard.wav",
            duration=2.0,
            transcripts={
                "qwen": {"text": "今天天气很好"},
                "sensevoice": {"text": "明天不用来了"},
            },
            quality={
                "max_vs_base_agreement_cer": 0.8,
                "vs_base_hypothesis_count": 1,
            },
        ),
        Sample(
            id="no_hyp",
            source_path="solo.wav",
            duration=2.0,
            transcripts={"qwen": {"text": "只有base"}},
            quality={
                "max_vs_base_agreement_cer": None,
                "vs_base_hypothesis_count": 0,
            },
        ),
    ]
    result = OperatorRegistry.get("quality.classify").run(
        samples,
        OperatorConfig(params={"config_path": "configs/selection/zh_asr_v1.yaml"}),
    )
    by_id = {sample.id: sample for sample in result}
    assert by_id["both_empty"].labels["classification_bucket"] == "noise"
    assert by_id["both_empty"].labels["type"] == "noise"
    assert by_id["both_empty"].labels["classification_reason_codes"] == [
        "empty_asr_transcripts"
    ]
    assert by_id["voicemail"].labels["classification_bucket"] == "voicemail"
    assert by_id["voicemail"].labels["type"] == "voicemail"
    assert by_id["voicemail"].labels["classification_reason_codes"] == [
        "voicemail_or_phone_assistant"
    ]
    assert by_id["gold_empty"].labels["classification_bucket"] == "review_queue"
    assert by_id["gold_empty"].labels["type"] == "review_queue"
    assert by_id["gold_empty"].labels["classification_reason_codes"] == [
        "empty_gold_source_transcript"
    ]
    assert by_id["agree"].labels["classification_bucket"] == "auto_gold"
    assert by_id["agree"].labels["type"] == "auto_gold"
    assert by_id["agree"].labels["gold_text"] == "你好世界"
    assert by_id["agree"].labels["label"] == "你好世界"
    assert by_id["agree"].labels["gold_source"] in {"qwen", "sensevoice", "doubao"}
    assert by_id["agree"].transcripts["qwen"]["text"] == "你好世界"
    assert by_id["agree"].transcripts["sensevoice"]["text"] == "你好世界"
    assert by_id["agree"].transcripts["sensevoice"]["extra"]["raw_text"] == (
        "<|zh|><|NEUTRAL|>你好世界。"
    )
    assert by_id["near_agree"].labels["classification_bucket"] == "auto_gold"
    assert by_id["near_agree"].labels["gold_text"] in {"不需要贷款", "不需要贷"}
    assert by_id["near_agree"].labels["gold_source"] in {"qwen", "sensevoice"}
    assert by_id["one_char_diff"].labels["classification_bucket"] == "hardcase"
    assert by_id["one_char_diff"].labels["type"] == "hardcase"
    assert by_id["hard"].labels["classification_bucket"] == "hardcase"
    assert by_id["no_hyp"].labels["classification_bucket"] == "review_queue"
    assert by_id["no_hyp"].labels["classification_reason_codes"] == [
        "missing_agreement_hypotheses"
    ]


def test_export_summary_writes_type_and_splits_xlsx(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "reviewed.parquet"
    samples = []
    for i, bucket in enumerate(
        ["auto_gold", "hardcase", "noise", "review_queue"] * 3
    ):
        samples.append(
            Sample(
                id=f"s{i}",
                source_path=f"{i}.wav",
                sha256=f"{i:064x}",
                labels={
                    "classification_bucket": bucket,
                    "classification_reason_codes": ["unit_test"],
                    "gold_text": "金标" if bucket == "auto_gold" else "",
                    "label": "金标" if bucket == "auto_gold" else "",
                },
                transcripts={"qwen": {"text": f"t{i}"}, "sensevoice": {"text": f"s{i}"}},
            )
        )
    Manifest(samples).save(source)
    runner = CliRunner()
    xlsx = tmp_path / "summary.xlsx"
    typed = tmp_path / "typed.parquet"
    result = runner.invoke(
        app,
        [
            "review",
            "export-summary",
            str(source),
            "--output",
            str(xlsx),
            "--output-manifest",
            str(typed),
            "--max-rows",
            "5",
        ],
    )
    assert result.exit_code == 0, result.output
    assert not xlsx.exists()  # split into parts
    part1 = tmp_path / "summary-part-001.xlsx"
    part2 = tmp_path / "summary-part-002.xlsx"
    part3 = tmp_path / "summary-part-003.xlsx"
    assert part1.exists() and part2.exists() and part3.exists()
    import pandas as pd

    frames = [pd.read_excel(p) for p in (part1, part2, part3)]
    assert sum(len(frame) for frame in frames) == 12
    assert all("type" in frame.columns for frame in frames)
    assert set(pd.concat(frames)["type"]) == {
        "auto_gold",
        "hardcase",
        "noise",
        "review_queue",
    }
    typed_samples = Manifest.load(typed).samples
    assert all(s.labels.get("type") == s.labels.get("classification_bucket") for s in typed_samples)

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
    assert resolved.output.strip().replace("\\", "/").endswith("data/releases/ds_v1/test.parquet")


def test_export_gold_fills_label_and_writes_manifest(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "classified.parquet"
    Manifest(
        [
            Sample(
                id="agree",
                source_path="a.wav",
                sha256="a" * 64,
                labels={
                    "classification_bucket": "auto_gold",
                    "classification_reason_codes": ["multi_asr_high_agreement"],
                    "annotation_state": "auto_accepted",
                    "gold_text": "你好世界",
                    "gold_source": "qwen",
                    "selection_policy_version": "selection_zh_asr_v1_2",
                },
                transcripts={"qwen": {"text": "你好世界"}, "sensevoice": {"text": "你好世界"}},
            ),
            Sample(
                id="hard",
                source_path="h.wav",
                sha256="b" * 64,
                labels={"classification_bucket": "hardcase"},
                transcripts={"qwen": {"text": "今天"}, "sensevoice": {"text": "明天"}},
            ),
        ]
    ).save(source)
    runner = CliRunner()
    xlsx = tmp_path / "auto_gold.xlsx"
    gold_manifest = tmp_path / "gold.parquet"
    exported = runner.invoke(
        app,
        [
            "review",
            "export-gold",
            str(source),
            "--output",
            str(xlsx),
            "--output-manifest",
            str(gold_manifest),
        ],
    )
    assert exported.exit_code == 0, exported.output
    frame = pd.read_excel(xlsx, dtype=str).fillna("")
    assert list(frame["sample_id"]) == ["agree"]
    assert list(frame["label"]) == ["你好世界"]
    assert list(frame["gold_text"]) == ["你好世界"]
    gold_samples = Manifest.load(gold_manifest).samples
    assert len(gold_samples) == 1
    assert gold_samples[0].labels["label"] == "你好世界"
    assert gold_samples[0].labels["gold_text"] == "你好世界"


def test_release_build_can_skip_unresolved_review(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "classified.parquet"
    Manifest(
        [
            Sample(
                id="gold",
                source_path="g.wav",
                sha256="1" * 64,
                labels={
                    "classification_bucket": "auto_gold",
                    "annotation_state": "auto_accepted",
                    "gold_text": "金标",
                    "label": "金标",
                    "speaker_id": "p0",
                },
            ),
            Sample(
                id="pending",
                source_path="p.wav",
                sha256="2" * 64,
                labels={
                    "classification_bucket": "hardcase",
                    "speaker_id": "p1",
                },
            ),
        ]
    ).save(source)
    runner = CliRunner()
    blocked = runner.invoke(
        app,
        [
            "release",
            "build",
            str(source),
            "--id",
            "ds_blocked",
            "--policy-version",
            "p1",
            "--normalization-version",
            "n1",
            "--gold-revision",
            "auto",
            "--group-key",
            "speaker_id",
        ],
    )
    assert blocked.exit_code != 0
    built = runner.invoke(
        app,
        [
            "release",
            "build",
            str(source),
            "--id",
            "ds_auto_only",
            "--policy-version",
            "p1",
            "--normalization-version",
            "n1",
            "--gold-revision",
            "auto",
            "--group-key",
            "speaker_id",
            "--allow-unresolved-review",
        ],
    )
    assert built.exit_code == 0, built.output
    assert (tmp_path / "data/catalog/releases/ds_auto_only.json").exists()
    total = sum(
        len(Manifest.load(tmp_path / f"data/releases/ds_auto_only/{split}.parquet"))
        for split in ("train", "dev", "test")
    )
    assert total == 1
