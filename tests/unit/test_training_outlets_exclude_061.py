"""061 · multi_speaker / exclude_from_training 在所有训练出口的硬排除。

覆盖出口：
- dataset_v3 ``build_sampling_plan`` / ``apply_sampling_plan_to_samples``（正式 Release）
- legacy ``release build`` CLI
- warehouse ``is_explicitly_excluded`` / categories 配置接受人工第六类

断言方式：按样本身份（id）取交集，而非仅凭类别列为空。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

import audio_engine.operators  # noqa: F401
from audio_engine.cli.main import app
from audio_engine.core import human_category as hc
from audio_engine.core.dataset_v3.sampling import (
    TRAIN_EXCLUDE_REASON_061,
    apply_sampling_plan_to_samples,
    build_sampling_plan,
)
from audio_engine.core.manifest import Manifest
from audio_engine.core.sample import Sample
from audio_engine.core.warehouse.categories import load_allowed_categories
from audio_engine.core.warehouse.completion import is_explicitly_excluded

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))
from test_dataset_v3_build import (  # noqa: E402
    _annotate_for_build,
    _gold,
    _prepare_pool,
    _small_cfg,
)


def _mark_multi_speaker(sample: Sample) -> Sample:
    """模拟 label-master v2 合并后的多人说话样本：形式完成、gold_text=None。"""
    s = sample.model_copy(deep=True)
    s.labels.update(
        {
            "annotation_type": hc.ANNOTATION_TYPE_MULTI_SPEAKER,
            "reviewed_category": hc.CATEGORY_MULTI_SPEAKER,
            "category": hc.CATEGORY_MULTI_SPEAKER,
            "source_category": "gold_candidate",
            "gold_text": None,
            "gold_kind": "ambiguous_target",
            "exclude_from_training": True,
            "training_exclusion_reason": "multi_speaker",
            "transcript_annotation_protocol": hc.PROTOCOL_WEB_V2,
            "annotation_state": "second_review",
            "is_human_verified": True,
            "label_source": "human_transcript_web",
        }
    )
    return s


def _mark_flag_only(sample: Sample) -> Sample:
    """有 gold 文本但人工显式 exclude_from_training=True。"""
    s = sample.model_copy(deep=True)
    s.labels["exclude_from_training"] = "true"
    return s


# ---------------------------------------------------------------------------
# dataset_v3 formal release sampling
# ---------------------------------------------------------------------------


def _train_pool_ids(samples: list[Sample], reservation) -> list[str]:
    return [s.id for s in samples if reservation.sample_role.get(s.id) == "train_pool"]


def test_sampling_plan_hard_excludes_multi_speaker_and_flag_by_identity():
    samples, reservation = _prepare_pool(48)
    samples = _annotate_for_build(samples, reservation)
    pool_ids = _train_pool_ids(samples, reservation)
    assert len(pool_ids) >= 6
    ms_ids = set(pool_ids[:3])
    flag_ids = set(pool_ids[3:5])
    by_id = {s.id: s for s in samples}
    for sid in ms_ids:
        by_id[sid] = _mark_multi_speaker(by_id[sid])
    for sid in flag_ids:
        by_id[sid] = _mark_flag_only(by_id[sid])
    samples = list(by_id.values())

    cfg = _small_cfg(build={"train_size": 6, "fail_on_shortfall": False})
    plan = build_sampling_plan(samples, reservation, cfg)

    train_ids = set(plan.train_ids)
    assert train_ids & ms_ids == set()
    assert train_ids & flag_ids == set()
    excluded = {e["id"]: e for e in plan.excluded if e.get("reason") == TRAIN_EXCLUDE_REASON_061}
    assert set(excluded) == ms_ids | flag_ids
    for sid in ms_ids:
        assert hc.REASON_CATEGORY in excluded[sid]["detail"]
        assert hc.REASON_FLAG in excluded[sid]["detail"]
    for sid in flag_ids:
        assert excluded[sid]["detail"] == hc.REASON_FLAG
    assert any(n.startswith("061_training_exclusion") for n in plan.notes)

    stamped = apply_sampling_plan_to_samples(samples, plan, reservation)
    by_id = {s.id: s for s in stamped}
    for sid in ms_ids | flag_ids:
        assert by_id[sid].labels.get("split") != "train"
        assert by_id[sid].labels.get("train_target_text") in (None, "")
    for sid in train_ids:
        assert not hc.training_exclusion_reasons(by_id[sid].labels)
    # 多人说话样本必须保留完整人工数据（不得改写 gold_text 为 ""）
    for sid in ms_ids:
        assert by_id[sid].labels["gold_text"] is None
        assert by_id[sid].labels["annotation_type"] == "multi_speaker"


def test_sampling_plan_excludes_multi_speaker_even_when_non_speech_train_allowed():
    """include 空语音（allow_non_speech_train=True）也不得把 multi_speaker 当成空目标纳入。"""
    samples, reservation = _prepare_pool(48)
    samples = _annotate_for_build(samples, reservation)
    pool_ids = _train_pool_ids(samples, reservation)
    by_id = {s.id: s for s in samples}
    ms_id = pool_ids[0]
    # 伪造一个“历史遗留写法”：gold_text 被错误填为 ""，但 category 仍是 multi_speaker
    ms = _mark_multi_speaker(by_id[ms_id])
    ms.labels["gold_text"] = ""
    ms.labels["gold_kind"] = "non_speech"
    by_id[ms_id] = ms
    noise_id = pool_ids[1]
    by_id[noise_id] = _gold(
        by_id[noise_id], gold_text="", gold_kind="non_speech", semantic="not_applicable"
    )
    by_id[noise_id].labels.update(
        annotation_type="human_noise", category="environment_noise", noise_kind="human_noise"
    )
    samples = list(by_id.values())
    cfg = _small_cfg(
        build={"train_size": 8, "fail_on_shortfall": False, "allow_non_speech_train": True}
    )
    plan = build_sampling_plan(samples, reservation, cfg)
    assert ms_id not in plan.train_ids
    assert any(e["id"] == ms_id and e["reason"] == TRAIN_EXCLUDE_REASON_061 for e in plan.excluded)
    # 人也听不清 → environment_noise 空目标，不是 multi_speaker，不受 061 硬排除
    assert not hc.training_exclusion_reasons(by_id[noise_id].labels)


def test_apply_plan_defends_against_stale_plan_with_multi_speaker_train_id():
    """旧采样器生成的 plan 若含 multi_speaker 训练 id，stamp 阶段仍必须拒绝写训练目标。"""
    samples, reservation = _prepare_pool(48)
    samples = _annotate_for_build(samples, reservation)
    cfg = _small_cfg(build={"train_size": 6, "fail_on_shortfall": False})
    plan = build_sampling_plan(samples, reservation, cfg)
    assert plan.train_ids
    victim = plan.train_ids[0]
    by_id = {s.id: s for s in samples}
    by_id[victim] = _mark_multi_speaker(by_id[victim])
    stamped = apply_sampling_plan_to_samples(list(by_id.values()), plan, reservation)
    v = {s.id: s for s in stamped}[victim]
    assert v.labels["split"] == "excluded"
    assert v.labels["exclude_reason"] == TRAIN_EXCLUDE_REASON_061
    assert v.labels["train_target_text"] is None
    assert v.labels["train_target_source"] == "blocked"
    assert v.labels["gold_text"] is None


# ---------------------------------------------------------------------------
# legacy release build CLI
# ---------------------------------------------------------------------------


def test_export_cli_drops_multi_speaker_by_identity(tmp_path, monkeypatch):
    """Outlet D: ``audio-data export`` is marketed as training-ready → hard-exclude 061."""
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "mix.parquet"
    Manifest(
        [
            Sample(
                id="ok1",
                source_path="a.wav",
                sha256="1" * 64,
                labels={"category": "gold_candidate", "gold_text": "好"},
            ),
            Sample(
                id="ms1",
                source_path="b.wav",
                sha256="2" * 64,
                labels={
                    "annotation_type": "multi_speaker",
                    "reviewed_category": "multi_speaker",
                    "exclude_from_training": True,
                    "gold_text": None,
                },
            ),
        ]
    ).save(source)
    # Put a named dataset where _resolve_dataset can find it as a path argument.
    out = tmp_path / "out.jsonl"
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["export", str(source), "--format", "jsonl", "--output", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert "061 training exclusion" in result.output
    lines = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
    ids = {row["id"] for row in lines}
    assert ids == {"ok1"}
    assert "ms1" not in ids


def test_release_build_cli_drops_multi_speaker_by_identity(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "classified.parquet"
    rows = []
    for i in range(6):
        rows.append(
            Sample(
                id=f"gold{i}",
                source_path=f"g{i}.wav",
                sha256=f"{i}" * 64,
                labels={
                    "classification_bucket": "auto_gold",
                    "annotation_state": "human_accepted",
                    "gold_text": f"金标{i}",
                    "label": f"金标{i}",
                    "speaker_id": f"p{i}",
                    "category": "gold_candidate",
                },
            )
        )
    # 多人说话：形式完成（human_accepted）、但 gold_text 残留了文本（最坏情况）也必须排除
    rows.append(
        Sample(
            id="ms_keep_text",
            source_path="m1.wav",
            sha256="a" * 64,
            labels={
                "annotation_state": "human_accepted",
                "gold_text": "两个人同时在说",
                "label": "两个人同时在说",
                "speaker_id": "pm1",
                "reviewed_category": "multi_speaker",
                "annotation_type": "multi_speaker",
                "source_category": "gold_candidate",
            },
        )
    )
    # 多人说话：gold_text=None（契约形态）。若不先排除，会被 “accepted without Gold” 误拦。
    rows.append(
        Sample(
            id="ms_null",
            source_path="m2.wav",
            sha256="b" * 64,
            labels={
                "annotation_state": "human_accepted",
                "gold_text": None,
                "speaker_id": "pm2",
                "category": "multi_speaker",
                "exclude_from_training": True,
            },
        )
    )
    Manifest(rows).save(source)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "release", "build", str(source),
            "--id", "ds_061",
            "--policy-version", "p1",
            "--normalization-version", "n1",
            "--gold-revision", "r1",
            "--group-key", "speaker_id",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "061 training exclusion" in result.output
    all_ids: set[str] = set()
    for split in ("train", "dev", "test"):
        m = Manifest.load(tmp_path / f"data/releases/ds_061/{split}.parquet")
        for s in m:
            all_ids.add(s.id)
            assert not hc.training_exclusion_reasons(s.labels)
    assert all_ids & {"ms_keep_text", "ms_null"} == set()
    assert all_ids == {f"gold{i}" for i in range(6)}


# ---------------------------------------------------------------------------
# warehouse
# ---------------------------------------------------------------------------


def test_warehouse_accepts_multi_speaker_as_human_final_category_but_excludes_training():
    allowed = load_allowed_categories(ROOT / "configs/warehouse/categories_five_class_v2_2.yaml")
    assert "multi_speaker" in allowed
    # 自动五分类集合保持不变
    from audio_engine.core.selection_v3.types import BUSINESS_CATEGORIES_FIVE_CLASS

    assert "multi_speaker" not in BUSINESS_CATEGORIES_FIVE_CLASS
    ms = Sample(
        id="x",
        source_path="x.wav",
        sha256="c" * 64,
        labels={
            "annotation_state": "second_review",
            "gold_kind": "ambiguous_target",
            "reviewed_category": "multi_speaker",
            "annotation_type": "multi_speaker",
        },
    )
    assert is_explicitly_excluded(ms)
    flag_only = Sample(
        id="y",
        source_path="y.wav",
        sha256="d" * 64,
        labels={"annotation_state": "second_review", "gold_kind": "speech", "gold_text": "好",
                "exclude_from_training": 1},
    )
    assert is_explicitly_excluded(flag_only)
    noise = Sample(
        id="z",
        source_path="z.wav",
        sha256="e" * 64,
        labels={"annotation_state": "second_review", "gold_kind": "non_speech", "gold_text": "",
                "category": "environment_noise", "annotation_type": "human_noise"},
    )
    assert not is_explicitly_excluded(noise)


@pytest.mark.parametrize(
    "labels",
    [
        {"category": "environment_noise", "annotation_type": "silence"},
        {"category": "environment_noise", "annotation_type": "background_suspected"},
        {"category": "environment_noise", "annotation_type": "human_noise"},
        {"category": "environment_noise", "decision": "empty"},  # 历史 empty 无 annotation_type
        {"category": "gold_candidate", "source_category": "multi_speaker"},  # 仅溯源，不参与
    ],
)
def test_noise_and_legacy_rows_not_hard_excluded(labels):
    assert hc.training_exclusion_reasons(labels) == []
