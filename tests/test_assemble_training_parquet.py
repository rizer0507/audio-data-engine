# -*- coding: utf-8 -*-
"""Tests for scripts/assemble_training_parquet.py (036)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from audio_engine.core.manifest import Manifest
from audio_engine.core.sample import Sample

import assemble_training_parquet as mod


def _sample(
    sid: str,
    *,
    category: str,
    sha: str,
    gold: str = "",
    duration: float = 1.5,
    reviewed: str | None = None,
) -> Sample:
    labels: dict = {
        "category": category,
        "type": category,
        "original_audio_sha256": sha,
    }
    if gold:
        labels["gold_text"] = gold
    if reviewed is not None:
        labels["reviewed_category"] = reviewed
    return Sample(
        id=sid,
        source_path=f"/data/{sid}.wav",
        sha256=sha,
        duration=duration,
        audio={"resampled_16k": f"/data/{sid}.wav"},
        labels=labels,
    )


def _write_batch(path: Path, samples: list[Sample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Manifest.from_samples(samples).save(path)


@pytest.fixture()
def markers_cfg(tmp_path: Path) -> Path:
    src = ROOT / "configs" / "selection" / "training_exclude_markers_v1.yaml"
    dst = tmp_path / "markers.yaml"
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    return dst


def test_multi_batch_merge_and_clean(tmp_path: Path, markers_cfg: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    b1 = tmp_path / "a.parquet"
    b2 = tmp_path / "b.parquet"
    _write_batch(
        b1,
        [
            _sample("s1", category="gold_candidate", sha="h1", gold="你好"),
            _sample("s2", category="semantic_risk", sha="h2", gold="不做训练"),
            _sample("s3", category="semantic_risk", sha="h3", gold="正常听写"),
            _sample("s4", category="voicemail", sha="h4", gold="请留言"),
        ],
    )
    _write_batch(
        b2,
        [
            _sample("s5", category="hardcase", sha="h5"),
            _sample("s6", category="semantic_risk", sha="h6", gold="不作训练语音信箱"),
            _sample("s1", category="gold_candidate", sha="h7", gold="同名不同音频"),  # 同 sample_id 跨 batch 允许
        ],
    )

    recipe = mod.AssembleRecipe(
        trainset_version="mix_v001",
        categories=["gold_candidate", "semantic_risk", "hardcase"],
        batches=[
            mod.BatchSource("batch_a", b1),
            mod.BatchSource("batch_b", b2),
        ],
        exclude_markers_config=markers_cfg,
        duplicate_audio_policy="fail",
    )
    result = mod.assemble(recipe)
    ids = {(s.labels["source_batch"], s.id) for s in result.samples}
    # s2 洗成 hardcase 且 hardcase 在配方内 → 入选
    # s6 洗成 hardcase → 入选
    # s4 voicemail 不在配方 → 丢弃
    assert ("batch_a", "s1") in ids
    assert ("batch_a", "s2") in ids
    assert ("batch_a", "s3") in ids
    assert ("batch_b", "s5") in ids
    assert ("batch_b", "s6") in ids
    assert ("batch_b", "s1") in ids
    assert ("batch_a", "s4") not in ids

    by_key = {(s.labels["source_batch"], s.id): s for s in result.samples}
    assert by_key[("batch_a", "s2")].labels["category"] == "hardcase"
    assert by_key[("batch_a", "s2")].labels["clean_reason"] == "exclude_from_training"
    assert by_key[("batch_a", "s2")].labels["gold_text"] == ""
    assert by_key[("batch_a", "s2")].labels["gold_text_raw"] == "不做训练"
    assert by_key[("batch_b", "s6")].labels["clean_reason"] == "exclude_voicemail_marked"
    assert result.summary["clean_stats"]["to_hardcase_total"] == 2
    assert result.summary["clean_stats"]["exclude_voicemail_marked"] == 1
    assert result.summary["total_selected"] == 6


def test_dropped_after_clean_when_hardcase_not_selected(
    tmp_path: Path, markers_cfg: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    path = tmp_path / "only.parquet"
    _write_batch(
        path,
        [
            _sample("keep", category="semantic_risk", sha="a", gold="真实听写"),
            _sample("drop", category="semantic_risk", sha="b", gold="不做训练"),
        ],
    )
    recipe = mod.AssembleRecipe(
        trainset_version="mix_v002",
        categories=["semantic_risk"],  # 不含 hardcase
        batches=[mod.BatchSource("b1", path)],
        exclude_markers_config=markers_cfg,
    )
    result = mod.assemble(recipe)
    assert len(result.samples) == 1
    assert result.samples[0].id == "keep"
    assert result.summary["clean_stats"]["dropped_after_clean"] == 1
    drop_rows = [d for d in result.details if d["sample_id"] == "drop"]
    assert drop_rows and drop_rows[0]["dropped_after_clean"] is True


def test_version_conflict_no_overwrite(
    tmp_path: Path, markers_cfg: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    path = tmp_path / "x.parquet"
    _write_batch(path, [_sample("s", category="gold_candidate", sha="z", gold="hi")])
    recipe_path = tmp_path / "recipe.yaml"
    recipe_path.write_text(
        "\n".join(
            [
                "trainset_version: conflict_v1",
                "categories: [gold_candidate]",
                "batches:",
                "  - batch: b1",
                f"    parquet: {path.as_posix()}",
                f"exclude_markers_config: {markers_cfg.as_posix()}",
            ]
        ),
        encoding="utf-8",
    )
    out = tmp_path / "datasets" / "stage1" / "trainsets" / "conflict_v1"
    rc = mod.main(["--recipe", str(recipe_path), "--version", "conflict_v1"])
    assert rc == 0
    assert (out / "manifest.parquet").is_file()
    assert (out / "summary.md").is_file()

    rc2 = mod.main(["--recipe", str(recipe_path), "--version", "conflict_v1"])
    assert rc2 == 2

    rc3 = mod.main(
        ["--recipe", str(recipe_path), "--version", "conflict_v1", "--overwrite"]
    )
    assert rc3 == 0


def test_duplicate_audio_fail_and_keep_first(
    tmp_path: Path, markers_cfg: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    p1 = tmp_path / "p1.parquet"
    p2 = tmp_path / "p2.parquet"
    _write_batch(p1, [_sample("a", category="gold_candidate", sha="SAME", gold="一")])
    _write_batch(p2, [_sample("b", category="gold_candidate", sha="SAME", gold="二")])

    recipe_fail = mod.AssembleRecipe(
        trainset_version="dup_fail",
        categories=["gold_candidate"],
        batches=[mod.BatchSource("b1", p1), mod.BatchSource("b2", p2)],
        exclude_markers_config=markers_cfg,
        duplicate_audio_policy="fail",
    )
    with pytest.raises(ValueError, match="重复音频"):
        mod.assemble(recipe_fail)

    recipe_keep = mod.AssembleRecipe(
        trainset_version="dup_keep",
        categories=["gold_candidate"],
        batches=[mod.BatchSource("b1", p1), mod.BatchSource("b2", p2)],
        exclude_markers_config=markers_cfg,
        duplicate_audio_policy="keep_first",
    )
    result = mod.assemble(recipe_keep)
    assert len(result.samples) == 1
    assert result.samples[0].id == "a"
    assert result.summary["warnings"]["skipped_duplicate_audio"] == 1


def test_reviewed_category_priority(tmp_path: Path, markers_cfg: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    path = tmp_path / "r.parquet"
    _write_batch(
        path,
        [
            _sample(
                "r1",
                category="semantic_risk",
                sha="r",
                gold="ok",
                reviewed="gold_candidate",
            )
        ],
    )
    recipe = mod.AssembleRecipe(
        trainset_version="rev_v1",
        categories=["gold_candidate"],
        batches=[mod.BatchSource("b", path)],
        exclude_markers_config=markers_cfg,
    )
    result = mod.assemble(recipe)
    assert len(result.samples) == 1
    assert result.samples[0].labels["training_category"] == "gold_candidate"


def test_categories_required(tmp_path: Path, markers_cfg: Path):
    recipe_path = tmp_path / "bad.yaml"
    recipe_path.write_text(
        "\n".join(
            [
                "trainset_version: bad",
                "batches:",
                "  - batch: b1",
                "    parquet: missing.parquet",
                f"exclude_markers_config: {markers_cfg.as_posix()}",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="categories"):
        mod.load_recipe_from_yaml(recipe_path)
