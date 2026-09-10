"""012-D: dataset_v3 sampling, leakage gates, atomic Release."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import audio_engine.operators  # noqa: F401
from audio_engine.core.catalog import ArtifactCatalog, DatasetRelease, is_dataset_policy_v3
from audio_engine.core.dataset_v3.grouping import (
    GroupingConfig,
    apply_grouping_to_samples,
    build_leakage_groups,
)
from audio_engine.core.dataset_v3.release import ReleaseBuildError, publish_release_v3
from audio_engine.core.dataset_v3.reservation import (
    ReservationConfig,
    apply_reservation_to_samples,
    build_reservation,
)
from audio_engine.core.dataset_v3.sampling import (
    SamplingConfig,
    apply_sampling_plan_to_samples,
    build_sampling_plan,
    is_formal_eval_gold,
)
from audio_engine.core.manifest import Manifest
from audio_engine.core.operator import OperatorConfig
from audio_engine.core.registry import OperatorRegistry
from audio_engine.core.sample import Sample
from audio_engine.core.selection_v3.types import DATASET_POLICY_VERSION


ROOT = Path(__file__).resolve().parents[1]
DATASET_CFG = ROOT / "configs" / "datasets" / "zh_asr_v3.yaml"


def _sample(
    sid: str,
    *,
    call_id: str,
    sha: str,
    text: str = "你好",
    duration: float = 1.5,
    labels: dict | None = None,
) -> Sample:
    lab = {
        "call_id": call_id,
        "original_audio_sha256": sha,
        "source_snapshot_id": "snap1",
        "source_audio_id": f"src_{call_id}",
        **(labels or {}),
    }
    return Sample(
        id=sid,
        source_path=f"/audio/{sid}.wav",
        sha256=sha,
        duration=duration,
        labels=lab,
        transcripts={
            "qwen_1": {"text": text},
            "qwen_2": {"text": text},
        },
    )


def _gold(
    sample: Sample,
    *,
    gold_text: str,
    gold_kind: str = "speech",
    semantic: str = "neutral",
    noise: str = "clean",
    crosstalk: str = "false",
    dual: bool = True,
    verified_errors: list[str] | None = None,
) -> Sample:
    s = sample.model_copy(deep=True)
    s.labels["gold_text"] = gold_text
    s.labels["gold_kind"] = gold_kind
    s.labels["human_semantic"] = semantic
    s.labels["human_noise"] = noise
    s.labels["human_crosstalk"] = crosstalk
    s.labels["is_human_verified"] = True
    s.labels["label_source"] = "human"
    s.labels["label_tier"] = "gold"
    s.labels["annotation_state"] = "second_review" if dual else "annotated"
    s.labels.update(annotator_id="a", reviewer_id="b" if dual else "",
                    speech_scope="none" if gold_kind == "non_speech" else "target",
                    audio_event_tags=["silence"] if gold_kind == "non_speech" else [])
    if verified_errors:
        s.labels["verified_error_tags"] = list(verified_errors)
    return s


def _pseudo(sample: Sample, text: str = "普通陈述") -> Sample:
    s = sample.model_copy(deep=True)
    s.labels["type"] = "pseudo_high"
    s.labels["label_tier"] = "pseudo_high"
    s.labels["candidate_text"] = text
    s.labels["decision"] = "auto_accept"
    s.labels["annotation_state"] = "auto_accept"
    s.labels["pseudo_audit_passed"] = True
    return s


def _prepare_pool(n_calls: int = 40) -> tuple[list[Sample], object]:
    """Many singleton leakage groups so reservation can fill roles."""
    samples: list[Sample] = []
    for i in range(n_calls):
        samples.append(
            _sample(
                f"s{i:03d}",
                call_id=f"c{i:03d}",
                sha=f"hash_{i:03d}",
                text="嗯" if i % 7 == 0 else ("不需要" if i % 5 == 0 else "需要"),
            )
        )
    grouping = build_leakage_groups(samples, GroupingConfig())
    samples = apply_grouping_to_samples(samples, grouping)
    reservation = build_reservation(
        samples,
        grouping,
        ReservationConfig(
            seed=7,
            eval_random_target=8,
            eval_core_reserve_ratio=0.25,
            dev_ratio_of_dev_pool=0.2,
            calibration_target=3,
        ),
    )
    samples = apply_reservation_to_samples(samples, reservation)
    return samples, reservation


def _annotate_for_build(samples: list[Sample], reservation) -> list[Sample]:
    """Attach dual-reviewed gold / pseudo so quotas can fill on a tiny corpus."""
    out: list[Sample] = []
    role = reservation.sample_role
    for i, src in enumerate(samples):
        sample = src.model_copy(deep=True)
        r = role.get(sample.id)
        if r == "eval_random":
            # Mix speech + one non_speech empty
            if i % 11 == 0:
                sample = _gold(sample, gold_text="", gold_kind="non_speech", semantic="not_applicable")
            elif i % 5 == 0:
                sample = _gold(sample, gold_text="不需要", semantic="negative")
            elif i % 3 == 0:
                sample = _gold(sample, gold_text="需要", semantic="positive")
            else:
                sample = _gold(sample, gold_text="嗯", semantic="neutral")
        elif r == "eval_core_reserve":
            bucket = i % 6
            if bucket == 0:
                sample = _gold(sample, gold_text="不要", semantic="negative")
            elif bucket == 1:
                sample = _gold(sample, gold_text="嗯", semantic="neutral")
                sample.duration = 1.0
            elif bucket == 2:
                sample = _gold(
                    sample, gold_text="", gold_kind="non_speech", semantic="not_applicable"
                )
            elif bucket == 3:
                sample = _gold(
                    sample,
                    gold_text="有点吵我需要",
                    semantic="positive",
                    noise="noisy",
                )
            elif bucket == 4:
                sample = _gold(sample, gold_text="可以", semantic="positive")
            else:
                sample = _gold(sample, gold_text="今天天气不错", semantic="neutral")
                sample.duration = 3.0
        elif r == "dev":
            sample = _gold(sample, gold_text="开发集金标", semantic="neutral", dual=False)
        elif r == "train_pool":
            if i % 4 == 0:
                sample = _gold(
                    sample,
                    gold_text="我不需要",
                    semantic="negative",
                    verified_errors=["negation_flip"],
                )
                sample.transcripts["qwen_1"] = {"text": "我需要"}
                sample.transcripts["qwen_2"] = {"text": "我需要"}
            elif i % 4 == 1:
                sample = _gold(sample, gold_text="不", semantic="negative")
                sample.duration = 0.8
            elif i % 4 == 2:
                sample = _gold(sample, gold_text="好的可以", semantic="positive")
                sample.duration = 2.5
            else:
                sample = _pseudo(sample, text="这是普通能力保持样本")
        else:
            # calibration / governance: leave pending
            sample.labels.setdefault("annotation_state", "pending")
        out.append(sample)
    return out


def _small_cfg(**overrides) -> SamplingConfig:
    params = {
        "dataset_policy_version": DATASET_POLICY_VERSION,
        "build": {
            "release_id": "zh_asr_v3_test_001",
            "train_size": 8,
            "sampling_seed": 7,
            "fail_on_shortfall": True,
            "allow_non_speech_train": False,
            "max_per_call": 3,
            "require_dual_review_for_eval": True,
            "require_pseudo_audit_for_pseudo_train": True,
            "require_reservation": True,
        },
        "eval_random": {"target": 4},
        "eval_core": {
            "strata": {
                "negative_reject": 1,
                "filler_neutral": 1,
                "confirmed_non_speech": 1,
                "noisy_crosstalk_speech": 1,
                "positive": 1,
                "other": 1,
            }
        },
        "train_pools": {
            "qwen_error_fix": 0.25,
            "human_high_risk": 0.25,
            "human_ordinary": 0.25,
            "pseudo_high_audited": 0.25,
        },
        # Relax cross constraints for tiny corpora
        "train_cross_constraints": {
            "min_positive_ratio": 0.0,
            "min_negative_ratio": 0.0,
            "min_noisy_crosstalk_ratio": 0.0,
            "max_non_speech_ratio": 1.0,
            "max_pseudo_high_ratio": 1.0,
        },
    }
    # deep merge overrides
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(params.get(key), dict):
            params[key] = {**params[key], **value}
        else:
            params[key] = value
    return SamplingConfig.from_params(params)


def test_config_yaml_loads_d_fields():
    cfg = SamplingConfig.from_params(
        __import__("yaml").safe_load(DATASET_CFG.read_text(encoding="utf-8"))
    )
    assert cfg.train_size == 10000
    assert cfg.eval_core.negative_reject == 500
    assert cfg.train_pools.qwen_error_fix == 0.40
    assert is_dataset_policy_v3(cfg.policy_version)


def test_non_speech_empty_is_formal_eval_gold():
    s = _gold(
        _sample("a", call_id="c1", sha="h1"),
        gold_text="",
        gold_kind="non_speech",
        semantic="not_applicable",
    )
    assert is_formal_eval_gold(s, require_dual=True)
    pending = s.model_copy(deep=True)
    pending.labels["annotation_state"] = "annotated"
    pending.labels["gold_text"] = None
    assert not is_formal_eval_gold(pending, require_dual=True)


def test_sampling_deterministic_and_respects_reservation(tmp_path: Path):
    samples, reservation = _prepare_pool(48)
    samples = _annotate_for_build(samples, reservation)
    cfg = _small_cfg()
    plan1 = build_sampling_plan(samples, reservation, cfg)
    plan2 = build_sampling_plan(list(reversed(samples)), reservation, cfg)
    assert plan1.sampling_digest == plan2.sampling_digest
    assert plan1.eval_random_ids == plan2.eval_random_ids
    assert plan1.train_ids == plan2.train_ids

    # Eval reserve unused members must not enter train
    train_groups = {
        reservation.group_mapping[sid]
        for sid in plan1.train_ids
        if sid in reservation.group_mapping
    }
    for gid, role in reservation.group_role.items():
        if role in {"eval_random", "eval_core_reserve"}:
            assert gid not in train_groups


def test_leakage_blocks_same_call_across_splits():
    samples, reservation = _prepare_pool(48)
    samples = _annotate_for_build(samples, reservation)
    cfg = _small_cfg(build={"release_id": "leak_test", "train_size": 4, "fail_on_shortfall": False})
    cfg.eval_random_target = 2
    plan = build_sampling_plan(samples, reservation, cfg)
    stamped = apply_sampling_plan_to_samples(samples, plan, reservation)
    from audio_engine.core.dataset_v3.release import validate_cross_split_leakage

    report = validate_cross_split_leakage(stamped, plan, reservation)
    assert report.ok


def test_shortfall_refuses_formal_release(tmp_path: Path):
    samples, reservation = _prepare_pool(20)
    # Leave almost everything pending → eval/train shortfall
    cfg = _small_cfg(
        build={
            "release_id": "shortfall_x",
            "train_size": 50,
            "fail_on_shortfall": True,
        }
    )
    cfg.eval_random_target = 10
    with pytest.raises(ReleaseBuildError, match="quota_shortfall|formal Release refused"):
        publish_release_v3(
            samples,
            config=cfg,
            reservation=reservation,
            catalog_dir=tmp_path / "catalog",
            output_dir=tmp_path / "releases",
            run_dir=tmp_path / "run",
        )
    assert not (tmp_path / "releases" / "shortfall_x").exists()


def test_atomic_publish_idempotent_and_collision(tmp_path: Path):
    samples, reservation = _prepare_pool(60)
    samples = _annotate_for_build(samples, reservation)
    cfg = _small_cfg(build={"release_id": "zh_asr_v3_rel_a", "train_size": 4})
    cfg.eval_random_target = 2
    # Soften eval_core quotas for small annotated core reserve
    cfg.eval_core.negative_reject = 1
    cfg.eval_core.filler_neutral = 0
    cfg.eval_core.confirmed_non_speech = 0
    cfg.eval_core.noisy_crosstalk_speech = 0
    cfg.eval_core.positive = 1
    cfg.eval_core.other = 0
    cfg.fail_on_shortfall = False

    catalog_dir = tmp_path / "catalog"
    cfg.train_pools.qwen_error_fix = 1.0
    cfg.train_pools.human_high_risk = cfg.train_pools.human_ordinary = cfg.train_pools.pseudo_high_audited = 0.0
    out_dir = tmp_path / "releases"
    first = publish_release_v3(
        samples,
        config=cfg,
        reservation=reservation,
        catalog_dir=catalog_dir,
        output_dir=out_dir,
        run_dir=tmp_path / "run1",
    )
    assert first.release_dir.is_dir()
    assert (first.release_dir / "train.parquet").is_file()
    assert (first.release_dir / "eval_core.parquet").is_file()
    assert (first.release_dir / "eval_random.parquet").is_file()
    assert (first.release_dir / "sampling.json").is_file()
    assert (first.release_dir / "leakage_report.json").is_file()
    assert first.release is not None
    assert "eval_core" in first.release.outputs
    assert "test" not in first.release.outputs

    second = publish_release_v3(
        samples,
        config=cfg,
        reservation=reservation,
        catalog_dir=catalog_dir,
        output_dir=out_dir,
        run_dir=tmp_path / "run2",
    )
    assert second.idempotent_hit is True

    # Different content → fail-fast, no overwrite
    cfg2 = _small_cfg(build={"release_id": "zh_asr_v3_rel_a", "train_size": 3})
    cfg2.eval_random_target = 2
    cfg2.eval_core.negative_reject = 1
    cfg2.eval_core.filler_neutral = 0
    cfg2.eval_core.confirmed_non_speech = 0
    cfg2.eval_core.noisy_crosstalk_speech = 0
    cfg2.eval_core.positive = 0
    cfg2.eval_core.other = 0
    cfg2.fail_on_shortfall = False
    cfg2.train_pools = cfg.train_pools
    with pytest.raises(ReleaseBuildError, match="different content|already"):
        publish_release_v3(
            samples,
            config=cfg2,
            reservation=reservation,
            catalog_dir=catalog_dir,
            output_dir=out_dir,
            run_dir=tmp_path / "run3",
        )


def test_dataset_release_v3_contract():
    release = DatasetRelease(
        release_id="ds_v3",
        source_artifact_id="manifest_source",
        outputs={
            "train": "manifest_train",
            "dev": "manifest_dev",
            "eval_core": "manifest_core",
            "eval_random": "manifest_random",
        },
        policy_version=DATASET_POLICY_VERSION,
        normalization_version="zh_v1",
        gold_revision="annotation_v3.0",
        split_seed=1,
        group_key="leakage_group_id",
        counts={"train": 1, "dev": 1, "eval_core": 1, "eval_random": 1},
    )
    assert release.outputs["eval_core"] == "manifest_core"

    with pytest.raises(Exception):
        DatasetRelease(
            release_id="bad_v3",
            source_artifact_id="manifest_source",
            outputs={"train": "a", "dev": "b", "test": "c"},
            policy_version=DATASET_POLICY_VERSION,
            normalization_version="zh_v1",
            gold_revision="g",
            split_seed=1,
            group_key="leakage_group_id",
        )


def test_build_dataset_v3_operator(tmp_path: Path, monkeypatch):
    samples, reservation = _prepare_pool(60)
    samples = _annotate_for_build(samples, reservation)
    res_path = tmp_path / "reservation.json"
    reservation.write_json(res_path)

    cfg = _small_cfg(build={"release_id": "op_rel_1", "train_size": 4})
    cfg.eval_random_target = 2
    cfg.eval_core.negative_reject = 1
    cfg.eval_core.filler_neutral = 0
    cfg.eval_core.confirmed_non_speech = 0
    cfg.eval_core.noisy_crosstalk_speech = 0
    cfg.eval_core.positive = 1
    cfg.eval_core.other = 0
    cfg.fail_on_shortfall = False

    # Write a minimal config file for the operator
    import yaml

    cfg_path = tmp_path / "ds.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "dataset_policy_version": DATASET_POLICY_VERSION,
                "build": {
                    "release_id": "op_rel_1",
                    "train_size": 4,
                    "sampling_seed": 7,
                    "fail_on_shortfall": False,
                    "require_reservation": True,
                },
                "eval_random": {"target": 2},
                "eval_core": {
                    "strata": {
                        "negative_reject": 1,
                        "filler_neutral": 0,
                        "confirmed_non_speech": 0,
                        "noisy_crosstalk_speech": 0,
                        "positive": 1,
                        "other": 0,
                    }
                },
                "train_pools": {
                    "qwen_error_fix": 1.0,
                    "human_high_risk": 0.0,
                    "human_ordinary": 0.0,
                    "pseudo_high_audited": 0.0,
                },
                "train_cross_constraints": {
                    "min_positive_ratio": 0.0,
                    "min_negative_ratio": 0.0,
                    "min_noisy_crosstalk_ratio": 0.0,
                    "max_non_speech_ratio": 1.0,
                    "max_pseudo_high_ratio": 1.0,
                },
            }
        ),
        encoding="utf-8",
    )

    op = OperatorRegistry.get("quality.build_dataset_v3")
    out = op.run(
        samples,
        OperatorConfig(
            params={
                "config_path": str(cfg_path),
                "reservation_path": str(res_path),
                "release_output_dir": str(tmp_path / "releases"),
                "catalog_dir": str(tmp_path / "catalog"),
            },
            run_dir=str(tmp_path / "run"),
        ),
    )
    assert any(s.labels.get("release_id") == "op_rel_1" for s in out)
    assert (tmp_path / "releases" / "op_rel_1" / "release.json").is_file()
    catalog = ArtifactCatalog(tmp_path / "catalog")
    release = catalog.get_release("op_rel_1")
    assert set(release.outputs) >= {"train", "dev", "eval_core", "eval_random"}


def test_missing_reservation_fail_fast():
    samples = [_sample("only", call_id="c1", sha="h1")]
    cfg = _small_cfg()
    with pytest.raises(ValueError, match="reservation"):
        build_sampling_plan(samples, None, cfg)


def test_pseudo_not_mapped_to_eval_gold():
    samples, reservation = _prepare_pool(48)
    samples = _annotate_for_build(samples, reservation)
    cfg = _small_cfg()
    plan = build_sampling_plan(samples, reservation, cfg)
    stamped = apply_sampling_plan_to_samples(samples, plan, reservation)
    for s in stamped:
        if s.labels.get("split") in {"eval_core", "eval_random"}:
            assert s.labels.get("label_source") in {"human", "trusted_external", None} or s.labels.get(
                "is_human_verified"
            )
            assert s.labels.get("type") != "pseudo_high" or s.labels.get("is_human_verified")
        if s.labels.get("train_pool") == "pseudo_high_audited":
            assert s.labels.get("train_target_text") == s.labels.get("candidate_text")
            # Must not rewrite human gold from pseudo
            assert s.labels.get("gold_text") in (None, "") or "gold_text" not in s.labels or s.labels.get(
                "label_source"
            ) != "model_consensus"
