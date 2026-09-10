"""012-A: selection_v3 contract + dataset_v3 grouping/reservation."""

from __future__ import annotations

from pathlib import Path

import audio_engine.operators  # noqa: F401
from audio_engine.core.dataset_v3.grouping import (
    GroupingConfig,
    apply_grouping_to_samples,
    build_leakage_groups,
)
from audio_engine.core.dataset_v3.reservation import (
    ReservationConfig,
    apply_reservation_to_samples,
    build_reservation,
)
from audio_engine.core.eval_ready import inspect_eval_manifest
from audio_engine.core.manifest import Manifest
from audio_engine.core.operator import OperatorConfig
from audio_engine.core.registry import OperatorRegistry
from audio_engine.core.sample import Sample
from audio_engine.core.selection_v3.config import SelectionV3Config
from audio_engine.core.selection_v3.input_contract import (
    apply_contract_to_samples,
    classify_run_status,
    evaluate_sample_contract,
    merge_field_by_join_key,
    original_audio_sha256,
)
from audio_engine.core.selection_v3.types import (
    RESERVATION_EVAL_RANDOM,
    RESERVATION_GOVERNANCE_HOLD,
    RESERVATION_TRAIN_POOL,
    RUN_STATUS_FAILED,
    RUN_STATUS_MISSING,
    RUN_STATUS_SUCCESS_EMPTY,
    RUN_STATUS_SUCCESS_TEXT,
    SAMPLE_CLASSIFIABLE,
    SAMPLE_INFERENCE_INCOMPLETE,
    SAMPLE_INVALID_AUDIO,
)


ROOT = Path(__file__).resolve().parents[1]
DATASET_CFG = ROOT / "configs" / "datasets" / "zh_asr_v3.yaml"

EIGHT_KEYS = [
    "kimi_1",
    "kimi_2",
    "glm_1",
    "glm_2",
    "sensevoice_1",
    "sensevoice_2",
    "qwen_1",
    "qwen_2",
]


def _cfg(**overrides) -> SelectionV3Config:
    params = {
        "engine": "consensus_v3",
        "rule_version": "selection_v3.0",
        "target_family": "qwen",
        "model_families": {
            "kimi": ["kimi_1", "kimi_2"],
            "glm": ["glm_1", "glm_2"],
            "sensevoice": ["sensevoice_1", "sensevoice_2"],
            "qwen": ["qwen_1", "qwen_2"],
        },
        "teacher_families": ["kimi", "glm", "sensevoice"],
        "expected_runs_per_family": 2,
    }
    params.update(overrides)
    return SelectionV3Config.from_params(params)


def _full_transcripts(text: str = "你好") -> dict:
    return {k: {"text": text} for k in EIGHT_KEYS}


def _sample(
    sid: str,
    *,
    text: str = "你好",
    sha: str = "hash_a",
    call_id: str | None = "call1",
    source_audio_id: str | None = None,
    duration: float = 1.5,
    labels: dict | None = None,
    transcripts: dict | None = None,
    broken: bool = False,
) -> Sample:
    lab = dict(labels or {})
    if call_id is not None:
        lab.setdefault("call_id", call_id)
    if source_audio_id is not None:
        lab.setdefault("source_audio_id", source_audio_id)
    lab.setdefault("original_audio_sha256", sha)
    lab.setdefault("source_snapshot_id", "snap1")
    if broken:
        lab["broken"] = True
    return Sample(
        id=sid,
        source_path=f"{sid}.wav",
        sha256=sha,
        duration=0.0 if broken else duration,
        transcripts=transcripts if transcripts is not None else _full_transcripts(text),
        labels=lab,
    )


def test_config_from_yaml_and_reject_duplicate_family_alias():
    cfg = SelectionV3Config.from_yaml(DATASET_CFG)
    assert cfg.engine == "consensus_v3"
    assert len(cfg.all_transcript_keys()) == 8
    assert cfg.configured_family_count == 4
    assert cfg.expected_total_runs == 8
    try:
        _cfg(model_families={"kimi": ["kimi_1", "kimi_2"], "glm": ["kimi_1", "glm_2"]})
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "both" in str(exc) or "duplicate" in str(exc).lower() or "assigned" in str(exc)


def test_three_family_config_accepted_and_two_family_rejected():
    three = SelectionV3Config.from_params(
        {
            "engine": "consensus_v3",
            "target_family": "qwen",
            "model_families": {
                "glm": ["glm_1", "glm_2"],
                "sensevoice": ["sensevoice_1", "sensevoice_2"],
                "qwen": ["qwen_1", "qwen_2"],
            },
            "teacher_families": ["glm", "sensevoice"],
            "expected_runs_per_family": 2,
        }
    )
    assert three.configured_family_count == 3
    assert three.expected_total_runs == 6
    assert len(three.all_transcript_keys()) == 6

    # Auto-derive teachers when omitted
    auto = SelectionV3Config.from_params(
        {
            "target_family": "qwen",
            "model_families": {
                "glm": ["glm_1", "glm_2"],
                "sensevoice": ["sensevoice_1", "sensevoice_2"],
                "qwen": ["qwen_1", "qwen_2"],
            },
            "expected_runs_per_family": 2,
        }
    )
    assert sorted(auto.teacher_families) == ["glm", "sensevoice"]

    try:
        SelectionV3Config.from_params(
            {
                "target_family": "qwen",
                "model_families": {
                    "glm": ["glm_1", "glm_2"],
                    "qwen": ["qwen_1", "qwen_2"],
                },
                "teacher_families": ["glm"],
                "expected_runs_per_family": 2,
            }
        )
        assert False, "expected ValueError for N=2"
    except ValueError as exc:
        assert "at least 3" in str(exc).lower() or "3" in str(exc)


def test_three_family_contract_classifiable():
    cfg = SelectionV3Config.from_params(
        {
            "target_family": "qwen",
            "model_families": {
                "glm": ["glm_1", "glm_2"],
                "sensevoice": ["sensevoice_1", "sensevoice_2"],
                "qwen": ["qwen_1", "qwen_2"],
            },
            "teacher_families": ["glm", "sensevoice"],
            "expected_runs_per_family": 2,
        }
    )
    keys = cfg.all_transcript_keys()
    s = Sample(
        id="t3",
        source_path="t3.wav",
        sha256="h3",
        duration=1.5,
        transcripts={k: {"text": "你好"} for k in keys},
        labels={
            "call_id": "c3",
            "original_audio_sha256": "h3",
            "source_snapshot_id": "snap1",
        },
    )
    result = evaluate_sample_contract(s, cfg)
    assert result.readiness == SAMPLE_CLASSIFIABLE
    updated, report, _ = apply_contract_to_samples([s], cfg)
    assert report.classifiable == 1
    report.assert_conserved()
    assert updated[0].labels["contract_readiness"] == SAMPLE_CLASSIFIABLE


def test_run_status_success_empty_failed_missing():
    s = _sample(
        "s1",
        transcripts={
            "kimi_1": {"text": "需要"},
            "kimi_2": {"text": ""},
            "glm_1": {"text": "", "status": "failed"},
            # glm_2 missing
            "sensevoice_1": {"text": "嗯"},
            "sensevoice_2": {"text": "嗯"},
            "qwen_1": {"text": "需要"},
            "qwen_2": {"text": "需要"},
        },
    )
    assert classify_run_status(s, "kimi_1") == RUN_STATUS_SUCCESS_TEXT
    assert classify_run_status(s, "kimi_2") == RUN_STATUS_SUCCESS_EMPTY
    assert classify_run_status(s, "glm_1") == RUN_STATUS_FAILED
    assert classify_run_status(s, "glm_2") == RUN_STATUS_MISSING


def test_incomplete_not_treated_as_all_empty():
    s = _sample(
        "inc1",
        transcripts={
            **{k: {"text": ""} for k in EIGHT_KEYS if k != "qwen_2"},
            # qwen_2 missing → incomplete, not all_empty
        },
    )
    result = evaluate_sample_contract(s, _cfg())
    assert result.readiness == SAMPLE_INFERENCE_INCOMPLETE
    assert "qwen_2" in result.missing_runs


def test_conservation_report_and_fail_fast_duplicate_id():
    samples = [_sample("a", sha="h1"), _sample("b", sha="h2", call_id="call2")]
    updated, report, _ = apply_contract_to_samples(samples, _cfg())
    assert report.input_count == 2
    assert report.classifiable == 2
    report.assert_conserved()
    assert all(s.labels["contract_readiness"] == SAMPLE_CLASSIFIABLE for s in updated)

    broken = _sample("x", broken=True)
    incomplete = _sample(
        "y",
        call_id="call_y",
        transcripts={k: {"text": "a"} for k in EIGHT_KEYS[:-1]},
    )
    ok = _sample("z", call_id="call_z")
    _, report2, _ = apply_contract_to_samples([broken, incomplete, ok], _cfg())
    assert report2.physically_invalid == 1
    assert report2.inference_incomplete == 1
    assert report2.classifiable == 1
    assert len(report2.retry_list) == 1

    try:
        apply_contract_to_samples(
            [_sample("dup"), _sample("dup", sha="other")], _cfg()
        )
        assert False, "expected duplicate id fail"
    except ValueError as exc:
        assert "duplicate" in str(exc).lower()


def test_join_by_id_and_original_hash_not_row_position():
    base = [
        _sample("a", sha="hash_a"),
        _sample("b", sha="hash_b", call_id="c2"),
    ]
    sidecar = {
        ("a", "hash_a"): {"dnsmos_ovrl": 3.1},
        ("b", "hash_b"): {"dnsmos_ovrl": 2.2},
    }
    merged = merge_field_by_join_key(base, sidecar, target="quality")
    assert merged[0].quality["dnsmos_ovrl"] == 3.1
    assert merged[1].quality["dnsmos_ovrl"] == 2.2
    # Wrong hash must not match
    bad = merge_field_by_join_key(
        base, {("a", "WRONG"): {"dnsmos_ovrl": 9.9}}, target="quality"
    )
    assert "dnsmos_ovrl" not in bad[0].quality


def test_leakage_groups_call_source_hash_and_near_dup():
    samples = [
        _sample("s1", sha="h1", call_id="C1"),
        _sample("s2", sha="h2", call_id="C1"),  # same call
        _sample("s3", sha="h3", call_id="C2", source_audio_id="SRC"),
        _sample("s4", sha="h4", call_id="C3", source_audio_id="SRC"),  # same source
        _sample("s5", sha="SAME", call_id="C4"),
        _sample("s6", sha="SAME", call_id="C5"),  # same file hash
        _sample(
            "s7",
            sha="h7",
            call_id="C6",
            labels={
                "near_duplicate_group_id": "nd1",
                "near_duplicate_algorithm": "fp_v1",
                "near_duplicate_threshold": "0.9",
                "source_snapshot_id": "snap1",
            },
        ),
        _sample(
            "s8",
            sha="h8",
            call_id="C7",
            labels={
                "near_duplicate_group_id": "nd1",
                "source_snapshot_id": "snap1",
            },
        ),
    ]
    grouping = build_leakage_groups(samples)
    assert grouping.leakage_group_id["s1"] == grouping.leakage_group_id["s2"]
    assert grouping.leakage_group_id["s3"] == grouping.leakage_group_id["s4"]
    assert grouping.leakage_group_id["s5"] == grouping.leakage_group_id["s6"]
    assert grouping.leakage_group_id["s7"] == grouping.leakage_group_id["s8"]


def test_missing_group_meta_goes_to_governance_not_silent_id_split():
    samples = [
        Sample(
            id="orphan",
            source_path="o.wav",
            sha256="hx",
            duration=1.0,
            transcripts=_full_transcripts(),
            labels={"original_audio_sha256": "hx"},  # no call/source/snapshot
        ),
        _sample("ok1", call_id="c1", sha="h1"),
        _sample("ok2", call_id="c2", sha="h2"),
        _sample("ok3", call_id="c3", sha="h3"),
    ]
    grouping = build_leakage_groups(samples, GroupingConfig(require_source_index=True))
    assert "orphan" in grouping.missing_group_meta_ids
    reservation = build_reservation(
        samples,
        grouping,
        ReservationConfig(
            seed=7,
            eval_random_target=1,
            eval_core_reserve_ratio=0.0,
            dev_ratio_of_dev_pool=0.0,
            calibration_target=0,
            isolate_missing_group_meta=True,
        ),
    )
    assert reservation.sample_role["orphan"] == RESERVATION_GOVERNANCE_HOLD
    assert "orphan" not in reservation.eval_random_ids
    # Must not silently put orphan into train via id-only split
    assert reservation.sample_role["orphan"] != RESERVATION_TRAIN_POOL


def test_reservation_deterministic_and_independent_of_asr_order():
    samples_a = [
        _sample(f"id{i}", call_id=f"c{i}", sha=f"h{i}") for i in range(20)
    ]
    samples_b = list(reversed(samples_a))
    # Attach dummy ASR-ish labels that must not affect reservation
    for s in samples_b:
        s.labels["fake_asr_type"] = "hardcase"

    gcfg = GroupingConfig()
    ga = build_leakage_groups(samples_a, gcfg)
    gb = build_leakage_groups(samples_b, gcfg)
    cfg = ReservationConfig(
        seed=123,
        eval_random_target=3,
        eval_core_reserve_ratio=0.2,
        dev_ratio_of_dev_pool=0.25,
        calibration_target=2,
    )
    ra = build_reservation(samples_a, ga, cfg, grouping_config=gcfg)
    rb = build_reservation(samples_b, gb, cfg, grouping_config=gcfg)
    assert ra.content_digest == rb.content_digest
    assert ra.eval_random_ids == rb.eval_random_ids
    assert ra.verify_digest()

    # Changing ASR texts must not change reservation when ids/groups fixed
    samples_c = [
        _sample(f"id{i}", call_id=f"c{i}", sha=f"h{i}", text=f"不同文本{i}")
        for i in range(20)
    ]
    gc = build_leakage_groups(samples_c, gcfg)
    rc = build_reservation(samples_c, gc, cfg, grouping_config=gcfg)
    assert rc.eval_random_ids == ra.eval_random_ids
    assert rc.content_digest == ra.content_digest


def test_eval_random_locks_whole_leakage_group():
    samples = [
        _sample("a1", call_id="CALLX", sha="ha"),
        _sample("a2", call_id="CALLX", sha="hb"),
        _sample("b1", call_id="CALLY", sha="hc"),
        _sample("c1", call_id="CALLZ", sha="hd"),
    ]
    grouping = build_leakage_groups(samples)
    art = build_reservation(
        samples,
        grouping,
        ReservationConfig(
            seed=1,
            eval_random_target=1,
            eval_core_reserve_ratio=0.0,
            calibration_target=0,
            dev_ratio_of_dev_pool=0.0,
        ),
    )
    selected = art.eval_random_ids[0]
    gid = art.group_mapping[selected]
    members = [sid for sid, g in art.group_mapping.items() if g == gid]
    for mid in members:
        assert art.sample_role[mid] == RESERVATION_EVAL_RANDOM
        assert art.sample_role[mid] != RESERVATION_TRAIN_POOL


def test_prepare_dataset_v3_operator(tmp_path: Path):
    samples = [
        _sample(f"p{i}", call_id=f"call{i}", sha=f"hash{i}") for i in range(12)
    ]
    # One incomplete
    samples[0].transcripts.pop("qwen_2")
    op = OperatorRegistry.get("quality.prepare_dataset_v3")
    out = op.run(
        samples,
        OperatorConfig(
            params={
                "config_path": str(DATASET_CFG),
                "reservation_output": str(tmp_path / "reservation.json"),
                "report_output": str(tmp_path / "report.json"),
            },
            run_dir=tmp_path,
            step_name="prepare",
        ),
    )
    assert (tmp_path / "reservation.json").exists()
    assert (tmp_path / "report.json").exists()
    assert out[0].labels["contract_readiness"] == SAMPLE_INFERENCE_INCOMPLETE
    assert "leakage_group_id" in out[1].labels
    assert "reservation" in out[1].labels
    assert out[1].labels["reservation_digest"]


def test_aggregate_manifests_eight_route_report(tmp_path: Path):
    base = [_sample("m1", sha="h1"), _sample("m2", sha="h2", call_id="c2")]
    # Write join manifests with sensevoice only
    for model in ("sensevoice_1", "sensevoice_2"):
        join_samples = []
        for s in base:
            js = s.model_copy(deep=True)
            js.transcripts = {model: {"text": "测"}}
            join_samples.append(js)
        Manifest(join_samples).save(tmp_path / f"{model}.parquet")

    # Base already has all keys; overwrite path test with left-side only joins
    base_manifest = [s.model_copy(deep=True) for s in base]
    for s in base_manifest:
        s.transcripts = {k: {"text": "基"} for k in ("kimi_1", "kimi_2", "glm_1", "glm_2", "qwen_1", "qwen_2")}

    op = OperatorRegistry.get("quality.aggregate_manifests")
    result = op.run(
        base_manifest,
        OperatorConfig(
            params={
                "overwrite": True,
                "hash_policy": "original_audio",
                "selection_config_path": str(DATASET_CFG),
                "manifests": [
                    {"model": "sensevoice_1", "path": str(tmp_path / "sensevoice_1.parquet")},
                    {"model": "sensevoice_2", "path": str(tmp_path / "sensevoice_2.parquet")},
                ],
            },
            run_dir=tmp_path,
            step_name="agg",
        ),
    )
    assert "sensevoice_1" in result[0].transcripts
    report = (tmp_path / "reports" / "agg_alignment.json")
    assert report.exists()
    import json

    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["aligned"] is True
    assert "eight_route_integrity" in data
    assert len(data["eight_route_integrity"]["expected_runs"]) == 8


def test_eval_ready_leakage_group_and_source_audio(tmp_path: Path):
    eval_samples = [
        _sample(
            "e1",
            call_id="CX",
            sha="he",
            labels={
                "call_id": "CX",
                "source_audio_id": "SRC1",
                "leakage_group_id": "lg_shared",
                "gold_text": "你好",
                "annotation_state": "human_accepted",
                "original_audio_sha256": "he",
                "source_snapshot_id": "snap",
            },
        )
    ]
    train_samples = [
        _sample(
            "t1",
            call_id="CX",
            sha="ht",
            labels={
                "call_id": "CX",
                "source_audio_id": "SRC1",
                "leakage_group_id": "lg_shared",
                "original_audio_sha256": "ht",
                "source_snapshot_id": "snap",
            },
        )
    ]
    eval_path = tmp_path / "eval.parquet"
    train_path = tmp_path / "train.parquet"
    Manifest(eval_samples).save(eval_path)
    Manifest(train_samples).save(train_path)
    report = inspect_eval_manifest(eval_path, train_manifest=train_path)
    assert report.leak_leakage_groups
    assert report.leak_source_audio_ids
    assert any("leakage_group_id" in e for e in report.errors)


def test_invalid_audio_readiness():
    s = _sample("bad", broken=True)
    assert evaluate_sample_contract(s, _cfg()).readiness == SAMPLE_INVALID_AUDIO


def test_apply_grouping_and_reservation_labels():
    samples = [_sample("g1", call_id="c1"), _sample("g2", call_id="c1")]
    grouping = build_leakage_groups(samples)
    grouped = apply_grouping_to_samples(samples, grouping)
    art = build_reservation(
        grouped,
        grouping,
        ReservationConfig(seed=0, eval_random_target=1, calibration_target=0, eval_core_reserve_ratio=0),
    )
    stamped = apply_reservation_to_samples(grouped, art)
    assert stamped[0].labels["leakage_group_id"] == stamped[1].labels["leakage_group_id"]
    assert original_audio_sha256(stamped[0])
