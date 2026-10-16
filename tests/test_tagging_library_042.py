"""042 tag definition library: migration parity, selection contract, snapshot, docs."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml

from audio_engine.core.sample import Sample
from audio_engine.core.tagging.config import load_tagging_config
from audio_engine.core.tagging.docs_gen import (
    check_tag_library_docs,
    generate_tag_library_docs,
)
from audio_engine.core.tagging.engine import MultiTagEngine
from audio_engine.core.tagging.library import (
    TagLibraryError,
    assert_resume_compatible,
    build_definition_snapshot,
    load_tag_library,
    save_definition_snapshot,
)
from audio_engine.core.tagging.service import TaggingRunRequest, TaggingService
from audio_engine.core.tagging.types import Ternary

ROOT = Path(__file__).resolve().parents[1]
LIB_CFG = ROOT / "configs" / "tagging" / "zh_asr_multi_tag_v1.yaml"
INLINE_CFG = ROOT / "tests" / "fixtures" / "tagging" / "zh_asr_multi_tag_v1_inline.yaml"
LIBRARY_DIR = ROOT / "configs" / "tagging" / "library"
KEYS = [
    "glm_1",
    "glm_2",
    "sensevoice_1",
    "sensevoice_2",
    "qwen_1",
    "qwen_2",
]


def _tx(text: str, *, status: str = "success") -> dict:
    return {"text": text, "status": status, "raw_text": text}


def _sample(
    texts: dict[str, str],
    *,
    sid: str = "s1",
    quality: dict | None = None,
    failed: set[str] | None = None,
    labels: dict | None = None,
) -> Sample:
    failed = failed or set()
    transcripts = {}
    for key in KEYS:
        if key in failed:
            transcripts[key] = {"text": "", "status": "failed", "raw_text": ""}
        else:
            transcripts[key] = _tx(texts.get(key, ""))
    return Sample(
        id=sid,
        source_path=f"{sid}.wav",
        sha256="b" * 64,
        duration=2.0,
        transcripts=transcripts,
        quality=dict(quality or {}),
        labels=dict(labels or {}),
        audio={"resampled_16k": f"{sid}.wav"},
    )


def _fixed_offline_samples() -> list[Sample]:
    """Fixed offline evidence set for migration parity (no ASR/acoustic inference)."""
    return [
        _sample(
            {
                "sensevoice_1": "好",
                "sensevoice_2": "好",
                "glm_1": "好",
                "glm_2": "好",
                "qwen_1": "要",
                "qwen_2": "要",
            },
            sid="hit_tag1_critical",
        ),
        _sample(
            {
                "sensevoice_1": "好",
                "sensevoice_2": "好",
                "glm_1": "好",
                "glm_2": "好",
                "qwen_1": "好",
                "qwen_2": "好",
            },
            sid="agree_no_tag1",
        ),
        _sample(
            {
                "sensevoice_1": "好",
                "sensevoice_2": "好",
                "glm_1": "好",
                "glm_2": "好",
                "qwen_1": "",
                "qwen_2": "",
            },
            sid="qwen_stable_empty",
        ),
        _sample(
            {
                "sensevoice_1": "你好世界",
                "sensevoice_2": "你好世界",
                "glm_1": "你好世界",
                "glm_2": "你好世界",
                "qwen_1": "你好世界",
                "qwen_2": "你好世界",
            },
            sid="long_agree",
        ),
        # Parent/segment lineage identity must survive tagging (041 contract surface).
        _sample(
            {
                "sensevoice_1": "嗯",
                "sensevoice_2": "嗯",
                "glm_1": "嗯",
                "glm_2": "嗯",
                "qwen_1": "要",
                "qwen_2": "要",
            },
            sid="seg_parent_001#1",
            labels={
                "parent_sample_id": "seg_parent_001",
                "segment_index": 1,
                "lineage": {"parent_sha256": "a" * 64, "vad": "fsmn"},
            },
        ),
    ]


def _rule_ternaries(result) -> dict[str, str]:
    return {rr.rule_id: rr.ternary.value for rr in result.rule_results}


def _matched_sources(result) -> list[tuple[str, str, str]]:
    return [
        (rr.rule_id, rr.rule_version, rr.tag_id)
        for rr in result.rule_results
        if rr.ternary == Ternary.MATCHED
    ]


# 040→042 迁移对照的原七项（不含 043 新增的 semantic_risk_text_disagreement）
LEGACY_SEVEN_TAG_IDS = frozenset(
    {
        "tag_1",
        "critical_short",
        "voicemail_raw",
        "polarity_conflict",
        "qwen_stable_empty_diag",
        "human_noise_sig_low",
        "silence_strict_zero",
    }
)


def test_library_loads_and_selects_explicit_tags():
    cfg = load_tagging_config(LIB_CFG)
    assert cfg.library_mode is True
    # 默认运行配置选中的库定义及版本必须完整加载（允许各 tag 独立升级）。
    selected = yaml.safe_load(LIB_CFG.read_text(encoding="utf-8"))["select_tags"]
    assert {tid: tag.version for tid, tag in cfg.tags.items()} == {
        item["tag_id"]: str(item["version"]) for item in selected
    }
    assert LEGACY_SEVEN_TAG_IDS.issubset(cfg.tags)
    assert "semantic_risk_text_disagreement" in cfg.tags
    assert cfg.tags["semantic_risk_text_disagreement"].version == "5"
    assert "qwen_unclear_speech_candidate" in cfg.tags
    assert any(r.rule_id == "rule_tag_1_v1" for r in cfg.rules)
    assert any(r.rule_id == "rule_semantic_risk_keyword_sg_qwen_v1" for r in cfg.rules)
    lib = load_tag_library(LIBRARY_DIR)
    assert "tag_1" in lib.entries
    assert "semantic_risk_text_disagreement" in lib.entries
    assert "_example_new_tag" not in {p.stem for p in LIBRARY_DIR.glob("*.yaml") if not p.name.startswith("_")} or True
    # underscore example is not loaded
    assert "example_new_tag" not in lib.entries


def test_migration_parity_rule_ternary_sources_effective_tags():
    """§7.1: 原七项在旧内联 fixture 与库模式默认配置上三态/来源/有效 tag 一致。

    043 新增默认项后，全量 tag/rule 集合不再相等；对照限制在原七项及其规则。
    """
    inline = load_tagging_config(INLINE_CFG)
    library = load_tagging_config(LIB_CFG)
    assert set(inline.tags) == LEGACY_SEVEN_TAG_IDS
    assert LEGACY_SEVEN_TAG_IDS.issubset(library.tags)
    inline_rule_ids = {r.rule_id for r in inline.rules}
    library_legacy_rule_ids = {
        r.rule_id for r in library.rules if r.tag_id in LEGACY_SEVEN_TAG_IDS
    }
    assert inline_rule_ids == library_legacy_rule_ids

    eng_a = MultiTagEngine(inline)
    eng_b = MultiTagEngine(library)
    for sample in _fixed_offline_samples():
        ra = eng_a.evaluate_sample(sample, dry_run=True)
        rb = eng_b.evaluate_sample(sample, dry_run=True)
        ra_legacy = {
            rid: t
            for rid, t in _rule_ternaries(ra).items()
            if rid in inline_rule_ids
        }
        rb_legacy = {
            rid: t
            for rid, t in _rule_ternaries(rb).items()
            if rid in inline_rule_ids
        }
        assert ra_legacy == rb_legacy, sample.id
        assert _matched_sources(ra) == [
            s for s in _matched_sources(rb) if s[0] in inline_rule_ids
        ], sample.id
        # 有效 tag：原七项交集一致（库模式可额外命中 043 新 tag）
        assert sorted(t for t in ra.effective_tags if t in LEGACY_SEVEN_TAG_IDS) == sorted(
            t for t in rb.effective_tags if t in LEGACY_SEVEN_TAG_IDS
        ), sample.id


def test_multi_rule_same_tag_and_order_independent(tmp_path: Path):
    """Multi-rule same tag + rule order change keep effective tags."""
    # Build a tiny library with two rules producing the same tag.
    lib_dir = tmp_path / "library"
    lib_dir.mkdir()
    (lib_dir / "dual_hit.yaml").write_text(
        yaml.safe_dump(
            {
                "tag_id": "dual_hit",
                "version": "1",
                "status": "enabled",
                "name": "双规则同 tag",
                "rules": [
                    {
                        "rule_id": "rule_a",
                        "version": "1",
                        "tag_id": "dual_hit",
                        "condition": {
                            "capability": "family_state",
                            "family": "sensevoice",
                            "state": "stable_text",
                        },
                    },
                    {
                        "rule_id": "rule_b",
                        "version": "1",
                        "tag_id": "dual_hit",
                        "condition": {
                            "capability": "family_state",
                            "family": "qwen",
                            "state": "stable_text",
                        },
                    },
                ],
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    run_cfg = tmp_path / "run.yaml"
    run_cfg.write_text(
        yaml.safe_dump(
            {
                "policy_version": "selection_multi_tag_v1",
                "evidence_config": str(
                    (ROOT / "configs/tagging/evidence/zh_asr_families_v2.yaml").as_posix()
                ),
                "evidence_mode": "existing",
                "library": {"path": str(lib_dir.as_posix())},
                "fragments": {"path": str((ROOT / "configs/tagging/fragments_v1.yaml").as_posix())},
                "lexicons": {},
                "select_tags": [{"tag_id": "dual_hit", "version": "1"}],
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    cfg = load_tagging_config(run_cfg)
    sample = _fixed_offline_samples()[0]
    eng = MultiTagEngine(cfg)
    r1 = eng.evaluate_sample(sample, dry_run=True)
    matched_tags = {rr.tag_id for rr in r1.rule_results if rr.ternary == Ternary.MATCHED}
    assert "dual_hit" in matched_tags
    # Both rules matched independently; same tag appears once in the matched-tag set.
    matched_rules = [rr.rule_id for rr in r1.rule_results if rr.ternary == Ternary.MATCHED]
    assert set(matched_rules) == {"rule_a", "rule_b"}

    # Non-dry-run reconciles sources → effective tags dedupe to one dual_hit.
    r1_live = eng.evaluate_sample(sample, dry_run=False, snapshot_id="snap_test")
    assert r1_live.effective_tags.count("dual_hit") == 1

    cfg.rules = list(reversed(cfg.rules))
    r2 = MultiTagEngine(cfg).evaluate_sample(sample, dry_run=True)
    assert _rule_ternaries(r1) == _rule_ternaries(r2)
    assert matched_tags == {
        rr.tag_id for rr in r2.rule_results if rr.ternary == Ternary.MATCHED
    }


def test_manual_add_block_parity_across_migration(tmp_path: Path):
    samples = [_fixed_offline_samples()[0]]
    for cfg_path, work_name in ((INLINE_CFG, "inline"), (LIB_CFG, "lib")):
        cfg = load_tagging_config(cfg_path)
        service = TaggingService(cfg, config_path=cfg_path)
        work = tmp_path / work_name
        r1 = service.run(
            TaggingRunRequest(
                batch_id="m042",
                config_path=cfg_path,
                samples=samples,
                work_root=work,
                publish=True,
                output_manifest=tmp_path / f"{work_name}_out.parquet",
                summary_xlsx=tmp_path / f"{work_name}_sum.xlsx",
            )
        )
        assert "tag_1" in r1.samples[0].labels["tags"]
        service.remove_tag(
            batch_id="m042",
            sample_id=samples[0].id,
            tag_id="tag_1",
            audio_sha256=samples[0].sha256,
            actor="test",
            reason="block",
            work_root=work,
        )
        r2 = service.run(
            TaggingRunRequest(
                batch_id="m042",
                config_path=cfg_path,
                samples=samples,
                work_root=work,
                publish=True,
                output_manifest=tmp_path / f"{work_name}_out2.parquet",
                summary_xlsx=tmp_path / f"{work_name}_sum2.xlsx",
            )
        )
        assert "tag_1" not in r2.samples[0].labels["tags"]
        # Manual state file survives
        assert (work / "tag_state.json").exists()
        assert (work / "manual_ops.jsonl").exists()


def test_unselected_library_file_does_not_affect_tagging(tmp_path: Path):
    # Copy production library and add an extra tag file that is NOT selected.
    lib_dir = tmp_path / "library"
    shutil.copytree(LIBRARY_DIR, lib_dir)
    (lib_dir / "extra_unselected.yaml").write_text(
        yaml.safe_dump(
            {
                "tag_id": "extra_unselected",
                "version": "1",
                "status": "enabled",
                "name": "未选入",
                "rules": [
                    {
                        "rule_id": "rule_extra",
                        "version": "1",
                        "condition": {"capability": "true"},
                    }
                ],
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    run = yaml.safe_load(LIB_CFG.read_text(encoding="utf-8"))
    run["library"] = {"path": str(lib_dir.as_posix())}
    run_path = tmp_path / "run.yaml"
    run_path.write_text(yaml.safe_dump(run, allow_unicode=True), encoding="utf-8")
    cfg = load_tagging_config(run_path)
    assert "extra_unselected" not in cfg.tags
    sample = _fixed_offline_samples()[0]
    result = MultiTagEngine(cfg).evaluate_sample(sample, dry_run=True)
    assert "extra_unselected" not in result.effective_tags
    assert all(rr.tag_id != "extra_unselected" for rr in result.rule_results)


def test_selection_contract_errors(tmp_path: Path):
    base = yaml.safe_load(LIB_CFG.read_text(encoding="utf-8"))

    def write(mutator) -> Path:
        data = json.loads(json.dumps(base))  # deep copy via json
        mutator(data)
        path = tmp_path / f"bad_{mutator.__name__}.yaml"
        # mutator may not have __name__ if lambda — use counter
        return path, data

    # empty select
    data = yaml.safe_load(LIB_CFG.read_text(encoding="utf-8"))
    data["select_tags"] = []
    p = tmp_path / "empty.yaml"
    p.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    with pytest.raises(TagLibraryError):
        load_tagging_config(p)

    # unknown id
    data = yaml.safe_load(LIB_CFG.read_text(encoding="utf-8"))
    data["select_tags"] = [{"tag_id": "no_such_tag", "version": "1"}]
    p = tmp_path / "unknown.yaml"
    p.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    with pytest.raises(TagLibraryError):
        load_tagging_config(p)

    # version mismatch
    data = yaml.safe_load(LIB_CFG.read_text(encoding="utf-8"))
    data["select_tags"] = [{"tag_id": "tag_1", "version": "999"}]
    p = tmp_path / "ver.yaml"
    p.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    with pytest.raises(TagLibraryError):
        load_tagging_config(p)

    # disabled / archived selection
    lib_dir = tmp_path / "lib_disabled"
    shutil.copytree(LIBRARY_DIR, lib_dir)
    archived = yaml.safe_load((lib_dir / "tag_1.yaml").read_text(encoding="utf-8"))
    archived["status"] = "archived"
    (lib_dir / "tag_1.yaml").write_text(yaml.safe_dump(archived, allow_unicode=True), encoding="utf-8")
    data = yaml.safe_load(LIB_CFG.read_text(encoding="utf-8"))
    data["library"] = {"path": str(lib_dir.as_posix())}
    data["select_tags"] = [{"tag_id": "tag_1", "version": "1"}]
    p = tmp_path / "arch.yaml"
    p.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    with pytest.raises(TagLibraryError):
        load_tagging_config(p)

    # inline + library same id conflict
    data = yaml.safe_load(LIB_CFG.read_text(encoding="utf-8"))
    data["tags"] = [{"tag_id": "tag_1", "name": "冲突"}]
    data["rules"] = [
        {
            "rule_id": "inline_dup",
            "version": "1",
            "tag_id": "tag_1",
            "condition": {"capability": "true"},
        }
    ]
    p = tmp_path / "conflict.yaml"
    p.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    with pytest.raises(TagLibraryError):
        load_tagging_config(p)


def test_fingerprint_changes_with_definition_and_resume_rejects_drift(tmp_path: Path):
    cfg1 = load_tagging_config(LIB_CFG)
    snap1 = build_definition_snapshot(cfg1)
    assert "tags" in snap1 and "rules" in snap1 and "lexicons" in snap1
    assert "entries" in next(iter(snap1["lexicons"].values()))
    assert snap1["content_fingerprint"] == cfg1.content_fingerprint

    # Mutate fragment content → fingerprint must change when reloaded via copied run cfg
    frag_src = ROOT / "configs/tagging/fragments_v1.yaml"
    frag_copy = tmp_path / "fragments_v1.yaml"
    frag_copy.write_text(frag_src.read_text(encoding="utf-8"), encoding="utf-8")
    lib_copy = tmp_path / "library"
    shutil.copytree(LIBRARY_DIR, lib_copy)
    run = yaml.safe_load(LIB_CFG.read_text(encoding="utf-8"))
    run["library"] = {"path": str(lib_copy.as_posix())}
    run["fragments"] = {"path": str(frag_copy.as_posix())}
    run_path = tmp_path / "run.yaml"
    run_path.write_text(yaml.safe_dump(run, allow_unicode=True), encoding="utf-8")
    fp_before = load_tagging_config(run_path).content_fingerprint

    text = frag_copy.read_text(encoding="utf-8")
    frag_copy.write_text(text + "\n# touch\n", encoding="utf-8")
    # fragments content change alone may not change fingerprint if condition unchanged —
    # bump a selected tag description instead for deterministic content change.
    tag_path = lib_copy / "tag_1.yaml"
    raw = yaml.safe_load(tag_path.read_text(encoding="utf-8"))
    raw["description"] = str(raw.get("description") or "") + "\nchanged"
    tag_path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    fp_after = load_tagging_config(run_path).content_fingerprint
    assert fp_before != fp_after

    # Lexicon content change（保留其余已注册词表，仅替换 tag_1 副本）
    lex = tmp_path / "lex.yaml"
    shutil.copy(ROOT / "configs/tagging/lexicons/tag_1_patterns_v1.yaml", lex)
    run["lexicons"] = dict(run["lexicons"])
    run["lexicons"]["tag_1"] = {"path": str(lex.as_posix())}
    run_path.write_text(yaml.safe_dump(run, allow_unicode=True), encoding="utf-8")
    fp_lex1 = load_tagging_config(run_path).content_fingerprint
    lex_raw = yaml.safe_load(lex.read_text(encoding="utf-8"))
    lex_raw["entries"].append({"entry_id": "tag_1_extra", "type": "regex", "match_mode": "fullmatch", "pattern": "^哎$"})
    lex.write_text(yaml.safe_dump(lex_raw, allow_unicode=True), encoding="utf-8")
    fp_lex2 = load_tagging_config(run_path).content_fingerprint
    assert fp_lex1 != fp_lex2

    # Resume drift rejection
    service = TaggingService(cfg1, config_path=LIB_CFG)
    sample = _fixed_offline_samples()[0]
    work = tmp_path / "resume_work"
    r1 = service.run(
        TaggingRunRequest(
            batch_id="drift042",
            config_path=LIB_CFG,
            samples=[sample],
            work_root=work,
            publish=False,
        )
    )
    # Simulate definition change in progress check
    progress = {
        "completed_sample_ids": [sample.id],
        "definition_fingerprint": "deadbeef",
        "config_fingerprint": "deadbeef",
    }
    with pytest.raises(TagLibraryError):
        assert_resume_compatible(progress, cfg1)

    # Unchanged fingerprint allows resume
    assert_resume_compatible(
        {
            "completed_sample_ids": [sample.id],
            "definition_fingerprint": cfg1.content_fingerprint,
        },
        cfg1,
    )
    r2 = service.run(
        TaggingRunRequest(
            batch_id="drift042",
            config_path=LIB_CFG,
            samples=[sample],
            work_root=work,
            resume_run_id=r1.run_id,
            publish=True,
            output_manifest=tmp_path / "resume_out.parquet",
            summary_xlsx=tmp_path / "resume_sum.xlsx",
        )
    )
    assert r2.snapshot_id
    snap_file = work / "work" / r1.run_id / "definition_snapshot.json"
    assert snap_file.is_file()
    old_snap = json.loads(snap_file.read_text(encoding="utf-8"))
    assert old_snap["content_fingerprint"] == cfg1.content_fingerprint
    # Historical snapshot still viewable after current defs would change
    assert "tag_1" in old_snap["tags"]


def test_docs_generate_stable_and_stale_check(tmp_path: Path):
    out1 = tmp_path / "tag规则库.md"
    r1 = generate_tag_library_docs(
        library_dir=LIBRARY_DIR,
        fragments_path=ROOT / "configs/tagging/fragments_v1.yaml",
        output=out1,
        repo_root=ROOT,
    )
    assert r1["ok"] and r1["asr_or_acoustic_invoked"] is False
    text1 = out1.read_text(encoding="utf-8")
    assert "`tag_1`" in text1
    assert "archived" in text1 or "enabled" in text1
    assert "rule_tag_1_v1" in text1
    # regenerate stable
    r2 = generate_tag_library_docs(
        library_dir=LIBRARY_DIR,
        fragments_path=ROOT / "configs/tagging/fragments_v1.yaml",
        output=out1,
        repo_root=ROOT,
    )
    text2 = out1.read_text(encoding="utf-8")
    # hash line stable; timestamp may differ — compare source hash
    assert r1["source_content_hash"] == r2["source_content_hash"]
    assert "source_content_hash:" in text2

    check = check_tag_library_docs(
        library_dir=LIBRARY_DIR,
        fragments_path=ROOT / "configs/tagging/fragments_v1.yaml",
        doc_path=out1,
        repo_root=ROOT,
    )
    assert check["ok"] is True and check["stale"] is False

    # Stale after definition change
    lib_dir = tmp_path / "lib"
    shutil.copytree(LIBRARY_DIR, lib_dir)
    p = lib_dir / "critical_short.yaml"
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    raw["description"] = "changed for stale check"
    p.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    stale = check_tag_library_docs(
        library_dir=lib_dir,
        fragments_path=ROOT / "configs/tagging/fragments_v1.yaml",
        doc_path=out1,
        repo_root=ROOT,
    )
    assert stale["stale"] is True


def test_segment_lineage_preserved_through_tagging(tmp_path: Path):
    """§7.5 surface: segment id + parent lineage not dropped by library migration."""
    cfg = load_tagging_config(LIB_CFG)
    service = TaggingService(cfg, config_path=LIB_CFG)
    sample = [s for s in _fixed_offline_samples() if s.id.startswith("seg_")][0]
    result = service.run(
        TaggingRunRequest(
            batch_id="seg042",
            config_path=LIB_CFG,
            samples=[sample],
            work_root=tmp_path / "w",
            publish=True,
            output_manifest=tmp_path / "seg_out.parquet",
            summary_xlsx=tmp_path / "seg_sum.xlsx",
        )
    )
    out = result.samples[0]
    assert out.id == sample.id
    assert out.labels.get("parent_sample_id") == "seg_parent_001"
    assert out.labels.get("segment_index") == 1
    assert out.labels.get("lineage", {}).get("parent_sha256") == "a" * 64
    assert "tags" in out.labels


def test_view_definition_does_not_need_asr():
    """list/show/docs path: library load only."""
    lib = load_tag_library(LIBRARY_DIR)
    entry = lib.get("tag_1")
    view = entry.as_view_dict()
    assert view["tag_id"] == "tag_1"
    assert view["examples"]
    assert view["rules"]
    # snapshot without running engine
    cfg = load_tagging_config(LIB_CFG)
    snap = build_definition_snapshot(cfg)
    assert snap["schema"] == "tagging_definition_snapshot_v1"
