"""044 scaffolding: free_tags_v2 configs, migration CLI, source naming."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from audio_engine.core.manifest import Manifest
from audio_engine.core.sample import Sample
from audio_engine.core.source_naming import (
    STAGE1_DERIVED_DIR,
    apply_source_name_to_single_pipeline,
    staged_manifest_path,
)
from audio_engine.core.tagging.config_interim_free_tags_v2 import (
    load_free_tags_v2_raw,
    validate_free_tags_v2_raw,
)
from audio_engine.core.tagging.migrate import migrate_tagging_config

ROOT = Path(__file__).resolve().parents[1]
FREE_TAGS_CFG = ROOT / "configs" / "tagging" / "zh_asr_free_tags_v2.yaml"
MULTI_TAG_CFG = ROOT / "configs" / "tagging" / "zh_asr_multi_tag_v1.yaml"


def test_free_tags_yaml_exists_without_selection_config():
    assert FREE_TAGS_CFG.is_file()
    raw = yaml.safe_load(FREE_TAGS_CFG.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    assert raw.get("policy_version") == "free_tags_v2"
    assert "selection_config" not in raw
    assert "selection_config_path" not in raw
    assert str(raw.get("evidence_config", "")).endswith("zh_asr_families_v2.yaml")
    errors = validate_free_tags_v2_raw(raw, path=str(FREE_TAGS_CFG))
    assert not errors
    load_free_tags_v2_raw(FREE_TAGS_CFG)


def test_free_tags_source_naming_rejects_classified_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)

    class _Step:
        operator = "quality.multi_tag_v1"
        params: dict = {}

    batch = "ft044batch"
    classified = staged_manifest_path(f"classified_five_class_v3_{batch}")
    classified.parent.mkdir(parents=True, exist_ok=True)
    Manifest([Sample(id="a", source_path="/tmp/a.wav", duration=1.0)]).save(classified)

    prepared = staged_manifest_path(f"prepared_asr_v3_{batch}")
    assert prepared.parent == STAGE1_DERIVED_DIR
    assert not prepared.is_file()

    with pytest.raises(FileNotFoundError, match="Classified_\\* fallback is disabled"):
        apply_source_name_to_single_pipeline(
            pipeline_name="tag_dataset_free_tags_v2",
            steps=[_Step()],
            source_name=batch,
        )


def test_migrate_config_dry_run_report_structure(tmp_path: Path):
    # Source must still be legacy selection_config schema (already-migrated configs are not sources).
    legacy = tmp_path / "legacy_multi_tag.yaml"
    legacy.write_text(
        yaml.safe_dump(
            {
                "policy_version": "selection_multi_tag_v1",
                "selection_config": str(
                    (ROOT / "configs/selection/zh_asr_five_class_v3.yaml").as_posix()
                ),
                "evidence_mode": "existing",
                "library": {"path": "configs/tagging/library"},
                "fragments": {"path": "configs/tagging/fragments_v1.yaml"},
                "lexicons": {
                    "tag_1": {"path": "configs/tagging/lexicons/tag_1_patterns_v1.yaml"},
                },
                "select_tags": [{"tag_id": "tag_1", "version": "1"}],
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    out = tmp_path / "zh_asr_free_tags_v2.yaml"
    report = migrate_tagging_config(legacy, out, dry_run=True)
    assert report.get("ok") is True
    assert report.get("dry_run") is True
    assert "targets" in report
    assert "tagging_config" in report["targets"]
    assert "evidence_config" in report["targets"]
    assert "difference_report" in report["targets"]
    assert "mapped_from_selection" in report
    assert "dropped_tagging_keys" in report
    assert "selection_config" in report["dropped_tagging_keys"]
    assert "would_write" in report
    assert "evidence_config" in report["would_write"]
    assert "tagging_config" in report["would_write"]
    assert "selection_config" not in report["would_write"]["tagging_config"]
