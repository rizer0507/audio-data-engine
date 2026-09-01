from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from audio_engine.cli.main import app
from audio_engine.core.catalog import (
    ArtifactCatalog,
    DatasetRelease,
    ModelVersion,
    ProducerRecord,
)


def test_register_is_idempotent_and_detects_payload_mutation(tmp_path: Path):
    payload = tmp_path / "manifest.jsonl"
    payload.write_text('{"id":"a"}\n', encoding="utf-8")
    catalog = ArtifactCatalog(tmp_path / "catalog")
    producer = ProducerRecord(pipeline="clean", run_id="run-1")

    first = catalog.register_file(
        payload,
        kind="manifest",
        producer=producer,
        metadata={"sample_count": 1},
    )
    second = catalog.register_file(
        payload,
        kind="manifest",
        producer=producer,
        metadata={"sample_count": 1},
    )

    assert first == second
    assert catalog.get(first.artifact_id, verify=True) == first
    assert catalog.list(kind="manifest") == [first]

    payload.write_text('{"id":"changed"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="payload changed"):
        catalog.get(first.artifact_id, verify=True)


def test_catalog_records_are_immutable(tmp_path: Path):
    payload = tmp_path / "result.txt"
    payload.write_text("one", encoding="utf-8")
    catalog = ArtifactCatalog(tmp_path / "catalog")
    record = catalog.register_file(payload, kind="report")
    record_path = catalog.records_dir / f"{record.artifact_id}.json"
    raw = json.loads(record_path.read_text(encoding="utf-8"))
    raw["uri"] = "/different/path"
    record_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="artifact id collision"):
        catalog.register_file(payload, kind="report")


def test_release_and_model_contract_validation():
    release = DatasetRelease(
        release_id="ds_v1",
        source_artifact_id="manifest_source",
        outputs={"train": "manifest_train", "dev": "manifest_dev", "test": "manifest_test"},
        policy_version="selection_v1",
        normalization_version="zh_v1",
        gold_revision="gold_v1",
        split_seed=7,
        group_key="conversation_id",
        counts={"train": 10, "dev": 2, "test": 2},
    )
    assert release.outputs["test"] == "manifest_test"

    with pytest.raises(ValidationError, match="outputs missing"):
        DatasetRelease(
            release_id="bad",
            source_artifact_id="manifest_source",
            outputs={"train": "a", "dev": "b"},
            policy_version="v1",
            normalization_version="v1",
            gold_revision="v1",
            split_seed=1,
            group_key="speaker_id",
        )

    model = ModelVersion(
        model_id="qwen_sft_v1",
        base_model="qwen",
        training_release_id="ds_v1",
        training_recipe="recipes/qwen.yaml",
        checkpoint_uri="/models/qwen_sft_v1",
        status="ready",
    )
    assert model.training_release_id == release.release_id


def test_catalog_persists_release_and_model_with_referential_integrity(tmp_path: Path):
    catalog = ArtifactCatalog(tmp_path / "catalog")
    ids = {}
    for name in ("source", "train", "dev", "test"):
        path = tmp_path / f"{name}.parquet"
        path.write_bytes(name.encode())
        ids[name] = catalog.register_file(path, kind="manifest").artifact_id
    release = DatasetRelease(
        release_id="ds_v1",
        source_artifact_id=ids["source"],
        outputs={key: ids[key] for key in ("train", "dev", "test")},
        policy_version="policy_v1",
        normalization_version="zh_v1",
        gold_revision="gold_v1",
        split_seed=1,
        group_key="speaker_id",
    )
    assert catalog.put_release(release) == release
    assert catalog.get_release("ds_v1") == release
    checkpoint = tmp_path / "model"
    checkpoint.mkdir()
    model = ModelVersion(
        model_id="model_v1",
        base_model="base",
        training_release_id="ds_v1",
        training_recipe="recipe.yaml",
        checkpoint_uri=str(checkpoint),
        status="ready",
    )
    assert catalog.put_model(model) == model
    assert catalog.get_model("model_v1") == model
    changed = model.model_copy(update={"status": "failed"})
    with pytest.raises(ValueError, match="immutable registry"):
        catalog.put_model(changed)


def test_artifact_cli_register_show_and_path(tmp_path: Path):
    payload = tmp_path / "report.json"
    payload.write_text("{}", encoding="utf-8")
    catalog_dir = tmp_path / "catalog"
    runner = CliRunner()

    registered = runner.invoke(
        app,
        [
            "artifact",
            "register",
            str(payload),
            "--kind",
            "report",
            "--catalog-dir",
            str(catalog_dir),
        ],
    )
    assert registered.exit_code == 0, registered.output
    artifact_id = next(token for token in registered.output.split() if token.startswith("report_"))

    shown = runner.invoke(
        app, ["artifact", "show", artifact_id, "--verify", "--catalog-dir", str(catalog_dir)]
    )
    assert shown.exit_code == 0, shown.output
    assert str(payload.resolve()) in shown.output

    resolved = runner.invoke(
        app, ["artifact", "path", artifact_id, "--catalog-dir", str(catalog_dir)]
    )
    assert resolved.exit_code == 0, resolved.output
    assert resolved.output.strip() == str(payload.resolve())
