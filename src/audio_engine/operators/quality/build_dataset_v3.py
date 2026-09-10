"""Build dataset v3: quota sampling + gates + atomic Release (012-D)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from audio_engine.core.artifacts import atomic_write_json
from audio_engine.core.dataset_v3.release import ReleaseBuildError, publish_release_v3
from audio_engine.core.dataset_v3.reservation import ReservationArtifact
from audio_engine.core.dataset_v3.sampling import SamplingConfig, build_sampling_plan
from audio_engine.core.operator import ManifestOperator, OperatorConfig
from audio_engine.core.registry import register_operator
from audio_engine.core.sample import Sample


def _load_dataset_config(path: str | Path | None, params: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    if path:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"dataset config must be a mapping: {path}")
        merged.update(raw)
    for key, value in params.items():
        if key in {
            "config_path",
            "reservation_path",
            "audit_report_path",
            "release_output_dir",
            "catalog_dir",
            "source_artifact_id",
            "source_manifest_path",
            "report_output",
        }:
            continue
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            nested = dict(merged[key])
            nested.update(value)
            merged[key] = nested
        else:
            merged[key] = value
    return merged


def _load_reservation(path: str | Path | None, samples: list[Sample]) -> ReservationArtifact | None:
    if path:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        art = ReservationArtifact.from_dict(data)
        if not art.verify_digest():
            raise ValueError(f"reservation digest mismatch: {path}")
        return art
    # Recover from sample stamps when possible
    digest = None
    for sample in samples:
        digest = sample.labels.get("reservation_digest")
        if digest:
            break
    res_path = None
    for sample in samples:
        res_path = sample.labels.get("reservation_path")
        if res_path:
            break
    if res_path and Path(str(res_path)).is_file():
        art = ReservationArtifact.from_dict(
            json.loads(Path(str(res_path)).read_text(encoding="utf-8"))
        )
        if not art.verify_digest():
            raise ValueError(f"reservation digest mismatch: {res_path}")
        return art
    return None


@register_operator
class BuildDatasetV3Operator(ManifestOperator):
    """Stage-D build: consume reviewed+reservation(+audit), freeze Release.

    Params:
      config_path: configs/datasets/zh_asr_v3.yaml
      reservation_path: explicit reservation.json (else labels.reservation_path)
      audit_report_path: optional pseudo-audit JSON from review audit-pseudo
      release_output_dir: default data/releases
      catalog_dir: ArtifactCatalog root
      release_id / train_size: may override config build.*
    """

    name = "build_dataset_v3"
    version = "1.0.0"
    category = "quality"

    def run(self, samples: list[Sample], config: OperatorConfig) -> list[Sample]:
        params = dict(config.params or {})
        cfg_path = params.get("config_path")
        merged = _load_dataset_config(cfg_path, params)
        sampling_cfg = SamplingConfig.from_params(merged)
        if params.get("release_id"):
            sampling_cfg.release_id = str(params["release_id"])
        if params.get("train_size") is not None:
            sampling_cfg.train_size = int(params["train_size"])

        reservation = _load_reservation(params.get("reservation_path"), samples)
        if sampling_cfg.require_reservation and reservation is None:
            raise ValueError(
                "build_dataset_v3 requires reservation.json "
                "(pass reservation_path or prepare_dataset_v3 stamps)"
            )

        audit_report = None
        audit_path = params.get("audit_report_path")
        if audit_path:
            audit_report = json.loads(Path(str(audit_path)).read_text(encoding="utf-8"))

        run_dir = Path(config.run_dir) if config.run_dir else Path("runs") / "build_dataset_v3"
        run_dir.mkdir(parents=True, exist_ok=True)

        plan = build_sampling_plan(samples, reservation, sampling_cfg)
        atomic_write_json(run_dir / "sampling_plan.json", plan.to_dict())

        catalog_dir = params.get("catalog_dir") or "data/catalog"
        output_dir = params.get("release_output_dir") or "data/releases"

        try:
            result = publish_release_v3(
                samples,
                config=sampling_cfg,
                reservation=reservation,
                catalog_dir=catalog_dir,
                output_dir=output_dir,
                source_artifact_id=params.get("source_artifact_id"),
                source_manifest_path=params.get("source_manifest_path"),
                run_dir=run_dir,
                audit_report=audit_report,
                plan=plan,
            )
        except ReleaseBuildError:
            # Stamp samples with plan exclusions for diagnostics; re-raise
            from audio_engine.core.dataset_v3.sampling import apply_sampling_plan_to_samples

            stamped = apply_sampling_plan_to_samples(samples, plan, reservation)
            report = {
                "operator": self.full_name,
                "version": self.version,
                "status": "failed",
                "sampling": plan.to_dict(),
            }
            rpath = params.get("report_output") or (run_dir / "build_dataset_v3_report.json")
            atomic_write_json(Path(str(rpath)), report)
            for sample in stamped:
                sample.mark_failed(self.full_name, "release_build_refused")
            # Propagate failure to pipeline
            raise

        from audio_engine.core.dataset_v3.sampling import apply_sampling_plan_to_samples

        stamped = apply_sampling_plan_to_samples(samples, plan, reservation)
        report = {
            "operator": self.full_name,
            "version": self.version,
            "status": "ok",
            "config_path": str(cfg_path) if cfg_path else None,
            "release": result.to_dict(),
            "sampling_digest": result.sampling_digest,
            "reservation_digest": result.reservation_digest,
        }
        rpath = params.get("report_output") or (run_dir / "build_dataset_v3_report.json")
        atomic_write_json(Path(str(rpath)), report)

        for sample in stamped:
            sample.mark_completed(self.full_name)
            sample.add_lineage(
                self.full_name,
                self.version,
                {
                    "config_path": str(cfg_path) if cfg_path else None,
                    "release_id": result.release_id,
                    "sampling_digest": result.sampling_digest,
                    "reservation_digest": result.reservation_digest,
                    "idempotent_hit": result.idempotent_hit,
                },
            )
            sample.labels["release_id"] = result.release_id
            sample.labels["build_report_path"] = str(rpath)
        return stamped
