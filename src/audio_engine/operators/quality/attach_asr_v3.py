"""Attach registered ASR executions (2N routes for N≥3 families) to the frozen reservation."""
from dataclasses import asdict
from pathlib import Path

from audio_engine.core.catalog import ArtifactCatalog
from audio_engine.core.dataset_v3.audit_plan import digest_payload
from audio_engine.core.operator import ManifestOperator, OperatorConfig
from audio_engine.core.registry import register_operator
from audio_engine.core.selection_v3.config import SelectionV3Config
from audio_engine.core.selection_v3.input_contract import apply_contract_to_samples, original_audio_sha256
from audio_engine.operators.quality.aggregate_manifests import AggregateManifestsOperator


@register_operator
class AttachAsrV3Operator(ManifestOperator):
    name = "attach_asr_v3"
    category = "quality"
    version = "1.1.0"

    def run(self, samples, config):
        cfg = SelectionV3Config.from_yaml(Path(config.params["config_path"]))
        expected = cfg.expected_total_runs
        if len(cfg.runs) != expected:
            raise ValueError(
                f"attach_asr_v3 requires {expected} registered run identities "
                f"(2N for N={cfg.configured_family_count} families), got {len(cfg.runs)}"
            )
        if not samples or any(not s.labels.get("reservation_digest") for s in samples):
            raise ValueError("freeze raw reservation before attaching ASR")
        snapshot_digest = digest_payload({s.id: original_audio_sha256(s) for s in sorted(samples, key=lambda s: s.id)})
        catalog = ArtifactCatalog(config.params.get("catalog_dir") or "data/catalog")
        joins, seen_artifacts, seen_executions = [], set(), set()
        for run in cfg.runs:
            if not run.artifact_id or not run.execution_id:
                raise ValueError("every ASR run requires artifact_id and execution_id")
            if run.artifact_id in seen_artifacts or (run.family, run.execution_id) in seen_executions:
                raise ValueError("copied artifact/execution cannot serve as an independent second run")
            seen_artifacts.add(run.artifact_id)
            seen_executions.add((run.family, run.execution_id))
            record = catalog.get(run.artifact_id, verify=True)
            if record.kind != "manifest" or record.producer.run_id != run.execution_id:
                raise ValueError("ASR artifact execution provenance mismatch")
            if run.input_audio_digest != snapshot_digest:
                raise ValueError("ASR input audio snapshot differs from frozen reservation")
            declared = record.metadata.get("run_identity") or {}
            for key in ("family", "model_checkpoint_digest", "decode_config_digest", "prompt_digest", "input_audio_digest", "input_audio_key", "created_at"):
                if declared.get(key) != getattr(run, key):
                    raise ValueError(f"ASR artifact metadata mismatch: {run.run_id}/{key}")
            joins.append({"model": run.transcript_key, "path": record.uri})
        params = {"manifests": joins, "id_policy": "left", "hash_policy": "original_audio", "require_hashes": True}
        joined = AggregateManifestsOperator().run(samples, OperatorConfig(
            params=params, run_dir=config.run_dir, step_name=config.step_name))
        updated, conservation, alignment = apply_contract_to_samples(joined, cfg)
        verified_digest = digest_payload([asdict(r) for r in cfg.runs])
        for sample in updated:
            sample.labels["run_identities_digest"] = verified_digest
            sample.labels["run_identities_verified"] = True
            sample.labels["asr_snapshot_digest"] = snapshot_digest
            sample.labels["configured_family_count"] = cfg.configured_family_count
        if config.run_dir:
            from audio_engine.core.artifacts import atomic_write_json
            # Legacy filename kept for existing tooling; also write route_contract.json.
            atomic_write_json(Path(config.run_dir) / "eight_route_contract.json", alignment.to_dict())
            atomic_write_json(Path(config.run_dir) / "run_identities.json", [asdict(r) for r in cfg.runs])
            atomic_write_json(
                Path(config.run_dir) / "route_contract.json",
                {
                    **alignment.to_dict(),
                    "configured_family_count": cfg.configured_family_count,
                    "expected_total_runs": expected,
                },
            )
        return updated
