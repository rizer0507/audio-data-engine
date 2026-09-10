from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from audio_engine.core.manifest import Manifest
from audio_engine.core.artifacts import atomic_write_json
from audio_engine.core.operator import ManifestOperator, OperatorConfig
from audio_engine.core.registry import register_operator
from audio_engine.core.sample import Sample
from audio_engine.core.selection_v3.config import SelectionV3Config
from audio_engine.core.selection_v3.input_contract import (
    original_audio_sha256,
    validate_base_snapshot,
)


@register_operator
class AggregateManifestsOperator(ManifestOperator):
    """Join independently produced ASR manifests by sample id (+ original audio hash).

    ``id_policy``:
      - ``exact`` (default): join id set must equal the base id set.
      - ``left``: extra ids are reported and ignored; missing base ids are retained
        with explicit missing status for retry/completeness gates.

    ``hash_policy``:
      - ``original_audio`` (default): compare ``original_audio_sha256`` lineage when
        present; do not require pad-file sha256 to equal original.
      - ``sha256``: legacy — compare ``sample.sha256`` when present on both sides.

    Optional ``selection_config_path`` / ``model_families`` enables an eight-route
    integrity section in the alignment report (012-A).
    """

    name = "aggregate_manifests"
    version = "2.2.0"
    category = "quality"

    def run(self, samples: list[Sample], config: OperatorConfig) -> list[Sample]:
        validate_base_snapshot(samples)
        base = {sample.id: sample.model_copy(deep=True) for sample in samples}
        if len(base) != len(samples):
            raise ValueError("aggregate input contains duplicate ids")
        id_policy = str(config.params.get("id_policy", "exact")).strip().lower()
        if id_policy not in {"exact", "left"}:
            raise ValueError("aggregate_manifests id_policy must be 'exact' or 'left'")
        hash_policy = str(
            config.params.get("hash_policy", "original_audio")
        ).strip().lower()
        if hash_policy not in {"original_audio", "sha256"}:
            raise ValueError(
                "aggregate_manifests hash_policy must be 'original_audio' or 'sha256'"
            )
        expected = set(base)
        alignment_report: dict[str, Any] = {
            "base_count": len(samples),
            "join_key": "id+original_audio_sha256"
            if hash_policy == "original_audio"
            else "id+sha256",
            "id_policy": id_policy,
            "hash_policy": hash_policy,
            "sha256_policy": "must_match_when_present_on_both_sides",
            "manifests": [],
        }

        def write_report(*, aligned: bool) -> None:
            alignment_report["aligned"] = aligned
            if config.run_dir is not None:
                report_path = (
                    Path(config.run_dir) / "reports" / f"{config.step_name}_alignment.json"
                )
                atomic_write_json(report_path, alignment_report)

        for item in config.params.get("manifests", []):
            model, path = str(item["model"]), Path(item["path"])
            incoming = Manifest.load(path).samples
            indexed = {sample.id: sample for sample in incoming}
            if len(indexed) != len(incoming):
                raise ValueError(f"manifest {path} contains duplicate ids")
            missing, extra = expected - indexed.keys(), indexed.keys() - expected
            hash_mismatches: list[str] = []
            unchecked_hashes = 0
            for sample_id in expected & indexed.keys():
                if hash_policy == "original_audio":
                    left = original_audio_sha256(base[sample_id])
                    right = original_audio_sha256(indexed[sample_id])
                else:
                    left = str(base[sample_id].sha256 or "").strip()
                    right = str(indexed[sample_id].sha256 or "").strip()
                if left and right and left != right:
                    hash_mismatches.append(sample_id)
                elif not left or not right:
                    unchecked_hashes += 1
            # Legacy field names kept for existing consumers
            sha_mismatches = hash_mismatches
            if config.params.get("require_hashes") and unchecked_hashes:
                raise ValueError("v3 ASR join requires audio hashes on both sides")
            report_item = {
                "model": model,
                "path": str(path.resolve()),
                "count": len(incoming),
                "missing_ids": len(missing),
                "extra_ids": len(extra),
                "sha256_mismatches": len(sha_mismatches),
                "sha256_unchecked": unchecked_hashes,
                "original_audio_sha256_mismatches": len(hash_mismatches)
                if hash_policy == "original_audio"
                else None,
                "missing_id_examples": sorted(missing)[:20],
                "extra_id_examples": sorted(extra)[:20],
                "sha256_mismatch_examples": sha_mismatches[:20],
            }
            alignment_report["manifests"].append(report_item)
            if id_policy == "exact" and (missing or extra):
                write_report(aligned=False)
                raise ValueError(
                    f"manifest {path} ids are not aligned: missing={len(missing)}, extra={len(extra)}"
                )
            if sha_mismatches:
                write_report(aligned=False)
                raise ValueError(
                    f"manifest {path} audio hashes are not aligned: "
                    f"hash_mismatches={len(sha_mismatches)}, examples={sha_mismatches[:5]}"
                )
            for sample_id, target in base.items():
                if sample_id not in indexed:
                    target.transcripts[model] = {"text": None, "status": "missing",
                                                 "extra": {"source_manifest": str(path)}}
                    continue
                text = indexed[sample_id].get_transcript_text(model)
                if model in target.transcripts and not config.params.get("overwrite", False):
                    raise ValueError(f"transcript `{model}` already exists for id {sample_id}")
                source = indexed[sample_id].transcripts.get(model, {})
                # Preserve explicit failed status from join side when present
                incoming_entry = source if isinstance(source, dict) else {}
                if model not in indexed[sample_id].transcripts:
                    incoming_entry = {"text": None, "status": "missing"}
                # If join sample itself failed ASR at operator level and has no text entry,
                # mark failed rather than silent empty success.
                if model not in indexed[sample_id].transcripts and indexed[sample_id].errors:
                    asr_failed = any(
                        indexed[sample_id].status.get(k) == "failed"
                        for k in indexed[sample_id].errors
                    )
                    if asr_failed:
                        incoming_entry = {
                            "text": "",
                            "status": "failed",
                            "extra": {
                                "inference_status": "failed",
                                "source_manifest": str(path),
                            },
                        }
                merged_extra = dict(incoming_entry.get("extra") or {})
                merged_extra["source_manifest"] = str(path)
                target.transcripts[model] = {
                    **incoming_entry,
                    "text": incoming_entry.get("text", text) if "text" in incoming_entry else text,
                    "extra": merged_extra,
                }
                # Propagate original audio hash onto base when missing
                if not original_audio_sha256(target):
                    ohash = original_audio_sha256(indexed[sample_id])
                    if ohash:
                        target.labels.setdefault("original_audio_sha256", ohash)
                target.add_lineage(
                    self.full_name, self.version, {"model": model, "source_manifest": str(path)}
                )

        # Eight-route integrity appendix (optional)
        eight_route = self._eight_route_integrity(base, config.params)
        if eight_route is not None:
            alignment_report["eight_route_integrity"] = eight_route

        write_report(aligned=True)
        return [base[sample.id] for sample in samples]

    def _eight_route_integrity(
        self, base: dict[str, Sample], params: dict[str, Any]
    ) -> dict[str, Any] | None:
        cfg_path = params.get("selection_config_path") or params.get("config_path")
        families = params.get("model_families")
        if not cfg_path and not families:
            return None
        if cfg_path:
            raw = yaml.safe_load(Path(str(cfg_path)).read_text(encoding="utf-8")) or {}
            selection = SelectionV3Config.from_params(raw if isinstance(raw, dict) else {})
        else:
            selection = SelectionV3Config.from_params({"model_families": families, **{
                k: params[k]
                for k in (
                    "teacher_families",
                    "target_family",
                    "expected_runs_per_family",
                    "rule_version",
                    "engine",
                )
                if k in params
            }})
        expected_keys = selection.all_transcript_keys()
        present_counts = {key: 0 for key in expected_keys}
        missing_counts = {key: 0 for key in expected_keys}
        failed_counts = {key: 0 for key in expected_keys}
        from audio_engine.core.selection_v3.input_contract import classify_run_status
        from audio_engine.core.selection_v3.types import (
            RUN_STATUS_FAILED,
            RUN_STATUS_MISSING,
        )

        for sample in base.values():
            for key in expected_keys:
                status = classify_run_status(sample, key)
                if status == RUN_STATUS_MISSING:
                    missing_counts[key] += 1
                elif status == RUN_STATUS_FAILED:
                    failed_counts[key] += 1
                else:
                    present_counts[key] += 1
        return {
            "rule_version": selection.rule_version,
            "expected_runs": expected_keys,
            "families": selection.model_families,
            "present_success_counts": present_counts,
            "missing_counts": missing_counts,
            "failed_counts": failed_counts,
            "base_count": len(base),
        }
