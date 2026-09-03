from __future__ import annotations

from pathlib import Path

from audio_engine.core.manifest import Manifest
from audio_engine.core.artifacts import atomic_write_json
from audio_engine.core.operator import ManifestOperator, OperatorConfig
from audio_engine.core.registry import register_operator
from audio_engine.core.sample import Sample


@register_operator
class AggregateManifestsOperator(ManifestOperator):
    """Join independently produced ASR manifests by sample id.

    ``id_policy``:
      - ``exact`` (default): join id set must equal the base id set.
      - ``left``: extra ids on the join side are ignored; missing base ids still fail.
        Use this when attaching a larger ASR dump onto a smaller eval set.
    """

    name = "aggregate_manifests"
    version = "2.1.0"
    category = "quality"

    def run(self, samples: list[Sample], config: OperatorConfig) -> list[Sample]:
        base = {sample.id: sample.model_copy(deep=True) for sample in samples}
        if len(base) != len(samples):
            raise ValueError("aggregate input contains duplicate ids")
        id_policy = str(config.params.get("id_policy", "exact")).strip().lower()
        if id_policy not in {"exact", "left"}:
            raise ValueError("aggregate_manifests id_policy must be 'exact' or 'left'")
        expected = set(base)
        alignment_report: dict = {
            "base_count": len(samples),
            "join_key": "id",
            "id_policy": id_policy,
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
            sha_mismatches = sorted(
                sample_id
                for sample_id in expected & indexed.keys()
                if base[sample_id].sha256
                and indexed[sample_id].sha256
                and base[sample_id].sha256 != indexed[sample_id].sha256
            )
            unchecked_hashes = sum(
                1
                for sample_id in expected & indexed.keys()
                if not base[sample_id].sha256 or not indexed[sample_id].sha256
            )
            report_item = {
                "model": model,
                "path": str(path.resolve()),
                "count": len(incoming),
                "missing_ids": len(missing),
                "extra_ids": len(extra),
                "sha256_mismatches": len(sha_mismatches),
                "sha256_unchecked": unchecked_hashes,
                "missing_id_examples": sorted(missing)[:20],
                "extra_id_examples": sorted(extra)[:20],
                "sha256_mismatch_examples": sha_mismatches[:20],
            }
            alignment_report["manifests"].append(report_item)
            if missing or (extra and id_policy == "exact"):
                write_report(aligned=False)
                raise ValueError(
                    f"manifest {path} ids are not aligned: missing={len(missing)}, extra={len(extra)}"
                )
            if sha_mismatches:
                write_report(aligned=False)
                raise ValueError(
                    f"manifest {path} audio hashes are not aligned: "
                    f"sha256_mismatches={len(sha_mismatches)}, examples={sha_mismatches[:5]}"
                )
            for sample_id, target in base.items():
                text = indexed[sample_id].get_transcript_text(model)
                if model in target.transcripts and not config.params.get("overwrite", False):
                    raise ValueError(f"transcript `{model}` already exists for id {sample_id}")
                source = indexed[sample_id].transcripts.get(model, {})
                target.transcripts[model] = {
                    **(source if isinstance(source, dict) else {}),
                    "text": text,
                    "extra": {
                        **(source.get("extra", {}) if isinstance(source, dict) else {}),
                        "source_manifest": str(path),
                    },
                }
                target.add_lineage(
                    self.full_name, self.version, {"model": model, "source_manifest": str(path)}
                )
        write_report(aligned=True)
        return [base[sample.id] for sample in samples]
