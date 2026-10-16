"""quality.evidence_join — align segment snapshot + N ASR routes + DNSMOS by identity."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from audio_engine.core.artifacts import atomic_write_json
from audio_engine.core.manifest import Manifest
from audio_engine.core.operator import ManifestOperator, OperatorConfig
from audio_engine.core.registry import register_operator
from audio_engine.core.sample import Sample
from audio_engine.core.selection_v3.input_contract import original_audio_sha256


def _index_by_id(samples: list[Sample], *, label: str) -> dict[str, Sample]:
    indexed: dict[str, Sample] = {}
    duplicates: list[str] = []
    for sample in samples:
        sid = str(sample.id).strip()
        if not sid:
            raise ValueError(f"{label}: empty sample id")
        digest = original_audio_sha256(sample)
        if not digest:
            raise ValueError(f"{label}: missing original_audio_sha256 for {sid}")
        if sid in indexed:
            duplicates.append(sid)
            continue
        indexed[sid] = sample
    if duplicates:
        raise ValueError(f"{label}: duplicate ids={sorted(set(duplicates))[:20]}")
    return indexed


def _snapshot_id(sample: Sample) -> str:
    for container in (sample.labels, sample.quality):
        for key in (
            "segment_input_snapshot_id",
            "input_snapshot_id",
            "vad_processing_version",
            "vad_snapshot_id",
        ):
            value = container.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
    return ""


@register_operator
class EvidenceJoinOperator(ManifestOperator):
    """Merge VAD segment base with ASR transcripts and DNSMOS quality by join_key.

    This is an explicit barrier node for the 045 production chain. It never
    invents DNSMOS scores, never reads quality_sidecar_*, and never treats
    missing branch coverage as success.
    """

    name = "evidence_join"
    version = "1.0.0"
    category = "quality"

    def run(self, samples: list[Sample], config: OperatorConfig) -> list[Sample]:
        params = dict(config.params)
        id_policy = str(params.get("id_policy") or "exact").strip().lower()
        if id_policy not in {"exact", "left"}:
            raise ValueError("evidence_join id_policy must be 'exact' or 'left'")

        base = _index_by_id(samples, label="segments")
        expected = set(base)
        report: dict[str, Any] = {
            "base_count": len(samples),
            "join_key": "id+original_audio_sha256",
            "id_policy": id_policy,
            "asr_routes": [],
            "dnsmos": None,
            "snapshot_mismatches": [],
            "aligned": False,
        }

        base_snapshots = {sid: _snapshot_id(sample) for sid, sample in base.items()}
        updated = {sid: sample.model_copy(deep=True) for sid, sample in base.items()}

        for item in params.get("manifests") or []:
            model = str(item.get("model") or "").strip()
            path = Path(str(item.get("path") or ""))
            if not model or not path:
                raise ValueError("evidence_join manifests entries need model and path")
            if not path.is_file():
                raise ValueError(f"ASR route manifest missing: {path}")
            incoming = Manifest.load(path).samples
            indexed = _index_by_id(incoming, label=f"asr:{model}")
            missing = expected - indexed.keys()
            extra = indexed.keys() - expected
            hash_mismatches: list[str] = []
            snapshot_mismatches: list[str] = []
            for sid in expected & indexed.keys():
                left = original_audio_sha256(updated[sid])
                right = original_audio_sha256(indexed[sid])
                if left and right and left != right:
                    hash_mismatches.append(sid)
                left_snap = base_snapshots.get(sid) or ""
                right_snap = _snapshot_id(indexed[sid])
                if left_snap and right_snap and left_snap != right_snap:
                    snapshot_mismatches.append(sid)
            route_report = {
                "model": model,
                "path": str(path),
                "count": len(incoming),
                "missing_ids": sorted(missing),
                "extra_ids": sorted(extra),
                "hash_mismatches": hash_mismatches,
                "snapshot_mismatches": snapshot_mismatches,
            }
            report["asr_routes"].append(route_report)
            if hash_mismatches:
                raise ValueError(
                    f"ASR route {model}: hash mismatches on {hash_mismatches[:10]}"
                )
            if snapshot_mismatches:
                report["snapshot_mismatches"].extend(snapshot_mismatches)
                raise ValueError(
                    f"ASR route {model}: input snapshot mismatches on "
                    f"{snapshot_mismatches[:10]}"
                )
            if id_policy == "exact" and (missing or extra):
                raise ValueError(
                    f"ASR route {model}: coverage gap missing={len(missing)} "
                    f"extra={len(extra)}"
                )
            if missing and id_policy == "left":
                for sid in missing:
                    sample = updated[sid]
                    transcripts = dict(sample.transcripts or {})
                    transcripts[model] = {
                        "text": None,
                        "status": "missing",
                        "error": "route_coverage_gap",
                    }
                    sample.transcripts = transcripts
            for sid, incoming_sample in indexed.items():
                if sid not in updated:
                    continue
                sample = updated[sid]
                transcripts = dict(sample.transcripts or {})
                src_tx = dict(incoming_sample.transcripts or {})
                if model in src_tx:
                    transcripts[model] = src_tx[model]
                elif len(src_tx) == 1:
                    transcripts[model] = next(iter(src_tx.values()))
                elif "text" in (incoming_sample.labels or {}):
                    transcripts[model] = {
                        "text": incoming_sample.labels.get("text"),
                        "status": incoming_sample.labels.get("asr_status")
                        or "success_text",
                    }
                else:
                    for _alias, payload in src_tx.items():
                        transcripts[model] = payload
                        break
                    if model not in transcripts:
                        transcripts[model] = {
                            "text": None,
                            "status": "missing",
                            "error": "no_transcript_payload",
                        }
                sample.transcripts = transcripts
                labels = dict(sample.labels or {})
                labels["evidence_asr_routes"] = sorted(
                    set(list(labels.get("evidence_asr_routes") or []) + [model])
                )
                sample.labels = labels

        dnsmos_path = (
            params.get("dnsmos_manifest")
            or params.get("dnsmos_path")
            or os.environ.get("AUDIO_ENGINE_DNSMOS_MANIFEST")
        )
        if not dnsmos_path:
            raise ValueError(
                "evidence_join requires dnsmos_manifest (or AUDIO_ENGINE_DNSMOS_MANIFEST)"
            )
        dnsmos_file = Path(str(dnsmos_path))
        if not dnsmos_file.is_file():
            raise ValueError(f"DNSMOS manifest missing: {dnsmos_file}")
        if "quality_sidecar" in dnsmos_file.name:
            raise ValueError(
                "evidence_join refuses quality_sidecar_* inputs; "
                "use datasets/stage1/quality/dnsmos_<batch>.parquet"
            )
        dnsmos_samples = Manifest.load(dnsmos_file).samples
        dnsmos_index = _index_by_id(dnsmos_samples, label="dnsmos")
        missing = expected - dnsmos_index.keys()
        extra = dnsmos_index.keys() - expected
        hash_mismatches = []
        for sid in expected & dnsmos_index.keys():
            left = original_audio_sha256(updated[sid])
            right = original_audio_sha256(dnsmos_index[sid])
            if left and right and left != right:
                hash_mismatches.append(sid)
        dnsmos_report = {
            "path": str(dnsmos_file),
            "count": len(dnsmos_samples),
            "missing_ids": sorted(missing),
            "extra_ids": sorted(extra),
            "hash_mismatches": hash_mismatches,
            "failed_scores": 0,
            "success_scores": 0,
        }
        if hash_mismatches:
            raise ValueError(f"DNSMOS hash mismatches on {hash_mismatches[:10]}")
        if id_policy == "exact" and (missing or extra):
            raise ValueError(
                f"DNSMOS coverage gap missing={len(missing)} extra={len(extra)}"
            )
        for sid, incoming in dnsmos_index.items():
            if sid not in updated:
                continue
            sample = updated[sid]
            quality = dict(sample.quality or {})
            quality.update(dict(incoming.quality or {}))
            sample.quality = quality
            status = str(quality.get("dnsmos_status") or "")
            if status == "success":
                dnsmos_report["success_scores"] += 1
            elif status:
                dnsmos_report["failed_scores"] += 1
            labels = dict(sample.labels or {})
            labels["evidence_dnsmos"] = True
            labels["evidence_dnsmos_status"] = status or "unknown"
            sample.labels = labels
        if missing and id_policy == "left":
            for sid in missing:
                sample = updated[sid]
                quality = dict(sample.quality or {})
                quality.setdefault("dnsmos_status", "missing")
                quality.setdefault("dnsmos_error", "dnsmos_coverage_gap")
                sample.quality = quality
        report["dnsmos"] = dnsmos_report

        if not report["asr_routes"]:
            raise ValueError("evidence_join requires at least one ASR route manifest")

        report["aligned"] = True
        report["output_count"] = len(updated)
        if config.run_dir is not None:
            report_path = (
                Path(config.run_dir) / "reports" / f"{config.step_name}_alignment.json"
            )
            atomic_write_json(report_path, report)

        return [updated[str(sample.id).strip()] for sample in samples if str(sample.id).strip() in updated]
