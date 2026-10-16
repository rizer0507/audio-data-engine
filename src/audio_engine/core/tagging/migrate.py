"""044 scaffolding: migrate legacy tagging configs to free_tags_v2 layout."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from audio_engine.core.selection_v3.config import SelectionV3Config
from audio_engine.core.tagging.paths import find_repo_root, resolve_tagging_path

FREE_TAGS_V2_POLICY = "free_tags_v2"
EVIDENCE_POLICY_V2 = "zh_asr_families_v2"
DEFAULT_EVIDENCE_REL = "configs/tagging/evidence/zh_asr_families_v2.yaml"

POLICY_WORK_DIR_MAP: dict[str, str] = {
    "selection_multi_tag_v1": "runs/tagging/{batch_id}/selection_multi_tag_v1",
    "free_tags_v2": "runs/tagging/{batch_id}/free_tags_v2",
}


def _load_yaml(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"YAML must be a mapping: {path}")
    return raw


def _dump_yaml(data: dict[str, Any]) -> str:
    return yaml.safe_dump(
        data,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    )


def _repo_relative(path: str, repo_root: Path) -> str:
    """Return ``path`` relative to the repo root when possible (posix style)."""
    p = Path(path)
    try:
        return p.resolve().relative_to(repo_root.resolve()).as_posix()
    except (ValueError, OSError):
        return p.as_posix()


def _dnsmos_weak_sig_from_path(path: Path) -> float | None:
    if not path.is_file():
        return None
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        return None
    thresholds = raw.get("thresholds") or {}
    if isinstance(thresholds, dict) and thresholds.get("weak_sig") is not None:
        return float(thresholds["weak_sig"])
    return None


def _resolve_write_path(target: Path) -> tuple[Path, str | None]:
    """Pick a write path; never silently overwrite an existing file."""
    if not target.exists():
        return target, None
    migrated = target.with_name(f"{target.stem}_migrated{target.suffix}")
    if not migrated.exists():
        return migrated, f"target exists; writing {migrated.name} instead of overwrite"
    stamped = target.with_name(
        f"{target.stem}_migrated_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}{target.suffix}"
    )
    return stamped, f"target exists; writing {stamped.name}"


def _backup_existing(path: Path) -> Path | None:
    if not path.is_file():
        return None
    bak = path.with_suffix(path.suffix + ".bak")
    if bak.exists():
        bak = path.with_suffix(
            path.suffix + f".bak.{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        )
    shutil.copy2(path, bak)
    return bak


def evidence_payload_from_selection(
    selection: SelectionV3Config,
    *,
    repo_root: Path,
    voicemail_patterns_path: str = "configs/tagging/lexicons/voicemail_patterns_v1.yaml",
) -> dict[str, Any]:
    """Build independent evidence YAML content from SelectionV3 effective values."""
    dnsmos_cfg = selection.dnsmos_decision_config_path or "configs/quality/dnsmos_decision_v2_2.yaml"
    dnsmos_resolved = resolve_tagging_path(dnsmos_cfg, base_dir=repo_root, repo_root=repo_root)
    weak_sig = _dnsmos_weak_sig_from_path(dnsmos_resolved)
    if weak_sig is None:
        weak_sig = float(getattr(selection, "dnsmos_weak_sig", 2.8) or 2.8)

    echo: dict[str, Any] = {}
    if selection.classify_text_echo:
        from audio_engine.core.evidence.classify_text import default_echo_spec_for_family

        for family, table in selection.classify_text_echo.items():
            if isinstance(table, dict):
                echo[str(family)] = dict(table)
                continue
            # Loaded EchoTable: recover the file spec from its source paths.
            spec: dict[str, str] = {}
            for src in getattr(table, "source_paths", ()) or ():
                rel = _repo_relative(str(src), repo_root)
                key = "blank_exact" if "blank_exact" in Path(rel).name else "asr_config"
                spec.setdefault(key, rel)
            echo[str(family)] = spec or default_echo_spec_for_family(str(family))

    return {
        "policy_version": EVIDENCE_POLICY_V2,
        "model_families": dict(selection.model_families),
        "expected_runs_per_family": int(selection.expected_runs_per_family),
        "run_aliases": dict(selection.run_aliases),
        "similarity": {"family_threshold": float(selection.family_threshold)},
        "classify_text": {
            "policy": selection.classify_text_policy,
            "keep_digits": bool(selection.classify_text_keep_digits),
            "echo_missing": selection.classify_text_echo_missing,
            "echo": echo,
        },
        "punctuation_to_strip": list(selection.punctuation_to_strip),
        "critical_short_responses": list(selection.critical_short_responses),
        "short_polarity_max_han_chars": int(
            getattr(selection, "short_polarity_max_han_chars", 4) or 4
        ),
        # Effective lexicon values inlined so the migrated evidence is self-contained.
        "semantic": {
            "negative": list(selection.negative_phrases or []),
            "positive": list(selection.positive_phrases or []),
            "critical_tokens": list(selection.critical_tokens or []),
            "profanity_or_reject": list(selection.profanity_or_reject or []),
            "filler": list(selection.filler_phrases or []),
            "affirmation": list(selection.affirmation_phrases or []),
        },
        "dnsmos_weak_sig": weak_sig,
        "voicemail_patterns_path": voicemail_patterns_path,
        "audio_energy": {
            "config_path": (
                f"configs/quality/{selection.audio_energy_policy_version}.yaml"
                if selection.audio_energy_policy_version
                else "configs/quality/audio_energy_v1.yaml"
            ),
        },
        "silero_vad": {
            "config_path": selection.silero_vad_config_path
            or "configs/quality/silero_vad_v1.yaml",
        },
        "dnsmos_decision": {
            "enabled": bool(selection.dnsmos_decision_enabled),
            "config_path": dnsmos_cfg,
        },
        "dnsmos_scoring": {"config_path": "configs/quality/dnsmos_p835.yaml"},
    }


def free_tags_tagging_payload_from_legacy(
    old_raw: dict[str, Any],
    *,
    evidence_config_rel: str,
    policy_version: str = FREE_TAGS_V2_POLICY,
) -> dict[str, Any]:
    """Strip selection_config and point at independent evidence + free_tags work dir."""
    drop_keys = {
        "selection_config",
        "selection_config_path",
    }
    out: dict[str, Any] = {}
    for key, value in old_raw.items():
        if key in drop_keys:
            continue
        out[key] = value

    out["policy_version"] = policy_version
    out["evidence_config"] = evidence_config_rel
    out.pop("selection_config", None)
    out.pop("selection_config_path", None)
    out["work_dir_template"] = POLICY_WORK_DIR_MAP.get(
        policy_version, "runs/tagging/{batch_id}/{policy_version}"
    )

    acoustic = dict(out.get("acoustic") or {})
    if acoustic.get("voicemail_patterns_path", "").startswith("configs/selection/"):
        acoustic["voicemail_patterns_path"] = "configs/tagging/lexicons/voicemail_patterns_v1.yaml"
    out["acoustic"] = acoustic

    lexicons = dict(out.get("lexicons") or {})
    normalized: dict[str, Any] = {}
    for lid, spec in lexicons.items():
        if isinstance(spec, dict):
            path = str(spec.get("path") or "")
            if path.startswith("configs/selection/voicemail_patterns_v1.yaml"):
                normalized[lid] = "configs/tagging/lexicons/voicemail_patterns_v1.yaml"
            else:
                normalized[lid] = spec
        elif isinstance(spec, str) and spec.startswith("configs/selection/voicemail_patterns_v1.yaml"):
            normalized[lid] = "configs/tagging/lexicons/voicemail_patterns_v1.yaml"
        else:
            normalized[lid] = spec
    out["lexicons"] = normalized
    return out


def migrate_tagging_config(
    old_path: str | Path,
    out_path: str | Path,
    *,
    dry_run: bool = False,
    evidence_out: str | Path | None = None,
) -> dict[str, Any]:
    """Migrate legacy tagging YAML (with selection_config) to free_tags_v2 files."""
    old_p = Path(old_path).resolve()
    out_p = Path(out_path).resolve()
    repo_root = find_repo_root(old_p.parent)

    old_raw = _load_yaml(old_p)
    selection_key = old_raw.get("selection_config") or old_raw.get("selection_config_path")
    if not selection_key:
        raise ValueError(f"{old_p}: expected selection_config for migration source")

    selection_resolved = resolve_tagging_path(
        str(selection_key), base_dir=old_p.parent, repo_root=repo_root
    )
    selection = SelectionV3Config.from_yaml(selection_resolved)

    evidence_rel = str(
        evidence_out
        or old_raw.get("evidence_config")
        or DEFAULT_EVIDENCE_REL
    )
    if Path(evidence_rel).is_absolute():
        evidence_path = Path(evidence_rel)
    else:
        evidence_path = (repo_root / evidence_rel).resolve()
        evidence_rel = str(Path(evidence_rel).as_posix())

    evidence_body = evidence_payload_from_selection(selection, repo_root=repo_root)
    tagging_body = free_tags_tagging_payload_from_legacy(
        old_raw, evidence_config_rel=evidence_rel
    )

    dropped = sorted(
        k
        for k in ("selection_config", "selection_config_path")
        if k in old_raw
    )
    report_path = out_p.with_suffix(".migrate_report.json")

    evidence_write, evidence_note = _resolve_write_path(evidence_path)
    tagging_write, tagging_note = _resolve_write_path(out_p)

    report: dict[str, Any] = {
        "ok": True,
        "dry_run": dry_run,
        "source_tagging_config": str(old_p.as_posix()),
        "source_selection_config": str(selection_resolved.as_posix()),
        "targets": {
            "tagging_config": str(tagging_write.as_posix()),
            "evidence_config": str(evidence_write.as_posix()),
            "difference_report": str(report_path.as_posix()),
        },
        "policy_version": tagging_body.get("policy_version"),
        "evidence_policy_version": evidence_body.get("policy_version"),
        "mapped_from_selection": [
            "model_families",
            "expected_runs_per_family",
            "run_aliases",
            "similarity.family_threshold",
            "classify_text",
            "punctuation_to_strip",
            "critical_short_responses",
            "dnsmos_weak_sig",
            "audio_energy/silero_vad/dnsmos paths",
        ],
        "dropped_tagging_keys": dropped,
        "excluded_from_evidence": [
            "queues A-E",
            "category",
            "gold quotas",
            "review_queue",
            "selection business block",
        ],
        "notes": [n for n in (evidence_note, tagging_note) if n],
        "files_written": [],
        "files_skipped": [],
    }

    diff = {
        "selection_config_path": str(selection_resolved.as_posix()),
        "evidence_fields": sorted(evidence_body.keys()),
        "tagging_keys": sorted(tagging_body.keys()),
        "selection_only_omitted": report["excluded_from_evidence"],
    }
    report["difference"] = diff

    if dry_run:
        report["would_write"] = {
            "evidence_config": evidence_body,
            "tagging_config": tagging_body,
        }
        return report

    for target, body in (
        (evidence_write, evidence_body),
        (tagging_write, tagging_body),
    ):
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            bak = _backup_existing(target)
            if bak:
                report.setdefault("backups", []).append(str(bak.as_posix()))
        target.write_text(_dump_yaml(body), encoding="utf-8")
        report["files_written"].append(str(target.as_posix()))

    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    report["files_written"].append(str(report_path.as_posix()))
    return report


def migrate_classified_to_neutral_evidence(samples_path: str | Path) -> dict[str, Any]:
    """Stub: strip legacy auto category fields into labels.legacy_auto (044 §8.2)."""
    path = Path(samples_path)
    return {
        "ok": False,
        "stub": True,
        "path": str(path.as_posix()),
        "message": (
            "Not implemented: will map category/type/queue fields to "
            "labels.legacy_auto and record evidence availability."
        ),
        "planned_labels_namespace": "labels.legacy_auto",
        "evidence_availability": "unknown",
    }


def migrate_manual_tag_state(
    src_work: str | Path,
    dst_work: str | Path,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Copy tag_state / manual_ops with explicit policy mapping."""
    src = Path(src_work).resolve()
    dst = Path(dst_work).resolve()
    src_policy = src.name
    dst_policy = dst.name
    mapping = {
        "source_work_dir": str(src.as_posix()),
        "target_work_dir": str(dst.as_posix()),
        "source_policy": src_policy,
        "target_policy": dst_policy,
        "known_policy_map": POLICY_WORK_DIR_MAP,
    }
    copies: list[dict[str, str]] = []
    for rel in ("tag_state.json", "tag_state.sqlite3", "manual_ops.jsonl"):
        s = src / rel
        if not s.is_file():
            continue
        t = dst / rel
        copies.append({"from": str(s.as_posix()), "to": str(t.as_posix())})
        if not dry_run:
            dst.mkdir(parents=True, exist_ok=True)
            if t.exists():
                _backup_existing(t)
            if rel == "tag_state.sqlite3":
                import sqlite3
                from contextlib import closing
                with closing(sqlite3.connect(s)) as source, closing(sqlite3.connect(t)) as target:
                    source.backup(target)
            else:
                shutil.copy2(s, t)

    return {
        "ok": True,
        "dry_run": dry_run,
        "policy_mapping": mapping,
        "copied": copies,
        "skipped_missing": [
            rel for rel in ("tag_state.json", "manual_ops.jsonl") if not (src / rel).is_file()
        ],
    }
