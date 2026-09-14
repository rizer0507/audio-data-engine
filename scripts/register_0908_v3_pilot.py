"""Pilot-register 0908-30000 three-family ASR runs and write runs: into dataset config.

Digests are derived from parquet model/version + file identity (pilot provenance).
Replace with true execution identities when available.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml
from audio_engine.core.catalog import ArtifactCatalog, ProducerRecord
from audio_engine.core.dataset_v3.audit_plan import digest_payload
from audio_engine.core.manifest import Manifest
from audio_engine.core.selection_v3.config import RunIdentity
from audio_engine.core.selection_v3.input_contract import (
    align_run_manifest,
    original_audio_sha256,
    validate_base_snapshot,
)

BATCH = "0908-30000"
ROOT = Path(".")
CLEANED = ROOT / f"datasets/stage1/cleaned/cleaned_{BATCH}.parquet"
DATASET_CFG = ROOT / "configs/datasets/zh_asr_v3_0908_30000.yaml"
ID_DIR = ROOT / "configs/datasets/run_identities/0908-30000"
CATALOG = ROOT / "data/catalog"

ALIASES = [
    ("glm", "glm-asr-1"),
    ("glm", "glm-asr-2"),
    ("qwen", "qwen3-asr-1"),
    ("qwen", "qwen3-asr-2"),
    ("sensevoice", "sensevoice-asr-1"),
    ("sensevoice", "sensevoice-asr-2"),
]


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _pilot_identity(family: str, alias: str, asr_path: Path) -> RunIdentity:
    import pyarrow.parquet as pq

    table = pq.read_table(asr_path, columns=["transcripts"]).slice(0, 1)
    raw = table["transcripts"][0].as_py()
    if isinstance(raw, str):
        raw = json.loads(raw)
    entry = raw.get(alias) if isinstance(raw, dict) else None
    if not isinstance(entry, dict):
        entry = next(iter(raw.values())) if isinstance(raw, dict) else {}
    model = str(entry.get("model") or alias)
    version = str(entry.get("version") or "unknown")
    mtime = int(asr_path.stat().st_mtime)
    file_digest = _sha(f"{asr_path.resolve()}|{asr_path.stat().st_size}|{mtime}")
    created = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
    return RunIdentity(
        run_id=alias,
        family=family,
        transcript_key=alias,
        execution_id=f"pilot_{alias}_{file_digest[:12]}",
        model_checkpoint_digest=_sha(f"pilot_ckpt|{model}|{version}"),
        decode_config_digest=_sha(f"pilot_decode|{alias}|{file_digest}"),
        prompt_digest=_sha(f"pilot_prompt|{alias}|default"),
        input_audio_digest="",  # filled from cleaned base
        input_audio_key="resampled_16k",
        created_at=created,
    )


def main() -> None:
    if not CLEANED.exists():
        raise SystemExit(f"missing cleaned base: {CLEANED}")
    ID_DIR.mkdir(parents=True, exist_ok=True)
    CATALOG.mkdir(parents=True, exist_ok=True)

    base = list(Manifest.load(CLEANED))
    indexed = validate_base_snapshot(base)
    audio_digest = digest_payload(
        {s.id: original_audio_sha256(s) for s in sorted(base, key=lambda s: s.id)}
    )
    catalog = ArtifactCatalog(CATALOG)
    runs: list[dict] = []

    for family, alias in ALIASES:
        asr_path = ROOT / f"datasets/stage1/asr/{alias}_asr_{BATCH}.parquet"
        if not asr_path.exists():
            raise SystemExit(f"missing ASR: {asr_path}")
        identity = _pilot_identity(family, alias, asr_path)
        identity.input_audio_digest = audio_digest

        identity_path = ID_DIR / f"{alias}_identity.yaml"
        registered_path = ID_DIR / f"{alias}_registered.yaml"
        identity_path.write_text(
            yaml.safe_dump(
                {k: v for k, v in identity.__dict__.items() if k != "artifact_id"},
                allow_unicode=True,
                sort_keys=False,
            ),
            encoding="utf-8",
        )

        incoming = list(Manifest.load(asr_path))
        alignment = align_run_manifest(
            indexed,
            incoming,
            transcript_key=identity.transcript_key,
            path=str(asr_path),
            id_policy="left",
        )
        if alignment["original_audio_sha256_unchecked"] or alignment["extra_ids"]:
            raise SystemExit(
                f"{alias}: alignment failed unchecked={alignment['original_audio_sha256_unchecked']} "
                f"extra={alignment['extra_ids']}"
            )

        metadata = {
            "run_identity": {
                k: v for k, v in identity.__dict__.items() if k != "artifact_id"
            }
        }
        record = catalog.register_file(
            asr_path,
            kind="manifest",
            producer=ProducerRecord(
                pipeline="asr_external_execution",
                run_id=identity.execution_id,
            ),
            metadata=metadata,
        )
        identity.artifact_id = record.artifact_id
        payload = yaml.safe_dump(identity.__dict__, allow_unicode=True, sort_keys=False)
        if registered_path.exists() and registered_path.read_text(encoding="utf-8") != payload:
            registered_path = ID_DIR / f"{alias}_registered_rerun.yaml"
        registered_path.write_text(payload, encoding="utf-8")
        runs.append(identity.__dict__)
        print(f"registered {alias} -> {record.artifact_id}")

    raw = yaml.safe_load(DATASET_CFG.read_text(encoding="utf-8")) or {}
    raw["runs"] = runs
    DATASET_CFG.write_text(
        yaml.safe_dump(raw, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    print(f"wrote {len(runs)} runs -> {DATASET_CFG}")


if __name__ == "__main__":
    main()
