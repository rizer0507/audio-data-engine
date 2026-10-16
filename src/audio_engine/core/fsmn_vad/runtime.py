"""Prepare a dedicated FSMN VAD runtime directory with fenshen config.yaml."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Any

# Fingerprints from ph_ai_model_asr_rt_cpu docs (migration baseline).
EXPECTED_FINGERPRINTS = {
    "model.pt": "b3be75be477f0780277f3bae0fe489f48718f585f3a6e45d7dd1fbb1a4255fc5",
    "am.mvn": "6820fef9687708c4fc3fab2530179c8fcea6262daa25514380056cd8f6eb1754",
    "configuration.json": "7bce8867e37d55c3dd8f672695ced18077a2be199ea529a5d432d5350fc0acba",
    "fenshen_vad_config.yaml": "9057c765320b48d5a1247528dd8c5ea9fa818d4d6e3d073db0036ec4b81b79e6",
}


def validate_runtime_model_dir(runtime_dir: str | Path) -> Path:
    """Reject missing local assets before FunASR can interpret a path as a hub ID."""
    raw = str(runtime_dir).strip()
    if not raw:
        raise ValueError("Set FSMN_VAD_RUNTIME_DIR/model_runtime_dir to a prepared local directory")
    runtime = Path(raw).expanduser().resolve()
    if not runtime.is_dir():
        raise FileNotFoundError(
            f"FSMN VAD runtime directory not found: {runtime}. "
            "Set FSMN_VAD_RUNTIME_DIR/model_runtime_dir to the actual prepared local directory; "
            "model hub download is not supported here."
        )
    required = ("config.yaml", "configuration.json", "model.pt", "am.mvn")
    missing = [name for name in required if not (runtime / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Incomplete FSMN VAD runtime {runtime}: missing {', '.join(missing)}. "
            "Run scripts/prepare_fsmn_vad_runtime.py first."
        )
    return runtime


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def prepare_runtime_model_dir(
    *,
    source_model_dir: str | Path,
    fenshen_config_path: str | Path,
    runtime_dir: str | Path,
    overwrite: bool = False,
    verify_fingerprints: bool = True,
) -> dict[str, Any]:
    """Copy model assets and overlay fenshen YAML as ``config.yaml``.

    Never writes into the source asset directory. If ``runtime_dir`` already
    exists and ``overwrite`` is False, reuse it after verifying config hash.
    """
    source = Path(source_model_dir)
    config_src = Path(fenshen_config_path)
    runtime = Path(runtime_dir)

    if not source.is_dir():
        raise FileNotFoundError(f"FSMN VAD source model dir not found: {source}")
    if not config_src.is_file():
        raise FileNotFoundError(f"fenshen VAD config not found: {config_src}")

    config_sha = file_sha256(config_src)
    if verify_fingerprints:
        expected = EXPECTED_FINGERPRINTS["fenshen_vad_config.yaml"]
        if config_sha != expected:
            raise ValueError(
                f"fenshen config sha256 mismatch: got {config_sha}, expected {expected}"
            )
        for name, expected_sha in (
            ("model.pt", EXPECTED_FINGERPRINTS["model.pt"]),
            ("am.mvn", EXPECTED_FINGERPRINTS["am.mvn"]),
            ("configuration.json", EXPECTED_FINGERPRINTS["configuration.json"]),
        ):
            path = source / name
            if not path.is_file():
                raise FileNotFoundError(f"Missing model asset: {path}")
            got = file_sha256(path)
            if got != expected_sha:
                raise ValueError(f"{name} sha256 mismatch: got {got}, expected {expected_sha}")

    if runtime.exists():
        existing_cfg = runtime / "config.yaml"
        if not overwrite:
            if not existing_cfg.is_file():
                raise FileExistsError(
                    f"runtime dir exists without config.yaml: {runtime}; "
                    "pass overwrite=True to rebuild"
                )
            existing_sha = file_sha256(existing_cfg)
            if existing_sha != config_sha:
                raise ValueError(
                    f"existing runtime config sha {existing_sha} != source {config_sha}; "
                    "pass overwrite=True to rebuild"
                )
        else:
            shutil.rmtree(runtime)
            shutil.copytree(source, runtime)
            shutil.copyfile(config_src, runtime / "config.yaml")
    else:
        runtime.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, runtime)
        shutil.copyfile(config_src, runtime / "config.yaml")

    runtime_cfg_sha = file_sha256(runtime / "config.yaml")
    if runtime_cfg_sha != config_sha:
        raise RuntimeError("Failed to overlay fenshen config.yaml into runtime dir")

    return {
        "runtime_dir": str(runtime.resolve()),
        "config_sha256": runtime_cfg_sha,
        "model_pt_sha256": file_sha256(runtime / "model.pt"),
        "am_mvn_sha256": file_sha256(runtime / "am.mvn"),
        "source_model_dir": str(source.resolve()),
        "fenshen_config_path": str(config_src.resolve()),
    }


def dependency_versions() -> dict[str, str]:
    """Best-effort dependency fingerprint; missing packages recorded as absent."""
    out: dict[str, str] = {}
    for name in ("funasr", "torch", "torchaudio", "numpy"):
        try:
            mod = __import__(name)
            out[name] = str(getattr(mod, "__version__", "unknown"))
        except Exception:
            out[name] = "absent"
    return out
