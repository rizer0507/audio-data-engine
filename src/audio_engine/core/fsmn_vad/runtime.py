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

# Required model_conf / top-level values from the full fenshen YAML.
REQUIRED_FENSHEN_VALUES: dict[str, Any] = {
    "model": "FsmnVADStreaming",
    "frontend": "WavFrontendOnline",
    "frontend_conf.fs": 16000,
    "model_conf.sample_rate": 16000,
    "model_conf.max_end_silence_time": 1100,
    "model_conf.max_start_silence_time": 4000,
    "model_conf.sil_to_speech_time_thres": 200,
    "model_conf.speech_to_sil_time_thres": 50,
    "model_conf.speech_noise_thres": 0.85,
    "model_conf.max_single_segment_time": 10000,
    "model_conf.lookback_time_start_point": 200,
    "model_conf.lookahead_time_end_point": 100,
}

# Stock FunASR defaults that must never be used as a silent production runtime.
DEFAULT_STOCK_MODEL_CONF = {
    "max_end_silence_time": 800,
    "speech_noise_thres": 0.6,
    "max_single_segment_time": 60000,
}


def _config_get(data: Any, key: str, default: Any = None) -> Any:
    cur = data
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def load_runtime_config_yaml(runtime_dir: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required to validate FSMN VAD config.yaml") from exc
    path = Path(runtime_dir) / "config.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid FSMN VAD config.yaml (not a mapping): {path}")
    return data


def assert_fenshen_config_data(data: dict[str, Any], *, source: str) -> dict[str, Any]:
    """Fail when YAML is not the full fenshen config (incl. stock defaults)."""
    model_conf = data.get("model_conf") or {}
    stock_hit = {
        k: model_conf.get(k)
        for k, v in DEFAULT_STOCK_MODEL_CONF.items()
        if model_conf.get(k) == v
    }
    if len(stock_hit) == len(DEFAULT_STOCK_MODEL_CONF):
        raise ValueError(
            f"Runtime config still has FunASR stock defaults {DEFAULT_STOCK_MODEL_CONF} "
            f"at {source}. Re-run scripts/prepare_fsmn_vad_runtime.py with "
            "configs/vad/fenshen_vad_config.yaml into an isolated runtime directory; "
            "do not modify the shared weight directory."
        )

    mismatches = {
        key: {"expected": expected, "actual": _config_get(data, key)}
        for key, expected in REQUIRED_FENSHEN_VALUES.items()
        if _config_get(data, key) != expected
    }
    if mismatches:
        raise ValueError(f"Not the required fenshen VAD configuration: {source}: {mismatches}")
    return data


def assert_fenshen_runtime_config(runtime_dir: str | Path) -> dict[str, Any]:
    runtime = Path(runtime_dir)
    data = load_runtime_config_yaml(runtime)
    return assert_fenshen_config_data(data, source=str(runtime / "config.yaml"))


def validate_loaded_vad_opts(expected_yaml: dict[str, Any], actual_opts: dict[str, Any]) -> None:
    model_conf = expected_yaml.get("model_conf") or {}
    if not actual_opts:
        raise RuntimeError("Loaded FSMN model has no vad_opts; cannot verify fenshen config")
    mismatches = {
        key: {"expected": value, "actual": actual_opts.get(key)}
        for key, value in model_conf.items()
        if actual_opts.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            f"Loaded FSMN options do not match the full fenshen model_conf: {mismatches}"
        )


def check_generate_options_against_fenshen(
    generate_options: dict[str, Any] | None,
    expected_yaml: dict[str, Any],
) -> None:
    """Reject runtime overrides that silently undo fenshen model_conf."""
    if not generate_options:
        return
    model_conf = expected_yaml.get("model_conf") or {}
    for key in ("max_end_silence_time", "speech_noise_thres", "max_single_segment_time"):
        if key in generate_options and generate_options[key] != model_conf.get(key):
            raise ValueError(
                f"generate_options.{key}={generate_options[key]!r} conflicts with "
                f"fenshen model_conf.{key}={model_conf.get(key)!r}"
            )


def validate_runtime_model_dir(runtime_dir: str | Path) -> Path:
    """Reject missing local assets and non-fenshen config.yaml before FunASR loads."""
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
    assert_fenshen_runtime_config(runtime)
    return runtime


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
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required to prepare FSMN VAD runtime") from exc
    config_data = yaml.safe_load(config_src.read_text(encoding="utf-8"))
    if not isinstance(config_data, dict):
        raise ValueError(f"Invalid fenshen VAD config (not a mapping): {config_src}")
    assert_fenshen_config_data(config_data, source=str(config_src))

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
            assert_fenshen_runtime_config(runtime)
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
    assert_fenshen_runtime_config(runtime)

    return {
        "runtime_dir": str(runtime.resolve()),
        "config_sha256": runtime_cfg_sha,
        "model_pt_sha256": file_sha256(runtime / "model.pt"),
        "am_mvn_sha256": file_sha256(runtime / "am.mvn"),
        "source_model_dir": str(source.resolve()),
        "fenshen_config_path": str(config_src.resolve()),
        "fenshen_required_values": dict(REQUIRED_FENSHEN_VALUES),
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
