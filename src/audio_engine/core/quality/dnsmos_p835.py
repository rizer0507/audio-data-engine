"""DNSMOS P.835 quality scoring helpers (vendor-isolated).

Scores SIG/BAK/OVRL via a fixed ONNX primary model. Thresholds for
``noise_band`` / ``noise_risk`` are applied separately so they can be
recomputed without re-running the model.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from audio_engine.core.evidence.types import (
    DNSMOS_STATUS_FAILED,
    DNSMOS_STATUS_SUCCESS,
    NOISE_BAND_CLEAN,
    NOISE_BAND_MODERATE,
    NOISE_BAND_NOISY,
    NOISE_BAND_UNKNOWN,
    QUALITY_POLICY_VERSION,
)

# Official DNSMOS primary model expects 16 kHz mono float32.
DNSMOS_SAMPLE_RATE = 16000
DNSMOS_INPUT_LENGTH_SEC = 9.01
DNSMOS_PREPROCESS_VERSION = "p835_official_nonpersonalized_v2"


@dataclass
class DnsmosScores:
    sig: float | None
    bak: float | None
    ovrl: float | None
    status: str
    error: str | None = None
    short_audio_strategy: str | None = None
    model_digest: str = ""
    preprocess_version: str = DNSMOS_PREPROCESS_VERSION


@dataclass
class DnsmosRisk:
    noise_band: str
    noise_risk: bool | None
    quality_policy_version: str = QUALITY_POLICY_VERSION


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@lru_cache(maxsize=32)
def _model_digest_for_stat(path: str, identity: tuple[int, ...]) -> str:
    return file_digest(Path(path))


def model_file_digest(path: Path) -> str:
    """Avoid re-reading model weights for every sample; invalidate on replacement/edit."""
    resolved = path.resolve()
    stat = resolved.stat()
    return _model_digest_for_stat(
        str(resolved), (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns),
    )


@dataclass(frozen=True)
class DnsmosRuntimeOptions:
    """Explicit session thread budget; source-level concurrency stays in the pipeline."""

    intra_op_num_threads: int = 2
    inter_op_num_threads: int = 1
    allow_spinning: bool = False
    window_batch_size: int = 4

    def __post_init__(self) -> None:
        for name in ("intra_op_num_threads", "inter_op_num_threads", "window_batch_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"DNSMOS runtime.{name} must be a positive integer")
        if not isinstance(self.allow_spinning, bool):
            raise ValueError("DNSMOS runtime.allow_spinning must be a boolean")  # noqa: TRY004

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> DnsmosRuntimeOptions:
        if config is None:
            return cls()
        if not isinstance(config, dict):
            raise ValueError("DNSMOS runtime must be a mapping")  # noqa: TRY004
        unknown = set(config) - {
            "intra_op_num_threads", "inter_op_num_threads", "allow_spinning", "window_batch_size",
        }
        if unknown:
            raise ValueError(f"Unknown DNSMOS runtime options: {sorted(unknown)}")
        return cls(**config)


def prepare_audio_for_dnsmos(
    audio: np.ndarray,
    sample_rate: int,
    *,
    target_sr: int = DNSMOS_SAMPLE_RATE,
    input_length_sec: float = DNSMOS_INPUT_LENGTH_SEC,
) -> tuple[np.ndarray, str]:
    """Return mono float32 buffer for ONNX + strategy label (does not mutate source file)."""
    mono = np.asarray(audio, dtype=np.float64)
    if mono.ndim > 1:
        mono = mono.mean(axis=1)
    if sample_rate <= 0 or not np.all(np.isfinite(mono)) or len(mono) == 0:
        raise ValueError("invalid or empty audio buffer")
    if sample_rate != target_sr:
        import librosa
        mono = librosa.resample(mono, orig_sr=sample_rate, target_sr=target_sr)

    mono = mono.astype(np.float32)
    desired = int(input_length_sec * target_sr)
    strategy = "exact"
    if len(mono) == 0:
        raise ValueError("empty audio buffer")
    if len(mono) < desired:
        # Official-style repeat/pad to fill the fixed window; recorded as strategy only.
        while len(mono) < desired:
            mono = np.concatenate((mono, mono))
        strategy = "repeat_pad_to_window"
    elif len(mono) > desired:
        # Hop through and average later in scorer; here return full signal.
        strategy = "multi_window"
    return mono, strategy


def derive_noise_risk(
    *,
    bak: float | None,
    ovrl: float | None,
    t_bak: float,
    t_ovrl: float,
    calibrated: bool,
    status: str,
) -> DnsmosRisk:
    """Derive noise_band / noise_risk from persisted scores.

    When not calibrated, band/risk are ``unknown`` / ``null`` even if scores exist.
    When any score missing/invalid → risk null, band unknown.
    """
    if status != DNSMOS_STATUS_SUCCESS or bak is None or ovrl is None:
        return DnsmosRisk(noise_band=NOISE_BAND_UNKNOWN, noise_risk=None)
    if not calibrated:
        return DnsmosRisk(noise_band=NOISE_BAND_UNKNOWN, noise_risk=None)
    try:
        bak_f = float(bak)
        ovrl_f = float(ovrl)
    except (TypeError, ValueError):
        return DnsmosRisk(noise_band=NOISE_BAND_UNKNOWN, noise_risk=None)
    if not np.isfinite(bak_f) or not np.isfinite(ovrl_f):
        return DnsmosRisk(noise_band=NOISE_BAND_UNKNOWN, noise_risk=None)

    risk = bool(bak_f < t_bak or ovrl_f < t_ovrl)
    # Ordered bands use the same BAK/OVRL thresholds (explicit in config).
    # noisy: risk true; moderate: below clean floors but not noisy; clean: both high.
    # Additional clean floors optional — default: not noisy and both >= thresholds.
    if risk:
        band = NOISE_BAND_NOISY
    else:
        # moderate band when scores sit near thresholds (within 0.5) else clean
        near = (bak_f < t_bak + 0.5) or (ovrl_f < t_ovrl + 0.5)
        band = NOISE_BAND_MODERATE if near else NOISE_BAND_CLEAN
    return DnsmosRisk(noise_band=band, noise_risk=risk)


def derive_noise_band_explicit(
    *,
    bak: float | None,
    ovrl: float | None,
    clean_bak: float,
    clean_ovrl: float,
    moderate_bak: float,
    moderate_ovrl: float,
    calibrated: bool,
    status: str,
) -> DnsmosRisk:
    """Ordered clean/moderate/noisy bands from explicit floors + risk OR rule."""
    if status != DNSMOS_STATUS_SUCCESS or bak is None or ovrl is None:
        return DnsmosRisk(noise_band=NOISE_BAND_UNKNOWN, noise_risk=None)
    if not calibrated:
        return DnsmosRisk(noise_band=NOISE_BAND_UNKNOWN, noise_risk=None)
    bak_f = float(bak)
    ovrl_f = float(ovrl)
    if not np.isfinite(bak_f) or not np.isfinite(ovrl_f):
        return DnsmosRisk(noise_band=NOISE_BAND_UNKNOWN, noise_risk=None)

    # Risk uses moderate floors as T_bak / T_ovrl by default.
    risk = bool(bak_f < moderate_bak or ovrl_f < moderate_ovrl)
    if bak_f >= clean_bak and ovrl_f >= clean_ovrl:
        band = NOISE_BAND_CLEAN
    elif bak_f >= moderate_bak and ovrl_f >= moderate_ovrl:
        band = NOISE_BAND_MODERATE
    else:
        band = NOISE_BAND_NOISY
    return DnsmosRisk(noise_band=band, noise_risk=risk)


class DnsmosP835Session:
    """Lazy ONNX session for the fixed primary DNSMOS P.835 model."""

    def __init__(
        self, model_path: Path, *, providers: list[str] | None = None,
        runtime_options: DnsmosRuntimeOptions | None = None,
    ):
        self.runtime_options = runtime_options or DnsmosRuntimeOptions()
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"DNSMOS model file missing: {self.model_path}. "
                "Place the official P.835 primary ONNX at the configured path."
            )
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime is required for quality.dnsmos; "
                "install optional dependency: pip install onnxruntime"
            ) from exc
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.intra_op_num_threads = self.runtime_options.intra_op_num_threads
        opts.inter_op_num_threads = self.runtime_options.inter_op_num_threads
        spinning = "1" if self.runtime_options.allow_spinning else "0"
        opts.add_session_config_entry("session.intra_op.allow_spinning", spinning)
        opts.add_session_config_entry("session.inter_op.allow_spinning", spinning)
        self._session = ort.InferenceSession(
            str(self.model_path),
            sess_options=opts,
            providers=providers or ["CPUExecutionProvider"],
        )
        self.model_digest = model_file_digest(self.model_path)
        model_input = self._session.get_inputs()[0]
        self.input_name = model_input.name
        shape = getattr(model_input, "shape", [1])
        self.window_batch_size = (
            self.runtime_options.window_batch_size if not isinstance(shape[0], int) else 1
        )
        logger.info(
            "DNSMOS session ready: intra_threads={} inter_threads={} spinning={} window_batch={}",
            self.runtime_options.intra_op_num_threads,
            self.runtime_options.inter_op_num_threads,
            self.runtime_options.allow_spinning,
            self.window_batch_size,
        )

    def score_array(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> DnsmosScores:
        try:
            mono, strategy = prepare_audio_for_dnsmos(audio, sample_rate)
            desired = int(DNSMOS_INPUT_LENGTH_SEC * DNSMOS_SAMPLE_RATE)
            n_hops = int(np.floor(len(mono) / DNSMOS_SAMPLE_RATE) - DNSMOS_INPUT_LENGTH_SEC) + 1
            chunks = [mono[i * DNSMOS_SAMPLE_RATE:i * DNSMOS_SAMPLE_RATE + desired]
                      for i in range(max(1, n_hops))]

            sigs: list[float] = []
            baks: list[float] = []
            ovrls: list[float] = []
            batch_size = getattr(self, "window_batch_size", 1)
            for left in range(0, len(chunks), batch_size):
                batch = chunks[left:left + batch_size]
                inp = np.zeros((len(batch), desired), dtype=np.float32)
                for index, chunk in enumerate(batch):
                    inp[index, :len(chunk)] = chunk
                out = self._session.run(None, {self.input_name: inp})
                # Preserve calibration per window, then average across all windows.
                arr = np.asarray(out[0]).reshape(len(batch), -1)
                if arr.shape[1] < 3:
                    raise RuntimeError(f"unexpected DNSMOS output shape: {arr.shape}")
                if not np.all(np.isfinite(arr[:, :3])):
                    raise RuntimeError("non-finite DNSMOS output")
                # Microsoft DNSMOS non-personalized P.835 calibration per window.
                sigs.extend(float(np.polyval([-0.08397278, 1.22083953, 0.0052439], row[0]))
                            for row in arr)
                baks.extend(float(np.polyval([-0.13166888, 1.60915514, -0.39604546], row[1]))
                            for row in arr)
                ovrls.extend(float(np.polyval([-0.06766283, 1.11546468, 0.04602535], row[2]))
                             for row in arr)
            return DnsmosScores(
                sig=float(np.mean(sigs)),
                bak=float(np.mean(baks)),
                ovrl=float(np.mean(ovrls)),
                status=DNSMOS_STATUS_SUCCESS,
                short_audio_strategy=strategy,
                model_digest=self.model_digest,
            )
        except Exception as exc:  # noqa: BLE001 — isolate per-utterance failures
            return DnsmosScores(
                sig=None,
                bak=None,
                ovrl=None,
                status=DNSMOS_STATUS_FAILED,
                error=str(exc),
                model_digest=getattr(self, "model_digest", ""),
            )


def scores_to_quality_dict(
    scores: DnsmosScores,
    risk: DnsmosRisk,
) -> dict[str, Any]:
    return {
        "dnsmos_sig": scores.sig,
        "dnsmos_bak": scores.bak,
        "dnsmos_ovrl": scores.ovrl,
        "dnsmos_status": scores.status,
        "dnsmos_model_digest": scores.model_digest,
        "dnsmos_preprocess_version": scores.preprocess_version,
        "dnsmos_short_audio_strategy": scores.short_audio_strategy,
        "dnsmos_error": scores.error,
        "noise_band": risk.noise_band,
        "noise_risk": risk.noise_risk,
        "quality_policy_version": risk.quality_policy_version,
    }
