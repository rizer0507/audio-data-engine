"""EfficientAT AudioSet tagging wrapper (candidate samples only)."""

from __future__ import annotations

import csv
import hashlib
import sys
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]


@dataclass
class EfficientAtConfig:
    policy_version: str = "efficientat_v1"
    input_audio_key: str = "resampled_16k"
    model_name: str = "mn10_as"
    checkpoint_path: str = "EfficientAT-main/resources/mn10_as_mAP_471.pt"
    allow_download: bool = False
    efficientat_root: str = "EfficientAT-main"
    labels_csv: str = "EfficientAT-main/metadata/class_labels_indices.csv"
    device: str = "cuda"
    sample_rate: int = 32000
    window_size: int = 800
    hop_size: int = 320
    n_mels: int = 128
    top_k: int = 10
    label_buckets: dict[str, list[str]] = field(default_factory=dict)
    audio_event: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> EfficientAtConfig:
        data = dict(raw or {})
        buckets = data.get("label_buckets") or {}
        event = data.get("audio_event") or {}
        return cls(
            policy_version=str(data.get("policy_version") or "efficientat_v1"),
            input_audio_key=str(data.get("input_audio_key") or "resampled_16k"),
            model_name=str(data.get("model_name") or "mn10_as"),
            checkpoint_path=str(data.get("checkpoint_path") or ""),
            allow_download=bool(data.get("allow_download", False)),
            efficientat_root=str(data.get("efficientat_root") or "EfficientAT-main"),
            labels_csv=str(
                data.get("labels_csv")
                or "EfficientAT-main/metadata/class_labels_indices.csv"
            ),
            device=str(data.get("device") or "cuda"),
            sample_rate=int(data.get("sample_rate") or 32000),
            window_size=int(data.get("window_size") or 800),
            hop_size=int(data.get("hop_size") or 320),
            n_mels=int(data.get("n_mels") or 128),
            top_k=int(data.get("top_k") or 10),
            label_buckets={
                str(k): [str(x) for x in (v or [])]
                for k, v in dict(buckets).items()
            },
            audio_event={str(k): float(v) for k, v in dict(event).items()},
        )


def load_efficientat_config(path: str | Path | None) -> EfficientAtConfig:
    if not path:
        return EfficientAtConfig()
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"efficientat config must be a mapping: {path}")
    return EfficientAtConfig.from_mapping(raw)


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (REPO_ROOT / p).resolve()


def _file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _load_label_names(csv_path: Path) -> list[str]:
    names: list[str] = []
    with csv_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            names.append(str(row.get("display_name") or "").strip())
    return names


def derive_audio_event_state(
    probs: dict[str, float | None],
    *,
    status: str,
    thresholds: dict[str, float] | None = None,
) -> str:
    """Map bucket probs → audio_event_state. Never forges on failed."""
    if status == "failed":
        return "failed"
    if status == "skipped":
        return "unknown"
    thr = {
        "speech_like_min": 0.35,
        "human_background_babble_min": 0.30,
        "human_background_conversation_min": 0.35,
        "non_speech_env_min": 0.30,
        "non_speech_speech_max": 0.20,
        "mixed_speech_min": 0.25,
        "mixed_noise_min": 0.25,
        **(thresholds or {}),
    }

    def _p(key: str) -> float:
        val = probs.get(key)
        return float(val) if val is not None else 0.0

    speech = max(_p("speech_prob"), _p("conversation_prob") * 0.8)
    babble = _p("speech_babble_prob")
    conversation = _p("conversation_prob")
    env = max(
        _p("environment_noise_prob"),
        _p("music_prob"),
        _p("vehicle_prob"),
        _p("mechanical_prob"),
    )
    if babble >= float(thr["human_background_babble_min"]) or conversation >= float(
        thr["human_background_conversation_min"]
    ):
        if speech >= float(thr["mixed_speech_min"]) and env >= float(thr["mixed_noise_min"]):
            return "mixed"
        return "human_background"
    if speech >= float(thr["speech_like_min"]) and env < float(thr["mixed_noise_min"]):
        return "speech_like"
    if env >= float(thr["non_speech_env_min"]) and speech <= float(
        thr["non_speech_speech_max"]
    ):
        return "non_speech_environment"
    if speech >= float(thr["mixed_speech_min"]) and env >= float(thr["mixed_noise_min"]):
        return "mixed"
    return "unknown"


class EfficientAtSession:
    """Lazy EfficientAT session. Does not download unless allow_download=true."""

    def __init__(self, cfg: EfficientAtConfig) -> None:
        self.cfg = cfg
        self._model = None
        self._mel = None
        self._device = None
        self._labels: list[str] = []
        self._name_to_idx: dict[str, int] = {}
        self.model_digest = ""
        self._autocast = None
        self._torch = None

    def _ensure_path(self) -> Path:
        root = _resolve(self.cfg.efficientat_root)
        if not root.is_dir():
            raise FileNotFoundError(f"EfficientAT root missing: {root}")
        text = str(root)
        if text not in sys.path:
            sys.path.insert(0, text)
        return root

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from torch import autocast

        self._autocast = autocast
        self._torch = torch
        self._ensure_path()
        labels_path = _resolve(self.cfg.labels_csv)
        if not labels_path.is_file():
            raise FileNotFoundError(f"EfficientAT labels missing: {labels_path}")
        self._labels = _load_label_names(labels_path)
        self._name_to_idx = {name: i for i, name in enumerate(self._labels) if name}

        from helpers.utils import NAME_TO_WIDTH
        from models.dymn.model import get_model as get_dymn
        from models.mn.model import get_model as get_mobilenet
        from models.preprocess import AugmentMelSTFT

        device_name = str(self.cfg.device or "cpu")
        if device_name.startswith("cuda") and not torch.cuda.is_available():
            device_name = "cpu"
        self._device = torch.device(device_name)

        model_name = str(self.cfg.model_name or "mn10_as")
        width = NAME_TO_WIDTH(model_name)
        ckpt = _resolve(self.cfg.checkpoint_path) if self.cfg.checkpoint_path else None

        if ckpt is not None and ckpt.is_file():
            if model_name.startswith("dymn"):
                model = get_dymn(width_mult=width, pretrained_name=None)
            else:
                model = get_mobilenet(width_mult=width, pretrained_name=None)
            state = torch.load(str(ckpt), map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state, strict=False)
            self.model_digest = _file_digest(ckpt)
        elif self.cfg.allow_download:
            if model_name.startswith("dymn"):
                model = get_dymn(width_mult=width, pretrained_name=model_name)
            else:
                model = get_mobilenet(width_mult=width, pretrained_name=model_name)
            self.model_digest = f"downloaded:{model_name}"
        else:
            raise FileNotFoundError(
                "EfficientAT checkpoint missing and allow_download=false: "
                f"{ckpt}"
            )

        model.to(self._device)
        model.eval()
        mel = AugmentMelSTFT(
            n_mels=int(self.cfg.n_mels),
            sr=int(self.cfg.sample_rate),
            win_length=int(self.cfg.window_size),
            hopsize=int(self.cfg.hop_size),
        )
        mel.to(self._device)
        mel.eval()
        self._model = model
        self._mel = mel

    def _bucket_prob(self, preds, bucket: str) -> float:
        names = self.cfg.label_buckets.get(bucket) or []
        values: list[float] = []
        for name in names:
            idx = self._name_to_idx.get(name)
            if idx is None:
                continue
            values.append(float(preds[idx]))
        return max(values) if values else 0.0

    def analyze(self, audio_path: str | Path) -> dict[str, Any]:
        """Return EfficientAT quality fields. Failure never forges probabilities."""
        empty_probs = {
            "speech_prob": None,
            "conversation_prob": None,
            "speech_babble_prob": None,
            "environment_noise_prob": None,
            "music_prob": None,
            "vehicle_prob": None,
            "mechanical_prob": None,
            "silence_tag_prob": None,
        }
        base = {
            "efficientat_status": "failed",
            **empty_probs,
            "top_k_labels": None,
            "efficientat_model_name": self.cfg.model_name,
            "efficientat_model_digest": self.model_digest,
            "efficientat_config_version": self.cfg.policy_version,
            "audio_event_state": "failed",
            "efficientat_error": None,
        }
        path = Path(audio_path) if audio_path else None
        if path is None or not path.is_file():
            base["efficientat_error"] = f"audio_unreadable:{audio_path}"
            return base
        try:
            self._load()
            import librosa
            import numpy as np
            import torch

            waveform, _ = librosa.core.load(
                str(path), sr=int(self.cfg.sample_rate), mono=True
            )
            wave = torch.from_numpy(np.asarray(waveform, dtype=np.float32)[None, :]).to(
                self._device
            )
            use_cuda = self._device is not None and self._device.type == "cuda"
            ctx = self._autocast(device_type="cuda") if use_cuda else nullcontext()
            with torch.no_grad(), ctx:
                spec = self._mel(wave)
                preds, _features = self._model(spec.unsqueeze(0))
            preds = torch.sigmoid(preds.float()).squeeze().cpu().numpy()
            probs = {
                "speech_prob": self._bucket_prob(preds, "speech"),
                "conversation_prob": self._bucket_prob(preds, "conversation"),
                "speech_babble_prob": self._bucket_prob(preds, "speech_babble"),
                "environment_noise_prob": self._bucket_prob(preds, "environment_noise"),
                "music_prob": self._bucket_prob(preds, "music"),
                "vehicle_prob": self._bucket_prob(preds, "vehicle"),
                "mechanical_prob": self._bucket_prob(preds, "mechanical"),
                "silence_tag_prob": self._bucket_prob(preds, "silence_tag"),
            }
            order = np.argsort(preds)[::-1]
            top_k = []
            for idx in order[: int(self.cfg.top_k)]:
                name = self._labels[int(idx)] if int(idx) < len(self._labels) else str(idx)
                top_k.append({"name": name, "prob": float(preds[int(idx)])})
            event_state = derive_audio_event_state(
                probs,
                status="success",
                thresholds=self.cfg.audio_event,
            )
            return {
                "efficientat_status": "success",
                **probs,
                "top_k_labels": top_k,
                "efficientat_model_name": self.cfg.model_name,
                "efficientat_model_digest": self.model_digest,
                "efficientat_config_version": self.cfg.policy_version,
                "audio_event_state": event_state,
                "efficientat_error": None,
            }
        except Exception as exc:  # noqa: BLE001
            base["efficientat_model_digest"] = self.model_digest
            base["efficientat_error"] = str(exc)
            base["audio_event_state"] = "failed"
            return base


def skipped_efficientat_fields(cfg: EfficientAtConfig | None = None) -> dict[str, Any]:
    policy = cfg.policy_version if cfg else "efficientat_v1"
    model_name = cfg.model_name if cfg else "mn10_as"
    return {
        "efficientat_status": "skipped",
        "speech_prob": None,
        "conversation_prob": None,
        "speech_babble_prob": None,
        "environment_noise_prob": None,
        "music_prob": None,
        "vehicle_prob": None,
        "mechanical_prob": None,
        "silence_tag_prob": None,
        "top_k_labels": None,
        "efficientat_model_name": model_name,
        "efficientat_model_digest": "",
        "efficientat_config_version": policy,
        "audio_event_state": "unknown",
        "efficientat_error": None,
    }
