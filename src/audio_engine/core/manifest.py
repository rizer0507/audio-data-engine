from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import soundfile as sf
from loguru import logger

from audio_engine.core.artifacts import sample_digest
from audio_engine.core.sample import Sample

AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3", ".m4a"}
PCM_EXTENSIONS = {".pcm", ".raw"}


def file_sha256(path: Path, chunk_size: int = 65536) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def probe_audio(path: Path) -> dict[str, Any]:
    """Read basic audio metadata; return empty dict on failure."""
    try:
        info = sf.info(str(path))
        return {
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "duration": info.duration,
            "valid": True,
        }
    except Exception as exc:
        logger.debug("Cannot probe {}: {}", path, exc)
        return {"valid": False, "error": str(exc)}


class Manifest:
    """Load / save / query a collection of Samples."""

    def __init__(self, samples: list[Sample] | None = None):
        self.samples = samples or []

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)

    @classmethod
    def from_samples(cls, samples: list[Sample]) -> Manifest:
        return cls(samples)

    @classmethod
    def load(cls, path: str | Path) -> Manifest:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Manifest not found: {path}")

        if path.suffix == ".jsonl":
            return cls._load_jsonl(path)
        if path.suffix == ".parquet":
            return cls._load_parquet(path)
        raise ValueError(f"Unsupported manifest format: {path.suffix}")

    @classmethod
    def _load_jsonl(cls, path: Path) -> Manifest:
        samples: list[Sample] = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                samples.append(Sample(**data))
        return cls(samples)

    @classmethod
    def _load_parquet(cls, path: Path) -> Manifest:
        df = pd.read_parquet(path)
        json_cols = ["audio", "transcripts", "quality", "labels", "lineage", "status", "errors"]
        samples: list[Sample] = []
        for _, row in df.iterrows():
            data = row.to_dict()
            for col in json_cols:
                if col in data and data[col] is not None:
                    if isinstance(data[col], str):
                        try:
                            data[col] = json.loads(data[col])
                        except json.JSONDecodeError:
                            data[col] = {}
                    elif not isinstance(data[col], (dict, list)):
                        data[col] = {}
            samples.append(Sample.from_flat_dict(data))
        return cls(samples)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.suffix == ".jsonl":
            self._save_jsonl(path)
        elif path.suffix == ".parquet":
            self._save_parquet(path)
        else:
            raise ValueError(f"Unsupported manifest format: {path.suffix}")
        return path

    def _save_jsonl(self, path: Path) -> None:
        with path.open("w", encoding="utf-8") as f:
            for sample in self.samples:
                f.write(sample.model_dump_json() + "\n")

    def _save_parquet(self, path: Path) -> None:
        rows = [sample.to_flat_dict() for sample in self.samples]
        df = pd.DataFrame(rows)
        json_cols = ["audio", "transcripts", "quality", "labels", "lineage", "status", "errors"]
        for col in json_cols:
            if col in df.columns:
                df[col] = df[col].apply(lambda v: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v)
        df.to_parquet(path, index=False)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([s.to_flat_dict() for s in self.samples])

    def filter(self, expr: str | None) -> Manifest:
        if not expr:
            return Manifest([s.model_copy(deep=True) for s in self.samples])
        df = self.to_dataframe()
        try:
            filtered = df.query(expr, engine="python")
        except Exception as exc:
            raise ValueError(f"Invalid filter expression: {expr!r} ({exc})") from exc
        ids = set(filtered["id"].tolist())
        return Manifest([s.model_copy(deep=True) for s in self.samples if s.id in ids])

    def stats(self) -> dict[str, Any]:
        df = self.to_dataframe()
        total = len(df)
        if total == 0:
            return {"files": 0}

        durations = df["duration"].dropna()
        sr_counts = df["sample_rate"].value_counts(dropna=False).to_dict()
        ext_counts: dict[str, int] = {}
        broken = 0
        for sample in self.samples:
            ext = Path(sample.source_path).suffix.lower()
            if ext in PCM_EXTENSIONS:
                ext_counts["pcm"] = ext_counts.get("pcm", 0) + 1
            elif ext in AUDIO_EXTENSIONS:
                ext_counts["wav"] = ext_counts.get("wav", 0) + 1
            else:
                ext_counts[ext or "unknown"] = ext_counts.get(ext or "unknown", 0) + 1
            if sample.duration is None and ext in AUDIO_EXTENSIONS:
                broken += 1

        return {
            "files": total,
            "duration_hours": round(durations.sum() / 3600, 2) if len(durations) else 0,
            "sample_rates": {str(int(k)) if pd.notna(k) else "unknown": int(v) for k, v in sr_counts.items()},
            "formats": ext_counts,
            "broken": broken,
        }

    @classmethod
    def ingest(
        cls,
        source_dir: str | Path,
        *,
        recursive: bool = True,
        extensions: set[str] | None = None,
    ) -> Manifest:
        source_dir = Path(source_dir)
        if not source_dir.is_dir():
            raise NotADirectoryError(f"Source is not a directory: {source_dir}")

        exts = extensions or (AUDIO_EXTENSIONS | PCM_EXTENSIONS)
        pattern = "**/*" if recursive else "*"
        files = sorted(
            p for p in source_dir.glob(pattern) if p.is_file() and p.suffix.lower() in exts
        )

        samples: list[Sample] = []
        for idx, path in enumerate(files, start=1):
            sample_id = path.stem
            sha = file_sha256(path)
            meta: dict[str, Any] = {}
            if path.suffix.lower() in AUDIO_EXTENSIONS:
                meta = probe_audio(path)

            sample = Sample(
                id=sample_id,
                source_path=str(path.resolve()),
                sha256=sha,
                sample_rate=meta.get("sample_rate"),
                channels=meta.get("channels"),
                duration=meta.get("duration"),
                audio={"raw": str(path.resolve())},
            )
            if not meta.get("valid", True):
                sample.labels["broken"] = True
            samples.append(sample)
            logger.debug("Ingested [{}/{}] {}", idx, len(files), sample_id)

        return cls(samples)

    def split(self, shards: int, *, strategy: str = "hash") -> list[Manifest]:
        """Split samples into shards.

        Strategies:
        - `hash`: stable content-hash buckets (same file always same shard).
        - `duration-balanced`: greedy LPT assignment by audio duration so total
          load per shard is closer when clip lengths vary widely. Deterministic
          via sort key `(duration desc, digest)`; best after duration is known.
        """
        if shards < 1:
            raise ValueError(f"shards must be >= 1, got {shards}")
        if strategy == "hash":
            buckets: list[list[Sample]] = [[] for _ in range(shards)]
            for sample in self.samples:
                buckets[int(sample_digest(sample)[:16], 16) % shards].append(sample)
            return [Manifest(bucket) for bucket in buckets]
        if strategy == "duration-balanced":
            return self._split_duration_balanced(shards)
        raise ValueError(
            f"Unknown split strategy '{strategy}'. Use 'hash' or 'duration-balanced'."
        )

    def _split_duration_balanced(self, shards: int) -> list[Manifest]:
        """Longest-processing-time-first: assign each clip to the lightest shard."""
        ordered = sorted(
            self.samples,
            key=lambda s: (-(s.duration if s.duration is not None else 1.0), sample_digest(s)),
        )
        buckets: list[list[Sample]] = [[] for _ in range(shards)]
        loads = [0.0] * shards
        for sample in ordered:
            weight = sample.duration if sample.duration is not None else 1.0
            # Prefer the lightest shard; tie-break by smaller index for stability.
            target = min(range(shards), key=lambda i: (loads[i], i))
            buckets[target].append(sample)
            loads[target] += weight
        # Restore deterministic order inside each shard (by digest), not LPT order.
        for bucket in buckets:
            bucket.sort(key=lambda s: (sample_digest(s), s.id))
        return [Manifest(bucket) for bucket in buckets]

    @classmethod
    def merge(
        cls, manifests: list[Manifest], *, keep_duplicates: bool = False
    ) -> tuple[Manifest, dict[str, Any]]:
        """Concatenate shard outputs, drop repeats and restore a deterministic order."""
        combined = [sample for manifest in manifests for sample in manifest.samples]
        seen: set[tuple[str, str]] = set()
        unique: list[Sample] = []
        duplicates = 0
        for sample in combined:
            key = (sample.sha256 or sample.source_path, sample.id)
            if key in seen:
                duplicates += 1
                if not keep_duplicates:
                    continue
            seen.add(key)
            unique.append(sample)

        unique.sort(key=lambda s: (s.sha256, s.id))
        report = {
            "inputs": len(manifests),
            "total_in": len(combined),
            "duplicates": duplicates,
            "total_out": len(unique),
        }
        return cls(unique), report

    def resolve_path(name: str, manifests_dir: Path = Path("datasets/manifests")) -> Path:
        """Resolve dataset name to manifest path (with or without extension)."""
        manifests_dir = Path(manifests_dir)
        candidates = [
            manifests_dir / name,
            manifests_dir / f"{name}.parquet",
            manifests_dir / f"{name}.jsonl",
            Path(name),
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"Dataset '{name}' not found. Searched: {[str(c) for c in candidates]}"
        )
