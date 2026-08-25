"""Unified dataset naming driven by CLI ``--source-name``.

Convention (source_name=mt3000):
  datasets/manifests/cleaned_mt3000.parquet
  datasets/manifests/qwen_asr_mt3000.parquet
  datasets/manifests/multi_asr_aggregate_mt3000.parquet
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

DEFAULT_MANIFESTS_DIR = Path("datasets/manifests")
_SOURCE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")

# Default stage wiring for multi_asr_aggregate when YAML omits source_name_layout.
DEFAULT_MULTI_ASR_LAYOUT: list[dict[str, str]] = [
    {"input": "cleaned_{source_name}", "output": "qwen_asr_{source_name}"},
    {"input": "qwen_asr_{source_name}", "output": "multi_asr_aggregate_{source_name}"},
]


def validate_source_name(source_name: str) -> str:
    name = (source_name or "").strip()
    if not name:
        raise ValueError("--source-name must be a non-empty string")
    if not _SOURCE_NAME_RE.fullmatch(name):
        raise ValueError(
            f"Invalid --source-name '{source_name}': use letters/digits/_/- "
            "(e.g. mt3000, mt-3000)"
        )
    return name


def manifest_stem(kind: str, source_name: str) -> str:
    """Return basename without extension, e.g. cleaned_mt3000."""
    name = validate_source_name(source_name)
    kind = kind.strip().strip("_")
    if not kind:
        raise ValueError("manifest kind must be non-empty")
    return f"{kind}_{name}"


def _posix(path: Path | str) -> str:
    """Stable relative path string for YAML/CLI (forward slashes)."""
    return Path(path).as_posix()


def manifest_path(
    kind: str,
    source_name: str,
    *,
    manifests_dir: Path | str = DEFAULT_MANIFESTS_DIR,
    ext: str = ".parquet",
) -> Path:
    return Path(manifests_dir) / f"{manifest_stem(kind, source_name)}{ext}"


def resolve_existing_manifest(
    stem_or_path: str,
    *,
    manifests_dir: Path | str = DEFAULT_MANIFESTS_DIR,
) -> Path:
    """Find an existing parquet/jsonl for a stem or relative path.

    Accepts:
      - cleaned_mt3000
      - datasets/manifests/cleaned_mt3000
      - datasets/manifests/cleaned_mt3000.parquet
    Prefers ``.parquet`` over ``.jsonl`` when both exist.
    """
    text = str(stem_or_path).strip()
    if not text:
        raise ValueError("manifest path/stem is empty")

    raw = Path(text)
    manifests_dir = Path(manifests_dir)
    candidates: list[Path] = []

    if raw.suffix.lower() in {".parquet", ".jsonl"}:
        candidates.append(raw)
        if not raw.is_absolute():
            candidates.append(Path.cwd() / raw)
            candidates.append(manifests_dir / raw.name)
    else:
        # Prefer parquet then jsonl under manifests_dir and CWD.
        for base in (
            manifests_dir,
            Path.cwd(),
            raw.parent if raw.parent != Path(".") else Path.cwd(),
        ):
            stem = raw.name
            candidates.append(base / f"{stem}.parquet")
            candidates.append(base / f"{stem}.jsonl")
            candidates.append(base / stem)

    seen: set[Path] = set()
    ordered: list[Path] = []
    for candidate in candidates:
        resolved = candidate if candidate.is_absolute() else (Path.cwd() / candidate)
        key = resolved.resolve() if resolved.exists() else resolved
        if key in seen:
            continue
        seen.add(key)
        ordered.append(resolved)

    for candidate in ordered:
        if candidate.is_file():
            return candidate.resolve()

    searched = ", ".join(str(p) for p in ordered[:8])
    raise FileNotFoundError(
        f"Manifest '{stem_or_path}' not found (tried parquet/jsonl). Searched: {searched}"
    )


def _as_output_parquet(formatted: str) -> str:
    """Normalize a layout output template to a ``.parquet`` path under manifests."""
    text = formatted.strip()
    path = Path(text)
    if path.suffix.lower() in {".parquet", ".jsonl"}:
        return _posix(path.with_suffix(".parquet"))
    if "/" in text or "\\" in text:
        out = Path(text).with_suffix(".parquet") if Path(text).suffix else Path(f"{text}.parquet")
        return _posix(out)
    return _posix(DEFAULT_MANIFESTS_DIR / f"{path.name}.parquet")


def expand_layout_templates(
    layout: list[dict[str, Any]] | None,
    source_name: str,
) -> list[tuple[str, str]]:
    """Expand ``{source_name}`` templates into (input_stem, output_parquet) pairs.

    ``input`` stays a stem/path for ``resolve_existing_manifest``.
    ``output`` becomes ``datasets/manifests/<stem>.parquet``.
    """
    name = validate_source_name(source_name)
    rows = layout if layout else DEFAULT_MULTI_ASR_LAYOUT
    if not rows:
        raise ValueError("source_name_layout is empty")

    expanded: list[tuple[str, str]] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"source_name_layout[{index}] must be a mapping")
        raw_in = str(row.get("input") or "").strip()
        raw_out = str(row.get("output") or "").strip()
        if not raw_in or not raw_out:
            raise ValueError(
                f"source_name_layout[{index}] needs both input and output templates"
            )
        inp = raw_in.format(source_name=name)
        out_path = _as_output_parquet(raw_out.format(source_name=name))
        expanded.append((inp, out_path))
    return expanded


def cleaned_output_path(source_name: str) -> str:
    return _posix(manifest_path("cleaned", source_name))


def pipeline_run_name(pipeline_name: str, source_name: str | None) -> str:
    """Build runs/ directory label; append ``--source-name`` when provided.

    ``data_cleaning_source_A`` + ``test_local`` → ``data_cleaning_test_local``
    ``multi_asr_aggregate`` + ``test_local`` → ``multi_asr_aggregate_test_local``
    """
    if not source_name:
        return pipeline_name
    name = validate_source_name(source_name)
    base = re.sub(r"_source_[A-Za-z0-9_-]+$", "", pipeline_name.strip()) or pipeline_name
    if base.endswith(f"_{name}"):
        return base
    return f"{base}_{name}"


def apply_source_name_to_cleaning(
    *,
    source_name: str,
    source_dir: str | Path,
) -> dict[str, str]:
    """Return overrides for the data-cleaning pipeline."""
    name = validate_source_name(source_name)
    path = Path(source_dir)
    if not path.is_dir():
        raise FileNotFoundError(f"--source-dir is not a directory: {source_dir}")
    return {
        "source_dir": str(path.resolve()),
        "output_manifest": cleaned_output_path(name),
    }


def apply_source_name_to_single_pipeline(
    *,
    pipeline_name: str,
    steps: list[Any],
    source_name: str,
    source_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Derive input/output overrides for a non-staged pipeline.

    - Cleaning (ingest steps or ``--source-dir``): write ``cleaned_<name>``.
    - ``qwen_asr*``: ``cleaned_<name>`` → ``qwen_asr_<name>``.
    - ``sensevoice*``: ``cleaned_<name>`` → ``sensevoice_asr_<name>``.
    - ``multi_asr*``: ``qwen_asr_<name>`` + model manifests → aggregate output.
    - ``asr_metric*``: aggregate output → standalone metric output.
    """
    name = validate_source_name(source_name)
    has_ingest = any(
        getattr(step, "operator", "").startswith("ingest.") for step in steps
    )
    if has_ingest or source_dir is not None:
        if source_dir is None:
            raise ValueError(
                "cleaning pipeline with --source-name also requires --source-dir"
            )
        overrides = apply_source_name_to_cleaning(
            source_name=name, source_dir=source_dir
        )
        return {
            "source_dir": overrides["source_dir"],
            "input_manifest": "",
            "source_id": None,
            "output_manifest": overrides["output_manifest"],
        }

    key = pipeline_name.lower()
    if "qwen" in key:
        resolved = resolve_existing_manifest(manifest_stem("cleaned", name))
        return {
            "source_dir": None,
            "input_manifest": str(resolved),
            "source_id": None,
            "output_manifest": _posix(manifest_path("qwen_asr", name)),
        }
    if "sensevoice" in key:
        resolved = resolve_existing_manifest(manifest_stem("cleaned", name))
        return {
            "source_dir": None,
            "input_manifest": str(resolved),
            "source_id": None,
            "output_manifest": _posix(manifest_path("sensevoice_asr", name)),
        }
    if "multi_asr" in key:
        resolved = resolve_existing_manifest(manifest_stem("qwen_asr", name))
        for step in steps:
            if getattr(step, "operator", "") != "quality.aggregate_manifests":
                continue
            for item in step.params.get("manifests", []):
                model = str(item.get("model") or "").strip()
                if model:
                    item["path"] = _posix(manifest_path(f"{model}_asr", name))
        return {
            "source_dir": None,
            "input_manifest": str(resolved),
            "source_id": None,
            "output_manifest": _posix(manifest_path("multi_asr_aggregate", name)),
        }
    if "asr_metric" in key:
        resolved = resolve_existing_manifest(manifest_stem("multi_asr_aggregate", name))
        return {
            "source_dir": None,
            "input_manifest": str(resolved),
            "source_id": None,
            "output_manifest": _posix(manifest_path("multi_asr_metrics", name)),
        }

    raise ValueError(
        f"--source-name on pipeline '{pipeline_name}' is unsupported without "
        "--source-dir; use multi_asr_aggregate.yaml or pass --input-manifest / "
        "--output-manifest"
    )
