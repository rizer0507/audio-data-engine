"""Unified dataset naming driven by CLI ``--source-name``.

Convention (source_name=mt3000):
  datasets/manifests/cleaned_mt3000.parquet
  datasets/manifests/qwen_asr_mt3000.parquet          # default ASR stem
  datasets/manifests/qwen1_asr_mt3000.parquet         # --asr-run qwen1
  datasets/manifests/sensevoice_asr_mt3000.parquet
  datasets/manifests/multi_asr_aggregate_mt3000.parquet
  datasets/manifests/multi_asr_metrics_mt3000.parquet

Eval convention (--eval-name eval_mt3000):
  datasets/manifests/eval_mt3000.parquet              # registered eval set
  datasets/manifests/{alias}_asr_eval_mt3000.parquet  # --asr-run on eval set
  datasets/manifests/eval_aggregate_eval_mt3000.parquet
  datasets/manifests/eval_metrics_eval_mt3000.parquet
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

DEFAULT_MANIFESTS_DIR = Path("datasets/manifests")
_SOURCE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
# kimi_asr_batch / qwen_asr / sensevoice_asr_batch → model stem before _asr
_ASR_PIPELINE_RE = re.compile(r"^(.+)_asr(?:_batch)?$")

# Legacy staged orchestrator layout (optional YAML ``stages:`` + source_name_layout).
DEFAULT_MULTI_ASR_LAYOUT: list[dict[str, str]] = [
    {"input": "cleaned_{source_name}", "output": "qwen_asr_{source_name}"},
    {"input": "cleaned_{source_name}", "output": "sensevoice_asr_{source_name}"},
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


def validate_asr_run(asr_run: str) -> str:
    """Validate a result alias used as transcript key / manifest stem prefix.

    Same character rules as ``--source-name``. Example aliases: ``qwen1``,
    ``sensevoice2``, ``doubao_a``.
    """
    name = (asr_run or "").strip()
    if not name:
        raise ValueError("--asr-run must be a non-empty string")
    if not _SOURCE_NAME_RE.fullmatch(name):
        raise ValueError(
            f"Invalid --asr-run '{asr_run}': use letters/digits/_/- "
            "(e.g. qwen1, sensevoice2)"
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


def model_asr_kind(model: str) -> str:
    """Map transcript model key to manifest kind: sensevoice → sensevoice_asr."""
    model = str(model or "").strip().strip("_")
    if not model:
        raise ValueError("model name must be non-empty")
    if model.endswith("_asr"):
        return model
    return f"{model}_asr"


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


def resolve_model_asr_manifest(model: str, source_name: str) -> str:
    """Resolve ``{model}_asr_{source_name}.parquet|.jsonl`` to an existing file path."""
    kind = model_asr_kind(model)
    return str(resolve_existing_manifest(manifest_stem(kind, source_name)))


def rewrite_join_manifests_for_source(
    manifests: list[dict[str, Any]] | None,
    source_name: str,
    *,
    require_existing: bool = True,
) -> list[dict[str, Any]]:
    """Rewrite aggregate ``manifests`` entries to ``{model}_asr_{source_name}`` paths."""
    name = validate_source_name(source_name)
    rows = list(manifests or [])
    if not rows:
        raise ValueError(
            "aggregate pipeline needs manifests to join "
            "(YAML params.manifests or --join-manifest)"
        )
    rewritten: list[dict[str, Any]] = []
    for index, item in enumerate(rows, start=1):
        if not isinstance(item, dict) or "model" not in item:
            raise ValueError(f"manifests[{index}] must be a mapping with `model`")
        model = str(item["model"]).strip()
        kind = model_asr_kind(model)
        if require_existing:
            path = str(resolve_existing_manifest(manifest_stem(kind, name)))
        else:
            path = _posix(manifest_path(kind, name))
        rewritten.append({**item, "model": model, "path": path})
    return rewritten


def parse_join_manifest_arg(raw: str, source_name: str | None = None) -> dict[str, str]:
    """Parse ``sensevoice`` or ``kimi=/path/to.parquet`` into `{model, path}`.

    When only a model name is given, ``source_name`` is required and the path becomes
    ``datasets/manifests/{model}_asr_{source_name}.parquet``.
    """
    text = (raw or "").strip()
    if not text:
        raise ValueError("--join-manifest value must be non-empty")
    if "=" in text:
        model, _, path = text.partition("=")
        model = model.strip()
        path = path.strip()
        if not model or not path:
            raise ValueError(
                "--join-manifest expected `model` or `model=/path/to.parquet`"
            )
        return {"model": model, "path": path}
    if source_name is None:
        raise ValueError(
            f"--join-manifest '{text}' needs --source-name "
            f"(or use model=/explicit/path.parquet)"
        )
    kind = model_asr_kind(text)
    return {
        "model": text.strip(),
        "path": _posix(manifest_path(kind, validate_source_name(source_name))),
    }


def _asr_output_kind(pipeline_name: str) -> str | None:
    """Return manifest kind for an ASR batch pipeline, or None if not ASR-named."""
    key = pipeline_name.lower().strip()
    if "multi_asr" in key or "aggregate" in key:
        return None
    if "metric" in key:
        return None
    match = _ASR_PIPELINE_RE.match(key)
    if match:
        stem = match.group(1)
        if stem in {"multi"}:
            return None
        return f"{stem}_asr"
    return None


def apply_source_name_to_single_pipeline(
    *,
    pipeline_name: str,
    steps: list[Any],
    source_name: str,
    source_dir: str | Path | None = None,
    join_manifests: list[dict[str, Any]] | None = None,
    asr_run: str | None = None,
    aggregate_base: str | None = None,
) -> dict[str, Any]:
    """Derive input/output (and aggregate join) overrides for a non-staged pipeline.

    - Cleaning (ingest steps or ``--source-dir``): write ``cleaned_<name>``.
    - ``qwen_asr*`` / ``sensevoice_asr*`` / ``{model}_asr*``:
      ``cleaned_<name>`` → ``{model}_asr_<name>`` (or ``{asr_run}_asr_<name>``).
    - ``multi_asr_aggregate*``: ``{aggregate_base|qwen}_asr_<name>`` →
      ``multi_asr_aggregate_<name>``, and rewrite join manifests to
      ``{model}_asr_<name>``.
    - ``asr_metric*``: ``multi_asr_aggregate_<name>`` → ``multi_asr_metrics_<name>``.
    """
    name = validate_source_name(source_name)
    run_alias = validate_asr_run(asr_run) if asr_run is not None else None
    base_alias = (
        validate_asr_run(aggregate_base) if aggregate_base is not None else None
    )
    has_ingest = any(
        getattr(step, "operator", "").startswith("ingest.") for step in steps
    )
    if has_ingest or source_dir is not None:
        if run_alias is not None:
            raise ValueError("--asr-run is only valid for ASR inference pipelines")
        if base_alias is not None:
            raise ValueError(
                "--aggregate-base is only valid for multi_asr_aggregate pipelines"
            )
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
            "aggregate_manifests": None,
            "asr_run": None,
            "aggregate_base": None,
        }

    key = pipeline_name.lower()

    if "multi_asr" in key or ("aggregate" in key and "asr" in key):
        if run_alias is not None:
            raise ValueError("--asr-run is only valid for ASR inference pipelines")
        base_model = base_alias or "qwen"
        resolved = resolve_existing_manifest(manifest_stem(model_asr_kind(base_model), name))
        if join_manifests is not None:
            # CLI --join-manifest: keep explicit paths; only require files exist.
            joins = [
                {
                    "model": str(item["model"]).strip(),
                    "path": str(resolve_existing_manifest(str(item["path"]))),
                }
                for item in join_manifests
            ]
            if not joins:
                raise ValueError("--join-manifest produced an empty join list")
        else:
            yaml_joins: list[dict[str, Any]] = []
            for step in steps:
                if getattr(step, "operator", "") == "quality.aggregate_manifests":
                    params = getattr(step, "params", None) or {}
                    yaml_joins = list(params.get("manifests") or [])
                    break
            joins = rewrite_join_manifests_for_source(yaml_joins, name)
        return {
            "source_dir": None,
            "input_manifest": str(resolved),
            "source_id": None,
            "output_manifest": _posix(manifest_path("multi_asr_aggregate", name)),
            "aggregate_manifests": joins,
            "asr_run": None,
            "aggregate_base": base_model,
        }

    if "asr_metric" in key or key in {"metric_pipeline", "text_metrics"}:
        if run_alias is not None:
            raise ValueError("--asr-run is only valid for ASR inference pipelines")
        if base_alias is not None:
            raise ValueError(
                "--aggregate-base is only valid for multi_asr_aggregate pipelines; "
                "use --agreement-base for asr_metric_pipeline"
            )
        resolved = resolve_existing_manifest(manifest_stem("multi_asr_aggregate", name))
        return {
            "source_dir": None,
            "input_manifest": str(resolved),
            "source_id": None,
            "output_manifest": _posix(manifest_path("multi_asr_metrics", name)),
            "aggregate_manifests": None,
            "asr_run": None,
            "aggregate_base": None,
        }

    asr_kind = _asr_output_kind(pipeline_name)
    if asr_kind is not None:
        if base_alias is not None:
            raise ValueError(
                "--aggregate-base is only valid for multi_asr_aggregate pipelines"
            )
        resolved = resolve_existing_manifest(manifest_stem("cleaned", name))
        out_kind = model_asr_kind(run_alias) if run_alias is not None else asr_kind
        return {
            "source_dir": None,
            "input_manifest": str(resolved),
            "source_id": None,
            "output_manifest": _posix(manifest_path(out_kind, name)),
            "aggregate_manifests": None,
            "asr_run": run_alias,
            "aggregate_base": None,
        }

    # Fallback: any pipeline whose steps are pure ASR inference.
    if any(getattr(step, "operator", "").startswith("asr.") for step in steps):
        raise ValueError(
            f"--source-name on pipeline '{pipeline_name}' needs a name like "
            f"qwen_asr_batch / sensevoice_asr_batch / kimi_asr_batch "
            f"(output becomes <model>_asr_{name}.parquet)"
        )

    raise ValueError(
        f"--source-name on pipeline '{pipeline_name}' is unsupported without "
        "--source-dir; use qwen/sensevoice/aggregate/metric pipelines or pass "
        "--input-manifest / --output-manifest"
    )


def apply_eval_name_to_single_pipeline(
    *,
    pipeline_name: str,
    steps: list[Any],
    eval_name: str,
    join_manifests: list[dict[str, Any]] | None = None,
    asr_run: str | None = None,
) -> dict[str, Any]:
    """Derive input/output for evaluation pipelines (decoupled from training).

    - ASR ``qwen_asr*`` / ``{model}_asr*``: registered eval set →
      ``{alias}_asr_{eval_name}``.
    - ``eval_aggregate*``: eval set as left table + ``--join-manifest`` aliases
      (``{alias}_asr_{eval_name}``) → ``eval_aggregate_{eval_name}``.
    - ``eval_metric*`` / ``asr_eval``: ``eval_aggregate_{eval_name}`` →
      ``eval_metrics_{eval_name}``.
    """
    name = validate_source_name(eval_name)
    run_alias = validate_asr_run(asr_run) if asr_run is not None else None
    eval_path = str(resolve_existing_manifest(name))
    key = pipeline_name.lower()

    if "eval_aggregate" in key or (key.startswith("eval") and "aggregate" in key):
        if run_alias is not None:
            raise ValueError("--asr-run is only valid for ASR inference pipelines")
        if not join_manifests:
            raise ValueError(
                "eval_aggregate requires --join-manifest "
                "(result aliases whose ids match the eval set)"
            )
        joins = [
            {
                "model": str(item["model"]).strip(),
                "path": str(resolve_existing_manifest(str(item["path"]))),
            }
            for item in join_manifests
        ]
        return {
            "source_dir": None,
            "input_manifest": eval_path,
            "source_id": None,
            "output_manifest": _posix(manifest_path("eval_aggregate", name)),
            "aggregate_manifests": joins,
            "asr_run": None,
            "aggregate_base": None,
        }

    if "eval_metric" in key or key in {"asr_eval", "eval_metrics"}:
        if run_alias is not None:
            raise ValueError("--asr-run is only valid for ASR inference pipelines")
        resolved = resolve_existing_manifest(manifest_stem("eval_aggregate", name))
        return {
            "source_dir": None,
            "input_manifest": str(resolved),
            "source_id": None,
            "output_manifest": _posix(manifest_path("eval_metrics", name)),
            "aggregate_manifests": None,
            "asr_run": None,
            "aggregate_base": None,
        }

    asr_kind = _asr_output_kind(pipeline_name)
    if asr_kind is not None:
        out_kind = model_asr_kind(run_alias) if run_alias is not None else asr_kind
        return {
            "source_dir": None,
            "input_manifest": eval_path,
            "source_id": None,
            "output_manifest": _posix(manifest_path(out_kind, name)),
            "aggregate_manifests": None,
            "asr_run": run_alias,
            "aggregate_base": None,
        }

    raise ValueError(
        f"--eval-name on pipeline '{pipeline_name}' is unsupported; "
        "use qwen_asr_batch / eval_aggregate / eval_metric_pipeline "
        "(or pass --input-manifest / --output-manifest)"
    )
