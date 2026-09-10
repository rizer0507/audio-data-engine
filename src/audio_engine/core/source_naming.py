"""Unified dataset naming driven by CLI ``--source-name`` / ``--eval-name``.

Stem convention is unchanged; directory roots are staged by process (009 Phase A):

  datasets/stage1/cleaned/cleaned_{source}.parquet
  datasets/stage1/asr/{alias}_asr_{source}.parquet          # expensive
  datasets/stage1/derived/multi_asr_aggregate_{source}.parquet
  datasets/stage1/derived/multi_asr_metrics_{source}.parquet
  datasets/stage1/derived/classified_{source}.parquet
  datasets/stage3/eval_sets/eval_{batch}.parquet
  datasets/stage3/asr/{alias}_asr_eval_{batch}.parquet      # expensive
  datasets/stage3/derived/eval_aggregate_eval_{batch}.parquet
  datasets/stage3/derived/eval_metrics_eval_{batch}.parquet
  datasets/stage3/reports/{eval_name}/evaluation.{json,xlsx}

Legacy flat ``datasets/manifests/`` remains readable via ``resolve_existing_manifest``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

DATASETS_ROOT = Path("datasets")
LEGACY_MANIFESTS_DIR = DATASETS_ROOT / "manifests"
# Kept as alias for callers / resolve fallback search.
DEFAULT_MANIFESTS_DIR = LEGACY_MANIFESTS_DIR

STAGE1_CLEANED_DIR = DATASETS_ROOT / "stage1" / "cleaned"
STAGE1_ASR_DIR = DATASETS_ROOT / "stage1" / "asr"
STAGE1_DERIVED_DIR = DATASETS_ROOT / "stage1" / "derived"
STAGE3_EVAL_SETS_DIR = DATASETS_ROOT / "stage3" / "eval_sets"
STAGE3_ASR_DIR = DATASETS_ROOT / "stage3" / "asr"
STAGE3_DERIVED_DIR = DATASETS_ROOT / "stage3" / "derived"
STAGE3_REPORTS_DIR = DATASETS_ROOT / "stage3" / "reports"

_SOURCE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
# glm_asr_batch / kimi_asr_batch / qwen_asr / sensevoice_asr_batch → model stem before _asr
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


def manifest_dir_for_stem(stem: str) -> Path:
    """Map a manifest basename (no extension) to its staged directory root."""
    text = str(stem or "").strip()
    if not text:
        raise ValueError("manifest stem must be non-empty")
    # Strip accidental suffixes if callers pass a filename.
    if text.lower().endswith((".parquet", ".jsonl")):
        text = Path(text).stem

    if text.startswith("cleaned_"):
        return STAGE1_CLEANED_DIR
    if (
        text.startswith("multi_asr_")
        or text.startswith("classified_")
        or text.startswith("prepared_v3_")
        or text.startswith("quality_sidecar_")
    ):
        return STAGE1_DERIVED_DIR
    if text.startswith("eval_aggregate_") or text.startswith("eval_metrics_"):
        return STAGE3_DERIVED_DIR
    # Eval-set ASR before generic ``_asr_`` / ``eval_`` rules.
    if "_asr_eval_" in text:
        return STAGE3_ASR_DIR
    if text.startswith("eval_"):
        return STAGE3_EVAL_SETS_DIR
    if "_asr_" in text or text.endswith("_asr"):
        return STAGE1_ASR_DIR
    # Unknown stems (reviewed_*, gold_*, …): keep writable under stage1/derived.
    return STAGE1_DERIVED_DIR


def staged_manifest_path(stem: str, *, ext: str = ".parquet") -> Path:
    """Canonical write path for a full stem under the staged layout."""
    name = str(stem).strip()
    if name.lower().endswith((".parquet", ".jsonl")):
        ext = Path(name).suffix.lower()
        name = Path(name).stem
    return manifest_dir_for_stem(name) / f"{name}{ext}"


def evaluation_report_dir(eval_name: str) -> Path:
    """Authoritative evaluation report directory for an eval set name."""
    name = validate_source_name(eval_name)
    return STAGE3_REPORTS_DIR / name


def manifest_path(
    kind: str,
    source_name: str,
    *,
    manifests_dir: Path | str | None = None,
    ext: str = ".parquet",
) -> Path:
    """Return the canonical parquet/jsonl path for ``{kind}_{source_name}``.

    When ``manifests_dir`` is omitted, writes go to the staged process directory.
    Passing ``manifests_dir`` explicitly forces that root (tests / overrides).
    """
    stem = manifest_stem(kind, source_name)
    if manifests_dir is not None:
        return Path(manifests_dir) / f"{stem}{ext}"
    return staged_manifest_path(stem, ext=ext)


def model_asr_kind(model: str) -> str:
    """Map transcript model key to manifest kind: sensevoice → sensevoice_asr."""
    model = str(model or "").strip().strip("_")
    if not model:
        raise ValueError("model name must be non-empty")
    if model.endswith("_asr"):
        return model
    return f"{model}_asr"


def _candidate_bases_for_stem(stem: str, manifests_dir: Path) -> list[Path]:
    """Search order: staged root → legacy manifests → explicit manifests_dir."""
    staged = manifest_dir_for_stem(stem)
    bases: list[Path] = [staged, LEGACY_MANIFESTS_DIR]
    manifests_dir = Path(manifests_dir)
    if manifests_dir.resolve() != LEGACY_MANIFESTS_DIR.resolve():
        bases.append(manifests_dir)
    # Deduplicate while preserving order.
    seen: set[Path] = set()
    ordered: list[Path] = []
    for base in bases:
        key = base if base.is_absolute() else (Path.cwd() / base)
        try:
            key = key.resolve()
        except OSError:
            pass
        if key in seen:
            continue
        seen.add(key)
        ordered.append(base)
    return ordered


def resolve_existing_manifest(
    stem_or_path: str,
    *,
    manifests_dir: Path | str = DEFAULT_MANIFESTS_DIR,
) -> Path:
    """Find an existing parquet/jsonl for a stem or relative path.

    Accepts:
      - cleaned_mt3000
      - datasets/manifests/cleaned_mt3000
      - datasets/stage1/cleaned/cleaned_mt3000.parquet
    Prefers staged locations over legacy ``datasets/manifests/``, and
    ``.parquet`` over ``.jsonl`` when both exist in the same root.
    """
    text = str(stem_or_path).strip()
    if not text:
        raise ValueError("manifest path/stem is empty")

    raw = Path(text)
    manifests_dir = Path(manifests_dir)
    candidates: list[Path] = []

    if raw.suffix.lower() in {".parquet", ".jsonl"}:
        stem = raw.stem
        candidates.append(raw)
        if not raw.is_absolute():
            candidates.append(Path.cwd() / raw)
        for base in _candidate_bases_for_stem(stem, manifests_dir):
            candidates.append(base / raw.name)
            # Prefer parquet sibling when caller pointed at missing jsonl path.
            if raw.suffix.lower() == ".jsonl":
                candidates.append(base / f"{stem}.parquet")
            else:
                candidates.append(base / f"{stem}.jsonl")
    else:
        stem = raw.name
        for base in (
            *_candidate_bases_for_stem(stem, manifests_dir),
            Path.cwd(),
            raw.parent if raw.parent != Path(".") else Path.cwd(),
        ):
            candidates.append(base / f"{stem}.parquet")
            candidates.append(base / f"{stem}.jsonl")
            candidates.append(base / stem)

    seen: set[Path] = set()
    ordered: list[Path] = []
    for candidate in candidates:
        resolved = candidate if candidate.is_absolute() else (Path.cwd() / candidate)
        try:
            key = resolved.resolve() if resolved.exists() else resolved
        except OSError:
            key = resolved
        if key in seen:
            continue
        seen.add(key)
        ordered.append(resolved)

    for candidate in ordered:
        if candidate.is_file():
            return candidate.resolve()

    searched = ", ".join(str(p) for p in ordered[:12])
    raise FileNotFoundError(
        f"Manifest '{stem_or_path}' not found (tried parquet/jsonl). Searched: {searched}"
    )


def _as_output_parquet(formatted: str) -> str:
    """Normalize a layout output template to a staged ``.parquet`` path."""
    text = formatted.strip()
    path = Path(text)
    if path.suffix.lower() in {".parquet", ".jsonl"}:
        if "/" in text or "\\" in text:
            return _posix(path.with_suffix(".parquet"))
        return _posix(staged_manifest_path(path.stem, ext=".parquet"))
    if "/" in text or "\\" in text:
        out = Path(text).with_suffix(".parquet") if Path(text).suffix else Path(f"{text}.parquet")
        return _posix(out)
    return _posix(staged_manifest_path(path.name, ext=".parquet"))


def expand_layout_templates(
    layout: list[dict[str, Any]] | None,
    source_name: str,
) -> list[tuple[str, str]]:
    """Expand ``{source_name}`` templates into (input_stem, output_parquet) pairs.

    ``input`` stays a stem/path for ``resolve_existing_manifest``.
    ``output`` becomes the staged parquet path for that stem.
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
    the staged ``{model}_asr_{source_name}.parquet``.
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
    - ``classify_external_gold*``: ``{aggregate_base|qwen}_asr_<name>`` →
      ``classified_<name>`` (external gold inject + classify).
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

    if "external_gold" in key or "classify_external" in key:
        if run_alias is not None:
            raise ValueError("--asr-run is only valid for ASR inference pipelines")
        base_model = base_alias or "qwen"
        resolved = resolve_existing_manifest(manifest_stem(model_asr_kind(base_model), name))
        return {
            "source_dir": None,
            "input_manifest": str(resolved),
            "source_id": None,
            "output_manifest": _posix(manifest_path("classified", name)),
            "aggregate_manifests": None,
            "asr_run": None,
            "aggregate_base": base_model,
        }

    if "prepare_dataset" in key or key.startswith("prepare_v3"):
        if run_alias is not None:
            raise ValueError("--asr-run is only valid for ASR inference pipelines")
        if base_alias is not None:
            raise ValueError(
                "--aggregate-base is only valid for multi_asr_aggregate pipelines"
            )
        resolved = resolve_existing_manifest(manifest_stem("cleaned", name))
        return {
            "source_dir": None,
            "input_manifest": str(resolved),
            "source_id": None,
            "output_manifest": _posix(manifest_path("prepared_v3", name)),
            "aggregate_manifests": None,
            "asr_run": None,
            "aggregate_base": None,
        }

    if "audio_quality" in key or "quality_sidecar" in key:
        if run_alias is not None:
            raise ValueError("--asr-run is only valid for ASR inference pipelines")
        if base_alias is not None:
            raise ValueError(
                "--aggregate-base is only valid for multi_asr_aggregate pipelines"
            )
        resolved = resolve_existing_manifest(manifest_stem("cleaned", name))
        return {
            "source_dir": None,
            "input_manifest": str(resolved),
            "source_id": None,
            "output_manifest": _posix(manifest_path("quality_sidecar", name)),
            "aggregate_manifests": None,
            "asr_run": None,
            "aggregate_base": None,
        }

    if "classify_dataset_v3" in key or key in {"classify_v3", "classified_v3"}:
        if run_alias is not None:
            raise ValueError("--asr-run is only valid for ASR inference pipelines")
        if base_alias is not None:
            raise ValueError(
                "--aggregate-base is only valid for multi_asr_aggregate pipelines"
            )
        resolved = resolve_existing_manifest(manifest_stem("prepared_asr_v3", name))
        sidecar = staged_manifest_path(manifest_stem("quality_sidecar", name))
        overrides = {
            "source_dir": None,
            "input_manifest": str(resolved),
            "source_id": None,
            "output_manifest": _posix(manifest_path("classified_v3", name)),
            "aggregate_manifests": None,
            "asr_run": None,
            "aggregate_base": None,
        }
        if sidecar.exists():
            overrides["quality_sidecar_manifest"] = _posix(sidecar)
        return overrides

    if key == "attach_asr_v3":
        resolved = resolve_existing_manifest(manifest_stem("prepared_v3", name))
        return {"source_dir": None, "input_manifest": str(resolved), "source_id": None,
                "output_manifest": _posix(manifest_path("prepared_asr_v3", name)),
                "aggregate_manifests": None, "asr_run": None, "aggregate_base": None}

    if key == "build_dataset_v3":
        if run_alias is not None or base_alias is not None:
            raise ValueError("build_dataset_v3 does not accept ASR run/base overrides")
        resolved = resolve_existing_manifest(manifest_stem("reviewed_v3", name))
        return {"source_dir": None, "input_manifest": str(resolved), "source_id": None,
                "output_manifest": _posix(manifest_path("built_v3", name)),
                "aggregate_manifests": None, "asr_run": None, "aggregate_base": None}

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
            f"qwen_asr_batch / sensevoice_asr_batch / kimi_asr_batch / glm_asr_batch "
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
        "use qwen_asr_batch / glm_asr_batch / eval_aggregate / eval_metric_pipeline "
        "(or pass --input-manifest / --output-manifest)"
    )
