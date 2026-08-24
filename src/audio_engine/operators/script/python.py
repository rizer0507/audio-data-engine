from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

from audio_engine.core.operator import BaseOperator, OperatorConfig
from audio_engine.core.registry import register_operator
from audio_engine.core.sample import Sample


def _load_module(path: Path) -> ModuleType:
    """Load a fresh module so edited scripts are never hidden by import caching."""
    identity = f"{path}:{time.time_ns()}"
    name = f"audio_engine_user_script_{hashlib.sha256(identity.encode()).hexdigest()[:16]}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load Python script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


@dataclass(frozen=True)
class ScriptContext:
    """Small, process-safe runtime API exposed to a user script."""

    sample_id: str
    step_name: str
    run_dir: Path
    log_path: Path

    def log(self, message: str, *, level: str = "INFO", **fields: Any) -> None:
        """Append one structured event with O_APPEND (safe across threads/processes)."""
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": level.upper(),
            "step": self.step_name,
            "sample_id": self.sample_id,
            "message": str(message),
            **fields,
        }
        payload = (json.dumps(event, ensure_ascii=False, default=str) + "\n").encode()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.log_path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
        try:
            os.write(fd, payload)
        finally:
            os.close(fd)

    def artifact_path(self, *parts: str) -> Path:
        """Return a path namespaced to this step and sample under the run directory."""
        root = self.run_dir / "script_artifacts" / self.step_name / self.sample_id
        path = root.joinpath(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path


@register_operator
class PythonScriptOperator(BaseOperator):
    """Run a user Python ``process`` function for every sample in a pipeline."""

    name = "python"
    version = "1.0.0"
    category = "script"

    def _script_path(self, config: OperatorConfig) -> Path:
        value = config.params.get("path")
        if not value:
            raise ValueError("script.python requires params.path")
        path = Path(str(value)).expanduser().resolve()
        if not path.is_file() or path.suffix != ".py":
            raise FileNotFoundError(f"Python processing script not found: {path}")
        return path

    def compute_cache_key(self, sample: Sample, config: OperatorConfig) -> str:
        # A script edit is a new implementation even if its path and YAML stay unchanged.
        path = self._script_path(config)
        params = dict(config.params)
        params["_script_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        return super().compute_cache_key(sample, config.model_copy(update={"params": params}))

    def _execute(self, sample: Sample, config: OperatorConfig) -> dict[str, Any]:
        path = self._script_path(config)
        entrypoint = str(config.params.get("entrypoint", "process"))
        function: Callable[..., Any] | None = getattr(_load_module(path), entrypoint, None)
        if not callable(function):
            raise ValueError(f"{path} must define callable {entrypoint}(sample, params, context)")

        step = re.sub(r"[^A-Za-z0-9_.-]", "_", config.step_name or path.stem)
        run_dir = (config.run_dir or Path("runs/untracked")).resolve()
        context = ScriptContext(
            sample_id=sample.id,
            step_name=step,
            run_dir=run_dir,
            log_path=run_dir / "script_logs" / f"{step}.jsonl",
        )
        user_params = {
            key: value for key, value in config.params.items() if key not in {"path", "entrypoint"}
        }
        started = time.perf_counter()
        context.log("script started", script=str(path))
        try:
            result = function(sample.model_dump(mode="python"), user_params, context)
        except Exception as exc:
            context.log("script failed", level="ERROR", error=repr(exc))
            raise
        if result is None:
            result = {}
        if not isinstance(result, dict):
            raise TypeError(f"Script {path} returned {type(result).__name__}; expected dict")
        allowed = {"audio", "transcripts", "quality", "labels", *self._SCALAR_FIELDS}
        unknown = set(result) - allowed
        if unknown:
            raise ValueError(f"Script returned unsupported update keys: {sorted(unknown)}")
        context.log("script finished", elapsed_seconds=round(time.perf_counter() - started, 6))
        result["lineage_entry"] = {
            "operator": self.full_name,
            "version": self.version,
            "params": {**user_params, "script": str(path)},
            "input_key": config.params.get("input_audio_key"),
            "output_key": config.params.get("output_audio_key"),
        }
        return result
