"""augmentation.tempo — pitch-preserving batch speed augmentation (047)."""

from __future__ import annotations

import time
import traceback
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
from typing import Any

from loguru import logger

from audio_engine.core.artifacts import atomic_write_json
from audio_engine.core.augmentation.audio_io import (
    TempoAudioError,
    duration_tolerance,
    read_audio_info,
    resolve_input_audio,
    validate_output_wav,
)
from audio_engine.core.augmentation.ffmpeg_atempo import (
    atempo_filter_graph,
    probe_ffmpeg,
    run_atempo,
)
from audio_engine.core.augmentation.identity import (
    audio_output_path,
    cache_key_for,
    derived_sample_id,
    processing_fingerprint,
)
from audio_engine.core.augmentation.params import OPERATOR_VERSION, TempoParams
from audio_engine.core.augmentation.preflight import (
    AugmentPreflightError,
    preflight_tempo,
)
from audio_engine.core.augmentation.provenance import build_derived_sample
from audio_engine.core.augmentation.publish import (
    AugmentBatchIncomplete,
    publish_batch_artifacts,
)
from audio_engine.core.augmentation.recipe import build_recipe
from audio_engine.core.operator import ManifestOperator, OperatorConfig
from audio_engine.core.registry import register_operator
from audio_engine.core.sample import Sample


def _mock_atempo(
    *,
    input_path: Path,
    output_path: Path,
    speed_factor: float,
) -> dict[str, Any]:
    """Deterministic mock: copy bytes and shrink duration metadata via sox-less trim.

    For tests without claiming pitch preservation of real speech. Uses soundfile
    to rewrite at original sr/channels with shortened frames (pitch-changing),
    but only when mock=True — never used in production path.
    """
    import numpy as np
    import soundfile as sf

    from audio_engine.core.artifacts import atomic_path

    data, sr = sf.read(str(input_path), always_2d=True)
    new_len = max(1, int(data.shape[0] / speed_factor))
    # Keep channels; truncate frames only for mock duration ≈ T/factor.
    out = data[:new_len, :]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(output_path) as tmp:
        sf.write(str(tmp), out.astype(np.float32), sr, subtype="PCM_16")
    return {
        "filter_graph": atempo_filter_graph(speed_factor),
        "stages": [speed_factor],
        "ffmpeg_version": "mock",
        "ffmpeg_bin": "mock",
        "command": ["mock-atempo", str(input_path), str(output_path)],
        "stderr_tail": "",
    }


@register_operator
class TempoAugmentOperator(ManifestOperator):
    """Whole-batch pitch-preserving tempo augmentation.

    Params (see configs/augmentation/speed.yaml):
      speed_factor, input_audio_key, source_batch, product_root, ...
    """

    name = "tempo"
    version = OPERATOR_VERSION
    category = "augmentation"

    def dependency_fingerprint(self, config: OperatorConfig) -> dict[str, Any] | None:
        params = dict(config.params or {})
        mock = bool(config.mock or params.get("mock"))
        try:
            if mock:
                version = "mock"
            else:
                version = probe_ffmpeg(str(params.get("ffmpeg_bin") or "ffmpeg")).version
        except Exception as exc:  # noqa: BLE001
            version = f"unavailable:{exc}"
        return {
            "operator_version": self.version,
            "speed_factor": params.get("speed_factor"),
            "backend": params.get("backend", "ffmpeg_atempo"),
            "ffmpeg_version": version,
            "input_audio_key": params.get("input_audio_key"),
            "mock": mock,
        }

    def input_fingerprint(
        self, samples: list[Sample], config: OperatorConfig
    ) -> str | None:
        import hashlib
        import json

        rows = []
        key = str((config.params or {}).get("input_audio_key") or "resampled_16k")
        for sample in samples:
            audio = (sample.audio or {}).get(key) or ""
            rows.append({"id": sample.id, "audio": str(audio), "sha256": sample.sha256})
        raw = json.dumps(rows, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def run(self, samples: list[Sample], config: OperatorConfig) -> list[Sample]:
        started = time.perf_counter()
        params_raw = dict(config.params or {})
        mock = bool(config.mock or params_raw.get("mock"))
        force = bool(config.force or params_raw.get("force"))

        source_batch = str(params_raw.get("source_batch") or params_raw.get("batch") or "").strip()
        input_manifest = params_raw.get("input_manifest") or params_raw.get("bound_input_manifest")
        if not source_batch:
            raise AugmentPreflightError(
                "augmentation.tempo requires params.source_batch "
                "(set via --source-name on augment_batch_speed)"
            )
        if not input_manifest:
            # Fall back to run_dir frozen path if caller injected it.
            raise AugmentPreflightError(
                "augmentation.tempo requires params.input_manifest "
                "(frozen absolute path of the bound snapshot)"
            )

        preflight = preflight_tempo(
            params_raw=params_raw,
            source_batch=source_batch,
            input_manifest=input_manifest,
            samples=list(samples),
            mock=mock,
        )
        tempo: TempoParams = preflight["params"]
        product_root: Path = Path(params_raw.get("product_root") or preflight["product_root"])
        recipe_digest = str(params_raw.get("recipe_digest") or preflight["recipe_digest"])
        derived_batch = str(params_raw.get("derived_batch") or preflight["derived_batch"])
        filter_graph = str(preflight["filter_graph"])
        ffmpeg_version = str(preflight["ffmpeg_version"])
        binding_source = str(preflight["snapshot"]["binding_source"])

        product_root.mkdir(parents=True, exist_ok=True)
        cache_dir = Path(config.cache_dir) / "augmentation_tempo"
        cache_dir.mkdir(parents=True, exist_ok=True)

        recipe = build_recipe(
            source_batch=source_batch,
            input_manifest=Path(input_manifest),
            input_row_count=len(samples),
            params=tempo,
            ffmpeg_version=ffmpeg_version,
            filter_graph=filter_graph,
            derived_batch=derived_batch,
            product_root=product_root,
            binding_source=binding_source,
        )
        # Authoritative digest comes from the frozen recipe payload.
        recipe_digest = str(recipe["recipe_digest"])
        derived_batch = str(recipe.get("derived_batch") or derived_batch)

        # Resume fingerprint guard: frozen recipe must match the rebuilt identity.
        existing_recipe_path = product_root / "recipe.json"
        if existing_recipe_path.is_file() and not force:
            import json

            frozen = json.loads(existing_recipe_path.read_text(encoding="utf-8"))
            if float(frozen.get("speed_factor")) != float(tempo.speed_factor):
                raise AugmentPreflightError(
                    f"resume recipe speed_factor={frozen.get('speed_factor')} "
                    f"!= current {tempo.speed_factor}; start a new run"
                )
            frozen_in = str(frozen.get("input_manifest") or "")
            if frozen_in and Path(frozen_in).resolve() != Path(input_manifest).resolve():
                raise AugmentPreflightError(
                    f"resume input manifest mismatch: frozen={frozen_in} "
                    f"current={Path(input_manifest).resolve()}"
                )
            if str(frozen.get("recipe_digest")) != recipe_digest:
                raise AugmentPreflightError(
                    f"resume recipe_digest mismatch: frozen={frozen.get('recipe_digest')} "
                    f"current={recipe_digest}; start a new run for the new recipe"
                )

        # Persist recipe early so resume / shards share the same identity.
        atomic_write_json(product_root / "recipe.json", recipe)

        backend_info = None if mock else probe_ffmpeg(tempo.ffmpeg_bin)
        workers = max(1, min(int(tempo.workers), len(samples) or 1))
        max_in_flight = max(workers, int(tempo.max_in_flight))
        # Total FFmpeg children bounded by max_in_flight (gate inside run_atempo).
        logger.info(
            "tempo augment: batch={} factor={} pitch=preserve backend={} "
            "ffmpeg={} workers={} max_in_flight={} backend_threads={} "
            "input_key={} rows={} product_root={}",
            source_batch,
            tempo.speed_factor,
            tempo.backend,
            ffmpeg_version,
            workers,
            max_in_flight,
            tempo.backend_threads,
            tempo.input_audio_key,
            len(samples),
            product_root,
        )

        source_order = [s.id for s in samples]
        results_rows: list[dict[str, Any]] = []
        success_samples: list[Sample] = []
        # Keep failed placeholders out of formal manifest but covered in results.
        out_samples: list[Sample] = []

        def _process_one(index: int, sample: Sample) -> tuple[Sample, dict[str, Any]]:
            row_started = time.perf_counter()
            row: dict[str, Any] = {
                "source_id": sample.id,
                "source_index": index,
                "status": "failed",
                "cache_hit": False,
                "error_code": "",
                "error_message": "",
                "derived_id": "",
                "output_path": "",
                "elapsed_sec": 0.0,
            }
            try:
                input_path = resolve_input_audio(
                    sample,
                    input_key=tempo.input_audio_key,
                    allow_raw_fallback=(tempo.input_audio_key == "raw"),
                )
                info = read_audio_info(input_path)
                row["input_path"] = str(info["path"])
                row["input_sha256"] = info["sha256"]
                row["input_duration"] = info["duration"]
                row["input_sample_rate"] = info["sample_rate"]
                row["input_channels"] = info["channels"]

                expected_out_dur = float(info["duration"]) / float(tempo.speed_factor)
                fp = processing_fingerprint(
                    source_batch=source_batch,
                    source_id=sample.id,
                    input_audio_sha256=info["sha256"],
                    params=tempo,
                    ffmpeg_version=ffmpeg_version,
                    filter_graph=filter_graph,
                )
                derived_id = derived_sample_id(fp)
                out_path = audio_output_path(product_root, derived_id)
                row["derived_id"] = derived_id
                row["output_path"] = str(out_path)
                ckey = cache_key_for(
                    source_batch=source_batch,
                    source_id=sample.id,
                    input_key=tempo.input_audio_key,
                    input_audio_sha256=info["sha256"],
                    params=tempo,
                    ffmpeg_version=ffmpeg_version,
                    filter_graph=filter_graph,
                )
                row["cache_key"] = ckey
                cache_path = cache_dir / f"{ckey}.json"

                reuse = False
                meta: dict[str, Any] = {
                    "filter_graph": filter_graph,
                    "ffmpeg_version": ffmpeg_version,
                }
                if not force and out_path.is_file() and cache_path.is_file():
                    try:
                        import json

                        cached = json.loads(cache_path.read_text(encoding="utf-8"))
                        validated = validate_output_wav(
                            out_path,
                            expected_sr=info["sample_rate"],
                            expected_channels=info["channels"],
                            expected_duration=expected_out_dur,
                            duration_tol=duration_tolerance(expected_out_dur),
                        )
                        if (
                            cached.get("output_sha256") == validated["sha256"]
                            and abs(float(cached.get("speed_factor")) - tempo.speed_factor) == 0
                            and cached.get("input_audio_sha256") == info["sha256"]
                        ):
                            reuse = True
                            row["cache_hit"] = True
                            meta = cached
                            out_info = validated
                    except Exception:  # noqa: BLE001 — treat as miss
                        reuse = False

                if not reuse:
                    last_exc: Exception | None = None
                    for attempt in range(1, tempo.max_attempts + 1):
                        try:
                            if mock:
                                meta = _mock_atempo(
                                    input_path=info["path"],
                                    output_path=out_path,
                                    speed_factor=tempo.speed_factor,
                                )
                            else:
                                meta = run_atempo(
                                    input_path=info["path"],
                                    output_path=out_path,
                                    params=tempo,
                                    backend=backend_info,
                                    gate_limit=max_in_flight,
                                )
                            out_info = validate_output_wav(
                                out_path,
                                expected_sr=info["sample_rate"],
                                expected_channels=info["channels"],
                                expected_duration=expected_out_dur,
                                duration_tol=duration_tolerance(expected_out_dur),
                            )
                            last_exc = None
                            break
                        except (TimeoutError, OSError) as exc:
                            last_exc = exc
                            if attempt >= tempo.max_attempts:
                                raise
                            logger.warning(
                                "tempo retry {}/{} for {}: {}",
                                attempt,
                                tempo.max_attempts,
                                sample.id,
                                exc,
                            )
                        except Exception:
                            raise
                    if last_exc is not None:
                        raise last_exc

                    cache_payload = {
                        "output_sha256": out_info["sha256"],
                        "output_path": str(out_path),
                        "output_duration": out_info["duration"],
                        "speed_factor": tempo.speed_factor,
                        "input_audio_sha256": info["sha256"],
                        "filter_graph": meta.get("filter_graph", filter_graph),
                        "ffmpeg_version": meta.get("ffmpeg_version", ffmpeg_version),
                        "derived_id": derived_id,
                        "processing_fingerprint": fp,
                    }
                    atomic_write_json(cache_path, cache_payload)
                    meta = {**meta, **cache_payload}
                else:
                    out_info = validate_output_wav(
                        out_path,
                        expected_sr=info["sample_rate"],
                        expected_channels=info["channels"],
                    )

                aug_label = {
                    "type": "tempo",
                    "speed_factor": tempo.speed_factor,
                    "preserve_pitch": True,
                    "backend": tempo.backend,
                    "ffmpeg_version": meta.get("ffmpeg_version", ffmpeg_version),
                    "filter_graph": meta.get("filter_graph", filter_graph),
                    "source_batch": source_batch,
                    "source_id": sample.id,
                    "source_index": index,
                    "source_manifest": str(Path(input_manifest).resolve()),
                    "source_manifest_sha256": recipe.get("input_manifest_sha256"),
                    "input_audio_key": tempo.input_audio_key,
                    "input_path": str(info["path"]),
                    "input_sha256": info["sha256"],
                    "input_duration": info["duration"],
                    "output_audio_key": tempo.output_audio_key,
                    "output_path": str(out_path),
                    "output_sha256": out_info["sha256"],
                    "output_duration": out_info["duration"],
                    "output_sample_rate": out_info["sample_rate"],
                    "output_channels": out_info["channels"],
                    "processing_fingerprint": fp,
                    "cache_key": ckey,
                    "derived_batch": derived_batch,
                    "recipe_digest": recipe_digest,
                    "operator_version": self.version,
                }
                derived = build_derived_sample(
                    sample,
                    derived_id=derived_id,
                    output_path=str(out_path.resolve()),
                    output_sha256=out_info["sha256"],
                    sample_rate=out_info["sample_rate"],
                    channels=out_info["channels"],
                    duration=out_info["duration"],
                    augmentation_label=aug_label,
                    operator_name=self.full_name,
                    operator_version=self.version,
                )
                row.update(
                    {
                        "status": "ok",
                        "output_duration": out_info["duration"],
                        "output_sha256": out_info["sha256"],
                        "error_code": "",
                        "error_message": "",
                    }
                )
                row["elapsed_sec"] = time.perf_counter() - row_started
                return derived, row
            except Exception as exc:  # noqa: BLE001 — per-row failure stays in results
                code = type(exc).__name__
                message = str(exc)
                if len(message) > 2000:
                    message = message[:2000] + "…"
                row["error_code"] = code
                row["error_message"] = message
                row["elapsed_sec"] = time.perf_counter() - row_started
                failed = sample.model_copy(deep=True)
                failed.mark_failed(self.full_name, message)
                failed.labels = dict(failed.labels or {})
                failed.labels["augmentation_attempt"] = {
                    "type": "tempo",
                    "speed_factor": tempo.speed_factor,
                    "source_id": sample.id,
                    "source_index": index,
                    "derived_id": row.get("derived_id"),
                    "error_code": code,
                }
                logger.error(
                    "tempo failed source_id={} path={} err={}",
                    sample.id,
                    row.get("input_path") or sample.source_path,
                    message,
                )
                logger.debug("tempo traceback:\n{}", traceback.format_exc())
                return failed, row

        # Bounded in-flight futures — never create one Future per million rows upfront.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending_idx = 0
            in_flight: dict[Any, tuple[int, Sample]] = {}
            completed_map: dict[int, tuple[Sample, dict[str, Any]]] = {}

            def _submit_more() -> None:
                nonlocal pending_idx
                while pending_idx < len(samples) and len(in_flight) < max_in_flight:
                    idx = pending_idx
                    sample = samples[idx]
                    fut = pool.submit(_process_one, idx, sample)
                    in_flight[fut] = (idx, sample)
                    pending_idx += 1

            _submit_more()
            while in_flight:
                done, _ = wait(in_flight.keys(), return_when=FIRST_COMPLETED)
                for fut in done:
                    idx, _sample = in_flight.pop(fut)
                    completed_map[idx] = fut.result()
                _submit_more()

        for index in range(len(samples)):
            sample_out, row = completed_map[index]
            results_rows.append(row)
            out_samples.append(sample_out)
            if row.get("status") == "ok":
                success_samples.append(sample_out)

        # Shard-safe: when defer_publish, only write shard result part; parent finalizes.
        defer = bool(params_raw.get("defer_publish"))
        if defer:
            part_dir = product_root / "results_parts"
            part_dir.mkdir(parents=True, exist_ok=True)
            part_name = str(
                params_raw.get("shard_id")
                or (Path(config.run_dir).name if config.run_dir else None)
                or f"part_{time.time_ns()}"
            )
            from audio_engine.core.augmentation.publish import write_results_parquet

            write_results_parquet(part_dir / f"{part_name}.parquet", results_rows)
            # Also keep a run-local copy for debugging.
            if config.run_dir:
                atomic_write_json(
                    Path(config.run_dir) / "tempo_shard_summary.json",
                    {
                        "success": sum(1 for r in results_rows if r["status"] == "ok"),
                        "failed": sum(1 for r in results_rows if r["status"] != "ok"),
                        "total": len(results_rows),
                    },
                )
            return out_samples

        summary = publish_batch_artifacts(
            product_root=product_root,
            recipe=recipe,
            results_rows=results_rows,
            success_samples=success_samples,
            source_order=source_order,
            started_at=started,
            raise_on_failure=True,
        )
        logger.info(
            "tempo publish: status={} success={} failed={} root={}",
            summary.get("status"),
            summary.get("success"),
            summary.get("failed"),
            product_root,
        )
        # Return successful derived samples in input order for pipeline output_manifest.
        return [
            s
            for s in out_samples
            if s.status.get(self.full_name) == "completed" and str(s.id).startswith("tempo_")
        ]
