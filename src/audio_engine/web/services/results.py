"""Dataset/sample/result helpers reusing existing manifests, tagging and annotation."""

from __future__ import annotations

import hashlib
import json
import mimetypes
from pathlib import Path
from typing import Any

from audio_engine.web.settings import WebSettings
from audio_engine.web.store import ConflictError, JobStore

TAG_ACTION_MAP = {
    "manual_add": "add_tag",
    "manual_block": "remove_tag",
    "restore_auto": "restore_tag",
    "retract_manual": "retract_manual",
}

DEFAULT_TAGGING_CONFIG = "configs/tagging/zh_asr_free_tags_v2.yaml"
DEFAULT_ANNOTATION_CONFIG = "configs/annotation/zh_asr_v3.yaml"


class ResultsService:
    def __init__(self, settings: WebSettings, store: JobStore | None = None) -> None:
        self.settings = settings
        self.store = store

    # ------------------------------------------------------------------ path helpers

    def _allowed_roots(self) -> list[Path]:
        return [self.settings.project_root.resolve(), *[r.resolve() for r in self.settings.allowed_source_roots]]

    def _ensure_allowed(self, path: Path) -> Path:
        path = path.resolve()
        for root in self._allowed_roots():
            try:
                path.relative_to(root)
                return path
            except ValueError:
                continue
        raise PermissionError(f"path outside allowed roots: {path}")

    def _binding_path(self, dataset_path: Path) -> Path:
        return dataset_path.with_suffix(dataset_path.suffix + ".web_meta.json")

    def _work_manifest_path(self, published_path: Path) -> Path:
        return published_path.with_name(f"{published_path.stem}.annotation_work{published_path.suffix}")

    def load_binding(self, dataset_path: Path) -> dict[str, Any]:
        meta = self._binding_path(dataset_path)
        if not meta.is_file():
            return {}
        try:
            data = json.loads(meta.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def save_binding(self, dataset_path: Path, binding: dict[str, Any]) -> Path:
        path = self._binding_path(dataset_path)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(binding, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(path)
        return path

    def _resolve_dataset(self, dataset_id: str) -> Path:
        text = str(dataset_id or "").strip()
        if not text:
            raise FileNotFoundError("empty dataset_id")

        candidates: list[Path] = []
        raw = Path(text)
        if raw.is_absolute():
            candidates.append(raw)
        else:
            candidates.append(self.settings.project_root / raw)
            try:
                from audio_engine.core.source_naming import resolve_existing_manifest

                resolved = resolve_existing_manifest(
                    text,
                    manifests_dir=self.settings.project_root / "datasets" / "manifests",
                )
                if not resolved.is_absolute():
                    resolved = self.settings.project_root / resolved
                candidates.append(resolved)
            except Exception:
                pass

        last_err: Exception | None = None
        for cand in candidates:
            try:
                path = self._ensure_allowed(cand)
                if path.is_file():
                    return path
            except Exception as exc:  # noqa: BLE001
                last_err = exc
        if isinstance(last_err, PermissionError):
            raise last_err
        raise FileNotFoundError(dataset_id)

    def _rel_id(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.settings.project_root.resolve()).as_posix()
        except ValueError:
            return str(path.resolve().as_posix())

    def _load_manifest(self, path: Path):
        from audio_engine.core.manifest import Manifest

        return Manifest.load(path)

    def _sample_tags(self, sample) -> list[str]:
        labels = sample.labels or {}
        tags = labels.get("tags")
        if isinstance(tags, list):
            return [str(t) for t in tags]
        if isinstance(tags, str) and tags:
            return [tags]
        return []

    def _sample_ternaries(self, sample) -> dict[str, str]:
        labels = sample.labels or {}
        tagging = labels.get("tagging") if isinstance(labels.get("tagging"), dict) else {}
        coverage = tagging.get("rule_coverage") if isinstance(tagging.get("rule_coverage"), dict) else {}
        return {str(k): str(v) for k, v in coverage.items()}

    def _sample_row(self, sample, *, tag_overlay: list[str] | None = None) -> dict[str, Any]:
        labels = sample.labels or {}
        quality = sample.quality or {}
        tags = tag_overlay if tag_overlay is not None else self._sample_tags(sample)
        ternary = self._sample_ternaries(sample)
        route_preview = None
        for key, val in (sample.transcripts or {}).items():
            if isinstance(val, dict) and val.get("text"):
                route_preview = {"route": key, "text": str(val.get("text"))[:120]}
                break
        return {
            "sample_id": sample.id,
            "duration": sample.duration,
            "tags": tags,
            "status": dict(sample.status or {}),
            "errors": dict(sample.errors or {}),
            "ternary": ternary,
            "quality": {
                k: quality.get(k)
                for k in ("dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl", "dnsmos_status")
                if k in quality
            },
            "annotation_state": labels.get("annotation_state"),
            "gold_kind": labels.get("gold_kind"),
            "has_formal_gold": None,  # filled on detail when needed
            "transcript_preview": route_preview,
            "lineage_parents": self._lineage_summary(sample),
        }

    def _lineage_summary(self, sample) -> dict[str, Any]:
        labels = sample.labels or {}
        parent = labels.get("parent_sample_id") or labels.get("parent_id") or labels.get("source_sample_id")
        segment = labels.get("segment_id") or labels.get("fragment_id")
        return {
            "parent_sample_id": parent,
            "segment_id": segment,
            "source_path": sample.source_path,
            "lineage_ops": [
                (e.operator if hasattr(e, "operator") else e.get("operator"))
                for e in (sample.lineage or [])[:8]
            ],
        }

    # ------------------------------------------------------------------ datasets

    def list_datasets(self) -> dict[str, Any]:
        root = self.settings.project_root / "datasets"
        items: list[dict[str, Any]] = []
        if root.is_dir():
            for path in sorted(root.rglob("*.parquet")):
                if path.name.endswith(".annotation_work.parquet"):
                    continue
                rel = path.relative_to(self.settings.project_root).as_posix()
                binding = self.load_binding(path)
                work = self._work_manifest_path(path)
                items.append(
                    {
                        "dataset_id": rel,
                        "path": rel,
                        "format": "parquet",
                        "registered": True,
                        "has_work_copy": work.is_file(),
                        "work_dataset_id": self._rel_id(work) if work.is_file() else None,
                        "state_kind": "published",
                        "tagging_binding": {
                            "batch_id": binding.get("batch_id"),
                            "work_root": binding.get("work_root"),
                            "tagging_config": binding.get("tagging_config"),
                        }
                        if binding
                        else None,
                    }
                )
        return {"items": items[:500], "next_cursor": None, "total": len(items)}

    def get_dataset(self, dataset_id: str) -> dict[str, Any]:
        path = self._resolve_dataset(dataset_id)
        binding = self.load_binding(path)
        work = self._work_manifest_path(path)
        is_work = path.name.endswith(".annotation_work.parquet") or path.name.endswith(
            ".annotation_work.jsonl"
        )
        manifest = self._load_manifest(path)
        formal = 0
        try:
            from audio_engine.core.annotation_v3.gold import has_formal_gold_evidence

            for s in manifest.samples:
                if has_formal_gold_evidence(s, require_dual=False):
                    formal += 1
        except Exception:
            formal = 0
        return {
            "dataset_id": self._rel_id(path),
            "path": str(path),
            "exists": True,
            "sample_count": len(manifest.samples),
            "formal_gold_count": formal,
            "state_kind": "working" if is_work else "published",
            "immutable_published": not is_work,
            "has_work_copy": work.is_file() and not is_work,
            "work_dataset_id": self._rel_id(work) if work.is_file() and not is_work else None,
            "published_dataset_id": None if is_work else self._rel_id(path),
            "tagging_binding": binding or None,
            "note": None
            if binding
            else "缺 tagging 绑定元数据时人工标签需在请求中提供 work_root/batch_id",
        }

    def _match_filters(self, sample, filters: dict[str, Any], *, overlay_tags: list[str] | None) -> bool:
        tags = overlay_tags if overlay_tags is not None else self._sample_tags(sample)
        tag_set = set(tags)

        tag = filters.get("tag")
        if tag and tag not in tag_set:
            return False
        tag_exclude = filters.get("tag_exclude")
        if tag_exclude and tag_exclude in tag_set:
            return False
        include_any = filters.get("tag_any") or filters.get("include_any")
        if include_any:
            wanted = {t.strip() for t in str(include_any).split(",") if t.strip()}
            if wanted and not (tag_set & wanted):
                return False
        include_all = filters.get("tag_all") or filters.get("include_all")
        if include_all:
            wanted = {t.strip() for t in str(include_all).split(",") if t.strip()}
            if wanted and not wanted.issubset(tag_set):
                return False

        ternary = filters.get("ternary")
        if ternary:
            states = self._sample_ternaries(sample)
            if not any(v == ternary for v in states.values()):
                return False
        tag_id_ternary = filters.get("tag_ternary")
        if tag_id_ternary and ":" in str(tag_id_ternary):
            tid, want = str(tag_id_ternary).split(":", 1)
            if self._sample_ternaries(sample).get(tid) != want:
                return False

        route_status = filters.get("route_status")
        if route_status:
            # form: key=value or key
            status = sample.status or {}
            if "=" in route_status:
                k, v = route_status.split("=", 1)
                if status.get(k) != v:
                    return False
            elif route_status not in status:
                return False

        review_status = filters.get("review_status") or filters.get("annotation_state")
        if review_status:
            if str((sample.labels or {}).get("annotation_state") or "") != review_status:
                return False

        q_key = filters.get("quality_key") or "dnsmos_sig"
        q_min = filters.get("quality_min")
        q_max = filters.get("quality_max")
        if q_min is not None or q_max is not None:
            quality = sample.quality or {}
            raw = quality.get(q_key)
            if raw is None:
                return False
            try:
                score = float(raw)
            except (TypeError, ValueError):
                return False
            if q_min is not None and score < float(q_min):
                return False
            if q_max is not None and score > float(q_max):
                return False
        return True

    def _overlay_tags_map(self, binding: dict[str, Any]) -> dict[str, list[str]]:
        work_root = binding.get("work_root")
        if not work_root:
            return {}
        state_path = Path(str(work_root)) / "tag_state.json"
        if not state_path.is_file():
            return {}
        try:
            from audio_engine.core.tagging.state import TagStateStore, effective_tags

            store = TagStateStore(state_path)
            try:
                return {
                    sid: effective_tags(st)
                    for sid, st in store.all_states().items()
                }
            finally:
                store.close()
        except Exception:
            return {}

    def list_samples(
        self,
        dataset_id: str,
        *,
        limit: int = 50,
        offset: int = 0,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        filters = {k: v for k, v in (filters or {}).items() if v not in (None, "")}
        path = self._resolve_dataset(dataset_id)
        binding = self.load_binding(path)
        overlay = self._overlay_tags_map(binding)
        samples = list(self._load_manifest(path).samples)

        matched: list[Any] = []
        for s in samples:
            ov = overlay.get(s.id)
            if self._match_filters(s, filters, overlay_tags=ov):
                matched.append((s, ov))

        limit = max(1, min(int(limit or 50), 200))
        offset = max(0, int(offset or 0))
        page = matched[offset : offset + limit]
        items = [self._sample_row(s, tag_overlay=ov) for s, ov in page]
        frozen_filters = dict(filters)
        return {
            "items": items,
            "next_cursor": offset + limit if offset + limit < len(matched) else None,
            "total": len(matched),
            "offset": offset,
            "limit": limit,
            "filters": frozen_filters,
            "dataset_id": self._rel_id(path),
            "state_kind": "working"
            if path.name.endswith(".annotation_work.parquet")
            else "published",
            "tag_overlay_applied": bool(overlay),
        }

    def get_sample(self, dataset_id: str, sample_id: str) -> dict[str, Any]:
        path = self._resolve_dataset(dataset_id)
        binding = self.load_binding(path)
        overlay = self._overlay_tags_map(binding)
        for s in self._load_manifest(path).samples:
            if s.id == sample_id:
                data = s.model_dump()
                data["sample_id"] = s.id
                if sample_id in overlay:
                    data["effective_tags_working"] = overlay[sample_id]
                    data["effective_tags_published"] = self._sample_tags(s)
                else:
                    data["effective_tags_working"] = self._sample_tags(s)
                    data["effective_tags_published"] = self._sample_tags(s)
                data["lineage_summary"] = self._lineage_summary(s)
                data["state_kind"] = (
                    "working"
                    if path.name.endswith(".annotation_work.parquet")
                    else "published"
                )
                try:
                    from audio_engine.core.annotation_v3.gold import has_formal_gold_evidence

                    data["has_formal_gold"] = has_formal_gold_evidence(s, require_dual=False)
                except Exception:
                    data["has_formal_gold"] = False
                # tag state revision if work store present
                if binding.get("work_root"):
                    try:
                        from audio_engine.core.tagging.state import TagStateStore

                        store = TagStateStore(Path(str(binding["work_root"])) / "tag_state.json")
                        try:
                            st = store.get(sample_id)
                            data["tag_revision"] = str(st.version)
                            data["tag_sources"] = [x.as_dict() for x in st.sources]
                        finally:
                            store.close()
                    except Exception:
                        pass
                return data
        raise FileNotFoundError(sample_id)

    def resolve_audio(
        self, dataset_id: str, sample_id: str, *, asset: str = "raw"
    ) -> tuple[Path, str, dict[str, Any]]:
        path = self._resolve_dataset(dataset_id)
        sample = None
        for s in self._load_manifest(path).samples:
            if s.id == sample_id:
                sample = s
                break
        if sample is None:
            raise FileNotFoundError(sample_id)

        meta: dict[str, Any] = {"asset": asset, "pcm_preview": False}
        try:
            path_value = sample.audio_path(asset)
        except KeyError as exc:
            # PCM without params cannot be previewed as browser audio.
            fmt = str((sample.labels or {}).get("audio_format") or "").lower()
            if fmt == "pcm" or asset.endswith(".pcm"):
                missing = [
                    k
                    for k in ("sample_rate", "bit_depth", "channels")
                    if not getattr(sample, k, None)
                    and not (sample.labels or {}).get(k)
                ]
                raise FileNotFoundError(
                    f"PCM 无法试听：缺少参数 {missing or ['解释参数']}"
                ) from exc
            raise FileNotFoundError(str(exc)) from exc

        audio_path = Path(str(path_value))
        if not audio_path.is_absolute():
            audio_path = (self.settings.project_root / audio_path).resolve()
        else:
            audio_path = audio_path.resolve()
        # Resolve symlinks then check allowed roots
        audio_path = self._ensure_allowed(audio_path.resolve())
        if not audio_path.is_file():
            raise FileNotFoundError(str(audio_path))
        ctype = mimetypes.guess_type(str(audio_path))[0] or "application/octet-stream"
        if audio_path.suffix.lower() == ".wav":
            ctype = "audio/wav"
        elif audio_path.suffix.lower() in {".mp3"}:
            ctype = "audio/mpeg"
        elif audio_path.suffix.lower() in {".flac"}:
            ctype = "audio/flac"
        meta["path"] = str(audio_path)
        return audio_path, ctype, meta

    def explain_sample(
        self,
        dataset_id: str,
        sample_id: str,
        *,
        tagging_config: str | None = None,
    ) -> dict[str, Any]:
        """Read-only explanation from existing sample fields; never runs inference."""

        path = self._resolve_dataset(dataset_id)
        sample_obj = None
        for s in self._load_manifest(path).samples:
            if s.id == sample_id:
                sample_obj = s
                break
        if sample_obj is None:
            raise FileNotFoundError(sample_id)

        labels = sample_obj.labels or {}
        tagging = labels.get("tagging") if isinstance(labels.get("tagging"), dict) else {}
        snapshot_id = tagging.get("snapshot_id")
        rule_coverage = tagging.get("rule_coverage") or {}
        binding = self.load_binding(path)

        current_rules: dict[str, Any] = {}
        rules_differ = False
        cfg_path = tagging_config or binding.get("tagging_config") or DEFAULT_TAGGING_CONFIG
        try:
            from audio_engine.core.tagging.service import TaggingService

            svc = TaggingService.from_path(self.settings.resolve(cfg_path))
            # Compare definition versions only — do not evaluate/infer.
            for tid, tag in (svc.config.tags or {}).items():
                current_rules[tid] = {
                    "version": getattr(tag, "version", None) or getattr(tag, "rule_version", None),
                    "enabled": bool(getattr(tag, "enabled", True)),
                    "archived": bool(getattr(tag, "archived", False)),
                }
            for rid in rule_coverage:
                if rid in current_rules:
                    # If sample stored rule versions under tagging, compare when present
                    pass
                elif rid not in current_rules:
                    rules_differ = True
        except Exception as exc:  # noqa: BLE001
            current_rules = {"_error": str(exc)}

        # Prefer published definition snapshot next to work_root if present
        snapshot_path = None
        work_root = binding.get("work_root")
        if work_root:
            for cand in (
                Path(str(work_root)) / "published" / "CURRENT.json",
                Path(str(work_root)) / "definition_snapshot.json",
            ):
                if cand.is_file():
                    snapshot_path = str(cand)
                    break

        return {
            "sample_id": sample_id,
            "dataset_id": self._rel_id(path),
            "tags": self._sample_tags(sample_obj),
            "transcripts": sample_obj.transcripts,
            "metrics": sample_obj.quality,
            "quality": sample_obj.quality,
            "lineage": [e.model_dump() if hasattr(e, "model_dump") else e for e in sample_obj.lineage],
            "lineage_summary": self._lineage_summary(sample_obj),
            "labels": labels,
            "rule_coverage": rule_coverage,
            "snapshot_id": snapshot_id,
            "snapshot_path": snapshot_path,
            "current_library_rules": current_rules,
            "rules_may_differ_from_run": rules_differ or bool(snapshot_id),
            "rules_differ_note": (
                "解释使用样本内已保存的规则结果/快照；当前库定义可能已变更，"
                "不能用最新规则重新解释旧命中。本接口未触发 ASR/DNSMOS。"
                if snapshot_id or rules_differ
                else "未找到运行快照时仅展示样本内已有字段。"
            ),
            "inference_invoked": False,
            "state_kind": "working"
            if path.name.endswith(".annotation_work.parquet")
            else "published",
        }

    # ------------------------------------------------------------------ mutations

    def _request_hash(self, payload: dict[str, Any]) -> str:
        blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _idempotent(
        self,
        *,
        scope: str,
        idempotency_id: str | None,
        request_payload: dict[str, Any],
        runner,
    ) -> dict[str, Any]:
        if not idempotency_id or self.store is None:
            return runner()
        req_hash = self._request_hash(request_payload)
        existing = self.store.get_mutation_idempotency(scope=scope, idempotency_id=idempotency_id)
        if existing:
            if existing["request_hash"] != req_hash:
                raise ConflictError(
                    "IDEMPOTENCY_CONFLICT",
                    "idempotency_id reused with different request",
                    {"idempotency_id": idempotency_id, "scope": scope},
                )
            cached = json.loads(existing["response_json"])
            cached["idempotent_replay"] = True
            return cached
        result = runner()
        return self.store.put_mutation_idempotency(
            scope=scope,
            idempotency_id=idempotency_id,
            request_hash=req_hash,
            response=result,
        )

    def apply_tag_action(
        self,
        dataset_id: str,
        sample_id: str,
        *,
        action: str,
        tag_id: str,
        operator_id: str,
        reason: str = "",
        expected_revision: str | None = None,
        idempotency_id: str | None = None,
        work_root: str | None = None,
        batch_id: str | None = None,
        tagging_config: str | None = None,
        audio_sha256: str | None = None,
    ) -> dict[str, Any]:
        if action not in TAG_ACTION_MAP:
            raise ValueError(f"unsupported action: {action}")

        path = self._resolve_dataset(dataset_id)
        binding = dict(self.load_binding(path))
        work_root = work_root or binding.get("work_root")
        batch_id = batch_id or binding.get("batch_id") or path.stem
        tagging_config = tagging_config or binding.get("tagging_config") or DEFAULT_TAGGING_CONFIG
        if not work_root:
            raise ValueError(
                "缺少 tagging work_root：请在请求中提供，或先写入 dataset.web_meta.json 绑定"
            )

        sample = None
        for s in self._load_manifest(path).samples:
            if s.id == sample_id:
                sample = s
                break
        if sample is None:
            raise FileNotFoundError(sample_id)
        sha = audio_sha256 or sample.sha256 or ""

        expected_version = None
        if expected_revision not in (None, ""):
            expected_version = int(expected_revision)

        cfg_path = self.settings.resolve(tagging_config)
        work_path = self.settings.resolve(work_root) if not Path(work_root).is_absolute() else Path(work_root)
        work_path = self._ensure_allowed(work_path)

        request_payload = {
            "dataset_id": self._rel_id(path),
            "sample_id": sample_id,
            "action": action,
            "tag_id": tag_id,
            "operator_id": operator_id,
            "reason": reason,
            "expected_revision": expected_revision,
            "work_root": str(work_path),
            "batch_id": batch_id,
            "tagging_config": str(cfg_path),
        }

        def _run() -> dict[str, Any]:
            from audio_engine.core.tagging.service import TaggingService
            from audio_engine.core.tagging.state import ConcurrentModificationError

            svc = TaggingService.from_path(cfg_path)
            method = getattr(svc, TAG_ACTION_MAP[action])
            try:
                result = method(
                    batch_id=batch_id,
                    sample_id=sample_id,
                    tag_id=tag_id,
                    audio_sha256=sha,
                    actor=operator_id,
                    reason=reason,
                    work_root=work_path,
                    expected_version=expected_version,
                )
            except ConcurrentModificationError as exc:
                raise ConflictError(
                    "REVISION_CONFLICT",
                    str(exc),
                    {"sample_id": sample_id, "expected_revision": expected_revision},
                ) from exc
            # Persist binding for subsequent UI calls
            try:
                cfg_ref = self._rel_id(cfg_path)
            except Exception:
                cfg_ref = str(cfg_path)
            binding.update(
                {
                    "batch_id": batch_id,
                    "work_root": str(work_path.as_posix()),
                    "tagging_config": cfg_ref,
                    "updated_by": operator_id,
                }
            )
            self.save_binding(path, binding)
            result["available"] = True
            result["ok"] = True
            result["dataset_id"] = self._rel_id(path)
            result["published_dataset_unchanged"] = True
            result["note"] = (
                "人工状态写入工作区 tag_state，未原地覆盖已发布 Manifest；"
                "需正式发布时另走 tagging publish/导出任务。"
            )
            return result

        return self._idempotent(
            scope=f"tag-action:{self._rel_id(path)}:{sample_id}",
            idempotency_id=idempotency_id,
            request_payload=request_payload,
            runner=_run,
        )

    def apply_review(
        self,
        dataset_id: str,
        sample_id: str,
        *,
        operator_id: str,
        reason: str = "",
        expected_revision: str | None = None,
        idempotency_id: str | None = None,
        review_pass: str = "first",
        queue_id: str | None = None,
        decision: str = "accept",
        gold_kind: str | None = None,
        gold_text: str | None = None,
        speech_scope: str | None = None,
        human_semantic: str | None = None,
        human_noise: str | None = None,
        human_crosstalk: str | None = None,
        verified_error_tags: list[str] | None = None,
        audio_event_tags: list[str] | None = None,
        annotation_config: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Apply annotation_v3 review into a working copy; never overwrite published parquet."""

        published_path = self._resolve_dataset(dataset_id)
        is_already_work = published_path.name.endswith(".annotation_work.parquet")
        if is_already_work:
            work_path = published_path
            # published sibling guessed by stripping suffix
            pub_guess = published_path.with_name(
                published_path.name.replace(".annotation_work.parquet", ".parquet")
            )
            published_id = self._rel_id(pub_guess) if pub_guess.is_file() else None
        else:
            work_path = self._work_manifest_path(published_path)
            published_id = self._rel_id(published_path)

        cfg_rel = annotation_config or DEFAULT_ANNOTATION_CONFIG
        cfg_path = self.settings.resolve(cfg_rel)

        request_payload = {
            "dataset_id": self._rel_id(published_path),
            "sample_id": sample_id,
            "operator_id": operator_id,
            "reason": reason,
            "expected_revision": expected_revision,
            "review_pass": review_pass,
            "queue_id": queue_id,
            "decision": decision,
            "gold_kind": gold_kind,
            "gold_text": gold_text,
            "speech_scope": speech_scope,
            "human_semantic": human_semantic,
            "human_noise": human_noise,
            "human_crosstalk": human_crosstalk,
            "verified_error_tags": verified_error_tags or [],
            "audio_event_tags": audio_event_tags or [],
            "annotation_config": str(cfg_path),
        }
        if extra:
            request_payload["extra"] = extra

        def _run() -> dict[str, Any]:
            from audio_engine.core.annotation_v3.config import AnnotationConfig
            from audio_engine.core.annotation_v3.gold import has_formal_gold_evidence
            from audio_engine.core.annotation_v3.import_workflow import apply_review_import_v3
            from audio_engine.core.manifest import Manifest

            source = work_path if work_path.is_file() else published_path
            manifest = self._load_manifest(source)
            samples = list(manifest.samples)
            sample = next((s for s in samples if s.id == sample_id), None)
            if sample is None:
                raise FileNotFoundError(sample_id)

            revision = expected_revision or str(
                sample.labels.get("annotation_revision")
                or sample.labels.get("queue_revision")
                or "web_r1"
            )
            qid = queue_id or str(
                sample.labels.get("queue_id")
                or f"web::{self._rel_id(published_path)}::{revision}"
            )

            # Seed queue identity onto working samples when missing (does not touch published file).
            for s in samples:
                s.labels.setdefault("queue_id", qid)
                s.labels.setdefault("annotation_revision", revision)
                s.labels.setdefault("queue_revision", revision)
                if not s.labels.get("original_audio_sha256"):
                    s.labels["original_audio_sha256"] = s.sha256 or ""

            # Optimistic revision check against current working sample
            current_rev = str(
                sample.labels.get("annotation_revision")
                or sample.labels.get("queue_revision")
                or revision
            )
            if expected_revision not in (None, "") and str(expected_revision) != current_rev:
                # Allow first-time seed when sample had no revision yet on published copy
                published_only = not work_path.is_file()
                if not published_only:
                    raise ConflictError(
                        "REVISION_CONFLICT",
                        "annotation revision mismatch",
                        {
                            "expected_revision": expected_revision,
                            "current_revision": current_rev,
                            "sample_id": sample_id,
                        },
                    )

            row = {
                "sample_id": sample_id,
                "queue_id": qid,
                "queue_revision": revision,
                "review_revision": revision,
                "original_audio_sha256": sample.sha256 or sample.labels.get("original_audio_sha256") or "",
                "sha256": sample.sha256 or "",
                "decision": decision,
                "gold_kind": gold_kind,
                "gold_text": gold_text,
                "speech_scope": speech_scope,
                "human_semantic": human_semantic,
                "human_noise": human_noise,
                "human_crosstalk": human_crosstalk,
                "verified_error_tags": verified_error_tags or [],
                "audio_event_tags": audio_event_tags or [],
                "reason": reason,
                "annotator_id": operator_id,
                "type": sample.labels.get("type") or sample.labels.get("classification_bucket") or "",
            }
            if extra:
                for k, v in extra.items():
                    if k not in row:
                        row[k] = v

            config = AnnotationConfig.load(cfg_path)
            result = apply_review_import_v3(
                samples,
                [row],
                config=config,
                expected_queue_id=qid,
                expected_revision=revision,
                review_pass=review_pass,
                actor_id=operator_id,
            )
            if result.issues:
                return {
                    "ok": False,
                    "available": True,
                    "code": "REVIEW_REJECTED",
                    "message": "审核未通过既有 Gold/契约门禁",
                    "issues": [
                        {"sample_id": i.sample_id, "code": i.code, "message": i.message}
                        for i in result.issues
                    ],
                    "artifact_payload": result.artifact_payload,
                    "published_dataset_id": published_id,
                    "work_dataset_id": self._rel_id(work_path),
                    "state_kind": "working",
                    "gold_promoted": result.gold_promoted,
                }

            # Write ONLY to work copy — never replace published parquet in place.
            work_path.parent.mkdir(parents=True, exist_ok=True)
            Manifest(samples=result.samples).save(work_path)

            updated = next(s for s in result.samples if s.id == sample_id)
            formal = has_formal_gold_evidence(updated, require_dual=False)
            return {
                "ok": True,
                "available": True,
                "sample_id": sample_id,
                "review_pass": review_pass,
                "applied": result.applied,
                "gold_promoted": result.gold_promoted,
                "has_formal_gold": formal,
                "annotation_state": updated.labels.get("annotation_state"),
                "annotation_revision": updated.labels.get("annotation_revision"),
                "published_dataset_id": published_id,
                "work_dataset_id": self._rel_id(work_path),
                "state_kind": "working",
                "immutable_published_unchanged": True,
                "note": "人工审核写入 annotation_work 副本；已发布版本未原地覆盖。",
                "artifact_payload": result.artifact_payload,
            }

        return self._idempotent(
            scope=f"review:{self._rel_id(published_path)}:{sample_id}",
            idempotency_id=idempotency_id,
            request_payload=request_payload,
            runner=_run,
        )

    def export_via_plugin_request(
        self,
        dataset_id: str,
        *,
        export_plugin_id: str = "warehouse_export_annotation",
        filters: dict[str, Any] | None = None,
        parameters: dict[str, Any] | None = None,
        plugin_version: str = "1",
    ) -> dict[str, Any]:
        """Build a task request for export plugin; filters are frozen into parameters."""

        path = self._resolve_dataset(dataset_id)
        frozen = {k: v for k, v in (filters or {}).items() if v not in (None, "")}
        # Prefer work copy when exporting annotation state if present and filters say so
        export_dataset = self._rel_id(path)
        use_work = bool((parameters or {}).get("use_work_copy"))
        work = self._work_manifest_path(path)
        if use_work and work.is_file():
            export_dataset = self._rel_id(work)

        params = dict(parameters or {})
        params["filters"] = frozen
        params["frozen_dataset_id"] = export_dataset
        params["source_dataset_id"] = self._rel_id(path)
        return {
            "task_type": "pipeline",
            "root_plugin": {
                "kind": "pipeline",
                "id": export_plugin_id,
                "version": plugin_version,
            },
            "input": {"kind": "dataset", "dataset_id": export_dataset},
            "parameters": params,
            "model_bindings": [],
            "tags": [],
            "execution": {},
            "frozen_filters": frozen,
            "note": "提交时走 /plugins/plan + /jobs；筛选与数据集版本已冻结在请求中",
        }

    def filtered_sample_ids(self, dataset_id: str, filters: dict[str, Any] | None = None) -> list[str]:
        listed = self.list_samples(dataset_id, limit=200, offset=0, filters=filters)
        # For consistency checks in tests; production export uses plugin with frozen filters
        ids = [i["sample_id"] for i in listed["items"]]
        total = listed.get("total") or 0
        offset = listed.get("limit") or 200
        while offset < total:
            more = self.list_samples(dataset_id, limit=200, offset=offset, filters=filters)
            ids.extend(i["sample_id"] for i in more["items"])
            offset += 200
        return ids

    def inspect_gold_gate(
        self,
        dataset_id: str,
        *,
        require_formal_gold: bool = True,
        min_gold_ratio: float = 0.0,
    ) -> dict[str, Any]:
        path = self._resolve_dataset(dataset_id)
        from audio_engine.core.eval_ready import inspect_eval_manifest

        report = inspect_eval_manifest(
            path,
            require_formal_gold=require_formal_gold,
            min_gold_ratio=min_gold_ratio,
        )
        payload = report.as_dict() if hasattr(report, "as_dict") else {
            "path": str(getattr(report, "path", path)),
            "total": getattr(report, "total", None),
            "with_gold": list(getattr(report, "with_gold", [])),
            "without_gold": list(getattr(report, "without_gold", [])),
            "errors": list(getattr(report, "errors", [])),
            "warnings": list(getattr(report, "warnings", [])),
            "eval_trust": getattr(report, "eval_trust", None),
            "missing_audio": list(getattr(report, "missing_audio", [])),
            "pseudo_gold_ids": list(getattr(report, "pseudo_gold_ids", [])),
        }
        with_gold = len(payload.get("with_gold") or [])
        total = int(payload.get("total") or 0)
        payload["denominator"] = {
            "total_samples": total,
            "with_gold": with_gold,
            "without_gold": len(payload.get("without_gold") or []),
            "scored_subset_ratio": (with_gold / total) if total else 0.0,
            "failed_or_blocked": len(payload.get("errors") or []),
        }
        payload["formal_gate_ok"] = not bool(payload.get("errors")) and (
            not require_formal_gold or with_gold > 0
        )
        payload["dataset_id"] = self._rel_id(path)
        payload["note"] = (
            "缺合法正式 Gold 时不得生成虚构准确率；本检查不启动推理。"
        )
        return payload
