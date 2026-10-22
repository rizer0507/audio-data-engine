"""Job submission, status, cancel, and retry orchestration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from audio_engine.web.services.engine import EngineService
from audio_engine.web.settings import WebSettings
from audio_engine.web.store import ConflictError, JobStore

# catalog typing is structural (PluginCatalog)


class JobService:
    def __init__(self, settings: WebSettings, store: JobStore, engine: EngineService) -> None:
        self.settings = settings
        self.store = store
        self.engine = engine

    def create_plan(self, request_data: dict[str, Any]) -> dict[str, Any]:
        store_dir = self.settings.runs_dir
        store_dir.mkdir(parents=True, exist_ok=True)
        plan = self.engine.plan(request_data, store_dir=store_dir)
        meta = self.store.save_plan(
            plan_id=plan["plan_id"],
            plan_digest=plan["plan_digest"],
            task_request=plan.get("task_request") or request_data,
            plan=plan,
            catalog_generation=int(plan.get("catalog_generation") or 0),
            definition_digest=plan.get("snapshot_digest"),
            snapshot_path=plan.get("snapshot_path"),
            ttl_seconds=self.settings.plan_ttl_seconds,
        )
        return {**plan, **meta}

    def submit(
        self,
        *,
        plan_id: str,
        plan_digest: str,
        idempotency_key: str | None,
    ) -> tuple[dict[str, Any], bool]:
        row = self.store.get_plan(plan_id)
        if not row:
            raise KeyError("plan not found")
        from datetime import datetime, timezone

        expires = datetime.fromisoformat(row["expires_at"].replace("Z", "+00:00"))
        if expires < datetime.now(timezone.utc):
            raise ConflictError("PLAN_EXPIRED", "plan expired; re-plan required")
        if row["plan_digest"] != plan_digest:
            raise ConflictError("PLAN_DIGEST_MISMATCH", "plan_digest does not match stored plan")

        # Catalog drift check (status archive/enable refreshes generation)
        catalog = self.engine.ensure_catalog()
        if int(row["catalog_generation"]) != catalog.generation:
            raise ConflictError(
                "CATALOG_DRIFT",
                "plugin catalog generation changed; re-plan required",
                {
                    "plan_generation": row["catalog_generation"],
                    "current_generation": catalog.generation,
                },
            )

        # Explicit archived / non-enabled guard for clearer errors on submit/rerun.
        task_req_raw = row.get("task_request_json") or "{}"
        task_req = json.loads(task_req_raw) if isinstance(task_req_raw, str) else (task_req_raw or {})
        blocked = self._archived_plugin_refs(task_req, catalog)
        if blocked:
            raise ConflictError(
                "PLUGIN_ARCHIVED",
                "selected plugin is archived; enable it before new submit/rerun",
                {"plugins": blocked},
            )

        idem_hash = None
        if idempotency_key:
            idem_hash = hashlib.sha256(
                f"{plan_id}:{plan_digest}".encode("utf-8")
            ).hexdigest()
        try:
            job_id, created = self.store.create_job_from_plan(
                plan_row=row,
                idempotency_key=idempotency_key,
                idempotency_hash=idem_hash,
            )
        except ConflictError:
            raise
        job = self.store.get_job(job_id)
        assert job is not None
        return self._job_view(job), created

    def get(self, job_id: str) -> dict[str, Any]:
        job = self.store.get_job(job_id)
        if not job:
            raise KeyError(job_id)
        return self._job_view(job)

    def list(self, *, status: str | None = None, cursor: int = 0, limit: int = 50) -> dict[str, Any]:
        items, next_cursor = self.store.list_jobs(status=status, cursor=cursor, limit=limit)
        return {
            "items": [self._job_view(j) for j in items],
            "next_cursor": next_cursor,
            "total": None,
        }

    def cancel(self, job_id: str) -> dict[str, Any]:
        return self.store.request_cancel(job_id)

    def logs(
        self,
        job_id: str,
        *,
        cursor: str | None = None,
        limit_bytes: int | None = None,
        node_id: str | None = None,
    ) -> dict[str, Any]:
        _ = node_id  # node-scoped logs reserved; first version reads attempt log file
        job = self.store.get_job(job_id)
        if not job:
            raise KeyError(job_id)
        attempts = self.store.list_attempts(job_id)
        if not attempts:
            return {"chunks": [], "next_cursor": cursor or "", "attempt_id": None}
        attempt = attempts[-1]
        log_path = attempt.get("log_path")
        max_bytes = limit_bytes or self.settings.worker.log_chunk_max_bytes
        offset = 0
        attempt_id = attempt["attempt_id"]
        if cursor:
            # format: attempt_id:offset
            parts = cursor.split(":")
            if len(parts) == 2 and parts[0]:
                attempt_id, offset_s = parts
                offset = int(offset_s)
                matched = next((a for a in attempts if a["attempt_id"] == attempt_id), None)
                if matched:
                    attempt = matched
                    log_path = matched.get("log_path")
        if not log_path or not Path(log_path).is_file():
            return {
                "chunks": [],
                "next_cursor": f"{attempt_id}:{offset}",
                "attempt_id": attempt_id,
            }
        data = Path(log_path).read_bytes()
        chunk = data[offset : offset + max_bytes]
        # UTF-8 boundary trim
        text = chunk.decode("utf-8", errors="replace")
        new_offset = offset + len(chunk)
        return {
            "chunks": [{"text": text, "offset": offset}],
            "next_cursor": f"{attempt_id}:{new_offset}",
            "attempt_id": attempt_id,
            "eof": new_offset >= len(data),
        }

    def snapshot(self, job_id: str) -> dict[str, Any]:
        job = self.store.get_job(job_id)
        if not job:
            raise KeyError(job_id)
        plan = self.store.get_plan(job["plan_id"])
        snap_path = job.get("snapshot_path") or (plan or {}).get("snapshot_path")
        payload: dict[str, Any] = {
            "job_id": job_id,
            "plan_id": job["plan_id"],
            "plan_digest": job["plan_digest"],
            "snapshot_path": snap_path,
        }
        if snap_path and Path(snap_path).is_dir():
            digest_file = Path(snap_path) / "definition_snapshot.digest"
            plan_file = Path(snap_path) / "plan.json"
            if digest_file.is_file():
                payload["definition_digest"] = digest_file.read_text(encoding="utf-8").strip()
            if plan_file.is_file():
                plan_json = json.loads(plan_file.read_text(encoding="utf-8"))
                # Strip any accidental secrets
                payload["plan_summary"] = {
                    "plugin": plan_json.get("plugin"),
                    "selected_tags": plan_json.get("selected_tags"),
                    "resolved_models": plan_json.get("resolved_models"),
                    "nodes": plan_json.get("nodes"),
                    "supports_cancel": plan_json.get("supports_cancel"),
                    "supports_resume": plan_json.get("supports_resume"),
                }
        return payload

    def retry(self, job_id: str, *, mode: str) -> dict[str, Any]:
        job = self.store.get_job(job_id)
        if not job:
            raise KeyError(job_id)
        if mode == "rerun":
            # New job from same plan if still present
            view, _ = self.submit(
                plan_id=job["plan_id"],
                plan_digest=job["plan_digest"],
                idempotency_key=None,
            )
            return {"mode": "rerun", "job": view}
        if mode != "resume":
            raise ValueError("mode must be resume|rerun")
        if job["status"] not in {"failed", "cancelled", "interrupted"}:
            raise ConflictError(
                "RESUME_NOT_ALLOWED",
                f"cannot resume job in status {job['status']}",
            )
        # Capability check from frozen plan
        plan_row = self.store.get_plan(job["plan_id"])
        plan: dict[str, Any] = {}
        if plan_row:
            plan = json.loads(plan_row["plan_json"])
            if not plan.get("supports_resume"):
                raise ConflictError(
                    "RESUME_UNSUPPORTED",
                    "frozen plan reports supports_resume=false",
                )
        # Implementation drift: refuse silent resume when frozen digest diverges.
        snap_path = job.get("snapshot_path") or (plan_row or {}).get("snapshot_path")
        if snap_path:
            digest_file = Path(snap_path) / "definition_snapshot.digest"
            impl_file = Path(snap_path) / "implementation_digest.txt"
            if digest_file.is_file():
                frozen_def = digest_file.read_text(encoding="utf-8").strip()
            else:
                frozen_def = plan.get("snapshot_digest") or (
                    plan_row.get("definition_digest") if plan_row else None
                )
            # Marker file written by tests / operators to force incompatibility.
            block = Path(snap_path) / "RESUME_INCOMPATIBLE"
            if block.is_file():
                raise ConflictError(
                    "RESUME_INCOMPATIBLE",
                    "snapshot marked incompatible; create a new job via re-plan",
                    {"snapshot_path": str(snap_path)},
                )
            if frozen_def and plan_row and plan_row.get("definition_digest"):
                if str(frozen_def) != str(plan_row["definition_digest"]):
                    raise ConflictError(
                        "RESUME_INCOMPATIBLE",
                        "frozen definition digest does not match plan record",
                        {
                            "frozen": frozen_def,
                            "plan": plan_row["definition_digest"],
                        },
                    )
            _ = impl_file  # reserved: compare live impl digests when adapters emit them
        # Re-queue same job with new attempt by setting queued
        ok = self.store.update_job_status(
            job_id, status="queued", expected_statuses={"failed", "cancelled", "interrupted"}
        )
        if not ok:
            raise ConflictError("RESUME_RACE", "job status changed")
        return {"mode": "resume", "job": self.get(job_id)}

    def _archived_plugin_refs(self, task_req: dict[str, Any], catalog: Any) -> list[dict[str, Any]]:
        blocked: list[dict[str, Any]] = []
        root = task_req.get("root_plugin")
        refs: list[dict[str, Any]] = []
        if isinstance(root, dict) and root.get("id"):
            refs.append(root)
        for tag in task_req.get("tags") or []:
            if isinstance(tag, dict) and tag.get("id"):
                refs.append(tag)
        for ref in refs:
            kind = str(ref.get("kind") or "pipeline")
            plugin_id = str(ref.get("id") or "")
            version = ref.get("version")
            try:
                entry = catalog.get(kind, plugin_id, version)
            except KeyError:
                blocked.append({**ref, "reason": "not_found"})
                continue
            if entry.descriptor.status != "enabled":
                blocked.append(
                    {
                        "kind": kind,
                        "id": plugin_id,
                        "version": entry.descriptor.version,
                        "status": entry.descriptor.status,
                    }
                )
        return blocked

    def _job_view(self, job: dict[str, Any]) -> dict[str, Any]:
        attempts = self.store.list_attempts(job["job_id"])
        plan_row = self.store.get_plan(job["plan_id"])
        plan_json: dict[str, Any] = {}
        if plan_row and plan_row.get("plan_json"):
            try:
                plan_json = json.loads(plan_row["plan_json"])
            except json.JSONDecodeError:
                plan_json = {}
        return {
            "job_id": job["job_id"],
            "task_type": job["task_type"],
            "status": job["status"],
            "plan_id": job["plan_id"],
            "plan_digest": job["plan_digest"],
            "current_attempt": job["current_attempt"],
            "request": json.loads(job["request_json"]),
            "progress": json.loads(job["progress_json"]) if job.get("progress_json") else None,
            "artifacts": json.loads(job["artifacts_json"]) if job.get("artifacts_json") else [],
            "error": json.loads(job["error_json"]) if job.get("error_json") else None,
            "snapshot_path": job.get("snapshot_path"),
            "created_at": job["created_at"],
            "updated_at": job["updated_at"],
            "capabilities": {
                "supports_cancel": plan_json.get("supports_cancel"),
                "supports_resume": plan_json.get("supports_resume"),
            },
            "attempts": [
                {
                    "attempt_id": a["attempt_id"],
                    "attempt_no": a["attempt_no"],
                    "status": a["status"],
                    "worker_id": a.get("worker_id"),
                    "started_at": a.get("started_at"),
                    "ended_at": a.get("ended_at"),
                    "heartbeat_at": a.get("lease_expires_at"),
                    "error": json.loads(a["error_json"]) if a.get("error_json") else None,
                    "work_dir": a.get("work_dir"),
                }
                for a in attempts
            ],
        }
