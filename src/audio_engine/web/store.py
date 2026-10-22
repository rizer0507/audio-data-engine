"""SQLite persistence for plans, jobs, attempts, events, and idempotency."""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

SCHEMA_VERSION = 1


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime | None = None) -> str:
    value = dt or _utc_now()
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _process_identity_alive(pid: int | None, process_start_token: str) -> bool:
    """Return True only when PID is alive AND matches our start token format.

    Token format written by worker: ``{pid}:{time_ns}``. A recycled PID alone
    is not enough to claim ownership of an attempt.
    """

    if not pid or int(pid) <= 0:
        return False
    pid = int(pid)
    expected_prefix = f"{pid}:"
    if process_start_token and not str(process_start_token).startswith(expected_prefix):
        # Token mismatch => different process identity; treat as not ours.
        return False
    try:
        import os

        if os.name == "nt":
            import ctypes

            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            SYNCHRONIZE = 0x00100000
            handle = kernel32.OpenProcess(SYNCHRONIZE, False, pid)
            if handle:
                kernel32.CloseHandle(handle)
                return True
            return False
        os.kill(pid, 0)
        return True
    except Exception:
        return False


class JobStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA busy_timeout = 30000")
        return conn

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            conn = self._connect()
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def _init_db(self) -> None:
        with self.connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS meta (
                  key TEXT PRIMARY KEY,
                  value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS plans (
                  plan_id TEXT PRIMARY KEY,
                  plan_digest TEXT NOT NULL,
                  task_request_json TEXT NOT NULL,
                  plan_json TEXT NOT NULL,
                  catalog_generation INTEGER NOT NULL,
                  definition_digest TEXT,
                  snapshot_path TEXT,
                  expires_at TEXT NOT NULL,
                  job_id TEXT,
                  created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS jobs (
                  job_id TEXT PRIMARY KEY,
                  task_type TEXT NOT NULL,
                  status TEXT NOT NULL,
                  plan_id TEXT NOT NULL,
                  plan_digest TEXT NOT NULL,
                  request_json TEXT NOT NULL,
                  snapshot_path TEXT,
                  current_attempt INTEGER NOT NULL DEFAULT 0,
                  progress_json TEXT,
                  artifacts_json TEXT,
                  error_json TEXT,
                  idempotency_key TEXT,
                  idempotency_hash TEXT,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  UNIQUE(idempotency_key)
                );
                CREATE TABLE IF NOT EXISTS job_attempts (
                  attempt_id TEXT PRIMARY KEY,
                  job_id TEXT NOT NULL,
                  attempt_no INTEGER NOT NULL,
                  status TEXT NOT NULL,
                  worker_id TEXT,
                  lease_token TEXT,
                  lease_expires_at TEXT,
                  pid INTEGER,
                  process_start_token TEXT,
                  work_dir TEXT,
                  checkpoint_ref TEXT,
                  log_path TEXT,
                  error_json TEXT,
                  started_at TEXT,
                  ended_at TEXT,
                  UNIQUE(job_id, attempt_no),
                  FOREIGN KEY(job_id) REFERENCES jobs(job_id)
                );
                CREATE TABLE IF NOT EXISTS job_events (
                  event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                  job_id TEXT NOT NULL,
                  attempt_id TEXT,
                  ts TEXT NOT NULL,
                  kind TEXT NOT NULL,
                  details_json TEXT
                );
                CREATE TABLE IF NOT EXISTS publications (
                  job_id TEXT PRIMARY KEY,
                  attempt_id TEXT NOT NULL,
                  lease_token TEXT NOT NULL,
                  published_at TEXT NOT NULL,
                  outcome TEXT NOT NULL,
                  artifacts_json TEXT,
                  FOREIGN KEY(job_id) REFERENCES jobs(job_id)
                );
                CREATE TABLE IF NOT EXISTS mutation_idempotency (
                  scope TEXT NOT NULL,
                  idempotency_id TEXT NOT NULL,
                  request_hash TEXT NOT NULL,
                  response_json TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  PRIMARY KEY(scope, idempotency_id)
                );
                CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
                CREATE INDEX IF NOT EXISTS idx_attempts_job ON job_attempts(job_id);
                CREATE INDEX IF NOT EXISTS idx_events_job ON job_events(job_id, event_id);
                """
            )
            row = conn.execute(
                "SELECT value FROM meta WHERE key='schema_version'"
            ).fetchone()
            if row is None:
                conn.execute(
                    "INSERT INTO meta(key, value) VALUES ('schema_version', ?)",
                    (str(SCHEMA_VERSION),),
                )

    # --- plans ---

    def save_plan(
        self,
        *,
        plan_id: str,
        plan_digest: str,
        task_request: dict[str, Any],
        plan: dict[str, Any],
        catalog_generation: int,
        definition_digest: str | None,
        snapshot_path: str | None,
        ttl_seconds: int,
    ) -> dict[str, Any]:
        now = _utc_now()
        expires = now + timedelta(seconds=ttl_seconds)
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO plans(
                  plan_id, plan_digest, task_request_json, plan_json,
                  catalog_generation, definition_digest, snapshot_path,
                  expires_at, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    plan_id,
                    plan_digest,
                    json.dumps(task_request, ensure_ascii=False),
                    json.dumps(plan, ensure_ascii=False),
                    catalog_generation,
                    definition_digest,
                    snapshot_path,
                    _iso(expires),
                    _iso(now),
                ),
            )
        return {
            "plan_id": plan_id,
            "plan_digest": plan_digest,
            "expires_at": _iso(expires),
            "catalog_generation": catalog_generation,
        }

    def get_plan(self, plan_id: str) -> dict[str, Any] | None:
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM plans WHERE plan_id=?", (plan_id,)
            ).fetchone()
        if not row:
            return None
        return dict(row)

    # --- jobs ---

    def create_job_from_plan(
        self,
        *,
        plan_row: dict[str, Any],
        idempotency_key: str | None,
        idempotency_hash: str | None,
    ) -> tuple[str, bool]:
        """Return (job_id, created). Honors idempotency key."""

        with self.connection() as conn:
            if idempotency_key:
                existing = conn.execute(
                    "SELECT job_id, plan_digest, idempotency_hash FROM jobs WHERE idempotency_key=?",
                    (idempotency_key,),
                ).fetchone()
                if existing:
                    if existing["plan_digest"] != plan_row["plan_digest"] or (
                        idempotency_hash and existing["idempotency_hash"] != idempotency_hash
                    ):
                        raise ConflictError(
                            "IDEMPOTENCY_CONFLICT",
                            "same Idempotency-Key with different plan",
                        )
                    return existing["job_id"], False

            job_id = f"job_{uuid.uuid4().hex[:16]}"
            now = _iso()
            request = json.loads(plan_row["task_request_json"])
            conn.execute(
                """
                INSERT INTO jobs(
                  job_id, task_type, status, plan_id, plan_digest, request_json,
                  snapshot_path, current_attempt, idempotency_key, idempotency_hash,
                  created_at, updated_at
                ) VALUES (?, ?, 'queued', ?, ?, ?, ?, 0, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    request.get("task_type") or "pipeline",
                    plan_row["plan_id"],
                    plan_row["plan_digest"],
                    plan_row["task_request_json"],
                    plan_row.get("snapshot_path"),
                    idempotency_key,
                    idempotency_hash,
                    now,
                    now,
                ),
            )
            conn.execute(
                "UPDATE plans SET job_id=? WHERE plan_id=?",
                (job_id, plan_row["plan_id"]),
            )
            self._add_event(conn, job_id, None, "queued", {"plan_id": plan_row["plan_id"]})
            return job_id, True

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self.connection() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        return dict(row) if row else None

    def list_jobs(
        self, *, status: str | None = None, limit: int = 50, cursor: int = 0
    ) -> tuple[list[dict[str, Any]], int | None]:
        limit = max(1, min(limit, 200))
        with self.connection() as conn:
            if status:
                rows = conn.execute(
                    """
                    SELECT * FROM jobs WHERE status=?
                    ORDER BY created_at DESC LIMIT ? OFFSET ?
                    """,
                    (status, limit + 1, cursor),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM jobs ORDER BY created_at DESC LIMIT ? OFFSET ?
                    """,
                    (limit + 1, cursor),
                ).fetchall()
        items = [dict(r) for r in rows[:limit]]
        next_cursor = cursor + limit if len(rows) > limit else None
        return items, next_cursor

    def update_job_status(
        self,
        job_id: str,
        *,
        status: str,
        expected_statuses: set[str] | None = None,
        error: dict[str, Any] | None = None,
        progress: dict[str, Any] | None = None,
        artifacts: list[dict[str, Any]] | None = None,
    ) -> bool:
        with self.connection() as conn:
            row = conn.execute("SELECT status FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if not row:
                return False
            if expected_statuses and row["status"] not in expected_statuses:
                return False
            fields = ["status=?", "updated_at=?"]
            values: list[Any] = [status, _iso()]
            if error is not None:
                fields.append("error_json=?")
                values.append(json.dumps(error, ensure_ascii=False))
            if progress is not None:
                fields.append("progress_json=?")
                values.append(json.dumps(progress, ensure_ascii=False))
            if artifacts is not None:
                fields.append("artifacts_json=?")
                values.append(json.dumps(artifacts, ensure_ascii=False))
            values.append(job_id)
            conn.execute(f"UPDATE jobs SET {', '.join(fields)} WHERE job_id=?", values)
            self._add_event(conn, job_id, None, status, error or {})
            return True

    # --- attempts / lease ---

    def claim_next_job(self, *, worker_id: str, lease_seconds: int) -> dict[str, Any] | None:
        """Atomically claim one queued job. CAS on job status first, then insert attempt."""

        with self.connection() as conn:
            row = conn.execute(
                """
                SELECT * FROM jobs WHERE status='queued'
                ORDER BY created_at ASC LIMIT 1
                """
            ).fetchone()
            if not row:
                return None
            job = dict(row)
            attempt_no = int(job["current_attempt"]) + 1
            now = _utc_now()
            # CAS first so a lost race never leaves an orphan attempt row.
            cur = conn.execute(
                """
                UPDATE jobs SET status='running', current_attempt=?, updated_at=?
                WHERE job_id=? AND status='queued'
                """,
                (attempt_no, _iso(now), job["job_id"]),
            )
            if cur.rowcount == 0:
                return None
            attempt_id = f"att_{uuid.uuid4().hex[:16]}"
            lease_token = uuid.uuid4().hex
            lease_exp = _iso(now + timedelta(seconds=lease_seconds))
            conn.execute(
                """
                INSERT INTO job_attempts(
                  attempt_id, job_id, attempt_no, status, worker_id, lease_token,
                  lease_expires_at, started_at
                ) VALUES (?, ?, ?, 'running', ?, ?, ?, ?)
                """,
                (
                    attempt_id,
                    job["job_id"],
                    attempt_no,
                    worker_id,
                    lease_token,
                    lease_exp,
                    _iso(now),
                ),
            )
            self._add_event(
                conn,
                job["job_id"],
                attempt_id,
                "running",
                {"worker_id": worker_id, "attempt_no": attempt_no},
            )
            job["attempt_id"] = attempt_id
            job["attempt_no"] = attempt_no
            job["lease_token"] = lease_token
            job["work_dir"] = ""
            job["status"] = "running"
            job["current_attempt"] = attempt_no
            return job

    def heartbeat(
        self, attempt_id: str, *, lease_token: str, lease_seconds: int, pid: int | None = None
    ) -> bool:
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM job_attempts WHERE attempt_id=?", (attempt_id,)
            ).fetchone()
            if not row or row["lease_token"] != lease_token:
                return False
            if row["status"] not in {"running", "cancelling"}:
                return False
            exp = _iso(_utc_now() + timedelta(seconds=lease_seconds))
            conn.execute(
                """
                UPDATE job_attempts SET lease_expires_at=?, pid=COALESCE(?, pid)
                WHERE attempt_id=? AND lease_token=?
                """,
                (exp, pid, attempt_id, lease_token),
            )
            return True

    def set_attempt_process(
        self,
        attempt_id: str,
        *,
        lease_token: str,
        pid: int,
        process_start_token: str,
        work_dir: str,
        log_path: str,
    ) -> bool:
        with self.connection() as conn:
            cur = conn.execute(
                """
                UPDATE job_attempts
                SET pid=?, process_start_token=?, work_dir=?, log_path=?
                WHERE attempt_id=? AND lease_token=?
                """,
                (pid, process_start_token, work_dir, log_path, attempt_id, lease_token),
            )
            return cur.rowcount > 0

    def finish_attempt(
        self,
        attempt_id: str,
        *,
        lease_token: str,
        status: str,
        job_status: str,
        error: dict[str, Any] | None = None,
        artifacts: list[dict[str, Any]] | None = None,
        progress: dict[str, Any] | None = None,
    ) -> bool:
        """Finalize attempt only when lease still owns it. Rejects late finishers."""

        with self.connection() as conn:
            return self._finish_attempt_locked(
                conn,
                attempt_id,
                lease_token=lease_token,
                status=status,
                job_status=job_status,
                error=error,
                artifacts=artifacts,
                progress=progress,
                record_publication=job_status == "succeeded",
            )

    def try_publish(
        self,
        attempt_id: str,
        *,
        lease_token: str,
        artifacts: list[dict[str, Any]] | None = None,
        progress: dict[str, Any] | None = None,
        outcome: str = "succeeded",
        error: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Atomic publish + terminal state. Rejects lost-lease late publishers.

        Returns {"accepted": bool, "reason": str|None, "publication": ...}.
        """

        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM job_attempts WHERE attempt_id=?", (attempt_id,)
            ).fetchone()
            if not row:
                return {"accepted": False, "reason": "ATTEMPT_NOT_FOUND"}
            if row["lease_token"] != lease_token:
                return {"accepted": False, "reason": "LEASE_MISMATCH"}
            if row["status"] not in {"running", "cancelling"}:
                return {"accepted": False, "reason": "ATTEMPT_NOT_ACTIVE", "status": row["status"]}

            existing = conn.execute(
                "SELECT * FROM publications WHERE job_id=?", (row["job_id"],)
            ).fetchone()
            if existing:
                if existing["attempt_id"] == attempt_id:
                    # Idempotent re-publish of same attempt after crash between
                    # publish record and SQLite job update.
                    ok = self._finish_attempt_locked(
                        conn,
                        attempt_id,
                        lease_token=lease_token,
                        status=outcome if outcome in {"succeeded", "failed", "cancelled"} else "succeeded",
                        job_status=outcome if outcome in {"succeeded", "failed", "cancelled"} else "succeeded",
                        error=error,
                        artifacts=artifacts,
                        progress=progress,
                        record_publication=False,
                    )
                    return {
                        "accepted": ok,
                        "reason": None if ok else "FINISH_REJECTED",
                        "reconciled": True,
                        "publication": dict(existing),
                    }
                return {
                    "accepted": False,
                    "reason": "ALREADY_PUBLISHED_BY_OTHER_ATTEMPT",
                    "publication": dict(existing),
                }

            job_status = outcome if outcome in {"succeeded", "failed", "cancelled"} else "succeeded"
            attempt_status = job_status
            ok = self._finish_attempt_locked(
                conn,
                attempt_id,
                lease_token=lease_token,
                status=attempt_status,
                job_status=job_status,
                error=error,
                artifacts=artifacts,
                progress=progress,
                record_publication=True,
            )
            if not ok:
                return {"accepted": False, "reason": "FINISH_REJECTED"}
            pub = conn.execute(
                "SELECT * FROM publications WHERE job_id=?", (row["job_id"],)
            ).fetchone()
            return {
                "accepted": True,
                "reason": None,
                "publication": dict(pub) if pub else None,
            }

    def _finish_attempt_locked(
        self,
        conn: sqlite3.Connection,
        attempt_id: str,
        *,
        lease_token: str,
        status: str,
        job_status: str,
        error: dict[str, Any] | None,
        artifacts: list[dict[str, Any]] | None,
        progress: dict[str, Any] | None,
        record_publication: bool,
    ) -> bool:
        row = conn.execute(
            "SELECT * FROM job_attempts WHERE attempt_id=?", (attempt_id,)
        ).fetchone()
        if not row or row["lease_token"] != lease_token:
            return False
        if row["status"] not in {"running", "cancelling", "interrupted"}:
            # Allow interrupted reconcile to still finish once.
            if row["status"] in {"succeeded", "failed", "cancelled"}:
                return False
        cur = conn.execute(
            """
            UPDATE job_attempts SET status=?, ended_at=?, error_json=?
            WHERE attempt_id=? AND lease_token=? AND status IN ('running','cancelling','interrupted')
            """,
            (
                status,
                _iso(),
                json.dumps(error, ensure_ascii=False) if error else None,
                attempt_id,
                lease_token,
            ),
        )
        if cur.rowcount == 0:
            return False
        fields = ["status=?", "updated_at=?"]
        values: list[Any] = [job_status, _iso()]
        if error is not None:
            fields.append("error_json=?")
            values.append(json.dumps(error, ensure_ascii=False))
        if artifacts is not None:
            fields.append("artifacts_json=?")
            values.append(json.dumps(artifacts, ensure_ascii=False))
        if progress is not None:
            fields.append("progress_json=?")
            values.append(json.dumps(progress, ensure_ascii=False))
        values.append(row["job_id"])
        conn.execute(f"UPDATE jobs SET {', '.join(fields)} WHERE job_id=?", values)
        if record_publication and job_status == "succeeded":
            conn.execute(
                """
                INSERT OR IGNORE INTO publications(
                  job_id, attempt_id, lease_token, published_at, outcome, artifacts_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    row["job_id"],
                    attempt_id,
                    lease_token,
                    _iso(),
                    job_status,
                    json.dumps(artifacts or [], ensure_ascii=False),
                ),
            )
        self._add_event(conn, row["job_id"], attempt_id, job_status, error or {})
        return True

    def get_publication(self, job_id: str) -> dict[str, Any] | None:
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM publications WHERE job_id=?", (job_id,)
            ).fetchone()
        return dict(row) if row else None

    def list_active_attempts(self) -> list[dict[str, Any]]:
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT a.*, j.status AS job_status
                FROM job_attempts a
                JOIN jobs j ON j.job_id = a.job_id
                WHERE a.status IN ('running', 'cancelling')
                """
            ).fetchall()
        return [dict(r) for r in rows]

    def reconcile_stale_attempts(self) -> list[dict[str, Any]]:
        """Mark attempts interrupted when lease expired and process is gone.

        Never starts a new attempt solely because lease expired while the
        original process identity is still alive.
        """

        actions: list[dict[str, Any]] = []
        for att in self.list_active_attempts():
            exp = att.get("lease_expires_at")
            if not exp:
                continue
            expires = datetime.fromisoformat(str(exp).replace("Z", "+00:00"))
            if expires > _utc_now():
                continue
            pid = att.get("pid")
            token = att.get("process_start_token") or ""
            alive = _process_identity_alive(pid, token)
            if alive:
                actions.append(
                    {
                        "attempt_id": att["attempt_id"],
                        "action": "leave_running",
                        "reason": "lease_expired_but_process_alive",
                    }
                )
                continue
            self.mark_interrupted(
                att["attempt_id"],
                reason="lease_expired_and_process_gone",
            )
            actions.append(
                {
                    "attempt_id": att["attempt_id"],
                    "action": "interrupted",
                    "reason": "lease_expired_and_process_gone",
                }
            )
        return actions

    def reconcile_publications(self) -> list[dict[str, Any]]:
        """Align job terminal status with publication records after crash."""

        fixed: list[dict[str, Any]] = []
        with self.connection() as conn:
            rows = conn.execute("SELECT * FROM publications").fetchall()
            for pub in rows:
                job = conn.execute(
                    "SELECT * FROM jobs WHERE job_id=?", (pub["job_id"],)
                ).fetchone()
                if not job:
                    continue
                if job["status"] == "succeeded":
                    continue
                if job["status"] in {"running", "cancelling", "interrupted", "queued"}:
                    conn.execute(
                        """
                        UPDATE jobs SET status='succeeded', artifacts_json=?, updated_at=?
                        WHERE job_id=?
                        """,
                        (pub["artifacts_json"], _iso(), pub["job_id"]),
                    )
                    conn.execute(
                        """
                        UPDATE job_attempts SET status='succeeded', ended_at=COALESCE(ended_at, ?)
                        WHERE attempt_id=?
                        """,
                        (_iso(), pub["attempt_id"]),
                    )
                    self._add_event(
                        conn,
                        pub["job_id"],
                        pub["attempt_id"],
                        "succeeded",
                        {"reconciled_from": "publication"},
                    )
                    fixed.append({"job_id": pub["job_id"], "attempt_id": pub["attempt_id"]})
        return fixed

    def request_cancel(self, job_id: str) -> dict[str, Any]:
        with self.connection() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if not row:
                raise KeyError(job_id)
            status = row["status"]
            if status in {"succeeded", "failed", "cancelled"}:
                return {"job_id": job_id, "status": status, "changed": False}
            if status == "queued":
                conn.execute(
                    "UPDATE jobs SET status='cancelled', updated_at=? WHERE job_id=?",
                    (_iso(), job_id),
                )
                self._add_event(conn, job_id, None, "cancelled", {"from": "queued"})
                return {"job_id": job_id, "status": "cancelled", "changed": True}
            if status == "cancelling":
                return {"job_id": job_id, "status": "cancelling", "changed": False}
            conn.execute(
                "UPDATE jobs SET status='cancelling', updated_at=? WHERE job_id=?",
                (_iso(), job_id),
            )
            att = conn.execute(
                """
                SELECT attempt_id FROM job_attempts
                WHERE job_id=? AND status='running'
                ORDER BY attempt_no DESC LIMIT 1
                """,
                (job_id,),
            ).fetchone()
            if att:
                conn.execute(
                    "UPDATE job_attempts SET status='cancelling' WHERE attempt_id=?",
                    (att["attempt_id"],),
                )
            self._add_event(conn, job_id, att["attempt_id"] if att else None, "cancelling", {})
            return {"job_id": job_id, "status": "cancelling", "changed": True}

    def get_attempt(self, attempt_id: str) -> dict[str, Any] | None:
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM job_attempts WHERE attempt_id=?", (attempt_id,)
            ).fetchone()
        return dict(row) if row else None

    def list_attempts(self, job_id: str) -> list[dict[str, Any]]:
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM job_attempts WHERE job_id=? ORDER BY attempt_no",
                (job_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def list_events(self, job_id: str, *, after_id: int = 0, limit: int = 100) -> list[dict[str, Any]]:
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM job_events WHERE job_id=? AND event_id>?
                ORDER BY event_id ASC LIMIT ?
                """,
                (job_id, after_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def mark_interrupted(self, attempt_id: str, *, reason: str) -> None:
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM job_attempts WHERE attempt_id=?", (attempt_id,)
            ).fetchone()
            if not row:
                return
            conn.execute(
                "UPDATE job_attempts SET status='interrupted', ended_at=?, error_json=? WHERE attempt_id=?",
                (_iso(), json.dumps({"reason": reason}), attempt_id),
            )
            conn.execute(
                "UPDATE jobs SET status='interrupted', updated_at=? WHERE job_id=?",
                (_iso(), row["job_id"]),
            )
            self._add_event(conn, row["job_id"], attempt_id, "interrupted", {"reason": reason})

    def _add_event(
        self,
        conn: sqlite3.Connection,
        job_id: str,
        attempt_id: str | None,
        kind: str,
        details: dict[str, Any],
    ) -> None:
        conn.execute(
            """
            INSERT INTO job_events(job_id, attempt_id, ts, kind, details_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                job_id,
                attempt_id,
                _iso(),
                kind,
                json.dumps(details, ensure_ascii=False)[:8000],
            ),
        )


    def get_mutation_idempotency(
        self, *, scope: str, idempotency_id: str
    ) -> dict[str, Any] | None:
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM mutation_idempotency WHERE scope=? AND idempotency_id=?",
                (scope, idempotency_id),
            ).fetchone()
            if not row:
                return None
            return dict(row)

    def put_mutation_idempotency(
        self,
        *,
        scope: str,
        idempotency_id: str,
        request_hash: str,
        response: dict[str, Any],
    ) -> dict[str, Any]:
        """Store mutation response. Raises ConflictError on same id different hash."""

        with self.connection() as conn:
            existing = conn.execute(
                "SELECT * FROM mutation_idempotency WHERE scope=? AND idempotency_id=?",
                (scope, idempotency_id),
            ).fetchone()
            if existing:
                if existing["request_hash"] != request_hash:
                    raise ConflictError(
                        "IDEMPOTENCY_CONFLICT",
                        "idempotency_id reused with different request",
                        {
                            "scope": scope,
                            "idempotency_id": idempotency_id,
                        },
                    )
                return json.loads(existing["response_json"])
            payload = json.dumps(response, ensure_ascii=False)
            conn.execute(
                """
                INSERT INTO mutation_idempotency(
                  scope, idempotency_id, request_hash, response_json, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (scope, idempotency_id, request_hash, payload, _iso()),
            )
            return response


class ConflictError(Exception):
    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}
