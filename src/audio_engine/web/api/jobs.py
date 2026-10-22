from __future__ import annotations

from fastapi import APIRouter, Header, Request

from audio_engine.web.schemas import JobRetryRequest, JobSubmitRequest
from audio_engine.web.store import ConflictError

router = APIRouter(prefix="/jobs", tags=["jobs"])


def _jobs(request: Request):
    return request.app.state.job_service


@router.get("")
def list_jobs(
    request: Request,
    status: str | None = None,
    cursor: int = 0,
    limit: int = 50,
):
    return _jobs(request).list(status=status, cursor=cursor, limit=limit)


@router.post("", status_code=202)
def submit_job(
    payload: JobSubmitRequest,
    request: Request,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
):
    try:
        job, created = _jobs(request).submit(
            plan_id=payload.plan_id,
            plan_digest=payload.plan_digest,
            idempotency_key=idempotency_key,
        )
        return {**job, "created": created}
    except KeyError:
        from audio_engine.web.app import raise_api_error

        raise_api_error("PLAN_NOT_FOUND", "plan_id not found", status=404)
    except ConflictError as exc:
        from audio_engine.web.app import raise_api_error

        raise_api_error(exc.code, exc.message, details=exc.details, status=409)


@router.get("/{job_id}")
def get_job(job_id: str, request: Request):
    try:
        return _jobs(request).get(job_id)
    except KeyError:
        from audio_engine.web.app import raise_api_error

        raise_api_error("JOB_NOT_FOUND", job_id, status=404)


@router.get("/{job_id}/logs")
def job_logs(
    job_id: str,
    request: Request,
    cursor: str | None = None,
    limit_bytes: int | None = None,
    node_id: str | None = None,
):
    try:
        return _jobs(request).logs(
            job_id, cursor=cursor, limit_bytes=limit_bytes, node_id=node_id
        )
    except KeyError:
        from audio_engine.web.app import raise_api_error

        raise_api_error("JOB_NOT_FOUND", job_id, status=404)


@router.post("/{job_id}/cancel")
def cancel_job(job_id: str, request: Request):
    try:
        return _jobs(request).cancel(job_id)
    except KeyError:
        from audio_engine.web.app import raise_api_error

        raise_api_error("JOB_NOT_FOUND", job_id, status=404)


@router.post("/{job_id}/retry")
def retry_job(job_id: str, payload: JobRetryRequest, request: Request):
    try:
        return _jobs(request).retry(job_id, mode=payload.mode)
    except KeyError:
        from audio_engine.web.app import raise_api_error

        raise_api_error("JOB_NOT_FOUND", job_id, status=404)
    except ConflictError as exc:
        from audio_engine.web.app import raise_api_error

        raise_api_error(exc.code, exc.message, details=exc.details, status=409)
    except ValueError as exc:
        from audio_engine.web.app import raise_api_error

        raise_api_error("RETRY_INVALID", str(exc), status=400)


@router.get("/{job_id}/snapshot")
def job_snapshot(job_id: str, request: Request):
    try:
        return _jobs(request).snapshot(job_id)
    except KeyError:
        from audio_engine.web.app import raise_api_error

        raise_api_error("JOB_NOT_FOUND", job_id, status=404)
