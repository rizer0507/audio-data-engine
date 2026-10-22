from __future__ import annotations

from fastapi import APIRouter, Header, Request, Response

from audio_engine.web.schemas import SourceCreate, SourcePatch
from audio_engine.web.services.sources import SourceRegistryError

router = APIRouter(prefix="/sources", tags=["sources"])


def _svc(request: Request):
    return request.app.state.source_service


@router.get("")
def list_sources(request: Request):
    return _svc(request).list_sources()


@router.post("")
def create_source(payload: SourceCreate, request: Request, response: Response):
    try:
        item = _svc(request).create(payload.model_dump())
        response.status_code = 200 if item.get("reused") else 201
        return item
    except SourceRegistryError as exc:
        from audio_engine.web.app import raise_api_error

        raise_api_error(exc.code, exc.message, details=exc.details, status=409 if "CONFLICT" in exc.code else 400)


@router.get("/{source_id}")
def get_source(source_id: str, request: Request):
    try:
        return _svc(request).get(source_id)
    except SourceRegistryError as exc:
        from audio_engine.web.app import raise_api_error

        raise_api_error(exc.code, exc.message, status=404)


@router.patch("/{source_id}")
def patch_source(
    source_id: str,
    payload: SourcePatch,
    request: Request,
    if_match: str | None = Header(default=None, alias="If-Match"),
):
    rev = int(if_match) if if_match and if_match.isdigit() else None
    try:
        return _svc(request).patch(
            source_id, payload.model_dump(exclude_unset=True), expected_revision=rev
        )
    except SourceRegistryError as exc:
        from audio_engine.web.app import raise_api_error

        status = 409 if "CONFLICT" in exc.code else 404 if "NOT_FOUND" in exc.code else 400
        raise_api_error(exc.code, exc.message, details=exc.details, status=status)


@router.post("/{source_id}/check")
def check_source(source_id: str, request: Request):
    try:
        return _svc(request).check(source_id)
    except SourceRegistryError as exc:
        from audio_engine.web.app import raise_api_error

        raise_api_error(exc.code, exc.message, status=404)
