from __future__ import annotations

from fastapi import APIRouter, Header, Request, Response

from audio_engine.web.schemas import ModelCheckRequest, ModelCreate, ModelPatch
from audio_engine.web.services.models import ModelServiceError

router = APIRouter(prefix="/models", tags=["models"])


def _svc(request: Request):
    return request.app.state.model_service


@router.get("")
def list_models(request: Request):
    return _svc(request).list_models()


@router.post("")
def create_model(payload: ModelCreate, request: Request, response: Response):
    try:
        item = _svc(request).create(payload.model_dump())
        response.status_code = 200 if item.get("reused") else 201
        return item
    except ModelServiceError as exc:
        from audio_engine.web.app import raise_api_error

        raise_api_error(exc.code, exc.message, details=exc.details, status=409 if "CONFLICT" in exc.code else 400)


@router.get("/{connection_id}")
def get_model(connection_id: str, request: Request):
    try:
        return _svc(request).get(connection_id).public_dict()
    except ModelServiceError as exc:
        from audio_engine.web.app import raise_api_error

        raise_api_error(exc.code, exc.message, status=404)


@router.patch("/{connection_id}")
def patch_model(
    connection_id: str,
    payload: ModelPatch,
    request: Request,
    if_match: str | None = Header(default=None, alias="If-Match"),
):
    rev = int(if_match) if if_match and if_match.isdigit() else None
    try:
        return _svc(request).patch(
            connection_id, payload.model_dump(exclude_unset=True), expected_revision=rev
        )
    except ModelServiceError as exc:
        from audio_engine.web.app import raise_api_error

        status = 409 if "CONFLICT" in exc.code else 404 if "NOT_FOUND" in exc.code else 400
        raise_api_error(exc.code, exc.message, details=exc.details, status=status)


@router.post("/{connection_id}/check")
def check_model(connection_id: str, payload: ModelCheckRequest, request: Request):
    sample_path = None
    if payload.sample_ref:
        sample_path = payload.sample_ref.get("path") or payload.sample_ref.get("sample_id")
    try:
        return _svc(request).check(
            connection_id, level=payload.level, sample_path=sample_path
        )
    except ModelServiceError as exc:
        from audio_engine.web.app import raise_api_error

        raise_api_error(exc.code, exc.message, details=exc.details, status=400)
