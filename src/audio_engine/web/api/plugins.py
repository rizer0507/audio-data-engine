from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Header, Request

from audio_engine.plugins.registry import PluginDefinitionError, PluginStatusError
from audio_engine.web.schemas import PluginDefinitionDraft, PluginStatusPatch

router = APIRouter(prefix="/plugins", tags=["plugins"])


def _engine(request: Request):
    return request.app.state.engine_service


def _jobs(request: Request):
    return request.app.state.job_service


@router.get("")
def list_plugins(
    request: Request,
    kind: str | None = None,
    purpose: str | None = None,
    status: str | None = None,
):
    return _engine(request).list_plugins(kind=kind, purpose=purpose, status=status)


@router.get("/{kind}/{plugin_id}")
def get_plugin(kind: str, plugin_id: str, request: Request, version: str | None = None):
    try:
        return _engine(request).get_plugin(kind, plugin_id, version)
    except KeyError:
        from audio_engine.web.app import raise_api_error

        raise_api_error("PLUGIN_NOT_FOUND", f"{kind}/{plugin_id}", status=404)


@router.patch("/{kind}/{plugin_id}/status")
def patch_plugin_status(
    kind: str,
    plugin_id: str,
    payload: PluginStatusPatch,
    request: Request,
    if_match: str | None = Header(default=None, alias="If-Match"),
):
    from audio_engine.web.app import raise_api_error

    rev = payload.expected_revision
    if rev is None and if_match and if_match.isdigit():
        rev = int(if_match)
    try:
        return _engine(request).set_plugin_status(
            kind, plugin_id, payload.status, expected_revision=rev
        )
    except PluginStatusError as exc:
        status_code = 409 if "CONFLICT" in exc.code else 404 if "NOT_FOUND" in exc.code else 400
        raise_api_error(exc.code, exc.message, details=exc.details, status=status_code)
    except KeyError:
        raise_api_error("PLUGIN_NOT_FOUND", f"{kind}/{plugin_id}", status=404)


@router.post("/{kind}/{plugin_id}/definition/validate")
def validate_plugin_definition(
    kind: str,
    plugin_id: str,
    payload: PluginDefinitionDraft,
    request: Request,
    version: str | None = None,
):
    from audio_engine.web.app import raise_api_error

    try:
        return _engine(request).validate_plugin_definition(
            kind,
            plugin_id,
            payload.draft,
            expected_revision=payload.expected_revision,
            expected_definition_digest=payload.expected_definition_digest,
            version=version,
        )
    except PluginDefinitionError as exc:
        status_code = 409 if "CONFLICT" in exc.code else 404 if "NOT_FOUND" in exc.code else 400
        raise_api_error(exc.code, exc.message, details=exc.details, status=status_code)
    except KeyError:
        raise_api_error("PLUGIN_NOT_FOUND", f"{kind}/{plugin_id}", status=404)


@router.patch("/{kind}/{plugin_id}/definition")
def patch_plugin_definition(
    kind: str,
    plugin_id: str,
    payload: PluginDefinitionDraft,
    request: Request,
    version: str | None = None,
    if_match: str | None = Header(default=None, alias="If-Match"),
):
    from audio_engine.web.app import raise_api_error

    rev = payload.expected_revision
    if rev is None and if_match and if_match.isdigit():
        rev = int(if_match)
    try:
        return _engine(request).save_plugin_definition(
            kind,
            plugin_id,
            payload.draft,
            expected_revision=rev,
            expected_definition_digest=payload.expected_definition_digest,
            version=version,
        )
    except PluginDefinitionError as exc:
        status_code = 409 if "CONFLICT" in exc.code else 404 if "NOT_FOUND" in exc.code else 400
        raise_api_error(exc.code, exc.message, details=exc.details, status=status_code)
    except KeyError:
        raise_api_error("PLUGIN_NOT_FOUND", f"{kind}/{plugin_id}", status=404)


@router.post("/refresh")
def refresh_plugins(request: Request):
    try:
        return _engine(request).refresh_plugins()
    except RuntimeError as exc:
        from audio_engine.web.app import raise_api_error

        raise_api_error("PLUGIN_REFRESH_FAILED", str(exc), status=400)


@router.post("/validate")
def validate_plugins(payload: dict[str, Any], request: Request):
    return _engine(request).validate(payload)


@router.post("/plan")
def plan_plugins(payload: dict[str, Any], request: Request):
    try:
        return _jobs(request).create_plan(payload)
    except ValueError as exc:
        from audio_engine.web.app import raise_api_error

        raise_api_error("PLAN_FAILED", str(exc), status=400)
