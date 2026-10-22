from __future__ import annotations



from typing import Any



from fastapi import APIRouter, Query, Request, Response

from fastapi.responses import FileResponse



from audio_engine.web.schemas import ExportRequest, ReviewRequest, TagActionRequest

from audio_engine.web.store import ConflictError



router = APIRouter(prefix="/datasets", tags=["datasets"])





def _results(request: Request):

    return request.app.state.results_service





def _raise(code: str, message: str, *, status: int = 400, details: dict | None = None):

    from audio_engine.web.app import raise_api_error



    raise_api_error(code, message, status=status, details=details)





@router.get("")

def list_datasets(request: Request):

    return _results(request).list_datasets()





@router.post("/{dataset_id:path}/exports")

def create_export(dataset_id: str, payload: ExportRequest, request: Request):

    """Create export job via plugin plan+submit facade; filters frozen into plan."""



    try:

        body = _results(request).export_via_plugin_request(

            dataset_id,

            export_plugin_id=payload.export_plugin_id,

            filters=payload.filters or {},

            parameters=payload.parameters or {},

            plugin_version=payload.plugin_version,

        )

    except FileNotFoundError:

        _raise("DATASET_NOT_FOUND", dataset_id, status=404)

    except PermissionError as exc:

        _raise("DATASET_FORBIDDEN", str(exc), status=403)



    frozen = body.pop("frozen_filters", payload.filters)

    plan = request.app.state.job_service.create_plan(body)

    job, created = request.app.state.job_service.submit(

        plan_id=plan["plan_id"],

        plan_digest=plan["plan_digest"],

        idempotency_key=payload.idempotency_key,

    )

    return {

        **job,

        "created": created,

        "frozen_filters": frozen,

        "frozen_dataset_id": (body.get("parameters") or {}).get("frozen_dataset_id"),

    }





@router.post("/{dataset_id:path}/gold-gate")

def gold_gate(dataset_id: str, payload: dict[str, Any] | None, request: Request):

    payload = payload or {}

    try:

        return _results(request).inspect_gold_gate(

            dataset_id,

            require_formal_gold=bool(payload.get("require_formal_gold", True)),

            min_gold_ratio=float(payload.get("min_gold_ratio") or 0.0),

        )

    except FileNotFoundError:

        _raise("DATASET_NOT_FOUND", dataset_id, status=404)

    except PermissionError as exc:

        _raise("DATASET_FORBIDDEN", str(exc), status=403)





@router.get("/{dataset_id:path}/samples/{sample_id}/audio")

def sample_audio(

    dataset_id: str,

    sample_id: str,

    request: Request,

    response: Response,

    asset: str = "raw",

):

    try:

        path, ctype, meta = _results(request).resolve_audio(dataset_id, sample_id, asset=asset)

    except FileNotFoundError as exc:

        _raise("AUDIO_NOT_FOUND", str(exc), status=404)

    except PermissionError as exc:

        _raise("AUDIO_FORBIDDEN", str(exc), status=403)

    # Starlette FileResponse honors HTTP Range → 206/416 for browser seeking.

    headers = {"X-Audio-Asset": asset, "Accept-Ranges": "bytes"}

    if meta.get("path"):

        headers["X-Resolved-Name"] = path.name

    return FileResponse(

        path,

        media_type=ctype,

        filename=path.name,

        headers=headers,

    )





@router.get("/{dataset_id:path}/samples/{sample_id}/explain")

def explain_sample(

    dataset_id: str,

    sample_id: str,

    request: Request,

    tagging_config: str | None = None,

):

    try:

        return _results(request).explain_sample(

            dataset_id, sample_id, tagging_config=tagging_config

        )

    except FileNotFoundError:

        _raise("SAMPLE_NOT_FOUND", sample_id, status=404)

    except PermissionError as exc:

        _raise("DATASET_FORBIDDEN", str(exc), status=403)





@router.post("/{dataset_id:path}/samples/{sample_id}/tag-actions")

def tag_actions(

    dataset_id: str, sample_id: str, payload: TagActionRequest, request: Request

):

    settings = request.app.state.settings

    operator = payload.operator_id or settings.operator_id

    try:

        return _results(request).apply_tag_action(

            dataset_id,

            sample_id,

            action=payload.action,

            tag_id=payload.tag_id,

            operator_id=operator,

            reason=payload.reason,

            expected_revision=payload.expected_revision,

            idempotency_id=payload.idempotency_id,

            work_root=payload.work_root,

            batch_id=payload.batch_id,

            tagging_config=payload.tagging_config,

            audio_sha256=payload.audio_sha256,

        )

    except ConflictError as exc:

        _raise(exc.code, exc.message, details=exc.details, status=409)

    except FileNotFoundError:

        _raise("SAMPLE_NOT_FOUND", sample_id, status=404)

    except PermissionError as exc:

        _raise("DATASET_FORBIDDEN", str(exc), status=403)

    except ValueError as exc:

        _raise("TAG_ACTION_INVALID", str(exc), status=400)





@router.post("/{dataset_id:path}/samples/{sample_id}/review")

def review_sample(dataset_id: str, sample_id: str, payload: ReviewRequest, request: Request):

    settings = request.app.state.settings

    operator = payload.operator_id or settings.operator_id

    extra = payload.model_dump(exclude={

        "operator_id",

        "reason",

        "expected_revision",

        "idempotency_id",

        "review_pass",

        "queue_id",

        "decision",

        "gold_kind",

        "gold_text",

        "speech_scope",

        "human_semantic",

        "human_noise",

        "human_crosstalk",

        "verified_error_tags",

        "audio_event_tags",

        "annotation_config",

    })

    try:

        result = _results(request).apply_review(

            dataset_id,

            sample_id,

            operator_id=operator,

            reason=payload.reason,

            expected_revision=payload.expected_revision,

            idempotency_id=payload.idempotency_id,

            review_pass=payload.review_pass,

            queue_id=payload.queue_id,

            decision=payload.decision,

            gold_kind=payload.gold_kind,

            gold_text=payload.gold_text,

            speech_scope=payload.speech_scope,

            human_semantic=payload.human_semantic,

            human_noise=payload.human_noise,

            human_crosstalk=payload.human_crosstalk,

            verified_error_tags=payload.verified_error_tags,

            audio_event_tags=payload.audio_event_tags,

            annotation_config=payload.annotation_config,

            extra=extra or None,

        )

        if result.get("ok") is False and result.get("code") == "REVIEW_REJECTED":

            # Contract / Gold gate rejection — 409 so UI can refresh revision

            _raise(

                "REVIEW_REJECTED",

                result.get("message") or "review rejected",

                status=409,

                details=result,

            )

        return result

    except ConflictError as exc:

        _raise(exc.code, exc.message, details=exc.details, status=409)

    except FileNotFoundError:

        _raise("SAMPLE_NOT_FOUND", sample_id, status=404)

    except PermissionError as exc:

        _raise("DATASET_FORBIDDEN", str(exc), status=403)

    except ValueError as exc:

        _raise("REVIEW_INVALID", str(exc), status=400)





@router.get("/{dataset_id:path}/samples/{sample_id}")

def get_sample(dataset_id: str, sample_id: str, request: Request):

    try:

        return _results(request).get_sample(dataset_id, sample_id)

    except FileNotFoundError:

        _raise("SAMPLE_NOT_FOUND", sample_id, status=404)

    except PermissionError as exc:

        _raise("DATASET_FORBIDDEN", str(exc), status=403)





@router.get("/{dataset_id:path}/samples")

def list_samples(

    dataset_id: str,

    request: Request,

    limit: int = 50,

    offset: int = 0,

    tag: str | None = None,

    tag_exclude: str | None = None,

    tag_any: str | None = None,

    tag_all: str | None = None,

    ternary: str | None = None,

    tag_ternary: str | None = None,

    route_status: str | None = None,

    review_status: str | None = None,

    quality_key: str | None = None,

    quality_min: float | None = Query(default=None),

    quality_max: float | None = Query(default=None),

):

    try:

        return _results(request).list_samples(

            dataset_id,

            limit=limit,

            offset=offset,

            filters={

                "tag": tag,

                "tag_exclude": tag_exclude,

                "tag_any": tag_any,

                "tag_all": tag_all,

                "ternary": ternary,

                "tag_ternary": tag_ternary,

                "route_status": route_status,

                "review_status": review_status,

                "quality_key": quality_key,

                "quality_min": quality_min,

                "quality_max": quality_max,

            },

        )

    except FileNotFoundError:

        _raise("DATASET_NOT_FOUND", dataset_id, status=404)

    except PermissionError as exc:

        _raise("DATASET_FORBIDDEN", str(exc), status=403)





@router.get("/{dataset_id:path}")

def get_dataset(dataset_id: str, request: Request):

    try:

        return _results(request).get_dataset(dataset_id)

    except FileNotFoundError:

        _raise("DATASET_NOT_FOUND", dataset_id, status=404)

    except PermissionError as exc:

        _raise("DATASET_FORBIDDEN", str(exc), status=403)


