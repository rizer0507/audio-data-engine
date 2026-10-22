from __future__ import annotations



import json

from typing import Any



from fastapi import APIRouter, Header, Request



from audio_engine.web.schemas import JobSubmitRequest

from audio_engine.web.store import ConflictError



router = APIRouter(prefix="/evaluations", tags=["evaluations"])





def _raise(code: str, message: str, *, status: int = 400, details: dict | None = None):

    from audio_engine.web.app import raise_api_error



    raise_api_error(code, message, status=status, details=details)





@router.get("")

def list_evaluations(request: Request, cursor: int = 0, limit: int = 50):

    jobs = request.app.state.job_service.list(cursor=cursor, limit=limit * 3)

    items = []

    for j in jobs["items"]:

        req = j.get("request") or {}

        root = req.get("root_plugin") or {}

        purpose = None

        try:

            desc = request.app.state.engine_service.get_plugin(

                root.get("kind") or "pipeline",

                root.get("id"),

                root.get("version"),

            )

            purpose = (desc.get("capabilities") or {}).get("purpose")

        except Exception:

            purpose = None

        if purpose == "evaluation" or j.get("task_type") == "evaluation":

            items.append({**j, "evaluation_id": j["job_id"], "purpose": purpose})

        if len(items) >= limit:

            break

    return {"items": items, "next_cursor": jobs["next_cursor"], "total": None}





@router.post("", status_code=202)

def create_evaluation(

    payload: JobSubmitRequest,

    request: Request,

    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),

):

    """Thin facade: same as POST /jobs but requires evaluation purpose plugin.



    Formal Gold gate is checked against the planned dataset when present; missing

    gold blocks submit with an explicit error (no fabricated accuracy).

    """



    plan = request.app.state.store.get_plan(payload.plan_id)

    if not plan:

        _raise("PLAN_NOT_FOUND", payload.plan_id, status=404)



    plan_json = json.loads(plan["plan_json"])

    plugin = plan_json.get("plugin") or plan_json.get("root_plugin") or {}

    # plan may nest differently depending on engine; also check request

    req = json.loads(plan["task_request_json"]) if plan.get("task_request_json") else {}

    root = req.get("root_plugin") or plugin

    try:

        desc = request.app.state.engine_service.get_plugin(

            root.get("kind"), root.get("id"), root.get("version")

        )

    except KeyError:

        _raise("PLUGIN_NOT_FOUND", str(root), status=404)

    if (desc.get("capabilities") or {}).get("purpose") != "evaluation":

        _raise(

            "NOT_EVALUATION_PLUGIN",

            "root plugin purpose must be evaluation",

            status=400,

        )



    # Gold gate on dataset input when available

    inp = req.get("input") or {}

    dataset_id = inp.get("dataset_id")

    gate = None

    if dataset_id:

        try:

            gate = request.app.state.results_service.inspect_gold_gate(

                dataset_id, require_formal_gold=True

            )

        except FileNotFoundError:

            _raise("DATASET_NOT_FOUND", str(dataset_id), status=404)

        except PermissionError as exc:

            _raise("DATASET_FORBIDDEN", str(exc), status=403)

        if not gate.get("formal_gate_ok"):

            _raise(

                "FORMAL_GOLD_GATE",

                "正式 Gold 门禁未通过，拒绝创建正式评测任务（不会生成虚构准确率）",

                status=409,

                details={

                    "denominator": gate.get("denominator"),

                    "errors": gate.get("errors"),

                    "eval_trust": gate.get("eval_trust"),

                    "with_gold": len(gate.get("with_gold") or []),

                    "without_gold": len(gate.get("without_gold") or []),

                },

            )



    try:

        job, created = request.app.state.job_service.submit(

            plan_id=payload.plan_id,

            plan_digest=payload.plan_digest,

            idempotency_key=idempotency_key,

        )

        return {

            **job,

            "created": created,

            "evaluation_id": job["job_id"],

            "gold_gate": gate,

        }

    except ConflictError as exc:

        _raise(exc.code, exc.message, details=exc.details, status=409)





@router.get("/{evaluation_id}")

def get_evaluation(evaluation_id: str, request: Request):

    try:

        job = request.app.state.job_service.get(evaluation_id)

    except KeyError:

        _raise("EVALUATION_NOT_FOUND", evaluation_id, status=404)



    req = job.get("request") or {}

    dataset_id = (req.get("input") or {}).get("dataset_id")

    gate = None

    if dataset_id:

        try:

            gate = request.app.state.results_service.inspect_gold_gate(

                dataset_id, require_formal_gold=True

            )

        except Exception:

            gate = None



    metrics = {}

    for art in job.get("artifacts") or []:

        if isinstance(art, dict) and art.get("artifact_kind") in {"report", "metrics"}:

            metrics["artifact"] = art

    # Prefer counts/metrics stored on job progress

    progress = job.get("progress") or {}

    return {

        **job,

        "evaluation_id": evaluation_id,

        "gold_gate": gate,

        "metrics_summary": {

            "progress_counts": progress.get("counts") or progress,

            "artifacts": job.get("artifacts") or [],

            "denominator_note": (

                "指标分母以引擎报告为准：正式准确率仅统计合法 Gold 子集；"

                "失败/缺失计入失败计数，不得用前端虚构补齐。"

            ),

            "extra": metrics,

        },

    }





@router.post("/{evaluation_id}/compare")

def compare_evaluations(evaluation_id: str, payload: dict[str, Any], request: Request):

    """Same-口径对比：仅当数据集版本与指标口径一致时返回并排摘要。"""



    other_id = str(payload.get("other_evaluation_id") or "")

    if not other_id:

        _raise("COMPARE_TARGET_REQUIRED", "other_evaluation_id required", status=400)

    try:

        a = request.app.state.job_service.get(evaluation_id)

        b = request.app.state.job_service.get(other_id)

    except KeyError as exc:

        _raise("EVALUATION_NOT_FOUND", str(exc), status=404)



    a_ds = ((a.get("request") or {}).get("input") or {}).get("dataset_id")

    b_ds = ((b.get("request") or {}).get("input") or {}).get("dataset_id")

    comparable = a_ds and b_ds and a_ds == b_ds

    return {

        "comparable": bool(comparable),

        "reason": None

        if comparable

        else "评测输入数据集版本不一致，禁止同口径对比",

        "left": {

            "evaluation_id": evaluation_id,

            "dataset_id": a_ds,

            "status": a.get("status"),

            "plan_digest": a.get("plan_digest"),

        },

        "right": {

            "evaluation_id": other_id,

            "dataset_id": b_ds,

            "status": b.get("status"),

            "plan_digest": b.get("plan_digest"),

        },

    }


