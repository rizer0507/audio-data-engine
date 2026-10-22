"""FastAPI application factory."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from audio_engine.plugins.registry import PluginRegistry
from audio_engine.web.api import datasets, evaluations, jobs, models, plugins, sources
from audio_engine.web.services.engine import EngineService
from audio_engine.web.services.jobs import JobService
from audio_engine.web.services.models import ModelService
from audio_engine.web.services.results import ResultsService
from audio_engine.web.services.sources import SourceService
from audio_engine.web.settings import WebSettings, load_web_settings
from audio_engine.web.store import JobStore


# Nginx 对外路径 = 原路径前面加这一段。中间件剥掉前缀后再走原路由，
# 因此 /asr_data/api/v1/... 与 /api/v1/... 都能命中。
PUBLIC_PATH_PREFIX = "/asr_data"


class PublicPrefixMiddleware:
    """Accept ``/asr_data/...`` and dispatch it as the unprefixed route."""

    def __init__(self, app, prefix: str = PUBLIC_PATH_PREFIX):
        self.app = app
        self.prefix = prefix.rstrip("/") or prefix

    async def __call__(self, scope, receive, send):
        if scope["type"] in {"http", "websocket"}:
            path = scope.get("path") or ""
            prefix = self.prefix
            if path == prefix or path.startswith(prefix + "/"):
                scope = dict(scope)
                stripped = path[len(prefix) :] or "/"
                scope["path"] = stripped
                raw = scope.get("raw_path")
                if isinstance(raw, (bytes, bytearray)):
                    raw_text = raw.decode("latin1")
                    if raw_text == prefix or raw_text.startswith(prefix + "/"):
                        scope["raw_path"] = (raw_text[len(prefix) :] or "/").encode("latin1")
        await self.app(scope, receive, send)


class APIError(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        status: int = 400,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status
        self.details = details or {}


def raise_api_error(
    code: str,
    message: str,
    *,
    status: int = 400,
    details: dict[str, Any] | None = None,
) -> None:
    raise APIError(code, message, status=status, details=details)


def create_app(settings: WebSettings | None = None, *, config_path: str | Path | None = None) -> FastAPI:
    if settings is None:
        if config_path is None:
            raise ValueError("settings or config_path required")
        settings = load_web_settings(config_path)

    app = FastAPI(title="Audio Data Engine Web", version="0.1.0")
    app.state.settings = settings
    app.state.store = JobStore(settings.sqlite_path)
    app.state.registry = PluginRegistry(project_root=settings.project_root)
    app.state.source_service = SourceService(settings.sources_manifest, settings.allowed_source_roots)
    app.state.engine_service = EngineService(
        settings, app.state.registry, source_service=app.state.source_service
    )
    app.state.job_service = JobService(settings, app.state.store, app.state.engine_service)
    app.state.model_service = ModelService(settings)
    app.state.results_service = ResultsService(settings, store=app.state.store)

    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        request_id = request.headers.get("X-Request-Id") or uuid.uuid4().hex
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-Id"] = request_id
        return response

    def error_payload(request: Request, code: str, message: str, details: dict[str, Any] | None = None):
        return {
            "error": {
                "code": code,
                "message": message,
                "details": details or {},
                "request_id": getattr(request.state, "request_id", None),
            }
        }

    @app.exception_handler(APIError)
    async def api_error_handler(request: Request, exc: APIError):
        return JSONResponse(
            status_code=exc.status,
            content=error_payload(request, exc.code, exc.message, exc.details),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content=error_payload(
                request,
                "REQUEST_VALIDATION",
                "request validation failed",
                {"errors": exc.errors()},
            ),
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_handler(request: Request, exc: StarletteHTTPException):
        if request.url.path.startswith("/api"):
            return JSONResponse(
                status_code=exc.status_code,
                content=error_payload(request, "HTTP_ERROR", str(exc.detail)),
            )
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "version": "0.1.0",
            "project_root": str(settings.project_root),
        }

    @app.get("/api/v1/capabilities")
    def capabilities():
        return {
            "providers": ["glm", "qwen"],
            "evidence_modes": ["existing", "compute_acoustic", "plan", "preview"],
            "task_types": ["pipeline", "tagging"],
            "input_kinds": ["source", "dataset"],
            "job_statuses": [
                "queued",
                "running",
                "succeeded",
                "failed",
                "cancelling",
                "cancelled",
                "interrupted",
            ],
            "pcm_required_fields": ["sample_rate", "bit_depth", "channels"],
            "operator_id": settings.operator_id,
            "review_passes": ["first", "second", "adjudication", "spot_check"],
            "tag_actions": [
                "manual_add",
                "manual_block",
                "restore_auto",
                "retract_manual",
            ],
            "sample_filter_keys": [
                "tag",
                "tag_exclude",
                "tag_any",
                "tag_all",
                "ternary",
                "tag_ternary",
                "route_status",
                "review_status",
                "quality_key",
                "quality_min",
                "quality_max",
            ],
        }

    api = FastAPI()  # unused; mount routers on main with prefix
    app.include_router(sources.router, prefix="/api/v1")
    app.include_router(models.router, prefix="/api/v1")
    app.include_router(plugins.router, prefix="/api/v1")
    app.include_router(jobs.router, prefix="/api/v1")
    app.include_router(datasets.router, prefix="/api/v1")
    app.include_router(evaluations.router, prefix="/api/v1")

    @app.get("/api/v1/artifacts/{artifact_id}/download")
    def download_artifact(artifact_id: str, request: Request):
        # artifact_id is a path relative to project root previously registered by a job
        from fastapi.responses import FileResponse

        path = Path(artifact_id)
        if not path.is_absolute():
            path = (settings.project_root / path).resolve()
        try:
            path.relative_to(settings.project_root.resolve())
        except ValueError:
            raise_api_error("ARTIFACT_FORBIDDEN", "artifact outside project root", status=403)
        if not path.is_file():
            raise_api_error("ARTIFACT_NOT_FOUND", artifact_id, status=404)
        return FileResponse(path, filename=path.name)

    @app.on_event("startup")
    def _startup():
        try:
            app.state.engine_service.refresh_plugins()
        except Exception:
            # Keep API up; plugins endpoint will surface diagnostics.
            pass

    # Outermost: strip /asr_data before routing, static files, and other middleware.
    app.add_middleware(PublicPrefixMiddleware)

    static_dir = settings.static_dir
    if static_dir.is_dir() and any(static_dir.iterdir()):
        from fastapi.responses import FileResponse

        assets_dir = static_dir / "assets"
        if assets_dir.is_dir():
            app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

        index_html = static_dir / "index.html"

        @app.get("/")
        def spa_index():
            return FileResponse(index_html)

        @app.get("/{full_path:path}")
        def spa_fallback(full_path: str):
            # API/health already registered above; unknown /api must not reach here as HTML.
            if full_path == "health" or full_path.startswith("api/") or full_path.startswith("api"):
                raise_api_error("HTTP_ERROR", "Not Found", status=404)
            candidate = (static_dir / full_path).resolve()
            try:
                candidate.relative_to(static_dir.resolve())
            except ValueError:
                return FileResponse(index_html)
            if candidate.is_file():
                return FileResponse(candidate)
            # Deep SPA routes (e.g. /jobs/:id) → index.html
            return FileResponse(index_html)

    return app


def main(argv: list[str] | None = None) -> None:
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(prog="audio-data-web")
    parser.add_argument("--config", required=True, help="absolute path to configs/web.yaml")
    parser.add_argument("--project-root", default=None)
    args = parser.parse_args(argv)
    settings = load_web_settings(args.config, project_root=args.project_root)
    app = create_app(settings)
    uvicorn.run(app, host=settings.bind_host, port=settings.bind_port)


if __name__ == "__main__":
    main()
