"""FastAPI application factory (`uvicorn --factory fabric_inspection.api.app:create_app`)."""

import asyncio
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from fabric_inspection import __version__
from fabric_inspection.api.controller import operations, predictions, quality
from fabric_inspection.api.dependencies import AppState
from fabric_inspection.api.errors import register_error_handlers
from fabric_inspection.api.security import (
    ApiKeyVerifier,
    BodySizeLimitMiddleware,
    RateLimiter,
    SecurityHeadersMiddleware,
)
from fabric_inspection.config import ApiSettings
from fabric_inspection.db.session import build_engine, build_session_factory, session_scope
from fabric_inspection.inference.inspector import FabricInspector
from fabric_inspection.service.model_service import register_models

logger = logging.getLogger(__name__)

DESCRIPTION = """
Fabric defect inspection: PatchCore anomaly detection + ResNet-18 defect classifier served with
ONNX Runtime (CPU), ASTM D5430 four-point roll grading and ISO 2859-1 lot acceptance.

Every `/api/v1` route requires the `X-API-Key` header (use **Authorize**). Weights are licensed
CC BY-NC-SA 4.0 (trained on MVTec AD): non-commercial use only.
"""
# Multipart overhead on top of the image itself.
_MULTIPART_OVERHEAD = 64 * 1024


def create_app(
    settings: ApiSettings | None = None, inspector: FabricInspector | None = None
) -> FastAPI:
    settings = settings or ApiSettings()  # API_KEYS comes from the environment
    logging.basicConfig(level=settings.log_level.upper())
    inspector = inspector or FabricInspector.from_settings(settings)
    engine = build_engine(settings.database_url)
    session_factory = build_session_factory(engine)
    with session_scope(session_factory) as session:
        register_models(session, inspector.registry)

    app = FastAPI(
        title="vision-defectos-tela API",
        version=__version__,
        description=DESCRIPTION,
        docs_url="/docs" if settings.enable_docs else None,
        redoc_url=None,
        openapi_url="/openapi.json" if settings.enable_docs else None,
        license_info={"name": "MIT (code) / CC BY-NC-SA 4.0 (weights)"},
    )
    app.state.services = AppState(
        settings=settings,
        session_factory=session_factory,
        inspector=inspector,
        verifier=ApiKeyVerifier(settings.api_keys),
        key_limiter=RateLimiter(settings.rate_limit_per_minute),
        ip_limiter=RateLimiter(max(5, settings.rate_limit_per_minute // 3)),
        inference_slots=asyncio.Semaphore(settings.max_concurrent_inferences),
    )
    register_error_handlers(app)
    app.include_router(operations.health_router)
    app.include_router(predictions.router)
    app.include_router(quality.router)
    app.include_router(operations.router)

    if settings.cors_allowed_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_allowed_origins,
            allow_methods=["GET", "POST"],
            allow_headers=["X-API-Key", "Content-Type"],
            max_age=600,
        )
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(
        BodySizeLimitMiddleware, max_bytes=settings.max_upload_bytes + _MULTIPART_OVERHEAD
    )
    logger.info(
        "API ready: detector %s, classifier %s",
        inspector.registry.detector.name,
        inspector.registry.classifier.name,
    )
    return app
