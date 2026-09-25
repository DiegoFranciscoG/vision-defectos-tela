"""POST /api/v1/predictions (inspect an image) and GET the prediction log."""

from typing import Annotated

import numpy as np
from fastapi import APIRouter, File, Form, Query, Response, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from PIL import Image

from fabric_inspection.api.dependencies import ApiKey, AppState, DbSession, State, inference_slot
from fabric_inspection.api.dto import (
    CODE_PATTERN,
    ErrorResponse,
    ModelRef,
    PredictionPage,
    PredictionResponse,
)
from fabric_inspection.api.mapper import prediction_to_dto
from fabric_inspection.api.upload import decode_image, read_limited
from fabric_inspection.db.session import session_scope
from fabric_inspection.inference.visualization import overlay, to_png_base64
from fabric_inspection.repository.repositories import PredictionRepository
from fabric_inspection.service.prediction_service import PredictionService

router = APIRouter(prefix="/api/v1/predictions", tags=["predictions"])

_ERRORS: dict[int | str, dict[str, object]] = {
    code: {"model": ErrorResponse} for code in (401, 404, 413, 415, 422, 429, 503)
}


def _inspect(
    state: AppState,
    image: Image.Image,
    image_sha256: str,
    roll_code: str | None,
    position_m: float | None,
    include_heatmap: bool,
) -> PredictionResponse:
    with session_scope(state.session_factory) as session:
        service = PredictionService(session, state.inspector)
        stored = service.inspect_and_store(
            image, image_sha256, roll_code=roll_code, position_m=position_m
        )
        dto = prediction_to_dto(stored.prediction)
    result = stored.result
    registry = state.inspector.registry
    dto.class_probabilities = result.class_probabilities
    dto.models = ModelRef(
        detector=f"{registry.detector.name}@{registry.detector.version}",
        classifier=f"{registry.classifier.name}@{registry.classifier.version}",
    )
    if include_heatmap:
        reference = registry.detector.reference_scores or [result.threshold * 0.5]
        boxes = [(r.x, r.y, r.width, r.height) for r in result.regions]
        dto.heatmap_png_base64 = to_png_base64(
            overlay(
                image,
                result.anomaly_map,
                low=float(np.median(reference)),
                high=max(result.threshold * 1.5, result.score),
                boxes=boxes,
            )
        )
        cam = np.maximum(result.cam, 0)
        dto.cam_png_base64 = to_png_base64(overlay(image, cam, low=0.0, high=float(cam.max()) or 1))
    return dto


@router.post(
    "",
    response_model=PredictionResponse,
    status_code=status.HTTP_201_CREATED,
    responses=_ERRORS,
    summary="Inspect a fabric image",
    description=(
        "Upload a PNG or JPEG (max 5 MB, max 4096 px per side). Returns the anomaly score, the "
        "decision against the validated threshold, the defect type, the defect regions with "
        "their ASTM D5430 points and, optionally, the heat map and CAM as PNG (base64). "
        "Only the SHA-256 of the image is stored."
    ),
)
async def create_prediction(
    response: Response,
    state: State,
    _: ApiKey,
    file: Annotated[UploadFile, File(description="PNG or JPEG image")],
    roll_code: Annotated[str | None, Form(pattern=CODE_PATTERN)] = None,
    position_m: Annotated[float | None, Form(ge=0, le=10_000)] = None,
    include_heatmap: Annotated[bool, Query()] = False,
) -> PredictionResponse:
    data = await read_limited(file, state.settings.max_upload_bytes)
    image, image_sha256 = await run_in_threadpool(decode_image, data, state.settings.max_image_side)
    async with inference_slot(state):
        dto = await run_in_threadpool(
            _inspect, state, image, image_sha256, roll_code, position_m, include_heatmap
        )
    response.headers["Server-Timing"] = f"inference;dur={dto.latency_ms}"
    return dto


@router.get("", response_model=PredictionPage, summary="Latest predictions (newest first)")
def list_predictions(
    session: DbSession,
    _: ApiKey,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0, le=100_000)] = 0,
) -> PredictionPage:
    repository = PredictionRepository(session)
    items = [prediction_to_dto(item) for item in repository.list_recent(limit=limit, offset=offset)]
    return PredictionPage(items=items, total=repository.count(), limit=limit, offset=offset)
