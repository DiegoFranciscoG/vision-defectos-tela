"""Health check, model information, drift monitoring and savings estimate."""

from dataclasses import asdict
from typing import Annotated

from fastapi import APIRouter, Query, Response, status
from sqlalchemy import text

from fabric_inspection.api.dependencies import ApiKey, DbSession, State
from fabric_inspection.api.dto import (
    DriftReportResponse,
    DriftResponse,
    ErrorResponse,
    HealthResponse,
    ModelInfo,
    ModelsResponse,
    SavingsRequest,
    SavingsResponse,
)
from fabric_inspection.domain.savings import SavingsInput, estimate_savings
from fabric_inspection.registry import ModelEntry
from fabric_inspection.service.monitoring_service import MonitoringService

health_router = APIRouter(tags=["health"])
router = APIRouter(prefix="/api/v1", tags=["operations"])
Window = Annotated[int, Query(ge=30, le=5000)]


@health_router.get("/health", response_model=HealthResponse, summary="Liveness and database check")
def health(state: State, response: Response) -> HealthResponse:
    database: str = "ok"
    try:
        with state.session_factory() as session:
            session.execute(text("SELECT 1"))
    except Exception:
        database = "error"
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    detector = state.inspector.registry.detector
    return HealthResponse(
        status="ok" if database == "ok" else "degraded",
        database=database,
        detector_version=f"{detector.name}@{detector.version}",
    )


def _model_info(entry: ModelEntry) -> ModelInfo:
    test_metrics = entry.experiment.metrics.get("test", entry.experiment.metrics)
    return ModelInfo(
        name=entry.name,
        version=entry.version,
        file=entry.file,
        sha256=entry.sha256,
        size_bytes=entry.size_bytes,
        quantized=entry.quantized,
        threshold=entry.threshold,
        threshold_policy=entry.threshold_policy,
        mlflow_run_id=entry.experiment.mlflow_run_id,
        test_metrics=test_metrics if isinstance(test_metrics, dict) else {},
    )


@router.get("/models/current", response_model=ModelsResponse, summary="Served models (model card)")
def current_models(state: State, _: ApiKey) -> ModelsResponse:
    registry = state.inspector.registry
    return ModelsResponse(
        bundle_version=registry.bundle_version,
        weights_license=registry.weights_license,
        image_size=registry.image_size,
        mm_per_pixel=state.settings.mm_per_pixel,
        detector=_model_info(registry.detector),
        classifier=_model_info(registry.classifier),
    )


@router.get(
    "/monitoring/drift",
    response_model=DriftResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Score drift of the latest predictions against the validation reference",
)
def drift(session: DbSession, state: State, _: ApiKey, window: Window = 200) -> DriftResponse:
    evaluation = MonitoringService(session, state.inspector.registry.detector.name).evaluate(window)
    result = evaluation.result
    return DriftResponse(
        detector_version=f"{evaluation.model.name}@{evaluation.model.version}",
        threshold=evaluation.model.threshold or 0.0,
        **asdict(result),
        window_scores=[round(score, 5) for score in evaluation.scores],
        reference_scores=evaluation.reference_scores,
    )


@router.post(
    "/monitoring/drift-reports",
    response_model=DriftReportResponse,
    status_code=status.HTTP_201_CREATED,
    responses={404: {"model": ErrorResponse}},
    summary="Evaluate drift and store the report",
)
def record_drift(
    session: DbSession, state: State, _: ApiKey, window: Window = 200
) -> DriftReportResponse:
    report = MonitoringService(session, state.inspector.registry.detector.name).record(window)
    return DriftReportResponse(
        id=report.id,
        created_at=report.created_at,
        n_predictions=report.n_predictions,
        p_value=report.p_value,
        drift_detected=report.drift_detected,
        reason=report.reason,
    )


@router.post(
    "/savings/estimate",
    response_model=SavingsResponse,
    summary="Monthly savings of automated inspection versus manual inspection",
)
def savings(body: SavingsRequest, state: State, _: ApiKey) -> SavingsResponse:
    test = state.inspector.registry.detector.experiment.metrics.get("test", {})
    measured_recall = float(test.get("recall", 0.9)) if isinstance(test, dict) else 0.9
    measured_fpr = float(test.get("false_positive_rate", 0.0)) if isinstance(test, dict) else 0.0
    values = body.model_dump()
    values["model_recall"] = body.model_recall if body.model_recall is not None else measured_recall
    if body.false_alarms_per_100_m is None:
        # One inspected frame per metre of fabric (assumption) x measured false-positive rate.
        values["false_alarms_per_100_m"] = round(measured_fpr * 100, 3)
    estimate = estimate_savings(SavingsInput(**values))
    return SavingsResponse(
        inputs={key: float(value) for key, value in values.items()},
        **asdict(estimate),
        assumptions=[
            "S7: manual inspectors detect about 70% of defects (editable, not verified).",
            "S8: automated line speed 30 m/min vs 8-20 m/min manual (Kumar 2008).",
            "S9: hourly cost from the 2026 minimum wage (USD 482 / 240 h), no employer costs.",
            "One inspected frame per metre when false alarms come from the measured FPR.",
            "Price loss of a defective metre: 45% (lower bound of the 45-65% range, Kumar 2008).",
        ],
    )
