"""Roll grading (ASTM D5430), alerts and lot acceptance (ISO 2859-1)."""

from decimal import Decimal
from typing import Annotated

from fastapi import APIRouter, Path, Query, status

from fabric_inspection.api.dependencies import ApiKey, DbSession, State
from fabric_inspection.api.dto import (
    CODE_PATTERN,
    AlertDto,
    AqlPlanResponse,
    AssessmentResponse,
    ErrorResponse,
    LotInspectionRequest,
    LotInspectionResponse,
    RollQualityResponse,
    RollSummary,
)
from fabric_inspection.api.mapper import alert_to_dto, roll_quality_to_dto, roll_summary
from fabric_inspection.domain.aql import STANDARD_EDITION, InspectionLevel, sampling_plan
from fabric_inspection.exceptions import BusinessRuleError
from fabric_inspection.service.quality_service import QualityService

router = APIRouter(prefix="/api/v1", tags=["quality"])
Code = Annotated[str, Path(pattern=CODE_PATTERN)]
_ERRORS: dict[int | str, dict[str, object]] = {
    code: {"model": ErrorResponse} for code in (401, 404, 422, 429)
}


@router.get("/rolls", response_model=list[RollSummary], summary="Rolls with their current grade")
def list_rolls(session: DbSession, state: State, _: ApiKey) -> list[RollSummary]:
    return [roll_summary(item) for item in QualityService(session, state.settings).list_rolls()]


@router.get(
    "/rolls/{code}/quality",
    response_model=RollQualityResponse,
    responses=_ERRORS,
    summary="Four-point grade of a roll and the textrack payload",
)
def roll_quality(code: Code, session: DbSession, state: State, _: ApiKey) -> RollQualityResponse:
    return roll_quality_to_dto(QualityService(session, state.settings).roll_quality(code))


@router.post(
    "/rolls/{code}/assessments",
    response_model=AssessmentResponse,
    status_code=status.HTTP_201_CREATED,
    responses=_ERRORS,
    summary="Grade the roll now and store an alert if it is not accepted",
)
def assess_roll(code: Code, session: DbSession, state: State, _: ApiKey) -> AssessmentResponse:
    quality, alert = QualityService(session, state.settings).record_assessment(code)
    return AssessmentResponse(
        quality=roll_quality_to_dto(quality),
        alert=alert_to_dto(alert, quality.roll.code) if alert else None,
    )


@router.get("/quality/alerts", response_model=list[AlertDto], summary="Latest quality alerts")
def latest_alerts(
    session: DbSession,
    state: State,
    _: ApiKey,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> list[AlertDto]:
    alerts = QualityService(session, state.settings).latest_alerts(limit)
    return [alert_to_dto(alert, alert.roll.code) for alert in alerts]


@router.get(
    "/quality/aql-plan",
    response_model=AqlPlanResponse,
    responses=_ERRORS,
    summary="ISO 2859-1 single sampling plan (normal inspection)",
)
def aql_plan(
    _: ApiKey,
    lot_size: Annotated[int, Query(ge=2, le=10_000_000)],
    aql: Annotated[Decimal, Query(description="0.10 to 6.5")] = Decimal("2.5"),
    inspection_level: Annotated[InspectionLevel, Query()] = InspectionLevel.II,
) -> AqlPlanResponse:
    try:
        plan = sampling_plan(lot_size, inspection_level, aql)
    except ValueError as error:
        raise BusinessRuleError(str(error)) from error
    return AqlPlanResponse(
        lot_size=lot_size,
        inspection_level=inspection_level,
        aql=aql,
        code_letter=plan.initial_letter,
        plan_letter=plan.plan_letter,
        sample_size=plan.sample_size,
        accept_number=plan.accept_number,
        reject_number=plan.reject_number,
        full_inspection=plan.full_inspection,
        standard=STANDARD_EDITION,
    )


@router.post(
    "/lots/{lot_code}/inspections",
    response_model=LotInspectionResponse,
    status_code=status.HTTP_201_CREATED,
    responses=_ERRORS,
    summary="Accept or reject a lot of rolls with ISO 2859-1",
)
def inspect_lot(
    lot_code: Code,
    body: LotInspectionRequest,
    session: DbSession,
    state: State,
    _: ApiKey,
) -> LotInspectionResponse:
    result = QualityService(session, state.settings).evaluate_lot(
        lot_code, aql=body.aql, level=body.inspection_level
    )
    inspection = result.inspection
    return LotInspectionResponse(
        lot_code=inspection.lot_code,
        lot_size=inspection.lot_size,
        inspection_level=inspection.inspection_level,
        aql=inspection.aql,
        code_letter=inspection.code_letter,
        sample_size=inspection.sample_size,
        accept_number=inspection.accept_number,
        reject_number=inspection.reject_number,
        rejected_rolls=inspection.rejected_rolls,
        decision=inspection.decision,
        sampled_rolls=[roll_summary(item) for item in result.sampled_rolls],
        standard=inspection.standard_edition,
    )
