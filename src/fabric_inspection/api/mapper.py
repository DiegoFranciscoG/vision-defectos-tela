"""Entity / domain object -> DTO conversions."""

from fabric_inspection.api.dto import (
    AlertDto,
    DefectDto,
    PredictionResponse,
    RollQualityResponse,
    RollSummary,
    ScoredDefectDto,
)
from fabric_inspection.db.models import Prediction, QualityAlert
from fabric_inspection.service.quality_service import RollQuality


def prediction_to_dto(prediction: Prediction) -> PredictionResponse:
    return PredictionResponse(
        id=prediction.id,
        created_at=prediction.created_at,
        image_sha256=prediction.image_sha256,
        width=prediction.width,
        height=prediction.height,
        score=round(prediction.score, 5),
        threshold=round(prediction.threshold, 5),
        score_ratio=round(prediction.score / prediction.threshold, 3),
        is_defective=prediction.is_defective,
        predicted_class=prediction.predicted_class,
        class_confidence=prediction.class_confidence,
        defects=[
            DefectDto(
                class_code=defect.class_code,
                x=defect.bbox_x,
                y=defect.bbox_y,
                width=defect.bbox_w,
                height=defect.bbox_h,
                length_mm=float(defect.length_mm),
                is_hole=defect.is_hole,
                points=defect.points,
            )
            for defect in prediction.defects
        ],
        latency_ms=prediction.latency_ms,
        roll_code=prediction.roll.code if prediction.roll else None,
        position_m=float(prediction.position_m) if prediction.position_m is not None else None,
    )


def roll_quality_to_dto(quality: RollQuality) -> RollQualityResponse:
    assessment = quality.assessment
    return RollQualityResponse(
        code=quality.roll.code,
        lot_code=quality.roll.lot_code,
        fabric_type=quality.roll.fabric_type,
        width_cm=float(quality.roll.width_cm),
        length_m=float(quality.roll.length_m),
        raw_points=assessment.raw_points,
        total_points=assessment.total_points,
        points_per_100_sq_yd=assessment.points_per_100_sq_yd,
        max_points_allowed=assessment.max_points_allowed,
        decision=assessment.decision.value,
        defects=[
            ScoredDefectDto(
                position_m=item.defect.position_m,
                length_mm=item.defect.length_mm,
                is_hole=item.defect.is_hole,
                class_code=item.defect.class_code,
                points=item.points,
            )
            for item in assessment.defects
        ],
        textrack_payload=quality.textrack_payload,
    )


def roll_summary(quality: RollQuality) -> RollSummary:
    assessment = quality.assessment
    return RollSummary(
        code=quality.roll.code,
        lot_code=quality.roll.lot_code,
        fabric_type=quality.roll.fabric_type,
        width_cm=float(quality.roll.width_cm),
        length_m=float(quality.roll.length_m),
        defects=len(assessment.defects),
        total_points=assessment.total_points,
        points_per_100_sq_yd=assessment.points_per_100_sq_yd,
        decision=assessment.decision.value,
    )


def alert_to_dto(alert: QualityAlert, roll_code: str) -> AlertDto:
    return AlertDto(
        id=alert.id,
        roll_code=roll_code,
        level=alert.level,
        total_points=alert.total_points,
        points_per_100_sq_yd=float(alert.points_per_100_sq_yd),
        max_points_allowed=float(alert.max_points_allowed),
        created_at=alert.created_at,
    )
