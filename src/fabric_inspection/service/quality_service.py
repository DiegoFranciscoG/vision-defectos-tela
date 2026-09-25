"""Roll grading (ASTM D5430 four-point) and lot acceptance (ISO 2859-1) from stored detections."""

import random
from dataclasses import dataclass
from decimal import Decimal

from sqlalchemy.orm import Session

from fabric_inspection.config import QualitySettings
from fabric_inspection.db.models import LotInspection, QualityAlert, Roll
from fabric_inspection.domain.aql import STANDARD_EDITION, InspectionLevel, sampling_plan
from fabric_inspection.domain.four_point import (
    RollAssessment,
    RollDecision,
    RollDefect,
    assess_roll,
    to_textrack_payload,
)
from fabric_inspection.exceptions import BusinessRuleError, NotFoundError
from fabric_inspection.repository.repositories import (
    PredictionRepository,
    QualityRepository,
    RollRepository,
)


@dataclass(frozen=True, slots=True)
class RollQuality:
    roll: Roll
    assessment: RollAssessment
    textrack_payload: dict[str, object]


@dataclass(frozen=True, slots=True)
class LotResult:
    inspection: LotInspection
    sampled_rolls: list[RollQuality]


class QualityService:
    def __init__(self, session: Session, settings: QualitySettings) -> None:
        self._rolls = RollRepository(session)
        self._predictions = PredictionRepository(session)
        self._quality = QualityRepository(session)
        self._settings = settings

    def _roll(self, code: str) -> Roll:
        roll = self._rolls.get_by_code(code)
        if roll is None:
            raise NotFoundError(f"Roll {code} does not exist")
        return roll

    def _grade(self, roll: Roll) -> RollQuality:
        defects = [
            RollDefect(
                position_m=float(prediction.position_m or 0),
                length_mm=float(defect.length_mm),
                is_hole=defect.is_hole,
                class_code=defect.class_code,
            )
            for defect, prediction in self._predictions.defects_for_roll(roll.id)
        ]
        assessment = assess_roll(
            defects,
            inspected_length_m=float(roll.length_m),
            width_cm=float(roll.width_cm),
            max_points_per_100_sq_yd=self._settings.max_points_per_100_sq_yd,
            warning_ratio=self._settings.warning_ratio,
        )
        textrack_codes = {
            code: item.textrack_code for code, item in self._quality.defect_classes().items()
        }
        return RollQuality(
            roll=roll,
            assessment=assessment,
            textrack_payload=to_textrack_payload(assessment, textrack_codes),
        )

    def roll_quality(self, code: str) -> RollQuality:
        return self._grade(self._roll(code))

    def list_rolls(self) -> list[RollQuality]:
        return [self._grade(roll) for roll in self._rolls.list_all()]

    def record_assessment(self, code: str) -> tuple[RollQuality, QualityAlert | None]:
        """Grade the roll and store an alert when it is not plainly accepted."""
        quality = self._grade(self._roll(code))
        result = quality.assessment
        if result.decision is RollDecision.ACCEPTED:
            return quality, None
        alert = QualityAlert(
            roll_id=quality.roll.id,
            level=result.decision.value,
            inspected_length_m=Decimal(str(result.inspected_length_m)),
            width_cm=Decimal(str(result.width_cm)),
            total_points=result.total_points,
            points_per_100_sq_yd=Decimal(str(result.points_per_100_sq_yd)),
            max_points_allowed=Decimal(str(result.max_points_allowed)),
        )
        self._quality.add(alert)
        return quality, alert

    def latest_alerts(self, limit: int) -> list[QualityAlert]:
        return list(self._quality.latest_alerts(limit))

    def evaluate_lot(
        self, lot_code: str, *, aql: Decimal, level: InspectionLevel = InspectionLevel.II
    ) -> LotResult:
        """Sample rolls of the lot with the ISO 2859-1 plan; a rejected roll is a defective unit.

        The sample is drawn with a generator seeded by the lot code, so the same lot always
        yields the same sample (auditable), as a random sample should be drawn once per lot.
        """
        rolls = list(self._rolls.list_by_lot(lot_code))
        if not rolls:
            raise NotFoundError(f"Lot {lot_code} has no registered rolls")
        if len(rolls) < 2:
            raise BusinessRuleError("ISO 2859-1 sampling needs a lot of at least 2 rolls")
        try:
            plan = sampling_plan(len(rolls), level, aql)
        except ValueError as error:
            raise BusinessRuleError(str(error)) from error
        sample = random.Random(lot_code).sample(rolls, plan.sample_size)  # noqa: S311 - not crypto
        graded = [self._grade(roll) for roll in sorted(sample, key=lambda roll: roll.code)]
        rejected = sum(item.assessment.decision is RollDecision.REJECTED for item in graded)
        inspection = LotInspection(
            lot_code=lot_code,
            lot_size=len(rolls),
            inspection_level=level.value,
            aql=aql,
            code_letter=plan.initial_letter,
            plan_letter=plan.plan_letter,
            sample_size=plan.sample_size,
            accept_number=plan.accept_number,
            reject_number=plan.reject_number,
            rejected_rolls=rejected,
            decision="ACCEPTED" if plan.accepts(rejected) else "REJECTED",
            standard_edition=STANDARD_EDITION,
        )
        self._quality.add(inspection)
        return LotResult(inspection=inspection, sampled_rolls=graded)
