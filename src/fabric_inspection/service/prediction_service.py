"""Run an inspection and store it (hash only, never the image: rule R12)."""

from dataclasses import dataclass
from decimal import Decimal

from PIL import Image
from sqlalchemy.orm import Session

from fabric_inspection.db.models import DetectedDefect, Prediction
from fabric_inspection.exceptions import BusinessRuleError, NotFoundError
from fabric_inspection.inference.inspector import FabricInspector, InspectionResult
from fabric_inspection.repository.repositories import (
    ModelVersionRepository,
    PredictionRepository,
    RollRepository,
)


@dataclass(frozen=True, slots=True)
class StoredPrediction:
    prediction: Prediction
    result: InspectionResult


class PredictionService:
    def __init__(self, session: Session, inspector: FabricInspector) -> None:
        self._session = session
        self._inspector = inspector
        self._models = ModelVersionRepository(session)
        self._rolls = RollRepository(session)
        self._predictions = PredictionRepository(session)

    def validate_roll(self, roll_code: str | None, position_m: float | None) -> int | None:
        """Check the roll before spending CPU on inference."""
        if roll_code is None:
            if position_m is not None:
                raise BusinessRuleError("position_m requires roll_code")
            return None
        roll = self._rolls.get_by_code(roll_code)
        if roll is None:
            raise NotFoundError(f"Roll {roll_code} does not exist")
        if position_m is None:
            raise BusinessRuleError("position_m is required when roll_code is given")
        if position_m > float(roll.length_m):
            raise BusinessRuleError(
                f"position_m {position_m} exceeds the roll length ({roll.length_m} m)"
            )
        return roll.id

    def inspect_and_store(
        self,
        image: Image.Image,
        image_sha256: str,
        *,
        roll_code: str | None,
        position_m: float | None,
    ) -> StoredPrediction:
        roll_id = self.validate_roll(roll_code, position_m)
        registry = self._inspector.registry
        detector = self._models.get_by_sha256(registry.detector.sha256)
        classifier = self._models.get_by_sha256(registry.classifier.sha256)
        if detector is None or classifier is None:
            raise NotFoundError("The served models are not registered in the database")
        result = self._inspector.inspect(image)
        prediction = Prediction(
            model_version_id=detector.id,
            classifier_version_id=classifier.id,
            roll_id=roll_id,
            position_m=Decimal(f"{position_m:.2f}") if position_m is not None else None,
            image_sha256=image_sha256,
            width=result.width,
            height=result.height,
            score=result.score,
            threshold=result.threshold,
            is_defective=result.is_defective,
            predicted_class=result.predicted_class,
            class_confidence=result.class_confidence,
            latency_ms=result.latency_ms,
            source="api",
        )
        for region in result.regions:
            prediction.defects.append(
                DetectedDefect(
                    class_code=region.class_code,
                    bbox_x=region.x,
                    bbox_y=region.y,
                    bbox_w=region.width,
                    bbox_h=region.height,
                    length_mm=Decimal(f"{region.length_mm:.1f}"),
                    is_hole=region.is_hole,
                    points=region.points,
                )
            )
        self._predictions.add(prediction)
        return StoredPrediction(prediction=prediction, result=result)
