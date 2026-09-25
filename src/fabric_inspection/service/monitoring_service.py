"""Drift monitoring over the latest predictions of the production detector."""

from dataclasses import dataclass
from datetime import datetime

from sqlalchemy.orm import Session

from fabric_inspection.db.models import DriftReport, ModelVersion
from fabric_inspection.domain.drift import DriftResult, detect_drift
from fabric_inspection.exceptions import NotFoundError
from fabric_inspection.repository.repositories import (
    ModelVersionRepository,
    PredictionRepository,
    QualityRepository,
)


@dataclass(frozen=True, slots=True)
class ScoreWindow:
    model: ModelVersion
    scores: list[float]
    reference_scores: list[float]
    result: DriftResult
    window_start: datetime | None
    window_end: datetime | None


class MonitoringService:
    def __init__(self, session: Session, detector_name: str) -> None:
        self._models = ModelVersionRepository(session)
        self._predictions = PredictionRepository(session)
        self._quality = QualityRepository(session)
        self._detector_name = detector_name

    def evaluate(self, window: int) -> ScoreWindow:
        model = self._models.get_production(self._detector_name)
        if model is None or model.threshold is None or not model.reference_scores:
            raise NotFoundError("No production detector with a reference distribution")
        recent = list(self._predictions.recent_scores(model.id, limit=window))
        scores = [prediction.score for prediction in recent]
        result = detect_drift(scores, model.reference_scores, threshold=model.threshold)
        return ScoreWindow(
            model=model,
            scores=scores,
            reference_scores=list(model.reference_scores),
            result=result,
            window_start=recent[-1].created_at if recent else None,
            window_end=recent[0].created_at if recent else None,
        )

    def record(self, window: int) -> DriftReport:
        evaluation = self.evaluate(window)
        result = evaluation.result
        report = DriftReport(
            model_version_id=evaluation.model.id,
            window_start=evaluation.window_start,
            window_end=evaluation.window_end,
            n_predictions=result.n_window,
            ks_statistic=result.ks_statistic,
            p_value=result.p_value,
            mean_score=result.window_mean,
            alert_rate=result.alert_rate,
            max_alert_rate=result.max_alert_rate,
            drift_detected=result.drift_detected,
            reason=result.reason[:120],
        )
        self._quality.add(report)
        return report
