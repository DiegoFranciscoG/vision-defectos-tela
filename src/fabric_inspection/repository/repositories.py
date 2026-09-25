"""Data access. Every query goes through SQLAlchemy with bound parameters (no string SQL)."""

from collections.abc import Sequence

from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from fabric_inspection.db.models import (
    DefectClass,
    DetectedDefect,
    DriftReport,
    Experiment,
    LotInspection,
    ModelVersion,
    Prediction,
    QualityAlert,
    Roll,
)


class ModelVersionRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def get_production(self, name: str) -> ModelVersion | None:
        return self._session.scalar(
            select(ModelVersion).where(
                ModelVersion.name == name, ModelVersion.status == "production"
            )
        )

    def get_by_sha256(self, sha256: str) -> ModelVersion | None:
        return self._session.scalar(select(ModelVersion).where(ModelVersion.onnx_sha256 == sha256))

    def get_by_name_version(self, name: str, version: str) -> ModelVersion | None:
        return self._session.scalar(
            select(ModelVersion).where(ModelVersion.name == name, ModelVersion.version == version)
        )

    def archive_production(self, name: str) -> None:
        current = self.get_production(name)
        if current is not None:
            current.status = "archived"
            self._session.flush()

    def get_experiment(self, mlflow_run_id: str) -> Experiment | None:
        return self._session.scalar(
            select(Experiment).where(Experiment.mlflow_run_id == mlflow_run_id)
        )

    def add(self, entity: ModelVersion | Experiment) -> None:
        self._session.add(entity)
        self._session.flush()


class RollRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def get_by_code(self, code: str) -> Roll | None:
        return self._session.scalar(select(Roll).where(Roll.code == code))

    def list_all(self) -> Sequence[Roll]:
        return self._session.scalars(select(Roll).order_by(Roll.code)).all()

    def list_by_lot(self, lot_code: str) -> Sequence[Roll]:
        return self._session.scalars(
            select(Roll).where(Roll.lot_code == lot_code).order_by(Roll.code)
        ).all()

    def add(self, roll: Roll) -> Roll:
        self._session.add(roll)
        self._session.flush()
        return roll


class PredictionRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, prediction: Prediction) -> Prediction:
        self._session.add(prediction)
        self._session.flush()
        return prediction

    def list_recent(self, *, limit: int, offset: int = 0) -> Sequence[Prediction]:
        return self._session.scalars(
            select(Prediction)
            .options(selectinload(Prediction.defects), selectinload(Prediction.roll))
            .order_by(Prediction.created_at.desc(), Prediction.id.desc())
            .limit(limit)
            .offset(offset)
        ).all()

    def count(self) -> int:
        return self._session.scalar(select(func.count(Prediction.id))) or 0

    def recent_scores(self, model_version_id: int, *, limit: int) -> Sequence[Prediction]:
        return self._session.scalars(
            select(Prediction)
            .where(Prediction.model_version_id == model_version_id)
            .order_by(Prediction.created_at.desc(), Prediction.id.desc())
            .limit(limit)
        ).all()

    def defects_for_roll(self, roll_id: int) -> Sequence[tuple[DetectedDefect, Prediction]]:
        rows = self._session.execute(
            select(DetectedDefect, Prediction)
            .join(Prediction, DetectedDefect.prediction_id == Prediction.id)
            .where(Prediction.roll_id == roll_id, Prediction.is_defective.is_(True))
            .order_by(Prediction.position_m, DetectedDefect.id)
        ).all()
        return [(row[0], row[1]) for row in rows]

    def max_position_for_roll(self, roll_id: int) -> float | None:
        value = self._session.scalar(
            select(func.max(Prediction.position_m)).where(Prediction.roll_id == roll_id)
        )
        return float(value) if value is not None else None


class QualityRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, entity: QualityAlert | LotInspection | DriftReport) -> None:
        self._session.add(entity)
        self._session.flush()

    def alerts_for_roll(self, roll_id: int) -> Sequence[QualityAlert]:
        return self._session.scalars(
            select(QualityAlert)
            .where(QualityAlert.roll_id == roll_id)
            .order_by(QualityAlert.created_at.desc())
        ).all()

    def latest_alerts(self, limit: int) -> Sequence[QualityAlert]:
        return self._session.scalars(
            select(QualityAlert)
            .options(selectinload(QualityAlert.roll))
            .order_by(QualityAlert.created_at.desc(), QualityAlert.id.desc())
            .limit(limit)
        ).all()

    def defect_classes(self) -> dict[str, DefectClass]:
        return {item.code: item for item in self._session.scalars(select(DefectClass))}
