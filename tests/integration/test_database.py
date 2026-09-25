from decimal import Decimal

import pytest
from sqlalchemy import inspect, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from fabric_inspection.config import QualitySettings
from fabric_inspection.db.models import (
    Dataset,
    DefectClass,
    DetectedDefect,
    DriftReport,
    LotInspection,
    ModelVersion,
    Prediction,
    QualityAlert,
    Roll,
)
from fabric_inspection.domain.catalog import CLASS_CODES
from fabric_inspection.exceptions import BusinessRuleError, NotFoundError
from fabric_inspection.registry import ModelRegistry
from fabric_inspection.service.demo_seed import seed_demo
from fabric_inspection.service.model_service import register_models
from fabric_inspection.service.monitoring_service import MonitoringService
from fabric_inspection.service.quality_service import QualityService
from tests.conftest import make_registry


def test_migrations_create_all_tables_and_catalogs(session: Session) -> None:
    tables = set(inspect(session.get_bind()).get_table_names())
    assert {
        "datasets",
        "images",
        "labels",
        "defect_classes",
        "experiments",
        "model_versions",
        "rolls",
        "predictions",
        "detected_defects",
        "quality_alerts",
        "lot_inspections",
        "drift_reports",
    } <= tables
    codes = set(session.scalars(select(DefectClass.code)))
    assert codes == set(CLASS_CODES)
    dataset = session.scalar(select(Dataset))
    assert dataset is not None
    assert dataset.license_spdx == "CC-BY-NC-SA-4.0"
    assert not dataset.commercial_use_allowed


def test_check_constraints_are_enforced(session: Session) -> None:
    session.add(
        Roll(
            code="RL-X",
            lot_code="LT-X",
            fabric_type="Test",
            width_cm=Decimal(10),
            length_m=Decimal(10),
        )
    )
    with pytest.raises(IntegrityError):
        session.flush()


def test_register_models_is_idempotent_and_promotes_new_versions(
    session: Session, registry: ModelRegistry
) -> None:
    first = register_models(session, registry)
    again = register_models(session, registry)
    assert first.detector.id == again.detector.id

    clash = make_registry(detector_sha="3" * 64)
    with pytest.raises(BusinessRuleError, match="new version"):
        register_models(session, clash)

    newer = make_registry(detector_sha="3" * 64, detector_version="0.0.2")
    promoted = register_models(session, newer)
    session.flush()
    assert promoted.detector.id != first.detector.id
    statuses = dict(
        session.execute(
            select(ModelVersion.onnx_sha256, ModelVersion.status).where(
                ModelVersion.name == "patchcore-test"
            )
        ).all()
    )
    assert statuses == {"1" * 64: "archived", "3" * 64: "production"}


def test_demo_seed_builds_a_consistent_story(
    session: Session, registry: ModelRegistry, quality_settings: QualitySettings
) -> None:
    summary = seed_demo(session, registry, quality_settings, frames_per_roll=30)
    session.commit()
    assert summary is not None
    assert summary.rolls == 10
    assert summary.predictions == 300
    assert set(summary.lots) == {"LT-2026-01", "LT-2026-02"}
    assert seed_demo(session, registry, quality_settings) is None  # idempotent

    defects = session.scalars(select(DetectedDefect)).all()
    assert defects
    assert all(1 <= defect.points <= 4 for defect in defects)
    assert all(defect.points == 4 for defect in defects if defect.is_hole)
    assert session.scalar(select(LotInspection).limit(1)) is not None

    quality = QualityService(session, quality_settings)
    rolls = quality.list_rolls()
    assert len(rolls) == 10
    alerts = session.scalars(select(QualityAlert)).all()
    flagged = {item.roll.code for item in rolls if item.assessment.decision.value != "ACCEPTED"}
    assert {alert.roll_id for alert in alerts} == {
        item.roll.id for item in rolls if item.roll.code in flagged
    }
    payload = rolls[0].textrack_payload
    assert set(payload) == {"inspectedLengthM", "widthCm", "defects"}


def test_quality_service_errors(session: Session, quality_settings: QualitySettings) -> None:
    quality = QualityService(session, quality_settings)
    with pytest.raises(NotFoundError):
        quality.roll_quality("RL-NOPE")
    with pytest.raises(NotFoundError):
        quality.evaluate_lot("LT-NOPE", aql=Decimal("2.5"))
    session.add(
        Roll(
            code="RL-1",
            lot_code="LT-ONE",
            fabric_type="Test",
            width_cm=Decimal(150),
            length_m=Decimal(50),
        )
    )
    session.flush()
    with pytest.raises(BusinessRuleError, match="at least 2"):
        quality.evaluate_lot("LT-ONE", aql=Decimal("2.5"))


def test_monitoring_detects_stable_seed_and_records_report(
    session: Session, registry: ModelRegistry, quality_settings: QualitySettings
) -> None:
    seed_demo(session, registry, quality_settings, frames_per_roll=30)
    monitoring = MonitoringService(session, registry.detector.name)
    evaluation = monitoring.evaluate(window=200)
    assert evaluation.result.n_window == 200
    assert not evaluation.result.drift_detected
    report = monitoring.record(window=200)
    assert isinstance(report, DriftReport)
    assert report.id is not None


def test_monitoring_without_model_is_not_found(session: Session) -> None:
    with pytest.raises(NotFoundError):
        MonitoringService(session, "missing").evaluate(window=50)


def test_prediction_defects_cascade(session: Session, registry: ModelRegistry) -> None:
    served = register_models(session, registry)
    prediction = Prediction(
        model_version_id=served.detector.id,
        image_sha256="f" * 64,
        width=10,
        height=10,
        score=2.0,
        threshold=1.4,
        is_defective=True,
        latency_ms=12.0,
        source="api",
    )
    prediction.defects.append(
        DetectedDefect(
            class_code="hole",
            bbox_x=0,
            bbox_y=0,
            bbox_w=5,
            bbox_h=5,
            length_mm=Decimal("0.5"),
            is_hole=True,
            points=4,
        )
    )
    session.add(prediction)
    session.flush()
    session.delete(prediction)
    session.flush()
    assert session.scalar(select(DetectedDefect)) is None
