"""SQLAlchemy ORM model. Mirrors docs/modelo-datos.md (rules R1-R12)."""

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    Numeric,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# SQLite only auto-increments INTEGER PRIMARY KEY columns; PostgreSQL gets BIGINT.
BigId = BigInteger().with_variant(Integer(), "sqlite")

NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


def utc_now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)
    type_annotation_map = {dict[str, Any]: JSON, list[float]: JSON}  # noqa: RUF012


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(primary_key=True)
    code: Mapped[str] = mapped_column(String(40), unique=True)
    name: Mapped[str] = mapped_column(String(120))
    version: Mapped[str] = mapped_column(String(20))
    source_url: Mapped[str] = mapped_column(String(300))
    license_spdx: Mapped[str] = mapped_column(String(40))
    license_url: Mapped[str] = mapped_column(String(300))
    commercial_use_allowed: Mapped[bool]
    redistribution_allowed: Mapped[bool]
    citation: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class DefectClass(Base):
    __tablename__ = "defect_classes"

    code: Mapped[str] = mapped_column(String(30), primary_key=True)
    name_es: Mapped[str] = mapped_column(String(80))
    is_defect: Mapped[bool]
    is_hole: Mapped[bool]
    textrack_code: Mapped[str | None] = mapped_column(String(20))


class Image(Base):
    __tablename__ = "images"
    __table_args__ = (
        UniqueConstraint("dataset_id", "relative_path", name="uq_images_dataset_path"),
        CheckConstraint("width > 0 AND height > 0", name="positive_size"),
        CheckConstraint("source_split IN ('train', 'test')", name="source_split"),
        CheckConstraint("split IN ('train', 'val', 'test')", name="split"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"))
    relative_path: Mapped[str] = mapped_column(String(300))
    sha256: Mapped[str] = mapped_column(String(64), unique=True)
    dhash: Mapped[str] = mapped_column(String(16))
    width: Mapped[int]
    height: Mapped[int]
    source_split: Mapped[str] = mapped_column(String(10))
    split: Mapped[str] = mapped_column(String(5), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    label: Mapped["Label"] = relationship(back_populates="image")


class Label(Base):
    __tablename__ = "labels"
    __table_args__ = (
        CheckConstraint(
            "class_code = 'good' OR (mask_path IS NOT NULL AND bbox_w > 0 AND bbox_h > 0)",
            name="defect_has_mask",
        ),
        CheckConstraint("source IN ('dataset', 'manual')", name="source"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    image_id: Mapped[int] = mapped_column(ForeignKey("images.id"), unique=True)
    class_code: Mapped[str] = mapped_column(ForeignKey("defect_classes.code"))
    mask_path: Mapped[str | None] = mapped_column(String(300))
    bbox_x: Mapped[int | None]
    bbox_y: Mapped[int | None]
    bbox_w: Mapped[int | None]
    bbox_h: Mapped[int | None]
    area_px: Mapped[int | None]
    annotator: Mapped[str] = mapped_column(String(40))
    source: Mapped[str] = mapped_column(String(10), default="dataset")

    image: Mapped[Image] = relationship(back_populates="label")


class Experiment(Base):
    __tablename__ = "experiments"
    __table_args__ = (
        CheckConstraint("model_type IN ('classifier', 'patchcore')", name="model_type"),
        CheckConstraint("protocol IN ('mvtec_official', 'stratified')", name="protocol"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    mlflow_run_id: Mapped[str] = mapped_column(String(32), unique=True)
    model_type: Mapped[str] = mapped_column(String(20))
    backbone: Mapped[str] = mapped_column(String(60))
    protocol: Mapped[str] = mapped_column(String(20))
    params: Mapped[dict[str, Any]]
    metrics: Mapped[dict[str, Any]]
    seed: Mapped[int]
    data_manifest_sha256: Mapped[str] = mapped_column(String(64))
    code_version: Mapped[str] = mapped_column(String(40))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class ModelVersion(Base):
    __tablename__ = "model_versions"
    __table_args__ = (
        UniqueConstraint("name", "version", name="uq_model_versions_name_version"),
        CheckConstraint("status IN ('candidate', 'production', 'archived')", name="status"),
        Index(
            "uq_model_versions_production",
            "name",
            unique=True,
            sqlite_where=text("status = 'production'"),
            postgresql_where=text("status = 'production'"),
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    experiment_id: Mapped[int] = mapped_column(ForeignKey("experiments.id"))
    name: Mapped[str] = mapped_column(String(60))
    version: Mapped[str] = mapped_column(String(20))
    onnx_sha256: Mapped[str] = mapped_column(String(64), unique=True)
    onnx_size_bytes: Mapped[int]
    quantized: Mapped[bool]
    threshold: Mapped[float | None]
    threshold_policy: Mapped[str | None] = mapped_column(String(20))
    pixel_threshold: Mapped[float | None]
    reference_scores: Mapped[list[float] | None]
    weights_license: Mapped[str] = mapped_column(String(40))
    release_url: Mapped[str | None] = mapped_column(String(300))
    status: Mapped[str] = mapped_column(String(12), default="candidate")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    experiment: Mapped[Experiment] = relationship()


class Roll(Base):
    __tablename__ = "rolls"
    __table_args__ = (
        CheckConstraint("width_cm >= 30 AND width_cm <= 400", name="width_range"),
        CheckConstraint("length_m > 0", name="positive_length"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    code: Mapped[str] = mapped_column(String(30), unique=True)
    lot_code: Mapped[str] = mapped_column(String(30), index=True)
    fabric_type: Mapped[str] = mapped_column(String(60))
    width_cm: Mapped[Decimal] = mapped_column(Numeric(6, 1))
    length_m: Mapped[Decimal] = mapped_column(Numeric(8, 2))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class Prediction(Base):
    __tablename__ = "predictions"
    __table_args__ = (
        CheckConstraint("position_m IS NULL OR position_m >= 0", name="position"),
        CheckConstraint("latency_ms >= 0", name="latency"),
        CheckConstraint("source IN ('api', 'batch', 'seed')", name="source"),
    )

    id: Mapped[int] = mapped_column(BigId, primary_key=True)
    model_version_id: Mapped[int] = mapped_column(ForeignKey("model_versions.id"), index=True)
    classifier_version_id: Mapped[int | None] = mapped_column(ForeignKey("model_versions.id"))
    roll_id: Mapped[int | None] = mapped_column(ForeignKey("rolls.id"), index=True)
    position_m: Mapped[Decimal | None] = mapped_column(Numeric(8, 2))
    image_sha256: Mapped[str] = mapped_column(String(64))
    width: Mapped[int]
    height: Mapped[int]
    score: Mapped[float]
    threshold: Mapped[float]
    is_defective: Mapped[bool]
    predicted_class: Mapped[str | None] = mapped_column(ForeignKey("defect_classes.code"))
    class_confidence: Mapped[float | None]
    latency_ms: Mapped[float]
    source: Mapped[str] = mapped_column(String(10))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, index=True
    )

    defects: Mapped[list["DetectedDefect"]] = relationship(
        back_populates="prediction", cascade="all, delete-orphan", order_by="DetectedDefect.id"
    )
    roll: Mapped[Roll | None] = relationship()


class DetectedDefect(Base):
    __tablename__ = "detected_defects"
    __table_args__ = (
        CheckConstraint("points BETWEEN 1 AND 4", name="points"),
        CheckConstraint("length_mm > 0", name="positive_length"),
    )

    id: Mapped[int] = mapped_column(BigId, primary_key=True)
    prediction_id: Mapped[int] = mapped_column(
        ForeignKey("predictions.id", ondelete="CASCADE"), index=True
    )
    class_code: Mapped[str] = mapped_column(ForeignKey("defect_classes.code"))
    bbox_x: Mapped[int]
    bbox_y: Mapped[int]
    bbox_w: Mapped[int]
    bbox_h: Mapped[int]
    length_mm: Mapped[Decimal] = mapped_column(Numeric(8, 1))
    is_hole: Mapped[bool]
    points: Mapped[int] = mapped_column(SmallInteger)

    prediction: Mapped[Prediction] = relationship(back_populates="defects")


class QualityAlert(Base):
    __tablename__ = "quality_alerts"
    __table_args__ = (
        CheckConstraint("level IN ('WARNING', 'REJECTED')", name="level"),
        CheckConstraint("total_points >= 0", name="points"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    roll_id: Mapped[int] = mapped_column(ForeignKey("rolls.id"), index=True)
    level: Mapped[str] = mapped_column(String(10))
    inspected_length_m: Mapped[Decimal] = mapped_column(Numeric(8, 2))
    width_cm: Mapped[Decimal] = mapped_column(Numeric(6, 1))
    total_points: Mapped[int]
    points_per_100_sq_yd: Mapped[Decimal] = mapped_column(Numeric(8, 2))
    max_points_allowed: Mapped[Decimal] = mapped_column(Numeric(8, 2))
    acknowledged: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    roll: Mapped[Roll] = relationship()


class LotInspection(Base):
    __tablename__ = "lot_inspections"
    __table_args__ = (
        CheckConstraint("lot_size >= 2", name="lot_size"),
        CheckConstraint("rejected_rolls >= 0", name="rejected_rolls"),
        CheckConstraint("decision IN ('ACCEPTED', 'REJECTED')", name="decision"),
        CheckConstraint(
            "inspection_level IN ('S-1', 'S-2', 'S-3', 'S-4', 'I', 'II', 'III')",
            name="inspection_level",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    lot_code: Mapped[str] = mapped_column(String(30), index=True)
    lot_size: Mapped[int]
    inspection_level: Mapped[str] = mapped_column(String(3))
    aql: Mapped[Decimal] = mapped_column(Numeric(5, 3))
    code_letter: Mapped[str] = mapped_column(String(1))
    plan_letter: Mapped[str] = mapped_column(String(1))
    sample_size: Mapped[int]
    accept_number: Mapped[int]
    reject_number: Mapped[int]
    rejected_rolls: Mapped[int]
    decision: Mapped[str] = mapped_column(String(10))
    standard_edition: Mapped[str] = mapped_column(String(80))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class DriftReport(Base):
    __tablename__ = "drift_reports"

    id: Mapped[int] = mapped_column(primary_key=True)
    model_version_id: Mapped[int] = mapped_column(ForeignKey("model_versions.id"), index=True)
    window_start: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    window_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    n_predictions: Mapped[int]
    ks_statistic: Mapped[float | None]
    p_value: Mapped[float | None]
    mean_score: Mapped[float | None]
    alert_rate: Mapped[float | None]
    max_alert_rate: Mapped[float]
    drift_detected: Mapped[bool]
    reason: Mapped[str] = mapped_column(String(120))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
