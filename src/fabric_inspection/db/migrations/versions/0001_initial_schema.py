"""Initial schema: the 12 tables of docs/modelo-datos.md.

Revision ID: 0001
Revises:
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0001"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "datasets",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("code", sa.String(length=40), nullable=False),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("version", sa.String(length=20), nullable=False),
        sa.Column("source_url", sa.String(length=300), nullable=False),
        sa.Column("license_spdx", sa.String(length=40), nullable=False),
        sa.Column("license_url", sa.String(length=300), nullable=False),
        sa.Column("commercial_use_allowed", sa.Boolean(), nullable=False),
        sa.Column("redistribution_allowed", sa.Boolean(), nullable=False),
        sa.Column("citation", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_datasets")),
        sa.UniqueConstraint("code", name=op.f("uq_datasets_code")),
    )
    op.create_table(
        "defect_classes",
        sa.Column("code", sa.String(length=30), nullable=False),
        sa.Column("name_es", sa.String(length=80), nullable=False),
        sa.Column("is_defect", sa.Boolean(), nullable=False),
        sa.Column("is_hole", sa.Boolean(), nullable=False),
        sa.Column("textrack_code", sa.String(length=20), nullable=True),
        sa.PrimaryKeyConstraint("code", name=op.f("pk_defect_classes")),
    )
    op.create_table(
        "experiments",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("mlflow_run_id", sa.String(length=32), nullable=False),
        sa.Column("model_type", sa.String(length=20), nullable=False),
        sa.Column("backbone", sa.String(length=60), nullable=False),
        sa.Column("protocol", sa.String(length=20), nullable=False),
        sa.Column("params", sa.JSON(), nullable=False),
        sa.Column("metrics", sa.JSON(), nullable=False),
        sa.Column("seed", sa.Integer(), nullable=False),
        sa.Column("data_manifest_sha256", sa.String(length=64), nullable=False),
        sa.Column("code_version", sa.String(length=40), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "model_type IN ('classifier', 'patchcore')", name=op.f("ck_experiments_model_type")
        ),
        sa.CheckConstraint(
            "protocol IN ('mvtec_official', 'stratified')", name=op.f("ck_experiments_protocol")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_experiments")),
        sa.UniqueConstraint("mlflow_run_id", name=op.f("uq_experiments_mlflow_run_id")),
    )
    op.create_table(
        "lot_inspections",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("lot_code", sa.String(length=30), nullable=False),
        sa.Column("lot_size", sa.Integer(), nullable=False),
        sa.Column("inspection_level", sa.String(length=3), nullable=False),
        sa.Column("aql", sa.Numeric(precision=5, scale=3), nullable=False),
        sa.Column("code_letter", sa.String(length=1), nullable=False),
        sa.Column("plan_letter", sa.String(length=1), nullable=False),
        sa.Column("sample_size", sa.Integer(), nullable=False),
        sa.Column("accept_number", sa.Integer(), nullable=False),
        sa.Column("reject_number", sa.Integer(), nullable=False),
        sa.Column("rejected_rolls", sa.Integer(), nullable=False),
        sa.Column("decision", sa.String(length=10), nullable=False),
        sa.Column("standard_edition", sa.String(length=80), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "decision IN ('ACCEPTED', 'REJECTED')", name=op.f("ck_lot_inspections_decision")
        ),
        sa.CheckConstraint(
            "inspection_level IN ('S-1', 'S-2', 'S-3', 'S-4', 'I', 'II', 'III')",
            name=op.f("ck_lot_inspections_inspection_level"),
        ),
        sa.CheckConstraint("lot_size >= 2", name=op.f("ck_lot_inspections_lot_size")),
        sa.CheckConstraint("rejected_rolls >= 0", name=op.f("ck_lot_inspections_rejected_rolls")),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_lot_inspections")),
    )
    with op.batch_alter_table("lot_inspections", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_lot_inspections_lot_code"), ["lot_code"], unique=False)

    op.create_table(
        "rolls",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("code", sa.String(length=30), nullable=False),
        sa.Column("lot_code", sa.String(length=30), nullable=False),
        sa.Column("fabric_type", sa.String(length=60), nullable=False),
        sa.Column("width_cm", sa.Numeric(precision=6, scale=1), nullable=False),
        sa.Column("length_m", sa.Numeric(precision=8, scale=2), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("length_m > 0", name=op.f("ck_rolls_positive_length")),
        sa.CheckConstraint("width_cm >= 30 AND width_cm <= 400", name=op.f("ck_rolls_width_range")),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_rolls")),
        sa.UniqueConstraint("code", name=op.f("uq_rolls_code")),
    )
    with op.batch_alter_table("rolls", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_rolls_lot_code"), ["lot_code"], unique=False)

    op.create_table(
        "images",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("dataset_id", sa.Integer(), nullable=False),
        sa.Column("relative_path", sa.String(length=300), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("dhash", sa.String(length=16), nullable=False),
        sa.Column("width", sa.Integer(), nullable=False),
        sa.Column("height", sa.Integer(), nullable=False),
        sa.Column("source_split", sa.String(length=10), nullable=False),
        sa.Column("split", sa.String(length=5), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "source_split IN ('train', 'test')", name=op.f("ck_images_source_split")
        ),
        sa.CheckConstraint("split IN ('train', 'val', 'test')", name=op.f("ck_images_split")),
        sa.CheckConstraint("width > 0 AND height > 0", name=op.f("ck_images_positive_size")),
        sa.ForeignKeyConstraint(
            ["dataset_id"], ["datasets.id"], name=op.f("fk_images_dataset_id_datasets")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_images")),
        sa.UniqueConstraint("dataset_id", "relative_path", name="uq_images_dataset_path"),
        sa.UniqueConstraint("sha256", name=op.f("uq_images_sha256")),
    )
    with op.batch_alter_table("images", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_images_split"), ["split"], unique=False)

    op.create_table(
        "model_versions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("experiment_id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=60), nullable=False),
        sa.Column("version", sa.String(length=20), nullable=False),
        sa.Column("onnx_sha256", sa.String(length=64), nullable=False),
        sa.Column("onnx_size_bytes", sa.Integer(), nullable=False),
        sa.Column("quantized", sa.Boolean(), nullable=False),
        sa.Column("threshold", sa.Double(), nullable=True),
        sa.Column("threshold_policy", sa.String(length=20), nullable=True),
        sa.Column("pixel_threshold", sa.Double(), nullable=True),
        sa.Column("reference_scores", sa.JSON(), nullable=True),
        sa.Column("weights_license", sa.String(length=40), nullable=False),
        sa.Column("release_url", sa.String(length=300), nullable=True),
        sa.Column("status", sa.String(length=12), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "status IN ('candidate', 'production', 'archived')",
            name=op.f("ck_model_versions_status"),
        ),
        sa.ForeignKeyConstraint(
            ["experiment_id"],
            ["experiments.id"],
            name=op.f("fk_model_versions_experiment_id_experiments"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_model_versions")),
        sa.UniqueConstraint("name", "version", name="uq_model_versions_name_version"),
        sa.UniqueConstraint("onnx_sha256", name=op.f("uq_model_versions_onnx_sha256")),
    )
    with op.batch_alter_table("model_versions", schema=None) as batch_op:
        batch_op.create_index(
            "uq_model_versions_production",
            ["name"],
            unique=True,
            sqlite_where=sa.text("status = 'production'"),
            postgresql_where=sa.text("status = 'production'"),
        )

    op.create_table(
        "quality_alerts",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("roll_id", sa.Integer(), nullable=False),
        sa.Column("level", sa.String(length=10), nullable=False),
        sa.Column("inspected_length_m", sa.Numeric(precision=8, scale=2), nullable=False),
        sa.Column("width_cm", sa.Numeric(precision=6, scale=1), nullable=False),
        sa.Column("total_points", sa.Integer(), nullable=False),
        sa.Column("points_per_100_sq_yd", sa.Numeric(precision=8, scale=2), nullable=False),
        sa.Column("max_points_allowed", sa.Numeric(precision=8, scale=2), nullable=False),
        sa.Column("acknowledged", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "level IN ('WARNING', 'REJECTED')", name=op.f("ck_quality_alerts_level")
        ),
        sa.CheckConstraint("total_points >= 0", name=op.f("ck_quality_alerts_points")),
        sa.ForeignKeyConstraint(
            ["roll_id"], ["rolls.id"], name=op.f("fk_quality_alerts_roll_id_rolls")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_quality_alerts")),
    )
    with op.batch_alter_table("quality_alerts", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_quality_alerts_roll_id"), ["roll_id"], unique=False)

    op.create_table(
        "drift_reports",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("model_version_id", sa.Integer(), nullable=False),
        sa.Column("window_start", sa.DateTime(timezone=True), nullable=True),
        sa.Column("window_end", sa.DateTime(timezone=True), nullable=True),
        sa.Column("n_predictions", sa.Integer(), nullable=False),
        sa.Column("ks_statistic", sa.Double(), nullable=True),
        sa.Column("p_value", sa.Double(), nullable=True),
        sa.Column("mean_score", sa.Double(), nullable=True),
        sa.Column("alert_rate", sa.Double(), nullable=True),
        sa.Column("max_alert_rate", sa.Double(), nullable=False),
        sa.Column("drift_detected", sa.Boolean(), nullable=False),
        sa.Column("reason", sa.String(length=120), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["model_version_id"],
            ["model_versions.id"],
            name=op.f("fk_drift_reports_model_version_id_model_versions"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_drift_reports")),
    )
    with op.batch_alter_table("drift_reports", schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f("ix_drift_reports_model_version_id"), ["model_version_id"], unique=False
        )

    op.create_table(
        "labels",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("image_id", sa.Integer(), nullable=False),
        sa.Column("class_code", sa.String(length=30), nullable=False),
        sa.Column("mask_path", sa.String(length=300), nullable=True),
        sa.Column("bbox_x", sa.Integer(), nullable=True),
        sa.Column("bbox_y", sa.Integer(), nullable=True),
        sa.Column("bbox_w", sa.Integer(), nullable=True),
        sa.Column("bbox_h", sa.Integer(), nullable=True),
        sa.Column("area_px", sa.Integer(), nullable=True),
        sa.Column("annotator", sa.String(length=40), nullable=False),
        sa.Column("source", sa.String(length=10), nullable=False),
        sa.CheckConstraint(
            "class_code = 'good' OR (mask_path IS NOT NULL AND bbox_w > 0 AND bbox_h > 0)",
            name=op.f("ck_labels_defect_has_mask"),
        ),
        sa.CheckConstraint("source IN ('dataset', 'manual')", name=op.f("ck_labels_source")),
        sa.ForeignKeyConstraint(
            ["class_code"],
            ["defect_classes.code"],
            name=op.f("fk_labels_class_code_defect_classes"),
        ),
        sa.ForeignKeyConstraint(
            ["image_id"], ["images.id"], name=op.f("fk_labels_image_id_images")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_labels")),
        sa.UniqueConstraint("image_id", name=op.f("uq_labels_image_id")),
    )
    op.create_table(
        "predictions",
        sa.Column("id", sa.BigInteger().with_variant(sa.Integer(), "sqlite"), nullable=False),
        sa.Column("model_version_id", sa.Integer(), nullable=False),
        sa.Column("classifier_version_id", sa.Integer(), nullable=True),
        sa.Column("roll_id", sa.Integer(), nullable=True),
        sa.Column("position_m", sa.Numeric(precision=8, scale=2), nullable=True),
        sa.Column("image_sha256", sa.String(length=64), nullable=False),
        sa.Column("width", sa.Integer(), nullable=False),
        sa.Column("height", sa.Integer(), nullable=False),
        sa.Column("score", sa.Double(), nullable=False),
        sa.Column("threshold", sa.Double(), nullable=False),
        sa.Column("is_defective", sa.Boolean(), nullable=False),
        sa.Column("predicted_class", sa.String(length=30), nullable=True),
        sa.Column("class_confidence", sa.Double(), nullable=True),
        sa.Column("latency_ms", sa.Double(), nullable=False),
        sa.Column("source", sa.String(length=10), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "source IN ('api', 'batch', 'seed')", name=op.f("ck_predictions_source")
        ),
        sa.CheckConstraint("latency_ms >= 0", name=op.f("ck_predictions_latency")),
        sa.CheckConstraint(
            "position_m IS NULL OR position_m >= 0", name=op.f("ck_predictions_position")
        ),
        sa.ForeignKeyConstraint(
            ["classifier_version_id"],
            ["model_versions.id"],
            name=op.f("fk_predictions_classifier_version_id_model_versions"),
        ),
        sa.ForeignKeyConstraint(
            ["model_version_id"],
            ["model_versions.id"],
            name=op.f("fk_predictions_model_version_id_model_versions"),
        ),
        sa.ForeignKeyConstraint(
            ["predicted_class"],
            ["defect_classes.code"],
            name=op.f("fk_predictions_predicted_class_defect_classes"),
        ),
        sa.ForeignKeyConstraint(
            ["roll_id"], ["rolls.id"], name=op.f("fk_predictions_roll_id_rolls")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_predictions")),
    )
    with op.batch_alter_table("predictions", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_predictions_created_at"), ["created_at"], unique=False)
        batch_op.create_index(
            batch_op.f("ix_predictions_model_version_id"), ["model_version_id"], unique=False
        )
        batch_op.create_index(batch_op.f("ix_predictions_roll_id"), ["roll_id"], unique=False)

    op.create_table(
        "detected_defects",
        sa.Column("id", sa.BigInteger().with_variant(sa.Integer(), "sqlite"), nullable=False),
        sa.Column(
            "prediction_id", sa.BigInteger().with_variant(sa.Integer(), "sqlite"), nullable=False
        ),
        sa.Column("class_code", sa.String(length=30), nullable=False),
        sa.Column("bbox_x", sa.Integer(), nullable=False),
        sa.Column("bbox_y", sa.Integer(), nullable=False),
        sa.Column("bbox_w", sa.Integer(), nullable=False),
        sa.Column("bbox_h", sa.Integer(), nullable=False),
        sa.Column("length_mm", sa.Numeric(precision=8, scale=1), nullable=False),
        sa.Column("is_hole", sa.Boolean(), nullable=False),
        sa.Column("points", sa.SmallInteger(), nullable=False),
        sa.CheckConstraint("length_mm > 0", name=op.f("ck_detected_defects_positive_length")),
        sa.CheckConstraint("points BETWEEN 1 AND 4", name=op.f("ck_detected_defects_points")),
        sa.ForeignKeyConstraint(
            ["class_code"],
            ["defect_classes.code"],
            name=op.f("fk_detected_defects_class_code_defect_classes"),
        ),
        sa.ForeignKeyConstraint(
            ["prediction_id"],
            ["predictions.id"],
            name=op.f("fk_detected_defects_prediction_id_predictions"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_detected_defects")),
    )
    with op.batch_alter_table("detected_defects", schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f("ix_detected_defects_prediction_id"), ["prediction_id"], unique=False
        )


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table("detected_defects", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_detected_defects_prediction_id"))

    op.drop_table("detected_defects")
    with op.batch_alter_table("predictions", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_predictions_roll_id"))
        batch_op.drop_index(batch_op.f("ix_predictions_model_version_id"))
        batch_op.drop_index(batch_op.f("ix_predictions_created_at"))

    op.drop_table("predictions")
    op.drop_table("labels")
    with op.batch_alter_table("drift_reports", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_drift_reports_model_version_id"))

    op.drop_table("drift_reports")
    with op.batch_alter_table("quality_alerts", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_quality_alerts_roll_id"))

    op.drop_table("quality_alerts")
    with op.batch_alter_table("model_versions", schema=None) as batch_op:
        batch_op.drop_index(
            "uq_model_versions_production",
            sqlite_where=sa.text("status = 'production'"),
            postgresql_where=sa.text("status = 'production'"),
        )

    op.drop_table("model_versions")
    with op.batch_alter_table("images", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_images_split"))

    op.drop_table("images")
    with op.batch_alter_table("rolls", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_rolls_lot_code"))

    op.drop_table("rolls")
    with op.batch_alter_table("lot_inspections", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_lot_inspections_lot_code"))

    op.drop_table("lot_inspections")
    op.drop_table("experiments")
    op.drop_table("defect_classes")
    op.drop_table("datasets")
