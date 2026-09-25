"""Request and response bodies of the REST API (English field names, validated by Pydantic)."""

from datetime import datetime
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from fabric_inspection.domain.aql import InspectionLevel

CODE_PATTERN = r"^[A-Za-z0-9-]{1,30}$"


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HealthResponse(ApiModel):
    status: Literal["ok", "degraded"]
    database: Literal["ok", "error"]
    detector_version: str | None


class DefectDto(ApiModel):
    class_code: str
    x: int
    y: int
    width: int
    height: int
    length_mm: float
    is_hole: bool
    points: int = Field(ge=1, le=4)


class ModelRef(ApiModel):
    detector: str
    classifier: str


class PredictionResponse(ApiModel):
    id: int
    created_at: datetime
    image_sha256: str
    width: int
    height: int
    score: float
    threshold: float
    score_ratio: float = Field(description="score / threshold; >= 1 means defective")
    is_defective: bool
    predicted_class: str | None
    class_confidence: float | None
    class_probabilities: dict[str, float] | None = None
    defects: list[DefectDto]
    latency_ms: float
    roll_code: str | None
    position_m: float | None
    models: ModelRef | None = None
    heatmap_png_base64: str | None = None
    cam_png_base64: str | None = None


class PredictionPage(ApiModel):
    items: list[PredictionResponse]
    total: int
    limit: int
    offset: int


class ScoredDefectDto(ApiModel):
    position_m: float
    length_mm: float
    is_hole: bool
    class_code: str
    points: int


class RollQualityResponse(ApiModel):
    code: str
    lot_code: str
    fabric_type: str
    width_cm: float
    length_m: float
    raw_points: int
    total_points: int
    points_per_100_sq_yd: float
    max_points_allowed: float
    decision: Literal["ACCEPTED", "WARNING", "REJECTED"]
    defects: list[ScoredDefectDto]
    textrack_payload: dict[str, object] = Field(
        description="Body for textrack POST /fabric-rolls/{id}/inspection"
    )


class RollSummary(ApiModel):
    code: str
    lot_code: str
    fabric_type: str
    width_cm: float
    length_m: float
    defects: int
    total_points: int
    points_per_100_sq_yd: float
    decision: Literal["ACCEPTED", "WARNING", "REJECTED"]


class AlertDto(ApiModel):
    id: int
    roll_code: str
    level: Literal["WARNING", "REJECTED"]
    total_points: int
    points_per_100_sq_yd: float
    max_points_allowed: float
    created_at: datetime


class AssessmentResponse(ApiModel):
    quality: RollQualityResponse
    alert: AlertDto | None


class AqlPlanResponse(ApiModel):
    lot_size: int
    inspection_level: InspectionLevel
    aql: Decimal
    code_letter: str
    plan_letter: str
    sample_size: int
    accept_number: int
    reject_number: int
    full_inspection: bool
    standard: str


class LotInspectionRequest(ApiModel):
    aql: Decimal = Field(default=Decimal("2.5"))
    inspection_level: InspectionLevel = InspectionLevel.II


class LotInspectionResponse(ApiModel):
    lot_code: str
    lot_size: int
    inspection_level: str
    aql: Decimal
    code_letter: str
    sample_size: int
    accept_number: int
    reject_number: int
    rejected_rolls: int
    decision: Literal["ACCEPTED", "REJECTED"]
    sampled_rolls: list[RollSummary]
    standard: str


class DriftResponse(ApiModel):
    detector_version: str
    threshold: float
    n_window: int
    n_window_normal: int
    n_reference: int
    ks_statistic: float | None
    p_value: float | None
    window_mean: float | None
    reference_mean: float
    alert_rate: float | None
    max_alert_rate: float
    drift_detected: bool
    reason: str
    window_scores: list[float]
    reference_scores: list[float]


class DriftReportResponse(ApiModel):
    id: int
    created_at: datetime
    n_predictions: int
    p_value: float | None
    drift_detected: bool
    reason: str


class ModelInfo(ApiModel):
    name: str
    version: str
    file: str
    sha256: str
    size_bytes: int
    quantized: bool
    threshold: float | None
    threshold_policy: str | None
    mlflow_run_id: str
    test_metrics: dict[str, object]


class ModelsResponse(ApiModel):
    bundle_version: str
    weights_license: str
    image_size: int
    mm_per_pixel: float
    detector: ModelInfo
    classifier: ModelInfo


class SavingsRequest(ApiModel):
    meters_per_month: float = Field(default=50_000, gt=0, le=10_000_000)
    manual_speed_m_min: float = Field(default=15, gt=0, le=200)
    automated_speed_m_min: float = Field(default=30, gt=0, le=500)
    hourly_cost_usd: float = Field(default=2.01, ge=0, le=1000)
    defects_per_100_m: float = Field(default=2.0, ge=0, le=100)
    manual_recall: float = Field(default=0.70, ge=0, le=1)
    model_recall: float | None = Field(
        default=None, ge=0, le=1, description="Defaults to the served detector's test recall"
    )
    meters_lost_per_escaped_defect: float = Field(default=1.0, ge=0, le=100)
    price_per_meter_usd: float = Field(default=4.0, ge=0, le=1000)
    price_reduction: float = Field(default=0.45, ge=0, le=1)
    false_alarms_per_100_m: float | None = Field(
        default=None,
        ge=0,
        le=100,
        description="Defaults to the detector's test false-positive rate",
    )
    review_minutes_per_false_alarm: float = Field(default=0.5, ge=0, le=60)


class SavingsResponse(ApiModel):
    inputs: dict[str, float]
    manual_hours: float
    automated_hours: float
    labor_saving_usd: float
    defects_per_month: float
    escaped_defects_manual: float
    escaped_defects_automated: float
    escaped_meters_manual: float
    escaped_meters_automated: float
    escaped_meters_avoided: float
    quality_saving_usd: float
    false_alarms_per_month: float
    review_cost_usd: float
    net_saving_usd: float
    assumptions: list[str]


class ErrorResponse(ApiModel):
    detail: str
    error_id: str | None = None
