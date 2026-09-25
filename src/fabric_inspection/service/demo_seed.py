"""Fictitious demo data: rolls, simulated inspection frames, alerts and lot decisions.

Nothing here is real: roll codes, fabrics and detections are generated with a fixed seed so the
demo always looks the same. Scores are drawn from the detector's own reference distribution, so
the monitoring page shows a stable process until real uploads arrive.
"""

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import numpy as np
from sqlalchemy.orm import Session

from fabric_inspection.config import QualitySettings
from fabric_inspection.db.models import DetectedDefect, Prediction, Roll
from fabric_inspection.domain.aql import InspectionLevel
from fabric_inspection.domain.catalog import get_class
from fabric_inspection.domain.four_point import points_for_defect
from fabric_inspection.registry import ModelRegistry
from fabric_inspection.repository.repositories import PredictionRepository, RollRepository
from fabric_inspection.service.model_service import register_models
from fabric_inspection.service.quality_service import QualityService

logger = logging.getLogger(__name__)

DEMO_AQL = Decimal("2.5")
FABRICS = ("Denim 12 oz", "Popelina 100 % algodón", "Jersey 30/1", "Gabardina stretch")
# Typical defect sizes in millimetres (min, max) per class, used only to generate demo data.
DEFECT_SIZES_MM = {
    "color": (8.0, 60.0),
    "cut": (20.0, 180.0),
    "hole": (3.0, 25.0),
    "metal_contamination": (2.0, 15.0),
    "thread": (30.0, 260.0),
}
DEFECT_WEIGHTS = (0.3, 0.15, 0.15, 0.1, 0.3)


@dataclass(frozen=True, slots=True)
class SeedSummary:
    rolls: int
    predictions: int
    alerts: int
    lots: dict[str, str]


def seed_demo(
    session: Session,
    registry: ModelRegistry,
    settings: QualitySettings,
    *,
    seed: int = 2026,
    frames_per_roll: int = 40,
) -> SeedSummary | None:
    """Insert demo data once; returns None when rolls already exist (idempotent)."""
    rolls_repo = RollRepository(session)
    if rolls_repo.list_all():
        logger.info("Demo data already present; nothing to do")
        return None
    served = register_models(session, registry)
    detector = registry.detector
    if detector.threshold is None or not detector.reference_scores:
        raise ValueError("The detector in the registry has no threshold or reference scores")
    threshold = detector.threshold
    normal_pool = np.array([s for s in detector.reference_scores if s < threshold])
    rng = np.random.default_rng(seed)
    predictions = PredictionRepository(session)
    start = datetime.now(UTC) - timedelta(days=7)
    total_frames = 10 * frames_per_roll
    frame_index = 0

    for number in range(1, 11):
        lot = "LT-2026-01" if number <= 5 else "LT-2026-02"
        defect_rate = 0.04 if lot == "LT-2026-01" else 0.12  # lot 2 comes from a worse supplier
        roll = rolls_repo.add(
            Roll(
                code=f"RL-2026-{number:04d}",
                lot_code=lot,
                fabric_type=FABRICS[number % len(FABRICS)],
                width_cm=Decimal("150.0") if number % 3 else Decimal("160.0"),
                length_m=Decimal(int(rng.integers(50, 101))),
            )
        )
        positions = np.sort(rng.uniform(0, float(roll.length_m), frames_per_roll))
        for position in positions:
            created = start + timedelta(minutes=int(frame_index * 7 * 24 * 60 / total_frames))
            frame_index += 1
            defective = bool(rng.random() < defect_rate)
            prediction = Prediction(
                model_version_id=served.detector.id,
                classifier_version_id=served.classifier.id,
                roll_id=roll.id,
                position_m=Decimal(f"{position:.2f}"),
                image_sha256=f"{rng.integers(0, 2**63):016x}".rjust(64, "0"),
                width=1024,
                height=1024,
                score=float(threshold * rng.uniform(1.1, 2.5))
                if defective
                else float(rng.choice(normal_pool)),
                threshold=threshold,
                is_defective=defective,
                latency_ms=float(rng.normal(180, 25)),
                source="seed",
                created_at=created,
            )
            if defective:
                code = str(rng.choice(list(DEFECT_SIZES_MM), p=DEFECT_WEIGHTS))
                low, high = DEFECT_SIZES_MM[code]
                length = round(float(rng.uniform(low, high)), 1)
                is_hole = get_class(code).is_hole
                prediction.predicted_class = code
                prediction.class_confidence = round(float(rng.uniform(0.55, 0.99)), 3)
                side = max(1, round(length / settings.mm_per_pixel))
                prediction.defects.append(
                    DetectedDefect(
                        class_code=code,
                        bbox_x=int(rng.integers(0, 512)),
                        bbox_y=int(rng.integers(0, 512)),
                        bbox_w=min(side, 1024),
                        bbox_h=max(1, min(side // 3, 1024)),
                        length_mm=Decimal(str(length)),
                        is_hole=is_hole,
                        points=points_for_defect(length, is_hole=is_hole),
                    )
                )
            predictions.add(prediction)

    quality = QualityService(session, settings)
    alerts = sum(
        quality.record_assessment(roll.code)[1] is not None for roll in rolls_repo.list_all()
    )
    lots = {
        lot: quality.evaluate_lot(lot, aql=DEMO_AQL, level=InspectionLevel.II).inspection.decision
        for lot in ("LT-2026-01", "LT-2026-02")
    }
    session.flush()
    return SeedSummary(rolls=10, predictions=total_frames, alerts=alerts, lots=lots)
