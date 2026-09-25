"""ASTM D5430 four-point system for fabric rolls.

Point allocation and the points-per-100-square-yards formula come from Das Gupta et al. (2024),
Heliyon 10(17) e35931, which applies ASTM D5430 (docs/investigacion.md #10, #11). The cap of four
points per linear yard (S2) and the default acceptance limit (S3) are documented assumptions.
"""

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

MM_PER_INCH = 25.4
CM_PER_INCH = 2.54
METERS_PER_YARD = 0.9144
MAX_POINTS_PER_LINEAR_YARD = 4
SQ_INCHES_PER_100_SQ_YD = 36 * 100  # 100 yd² expressed with length in yards and width in inches
TEXTRACK_MAX_LENGTH_MM = 10_000
_TOLERANCE_MM = 1e-6

# Upper bound (inclusive) of each point band, in millimetres: 3, 6 and 9 inches (S4).
_POINT_BANDS_MM: tuple[tuple[float, int], ...] = (
    (3 * MM_PER_INCH, 1),
    (6 * MM_PER_INCH, 2),
    (9 * MM_PER_INCH, 3),
)


class RollDecision(StrEnum):
    ACCEPTED = "ACCEPTED"
    WARNING = "WARNING"
    REJECTED = "REJECTED"


def points_for_defect(length_mm: float, *, is_hole: bool) -> int:
    """Penalty points of a single defect: 1-4 by length, 4 for any hole (S5)."""
    if not math.isfinite(length_mm) or length_mm <= 0:
        raise ValueError("Defect length must be a positive number of millimetres")
    if is_hole:
        return 4
    for upper_mm, points in _POINT_BANDS_MM:
        if length_mm <= upper_mm + _TOLERANCE_MM:
            return points
    return 4


@dataclass(frozen=True, slots=True)
class RollDefect:
    position_m: float
    length_mm: float
    is_hole: bool
    class_code: str


@dataclass(frozen=True, slots=True)
class ScoredDefect:
    defect: RollDefect
    points: int


@dataclass(frozen=True, slots=True)
class RollAssessment:
    inspected_length_m: float
    width_cm: float
    raw_points: int
    total_points: int
    points_per_100_sq_yd: float
    max_points_allowed: float
    decision: RollDecision
    defects: tuple[ScoredDefect, ...]


def points_per_100_sq_yd(total_points: int, inspected_length_m: float, width_cm: float) -> float:
    """points x 36 x 100 / (length in yards x width in inches)."""
    if inspected_length_m <= 0 or width_cm <= 0:
        raise ValueError("Inspected length and width must be positive")
    length_yd = inspected_length_m / METERS_PER_YARD
    width_in = width_cm / CM_PER_INCH
    return total_points * SQ_INCHES_PER_100_SQ_YD / (length_yd * width_in)


def assess_roll(
    defects: Sequence[RollDefect],
    *,
    inspected_length_m: float,
    width_cm: float,
    max_points_per_100_sq_yd: float,
    warning_ratio: float = 0.8,
) -> RollAssessment:
    """Grade a roll: points per defect, cap per linear yard, normalised score and decision."""
    if not 0 < warning_ratio < 1:
        raise ValueError("warning_ratio must be between 0 and 1")
    if max_points_per_100_sq_yd <= 0:
        raise ValueError("max_points_per_100_sq_yd must be positive")
    scored: list[ScoredDefect] = []
    points_by_yard: dict[int, int] = defaultdict(int)
    for defect in defects:
        if not 0 <= defect.position_m <= inspected_length_m:
            raise ValueError(
                f"Defect position {defect.position_m} m is outside the inspected length"
            )
        points = points_for_defect(defect.length_mm, is_hole=defect.is_hole)
        scored.append(ScoredDefect(defect=defect, points=points))
        points_by_yard[int(defect.position_m // METERS_PER_YARD)] += points

    raw_points = sum(item.points for item in scored)
    total_points = sum(
        min(points, MAX_POINTS_PER_LINEAR_YARD) for points in points_by_yard.values()
    )
    score = points_per_100_sq_yd(total_points, inspected_length_m, width_cm)
    if score > max_points_per_100_sq_yd:
        decision = RollDecision.REJECTED
    elif score >= warning_ratio * max_points_per_100_sq_yd:
        decision = RollDecision.WARNING
    else:
        decision = RollDecision.ACCEPTED
    return RollAssessment(
        inspected_length_m=inspected_length_m,
        width_cm=width_cm,
        raw_points=raw_points,
        total_points=total_points,
        points_per_100_sq_yd=round(score, 2),
        max_points_allowed=max_points_per_100_sq_yd,
        decision=decision,
        defects=tuple(sorted(scored, key=lambda item: item.defect.position_m)),
    )


def to_textrack_payload(
    assessment: RollAssessment, textrack_codes: dict[str, str | None]
) -> dict[str, object]:
    """Body for textrack `POST /fabric-rolls/{id}/inspection` (FabricInspectionRequest).

    textrack validates `defectTypeCode` against its own catalog, so classes without an equivalent
    are sent as null instead of inventing a code.
    """
    return {
        "inspectedLengthM": round(assessment.inspected_length_m, 2),
        "widthCm": round(assessment.width_cm, 1),
        "defects": [
            {
                "positionM": round(item.defect.position_m, 2),
                "lengthMm": min(TEXTRACK_MAX_LENGTH_MM, max(1, math.ceil(item.defect.length_mm))),
                "hole": item.defect.is_hole,
                "defectTypeCode": textrack_codes.get(item.defect.class_code),
            }
            for item in assessment.defects
        ],
    }
