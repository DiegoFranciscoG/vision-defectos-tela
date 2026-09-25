import pytest

from fabric_inspection.domain.catalog import TEXTRACK_CODES
from fabric_inspection.domain.four_point import (
    RollDecision,
    RollDefect,
    assess_roll,
    points_for_defect,
    points_per_100_sq_yd,
    to_textrack_payload,
)


@pytest.mark.parametrize(
    ("length_mm", "expected"),
    [
        (1.0, 1),
        (76.2, 1),  # exactly 3 in
        (76.3, 2),
        (152.4, 2),  # exactly 6 in
        (152.5, 3),
        (228.6, 3),  # exactly 9 in
        (228.7, 4),
        (5000.0, 4),
    ],
)
def test_points_by_length_band(length_mm: float, expected: int) -> None:
    assert points_for_defect(length_mm, is_hole=False) == expected


def test_hole_is_always_four_points() -> None:
    assert points_for_defect(3.0, is_hole=True) == 4


@pytest.mark.parametrize("length_mm", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_length_is_rejected(length_mm: float) -> None:
    with pytest.raises(ValueError, match="positive"):
        points_for_defect(length_mm, is_hole=False)


def test_points_per_100_sq_yd_formula() -> None:
    # 100 yd long x 36 in wide = 100 yd²: the score equals the raw points.
    assert points_per_100_sq_yd(40, 100 * 0.9144, 36 * 2.54) == pytest.approx(40.0)


def test_roll_is_accepted_warned_or_rejected_against_the_limit() -> None:
    width_cm, length_m = 150.0, 91.44  # 100 yd long, 59.06 in wide
    one_defect = [RollDefect(10.0, 50.0, is_hole=False, class_code="color")]
    accepted = assess_roll(
        one_defect, inspected_length_m=length_m, width_cm=width_cm, max_points_per_100_sq_yd=40
    )
    assert accepted.decision is RollDecision.ACCEPTED
    assert accepted.total_points == 1

    many = [RollDefect(float(i), 10.0, is_hole=True, class_code="hole") for i in range(0, 90, 2)]
    rejected = assess_roll(
        many, inspected_length_m=length_m, width_cm=width_cm, max_points_per_100_sq_yd=40
    )
    assert rejected.decision is RollDecision.REJECTED
    assert rejected.points_per_100_sq_yd > 40

    warning = assess_roll(
        many[:14], inspected_length_m=length_m, width_cm=width_cm, max_points_per_100_sq_yd=40
    )
    assert warning.decision is RollDecision.WARNING


def test_points_are_capped_at_four_per_linear_yard() -> None:
    same_yard = [
        RollDefect(0.10, 10.0, is_hole=True, class_code="hole"),
        RollDefect(0.50, 10.0, is_hole=True, class_code="hole"),
        RollDefect(0.80, 200.0, is_hole=False, class_code="cut"),
    ]
    result = assess_roll(
        same_yard, inspected_length_m=50, width_cm=150, max_points_per_100_sq_yd=40
    )
    assert result.raw_points == 11
    assert result.total_points == 4


def test_defect_outside_the_roll_is_rejected() -> None:
    with pytest.raises(ValueError, match="outside"):
        assess_roll(
            [RollDefect(60.0, 10.0, is_hole=False, class_code="cut")],
            inspected_length_m=50,
            width_cm=150,
            max_points_per_100_sq_yd=40,
        )


def test_textrack_payload_matches_its_contract() -> None:
    assessment = assess_roll(
        [
            RollDefect(12.345, 30.2, is_hole=False, class_code="cut"),
            RollDefect(3.0, 0.4, is_hole=True, class_code="hole"),
        ],
        inspected_length_m=80,
        width_cm=150,
        max_points_per_100_sq_yd=40,
    )
    payload = to_textrack_payload(assessment, TEXTRACK_CODES)
    assert payload["inspectedLengthM"] == 80
    assert payload["widthCm"] == 150
    assert payload["defects"] == [
        {"positionM": 3.0, "lengthMm": 1, "hole": True, "defectTypeCode": "FABRIC_HOLE"},
        {"positionM": 12.35, "lengthMm": 31, "hole": False, "defectTypeCode": None},
    ]
