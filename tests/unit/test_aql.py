from decimal import Decimal

import pytest

from fabric_inspection.domain.aql import (
    InspectionLevel,
    code_letter,
    sampling_plan,
    supported_aqls,
)


@pytest.mark.parametrize(
    ("lot_size", "level", "letter"),
    [
        (2, InspectionLevel.II, "A"),
        (8, InspectionLevel.II, "A"),
        (9, InspectionLevel.II, "B"),
        (50, InspectionLevel.II, "D"),
        (500, InspectionLevel.II, "H"),
        (501, InspectionLevel.II, "J"),
        (10_000, InspectionLevel.II, "L"),
        (500_001, InspectionLevel.II, "Q"),
        (500_001, InspectionLevel.III, "R"),
        (1200, InspectionLevel.S1, "C"),
        (150, InspectionLevel.S4, "D"),
    ],
)
def test_code_letter_follows_table_i(lot_size: int, level: InspectionLevel, letter: str) -> None:
    assert code_letter(lot_size, level) == letter


# Cells read from MIL-STD-105E Table II-A (single sampling, normal inspection).
@pytest.mark.parametrize(
    ("lot_size", "aql", "letter", "n", "ac", "re"),
    [
        (500, "1.0", "H", 50, 1, 2),  # letter H, AQL 1.0
        (500, "2.5", "H", 50, 3, 4),
        (500, "4.0", "H", 50, 5, 6),
        (500, "6.5", "H", 50, 7, 8),
        (500, "0.65", "J", 80, 1, 2),  # H / 0.65 is a down-arrow to J (n = 80)
        (150, "1.0", "E", 13, 0, 1),  # F / 1.0 is an up-arrow to E (n = 13)
        (280, "1.0", "H", 50, 1, 2),  # G / 1.0 is a down-arrow to H (n = 50)
        (3200, "1.5", "K", 125, 5, 6),
        (35_000, "6.5", "L", 200, 21, 22),  # M / 6.5 is an up-arrow to L
    ],
)
def test_sampling_plan_matches_table_ii_a(
    lot_size: int, aql: str, letter: str, n: int, ac: int, re: int
) -> None:
    plan = sampling_plan(lot_size, InspectionLevel.II, Decimal(aql))
    assert (plan.plan_letter, plan.sample_size, plan.accept_number, plan.reject_number) == (
        letter,
        n,
        ac,
        re,
    )


def test_full_inspection_when_sample_reaches_the_lot_size() -> None:
    plan = sampling_plan(5, InspectionLevel.II, Decimal("2.5"))
    assert plan.full_inspection
    assert plan.sample_size == 5
    assert plan.accepts(0)
    assert not plan.accepts(1)


def test_unsupported_aql_and_lot_size_are_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported AQL"):
        sampling_plan(100, InspectionLevel.II, Decimal("0.010"))
    with pytest.raises(ValueError, match="at least 2"):
        code_letter(1, InspectionLevel.II)
    with pytest.raises(ValueError, match="negative"):
        sampling_plan(100, InspectionLevel.II, Decimal("1.0")).accepts(-1)


def test_supported_aqls_range() -> None:
    values = supported_aqls()
    assert values[0] == Decimal("0.10")
    assert values[-1] == Decimal("6.5")
