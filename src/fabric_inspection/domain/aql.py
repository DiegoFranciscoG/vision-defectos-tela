"""Single sampling plans for normal inspection indexed by AQL (ISO 2859-1).

Table I (sample size code letters) and Table II-A (n, Ac, Re) were checked against the public-domain
MIL-STD-105E scan, origin of the ISO 2859-1 tables (docs/investigacion.md #12, #13).

Table II-A is diagonal: for every AQL there is a "zero row" whose plan is Ac=0/Re=1. Moving one row
down (larger sample) the sequence of cells is always 0/1, up-arrow, down-arrow, 1/2, 2/3, 3/4, 5/6,
7/8, 10/11, 14/15, 21/22 and then up-arrows. Modelling that pattern avoids transcribing 256 cells
by hand and is tested against cells read from the published table.
"""

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum

STANDARD_EDITION = "ISO 2859-1 Table 2-A (values of MIL-STD-105E Table II-A)"
CODE_LETTERS = "ABCDEFGHJKLMNPQR"
SAMPLE_SIZES = (2, 3, 5, 8, 13, 20, 32, 50, 80, 125, 200, 315, 500, 800, 1250, 2000)
# Inclusive upper bound of each lot size range of Table I (the last one is open).
LOT_UPPER_BOUNDS = (8, 15, 25, 50, 90, 150, 280, 500, 1200, 3200, 10_000, 35_000, 150_000, 500_000)

# Preferred AQLs of Table II-A in column order; the zero row of the AQL at index i is 14 - i.
AQL_VALUES = tuple(
    Decimal(value)
    for value in (
        "0.010", "0.015", "0.025", "0.040", "0.065", "0.10", "0.15", "0.25",
        "0.40", "0.65", "1.0", "1.5", "2.5", "4.0", "6.5",
    )
)  # fmt: skip
# Only AQLs whose plans fit completely inside rows A..R are supported (0.10 to 6.5).
_MIN_SUPPORTED_AQL_INDEX = 5
# Ac/Re by distance to the zero row (d = 0 and d = 3..10; d = 1 and 2 are arrows).
_ACCEPT_REJECT: dict[int, tuple[int, int]] = {
    0: (0, 1), 3: (1, 2), 4: (2, 3), 5: (3, 4), 6: (5, 6),
    7: (7, 8), 8: (10, 11), 9: (14, 15), 10: (21, 22),
}  # fmt: skip


class InspectionLevel(StrEnum):
    S1 = "S-1"
    S2 = "S-2"
    S3 = "S-3"
    S4 = "S-4"
    I = "I"  # noqa: E741 - official name of the level
    II = "II"
    III = "III"


# Code letters per lot size range (Table I), one character per range.
_LEVEL_LETTERS: dict[InspectionLevel, str] = {
    InspectionLevel.S1: "AAAABBBBCCCCDDD",
    InspectionLevel.S2: "AAABBBCCCDDDEEE",
    InspectionLevel.S3: "AABBCCDDEEFFGGH",
    InspectionLevel.S4: "AABCCDEEFGGHJJK",
    InspectionLevel.I: "AABCCDEFGHJKLMN",
    InspectionLevel.II: "ABCDEFGHJKLMNPQ",
    InspectionLevel.III: "BCDEFGHJKLMNPQR",
}


@dataclass(frozen=True, slots=True)
class SamplingPlan:
    initial_letter: str
    plan_letter: str
    sample_size: int
    accept_number: int
    reject_number: int
    full_inspection: bool

    def accepts(self, defectives_found: int) -> bool:
        if defectives_found < 0:
            raise ValueError("The number of defective units cannot be negative")
        return defectives_found <= self.accept_number


def supported_aqls() -> tuple[Decimal, ...]:
    return AQL_VALUES[_MIN_SUPPORTED_AQL_INDEX:]


def code_letter(lot_size: int, level: InspectionLevel) -> str:
    if lot_size < 2:
        raise ValueError("Lot size must be at least 2")
    lot_range = next(
        (index for index, bound in enumerate(LOT_UPPER_BOUNDS) if lot_size <= bound),
        len(LOT_UPPER_BOUNDS),
    )
    return _LEVEL_LETTERS[level][lot_range]


def _aql_index(aql: Decimal) -> int:
    for index in range(_MIN_SUPPORTED_AQL_INDEX, len(AQL_VALUES)):
        if AQL_VALUES[index] == aql:
            return index
    valid = ", ".join(str(value) for value in supported_aqls())
    raise ValueError(f"Unsupported AQL {aql}. Valid values: {valid}")


def sampling_plan(lot_size: int, level: InspectionLevel, aql: Decimal) -> SamplingPlan:
    zero_row = 14 - _aql_index(aql)
    initial = code_letter(lot_size, level)
    row = CODE_LETTERS.index(initial)
    distance = row - zero_row
    if distance < 0 or distance == 1:
        plan_row = zero_row  # down-arrow to the first plan, or up-arrow to the plan above
    elif distance == 2:
        plan_row = zero_row + 3  # down-arrow to the 1/2 plan
    elif distance > 10:
        plan_row = zero_row + 10  # up-arrow to the 21/22 plan
    else:
        plan_row = row
    accept, reject = _ACCEPT_REJECT[plan_row - zero_row]
    sample_size = SAMPLE_SIZES[plan_row]
    full_inspection = sample_size >= lot_size
    return SamplingPlan(
        initial_letter=initial,
        plan_letter=CODE_LETTERS[plan_row],
        sample_size=lot_size if full_inspection else sample_size,
        accept_number=accept,
        reject_number=reject,
        full_inspection=full_inspection,
    )
