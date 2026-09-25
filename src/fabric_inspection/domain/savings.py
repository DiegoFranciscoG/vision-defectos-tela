"""Monthly savings of automated inspection versus manual inspection.

A transparent, parametric estimate: every input is editable and the defaults point to their source
in docs/investigacion.md (#14 for speeds and price loss; S7-S9 for assumptions).
"""

from dataclasses import dataclass

SBU_2026_USD = 482.0
HOURS_PER_MONTH = 240.0


@dataclass(frozen=True, slots=True)
class SavingsInput:
    meters_per_month: float = 50_000.0
    manual_speed_m_min: float = 15.0  # 8-20 m/min on an inspection table (#14)
    automated_speed_m_min: float = 30.0  # S8
    hourly_cost_usd: float = round(SBU_2026_USD / HOURS_PER_MONTH, 2)  # S9
    defects_per_100_m: float = 2.0
    manual_recall: float = 0.70  # S7
    model_recall: float = 0.90  # replaced by the measured test recall
    meters_lost_per_escaped_defect: float = 1.0
    price_per_meter_usd: float = 4.0
    price_reduction: float = 0.45  # lower bound of the 45-65 % loss reported in #14
    false_alarms_per_100_m: float = 0.5
    review_minutes_per_false_alarm: float = 0.5

    def __post_init__(self) -> None:
        positives = {
            "meters_per_month": self.meters_per_month,
            "manual_speed_m_min": self.manual_speed_m_min,
            "automated_speed_m_min": self.automated_speed_m_min,
        }
        for name, value in positives.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        fractions = {
            "manual_recall": self.manual_recall,
            "model_recall": self.model_recall,
            "price_reduction": self.price_reduction,
        }
        for name, value in fractions.items():
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        non_negative = {
            "hourly_cost_usd": self.hourly_cost_usd,
            "defects_per_100_m": self.defects_per_100_m,
            "meters_lost_per_escaped_defect": self.meters_lost_per_escaped_defect,
            "price_per_meter_usd": self.price_per_meter_usd,
            "false_alarms_per_100_m": self.false_alarms_per_100_m,
            "review_minutes_per_false_alarm": self.review_minutes_per_false_alarm,
        }
        for name, value in non_negative.items():
            if value < 0:
                raise ValueError(f"{name} cannot be negative")


@dataclass(frozen=True, slots=True)
class SavingsEstimate:
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


def estimate_savings(params: SavingsInput) -> SavingsEstimate:
    manual_hours = params.meters_per_month / params.manual_speed_m_min / 60
    automated_hours = params.meters_per_month / params.automated_speed_m_min / 60
    labor_saving = (manual_hours - automated_hours) * params.hourly_cost_usd

    defects = params.meters_per_month / 100 * params.defects_per_100_m
    escaped_manual = defects * (1 - params.manual_recall)
    escaped_automated = defects * (1 - params.model_recall)
    meters_manual = escaped_manual * params.meters_lost_per_escaped_defect
    meters_automated = escaped_automated * params.meters_lost_per_escaped_defect
    loss_per_meter = params.price_per_meter_usd * params.price_reduction
    quality_saving = (meters_manual - meters_automated) * loss_per_meter

    false_alarms = params.meters_per_month / 100 * params.false_alarms_per_100_m
    review_cost = false_alarms * params.review_minutes_per_false_alarm / 60 * params.hourly_cost_usd

    return SavingsEstimate(
        manual_hours=round(manual_hours, 1),
        automated_hours=round(automated_hours, 1),
        labor_saving_usd=round(labor_saving, 2),
        defects_per_month=round(defects, 1),
        escaped_defects_manual=round(escaped_manual, 1),
        escaped_defects_automated=round(escaped_automated, 1),
        escaped_meters_manual=round(meters_manual, 1),
        escaped_meters_automated=round(meters_automated, 1),
        escaped_meters_avoided=round(meters_manual - meters_automated, 1),
        quality_saving_usd=round(quality_saving, 2),
        false_alarms_per_month=round(false_alarms, 1),
        review_cost_usd=round(review_cost, 2),
        net_saving_usd=round(labor_saving + quality_saving - review_cost, 2),
    )
