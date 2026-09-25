"""Score drift detection: two-sample Kolmogorov-Smirnov test and alert-rate change (rule R8)."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True, slots=True)
class DriftResult:
    n_window: int
    n_reference: int
    ks_statistic: float | None
    p_value: float | None
    window_mean: float | None
    reference_mean: float
    alert_rate: float | None
    reference_alert_rate: float
    drift_detected: bool
    reason: str


def _smoothed_rate(scores: np.ndarray, threshold: float) -> float:
    """Alert rate with Laplace smoothing so an empty reference rate never divides by zero."""
    return float((np.sum(scores >= threshold) + 1) / (scores.size + 2))


def detect_drift(
    window_scores: Sequence[float],
    reference_scores: Sequence[float],
    *,
    threshold: float,
    p_value_threshold: float = 0.01,
    min_samples: int = 30,
    alert_rate_factor: float = 2.0,
) -> DriftResult:
    reference = np.asarray(reference_scores, dtype=np.float64)
    window = np.asarray(window_scores, dtype=np.float64)
    if reference.size < 2:
        raise ValueError("The reference distribution needs at least two scores")
    reference_rate = _smoothed_rate(reference, threshold)
    if window.size < min_samples:
        return DriftResult(
            n_window=int(window.size),
            n_reference=int(reference.size),
            ks_statistic=None,
            p_value=None,
            window_mean=float(window.mean()) if window.size else None,
            reference_mean=float(reference.mean()),
            alert_rate=None,
            reference_alert_rate=reference_rate,
            drift_detected=False,
            reason=f"insufficient_data: {window.size} < {min_samples} predictions",
        )
    test = stats.ks_2samp(window, reference)
    window_rate = _smoothed_rate(window, threshold)
    ratio = window_rate / reference_rate
    reasons = []
    if test.pvalue < p_value_threshold:
        reasons.append(f"ks_p_value<{p_value_threshold}")
    if ratio > alert_rate_factor or ratio < 1 / alert_rate_factor:
        reasons.append(f"alert_rate_changed_x{ratio:.2f}")
    return DriftResult(
        n_window=int(window.size),
        n_reference=int(reference.size),
        ks_statistic=float(test.statistic),
        p_value=float(test.pvalue),
        window_mean=float(window.mean()),
        reference_mean=float(reference.mean()),
        alert_rate=window_rate,
        reference_alert_rate=reference_rate,
        drift_detected=bool(reasons),
        reason=", ".join(reasons) if reasons else "stable",
    )
