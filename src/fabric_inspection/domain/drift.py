"""Score drift monitoring (rule R8).

Two separate signals:

* **Normal-score shift**: a two-sample Kolmogorov-Smirnov test between the scores of frames that
  look normal (below the threshold) and the scores of good validation images. Real defects are
  excluded on both sides, so more defects in production do not look like drift; a change of
  camera, lighting or fabric does.
* **Alert rate**: share of frames above the threshold. Too many alerts means either a process
  problem or a miscalibrated model; both need a person to look at it.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True, slots=True)
class DriftResult:
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


def detect_drift(
    window_scores: Sequence[float],
    reference_scores: Sequence[float],
    *,
    threshold: float,
    p_value_threshold: float = 0.01,
    min_samples: int = 30,
    max_alert_rate: float = 0.25,
) -> DriftResult:
    reference = np.asarray(reference_scores, dtype=np.float64)
    reference = reference[reference < threshold]
    if reference.size < 2:
        raise ValueError("The reference distribution needs at least two scores below the threshold")
    window = np.asarray(window_scores, dtype=np.float64)
    normal = window[window < threshold]
    alert_rate = float(np.mean(window >= threshold)) if window.size else None

    def result(
        ks_statistic: float | None, p_value: float | None, drift: bool, reason: str
    ) -> DriftResult:
        return DriftResult(
            n_window=int(window.size),
            n_window_normal=int(normal.size),
            n_reference=int(reference.size),
            ks_statistic=ks_statistic,
            p_value=p_value,
            window_mean=float(normal.mean()) if normal.size else None,
            reference_mean=float(reference.mean()),
            alert_rate=alert_rate,
            max_alert_rate=max_alert_rate,
            drift_detected=drift,
            reason=reason,
        )

    if normal.size < min_samples:
        return result(None, None, False, f"insufficient_data: {normal.size} < {min_samples}")
    test = stats.ks_2samp(normal, reference)
    reasons = []
    if test.pvalue < p_value_threshold:
        reasons.append("normal_score_shift")
    if alert_rate is not None and alert_rate > max_alert_rate:
        reasons.append("alert_rate_above_limit")
    return result(
        float(test.statistic),
        float(test.pvalue),
        bool(reasons),
        ", ".join(reasons) if reasons else "stable",
    )
