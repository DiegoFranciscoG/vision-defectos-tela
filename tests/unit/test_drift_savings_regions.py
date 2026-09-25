import numpy as np
import pytest

from fabric_inspection.domain.drift import detect_drift
from fabric_inspection.domain.regions import extract_regions
from fabric_inspection.domain.savings import SavingsInput, estimate_savings


class TestDrift:
    reference = np.random.default_rng(0).normal(1.0, 0.1, 200).tolist()

    def test_same_distribution_is_stable(self) -> None:
        window = np.random.default_rng(1).normal(1.0, 0.1, 200).tolist()
        result = detect_drift(window, self.reference, threshold=1.3)
        assert not result.drift_detected
        assert result.reason == "stable"
        assert result.p_value is not None
        assert result.p_value > 0.01

    def test_shifted_distribution_is_detected(self) -> None:
        window = np.random.default_rng(2).normal(1.4, 0.1, 200).tolist()
        result = detect_drift(window, self.reference, threshold=1.3)
        assert result.drift_detected
        assert "ks_p_value" in result.reason
        assert "alert_rate_changed" in result.reason

    def test_small_window_is_not_evaluated(self) -> None:
        result = detect_drift([1.0] * 5, self.reference, threshold=1.3)
        assert not result.drift_detected
        assert result.reason.startswith("insufficient_data")
        assert result.ks_statistic is None

    def test_reference_needs_two_scores(self) -> None:
        with pytest.raises(ValueError, match="at least two"):
            detect_drift([1.0] * 50, [1.0], threshold=1.0)


class TestSavings:
    def test_default_estimate_is_consistent(self) -> None:
        result = estimate_savings(SavingsInput())
        assert result.manual_hours == pytest.approx(55.6, abs=0.1)
        assert result.automated_hours == pytest.approx(27.8, abs=0.1)
        assert result.defects_per_month == 1000
        assert result.escaped_defects_manual == 300
        assert result.escaped_defects_automated == 100
        assert result.escaped_meters_avoided == 200
        assert result.quality_saving_usd == pytest.approx(200 * 4.0 * 0.45)
        assert result.net_saving_usd == pytest.approx(
            result.labor_saving_usd + result.quality_saving_usd - result.review_cost_usd, abs=0.02
        )

    def test_worse_model_gives_negative_quality_saving(self) -> None:
        result = estimate_savings(SavingsInput(model_recall=0.5))
        assert result.quality_saving_usd < 0

    @pytest.mark.parametrize(
        "kwargs",
        [{"manual_recall": 1.5}, {"meters_per_month": 0}, {"price_per_meter_usd": -1}],
    )
    def test_invalid_input(self, kwargs: dict[str, float]) -> None:
        with pytest.raises(ValueError, match=next(iter(kwargs))):
            SavingsInput(**kwargs)


class TestRegions:
    def test_regions_are_scaled_to_the_original_image(self) -> None:
        anomaly = np.zeros((64, 64), dtype=np.float32)
        anomaly[10:20, 30:34] = 2.0  # 10 x 4 blob
        anomaly[50:52, 5:7] = 1.5  # 2 x 2 blob
        regions = extract_regions(anomaly, 1.0, original_size=(1024, 1024))
        assert len(regions) == 2
        top = regions[0]
        assert (top.x, top.y, top.width, top.height) == (480, 160, 64, 160)
        assert top.peak_score == 2.0
        assert top.length_mm(0.1) == 16.0

    def test_tiny_components_fall_back_to_the_peak(self) -> None:
        anomaly = np.zeros((32, 32), dtype=np.float32)
        anomaly[3, 3] = 5.0
        regions = extract_regions(anomaly, 1.0, original_size=(32, 32), min_area_px=4)
        assert len(regions) == 1
        assert regions[0].peak_score == 5.0

    def test_no_region_below_threshold(self) -> None:
        assert extract_regions(np.zeros((8, 8)), 1.0, original_size=(8, 8)) == []

    def test_rejects_non_2d_maps(self) -> None:
        with pytest.raises(ValueError, match="2-D"):
            extract_regions(np.zeros((1, 8, 8)), 1.0, original_size=(8, 8))
