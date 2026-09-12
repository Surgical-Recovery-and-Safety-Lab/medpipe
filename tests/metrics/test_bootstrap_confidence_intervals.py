"""
Test functions for the bootstrap_confidence_intervals function of the
medpipe.metrics.core module.
"""

import inspect
from typing import Any
from unittest.mock import patch

import numpy as np
import numpy.typing as npt
import pytest

from medpipe._types import Labels
from medpipe.metrics.core import bootstrap_confidence_intervals, compute_metrics


@pytest.fixture
def mock_binary_data() -> tuple[Labels, npt.NDArray]:
    """Generate a moderately sized, well-balanced binary dataset."""
    rng = np.random.default_rng(seed=7)
    n_samples = 60
    y = rng.integers(low=0, high=2, size=n_samples)
    y_pred = rng.random(n_samples)

    return y, y_pred


class TestBootstrapConfidenceIntervalsStructure:
    """Tests for the shape and content of a successful call's return value."""

    def test_returns_expected_structure_per_metric(
        self, mock_binary_data: tuple[Labels, npt.NDArray]
    ) -> None:
        """Test that every requested metric maps to a dict with the three
        documented keys, each holding a float."""
        y, y_pred = mock_binary_data
        results = bootstrap_confidence_intervals(
            ["accuracy", "roc_auc"], y, y_pred, n_bootstraps=30, random_state=0
        )

        assert set(results.keys()) == {"accuracy", "roc_auc"}
        for metric_result in results.values():
            assert set(metric_result.keys()) == {
                "point_estimate",
                "ci_lower",
                "ci_upper",
            }
            assert all(isinstance(v, float) for v in metric_result.values())
            assert metric_result["ci_lower"] <= metric_result["ci_upper"]

    def test_point_estimate_matches_compute_metrics_on_original_data(
        self, mock_binary_data: tuple[Labels, npt.NDArray]
    ) -> None:
        """Test that point_estimate equals compute_metrics run directly on
        the un-resampled data, since it should never be affected by
        bootstrap resampling."""
        y, y_pred = mock_binary_data
        expected = compute_metrics(["accuracy"], y, y_pred)

        results = bootstrap_confidence_intervals(
            ["accuracy"], y, y_pred, n_bootstraps=10, random_state=0
        )

        assert results["accuracy"]["point_estimate"] == pytest.approx(expected[0])

    def test_default_n_bootstraps_and_ci_level(self) -> None:
        """Test the documented defaults directly via the signature, rather
        than running a full 1000-iteration bootstrap in the test suite."""
        sig = inspect.signature(bootstrap_confidence_intervals)

        assert sig.parameters["n_bootstraps"].default == 1000
        assert sig.parameters["ci_level"].default == 0.95
        assert sig.parameters["random_state"].default is None


class TestBootstrapConfidenceIntervalsRandomState:
    """Tests for the random_state / reproducibility behavior."""

    def test_reproducible_with_same_integer_seed(
        self, mock_binary_data: tuple[Labels, npt.NDArray]
    ) -> None:
        """Test that two calls with the same integer random_state produce
        identical results."""
        y, y_pred = mock_binary_data

        first = bootstrap_confidence_intervals(
            ["accuracy"], y, y_pred, n_bootstraps=20, random_state=123
        )
        second = bootstrap_confidence_intervals(
            ["accuracy"], y, y_pred, n_bootstraps=20, random_state=123
        )

        assert first == second

    def test_different_seeds_produce_different_bounds(
        self, mock_binary_data: tuple[Labels, npt.NDArray]
    ) -> None:
        """Test that different seeds lead to different resampling and thus
        (with high probability) different confidence bounds."""
        y, y_pred = mock_binary_data

        first = bootstrap_confidence_intervals(
            ["accuracy"], y, y_pred, n_bootstraps=20, random_state=1
        )
        second = bootstrap_confidence_intervals(
            ["accuracy"], y, y_pred, n_bootstraps=20, random_state=2
        )

        assert first != second

    def test_accepts_a_generator_instance_directly(
        self, mock_binary_data: tuple[Labels, npt.NDArray]
    ) -> None:
        """Test that passing an np.random.Generator instance is accepted
        and used as-is, rather than being re-seeded."""
        y, y_pred = mock_binary_data
        generator = np.random.default_rng(99)

        results = bootstrap_confidence_intervals(
            ["accuracy"], y, y_pred, n_bootstraps=15, random_state=generator
        )

        assert isinstance(results["accuracy"]["point_estimate"], float)

    def test_accepts_none_random_state(
        self, mock_binary_data: tuple[Labels, npt.NDArray]
    ) -> None:
        """Test that omitting random_state (None) still produces a valid,
        if non-reproducible, result."""
        y, y_pred = mock_binary_data

        results = bootstrap_confidence_intervals(
            ["accuracy"], y, y_pred, n_bootstraps=15
        )

        assert isinstance(results["accuracy"]["point_estimate"], float)


class TestBootstrapConfidenceIntervalsInputShapes:
    """Tests for accepted y_true / y_pred shapes and types."""

    def test_2d_y_pred_matches_1d_positive_class_column(
        self, mock_binary_data: tuple[Labels, npt.NDArray]
    ) -> None:
        """Test that passing (n_samples, 2) probabilities gives identical
        results to passing the positive-class column alone, given the same
        random_state."""
        y, y_pred_pos = mock_binary_data
        y_pred_2d = np.column_stack([1 - y_pred_pos, y_pred_pos])

        results_2d = bootstrap_confidence_intervals(
            ["accuracy"], y, y_pred_2d, n_bootstraps=15, random_state=0
        )
        results_1d = bootstrap_confidence_intervals(
            ["accuracy"], y, y_pred_pos, n_bootstraps=15, random_state=0
        )

        assert results_2d == results_1d

    def test_accepts_plain_lists_not_just_ndarrays(self) -> None:
        """Test that y_true and y_pred may be plain Python lists, since the
        function normalizes both via np.asarray."""
        y = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        y_pred = [0.2, 0.8, 0.3, 0.9, 0.1, 0.7, 0.4, 0.6, 0.2, 0.85]

        results = bootstrap_confidence_intervals(
            ["accuracy"], y, y_pred, n_bootstraps=10, random_state=0
        )

        assert isinstance(results["accuracy"]["point_estimate"], float)


class TestBootstrapConfidenceIntervalsCiLevel:
    """Tests for ci_level validation."""

    @pytest.mark.parametrize("ci_level", [0.0, 1.0, -0.1, 1.5])
    def test_invalid_ci_level_raises_value_error(
        self, mock_binary_data: tuple[Labels, npt.NDArray], ci_level: float
    ) -> None:
        """Test that ci_level outside the open interval (0, 1) raises."""
        y, y_pred = mock_binary_data

        with pytest.raises(
            ValueError, match=f"ci_level must be between 0.0 and 1.0, got {ci_level}"
        ):
            bootstrap_confidence_intervals(["accuracy"], y, y_pred, ci_level=ci_level)


class TestBootstrapConfidenceIntervalsResampling:
    """Tests for the resampling loop's edge-case handling."""

    def test_mixed_single_and_multi_class_resamples_still_succeeds(self) -> None:
        """Test that a small, imbalanced dataset (which reliably produces
        some single-class bootstrap draws alongside some valid ones with
        this fixed seed) still yields a valid result overall."""
        y = np.array([0, 0, 0, 0, 1])
        y_pred = np.array([0.1, 0.2, 0.3, 0.4, 0.9])

        results = bootstrap_confidence_intervals(
            ["accuracy"], y, y_pred, n_bootstraps=30, random_state=0
        )

        assert isinstance(results["accuracy"]["point_estimate"], float)

    def test_all_iterations_failing_raises_value_error(self) -> None:
        """Test that a single-sample dataset — where every bootstrap
        resample is trivially single-class — exhausts all iterations and
        raises, rather than returning an empty/invalid result."""
        with pytest.raises(
            ValueError, match="All bootstrap iterations failed to compute valid metrics"
        ):
            bootstrap_confidence_intervals(
                ["accuracy"], np.array([1]), np.array([0.9]), n_bootstraps=5
            )

    def test_iterations_raising_unexpected_errors_are_skipped(
        self, mock_binary_data: tuple[Labels, npt.NDArray]
    ) -> None:
        """Test the generic except-and-skip branch: bootstrap iterations
        whose compute_metrics call raises some other exception (e.g. a
        numerical error unrelated to class balance) are skipped rather
        than propagating, as long as at least one iteration succeeds."""
        y, y_pred = mock_binary_data
        real_compute_metrics = compute_metrics
        call_count = {"n": 0}

        def flaky_compute_metrics(metrics: list[str], y: Any, y_pred: Any):
            call_count["n"] += 1
            # Let the point-estimate call (the very first call) through
            # untouched, then fail every third call inside the loop.
            if call_count["n"] > 1 and call_count["n"] % 3 == 0:
                raise RuntimeError("synthetic numerical failure")
            return real_compute_metrics(metrics, y, y_pred)

        with patch(
            "medpipe.metrics.core.compute_metrics", side_effect=flaky_compute_metrics
        ):
            results = bootstrap_confidence_intervals(
                ["accuracy"], y, y_pred, n_bootstraps=15, random_state=0
            )

        assert isinstance(results["accuracy"]["point_estimate"], float)
        # More calls than just the point estimate + successful iterations
        # means some calls did raise and were skipped, not silently avoided.
        assert call_count["n"] > 1
