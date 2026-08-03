#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test functions for the metrics.core module
"""

from re import escape
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import numpy.typing as npt
import pytest

from medpipe._types import Labels
from medpipe.metrics.core import METRICS, build_scorers, compute_metrics, ici_score
from medpipe.metrics.registry import MetricRegistry, MetricSpec


@pytest.fixture
def mock_data() -> tuple[Labels, npt.NDArray]:
    """Generate some mock labels and predictions for tests."""
    rng = np.random.default_rng(seed=42)
    n_samples = 100
    y = rng.integers(low=0, high=2, size=n_samples)
    y_pred = np.zeros((n_samples, 2))  # Full probabilities
    y_pred[:, 0] = rng.random(100)  # Generate 100 probabilities
    y_pred[:, 1] = 1 - y_pred[:, 0]  # Get positive class probabilities

    return y, y_pred


class TestIciScore:
    """Test class for the ici_score function."""

    def test_ici_score_success(self, mock_data: tuple[Labels, npt.NDArray]) -> None:
        """Test successful function call."""
        y, y_pred = mock_data
        ici = ici_score(y, y_pred)

        assert isinstance(ici, float)

    def test_ici_score_pos_proba(self, mock_data: tuple[Labels, npt.NDArray]) -> None:
        """Test successful function call with positive class probabilities only."""
        y, y_pred = mock_data  # Unpack mock data
        y_pred = y_pred[:, 1]
        ici = ici_score(y, y_pred)

        assert isinstance(ici, float)

    @patch("medpipe.metrics.core.SplineCalib")
    def test_ici_score_spline_prediction_failure_raises(
        self, mock_spline_cls: MagicMock, mock_data: tuple[Labels, npt.NDArray]
    ) -> None:
        """Test ValueError handling when spline prediction fails/returns None."""
        y, y_pred = mock_data
        mock_instance = MagicMock()
        mock_instance.predict.return_value = None
        mock_spline_cls.return_value = mock_instance

        with pytest.raises(
            ValueError, match="Error predicting probabilities with spline"
        ):
            ici_score(y, y_pred[:, 1])


class TestBuildScorers:
    """Test class for the build_scorers function."""

    def test_build_scorers_success(self) -> None:
        """Test successful scorer generation across all registered default metrics."""
        scorers = build_scorers(METRICS)
        assert len(scorers) == len(METRICS)
        for _, scorer in scorers.items():
            assert callable(scorer)

    @pytest.mark.parametrize(
        "metrics",
        [
            3.14,
            42,
            "llama",
            {},
            (),
            [],
            [42],
            [3.14],
        ],
    )
    def test_build_scorers_invalid_metric_type(self, metrics) -> None:
        """Test case when metrics is not a list of strings."""
        with pytest.raises(
            TypeError, match="Input metrics should be a list of strings"
        ):
            build_scorers(metrics)

    def test_build_scorers_invalid_metric(self) -> None:
        """Test case when metrics has invalid value."""
        match_expr = (
            "'invalid' was not found in available metrics. "
            f"Available metrics are {METRICS}"
        )

        with pytest.raises(ValueError, match=escape(match_expr)):
            build_scorers(["invalid"])


class TestComputeMetrics:
    """Test class for the compute_metrics function."""

    def test_compute_metrics_success(
        self, mock_data: tuple[Labels, npt.NDArray]
    ) -> None:
        """Test metric calculation with 2D probabilities."""
        y, y_pred = mock_data
        scores = compute_metrics(METRICS, y, y_pred)
        assert isinstance(scores, np.ndarray)
        assert len(scores) == len(METRICS)

    def test_compute_metrics_success_pos_proba(
        self, mock_data: tuple[Labels, npt.NDArray]
    ) -> None:
        """Test metric calculation with 1D positive class probabilities."""
        y, y_pred = mock_data
        scores = compute_metrics(METRICS, y, y_pred[:, 1])
        assert isinstance(scores, np.ndarray)
        assert len(scores) == len(METRICS)

    def test_compute_metrics_numpy_array_input_for_metrics(
        self, mock_data: tuple[Labels, npt.NDArray]
    ) -> None:
        """Test passing metrics as a NumPy array of string identifiers."""
        y, y_pred = mock_data
        metrics_arr = np.array(["roc_auc", "accuracy"])
        scores = compute_metrics(metrics_arr, y, y_pred)
        assert len(scores) == 2

    @pytest.mark.parametrize(
        "metrics",
        [
            3.14,
            42,
            "llama",
            {},
            (),
            [],
            [42],
            [3.14],
        ],
    )
    def test_compute_metricss_invalid_metric_type(self, metrics: Any) -> None:
        """Test case when metrics is not a list of strings."""
        with pytest.raises(
            TypeError, match="Input metrics should be a list of strings"
        ):
            compute_metrics(metrics, np.array([]), np.array([]))

    def test_compute_metrics_invalid_metric(self) -> None:
        """Test case when metrics has invalid value."""
        match_expr = (
            "'invalid' was not found in available metrics. "
            f"Available metrics are {METRICS}"
        )

        with pytest.raises(ValueError, match=escape(match_expr)):
            compute_metrics(["invalid"], np.array([]), np.array([]))

    def test_compute_metrics_invalid_metric_value(self) -> None:
        """Test ValueError when invalid metric name is requested."""
        match_expr = (
            "'invalid' was not found in available metrics. "
            f"Available metrics are {METRICS}"
        )
        with pytest.raises(ValueError, match=escape(match_expr)):
            compute_metrics(["invalid"], np.array([0, 1]), np.array([0.1, 0.9]))

    @pytest.mark.parametrize("y", [3.14, 42, "llama", {}, ()])
    def test_compute_metrics_invalid_y(self, y: Any) -> None:
        """Test case when y is not a np.array."""
        match_expr = f"Input y should be a np.ndarray, but got {type(y)}"
        with pytest.raises(TypeError, match=match_expr):
            compute_metrics(METRICS, y, np.array([]))

    @pytest.mark.parametrize("y_pred", [3.14, 42, "llama", {}, ()])
    def test_compute_metrics_invalid_y_pred(self, y_pred: Any) -> None:
        """Test case when y is not a np.array."""
        match_expr = f"Input y_pred should be a np.ndarray, but got {type(y_pred)}"
        with pytest.raises(TypeError, match=match_expr):
            compute_metrics(METRICS, np.array([]), y_pred)

    def test_compute_metrics_single_class_target_raises(self) -> None:
        """
        Test that metrics requiring two classes (e.g. roc_auc) raise a ValueError
        when y contains only a single class (all 0s or all 1s).
        """
        y_single_class = np.array([0, 0, 0, 0, 0])
        y_pred = np.array([0.1, 0.2, 0.4, 0.3, 0.5])

        with pytest.raises(ValueError, match="Only one class present in y_true"):
            compute_metrics(["roc_auc"], y_single_class, y_pred)

    def test_compute_metrics_single_class_target_supported_metric(self) -> None:
        """
        Test that metrics that DO support single-class ground truth
        (e.g. accuracy, brier_score) compute without raising errors.
        """
        y_single_class = np.array([0, 0, 0, 0, 0])
        y_pred = np.array([0.1, 0.2, 0.4, 0.3, 0.5])

        scores = compute_metrics(["accuracy", "brier_score"], y_single_class, y_pred)

        assert isinstance(scores, np.ndarray)
        assert len(scores) == 2
        assert scores[0] == 1.0  # All predictions rounded to 0 match target 0

    def test_compute_metrics_empty_arrays(self) -> None:
        """Test behavior when passing empty NumPy arrays."""
        y_empty = np.array([], dtype=int)
        y_pred_empty = np.array([], dtype=float)

        # Scikit-learn metrics raise ValueError on empty arrays
        with pytest.raises(ValueError):
            compute_metrics(["accuracy"], y_empty, y_pred_empty)

    def test_compute_metrics_custom_metric_dynamic_registration(
        self, mock_data: tuple[Labels, npt.NDArray]
    ) -> None:
        """
        Test registering a custom metric dynamically at runtime and evaluating it
        via compute_metrics.
        """
        y, y_pred = mock_data

        # Define a custom metric function (e.g., Mean Absolute Error of probabilities)
        def custom_mae_func(y_true: np.ndarray, y_score: np.ndarray) -> float:
            return float(np.mean(np.abs(y_true - y_score)))

        # Register it dynamically
        custom_spec = MetricSpec(
            name="custom_prob_mae",
            func=custom_mae_func,
            response_method="predict_proba",
            display_name="Custom Prob MAE",
        )

        with patch.dict(MetricRegistry._registry, {}, clear=False):
            MetricRegistry.register_spec(custom_spec)

            # Evaluate using compute_metrics
            scores = compute_metrics(["custom_prob_mae", "accuracy"], y, y_pred)

            assert isinstance(scores, np.ndarray)
            assert len(scores) == 2
            assert 0.0 <= scores[0] <= 1.0

        # Verify registration was cleaned up and registry is pristine
        with pytest.raises(ValueError, match="custom_prob_mae"):
            MetricRegistry.get("custom_prob_mae")

        assert isinstance(scores, np.ndarray)
        assert len(scores) == 2
        assert 0.0 <= scores[0] <= 1.0  # Mean absolute error is within valid range
