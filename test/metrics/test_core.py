#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test functions for the metrics.core module
"""

from re import escape
from typing import Any

import numpy as np
import pytest

from medpipe._types import FullProba, Labels
from medpipe.metrics.core import METRICS, build_scorers, compute_metrics, ici_score


@pytest.fixture
def mock_data() -> tuple[Labels, FullProba]:
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

    def test_ici_score_success(self, mock_data: tuple[Labels, FullProba]) -> None:
        """Test successful function call."""
        y, y_pred = mock_data
        ici = ici_score(y, y_pred)

        assert isinstance(ici, float)

    def test_ici_score_pos_proba(self, mock_data: tuple[Labels, FullProba]) -> None:
        """Test successful function call with positive class probabilities only."""
        y, y_pred = mock_data  # Unpack mock data
        y_pred = y_pred[:, 1]
        ici = ici_score(y, y_pred)

        assert isinstance(ici, float)


class TestBuildScorers:
    """Test class for the build_scorers function."""

    def test_build_scorers_success(self) -> None:
        """Test successful function call."""
        scorers = build_scorers(METRICS)

        assert len(scorers) == len(METRICS)
        assert (metric in METRICS for metric in scorers.keys())

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
            "invalid was not found in available metric "
            f"list. Available metrics are {METRICS}"
        )

        with pytest.raises(ValueError, match=escape(match_expr)):
            build_scorers(["invalid"])


class TestComputeMetrics:
    """Test class for the compute_metrics function."""

    def test_compute_metrics_success(self, mock_data: tuple[Labels, FullProba]) -> None:
        """Test successful function call."""
        y, y_pred = mock_data  # Unpack mock data
        scores = compute_metrics(METRICS, y, y_pred)

        assert isinstance(scores, np.ndarray)
        assert len(scores) == len(METRICS)

    def test_compute_metrics_success_pos_proba(
        self, mock_data: tuple[Labels, FullProba]
    ) -> None:
        """Test successful function call."""
        y, y_pred = mock_data  # Unpack mock data
        y_pred = y_pred[:, 1]  # Get only positive class
        scores = compute_metrics(METRICS, y, y_pred)

        assert isinstance(scores, np.ndarray)
        assert len(scores) == len(METRICS)

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
            "invalid was not found in available metric "
            f"list. Available metrics are {METRICS}"
        )

        with pytest.raises(ValueError, match=escape(match_expr)):
            compute_metrics(["invalid"], np.array([]), np.array([]))

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
