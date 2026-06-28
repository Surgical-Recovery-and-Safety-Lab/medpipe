#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test functions for the metrics.core module
"""

from re import escape

import numpy as np
import numpy.typing as npt
import pytest

from medpipe.metrics.core import METRICS, build_scorers, ici_score


class TestIciScore:
    """Test class for the ici_score function."""

    @pytest.mark.parametrize(
        "y, y_pred",
        [
            (
                np.array([0, 0, 0, 0, 1, 1, 1, 1]),
                np.array([0.01, 0.2, 0.1, 0.2, 0.4, 0.8, 0.9, 0.5]),
            ),
            (
                np.array([0, 0, 0, 0, 1, 1, 1, 1]),
                np.array(
                    [
                        [0.99, 0.01],
                        [0.8, 0.2],
                        [0.9, 0.1],
                        [0.8, 0.2],
                        [0.6, 0.4],
                        [0.2, 0.8],
                        [0.1, 0.9],
                        [0.5, 0.5],
                    ]
                ),
            ),
        ],
    )
    def test_ici_score_success(self, y: npt.NDArray, y_pred: npt.NDArray) -> None:
        """Test successful function call."""
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
