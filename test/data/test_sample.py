#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test functions for the data.sample module
"""

from typing import Type
from unittest.mock import patch

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from medpipe.data.sample import _check_strategy, sample_data, sample_group_data
from medpipe.utils.config import SamplingConfig


class TestSampleData:
    """Test class for the sample_data function."""

    @pytest.fixture
    def imbalanced_dataset(self) -> tuple[pd.DataFrame, npt.NDArray]:
        """Fixture to generate imbalanced data."""
        # Extremely skewed binary classification dataset
        X = np.random.rand(100, 4)
        y = np.array([0] * 90 + [1] * 10)
        return pd.DataFrame(X), y

    def test_sample_data_no_config(
        self, imbalanced_dataset: tuple[pd.DataFrame, npt.NDArray]
    ) -> None:
        """Test case when there is no configuration."""
        X, y = imbalanced_dataset
        X_out, y_out = sample_data(X, y, None)

        pd.testing.assert_frame_equal(X, X_out)
        np.testing.assert_array_equal(y, y_out)

    def test_sample_data_balance_using_smote(
        self, imbalanced_dataset: tuple[pd.DataFrame, npt.NDArray]
    ) -> None:
        """Test case when a valid over sampling strategy is passed."""
        X, y = imbalanced_dataset
        config = SamplingConfig(strategy="SMOTE", k_neighbors=3)  # type: ignore

        X_out, y_out = sample_data(X, y, config)

        # SMOTE upsamples the minority class to match the majority class (90 vs 90)
        assert len(y_out) == 180
        assert len(X_out) == 180
        assert np.sum(y_out == 1) == 90
        assert np.sum(y_out == 0) == 90

    def test_sample_data_balance_using_rus(
        self, imbalanced_dataset: tuple[pd.DataFrame, npt.NDArray]
    ) -> None:
        """Test case when a valid under sampling strategy is passed."""
        X, y = imbalanced_dataset
        config = SamplingConfig(strategy="RandomUnderSampler")

        X_out, y_out = sample_data(X, y, config)

        assert len(y_out) == 20
        assert len(X_out) == 20
        assert np.sum(y_out == 1) == 10
        assert np.sum(y_out == 0) == 10


class TestCheckStrategy:
    """Test class for the _check_strategy function."""

    @pytest.mark.parametrize(
        "strategy, inst",
        [
            ("RandomOverSampler", RandomOverSampler),
            ("RandomUnderSampler", RandomUnderSampler),
            ("SMOTE", SMOTE),
        ],
    )
    def test_check_strategy_success(self, strategy: str, inst: Type) -> None:
        """Test successful function call."""
        sampler = _check_strategy(strategy)
        assert sampler is inst

    def test_check_strategy_invalid_strategy(self) -> None:
        """Test case when an invalid strategy is provided."""
        match_expr = (
            "invalid is not found in imblearn.under_sampling, or "
            "imblearn.over_sampling modules, please check that the strategy matches"
        )
        with pytest.raises(ValueError, match=match_expr):
            _check_strategy("invalid")


class TestSampleGroupData:
    """Test class for the sample_group_data function."""

    @pytest.fixture
    def sample_inputs(self) -> tuple[pd.DataFrame, npt.NDArray, list[npt.NDArray]]:
        """Provides sample DataFrame, labels, and group index splits."""
        X = pd.DataFrame({"feat1": [1, 2, 3, 4], "feat2": [10, 20, 30, 40]})
        y = np.array([0, 1, 0, 1])
        # Two groups: Group 0 -> indices [0, 1], Group 1 -> indices [2, 3]
        group_idx = [np.array([0, 1]), np.array([2, 3])]
        return X, y, group_idx

    def test_sample_group_data_success(
        self, sample_inputs: tuple[pd.DataFrame, npt.NDArray, list[npt.NDArray]]
    ) -> None:
        """Tests successful resampling across each group using a valid configuration."""
        X, y, group_idx = sample_inputs
        config = SamplingConfig(strategy="RandomOverSampler", random_state=42)

        # Mock sample_data to return predictable outputs per group call
        mock_returns = [
            (X.iloc[[0, 1]], np.array([0, 1])),
            (X.iloc[[2, 3]], np.array([0, 1])),
        ]

        with patch(
            "medpipe.data.sample.sample_data", side_effect=mock_returns
        ) as mock_sample:
            X_resampled, y_resampled = sample_group_data(X, y, group_idx, config)

            # Check that sample_data was called once for each group
            assert mock_sample.call_count == 2

            # Verify output data structure concatenation
            assert isinstance(X_resampled, pd.DataFrame)
            assert isinstance(y_resampled, np.ndarray)
            assert len(X_resampled) == 4
            assert len(y_resampled) == 4

    def test_sample_group_data_none_config(
        self, sample_inputs: tuple[pd.DataFrame, npt.NDArray, list[npt.NDArray]]
    ) -> None:
        """Tests function behaviour when config is None,
        ensuring it passes None downstream."""
        X, y, group_idx = sample_inputs

        mock_returns = [
            (X.iloc[[0, 1]], y[[0, 1]]),
            (X.iloc[[2, 3]], y[[2, 3]]),
        ]

        with patch(
            "medpipe.data.sample.sample_data", side_effect=mock_returns
        ) as mock_sample:
            X_resampled, y_resampled = sample_group_data(X, y, group_idx, config=None)

            # Verify None config was passed to sample_data
            for call in mock_sample.call_args_list:
                assert call.args[2] is None

            pd.testing.assert_frame_equal(X_resampled, X)
            np.testing.assert_array_equal(y_resampled, y)

    def test_sample_group_data_propagates_downstream_errors(
        self, sample_inputs: tuple[pd.DataFrame, npt.NDArray, list[npt.NDArray]]
    ) -> None:
        """Tests that exceptions raised by sample_data propagate upwards."""
        X, y, group_idx = sample_inputs
        config = SamplingConfig(strategy="invalid_strategy")  # type: ignore[arg-type]

        with patch(
            "medpipe.data.sample.sample_data",
            side_effect=ValueError("Invalid strategy"),
        ):
            with pytest.raises(ValueError, match="Invalid strategy"):
                sample_group_data(X, y, group_idx, config)
