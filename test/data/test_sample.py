#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test functions for the data.sample module
"""

from typing import Type

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from medpipe.data.sample import _check_strategy, sample_data
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
