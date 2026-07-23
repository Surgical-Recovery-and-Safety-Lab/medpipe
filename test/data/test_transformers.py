# -*- coding: utf-8 -*-
"""Test functions for the data.transformers module."""

from typing import Any

import numpy as np
import pytest

from medpipe.data.transformers import BoundedLogitTransformer


class TestBoundedLogitTransformer:
    """Test class for the BoundedLogitTransformer class."""

    @pytest.fixture
    def default_transformer(self) -> BoundedLogitTransformer:
        """Provide a BoundedLogitTransformer instance with default settings."""
        return BoundedLogitTransformer(max_value=1.0)

    def test_bounded_logit_transformer_fit_success(
        self, default_transformer: BoundedLogitTransformer
    ) -> None:
        """Test that the fit method returns the transformer instance itself."""
        y: np.ndarray[Any, np.dtype[np.float64]] = np.array([0.1, 0.5, 0.9])

        result = default_transformer.fit(y)

        assert result is default_transformer

    def test_bounded_logit_transformer_transform_success(
        self, default_transformer: BoundedLogitTransformer
    ) -> None:
        """Test successful forward transformation into logit space."""
        # np.log(0.99999/0.00001) = 16.11
        y: np.ndarray[Any, np.dtype[np.float64]] = np.array([1.0])

        transformed = default_transformer.transform(y)
        assert np.isclose(transformed[0], 16.11809)

    def test_bounded_logit_transformer_inverse_transform_success(
        self, default_transformer: BoundedLogitTransformer
    ) -> None:
        """Test successful backward transformation from logit space."""
        # 0 in logit space should map back to 0.5 (half of max_value 1.0)
        y: np.ndarray[Any, np.dtype[np.float64]] = np.array([0.0])

        inverse_transformed = default_transformer.inverse_transform(y)

        assert np.isclose(inverse_transformed[0], 0.5)

    def test_bounded_logit_transformer_transform_clips_boundaries(self) -> None:
        """Test that values exactly at or exceeding boundaries are
        clipped by epsilon."""
        transformer = BoundedLogitTransformer(max_value=10.0, epsilon=1e-5)

        # Test 0 (lower bound) and 10 (upper bound)
        y: np.ndarray[Any, np.dtype[np.float64]] = np.array([0.0, 10.0])

        transformed = transformer.transform(y)

        # Values should be finite because epsilon prevents
        # division by zero / log(0)
        assert np.isfinite(transformed).all()

    def test_bounded_logit_transformer_round_trip_success(self) -> None:
        """Test that transforming and inverse transforming returns
        the original values."""
        transformer = BoundedLogitTransformer(max_value=100.0)
        original_data: np.ndarray[Any, np.dtype[np.float64]] = np.array(
            [10.0, 50.0, 90.0]
        )

        transformed = transformer.transform(original_data)
        reconstructed = transformer.inverse_transform(transformed)

        assert np.allclose(original_data, reconstructed)
