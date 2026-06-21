#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Models core module test suite.
"""

from re import escape
from typing import Any, TypeAlias

import pytest
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from medpipe.models.core import create_model

ModelTypes: TypeAlias = type[
    HistGradientBoostingClassifier | IsotonicRegression | LogisticRegression
]


class TestCreateModel:
    """Test class for the create_model function."""

    @pytest.mark.parametrize(
        "model_type, model_instance",
        [
            ("hgb-c", HistGradientBoostingClassifier),
            ("logistic", LogisticRegression),
            ("isotonic", IsotonicRegression),
        ],
    )
    def test_create_model_success(
        self, model_type: str, model_instance: ModelTypes
    ) -> None:
        """Test successful function call."""
        model = create_model(
            model_type,
            logger=None,
        )

        assert isinstance(model, model_instance)

    @pytest.mark.parametrize(
        "model_type, config_params",
        [
            ("hgb-c", {"learning_rate": 0.5, "max_iter": 2}),
            ("logistic", {"l1_ratio": 0.3, "max_iter": 10}),
            ("isotonic", {"out_of_bounds": "clip"}),
        ],
    )
    def test_create_model_config_params(
        self, model_type: str, config_params: dict[str, str | int | float]
    ) -> None:
        """Test that configuration parameters are passed correctly."""
        model = create_model(model_type, logger=None, quiet=False, **config_params)

        for param, value in config_params.items():
            # Check that parameters have been changed correctly
            assert model.__getattribute__(param) == value

    @pytest.mark.parametrize(
        "model_type",
        [
            42,
            3.14,
            {"a": 1},
            (1, "a"),
            [1, 2, 3],
        ],
    )
    def test_create_model_invalid_type(self, model_type: Any) -> None:
        """Test case when model type is not a string."""
        match_expr = f"{model_type} should be a string"
        with pytest.raises(TypeError, match=escape(match_expr)):
            create_model(model_type)

    def test_create_model_invalid_value(self) -> None:
        """Test case when model type is invalid."""
        with pytest.raises(
            ValueError, match="invalid_type invalid model type. See function docstring"
        ):
            create_model(model_type="invalid_type")
