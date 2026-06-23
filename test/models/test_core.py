#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Models core module test suite.
"""

from pathlib import Path
from re import escape
from typing import Any, Type

import numpy as np
import pytest
from numpy.typing import NDArray
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from medpipe._types import FullProba, PosProba
from medpipe.models.core import (
    _check_model_type,
    create_estimator,
    get_full_proba,
    get_positive_proba,
    load_pipeline,
    save_pipeline,
)
from medpipe.pipeline.pipeline import MedpipePipeline


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


class TestSavePipeline:
    """Test class for the save_pipeline function."""

    @pytest.fixture
    def example_config_dir(self) -> Path:
        """Provide the location of the example configuration files."""
        base_dir = Path(__file__).parent.parent.parent

        return base_dir / "config-examples/"

    def test_save_pipeline_success(
        self, tmp_path: Path, example_config_dir: Path
    ) -> None:
        """Test successful function call."""
        pipeline = MedpipePipeline(example_config_dir / "HGBc_config.toml", logger=None)
        save_pipeline(pipeline, tmp_path / "pipeline.joblib")

        assert (tmp_path / "pipeline.joblib").exists()


class TestLoadPipeline:
    """Test class for the load_pipeline function."""

    @pytest.fixture
    def example_config_dir(self) -> Path:
        """Provide the location of the example configuration files."""
        base_dir = Path(__file__).parent.parent.parent

        return base_dir / "config-examples/"

    def test_load_pipeline_success(
        self, tmp_path: Path, example_config_dir: Path
    ) -> None:
        """Test successful function call."""
        # Create and save a MedpipePipeline
        pipeline = MedpipePipeline(example_config_dir / "HGBc_config.toml", logger=None)
        save_pipeline(pipeline, tmp_path / "pipeline.joblib")

        loaded_pipeline = load_pipeline(tmp_path / "pipeline.joblib")

        assert isinstance(loaded_pipeline, MedpipePipeline)


class TestGetPositiveProba:
    """Test class for the get_positive_proba function."""

    @pytest.mark.parametrize(
        "full_proba, expected_proba",
        [
            (
                np.array([[0, 1], [0.5, 0.5], [0.9, 0.1]]),
                np.array([[1], [0.5], [0.1]]),
            ),
            (
                [np.array([[0, 1]]), np.array([[0.5, 0.5]])],
                np.array([1, 0.5]),
            ),
            (
                [np.array([[0, 1], [0.5, 0.5]]), np.array([[0.5, 0.5], [0.9, 0.1]])],
                np.array([[1, 0.5], [0.5, 0.1]]),
            ),
        ],
    )
    def test_get_positive_proba_success(
        self, full_proba: FullProba | list[NDArray], expected_proba: PosProba
    ) -> None:
        """Test successful function call."""
        pos_proba = get_positive_proba(full_proba)
        assert (pos_proba == expected_proba).all()

    def test_get_positive_proba_mismatch(self) -> None:
        """Test case when list has mismatched lenghts."""
        mismatched_proba = [
            np.array([[0.2, 0.8], [1, 0]]),
            np.array([[0.1, 0.9]]),
        ]
        match_expr = "setting an array element with a sequence. "
        "The requested array has an inhomogeneous shape after 1 dimensions. "
        "The detected shape was (2,) + inhomogeneous part."
        with pytest.raises(ValueError, match=match_expr):
            get_positive_proba(mismatched_proba)

    @pytest.mark.parametrize(
        "full_proba",
        [
            np.array([[0], [0.5], [0.9]]),
            np.array([[0, 0.5, 0.9]]),
        ],
    )
    def test_get_pos_proba_incorrect_shape(self, full_proba: FullProba) -> None:
        """Test case where full_proba has incorrect shape."""
        match_expr = (
            "Input probabilities should have shape (n_samples, 2), "
            f"but got {full_proba.shape}"
        )
        with pytest.raises(ValueError, match=escape(match_expr)):
            get_positive_proba(full_proba)


class TestGetFullProba:
    """Test class for the get_full_proba function."""

    @pytest.mark.parametrize(
        "pos_proba, expected_proba",
        [
            (
                np.array([[1], [0.5], [0.1]]),
                np.array([[0, 1], [0.5, 0.5], [0.9, 0.1]]),
            ),
            (
                np.array([1, 0.5, 0]),
                np.array([[0, 1], [0.5, 0.5], [1, 0]]),
            ),
        ],
    )
    def test_get_full_proba_success(
        self, pos_proba: PosProba, expected_proba: FullProba
    ) -> None:
        """Test successful function call."""
        pos_proba = get_full_proba(pos_proba)
        assert (pos_proba == expected_proba).all()

    @pytest.mark.parametrize(
        "pos_proba",
        [
            np.array([[1, 0.5], [0.5, 0.1]]),
            np.array([[1, 0.5, 0.2], [0.5, 0.1, 0.8]]),
        ],
    )
    def test_get_full_proba_incorrect_shape(self, pos_proba: PosProba) -> None:
        """Test case where pos_proba has incorrect shape."""
        match_expr = (
            "Input probabilities should have shape (n_samples,) or "
            f"(n_samples, 1), but got {pos_proba.shape}"
        )
        with pytest.raises(ValueError, match=escape(match_expr)):
            pos_proba = get_full_proba(pos_proba)

    @pytest.mark.parametrize("pos_proba", [42, 3.14, {}, (), []])
    def test_get_full_proba_invalid_type(self, pos_proba: Any) -> None:
        """Test case when pos_proba is invalid type."""
        match_expr = (
            f"Input probabilities should be a np.ndarray, but got {type(pos_proba)}"
        )
        with pytest.raises(TypeError, match=escape(match_expr)):
            get_full_proba(pos_proba)
