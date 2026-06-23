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


class TestCreateEstimator:
    """Test class for the create_estimator function."""

    @pytest.mark.parametrize(
        "model_type, model_instance",
        [
            ("HistGradientBoostingClassifier", HistGradientBoostingClassifier),
            ("LogisticRegression", LogisticRegression),
            ("IsotonicRegression", IsotonicRegression),
        ],
    )
    def test_create_estimator_success(
        self, model_type: str, model_instance: Type
    ) -> None:
        """Test successful function call."""
        model = create_estimator(model_type)

        assert isinstance(model, model_instance)

    @pytest.mark.parametrize(
        "model_type, hyperparameters",
        [
            ("HistGradientBoostingClassifier", {"learning_rate": 0.5, "max_iter": 2}),
            ("LogisticRegression", {"l1_ratio": 0.3, "max_iter": 10}),
            ("IsotonicRegression", {"out_of_bounds": "clip"}),
        ],
    )
    def test_create_estimator_config_params(
        self, model_type: str, hyperparameters: dict[str, str | int | float]
    ) -> None:
        """Test that configuration parameters are passed correctly."""
        model = create_estimator(model_type, **hyperparameters)

        for param, value in hyperparameters.items():
            # Check that parameters have been changed correctly
            assert model.__getattribute__(param) == value

    @pytest.mark.parametrize(
        "model_type, hyperparameters",
        [
            ("HistGradientBoostingClassifier", {"invalid": 0.5, "max_iter": 2}),
            ("LogisticRegression", {"invalid": 0.3, "max_iter": 10}),
            ("IsotonicRegression", {"invalid": "clip"}),
        ],
    )
    def test_create_estimator_incorrect_hyperparameter(
        self, model_type: str, hyperparameters: dict[str, str | int | float]
    ) -> None:
        """Test case when incorrect hyperparameters are passed."""
        match_expr = (
            f"{model_type}.__init__() got an unexpected keyword argument 'invalid'"
        )
        with pytest.raises(TypeError, match=escape(match_expr)):
            create_estimator(model_type, **hyperparameters)

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
    def test_create_estimator_invalid_type(self, model_type: Any) -> None:
        """Test case when model type is not a string."""
        match_expr = f"{model_type} should be a string"
        with pytest.raises(TypeError, match=escape(match_expr)):
            create_estimator(model_type)


class TestCheckModelType:
    """Test class for the _check_model_type function."""

    @pytest.mark.parametrize(
        "model_type, model_instance",
        [
            ("HistGradientBoostingClassifier", HistGradientBoostingClassifier),
            ("LogisticRegression", LogisticRegression),
            ("IsotonicRegression", IsotonicRegression),
        ],
    )
    def test_check_model_type_success(
        self, model_type: str, model_instance: Type
    ) -> None:
        """Test successful function call."""
        model = _check_model_type(model_type)

        assert isinstance(model, model_instance)

    def test_check_model_type_invalid(self) -> None:
        """Test case when model type is invalid."""
        match_expr = f"invalid_type is not found in sklearn.ensemble, "
        "sklearn.linear_model, or sklearn.isotonic, "
        "please check that the operation matches"

        with pytest.raises(ValueError, match=match_expr):
            _check_model_type(model_type="invalid_type")


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
