#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test functions for the models.core module
"""

from pathlib import Path
from typing import Any, Generator
from unittest.mock import patch

import ngboost
import pytest
import sklearn.ensemble
import sklearn.linear_model

from medpipe import MedpipePipeline
from medpipe.models.core import (
    _check_model_type,
    create_estimator,
    load_pipeline,
    save_pipeline,
)


@pytest.fixture
def shield_local_filesystem() -> Generator[Any, Any, Any]:
    """Automatically prevent any test in this file from creating folders on disk."""
    with (
        patch("pathlib.Path.mkdir") as mock_path_mkdir,
        patch("os.makedirs") as mock_os_makedirs,
    ):
        # Relinquish control back to the test runner loop
        yield mock_path_mkdir, mock_os_makedirs


@pytest.fixture
def example_config_dir() -> Path:
    """Provide the location of the example configuration files."""
    base_dir = Path(__file__).parent.parent.parent

    return base_dir / "config-examples/"


@pytest.fixture
def mp_pipeline(
    example_config_dir: Path, shield_local_filesystem: Generator[Any, Any, Any]
) -> MedpipePipeline:
    """Create a Medpipe Pipeline for tests."""
    return MedpipePipeline(example_config_dir / "HGBc_config.toml", logger=None)


class TestCreateEstimator:
    """Test class for the create_estimator function."""

    def test_create_estimator_classifier_success(self) -> None:
        """Tests successful instantiation of a valid classifier
        with hyperparameters."""
        estimator = create_estimator(
            "RandomForestClassifier", n_estimators=10, random_state=42
        )

        assert isinstance(estimator, sklearn.ensemble.RandomForestClassifier)
        assert estimator.n_estimators == 10
        assert estimator.random_state == 42

    @pytest.mark.parametrize(
        "model_type", ["NGBRegressor", "HistGradientBoostingRegressor"]
    )
    def test_create_estimator_regressor_success(self, model_type: str) -> None:
        """Tests successful instantiation of a valid regressor with hyperparameters."""
        estimator = create_estimator(model_type)

        assert isinstance(estimator, sklearn.compose.TransformedTargetRegressor)

    def test_create_estimator_type_error_for_non_string_model_type(self) -> None:
        """Tests that a TypeError is raised when model_type is not a string."""
        with pytest.raises(TypeError, match="123 should be a string"):
            create_estimator(123)  # type: ignore[arg-type]

    def test_create_estimator_type_error_for_invalid_hyperparameters(self) -> None:
        """Tests that a TypeError is raised when unexpected keyword
        arguments are passed."""
        with pytest.raises(TypeError, match="got an unexpected keyword argument"):
            create_estimator("LogisticRegression", non_existent_param=123)


class TestCheckModelType:
    """Test class for the _check_model_type function."""

    @pytest.mark.parametrize(
        "model_type, expected_class",
        [
            ("RandomForestClassifier", sklearn.ensemble.RandomForestClassifier),
            ("LogisticRegression", sklearn.linear_model.LogisticRegression),
            ("IsotonicRegression", sklearn.isotonic.IsotonicRegression),
            ("NGBClassifier", ngboost.NGBClassifier),
        ],
    )
    def test_check_model_type_success(
        self, model_type: str, expected_class: type
    ) -> None:
        """Tests that valid estimator names return their respective
        uninstantiated class type."""
        estimator_cls = _check_model_type(model_type)

        assert estimator_cls is expected_class

    def test_check_model_type_value_error_for_unknown_model(self) -> None:
        """Tests that a ValueError is raised when an unsupported model_type
        string is provided."""
        invalid_type = "NonExistentModel"

        with pytest.raises(ValueError, match="NonExistentModel is not found in"):
            _check_model_type(invalid_type)


class TestSavePipeline:
    """Test class for the save_pipeline function."""

    def test_save_pipeline_success(
        self, tmp_path: Path, mp_pipeline: MedpipePipeline
    ) -> None:
        """Tests saving a pipeline object to disk successfully using a Path object."""
        save_file = tmp_path / "test_pipeline.joblib"

        save_pipeline(mp_pipeline, save_file)

        assert save_file.exists()
        assert save_file.stat().st_size > 0

    def test_save_pipeline_success_with_string_path(
        self, tmp_path: Path, mp_pipeline: MedpipePipeline
    ) -> None:
        """Tests saving a pipeline object using a string file path."""
        save_file_str = str(tmp_path / "test_pipeline_str.joblib")

        save_pipeline(mp_pipeline, save_file_str)

        assert Path(save_file_str).exists()


class TestLoadPipeline:
    """Test class for the load_pipeline function."""

    def test_load_pipeline_success(
        self, tmp_path: Path, mp_pipeline: MedpipePipeline
    ) -> None:
        """Tests loading a saved pipeline object successfully."""
        save_file = tmp_path / "saved_pipeline.joblib"

        save_pipeline(mp_pipeline, save_file)
        loaded_pipeline = load_pipeline(save_file)

        assert loaded_pipeline.version == mp_pipeline.version
