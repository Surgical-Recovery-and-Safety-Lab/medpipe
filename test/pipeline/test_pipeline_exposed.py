#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline class exposed functions test suites.
"""

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from fixtures import (
    MockData,
    MockLabels,
    _mp_pipeline,
    example_config_dir,
    mock_data,
    mock_labels,
    mp_pipeline,
    shield_local_filesystem,
)
from pandas.testing import assert_frame_equal
from pytest import MonkeyPatch
from sklearn.base import check_is_fitted
from sklearn.isotonic import IsotonicRegression
from sklearn.pipeline import NotFittedError

from medpipe.pipeline.pipeline import MedpipePipeline

# ==============================================================================
# Test classes for exposed functions
# ==============================================================================


class TestTransform:
    """Test class for the transform function of the MedpipePipeline class."""

    def test_pipeline_transform_success(
        self, mp_pipeline: MedpipePipeline, mock_data: MockData
    ) -> None:
        """Test successful function call."""
        X_train, X_test, _ = mock_data
        mp_pipeline.fit_transform(X_train)  # Fit the preprocessor
        X_transformed = mp_pipeline.transform(X_test)

        assert X_transformed.shape == X_test.shape
        assert isinstance(X_transformed, np.ndarray)

    def test_pipeline_transform_not_fitted(self, mp_pipeline: MedpipePipeline) -> None:
        """Test case when preprocessor has not been fitted."""
        with pytest.raises(NotFittedError):
            mp_pipeline.transform(pd.DataFrame({}))

    def test_pipeline_transform_warning(
        self, mp_pipeline: MedpipePipeline, mock_data: MockData
    ) -> None:
        """Test case when function raises a UserWarning."""
        _, X_test, _ = mock_data
        mp_pipeline.preprocessor = None  # Set preprocessor to None
        match_expr = "No preprocessor object created so data not transformed"
        with pytest.warns(UserWarning, match=match_expr):
            mp_pipeline.transform(X_test)

    @pytest.mark.filterwarnings("ignore")
    def test_pipeline_transform_no_transformation(
        self, mp_pipeline: MedpipePipeline, mock_data: MockData
    ) -> None:
        """Test case when data has not been transformed."""
        _, X_test, _ = mock_data
        mp_pipeline.preprocessor = None  # Set preprocessor to None
        X_unprocessed = mp_pipeline.transform(X_test)

        assert_frame_equal(X_unprocessed, X_test)


class TestFitTransform:
    """Test class for the fit_fit_transform function of the
    MedpipePipeline class."""

    def test_pipeline_fit_transform_success(
        self, mp_pipeline: MedpipePipeline, mock_data: MockData
    ) -> None:
        """Test successful function call."""
        _, X_test, _ = mock_data
        X_fit_transformed = mp_pipeline.fit_transform(X_test)

        check_is_fitted(mp_pipeline.preprocessor)
        assert X_fit_transformed.shape == X_test.shape
        assert isinstance(X_fit_transformed, np.ndarray)

    def test_pipeline_fit_transform_warning(
        self, mp_pipeline: MedpipePipeline, mock_data: MockData
    ) -> None:
        """Test case when function raises a UserWarning."""
        _, X_test, _ = mock_data
        mp_pipeline.preprocessor = None  # Set preprocessor to None
        match_expr = "No preprocessor object created so data not transformed"
        with pytest.warns(UserWarning, match=match_expr):
            mp_pipeline.fit_transform(X_test)

    @pytest.mark.filterwarnings("ignore")
    def test_pipeline_fit_transform_no_fit_transformation(
        self, mp_pipeline: MedpipePipeline, mock_data: MockData
    ) -> None:
        """Test case when data has not been fit_transformed."""
        _, X_test, _ = mock_data
        mp_pipeline.preprocessor = None  # Set preprocessor to None
        X_unprocessed = mp_pipeline.fit_transform(X_test)

        assert_frame_equal(X_unprocessed, X_test)


class TestFit:
    """Test class for the fit function of the MedpipePipeline class."""

    @pytest.mark.parametrize(
        "version, top_level_config",
        [
            (["0", "0", "0"], "HGBc_no_recal_config.toml"),
            (["0", "2", "2"], "HGBc_no_recal_config.toml"),
        ],
    )
    def test_pipeline_fit_success_no_recal(
        self,
        monkeypatch: MonkeyPatch,
        example_config_dir: Path,
        mock_data: MockData,
        mock_labels: MockLabels,
        version: list[str],
        top_level_config: str,
    ) -> None:
        """Test successful function call without recalibration."""
        X_train, _, _ = mock_data
        y_train, _, _ = mock_labels
        pipe = _mp_pipeline(monkeypatch, example_config_dir, top_level_config, version)
        pipe.fit(X_train, y_train)

    @pytest.mark.parametrize(
        "version, top_level_config",
        [
            (["0", "2", "1"], "HGBc_config.toml"),
            (["0", "2", "2"], "HGBc_config.toml"),
        ],
    )
    def test_pipeline_fit_success_with_recal(
        self,
        monkeypatch: MonkeyPatch,
        example_config_dir: Path,
        mock_data: MockData,
        mock_labels: MockLabels,
        version: list[str],
        top_level_config: str,
    ) -> None:
        """Test successful function call with recalibration."""
        X_train, X_recal, _ = mock_data
        y_train, y_recal, _ = mock_labels
        pipe = _mp_pipeline(monkeypatch, example_config_dir, top_level_config, version)
        pipe.fit(X_train, y_train, X_recal, y_recal)

    @pytest.mark.parametrize("X", [3.14, 42, "llama", [], {}, ()])
    def test_pipeline_fit_incorrect_X(
        self, X: Any, mp_pipeline: MedpipePipeline
    ) -> None:
        """Test case when X is of incorrect type."""
        match_expr = f"Input X should be pd.DataFrame, but got {type(X)}"
        with pytest.raises(TypeError, match=match_expr):
            mp_pipeline.fit(X, np.array([]))

    @pytest.mark.parametrize("X_recal", [3.14, 42, "llama", [], {}, ()])
    def test_pipeline_fit_incorrect_X_recal(
        self, X_recal: Any, mp_pipeline: MedpipePipeline
    ) -> None:
        """Test case when X_recal is of incorrect type."""
        match_expr = f"Input X_recal should be pd.DataFrame, but got {type(X_recal)}"
        with pytest.raises(TypeError, match=match_expr):
            mp_pipeline.fit(pd.DataFrame({}), np.array([]), X_recal, np.array([]))


class TestRun:
    """Test class for the run function of the MedpipePipeline class."""

    @pytest.mark.parametrize(
        "version, top_level_config",
        [
            (["0", "0", "0"], "HGBc_no_recal_config.toml"),
            (["0", "1", "1"], "HGBc_config.toml"),
            (["1", "2", "2"], "HGBc_config.toml"),
        ],
    )
    def test_pipeline_run_success(
        self,
        monkeypatch: MonkeyPatch,
        example_config_dir: Path,
        version: list[str],
        top_level_config: str,
    ) -> None:
        """Test successful function call."""
        pipe = _mp_pipeline(monkeypatch, example_config_dir, top_level_config, version)
        pipe.run()

    def test_pipeline_run_success_with_data(
        self, mp_pipeline: MedpipePipeline, mock_data: MockData
    ) -> None:
        """Test successful function call with data."""
        data = pd.concat(mock_data)
        y = [0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1]
        data["MORTALITY_30D"] = y

        mp_pipeline.run(data)

    @pytest.mark.parametrize("data", [3.12, 42, "llama", {}, [], ()])
    def test_pipeline_run_incorrect_data(
        self, mp_pipeline: MedpipePipeline, data: Any
    ) -> None:
        """Test case when incorrect data type is passed."""
        match_expr = f"Input data should be a pd.DataFrame, but got {type(data)}"
        with pytest.raises(TypeError, match=match_expr):
            mp_pipeline.run(data)

    def test_pipeline_run_no_preprocessing(self, mp_pipeline: MedpipePipeline) -> None:
        """Test successful function call with no preprocessing."""
        mp_pipeline.preprocessor = None
        mp_pipeline.run()

    def test_pipeline_run_no_recalibration(self, mp_pipeline: MedpipePipeline) -> None:
        """Test successful function call with no preprocessing."""
        mp_pipeline.recalibrator_method = None
        mp_pipeline.recalibrator = {}
        mp_pipeline.run()

    def test_pipeline_run_no_cv(self, mp_pipeline: MedpipePipeline) -> None:
        """Test successful function call with no cross-validation."""
        mp_pipeline.medpipe_config.top_level.meta.run_mode = "fast"
        mp_pipeline.medpipe_config.workflow.validation.cross_validation = None
        mp_pipeline.run()


class TestPredictProba:
    """Test class for the predict_proba function of
    the MedpipePipeline class."""

    @pytest.mark.parametrize(
        "version, top_level_config",
        [
            (["0", "1", "1"], "HGBc_config.toml"),
            (["1", "2", "2"], "HGBc_config.toml"),
        ],
    )
    def test_pipeline_predict_proba_success(
        self,
        monkeypatch: MonkeyPatch,
        mock_data: MockData,
        mock_labels: MockLabels,
        example_config_dir: Path,
        version: list[str],
        top_level_config: str,
    ) -> None:
        """Test successful function call."""
        mp_pipeline = _mp_pipeline(
            monkeypatch, example_config_dir, top_level_config, version
        )
        X_train, X_test, X_recal, _ = mp_pipeline._drop_group_columns(*mock_data)
        y_train, _, y_recal = mock_labels

        assert X_test is not None
        mp_pipeline.fit(X_train, y_train, X_recal, y_recal)  # Fit the MedpipePipeline
        outputs = mp_pipeline.predict_proba(X_test)

        assert isinstance(outputs, list)

        for output in outputs:
            assert isinstance(output, np.ndarray)
            assert output.shape == (len(X_test), 2)

    def test_pipeline_predict_proba_success_with_recal(
        self, mp_pipeline: MedpipePipeline, mock_data: MockData, mock_labels: MockLabels
    ) -> None:
        """Test successful function call with recalibrator data."""
        X_train, X_test, X_recal, _ = mp_pipeline._drop_group_columns(*mock_data)
        y_train, _, y_recal = mock_labels

        assert X_test is not None
        mp_pipeline.fit(X_train, y_train, X_recal, y_recal)  # Fit the MedpipePipeline
        outputs = mp_pipeline.predict_proba(X_test, "all", "recalibrator")

        assert isinstance(outputs, list)

        for output in outputs:
            assert isinstance(output, np.ndarray)
            assert output.shape == (len(X_test), 2)

    def test_pipeline_predict_proba_with_no_recal_but_data(
        self, mp_pipeline: MedpipePipeline, mock_data: MockData, mock_labels: MockLabels
    ) -> None:
        """Test case when not recalibrator but recalibrator is called."""
        X_train, X_test, X_recal, _ = mp_pipeline._drop_group_columns(*mock_data)
        y_train, _, y_recal = mock_labels
        assert X_test is not None

        mp_pipeline.recalibrator = {}
        mp_pipeline.fit(X_train, y_train, X_recal, y_recal)  # Fit the MedpipePipeline
        with pytest.raises(ValueError, match="No recalibrator present in pipeline"):
            mp_pipeline.predict_proba(
                X_test, outcomes="all", estimator_type="recalibrator"
            )

    def test_pipeline_predict_proba_not_implemented(
        self, mp_pipeline: MedpipePipeline
    ) -> None:
        """Test case when predict_proba is not implemented."""
        X_train = np.array([[0.8, 0.9, 0.7, 0.4, 0.1, 0.02, 0.3]]).T
        y_train = np.array([[1, 1, 1, 1, 0, 0, 0]]).T

        mp_pipeline.preprocessor = None
        mp_pipeline.recalibrator = {}

        for outcome in mp_pipeline.predictor.keys():
            mp_pipeline.predictor[outcome] = IsotonicRegression()

        mp_pipeline.fit(X_train, y_train)  # Fit the MedpipePipeline

        match_expr = (
            "Predictor of type IsotonicRegression does not implement "
            "the predict_proba method"
        )
        with pytest.raises(NotImplementedError, match=match_expr):
            mp_pipeline.predict_proba(
                X_train, outcomes="all", estimator_type="recalibrator"
            )


class TestPredict:
    """Test class for the predict function of
    the MedpipePipeline class."""

    @pytest.mark.parametrize(
        "version, top_level_config",
        [
            (["0", "1", "1"], "HGBc_config.toml"),
            (["1", "2", "2"], "HGBc_config.toml"),
        ],
    )
    def test_pipeline_predict_success(
        self,
        monkeypatch: MonkeyPatch,
        mock_data: MockData,
        example_config_dir: Path,
        version: list[str],
        top_level_config: str,
    ) -> None:
        """Test successful function call."""
        mp_pipeline = _mp_pipeline(
            monkeypatch, example_config_dir, top_level_config, version
        )
        X_train, X_test, X_recal, _ = mp_pipeline._drop_group_columns(*mock_data)
        y_train: npt.NDArray = np.zeros((len(X_train), mp_pipeline.n_outcomes))
        y_train[1, :] = 1  # Make at least one example positive
        y_recal: npt.NDArray = np.zeros((3, mp_pipeline.n_outcomes))
        y_recal[0, :] = 1

        assert X_test is not None
        mp_pipeline.fit(X_train, y_train, X_recal, y_recal)  # Fit the MedpipePipeline
        outputs = mp_pipeline.predict(X_test)

        assert isinstance(outputs, list)

        for output in outputs:
            assert isinstance(output, np.ndarray)
            assert len(output) == len(X_test)

    def test_pipeline_predict_success_with_recal(
        self, mp_pipeline: MedpipePipeline, mock_data: MockData, mock_labels: MockLabels
    ) -> None:
        """Test successful function call with recalibrator data."""
        X_train, X_test, X_recal, _ = mp_pipeline._drop_group_columns(*mock_data)
        y_train, _, y_recal = mock_labels

        assert X_test is not None
        mp_pipeline.fit(X_train, y_train, X_recal, y_recal)  # Fit the MedpipePipeline
        outputs = mp_pipeline.predict(X_test, "all", "recalibrator")

        assert isinstance(outputs, list)

        for output in outputs:
            assert isinstance(output, np.ndarray)
            assert len(output) == len(X_test)

    def test_pipeline_predict_with_no_recal_but_data(
        self, mp_pipeline: MedpipePipeline, mock_data: MockData
    ) -> None:
        """Test case when not recalibrator but recalibrator is called."""
        X_train, X_test, X_recal, _ = mp_pipeline._drop_group_columns(*mock_data)
        assert X_test is not None

        mp_pipeline.recalibrator = {}
        y_train = np.array([[1, 1, 1, 1, 0, 0, 0]]).T
        y_recal = np.array([[1, 0, 0]]).T

        mp_pipeline.fit(X_train, y_train, X_recal, y_recal)  # Fit the MedpipePipeline
        with pytest.raises(ValueError, match="No recalibrator present in pipeline"):
            mp_pipeline.predict(X_test, outcomes="all", estimator_type="recalibrator")
