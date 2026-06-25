#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline class tests suite.
"""

from pathlib import Path
from typing import Any, Generator, Literal
from unittest.mock import patch

import numpy as np
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GroupKFold, StratifiedKFold

from medpipe.pipeline.pipeline import MedpipePipeline
from medpipe.utils.io import load_data, read_toml_configuration

# ==============================================================================
# Fixtures for all tests
# ==============================================================================


@pytest.fixture(autouse=True)
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
def mp_pipeline(example_config_dir: Path) -> MedpipePipeline:
    """Create a Medpipe Pipeline for tests."""
    return MedpipePipeline(example_config_dir / "HGBc_config.toml", logger=None)


# ==============================================================================
# Test classes
# ==============================================================================


class TestPipeline:
    """Test class for the MedpipePipeline class"""

    def test_create_pipeline_from_file(self, example_config_dir: Path) -> None:
        """Test successful pipeline creation from configuration file."""
        pipe = MedpipePipeline(example_config_dir / "HGBc_config.toml", logger=None)
        assert pipe.version == "v0.1.1"
        assert pipe.predictor_algo == "HistGradientBoostingClassifier"
        assert pipe.calibrator_method == "IsotonicRegression"
        assert pipe.n_outcomes == 1
        assert isinstance(pipe.preprocessor, ColumnTransformer)

    def test_create_pipeline_from_config(self, example_config_dir: Path) -> None:
        """Test successful pipeline creation from MedpipeConfig."""
        config = read_toml_configuration(example_config_dir / "HGBc_config.toml")
        pipe = MedpipePipeline(config, logger=None)
        assert pipe.version == "v0.1.1"
        assert pipe.predictor_algo == "HistGradientBoostingClassifier"
        assert pipe.calibrator_method == "IsotonicRegression"
        assert pipe.n_outcomes == 1
        assert isinstance(pipe.preprocessor, ColumnTransformer)


class TestCheckOp:
    """Test class for the _check_op function of the MedpipePipeline class."""

    @pytest.mark.parametrize("op", ("StandardScaler", "SimpleImputer"))
    def test_pipeline_check_operation_success(
        self, mp_pipeline: MedpipePipeline, op: str
    ) -> None:
        """Test successful function call."""
        assert mp_pipeline._check_operation(op)

    def test_pipeline_check_operation_invalid_op(
        self, mp_pipeline: MedpipePipeline
    ) -> None:
        """Test case when invalid operation is provided."""
        match_expr = f"invalid is not found in sklearn.preprocessing or "
        "sklearn.impute, please check that the operation matches"

        with pytest.raises(ValueError, match=match_expr):
            mp_pipeline._check_operation("invalid")


class TestSetPreprocessingSteps:
    """Test class for the _set_preprocesing_steps function of the
    MedpipePipeline class."""

    def test_pipeline_set_preprocessing_steps_success(
        self, mp_pipeline: MedpipePipeline
    ) -> None:
        """Test successful function call."""
        ct = mp_pipeline._set_preprocessing_steps()

        assert isinstance(ct, ColumnTransformer)
        assert ct.transformers[0][0] == "op_1"
        assert ct.transformers[1][0] == "op_2"
        assert ct.transformers[0][2] == ["SEX", "CATEGORY_LEVEL_1"]
        assert ct.transformers[1][2] == ["CATEGORY_LEVEL_1"]
        assert ct.remainder == "passthrough"

    def test_pipeline_set_preprocessing_steps_None(
        self, mp_pipeline: MedpipePipeline
    ) -> None:
        """Test that None is returned correctly."""
        mp_pipeline.medpipe_config.workflow.preprocessing.preprocess = False  # type: ignore
        assert mp_pipeline._set_preprocessing_steps() == None

        mp_pipeline.medpipe_config.workflow.preprocessing = None
        assert mp_pipeline._set_preprocessing_steps() == None


class TestHasPreprocessor:
    """Test class for the _has_preprocessor function of the
    MedpipePipeline class."""

    def test_pipeline_has_preprocessor_success(
        self, mp_pipeline: MedpipePipeline
    ) -> None:
        """Test successful function call."""
        # Should be true in the default configuration file
        assert mp_pipeline._has_preprocessor() == True

    def test_pipeline_has_preprocessor_False(
        self, mp_pipeline: MedpipePipeline
    ) -> None:
        """Test cases when _has_preprocesor return False."""
        mp_pipeline.medpipe_config.workflow.preprocessing.preprocess = False  # type: ignore

        assert mp_pipeline._has_preprocessor() == False

        mp_pipeline.medpipe_config.workflow.preprocessing = None

        assert mp_pipeline._has_preprocessor() == False


class TestGetDataSets:
    """Test class for the _get_data_sets function of the
    MedpipePipeline class."""

    def test_pipeline_get_data_sets_success(self, mp_pipeline: MedpipePipeline) -> None:
        """Test successful function call."""
        mock_data = load_data(mp_pipeline.medpipe_config.data.path)
        validation_config = mp_pipeline.medpipe_config.workflow.validation

        test_group_column = validation_config.test_split.group_column
        cv_group_column = validation_config.cross_validation.group_column

        X_train, _, X_test, _, X_recal, _, groups = mp_pipeline._get_data_sets(
            mock_data
        )

        assert test_group_column not in X_train.columns
        assert test_group_column not in X_test.columns
        assert X_recal is not None  # Default config has recalibration set
        assert test_group_column not in X_recal.columns

        # Check group_column and X_train
        assert groups is not None  # Default config has group_column
        assert cv_group_column not in X_train.columns
        assert len(groups) == len(X_train)

    def test_pipeline_get_data_sets_no_recal(
        self, mp_pipeline: MedpipePipeline
    ) -> None:
        """Test case when there is no recalibration."""
        mock_data = load_data(mp_pipeline.medpipe_config.data.path)

        validation_config = mp_pipeline.medpipe_config.workflow.validation
        validation_config.recalibration_split = None

        _, _, _, _, X_recal, y_recal, _ = mp_pipeline._get_data_sets(mock_data)

        assert X_recal is None
        assert y_recal is None

    def test_pipeline_get_data_sets_no_groups(
        self, mp_pipeline: MedpipePipeline
    ) -> None:
        """Test case when there are no cross-validation groups."""
        mock_data = load_data(mp_pipeline.medpipe_config.data.path)

        validation_config = mp_pipeline.medpipe_config.workflow.validation
        validation_config.cross_validation.group_column = None

        _, _, _, _, _, _, groups = mp_pipeline._get_data_sets(mock_data)

        assert groups is None

    @pytest.mark.parametrize(
        "data",
        [
            3.14,
            42,
            "llama",
            [],
            {},
            (),
            np.array([]),
        ],
    )
    def test_pipeline_get_data_sets_invalid_data(
        self, mp_pipeline: MedpipePipeline, data: Any
    ) -> None:
        """Test case when the data is not pd.DataFrame."""
        match_expr = f"data should be a pd.DataFrame, but got {type(data)}"
        with pytest.raises(TypeError, match=match_expr):
            mp_pipeline._get_data_sets(data)


class TestGetCvGenerator:
    """Test class for the _get_cv_generator function of the
    MedpipePipeline class."""

    @pytest.mark.parametrize("strategy", ["random", "group"])
    def test_pipeline_get_cv_generator_success(
        self, mp_pipeline: MedpipePipeline, strategy: Literal["random", "group"]
    ) -> None:
        """Test successful function call."""
        cv_config = mp_pipeline.medpipe_config.workflow.validation.cross_validation
        cv_config.strategy = strategy  # Test both strategies
        cv_generator = mp_pipeline._get_cv_generator()

        if strategy == "random":
            assert isinstance(cv_generator, StratifiedKFold)
        if strategy == "group":
            assert isinstance(cv_generator, GroupKFold)
        assert cv_generator.n_splits == cv_config.n_splits
        assert cv_generator.shuffle == cv_config.shuffle
        assert cv_generator.random_state == cv_config.random_state
