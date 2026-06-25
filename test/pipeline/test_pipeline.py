#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline class tests suite.
"""

from pathlib import Path
from typing import Any, Generator
from unittest.mock import patch

import pytest
from sklearn.compose import ColumnTransformer

from medpipe.pipeline.pipeline import MedpipePipeline
from medpipe.utils.io import read_toml_configuration

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
        mp_pipeline.medpipe_config.workflow.preprocessing.preprocess = False

        assert mp_pipeline._has_preprocessor() == False

        mp_pipeline.medpipe_config.workflow.preprocessing = None

        assert mp_pipeline._has_preprocessor() == False


