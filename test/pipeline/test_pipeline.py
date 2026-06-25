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

