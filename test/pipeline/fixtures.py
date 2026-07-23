#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline class helper functions test suites.
"""

from pathlib import Path
from typing import Any, Generator, TypeAlias
from unittest.mock import patch

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from pytest import MonkeyPatch
from sklearn.model_selection import GroupKFold, StratifiedKFold

from medpipe._types import Labels, TransformedData
from medpipe.pipeline.pipeline import MedpipePipeline
from medpipe.utils.io import read_toml_configuration

# ==============================================================================
# Fixtures for all tests
# ==============================================================================

MockData: TypeAlias = tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
MockLabels: TypeAlias = tuple[npt.NDArray, npt.NDArray, npt.NDArray]
DataPrep: TypeAlias = tuple[
    TransformedData,
    Labels,
    TransformedData | None,
    Labels | None,
    npt.NDArray | None,
    StratifiedKFold | GroupKFold,
]


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


@pytest.fixture
def mock_data() -> MockData:
    """Generate some mock data for the some tests."""
    X_train = pd.DataFrame(
        {
            "SEX": ["F", "M", "M", "F", "M", "M", "F"],
            "AGE": [20, 43, 84, 19, 43, 28, 71],
            "PRIOR_CANCER": [0, 1, 0, 1, 0, 0, 0],
            "ADMISSION_ACUITY": [
                "Acute",
                "Elective",
                "Acute",
                "Acute",
                "Elective",
                "Elective",
                "Elective",
            ],
            "ADMISSION_SOURCE": [
                "Routine",
                "Routine",
                "Routine",
                "Transfer",
                "Routine",
                "Transfer",
                "Routine",
            ],
            "CATEGORY_LEVEL_1": [
                "Plastics",
                "Plastics",
                "Plastics",
                "General Surgery",
                "General Surgery",
                "General Surgery",
                "Plastics",
            ],
            "OP_SEVERITY": [1, 3, 2, 1, 1, 2, 4],
            "OP_YEAR": [2022, 2021, 2020, 2022, 2021, 2022, 2020],
            "DHB_NAME": [
                "Auckland",
                "Christchurch",
                "Wellington",
                "Auckland",
                "Auckland",
                "Wellington",
                "Christchurch",
            ],
        }
    )
    X_test = pd.DataFrame(
        {
            "SEX": ["M", "M", "F"],
            "AGE": [69, 43, 24],
            "PRIOR_CANCER": [0, 1, 0],
            "ADMISSION_ACUITY": ["Acute", "Elective", "Elective"],
            "ADMISSION_SOURCE": ["Transfer", "Transfer", "Routine"],
            "CATEGORY_LEVEL_1": ["Plastics", "General Surgery", "General Surgery"],
            "OP_SEVERITY": [1, 4, 2],
            "OP_YEAR": [2024, 2024, 2024],
            "DHB_NAME": ["Auckland", "Wellington", "Auckland"],
        }
    )
    X_recal = pd.DataFrame(
        {
            "SEX": ["M", "F", "F"],
            "AGE": [23, 45, 72],
            "PRIOR_CANCER": [0, 1, 1],
            "ADMISSION_ACUITY": ["Elective", "Acute", "Elective"],
            "ADMISSION_SOURCE": ["Routine", "Routine", "Routine"],
            "CATEGORY_LEVEL_1": ["Plastics", "General Surgery", "Plastics"],
            "OP_SEVERITY": [1, 2, 2],
            "OP_YEAR": [2023, 2023, 2023],
            "DHB_NAME": ["Wellington", "Christchurch", "Christchurch"],
        }
    )
    return (X_train, X_test, X_recal)


@pytest.fixture
def mock_labels(mp_pipeline: MedpipePipeline, mock_data: MockData) -> MockLabels:
    """Generate some mock data for the some tests."""
    X_train, X_test, X_recal = mock_data
    y_train: npt.NDArray = np.zeros((len(X_train), mp_pipeline.n_outcomes))
    y_test: npt.NDArray = np.zeros((len(X_test), mp_pipeline.n_outcomes))
    y_recal: npt.NDArray = np.zeros((len(X_recal), mp_pipeline.n_outcomes))

    # Make at least one example positive
    y_test[0, :] = 1
    y_train[0, :] = 1
    y_recal[0, :] = 1

    return (y_train, y_test, y_recal)


@pytest.fixture
def cv_data_prep(mp_pipeline: MedpipePipeline, mock_data: MockData) -> DataPrep:
    """Run preparation code before calling _cv_fit."""
    # Get different split data sets
    X_train, X_test, X_recal = mock_data
    y_train = np.array([1, 1, 1, 1, 0, 0, 0])
    y_recal = np.array([0, 1, 1])

    X_train, X_test, X_recal, groups = mp_pipeline._drop_group_columns(
        X_train, X_test, X_recal
    )
    X_train, X_recal = mp_pipeline._prepare_features(X_train, X_recal)
    # Create cross-validation generator
    cv_generator = mp_pipeline._get_cv_generator()

    return (X_train, y_train, X_recal, y_recal, groups, cv_generator)


@pytest.fixture
def mock_cv_results() -> dict[str, npt.NDArray]:
    """Generate mock cv_results."""
    cv_results = {
        "test_auroc": np.array([0.94, 0.99, 0.952]),
        "test_ici": np.array([0.01, 0.0223, 0.001]),
        "recal_auroc": np.array([0.9900, 0.93, 0.92]),
        "recal_ici": np.array([0.01, 0.01, 0.00]),
    }
    return cv_results


def _mp_pipeline(
    monkeypatch: MonkeyPatch,
    example_config_dir: Path,
    top_level_config: str,
    version: list[str],
) -> MedpipePipeline:
    """
    Creates a mp_pipeline with specific version numbers.

    Parameters
    ----------
    monkeypatch : MonkeyPatch
        Patch for the parse_version_number function.
    example_config_dir : Path
        Path to the example configuration directory.
    top_level_config : str
        Name of the top-level file to load.
    version : list[str]
        List of version numbers to load subconfigurations.

    Returns
    -------
    mp_pipeline : MedpipePipeline
        MedpipePipeline object.

    """
    monkeypatch.setattr(  # Patch to get the desired version number
        "medpipe.utils.io.parse_version_number", lambda v_list: version
    )
    config = read_toml_configuration(example_config_dir / top_level_config)
    config.top_level.meta.version = "v" + ".".join(version)
    return MedpipePipeline(config, logger=None)
