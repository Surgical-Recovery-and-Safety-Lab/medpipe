"""
Shared fixtures for the MedpipeClassifierRunner test suite.
"""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from medpipe.pipeline.orchestrator import MedpipeOrchestrator


@pytest.fixture
def mock_orchestrator() -> MagicMock:
    """Creates a mocked MedpipeOrchestrator with standard configurations."""
    orchestrator = MagicMock(spec=MedpipeOrchestrator)
    orchestrator.run_dir = Path("/fake/run/dir")
    orchestrator.build_preprocessor.return_value = None

    mock_config = MagicMock()
    mock_config.data = MagicMock()
    mock_config.workflow = MagicMock()
    mock_config.meta = MagicMock()

    mock_config.meta.run_mode = "cv"
    mock_config.meta.project_name = "test_project"
    mock_config.data.outcomes = ["MORTALITY_30D"]
    mock_config.workflow.n_jobs = 4
    mock_config.workflow.validation.cross_validation.strategy = "random"
    mock_config.workflow.validation.cross_validation.n_splits = 2
    mock_config.workflow.validation.cross_validation.random_state = 42
    mock_config.workflow.validation.cross_validation.grid_search = False
    mock_config.workflow.evaluation.metrics.metrics = ["accuracy"]

    orchestrator.config = mock_config

    orchestrator.resolved_model_configs = {
        "MORTALITY_30D": {
            "algorithm": "RandomForestClassifier",
            "hyperparameters": {"n_estimators": 5, "max_depth": 3},
            "recalibration": {"recalibrate": True, "method": "temperature"},
        }
    }
    return orchestrator


@pytest.fixture
def dummy_data() -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    """Provides small, consistent dummy data for fitting."""
    X_train = pd.DataFrame(
        {"feature1": [1, 2, 3, 4, 5, 6], "feature2": [6, 5, 4, 3, 2, 1]}
    )
    y_train = np.array([0, 1, 0, 1, 0, 1])

    X_recal = pd.DataFrame(
        {"feature1": [2, 3, 4, 5, 6, 7], "feature2": [7, 6, 5, 4, 3, 2]}
    )
    y_recal = np.array([1, 0, 1, 0, 1, 0])

    return X_train, y_train, X_recal, y_recal
