"""
Shared fixtures for the MedpipeOrchestrator test suite.
"""

from unittest.mock import MagicMock

import pytest

from medpipe.utils.config import MedpipeConfig


@pytest.fixture
def mock_config() -> MagicMock:
    """Creates a mock MedpipeConfig with necessary attributes."""
    config = MagicMock(spec=MedpipeConfig)

    # Mock Meta section
    config.meta = MagicMock()
    config.meta.project_name = "demo"
    config.meta.verbose = 0

    # Mock Data section
    config.data = MagicMock()
    config.data.path = "dummy/path/data.csv"
    config.data.predictors = ["AGE", "BMI"]

    # Mock Workflow section
    config.workflow = MagicMock()
    config.workflow.random_state = 42
    config.workflow.preprocessing = MagicMock()
    config.workflow.preprocessing.preprocess = True
    config.workflow.evaluation = MagicMock()
    config.workflow.evaluation.fairness = None

    # Mock a single operation
    op1 = MagicMock()
    op1.name = "StandardScaler"
    op1.columns = ["AGE"]
    op1.model_extra = {"with_mean": True}
    config.workflow.preprocessing.operations = [op1]

    # Mock model_dump for reproducibility artifacts
    config.model_dump.return_value = {"mocked": "config"}

    return config
