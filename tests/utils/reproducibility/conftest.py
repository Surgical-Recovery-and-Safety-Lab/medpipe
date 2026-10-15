"""
Shared fixtures for the medpipe.utils.reproducibility test suite.
"""

from pathlib import Path

import pytest


@pytest.fixture
def sample_config() -> dict:
    """Fixture providing a standard mock configuration dictionary."""
    return {
        "model": {"name": "RandomForest", "params": {"n_estimators": 100}},
        "data": {"target": "outcome", "test_size": 0.2},
    }


@pytest.fixture
def sample_dataset(tmp_path: Path) -> Path:
    """Fixture creating a temporary dummy dataset file."""
    dataset_file = tmp_path / "dummy_data.csv"
    dataset_file.write_text("col1,col2,outcome\n1,2,1\n3,4,0")
    return dataset_file
