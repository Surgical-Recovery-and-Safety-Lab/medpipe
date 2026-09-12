"""
Shared fixtures for the medpipe.utils.io test suite.
"""

from collections.abc import Generator

import pandas as pd
import pytest

from medpipe.utils.io import DataLoaderRegistry


@pytest.fixture(autouse=True)
def restore_registry_state() -> Generator:
    """Fixture to snapshot and restore DataLoaderRegistry state after each test."""
    original_registry = DataLoaderRegistry._registry.copy()
    yield
    DataLoaderRegistry._registry.clear()
    DataLoaderRegistry._registry.update(original_registry)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Provides a small dummy DataFrame for file writing/reading tests."""
    return pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
