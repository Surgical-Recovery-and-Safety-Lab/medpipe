#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
I/O functions and classes tests suite.
"""

from pathlib import Path
from typing import Any, Generator
from unittest.mock import patch

import pandas as pd
import pytest

from medpipe.utils.io import load_data, read_toml_configuration


class TestReadTOMLConfiguration:
    """Test class for the read_toml_configuration function."""

    @pytest.fixture(autouse=True)
    def shield_local_filesystem(self) -> Generator[Any, Any, Any]:
        """Automatically prevent any test in this file from creating folders on disk."""
        with (
            patch("pathlib.Path.mkdir") as mock_path_mkdir,
            patch("os.makedirs") as mock_os_makedirs,
        ):
            # Relinquish control back to the test runner loop
            yield mock_path_mkdir, mock_os_makedirs

    @pytest.fixture
    def example_config_dir(self) -> Path:
        """Provide the location of the example configuration files."""
        base_dir = Path(__file__).parent.parent.parent

        return base_dir / "config-examples/"

    def test_read_configuration(self, example_config_dir: Path) -> None:
        """Test successful function call."""
        read_toml_configuration(example_config_dir / "HGBc_config.toml")


class TestLoadData:
    """Test class for the load_data function."""

    @pytest.fixture
    def write_data(self, tmp_path: Path) -> None:
        """Write a csv and parquet file to test loading."""
        dummy_data = pd.DataFrame([0, 1, 2, 3])
        dummy_data.to_csv(tmp_path / "dummy.csv")
        dummy_data.to_parquet(tmp_path / "dummy.parquet")

    @pytest.mark.parametrize("file", ["dummy.csv", "dummy.parquet"])
    def test_load_data_success(
        self, tmp_path: Path, file: str, write_data: None
    ) -> None:
        """Test successful function call."""
        data = load_data(tmp_path / file)

        assert isinstance(data, pd.DataFrame)
