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

    def test_read_configuration(self) -> None:
        """Test successful function call."""
        base_dir = Path(__file__).parent.parent.parent

        example_config_dir = base_dir / "config-examples/"
        read_toml_configuration(example_config_dir / "medpipe.toml")


class TestLoadData:
    """Test class for the load_data function."""

    @pytest.mark.parametrize("file", ["dummy.csv", "dummy.parquet"])
    def test_load_data_success(self, tmp_path: Path, file: str) -> None:
        """Test successful function call."""
        # Write some dummy data
        dummy_data = pd.DataFrame([0, 1, 2, 3])
        dummy_data.to_csv(tmp_path / "dummy.csv")
        dummy_data.to_parquet(tmp_path / "dummy.parquet")

        data = load_data(tmp_path / file)

        assert isinstance(data, pd.DataFrame)
