"""
I/O functions and classes tests suite.
"""

from pathlib import Path
from typing import Any, Generator
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from medpipe.utils.io import (
    DataLoaderRegistry,
    load_data,
    read_toml_configuration,
    register_data_loader,
)


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

        example_config_dir = base_dir / "examples/"
        read_toml_configuration(example_config_dir / "default_config.toml")


@pytest.fixture(autouse=True)
def restore_registry_state():
    """Fixture to snapshot and restore DataLoaderRegistry state after each test."""
    original_registry = DataLoaderRegistry._registry.copy()
    yield
    DataLoaderRegistry._registry.clear()
    DataLoaderRegistry._registry.update(original_registry)


@pytest.fixture
def sample_df():
    """Provides a small dummy DataFrame for file writing/reading tests."""
    return pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})


class TestDataLoaderRegistry:
    """Tests for the DataLoaderRegistry class methods."""

    @pytest.mark.parametrize(
        "raw_ext, expected",
        [
            ("feather", ".feather"),
            (".FEATHER", ".feather"),
            (" .parquet ", ".parquet"),
            ("CSV", ".csv"),
            (".CustomExt", ".customext"),
        ],
    )
    def test_normalize_ext(self, raw_ext: str, expected: str) -> None:
        """Test extension string normalization (lowercasing, trimming, leading dot)."""
        assert DataLoaderRegistry._normalize_ext(raw_ext) == expected

    def test_list_registered(self) -> None:
        """Test listing default registered file extensions."""
        registered = DataLoaderRegistry.list_registered()
        assert ".csv" in registered
        assert ".parquet" in registered
        assert ".json" in registered
        assert ".pkl" in registered

    def test_get_valid_extension(self) -> None:
        """Test retrieving registered loader callables."""
        loader = DataLoaderRegistry.get(".csv")
        assert loader == pd.read_csv

        loader_upper = DataLoaderRegistry.get("PARQUET")
        assert loader_upper == pd.read_parquet

    def test_get_unsupported_extension_raises_value_error(self) -> None:
        """Test retrieving an unregistered extension raises ValueError."""
        with pytest.raises(
            ValueError, match="Unsupported file extension '.unsupported'"
        ):
            DataLoaderRegistry.get(".unsupported")

    def test_register_decorator(self) -> None:
        """Test registering a custom loader function via the class decorator."""
        mock_func = MagicMock()

        @DataLoaderRegistry.register("custom")
        def custom_loader(filepath, **kwargs):
            return mock_func(filepath, **kwargs)

        assert ".custom" in DataLoaderRegistry.list_registered()
        assert DataLoaderRegistry.get(".custom") == custom_loader


class TestRegisterDataLoader:
    """Tests for the standalone register_data_loader helper function."""

    def test_register_data_loader_adds_to_registry(self) -> None:
        """Verify register_data_loader adds an extension to DataLoaderRegistry."""
        mock_loader = MagicMock()
        register_data_loader("rds", mock_loader)

        assert ".rds" in DataLoaderRegistry.list_registered()
        assert DataLoaderRegistry.get(".rds") == mock_loader

    def test_register_data_loader_execution(
        self, tmp_path: Path, sample_df: pd.DataFrame
    ) -> None:
        """Verify load_data executes custom loader functions registered via register_data_loader."""
        custom_file = tmp_path / "test.dat"
        custom_file.write_text("dummy content")

        mock_loader = MagicMock(return_value=sample_df)
        register_data_loader(".dat", mock_loader)

        loaded_df = load_data(custom_file)

        mock_loader.assert_called_once_with(custom_file)
        pd.testing.assert_frame_equal(loaded_df, sample_df)


class TestLoadData:
    """Tests for the load_data function across standard file formats."""

    def test_load_csv_success(self, tmp_path: Path, sample_df: pd.DataFrame) -> None:
        """Test reading a standard .csv file."""
        csv_file = tmp_path / "data.csv"
        sample_df.to_csv(csv_file, index=False)

        loaded_df = load_data(csv_file)
        pd.testing.assert_frame_equal(loaded_df, sample_df)

    def test_load_tsv_success(self, tmp_path: Path, sample_df: pd.DataFrame) -> None:
        """Test reading a tab-separated .tsv file."""
        tsv_file = tmp_path / "data.tsv"
        sample_df.to_csv(tsv_file, sep="\t", index=False)

        loaded_df = load_data(tsv_file)
        pd.testing.assert_frame_equal(loaded_df, sample_df)

    def test_load_parquet_success(
        self, tmp_path: Path, sample_df: pd.DataFrame
    ) -> None:
        """Test reading a .parquet file."""
        parquet_file = tmp_path / "data.parquet"
        sample_df.to_parquet(parquet_file, index=False)

        loaded_df = load_data(parquet_file)
        pd.testing.assert_frame_equal(loaded_df, sample_df)

    def test_load_json_success(self, tmp_path: Path, sample_df: pd.DataFrame) -> None:
        """Test reading a .json file."""
        json_file = tmp_path / "data.json"
        sample_df.to_json(json_file)

        loaded_df = load_data(json_file)
        pd.testing.assert_frame_equal(loaded_df, sample_df)

    def test_load_jsonl_success(self, tmp_path: Path, sample_df: pd.DataFrame) -> None:
        """Test reading a JSON lines (.jsonl) file."""
        jsonl_file = tmp_path / "data.jsonl"
        sample_df.to_json(jsonl_file, orient="records", lines=True)

        loaded_df = load_data(jsonl_file)
        pd.testing.assert_frame_equal(loaded_df, sample_df)

    def test_load_pickle_success(self, tmp_path: Path, sample_df: pd.DataFrame) -> None:
        """Test reading a pickled (.pkl) DataFrame file."""
        pkl_file = tmp_path / "data.pkl"
        sample_df.to_pickle(pkl_file)

        loaded_df = load_data(pkl_file)
        pd.testing.assert_frame_equal(loaded_df, sample_df)

    def test_load_data_passes_kwargs_to_loader(
        self, tmp_path: Path, sample_df: pd.DataFrame
    ) -> None:
        """Verify that extra kwargs (e.g., usecols) are passed through to the reader."""
        csv_file = tmp_path / "data.csv"
        sample_df.to_csv(csv_file, index=False)

        loaded_df = load_data(csv_file, usecols=["col1"])

        assert list(loaded_df.columns) == ["col1"]
        assert len(loaded_df) == 3

    def test_load_data_unsupported_extension_raises_value_error(
        self, tmp_path: Path
    ) -> None:
        """Test attempting to load an unregistered file extension raises ValueError."""
        unsupported_file = tmp_path / "data.unsupported"
        unsupported_file.write_text("dummy data")

        with pytest.raises(ValueError, match="File suffix should be one of"):
            load_data(unsupported_file)

    def test_load_data_nonexistent_file_raises_error(self, tmp_path: Path) -> None:
        """Test providing a non-existent file path raises FileNotFoundError."""
        missing_file = tmp_path / "missing.csv"

        with pytest.raises(FileNotFoundError):
            load_data(missing_file)

    def test_load_data_directory_path_raises_error(self, tmp_path: Path) -> None:
        """Test passing a directory path raises IsADirectoryError."""
        dir_path = tmp_path / "sub_folder.csv"
        dir_path.mkdir()

        with pytest.raises(IsADirectoryError):
            load_data(dir_path)
