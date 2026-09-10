"""
Tests for the DataLoaderRegistry class and register_data_loader function
of the medpipe.utils.io module.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from medpipe.utils.io import DataLoaderRegistry, load_data, register_data_loader


class TestDataLoaderRegistryNormalizeExt:
    """Tests for DataLoaderRegistry._normalize_ext."""

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


class TestDataLoaderRegistryListRegistered:
    """Tests for DataLoaderRegistry.list_registered."""

    def test_list_registered_contains_common_extensions(self) -> None:
        """Test listing default registered file extensions."""
        registered = DataLoaderRegistry.list_registered()
        assert ".csv" in registered
        assert ".parquet" in registered
        assert ".json" in registered
        assert ".pkl" in registered

    def test_list_registered_matches_full_default_set(self) -> None:
        """Test that the out-of-the-box registry exposes exactly the
        documented set of default extensions."""
        expected = {
            ".csv",
            ".tsv",
            ".txt",
            ".parquet",
            ".pq",
            ".feather",
            ".xlsx",
            ".xls",
            ".json",
            ".jsonl",
            ".pkl",
            ".pickle",
        }

        assert set(DataLoaderRegistry.list_registered()) == expected


class TestDataLoaderRegistryGet:
    """Tests for DataLoaderRegistry.get."""

    def test_get_valid_extension(self) -> None:
        """Test retrieving registered loader callables."""
        loader = DataLoaderRegistry.get(".csv")
        assert loader == pd.read_csv

        loader_upper = DataLoaderRegistry.get("PARQUET")
        assert loader_upper == pd.read_parquet

    @pytest.mark.parametrize(
        "ext, expected_func",
        [
            (".csv", pd.read_csv),
            (".txt", pd.read_csv),
            (".parquet", pd.read_parquet),
            (".pq", pd.read_parquet),
            (".feather", pd.read_feather),
            (".xlsx", pd.read_excel),
            (".xls", pd.read_excel),
            (".json", pd.read_json),
            (".pkl", pd.read_pickle),
            (".pickle", pd.read_pickle),
        ],
    )
    def test_get_default_mappings(self, ext: str, expected_func) -> None:
        """Test that every directly-mapped default extension resolves to
        the expected pandas reader function."""
        assert DataLoaderRegistry.get(ext) is expected_func

    def test_get_unsupported_extension_raises_value_error(self) -> None:
        """Test retrieving an unregistered extension raises ValueError."""
        with pytest.raises(
            ValueError, match="Unsupported file extension '.unsupported'"
        ):
            DataLoaderRegistry.get(".unsupported")


class TestDataLoaderRegistryRegister:
    """Tests for DataLoaderRegistry.register."""

    def test_register_decorator(self) -> None:
        """Test registering a custom loader function via the class decorator."""
        mock_func = MagicMock()

        @DataLoaderRegistry.register("custom")
        def custom_loader(filepath, **kwargs):
            return mock_func(filepath, **kwargs)

        assert ".custom" in DataLoaderRegistry.list_registered()
        assert DataLoaderRegistry.get(".custom") == custom_loader

    def test_register_normalizes_extension(self) -> None:
        """Test that the extension passed to register() is normalized
        regardless of case or a missing leading dot."""

        @DataLoaderRegistry.register("MyExt")
        def my_loader(filepath, **kwargs):
            return None

        assert DataLoaderRegistry.get(".myext") is my_loader
        assert DataLoaderRegistry.get("MYEXT") is my_loader

    def test_register_overwrites_existing_extension(
        self, tmp_path: Path, sample_df: pd.DataFrame
    ) -> None:
        """Test that re-registering an already-supported extension replaces
        the previous loader instead of erroring or duplicating it."""
        mock_loader = MagicMock(return_value=sample_df)

        DataLoaderRegistry.register(".csv")(mock_loader)

        csv_file = tmp_path / "data.csv"
        csv_file.write_text("col1,col2\n1,a\n")

        loaded_df = load_data(csv_file)

        mock_loader.assert_called_once_with(csv_file)
        pd.testing.assert_frame_equal(loaded_df, sample_df)


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
