#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exception functions tests suite.
"""

from __future__ import annotations

from pathlib import Path
from re import escape
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from pandas import Series

from medpipe.utils.exceptions import (
    array_check,
    array_dim_check,
    file_checks,
    path_checks,
)

if TYPE_CHECKING:
    import numpy.typing as npt


class TestFileChecks:
    """Test class for the file_checks function."""

    @pytest.fixture
    def example_config_dir(self) -> Path:
        """Provide the location of the example configuration files."""
        base_dir = Path(__file__).parent.parent.parent

        return base_dir / "examples/"

    @pytest.mark.parametrize(
        "file_name, extension, exists",
        [
            ("default_config.toml", ".toml", True),
            ("not_a_file.csv", ".csv", False),
            (Path("data.parquet"), [".csv", ".parquet"], False),
        ],
    )
    def test_file_checks_success(
        self,
        file_name: str | Path,
        extension: str | list[str],
        exists: bool,
        example_config_dir: Path,
    ) -> None:
        """Test successful function call."""
        file_checks(example_config_dir / file_name, extension, exists)

    @pytest.mark.parametrize("file", [42, 3.14, ("a", 1), {1: "a"}, [1, 2]])
    def test_file_checks_not_str(self, file: Any) -> None:
        """Test case when file is not a string or Path."""
        match_expr = f"File should be a string or Path"
        with pytest.raises(TypeError, match=match_expr):
            file_checks(file, ".txt")

    @pytest.mark.parametrize("extension", [42, 3.14, ("a", 1), {1: "a"}])
    def test_file_checks_extension_type_error(self, extension: Any) -> None:
        """Test case when extension is not a string or list of strings."""
        match_expr = f"Extension should be a string or list of strings"
        with pytest.raises(TypeError, match=match_expr):
            file_checks("file.txt", extension)

    @pytest.mark.parametrize(
        "file, extension, match_expr",
        [
            (
                "file.txt",
                ".toml",
                "File suffix should be .toml, but got .txt",
            ),
            (
                "data.csv",
                ".parquet",
                "File suffix should be .parquet, but got .csv",
            ),
            (
                "config.toml",
                [".csv", ".parquet"],
                "File suffix should be one of ['.csv', '.parquet'], but got .toml",
            ),
        ],
    )
    def test_file_checks_incorrect_extension(
        self, file: str | Path, extension: str | list[str], match_expr: str
    ) -> None:
        """Test case when extension is incorrect."""
        with pytest.raises(ValueError, match=escape(match_expr)):
            file_checks(file, extension, False)

    @pytest.mark.parametrize(
        "file, extension",
        [
            (Path("not_config.toml"), ".toml"),
            ("not_data.parquet", [".csv", ".parquet"]),
        ],
    )
    def test_file_checks_file_not_found(
        self, file: str | Path, extension: str | list[str]
    ) -> None:
        """Test case when file is not found."""
        with pytest.raises(FileNotFoundError, match=f"{file} does not exist"):
            file_checks(file, extension)

    def test_file_checks_is_directory(self, example_config_dir: Path) -> None:
        """Test case when file points to a directory."""
        with pytest.raises(
            IsADirectoryError, match=f"{example_config_dir} should be a file"
        ):
            file_checks(example_config_dir, ".txt")


class TestPathChecks:
    """Test class for the path_checks function."""

    def test_path_checks_success(self, tmp_path: Path) -> None:
        """Test successful function call."""
        path_checks(str(tmp_path))
        path_checks(tmp_path)
        path_checks(tmp_path / "v0.1.2")  # Check that version numbers are ok

        assert tmp_path.exists()  # Check that creation occured

    @pytest.mark.parametrize(
        "wrong_type",
        [42, 3.14, [1, 2, 3], None],
    )
    def test_path_checks_type(self, wrong_type: Any) -> None:
        """Test case when path is not a string."""
        with pytest.raises(
            TypeError,
            match="Path should be a string or a Path",
        ):
            path_checks(wrong_type)

    @pytest.mark.parametrize(
        "file",
        ["config.toml", "data.csv"],
    )
    def test_path_checks_is_a_file(self, tmp_path: Path, file: str | Path) -> None:
        """Test case when path points to a file."""
        # Create the file in tmp_path
        file_path = tmp_path / file
        file_path.touch()
        with pytest.raises(
            NotADirectoryError,
            match=f"{file_path} should be a directory",
        ):
            path_checks(file_path)


class TestArrayCheck:
    """Test class for the array_check function."""

    @pytest.mark.parametrize(
        "arr",
        [
            np.array([]),
            np.array([1]),
            np.array([[1, 2], [1, 2]]),
            np.array([[1], [2]]),
            [1, 2, 3],  # List
            Series([1, 2]),  # Series
        ],
    )
    def test_array_check_success(self, arr: npt.NDArray | Series) -> None:
        """Test successful function call."""
        array_check(arr)

    @pytest.mark.parametrize(
        "not_arr",
        [
            5,  # Int
            3.14,  # Float
            "not_an_arr",  # Str
            {"A", 1},  # Dict
            (42, 3.14, "pi"),  # tuple
        ],
    )
    def test_array_check_not_array(self, not_arr: Any) -> None:
        """Test case when argument is not an array."""
        match_expr = f"Input should be an array-like but instead got {type(not_arr)}"
        with pytest.raises(TypeError, match=match_expr):
            array_check(not_arr)


class TestArrayDimCheck:
    """Test class for the array_dim_check function."""

    @pytest.mark.parametrize(
        "arr1, arr2, dim",
        [
            (np.array([]), np.array([]), None),
            (Series([]), Series([]), None),
            (np.array([1]), np.array([2]), None),
            (Series([1]), Series([2]), None),
            (np.array([1, 2]), np.array([2, 3]), None),
            (np.array([1, 2]), np.array([2, 3]), 0),
            (np.array([[1], [2]]), np.array([[1], [3]]), None),
            (np.array([[1], [2]]), np.array([[1], [3]]), 1),
            (np.array([[1, 2], [1, 2]]), np.array([[1, 4], [2, 3]]), None),
            (np.array([[1, 2], [1, 2]]), np.array([[1, 4], [2, 3]]), 0),
            (np.array([[1, 2], [1, 2]]), np.array([[1, 4], [2, 3]]), 1),
        ],
    )
    def test_array_dim_check_success(
        self, arr1: npt.NDArray | Series, arr2: npt.NDArray | Series, dim: int | None
    ) -> None:
        """Test successful function call."""
        array_dim_check(arr1, arr2, dim)

    @pytest.mark.parametrize(
        "arr1, arr2, dim, match_expr",
        [
            (np.array([]), np.array([1]), None, "The dimensions do not agree"),
            (
                np.array([1]),
                np.array([2, 2]),
                None,
                "The dimensions do not agree",
            ),
            (np.array([1, 2]), np.array([3]), 0, "The 0 axis does not agree"),
            (Series([1, 2]), Series([3]), 0, "The 0 axis does not agree"),
            (
                np.array([[1], [2]]),
                np.array([[1, 1], [2, 2]]),
                None,
                "The dimensions do not agree",
            ),
            (
                np.array([[1], [2]]),
                np.array([[1, 1], [2, 2]]),
                1,
                "The 1 axis does not agree",
            ),
            (
                np.array([[1, 2], [1, 2]]),
                np.array([1, 4]),
                None,
                "The dimensions do not agree",
            ),
        ],
    )
    def test_array_dim_check_not_equal(
        self,
        arr1: npt.NDArray | Series,
        arr2: npt.NDArray | Series,
        dim: int | None,
        match_expr: str,
    ) -> None:
        """Test case when dimensions are not equal."""
        with pytest.raises(ValueError, match=match_expr):
            array_dim_check(arr1, arr2, dim)

    @pytest.mark.parametrize(
        "arr1, arr2, dim",
        [
            (np.array([]), np.array([1]), 1),
            (np.array([1]), np.array([2, 2]), 1),
            (Series([1]), Series([2, 2]), 1),
            (np.array([1, 2]), np.array([2, 3]), 1),
            (np.array([[1], [2]]), np.array([[1], [3]]), 2),
        ],
    )
    def test_array_dim_check_index_error(
        self, arr1: npt.NDArray | Series, arr2: npt.NDArray | Series, dim: int
    ) -> None:
        """Test case when dimension does not exist."""
        with pytest.raises(IndexError):
            array_dim_check(arr1, arr2, dim)

    @pytest.mark.parametrize(
        "dim",
        [
            3.14,  # Float
            "not_an_arr",  # Str
            [1, 2, 3],  # List
            ("arr", 1),  # Tuple
        ],
    )
    def test_array_check_incorrect_dim(self, dim: Any) -> None:
        """Test case when the dimesion is incorrect."""
        with pytest.raises(TypeError, match="Input dim should be an integer"):
            array_dim_check(np.array([]), np.array([]), dim)
