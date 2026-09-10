"""
Validation functions tests suite.
"""

from __future__ import annotations

from pathlib import Path
from re import escape
from typing import Any

import pytest

from medpipe.utils.validation import file_checks, path_checks


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
