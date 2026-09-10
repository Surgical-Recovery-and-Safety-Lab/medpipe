"""
Tests for the load_data function of the medpipe.utils.io module.
"""

from pathlib import Path

import pandas as pd
import pytest

from medpipe.utils.io import load_data


class TestLoadDataFormats:
    """Tests for the load_data function across standard file formats."""

    def test_load_csv_success(self, tmp_path: Path, sample_df: pd.DataFrame) -> None:
        """Test reading a standard .csv file."""
        csv_file = tmp_path / "data.csv"
        sample_df.to_csv(csv_file, index=False)

        loaded_df = load_data(csv_file)
        pd.testing.assert_frame_equal(loaded_df, sample_df)

    def test_load_txt_success(self, tmp_path: Path, sample_df: pd.DataFrame) -> None:
        """Test reading a comma-separated .txt file (mapped to pd.read_csv)."""
        txt_file = tmp_path / "data.txt"
        sample_df.to_csv(txt_file, index=False)

        loaded_df = load_data(txt_file)
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

    def test_load_pq_alias_success(
        self, tmp_path: Path, sample_df: pd.DataFrame
    ) -> None:
        """Test reading a .pq file, the short alias for parquet."""
        pq_file = tmp_path / "data.pq"
        sample_df.to_parquet(pq_file, index=False)

        loaded_df = load_data(pq_file)
        pd.testing.assert_frame_equal(loaded_df, sample_df)

    def test_load_feather_success(
        self, tmp_path: Path, sample_df: pd.DataFrame
    ) -> None:
        """Test reading a .feather file."""
        feather_file = tmp_path / "data.feather"
        sample_df.to_feather(feather_file)

        loaded_df = load_data(feather_file)
        pd.testing.assert_frame_equal(loaded_df, sample_df)

    def test_load_xlsx_success(
        self, tmp_path: Path, sample_df: pd.DataFrame
    ) -> None:
        """Test reading an .xlsx file (requires the openpyxl engine)."""
        pytest.importorskip("openpyxl")
        xlsx_file = tmp_path / "data.xlsx"
        sample_df.to_excel(xlsx_file, index=False)

        loaded_df = load_data(xlsx_file)
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

    def test_load_pickle_alt_extension_success(
        self, tmp_path: Path, sample_df: pd.DataFrame
    ) -> None:
        """Test reading a pickled DataFrame using the .pickle extension alias."""
        pickle_file = tmp_path / "data.pickle"
        sample_df.to_pickle(pickle_file)

        loaded_df = load_data(pickle_file)
        pd.testing.assert_frame_equal(loaded_df, sample_df)


class TestLoadDataBehavior:
    """Tests for load_data's argument handling and error behavior."""

    def test_load_data_passes_kwargs_to_loader(
        self, tmp_path: Path, sample_df: pd.DataFrame
    ) -> None:
        """Verify that extra kwargs (e.g., usecols) are passed through to the reader."""
        csv_file = tmp_path / "data.csv"
        sample_df.to_csv(csv_file, index=False)

        loaded_df = load_data(csv_file, usecols=["col1"])

        assert list(loaded_df.columns) == ["col1"]
        assert len(loaded_df) == 3

    def test_load_data_accepts_string_path(
        self, tmp_path: Path, sample_df: pd.DataFrame
    ) -> None:
        """Test that a plain string path works the same as a Path object."""
        csv_file = tmp_path / "data.csv"
        sample_df.to_csv(csv_file, index=False)

        loaded_df = load_data(str(csv_file))
        pd.testing.assert_frame_equal(loaded_df, sample_df)

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

    def test_load_data_invalid_type_raises_type_error(self) -> None:
        """Test that a data_file argument that is neither a str nor a Path
        raises TypeError, as documented."""
        with pytest.raises(TypeError):
            load_data(123)  # type: ignore

    def test_load_data_uppercase_extension_is_accepted(
        self, tmp_path: Path, sample_df: pd.DataFrame
    ) -> None:
        """Test that an uppercase file extension like '.CSV' is accepted
        and dispatched to the same loader as '.csv', consistent with
        DataLoaderRegistry's own case-insensitive extension handling."""
        csv_file = tmp_path / "data.CSV"
        sample_df.to_csv(csv_file, index=False)

        loaded_df = load_data(csv_file)
        pd.testing.assert_frame_equal(loaded_df, sample_df)

    def test_load_data_mixed_case_extension_is_accepted(
        self, tmp_path: Path, sample_df: pd.DataFrame
    ) -> None:
        """Test that a mixed-case extension like '.Csv' is also accepted."""
        csv_file = tmp_path / "data.Csv"
        sample_df.to_csv(csv_file, index=False)

        loaded_df = load_data(csv_file)
        pd.testing.assert_frame_equal(loaded_df, sample_df)

    def test_load_data_unsupported_extension_error_reports_raw_suffix(
        self, tmp_path: Path
    ) -> None:
        """Test that the ValueError for an unsupported extension still
        reports the file's original, un-normalized suffix."""
        unsupported_file = tmp_path / "data.UNSUPPORTED"
        unsupported_file.write_text("dummy data")

        with pytest.raises(ValueError, match="but got .UNSUPPORTED"):
            load_data(unsupported_file)
