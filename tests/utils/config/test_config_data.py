"""
Test functions for the DataConfig schema of the config module.
"""

from pathlib import Path
from re import escape

import pytest
from pydantic import ValidationError

from medpipe.utils.config import DataConfig


class TestDataConfig:
    """Test class for the DataConfig class"""

    def _get_valid_config_dict(self, tmp_path: Path, **overrides) -> dict:
        """Creates a fresh valid config dict to override."""
        config_dict = {
            "path": str(tmp_path / "path/to/data.csv"),
            "predictors": ["AGE", "SEX", "OP_SEVERITY"],
            "outcomes": ["MORTALITY_30D"],
            "kwargs": {},
        }

        config_dict.update(overrides)

        return config_dict

    @pytest.mark.parametrize(
        "path",
        ["path/to/data.csv", "path/to/data.parquet"],
    )
    def test_valid_config(self, tmp_path: Path, path: str) -> None:
        """Pass valid configuration to DataConfig."""
        raw_config = self._get_valid_config_dict(
            tmp_path, **{"path": str(tmp_path / path)}
        )
        config = DataConfig.model_validate(raw_config)

        assert config.model_dump() == raw_config

    def test_default_kwargs(self, tmp_path: Path) -> None:
        """Test that kwargs defaults to an empty dict when omitted."""
        raw_config = self._get_valid_config_dict(tmp_path)
        del raw_config["kwargs"]

        config = DataConfig.model_validate(raw_config)

        assert config.kwargs == {}

    def test_custom_kwargs(self, tmp_path: Path) -> None:
        """Test that custom kwargs are preserved as passed."""
        raw_config = self._get_valid_config_dict(
            tmp_path, kwargs={"sep": ";", "header": 0}
        )
        config = DataConfig.model_validate(raw_config)

        assert config.kwargs == {"sep": ";", "header": 0}

    def test_validate_path(self, tmp_path: Path) -> None:
        """Test data path is not a file."""
        path = "path/"
        match_expr = "path should be a file, but got no suffix"
        with pytest.raises(ValidationError, match=match_expr):
            DataConfig.model_validate(
                self._get_valid_config_dict(tmp_path=tmp_path, path=path)
            )

    def test_data_config_no_extension(self) -> None:
        """Test that a path with no extension fails validation."""
        with pytest.raises(
            ValidationError, match="path should be a file, but got no suffix"
        ):
            DataConfig(path="dataset", predictors=["age"], outcomes=["mortality"])

    @pytest.mark.parametrize(
        "predictors, outcomes, overlap",
        [
            (["AGE", "SEX", "ANY_COMP"], ["ANY_COMP"], ["ANY_COMP"]),
            (["AGE", "SEX", "ANY_COMP"], ["AGE", "SEX"], ["AGE", "SEX"]),
        ],
    )
    def test_data_leakage(
        self,
        tmp_path: Path,
        predictors: list[str],
        outcomes: list[str],
        overlap: list[str],
    ) -> None:
        """Test data leakage safety check."""
        match_expr = "Overlap between predictors and outcomes which will "
        f"break model validity: {overlap}"

        with pytest.raises(ValidationError, match=escape(match_expr)):
            DataConfig.model_validate(
                self._get_valid_config_dict(
                    tmp_path, predictors=predictors, outcomes=outcomes
                )
            )

    def test_extra_fields_forbidden(self, tmp_path: Path) -> None:
        """Test that extra fields at the data config level are strictly forbidden."""
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            DataConfig.model_validate(
                self._get_valid_config_dict(tmp_path, unexpected_flag=True)
            )
