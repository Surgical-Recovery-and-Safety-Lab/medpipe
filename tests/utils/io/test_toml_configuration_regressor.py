"""
Tests for the read_regressor_toml_configuration function of the
medpipe.utils.io module.
"""

import tomllib
from pathlib import Path

import pytest
from pydantic import ValidationError

from medpipe.utils.config import MedpipeRegressorConfig
from medpipe.utils.io import read_regressor_toml_configuration

MINIMAL_VALID_TOML = """
[meta]
project_name = "test_project"
run_mode = "fast"

[data]
path = "data.csv"
predictors = ["AGE"]
outcomes = ["LOS_DAYS"]

[default_model]
algorithm = "LinearRegression"

[workflow.validation.test_split]
strategy = "random"
test_size = 0.2

[workflow.evaluation.metrics]
metrics = ["rmse"]
"""


@pytest.fixture
def minimal_config_file(tmp_path: Path) -> Path:
    """Write a minimal, schema-valid regression TOML configuration to disk."""
    config_path = tmp_path / "config.toml"
    config_path.write_text(MINIMAL_VALID_TOML)
    return config_path


class TestReadRegressorTOMLConfiguration:
    """Test class for the read_regressor_toml_configuration function."""

    def test_read_configuration_returns_medpipe_regressor_config(
        self, minimal_config_file: Path
    ) -> None:
        """Test that a minimal valid TOML file is parsed into a fully
        populated MedpipeRegressorConfig instance."""
        config = read_regressor_toml_configuration(minimal_config_file)

        assert isinstance(config, MedpipeRegressorConfig)
        assert config.meta.project_name == "test_project"
        assert config.meta.run_mode == "fast"
        assert config.data.outcomes == ["LOS_DAYS"]
        assert config.default_model.algorithm == "LinearRegression"

    def test_read_configuration_accepts_string_path(
        self, minimal_config_file: Path
    ) -> None:
        """Test that config_file may be passed as a plain string as well
        as a Path."""
        config = read_regressor_toml_configuration(str(minimal_config_file))

        assert isinstance(config, MedpipeRegressorConfig)

    def test_read_configuration_missing_file_raises_file_not_found(
        self, tmp_path: Path
    ) -> None:
        """Test that a non-existent config path raises FileNotFoundError."""
        missing_file = tmp_path / "missing.toml"

        with pytest.raises(FileNotFoundError):
            read_regressor_toml_configuration(missing_file)

    def test_read_configuration_wrong_extension_raises_value_error(
        self, tmp_path: Path
    ) -> None:
        """Test that a config file without a .toml extension is rejected."""
        wrong_ext_file = tmp_path / "config.json"
        wrong_ext_file.write_text("{}")

        with pytest.raises(ValueError, match=r"File suffix should be \.toml"):
            read_regressor_toml_configuration(wrong_ext_file)

    def test_read_configuration_malformed_toml_raises_decode_error(
        self, tmp_path: Path
    ) -> None:
        """Test that syntactically invalid TOML content surfaces tomllib's
        own decode error rather than being silently swallowed."""
        malformed_file = tmp_path / "malformed.toml"
        malformed_file.write_text("this is not = [valid toml")

        with pytest.raises(tomllib.TOMLDecodeError):
            read_regressor_toml_configuration(malformed_file)

    def test_read_configuration_schema_violation_raises_validation_error(
        self, tmp_path: Path
    ) -> None:
        """Test that syntactically valid TOML which fails
        MedpipeRegressorConfig's schema surfaces a pydantic ValidationError."""
        invalid_file = tmp_path / "invalid.toml"
        # Missing every required field.
        invalid_file.write_text("[meta]\nproject_name = \"only_meta\"\n")

        with pytest.raises(ValidationError):
            read_regressor_toml_configuration(invalid_file)
