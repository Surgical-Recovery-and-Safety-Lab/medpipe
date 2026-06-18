"""
Configuration function and classes tests suite.

"""

import pytest
from pydantic import ValidationError

from medpipe.utils.config import MetaConfig, PathsConfig

# ==============================================================================
# SCHEMA VALIDATION TESTS
# ==============================================================================


class TestMetaConfig:
    """Test class for the MetaConfig class"""

    def _get_valid_config_dict(self, **overrides) -> dict:
        """Creates a fresh valid config dict to override."""
        config_dict = {
            "version": "v0.0.1",
            "project_name": "mepipe-test",
            "run_mode": "audit",
        }
        config_dict.update(overrides)

        return config_dict

    def test_valid_config(self) -> None:
        """Pass valid configuration to MetaConfig."""
        MetaConfig.model_validate(self._get_valid_config_dict())

    @pytest.mark.parametrize("version", ["v0.1", "not_a_version"])
    def test_version_format(self, version: str) -> None:
        """Test incorrect version number formats."""
        with pytest.raises(
            ValidationError,
            match=f"Version should be formatted as vX.Y.Z, but got {version}",
        ):
            MetaConfig.model_validate(self._get_valid_config_dict(version=version))

    def test_project_name_empty(self) -> None:
        """Test that project name is not empty."""
        with pytest.raises(
            ValidationError, match="Project name should not be an empty string."
        ):
            MetaConfig.model_validate(self._get_valid_config_dict(project_name=""))


class TestPathsConfig:
    """Test class for the PathsConfig class"""

    def _get_valid_config_dict(self, **overrides) -> dict:
        """Creates a fresh valid config dict to override."""
        config_dict = {
            "config_dir": "path/to/config",
            "model_dir": "path/to/models",
            "figure_dir": "path/to/figures",
        }
        config_dict.update(overrides)

        return config_dict

    def test_valid_config(self) -> None:
        """Pass valid configuration to PathsConfig."""
        PathsConfig.model_validate(self._get_valid_config_dict())

    def test_validate_config_dir(self) -> None:
        """Test config dir is not a file."""
        with pytest.raises(
            ValidationError,
            match=f"config_dir should be a directory, but got suffix .toml",
        ):
            PathsConfig.model_validate(
                self._get_valid_config_dict(config_dir="config.toml")
            )

    def test_validate_model_dir(self) -> None:
        """Test model dir is not a file."""
        with pytest.raises(
            ValidationError,
            match=f"model_dir should be a directory, but got suffix .joblib",
        ):
            PathsConfig.model_validate(
                self._get_valid_config_dict(model_dir="model.joblib")
            )

    def test_validate_figure_dir(self) -> None:
        """Test figure dir is not a file."""
        with pytest.raises(
            ValidationError,
            match=f"figure_dir should be a directory, but got suffix .png",
        ):
            PathsConfig.model_validate(
                self._get_valid_config_dict(figure_dir="figure.png")
            )
