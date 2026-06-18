"""
Configuration function and classes tests suite.

"""

import pytest
from pydantic import ValidationError

from medpipe.utils.config import MetaConfig

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
