"""
Test functions for the MetaConfig schema of the config module.
"""

import pytest
from pydantic import ValidationError

from medpipe.utils.config import MetaConfig


class TestMetaConfig:
    """Test class for the MetaConfig class"""

    def _get_valid_config_dict(self, **overrides) -> dict:
        """Creates a fresh valid config dict to override."""
        config_dict = {
            "project_name": "mepipe-test",
            "run_mode": "audit",
            "verbose": "compact",
        }
        config_dict.update(overrides)

        return config_dict

    def test_valid_config(self) -> None:
        """Pass valid configuration to MetaConfig."""
        raw_config = self._get_valid_config_dict()
        config = MetaConfig.model_validate(raw_config)

        assert config.model_dump() == raw_config

    def test_default_run_mode(self) -> None:
        """Test that run_mode defaults to 'audit' when omitted."""
        config = MetaConfig.model_validate({"project_name": "demo_project"})

        assert config.run_mode == "audit"

    def test_default_verbose(self) -> None:
        """Test that verbose defaults to 'compact' when omitted from the
        configuration."""
        raw_config = {"project_name": "demo_project", "run_mode": "cv"}
        config = MetaConfig.model_validate(raw_config)

        assert config.verbose == "compact"

    @pytest.mark.parametrize(
        "valid_verbose",
        [
            "quiet",
            "compact",
            "progress",
            "info",
            "detailed",
            "debug",
            "warning",
            True,
            False,
            0,
            1,
            2,
            3,
        ],
    )
    def test_valid_verbose_options(self, valid_verbose) -> None:
        """Test valid verbosity string literals, booleans, and allowed
        integer levels."""
        raw_config = self._get_valid_config_dict(verbose=valid_verbose)
        config = MetaConfig.model_validate(raw_config)

        assert config.verbose == valid_verbose

    def test_project_name_empty(self) -> None:
        """Test that project name is not empty."""
        with pytest.raises(
            ValidationError, match=r"Project name should not be an empty string\."
        ):
            MetaConfig.model_validate(self._get_valid_config_dict(project_name=""))

    def test_meta_config_invalid_run_mode(self) -> None:
        """Test that invalid literal for run_mode raises a validation error."""
        with pytest.raises(
            ValidationError, match="Input should be 'fast', 'eval', 'cv' or 'audit'"
        ):
            MetaConfig(project_name="my_project", run_mode="unsupported_mode")  # type: ignore

    def test_invalid_verbose_string(self) -> None:
        """Test that an invalid string mode raises a ValidationError."""
        with pytest.raises(ValidationError):
            MetaConfig(project_name="my_project", verbose="ultra_verbose")  # type: ignore

    def test_invalid_verbose_integer_out_of_bounds(self) -> None:
        """Test that integers outside the allowed [0, 3] range raise a
        ValidationError."""
        with pytest.raises(ValidationError):
            MetaConfig(project_name="my_project", verbose=5)  # type: ignore

        with pytest.raises(ValidationError):
            MetaConfig(project_name="my_project", verbose=-1)  # type: ignore

    def test_extra_fields_forbidden(self) -> None:
        """Test that extra fields at the meta config level are strictly forbidden."""
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            MetaConfig.model_validate(
                self._get_valid_config_dict(unexpected_flag=True)
            )
