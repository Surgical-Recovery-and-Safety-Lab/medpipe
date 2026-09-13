"""
Test functions for the RegressorValidationSubConfig schema of the config
module.
"""

import pytest
from pydantic import ValidationError

from medpipe.utils.config import RegressorValidationSubConfig


class TestRegressorValidationSubConfig:
    """Test class for the RegressorValidationSubConfig class"""

    def _get_valid_config_dict(self, **overrides) -> dict:
        """Creates a fresh valid config dict to override."""
        config_dict = {
            "test_split": {
                "strategy": "random",
                "group_column": None,
                "values": None,
                "test_size": 0.1,
            },
            "cross_validation": {
                "strategy": "random",
                "grid_search": None,
                "group_column": None,
                "n_splits": 2,
                "shuffle": True,
            },
        }
        config_dict.update(overrides)

        return config_dict

    def test_valid_config(self) -> None:
        """Pass valid configuration to RegressorValidationSubConfig."""
        raw_config = self._get_valid_config_dict()
        config = RegressorValidationSubConfig.model_validate(raw_config)

        assert config.model_dump() == raw_config

    def test_cross_validation_optional(self) -> None:
        """Test that cross_validation defaults to None when omitted."""
        raw_config = self._get_valid_config_dict()
        del raw_config["cross_validation"]

        config = RegressorValidationSubConfig.model_validate(raw_config)

        assert config.cross_validation is None

    def test_test_split_required(self) -> None:
        """Test that test_split is a required field."""
        with pytest.raises(ValidationError, match="Field required"):
            RegressorValidationSubConfig.model_validate({})

    def test_no_recalibration_split_field(self) -> None:
        """Test that recalibration_split is not a recognized field, since
        post-hoc recalibration is not supported on the regression track."""
        raw_config = self._get_valid_config_dict(
            recalibration_split={
                "strategy": "random",
                "recalibration_size": 0.1,
            }
        )
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            RegressorValidationSubConfig.model_validate(raw_config)

    def test_extra_fields_forbidden(self) -> None:
        """Test that extra fields at the validation sub config level are
        forbidden."""
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            RegressorValidationSubConfig.model_validate(
                self._get_valid_config_dict(unexpected_flag=True)
            )
