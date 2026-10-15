"""
Test functions for the RegressorWorkflowConfig schema of the config module.
"""

import pytest
from pydantic import ValidationError

from medpipe.utils.config import RegressorWorkflowConfig


class TestRegressorWorkflowConfig:
    """Test class for the RegressorWorkflowConfig class"""

    def _get_valid_config_dict(self, **overrides) -> dict:
        """Creates a fresh valid config dict to override."""
        config_dict = {
            "random_state": None,
            "n_jobs": 1,
            "preprocessing": {
                "preprocess": True,
                "operations": [
                    {
                        "name": "StandardScaler",
                        "columns": ["AGE", "BMI"],
                        "with_mean": True,
                    },
                ],
            },
            "validation": {
                "test_split": {
                    "strategy": "group",
                    "test_size": None,
                    "group_column": "OP_YEAR",
                    "values": [2023],
                },
                "recalibration_split": {
                    "strategy": "group",
                    "group_column": "OP_YEAR",
                    "values": [2024],
                    "recalibration_size": None,
                },
                "cross_validation": {
                    "strategy": "group",
                    "grid_search": None,
                    "group_column": "DHB_NAME",
                    "n_splits": 2,
                    "shuffle": True,
                },
            },
            "evaluation": {
                "metrics": {
                    "metrics": ["rmse", "mae"],
                    "ci_level": 0.95,
                    "n_bootstraps": 200,
                },
                "fairness": {
                    "strata": ["AGE", "SEX"],
                    "groups": {"AGE": [[18, 50], [51, 120]]},
                },
            },
        }
        config_dict.update(overrides)

        return config_dict

    def test_valid_config(self) -> None:
        """Pass valid configuration to RegressorWorkflowConfig."""
        raw_config = self._get_valid_config_dict()
        config = RegressorWorkflowConfig.model_validate(raw_config)

        assert config.model_dump() == raw_config

    def test_preprocessing_optional(self) -> None:
        """Test that preprocessing defaults to None when omitted."""
        raw_config = self._get_valid_config_dict(preprocessing=None)
        config = RegressorWorkflowConfig.model_validate(raw_config)

        assert config.preprocessing is None

    def test_default_random_state_and_n_jobs(self) -> None:
        """Test that random_state and n_jobs default to 42 and 1 respectively."""
        raw_config = self._get_valid_config_dict()
        del raw_config["random_state"]
        del raw_config["n_jobs"]

        config = RegressorWorkflowConfig.model_validate(raw_config)

        assert config.random_state == 42
        assert config.n_jobs == 1

    def test_validation_required(self) -> None:
        """Test that validation is a required field."""
        raw_config = self._get_valid_config_dict()
        del raw_config["validation"]

        with pytest.raises(ValidationError, match="Field required"):
            RegressorWorkflowConfig.model_validate(raw_config)

    def test_evaluation_required(self) -> None:
        """Test that evaluation is a required field."""
        raw_config = self._get_valid_config_dict()
        del raw_config["evaluation"]

        with pytest.raises(ValidationError, match="Field required"):
            RegressorWorkflowConfig.model_validate(raw_config)

    def test_extra_fields_forbidden(self) -> None:
        """Test that extra fields at the workflow config level are forbidden."""
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            RegressorWorkflowConfig.model_validate(
                self._get_valid_config_dict(unexpected_flag=True)
            )
