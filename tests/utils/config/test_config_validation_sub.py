"""
Test functions for the ValidationSubConfig schema of the config module.
"""

import pytest
from pydantic import ValidationError

from medpipe.utils.config import ValidationSubConfig


class TestValidationSubConfig:
    """Test class for the ValidationSubConfig class"""

    def _get_valid_config_dict_group(self, **overrides) -> dict:
        """Creates a fresh valid config dict with group strategy
        to override."""
        config_dict = {
            "test_split": {
                "strategy": "group",
                "group_column": "OP_YEAR",
                "values": [2023],
                "test_size": 0.1,
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
        }
        config_dict.update(overrides)

        return config_dict

    def _get_valid_config_dict_random(self, **overrides) -> dict:
        """Creates a fresh valid config dict with random strategy
        to override."""
        config_dict = {
            "test_split": {
                "strategy": "random",
                "group_column": None,
                "values": None,
                "test_size": 0.1,
            },
            "recalibration_split": {
                "strategy": "random",
                "group_column": "OP_YEAR",
                "values": [2024],
                "recalibration_size": 0.1,
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
        """Pass valid configuration to TestValidationSubConfig."""
        raw_config = self._get_valid_config_dict_group()
        config = ValidationSubConfig.model_validate(raw_config)

        raw_config = self._get_valid_config_dict_random()
        config = ValidationSubConfig.model_validate(raw_config)

        assert config.model_dump() == raw_config

    def test_optional_sub_configs_omitted(self) -> None:
        """Test that recalibration_split and cross_validation are optional."""
        raw_config = self._get_valid_config_dict_random()
        del raw_config["recalibration_split"]
        del raw_config["cross_validation"]

        config = ValidationSubConfig.model_validate(raw_config)

        assert config.recalibration_split is None
        assert config.cross_validation is None

    def test_test_split_required(self) -> None:
        """Test that test_split is a required field."""
        with pytest.raises(ValidationError, match="Field required"):
            ValidationSubConfig.model_validate({})

    @pytest.mark.parametrize(
        "strategy, recalibration_dict",
        [
            (
                "random",
                {
                    "strategy": "group",
                    "recalibration_size": None,
                    "values": [2023],
                    "group_column": "OP_YEAR",
                },
            ),
            (
                "group",
                {
                    "strategy": "random",
                    "recalibration_size": 0.1,
                    "values": None,
                    "group_column": "OP_YEAR",
                },
            ),
        ],
    )
    def test_invalid_strategies(
        self, strategy: str, recalibration_dict: dict[str, str | list[int | str] | None]
    ) -> None:
        """Test case when test and recalibration strategies differ."""
        with pytest.raises(
            ValidationError, match="Recalibration and test strategies should match"
        ):
            if strategy == "group":
                ValidationSubConfig.model_validate(
                    self._get_valid_config_dict_group(
                        **{"recalibration_split": recalibration_dict}
                    )
                )
            elif strategy == "random":
                ValidationSubConfig.model_validate(
                    self._get_valid_config_dict_random(
                        **{"recalibration_split": recalibration_dict}
                    )
                )

    def test_invalid_columns(self) -> None:
        """Test case when test and recalibration groups differ."""
        recalibration_dict = {
            "strategy": "group",
            "recalibration_size": None,
            "values": [2023],
            "group_column": "invalid",
        }
        with pytest.raises(
            ValidationError, match="Recalibration and test group columns should match"
        ):
            ValidationSubConfig.model_validate(
                self._get_valid_config_dict_group(
                    **{"recalibration_split": recalibration_dict}
                )
            )

    @pytest.mark.parametrize(
        "test_values, recal_values",
        [
            (["test"], ["test"]),
            ([2023], [2023]),
            (["test_1", "test_2"], ["test_2"]),
            ([2023, 2024], [2024]),
        ],
    )
    def test_same_values(
        self, test_values: list[str | int], recal_values: list[str | int]
    ) -> None:
        """Test case when test and recalibration values are similar."""
        recalibration_dict = {
            "strategy": "group",
            "recalibration_size": None,
            "values": recal_values,
            "group_column": "group",
        }
        test_dict = {
            "strategy": "group",
            "test_size": None,
            "values": test_values,
            "group_column": "group",
        }
        with pytest.raises(
            ValidationError, match="Recalibration and test values should be different"
        ):
            ValidationSubConfig.model_validate(
                self._get_valid_config_dict_group(
                    **{
                        "recalibration_split": recalibration_dict,
                        "test_split": test_dict,
                    }
                )
            )

    def test_extra_fields_forbidden(self) -> None:
        """Test that extra fields at the validation sub config level are
        forbidden."""
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            ValidationSubConfig.model_validate(
                self._get_valid_config_dict_random(unexpected_flag=True)
            )
