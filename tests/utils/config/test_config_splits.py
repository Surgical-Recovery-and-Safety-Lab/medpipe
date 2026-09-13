"""
Test functions for the SplitTestConfig and SplitRecalibrationConfig
schemas of the config module.
"""

from typing import Literal

import pytest
from pydantic import ValidationError

from medpipe.utils.config import SplitRecalibrationConfig, SplitTestConfig


class TestSplitTestConfig:
    """Test class for the SplitTestConfig class"""

    def _get_valid_random_config_dict(self, **overrides) -> dict:
        """Creates a fresh valid config dict with random strategy to override."""
        config_dict = {
            "strategy": "random",
            "group_column": None,
            "values": None,
            "test_size": 0.1,
        }

        config_dict.update(overrides)

        return config_dict

    def _get_valid_group_config_dict(self, **overrides) -> dict:
        """Creates a fresh valid config dict with group strategy to override."""
        config_dict = {
            "strategy": "group",
            "group_column": "OP_YEAR",
            "values": [2024],
            "test_size": None,
        }

        config_dict.update(overrides)

        return config_dict

    def test_valid_config_random(self) -> None:
        """Pass valid configuration from _get_valid_random_config_dict
        to SplitTestConfig."""
        raw_config = self._get_valid_random_config_dict()
        config = SplitTestConfig.model_validate(raw_config)

        assert config.model_dump() == raw_config

    def test_default_strategy(self) -> None:
        """Test that strategy defaults to 'random' when omitted."""
        config = SplitTestConfig.model_validate({"test_size": 0.2})

        assert config.strategy == "random"

    @pytest.mark.parametrize(
        "strategy, test_size", [("group", None), ("group", 0.1), ("random", 0.1)]
    )
    def test_valid_config_group(
        self, strategy: Literal["group", "random"], test_size: float | None
    ) -> None:
        """Pass valid configuration with _get_valid_group_config_dict
        to SplitTestConfig."""
        raw_config = self._get_valid_group_config_dict(
            strategy=strategy, test_size=test_size
        )
        config = SplitTestConfig.model_validate(raw_config)

        assert config.model_dump() == raw_config

    @pytest.mark.parametrize(
        "test_size, match_expr",
        [
            (-0.1, "Input should be greater than 0"),
            (0.0, "Input should be greater than 0"),
            (1.2, "Input should be less than 1"),
            (1.0, "Input should be less than 1"),
        ],
    )
    def test_test_size_limits(self, test_size: float, match_expr: str) -> None:
        """Test the test size limits with the random strategy."""
        with pytest.raises(ValidationError, match=match_expr):
            SplitTestConfig.model_validate(
                self._get_valid_random_config_dict(test_size=test_size)
            )

    def test_random_stragey_interactions(self) -> None:
        """Tests interactions between random stragey flag and
        other parameters."""
        with pytest.raises(
            ValidationError, match="The random strategy requires a test size"
        ):
            SplitTestConfig.model_validate(
                self._get_valid_random_config_dict(test_size=None)
            )

    @pytest.mark.parametrize(
        "group_column, values, match_expr",
        [
            (None, [2024], "a group column to be specified"),
            ("OP_YEAR", [], "values to be specified"),
            ("OP_YEAR", None, "values to be specified"),
        ],
    )
    def test_group_stragey_interactions(
        self,
        group_column: str | None,
        values: list[str | int] | None,
        match_expr: str,
    ) -> None:
        """Tests interactions between group stragey flag and
        other parameters."""
        with pytest.raises(
            ValidationError, match="The group strategy requires " + match_expr
        ):
            SplitTestConfig.model_validate(
                self._get_valid_group_config_dict(
                    group_column=group_column,
                    values=values,
                )
            )

    def test_invalid_strategy(self) -> None:
        """Test that an invalid strategy literal raises a ValidationError."""
        with pytest.raises(
            ValidationError, match="Input should be 'random' or 'group'"
        ):
            SplitTestConfig.model_validate(
                self._get_valid_random_config_dict(strategy="unsupported")
            )

    def test_extra_fields_forbidden(self) -> None:
        """Test that extra fields at the split test config level are forbidden."""
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            SplitTestConfig.model_validate(
                self._get_valid_random_config_dict(unexpected_flag=True)
            )


class TestSplitRecalibrationConfig:
    """Test class for the SplitRecalibrationConfig class"""

    def _get_valid_random_config_dict(self, **overrides) -> dict:
        """Creates a fresh valid config dict with random strategy to override."""
        config_dict = {
            "strategy": "random",
            "group_column": None,
            "values": None,
            "recalibration_size": 0.1,
        }

        config_dict.update(overrides)

        return config_dict

    def _get_valid_group_config_dict(self, **overrides) -> dict:
        """Creates a fresh valid config dict with group strategy to override."""
        config_dict = {
            "strategy": "group",
            "group_column": "OP_YEAR",
            "values": [2024],
            "recalibration_size": None,
        }

        config_dict.update(overrides)

        return config_dict

    def test_valid_config_random(self) -> None:
        """Pass valid configuration from _get_valid_random_config_dict
        to SplitRecalibrationConfig."""
        raw_config = self._get_valid_random_config_dict()
        config = SplitRecalibrationConfig.model_validate(raw_config)

        assert config.model_dump() == raw_config

    @pytest.mark.parametrize(
        "strategy, recalibration_size",
        [("group", None), ("group", 0.1), ("random", 0.1)],
    )
    def test_valid_config_group(
        self, strategy: Literal["group", "random"], recalibration_size: float | None
    ) -> None:
        """Pass valid configuration with _get_valid_group_config_dict
        to SplitRecalibrationConfig."""
        raw_config = self._get_valid_group_config_dict(
            strategy=strategy, recalibration_size=recalibration_size
        )
        config = SplitRecalibrationConfig.model_validate(raw_config)

        assert config.model_dump() == raw_config

    def test_valid_config_None(self) -> None:
        """Test case when configuration is empty dictionary."""
        config = SplitRecalibrationConfig.model_validate({})

        for value in config.model_dump().values():
            assert value is None

    @pytest.mark.parametrize(
        "recalibration_size, match_expr",
        [
            (-0.1, "Input should be greater than 0"),
            (0.0, "Input should be greater than 0"),
            (1.2, "Input should be less than 1"),
            (1.0, "Input should be less than 1"),
        ],
    )
    def test_recalibration_size_limits(
        self, recalibration_size: float, match_expr: str
    ) -> None:
        """Test the test size limits with the random strategy."""
        with pytest.raises(ValidationError, match=match_expr):
            SplitRecalibrationConfig.model_validate(
                self._get_valid_random_config_dict(
                    recalibration_size=recalibration_size
                )
            )

    def test_random_stragey_interactions(self) -> None:
        """Tests interactions between random stragey flag and
        other parameters."""
        with pytest.raises(
            ValidationError, match="The random strategy requires a test size"
        ):
            SplitRecalibrationConfig.model_validate(
                self._get_valid_random_config_dict(recalibration_size=None)
            )

    @pytest.mark.parametrize(
        "group_column, values, match_expr",
        [
            (None, [2024], "a group column to be specified"),
            ("OP_YEAR", [], "values to be specified"),
            ("OP_YEAR", None, "values to be specified"),
        ],
    )
    def test_group_stragey_interactions(
        self,
        group_column: str | None,
        values: list[str | int] | None,
        match_expr: str,
    ) -> None:
        """Tests interactions between group stragey flag and
        other parameters."""
        with pytest.raises(
            ValidationError, match="The group strategy requires " + match_expr
        ):
            SplitRecalibrationConfig.model_validate(
                self._get_valid_group_config_dict(
                    group_column=group_column,
                    values=values,
                )
            )

    def test_extra_fields_forbidden(self) -> None:
        """Test that extra fields at the split recalibration config level are
        forbidden."""
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            SplitRecalibrationConfig.model_validate(
                self._get_valid_random_config_dict(unexpected_flag=True)
            )
