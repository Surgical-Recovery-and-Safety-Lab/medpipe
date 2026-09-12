"""
Test functions for the CrossValConfig schema of the config module.
"""

from typing import Literal

import pytest
from pydantic import ValidationError

from medpipe.utils.config import CrossValConfig


class TestCrossValConfig:
    """Test class for the CrossValConfig class"""

    def _get_valid_random_config_dict(self, **overrides) -> dict:
        """Creates a fresh valid config dict with random strategy to override."""
        config_dict = {
            "strategy": "random",
            "grid_search": None,
            "group_column": None,
            "n_splits": 2,
            "shuffle": True,
        }

        config_dict.update(overrides)

        return config_dict

    def _get_valid_group_config_dict(self, **overrides) -> dict:
        """Creates a fresh valid config dict with group strategy to override."""
        config_dict = {
            "strategy": "group",
            "grid_search": True,
            "group_column": "DHB_NAME",
            "n_splits": 2,
            "shuffle": True,
        }

        config_dict.update(overrides)

        return config_dict

    def test_valid_config_random(self) -> None:
        """Pass valid configuration from _get_valid_random_config_dict
        to CrossValConfig."""
        raw_config = self._get_valid_random_config_dict()
        config = CrossValConfig.model_validate(raw_config)

        assert config.model_dump() == raw_config

    @pytest.mark.parametrize(
        "strategy, group_column",
        [
            ("group", "DHB_NAME"),
            ("random", "DHB_NAME"),
            ("random", None),
        ],
    )
    def test_valid_config_group(
        self,
        strategy: Literal["group", "random"],
        group_column: str | None,
    ) -> None:
        """Pass valid configuration from _get_valid_group_config_dict
        to CrossValConfig."""
        raw_config = self._get_valid_group_config_dict(
            strategy=strategy,
            group_column=group_column,
        )
        config = CrossValConfig.model_validate(raw_config)

        assert config.model_dump() == raw_config

    def test_default_values(self) -> None:
        """Test that optional fields default to None when only strategy is given."""
        config = CrossValConfig.model_validate({"strategy": "random"})

        assert config.grid_search is None
        assert config.group_column is None
        assert config.n_splits is None
        assert config.shuffle is None

    def test_strategy_required(self) -> None:
        """Test that strategy is a required field."""
        with pytest.raises(ValidationError, match="Field required"):
            CrossValConfig.model_validate({})

    def test_invalid_strategy(self) -> None:
        """Test that an invalid strategy literal raises a ValidationError."""
        with pytest.raises(
            ValidationError, match="Input should be 'random' or 'group'"
        ):
            CrossValConfig.model_validate(
                self._get_valid_random_config_dict(strategy="unsupported")
            )

    @pytest.mark.parametrize("strategy", ["random", "group"])
    def test_n_splits_limits(self, strategy: str) -> None:
        """Test the n_splits limits strategy."""
        config_dict = {}

        if strategy == "random":
            config_dict = self._get_valid_random_config_dict(n_splits=-5)
        elif strategy == "group":
            config_dict = self._get_valid_group_config_dict(n_splits=-5)

        with pytest.raises(
            ValidationError, match="Input should be greater than or equal to 2"
        ):
            CrossValConfig.model_validate(config_dict)

    def test_group_stragey_interactions(self) -> None:
        """Tests interactions between group stragey flag and
        other parameters."""
        with pytest.raises(
            ValidationError,
            match="The group strategy requires a group column to be specified",
        ):
            CrossValConfig.model_validate(
                self._get_valid_group_config_dict(group_column=None)
            )

    def test_extra_fields_forbidden(self) -> None:
        """Test that extra fields at the cross validation config level are
        forbidden."""
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            CrossValConfig.model_validate(
                self._get_valid_random_config_dict(unexpected_flag=True)
            )
