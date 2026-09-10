"""
Test functions for the PreprocessOperationConfig and PreprocessingConfig
schemas of the config module.
"""

import pytest
from pydantic import ValidationError

from medpipe.utils.config import PreprocessingConfig, PreprocessOperationConfig


class TestPreprocessOperationConfig:
    """Test class for the PreprocessOperationConfig class"""

    def _get_valid_config_dict(self, **overrides) -> dict:
        """Creates a fresh valid config dict to override."""
        config_dict = {
            "name": "OrdinalEncoder",
            "columns": ["SEX", "ETHNICITY"],
        }

        config_dict.update(overrides)

        return config_dict

    def test_valid_config(self) -> None:
        """Pass valid configuration to PreprocessOperationConfig."""
        raw_config = self._get_valid_config_dict()
        config = PreprocessOperationConfig.model_validate(raw_config)

        assert config.model_dump() == raw_config

    def test_valid_columns(self) -> None:
        """Test case where columns is an empty list."""
        with pytest.raises(ValueError, match="Columns cannot be an empty list"):
            PreprocessOperationConfig.model_validate(
                self._get_valid_config_dict(columns=[])
            )

    def test_extra_fields_allowed(self) -> None:
        """Test that extra fields are allowed to carry transformer-specific kwargs."""
        config = PreprocessOperationConfig.model_validate(
            self._get_valid_config_dict(with_mean=True)
        )

        assert getattr(config, "with_mean", None) is True


class TestPreprocessingConfig:
    """Test class for the PreprocessingConfig class"""

    def _get_valid_config_dict(self, **overrides) -> dict:
        """Creates a fresh valid config dict to override."""
        config_dict = {
            "preprocess": True,
            "operations": [
                {
                    "name": "OrdinalEncoder",
                    "columns": ["SEX", "ETHNICITY"],
                },
                {
                    "name": "StandardScaler",
                    "columns": ["SEX", "ETHNICITY"],
                    "with_mean": True,
                },
            ],
        }

        config_dict.update(overrides)

        return config_dict

    def test_valid_config(self) -> None:
        """Pass valid configuration to PreprocessingConfig."""
        raw_config = self._get_valid_config_dict()
        config = PreprocessingConfig.model_validate(raw_config)

        assert config.model_dump() == raw_config

    def test_default_values(self) -> None:
        """Test that preprocess and operations default to None when omitted."""
        config = PreprocessingConfig.model_validate({})

        assert config.preprocess is None
        assert config.operations is None

    @pytest.mark.parametrize("operations", [None, []])
    def test_validate_operations_true_flag(self, operations: None | list) -> None:
        """Test interaction between preprocess flag True and operations."""
        with pytest.raises(
            ValidationError, match="Operations must be specified if preprocess is True"
        ):
            PreprocessingConfig.model_validate(
                self._get_valid_config_dict(operations=operations)
            )

    @pytest.mark.parametrize(
        "preprocess, operations", [(False, None), (False, []), (None, None), (None, [])]
    )
    def test_validate_operations_false_flag(
        self, preprocess: None | bool, operations: None | list
    ) -> None:
        """Test interaction between preprocess flag None or False and operations."""
        PreprocessingConfig.model_validate(
            self._get_valid_config_dict(operations=operations, preprocess=preprocess)
        )

    def test_extra_fields_forbidden(self) -> None:
        """Test that extra fields at the preprocessing config level are forbidden."""
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            PreprocessingConfig.model_validate(
                self._get_valid_config_dict(unexpected_flag=True)
            )
