"""
Configuration function and classes tests suite.

"""

from re import escape

import pytest
from pydantic import ValidationError

from medpipe.utils.config import (
    CalibrationConfig,
    DataConfig,
    MetaConfig,
    ModelConfig,
    PathsConfig,
    PreprocessingConfig,
    PreprocessOperationConfig,
    TopLevelConfig,
)

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


class TestModelConfig:
    """Test class for the ModelConfig class"""

    def _get_valid_config_dict(self, **overrides) -> dict:
        """Creates a fresh valid config dict to override."""
        config_dict = {
            "algorithm": "HistGradientBoostingClassifier",
        }
        config_dict.update(overrides)

        return config_dict

    def test_valid_config(self) -> None:
        """Pass valid configuration to ModelConfig."""
        ModelConfig.model_validate(self._get_valid_config_dict())


class TestCalibrationConfig:
    """Test class for the CalibrationConfig class"""

    def _get_valid_config_dict(self, **overrides) -> dict:
        """Creates a fresh valid config dict to override."""
        config_dict = {
            "method": "isotonic",
        }
        config_dict.update(overrides)

        return config_dict

    def test_valid_config(self) -> None:
        """Pass valid configuration to CalibrationConfig."""
        CalibrationConfig.model_validate(self._get_valid_config_dict())


class TestTopLevelConfig:
    """Test class for the TopLevelConfig class"""

    def _get_valid_config_dict(self, **overrides) -> dict:
        """Creates a fresh valid config dict to override."""
        config_dict = {
            "meta": {
                "version": "v0.0.1",
                "project_name": "mepipe-test",
                "run_mode": "audit",
            },
            "paths": {
                "config_dir": "path/to/config",
                "model_dir": "path/to/models",
                "figure_dir": "path/to/figures",
            },
            "model": {"algorithm": "HistGradientBoostingClassifier"},
            "calibration": {"method": "isotonic"},
        }
        config_dict.update(overrides)

        return config_dict

    def test_valid_config(self) -> None:
        """Pass valid configuration to TopLevelConfig."""
        TopLevelConfig.model_validate(self._get_valid_config_dict())


class TestDataConfig:
    """Test class for the DataConfig class"""

    def _get_valid_config_dict(self, **overrides) -> dict:
        """Creates a fresh valid config dict to override."""
        config_dict = {
            "path": "path/to/data",
            "predictors": ["AGE", "SEX", "OP_SEVERITY"],
            "outcomes": ["MORTALITY_30D"],
        }

        config_dict.update(overrides)

        return config_dict

    def test_valid_config(self) -> None:
        """Pass valid configuration to DataConfig."""
        DataConfig.model_validate(self._get_valid_config_dict())

    def test_validate_path(self) -> None:
        """Test data path is not a file."""
        with pytest.raises(
            ValidationError,
            match=f"path should be a directory, but got suffix .csv",
        ):
            DataConfig.model_validate(self._get_valid_config_dict(path="path.csv"))

    @pytest.mark.parametrize(
        "predictors, outcomes, overlap",
        [
            (["AGE", "SEX", "ANY_COMP"], ["ANY_COMP"], ["ANY_COMP"]),
            (["AGE", "SEX", "ANY_COMP"], ["AGE", "SEX"], ["AGE", "SEX"]),
        ],
    )
    def test_data_leakage(
        self, predictors: list[str], outcomes: list[str], overlap: list[str]
    ) -> None:
        """Test data leakage safety check."""
        match_expr = "Overlap between predictors and outcomes which will "
        f"break model validity: {overlap}"

        with pytest.raises(ValidationError, match=escape(match_expr)):
            DataConfig.model_validate(
                self._get_valid_config_dict(predictors=predictors, outcomes=outcomes)
            )


class TestPreprocessOperationConfig:
    """Test class for the PreprocessOperationConfig class"""

    def _get_valid_config_dict(self, **overrides) -> dict:
        """Creates a fresh valid config dict to override."""
        config_dict = {
            "name": "OrdinalEncoder",
            "feature_list": ["SEX", "ETHNICITY"],
        }

        config_dict.update(overrides)

        return config_dict

    def test_valid_config(self) -> None:
        """Pass valid configuration to PreprocessOperationConfig."""
        PreprocessOperationConfig.model_validate(self._get_valid_config_dict())


class TestPreprocessingConfig:
    """Test class for the PreprocessingConfig class"""

    def _get_valid_config_dict(self, **overrides) -> dict:
        """Creates a fresh valid config dict to override."""
        config_dict = {
            "preprocess": True,
            "operations": [
                {
                    "name": "OrdinalEncoder",
                    "feature_list": ["SEX", "ETHNICITY"],
                },
                {
                    "name": "StandarScaler",
                    "feature_list": ["SEX", "ETHNICITY"],
                    "with_mean": True,
                },
            ],
        }

        config_dict.update(overrides)

        return config_dict

    def test_valid_config(self) -> None:
        """Pass valid configuration to PreprocessingConfig."""
        PreprocessingConfig.model_validate(self._get_valid_config_dict())

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
