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
    SplitRecalibrationConfig,
    SplitTestConfig,
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


class TestSplitTestConfig:
    """Test class for the SplitTestConfig class"""

    def _get_valid_random_config_dict(self, **overrides) -> dict:
        """Creates a fresh valid config dict with random strategy to override."""
        config_dict = {
            "strategy": "random",
            "group_column": None,
            "values": None,
            "test_size": 0.1,
            "drop_group_column": None,
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
            "drop_group_column": True,
        }

        config_dict.update(overrides)

        return config_dict

    def test_valid_config(self) -> None:
        """Pass valid configuration to SplitTestConfig."""
        SplitTestConfig.model_validate(self._get_valid_random_config_dict())
        SplitTestConfig.model_validate(self._get_valid_group_config_dict())

        SplitTestConfig.model_validate(
            self._get_valid_group_config_dict(strategy="random", test_size=0.1)
        )
        SplitTestConfig.model_validate(self._get_valid_group_config_dict(test_size=0.1))

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
        "group_column, values, drop_group_column, match_expr",
        [
            (None, [2024], True, "a group column to be specified"),
            ("OP_YEAR", [], True, "values to be specified"),
            ("OP_YEAR", None, True, "values to be specified"),
            ("OP_YEAR", [2024], None, "the drop flag to be specified"),
        ],
    )
    def test_group_stragey_interactions(
        self,
        group_column: str | None,
        values: list[str | int] | None,
        drop_group_column: bool | None,
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
                    drop_group_column=drop_group_column,
                )
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
            "drop_group_column": None,
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
            "drop_group_column": True,
        }

        config_dict.update(overrides)

        return config_dict

    def test_valid_config(self) -> None:
        """Pass valid configuration to SplitRecalibrationConfig."""
        SplitRecalibrationConfig.model_validate(self._get_valid_random_config_dict())
        SplitRecalibrationConfig.model_validate(self._get_valid_group_config_dict())

        # Test unmatched parameters
        SplitRecalibrationConfig.model_validate(
            self._get_valid_group_config_dict(strategy="random", recalibration_size=0.1)
        )
        SplitRecalibrationConfig.model_validate(
            self._get_valid_group_config_dict(recalibration_size=0.1)
        )

        # Test None case
        SplitRecalibrationConfig.model_validate({})

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
        "group_column, values, drop_group_column, match_expr",
        [
            (None, [2024], True, "a group column to be specified"),
            ("OP_YEAR", [], True, "values to be specified"),
            ("OP_YEAR", None, True, "values to be specified"),
            ("OP_YEAR", [2024], None, "the drop flag to be specified"),
        ],
    )
    def test_group_stragey_interactions(
        self,
        group_column: str | None,
        values: list[str | int] | None,
        drop_group_column: bool | None,
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
                    drop_group_column=drop_group_column,
                )
            )
