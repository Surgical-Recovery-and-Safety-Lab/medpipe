"""
Configuration function and classes tests suite.

"""

from pathlib import Path
from re import escape

import pytest
from pydantic import ValidationError

from medpipe.data.sampler import VALID_SAMPLER_FN
from medpipe.data.weighting import VALID_WEIGHTING_FN
from medpipe.utils.config import *

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

    def _get_valid_config_dict(self, tmp_path: Path, **overrides) -> dict:
        """Creates a fresh valid config dict to override."""
        config_dict = {
            "config_dir": str(tmp_path / "config"),
            "model_dir": str(tmp_path / "models"),
            "figure_dir": str(tmp_path / "figures"),
        }
        config_dict.update(overrides)

        return config_dict

    def test_valid_config(self, tmp_path: Path) -> None:
        """Pass valid configuration to PathsConfig."""
        PathsConfig.model_validate(self._get_valid_config_dict(tmp_path))

    @pytest.mark.parametrize(
        "parameter, path",
        [
            ("config_dir", "config.toml"),
            ("model_dir", "model.joblib"),
            ("figure_dir", "figure.png"),
        ],
    )
    def test_validate_paths(self, tmp_path: Path, parameter: str, path: str) -> None:
        """Test paths are not a file."""
        file_path = tmp_path / path
        file_path.touch()  # Write file to tmp_path

        args = {parameter: str(file_path)}  # Create new kwargs
        with pytest.raises(
            ValidationError, match=f"{file_path} points to an existing file"
        ):
            PathsConfig.model_validate(self._get_valid_config_dict(tmp_path, **args))


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
            "path": "path/to/data.csv",
            "predictors": ["AGE", "SEX", "OP_SEVERITY"],
            "outcomes": ["MORTALITY_30D"],
        }

        config_dict.update(overrides)

        return config_dict

    def test_valid_config(self) -> None:
        """Pass valid configuration to DataConfig."""
        DataConfig.model_validate(self._get_valid_config_dict())

    @pytest.mark.parametrize(
        "path, match_expr",
        [
            ("path/", "path should be a file, but got no suffix"),
            ("path.db", "path should be a .csv file, but got suffix .db"),
        ],
    )
    def test_validate_path(self, path: str, match_expr: str) -> None:
        """Test data path is not a file."""
        with pytest.raises(ValidationError, match=match_expr):
            DataConfig.model_validate(self._get_valid_config_dict(path=path))

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


class TestCrossValConfig:
    """Test class for the CrossValConfig class"""

    def _get_valid_random_config_dict(self, **overrides) -> dict:
        """Creates a fresh valid config dict with random strategy to override."""
        config_dict = {
            "strategy": "random",
            "group_column": None,
            "n_splits": 2,
            "shuffle": True,
            "random_state": 0,
            "drop_group_column": None,
        }

        config_dict.update(overrides)

        return config_dict

    def _get_valid_group_config_dict(self, **overrides) -> dict:
        """Creates a fresh valid config dict with group strategy to override."""
        config_dict = {
            "strategy": "group",
            "group_column": "DHB_NAME",
            "n_splits": 2,
            "shuffle": True,
            "random_state": 0,
            "drop_group_column": True,
        }

        config_dict.update(overrides)

        return config_dict

    def test_valid_config(self) -> None:
        """Pass valid configuration to CrossValConfig."""
        CrossValConfig.model_validate(self._get_valid_random_config_dict())
        CrossValConfig.model_validate(self._get_valid_group_config_dict())

        CrossValConfig.model_validate(
            self._get_valid_group_config_dict(
                strategy="random", group_column="DHB_NAME"
            )
        )
        CrossValConfig.model_validate(
            self._get_valid_group_config_dict(
                strategy="random", drop_group_column=False
            )
        )
        CrossValConfig.model_validate({})  # Test the None config

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

    @pytest.mark.parametrize("strategy", ["random", "group"])
    def test_random_state_limits(self, strategy: str) -> None:
        """Test the random state limits."""
        config_dict = {}

        if strategy == "random":
            config_dict = self._get_valid_random_config_dict(random_state=-5)
        elif strategy == "group":
            config_dict = self._get_valid_group_config_dict(random_state=-5)

        with pytest.raises(
            ValidationError, match="Input should be greater than or equal to 0"
        ):
            CrossValConfig.model_validate(config_dict)

    @pytest.mark.parametrize(
        "group_column, drop_group_column, match_expr",
        [
            (None, True, "a group column to be specified"),
            ("DHB_NAME", None, "the drop flag to be specified"),
        ],
    )
    def test_group_stragey_interactions(
        self,
        group_column: str | None,
        drop_group_column: bool | None,
        match_expr: str,
    ) -> None:
        """Tests interactions between group stragey flag and
        other parameters."""
        with pytest.raises(
            ValidationError, match="The group strategy requires " + match_expr
        ):
            CrossValConfig.model_validate(
                self._get_valid_group_config_dict(
                    group_column=group_column, drop_group_column=drop_group_column
                )
            )


class TestValidationSubConfig:
    """Test class for the ValidationSubConfig class"""

    def _get_valid_config_dict(self, **overrides) -> dict:
        """Creates a fresh valid config dict to override."""
        config_dict = {
            "test_split": {
                "strategy": "random",
                "test_size": 0.1,
            },
            "recalibration_split": {
                "strategy": "group",
                "group_column": "OP_YEAR",
                "values": [2024],
                "recalibration_size": None,
                "drop_group_column": True,
            },
            "cross_validation": {
                "strategy": "group",
                "group_column": "DHB_NAME",
                "n_splits": 2,
                "shuffle": True,
                "random_state": 0,
                "drop_group_column": True,
            },
        }
        config_dict.update(overrides)

        return config_dict

    def test_valid_config(self) -> None:
        """Pass valid configuration to TopLevelConfig."""
        ValidationSubConfig.model_validate(self._get_valid_config_dict())
        ValidationSubConfig.model_validate(
            {"test_split": {"strategy": "random", "test_size": 0.1}}
        )


class TestWorkflowConfig:
    """Test class for the WorkflowConfig class"""

    def _get_valid_config_dict(self, **overrides) -> dict:
        """Creates a fresh valid config dict to override."""
        config_dict = {
            "preprocessing": {
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
            },
            "validation": {
                "test_split": {
                    "strategy": "random",
                    "test_size": 0.1,
                },
                "recalibration_split": {
                    "strategy": "group",
                    "group_column": "OP_YEAR",
                    "values": [2024],
                    "recalibration_size": None,
                    "drop_group_column": True,
                },
                "cross_validation": {
                    "strategy": "group",
                    "group_column": "DHB_NAME",
                    "n_splits": 2,
                    "shuffle": True,
                    "random_state": 0,
                    "drop_group_column": True,
                },
            },
        }
        config_dict.update(overrides)

        return config_dict

    def test_valid_config(self) -> None:
        """Pass valid configuration to WorkflowConfig."""
        WorkflowConfig.model_validate(self._get_valid_config_dict())
        WorkflowConfig.model_validate(
            self._get_valid_config_dict(
                preprocessing={},
            )
        )


class TestPredictorConfig:
    """Test class for the PredictorConfig class"""

    def _get_valid_config_dict(self, **overrides) -> dict:
        """Creates a fresh valid config dict to override."""
        config_dict = {"learning_rate": 0.1}
        config_dict.update(overrides)

        return config_dict

    def test_valid_config(self) -> None:
        """Pass valid configuration to PredictorConfig."""
        PredictorConfig.model_validate(self._get_valid_config_dict())

    @pytest.mark.parametrize("learning_rate", [-0.5, 0.0])
    def test_learning_rate_limits(self, learning_rate: float) -> None:
        """Test the learning rate value limits."""
        with pytest.raises(ValidationError, match="Input should be greater than 0"):
            PredictorConfig.model_validate(
                self._get_valid_config_dict(learning_rate=learning_rate)
            )


class TestCalibratorConfig:
    """Test class for the CalibratorConfig class"""

    def _get_valid_config_dict(self, **overrides) -> dict:
        """Creates a fresh valid config dict to override."""
        config_dict = {"out_of_bounds": "clip"}
        config_dict.update(overrides)

        return config_dict

    def test_valid_config(self) -> None:
        """Pass valid configuration to CalibratorConfig."""
        CalibratorConfig.model_validate(self._get_valid_config_dict())
        CalibratorConfig.model_validate({})


class TestModelHyperparamsSubConfigConfig:
    """Test class for the ModelHyperparamsSubConfigConfig class"""

    def _get_valid_config_dict(self, **overrides) -> dict:
        """Creates a fresh valid config dict to override."""
        config_dict = {
            "predictor": {"learning_rate": 0.1},
            "calibrator": {"out_of_bounds": "clip"},
        }
        config_dict.update(overrides)

        return config_dict

    def test_valid_config(self) -> None:
        """Pass valid configuration to ModelHyperparamsSubConfigConfig."""
        ModelHyperparamSubConfig.model_validate(self._get_valid_config_dict())
        ModelHyperparamSubConfig.model_validate(
            self._get_valid_config_dict(calibrator=None)
        )


class TestWeightingConfig:
    """Test class for the WeightingConfig class"""

    def _get_valid_config_dict(self, **overrides) -> dict:
        """Creates a fresh valid config dict to override."""
        config_dict = {"weighting_fn": "inverse_frequency_single_sample_weights"}
        config_dict.update(overrides)

        return config_dict

    def test_valid_config(self) -> None:
        """Pass valid configuration to WeightingConfig."""
        WeightingConfig.model_validate(self._get_valid_config_dict())
        WeightingConfig.model_validate({})

    def test_invalid_weighting_fn(self) -> None:
        """Test an invalid weighting function."""
        fn = "invalid_fn"
        match_expr = f"Unknown weighting function {fn} "
        f"should be one of {VALID_WEIGHTING_FN}"

        with pytest.raises(ValidationError, match=escape(match_expr)):
            WeightingConfig.model_validate(self._get_valid_config_dict(weighting_fn=fn))


class TestSamplingConfig:
    """Test class for the SamplingConfig class"""

    def _get_valid_config_dict(self, **overrides) -> dict:
        """Creates a fresh valid config dict to override."""
        config_dict = {
            "sampler_fn": "random_oversampler",
            "reduction_factor": 0.5,
            "hard_percent": 0.25,
        }
        config_dict.update(overrides)

        return config_dict

    def test_valid_config(self) -> None:
        """Pass valid configuration to SamplingConfig."""
        SamplingConfig.model_validate(self._get_valid_config_dict())
        SamplingConfig.model_validate({})

    def test_invalid_sampler_fn(self) -> None:
        """Test an invalid sampler function."""
        fn = "invalid_fn"
        match_expr = f"Unknown sampler function {fn} "
        f"should be one of {VALID_SAMPLER_FN}"

        with pytest.raises(ValidationError, match=escape(match_expr)):
            SamplingConfig.model_validate(self._get_valid_config_dict(sampler_fn=fn))

    @pytest.mark.parametrize(
        "reduction_factor, match_expr",
        [
            (-0.1, "Input should be greater than or equal to 0"),
            (1.2, "Input should be less than or equal to 1"),
            (None, "The sampler function requires a reduction factor to be specified"),
        ],
    )
    def test_reduction_factor(
        self, reduction_factor: float | None, match_expr: str
    ) -> None:
        """Test the reduction factor checks."""
        with pytest.raises(ValidationError, match=match_expr):
            SamplingConfig.model_validate(
                self._get_valid_config_dict(reduction_factor=reduction_factor)
            )

    @pytest.mark.parametrize(
        "hard_percent, match_expr",
        [
            (-0.1, "Input should be greater than 0"),
            (0.0, "Input should be greater than 0"),
            (1.2, "Input should be less than 1"),
            (1.0, "Input should be less than 1"),
        ],
    )
    def test_hard_percent_limits(
        self, hard_percent: float | None, match_expr: str
    ) -> None:
        """Test the hard percent limits."""
        with pytest.raises(ValidationError, match=match_expr):
            SamplingConfig.model_validate(
                self._get_valid_config_dict(hard_percent=hard_percent)
            )

    @pytest.mark.parametrize(
        "sampler_fn", ["mean_dist_sampler", "group_mean_dist_sampler"]
    )
    def test_hard_percent(self, sampler_fn: str) -> None:
        """Test functions that require hard percent."""
        match_expr = f"The {sampler_fn} function requires hard percent to be specified"

        with pytest.raises(ValidationError, match=match_expr):
            SamplingConfig.model_validate(
                self._get_valid_config_dict(hard_percent=None, sampler_fn=sampler_fn)
            )


class TestBalancingSubConfig:
    """Test class for the BalancingSubConfig class"""

    def _get_valid_config_dict(self, **overrides) -> dict:
        """Creates a fresh valid config dict to override."""
        config_dict = {
            "weighting": {
                "weighting_fn": "inverse_frequency_single_sample_weights",
            },
            "sampling": {
                "sampler_fn": "random_oversampler",
                "reduction_factor": 0.5,
            },
        }
        config_dict.update(overrides)

        return config_dict

    def test_valid_config(self) -> None:
        """Pass valid configuration to BalancingSubConfig."""
        BalancingSubConfig.model_validate(self._get_valid_config_dict())
        BalancingSubConfig.model_validate({})


class TestHyperparameterConfig:
    """Test class for the HyperparameterConfig class"""

    def _get_valid_config_dict(self, **overrides) -> dict:
        """Creates a fresh valid config dict to override."""
        config_dict = {
            "hyperparameters": {
                "predictor": {"learning_rate": 0.1},
                "calibrator": {"out_of_bounds": "clip"},
            },
            "balancing": {
                "weighting": {
                    "weighting_fn": "inverse_frequency_single_sample_weights",
                },
                "sampling": {
                    "sampler_fn": "random_oversampler",
                    "reduction_factor": 0.5,
                },
            },
        }
        config_dict.update(overrides)

        return config_dict

    def test_valid_config(self) -> None:
        """Pass valid configuration to HyperparameterConfig."""
        HyperparameterConfig.model_validate(self._get_valid_config_dict())
        HyperparameterConfig.model_validate(self._get_valid_config_dict(balancing={}))


class TestMedpipeConfig:
    """Test class for the MedpipeConfig class"""

    def _get_valid_config_dict(self, **overrides) -> dict:
        """Creates a fresh valid config dict to override."""
        config_dict = {
            "top_level": {
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
            },
            "data": {
                "path": "path/to/data.csv",
                "predictors": ["AGE", "SEX", "OP_SEVERITY"],
                "outcomes": ["MORTALITY_30D"],
            },
            "workflow": {
                "validation": {
                    "test_split": {
                        "strategy": "random",
                        "test_size": 0.1,
                    },
                },
            },
            "hyperparameters": {
                "hyperparameters": {
                    "predictor": {"learning_rate": 0.1},
                    "calibrator": {"out_of_bounds": "clip"},
                },
            },
        }
        config_dict.update(overrides)

        return config_dict

    def test_valid_config(self) -> None:
        """Pass valid configuration to MedpipeConfig."""
        MedpipeConfig.model_validate(self._get_valid_config_dict())


# ==============================================================================
# CONFIGURATION FUNCTION TESTS
# ==============================================================================


class TestReadSubconfigurationFile:
    """Test class for the read_subconfiguration_file function"""

    @pytest.fixture
    def example_config_dir(self) -> Path:
        """Provide the location of the example configuration files."""
        base_dir = Path(__file__).parent.parent.parent

        return base_dir / "config-examples/HGBc"

    @pytest.mark.parametrize("subtype", ["data", "workflow", "hyperparameters"])
    def test_read_subconfiguration_file_success(
        self, example_config_dir: Path, subtype: SubConfigTypes
    ) -> None:
        """Test successfull function call."""
        config_file = example_config_dir / subtype / (subtype + "_v0.toml")
        read_subconfiguration_file(config_file, subtype)

    def test_incorrect_subtype(self) -> None:
        """Test safety check on wrong subtype."""
        match_expr = f"Unexpected subtype invalid_subtype, expecting one "
        f"of {list(SUBCONFIG_REGISTRY.keys())}"
        with pytest.raises(ValueError, match=match_expr):
            read_subconfiguration_file("data.toml", "invalid_subtype")


class TestParseVersionNumber:
    """Test class for the parse_version_number function"""

    @pytest.mark.parametrize(
        "version, output",
        [
            ("v0.1.1", ["0", "1", "1"]),
            ("v10.20.10", ["10", "20", "10"]),
            ("v300.100.200", ["300", "100", "200"]),
            ("vX.Y.Z", ["X", "Y", "Z"]),
            ("1.2.3", ["1", "2", "3"]),
        ],
    )
    def test_parse_version_number_success(
        self, version: str, output: list[str]
    ) -> None:
        """Test successfull function call."""
        v_list = parse_version_number(version)
        assert len(v_list) == 3  # Check that list len is correct
        assert v_list == output

    @pytest.mark.parametrize(
        "version",
        [
            42,
            3.14,
            ("a", 1),
            {1: "a"},
            [1, 2, 3],
        ],
    )
    def test_not_a_str(self, version) -> None:
        """Test case when version is not a string."""
        match_expr = f"Version should be a string, but got {type(version)}"
        with pytest.raises(TypeError, match=match_expr):
            parse_version_number(version)

    @pytest.mark.parametrize(
        "version, error_msg",
        [
            ("", "Version is empty. "),
            ("v0", "Expecting 3 values, but got 1. "),
            ("v0.1", "Expecting 3 values, but got 2. "),
            ("..", "Element 0 in version is empty. "),
            ("v0..1", "Element 1 in version is empty. "),
            ("v0.1.", "Element 2 in version is empty. "),
        ],
    )
    def test_incorrect_str(self, version: str, error_msg: str) -> None:
        """Test case when version string is not formatted correctly."""
        match_expr = error_msg + "Check the version number is formatted "
        "as vX.Y.Z"
        with pytest.raises(ValueError, match=match_expr):
            parse_version_number(version)

    def test_longer_string(self) -> None:
        """Test case when version is longer than 3."""
        match_expr = "Expecting 3 values, but got 4. "
        "Everything after 3 is ignored."

        with pytest.warns(UserWarning, match=match_expr):
            v_list = parse_version_number("v0.1.2.3")

        assert len(v_list) == 3
        assert v_list == ["0", "1", "2"]
