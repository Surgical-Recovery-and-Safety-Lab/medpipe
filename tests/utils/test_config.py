#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration functions and classes tests suite.
"""

from pathlib import Path
from re import escape

import pytest
from pydantic import ValidationError

from medpipe.metrics.core import METRICS
from medpipe.utils.config import *

# ------------------------------------------------------------------------------
# FIXTURES for minimal required nested configurations
# ------------------------------------------------------------------------------


@pytest.fixture
def valid_workflow_dict():
    """Provides a minimal valid workflow dictionary to satisfy
    MedpipeConfig requirements."""
    return {
        "validation": {
            "test_split": {
                "strategy": "group",
                "group_column": "OP_YEAR",
                "values": [2024],
            },
            "cross_validation": {
                "strategy": "group",
                "grid_search": None,
                "group_column": "DHB_NAME",
                "n_splits": 3,
            },
        },
        "evaluation": {
            "metrics": {"metrics": ["roc_auc"]},
            "calibration": {"strategy": "uniform", "n_bootstraps": 10},
            "fairness": {"strata": ["SEX"]},
        },
    }


# ==============================================================================
# SCHEMA VALIDATION TESTS
# ==============================================================================


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

    def test_default_verbose(self) -> None:
        """Test that verbose defaults to 'compact' when omitted from the configuration."""
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
        """Test valid verbosity string literals, booleans, and allowed integer levels."""
        raw_config = self._get_valid_config_dict(verbose=valid_verbose)
        config = MetaConfig.model_validate(raw_config)

        assert config.verbose == valid_verbose

    def test_project_name_empty(self) -> None:
        """Test that project name is not empty."""
        with pytest.raises(
            ValidationError, match="Project name should not be an empty string."
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
        """Test that integers outside the allowed [0, 3] range raise a ValidationError."""
        with pytest.raises(ValidationError):
            MetaConfig(project_name="my_project", verbose=5)  # type: ignore

        with pytest.raises(ValidationError):
            MetaConfig(project_name="my_project", verbose=-1)  # type: ignore


class TestDataConfig:
    """Test class for the DataConfig class"""

    def _get_valid_config_dict(self, tmp_path: Path, **overrides) -> dict:
        """Creates a fresh valid config dict to override."""
        config_dict = {
            "path": str(tmp_path / "path/to/data.csv"),
            "predictors": ["AGE", "SEX", "OP_SEVERITY"],
            "outcomes": ["MORTALITY_30D"],
        }

        config_dict.update(overrides)

        return config_dict

    @pytest.mark.parametrize(
        "path",
        ["path/to/data.csv", "path/to/data.parquet"],
    )
    def test_valid_config(self, tmp_path: Path, path: str) -> None:
        """Pass valid configuration to DataConfig."""
        raw_config = self._get_valid_config_dict(
            tmp_path, **{"path": str(tmp_path / path)}
        )
        config = DataConfig.model_validate(raw_config)

        assert config.model_dump() == raw_config

    @pytest.mark.parametrize(
        "path, match_expr",
        [
            ("path/", "path should be a file, but got no suffix"),
            ("path.db", "path should be a .csv or .parquet file, but got suffix .db"),
        ],
    )
    def test_validate_path(self, tmp_path: Path, path: str, match_expr: str) -> None:
        """Test data path is not a file."""
        with pytest.raises(ValidationError, match=match_expr):
            DataConfig.model_validate(
                self._get_valid_config_dict(tmp_path=tmp_path, path=path)
            )

    @pytest.mark.parametrize(
        "predictors, outcomes, overlap",
        [
            (["AGE", "SEX", "ANY_COMP"], ["ANY_COMP"], ["ANY_COMP"]),
            (["AGE", "SEX", "ANY_COMP"], ["AGE", "SEX"], ["AGE", "SEX"]),
        ],
    )
    def test_data_leakage(
        self,
        tmp_path: Path,
        predictors: list[str],
        outcomes: list[str],
        overlap: list[str],
    ) -> None:
        """Test data leakage safety check."""
        match_expr = "Overlap between predictors and outcomes which will "
        f"break model validity: {overlap}"

        with pytest.raises(ValidationError, match=escape(match_expr)):
            DataConfig.model_validate(
                self._get_valid_config_dict(
                    tmp_path, predictors=predictors, outcomes=outcomes
                )
            )

    def test_data_config_no_extension(self):
        """Test that a path with no extension fails validation."""
        with pytest.raises(
            ValidationError, match="path should be a file, but got no suffix"
        ):
            DataConfig(path="dataset", predictors=["age"], outcomes=["mortality"])


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
            assert value == None

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


class TestMetricsConfig:
    """Test class for the MetricsConfig class"""

    def _get_valid_config_dict(self, **overrides) -> dict:
        """Creates a fresh valid config dict to override."""
        config_dict = {
            "metrics": ["roc_auc", "log_loss", "ici"],
            "n_bootstraps": 1000,
            "ci_level": 0.95,
        }
        config_dict.update(overrides)

        return config_dict

    def test_valid_config(self) -> None:
        """Pass valid configuration to MetricsConfig."""
        raw_config = self._get_valid_config_dict()
        config = MetricsConfig.model_validate(raw_config)

        assert config.model_dump() == raw_config

    def test_invalid_metric(self) -> None:
        """Test case when invalid metric is provided."""
        match_expr = (
            "invalid was not found in available metric "
            f"list. Available metrics are {METRICS}"
        )

        with pytest.raises(ValidationError, match=escape(match_expr)):
            MetricsConfig.model_validate(
                self._get_valid_config_dict(metrics=["invalid"])
            )


class TestFairnessConfig:
    """Test class for the FairnessConfig class"""

    def _get_valid_config_dict(self, **overrides) -> dict:
        """Creates a fresh valid config dict to override."""
        config_dict = {
            "strata": ["AGE", "SEX"],
            "groups": {"AGE": [[18, 50], [51, 120]]},
        }
        config_dict.update(overrides)

        return config_dict

    def test_valid_config(self) -> None:
        """Pass valid configuration to TestFairnessConfig."""
        raw_config = self._get_valid_config_dict()
        config = FairnessConfig.model_validate(raw_config)

        assert config.model_dump() == raw_config

    def test_group_not_in_strata(self) -> None:
        """Test case when groups are not in strata."""
        with pytest.raises(
            ValidationError, match="invalid should be in the strata list"
        ):
            FairnessConfig.model_validate(
                self._get_valid_config_dict(groups={"invalid": []})
            )

    def test_group_is_empty(self) -> None:
        """Test case when group has an empty list as value."""
        with pytest.raises(ValidationError, match="AGE should not have an empty list"):
            FairnessConfig.model_validate(
                self._get_valid_config_dict(groups={"AGE": []})
            )


class TestEvaluationSubConfig:
    """Test class for the EvaluationSubConfig class"""

    def _get_valid_config_dict(self, **overrides) -> dict:
        """Creates a fresh valid config dict to override."""
        config_dict = {
            "metrics": {
                "metrics": ["roc_auc", "ici"],
                "n_bootstraps": 1000,
                "ci_level": 0.95,
            },
            "fairness": {
                "strata": ["AGE", "SEX"],
                "groups": {"AGE": [[18, 50], [51, 120]]},
            },
        }
        config_dict.update(overrides)

        return config_dict

    def test_valid_config(self) -> None:
        """Pass valid configuration to TestEvaluationSubConfig."""
        raw_config = self._get_valid_config_dict()
        config = EvaluationSubConfig.model_validate(raw_config)

        assert config.model_dump() == raw_config


class TestWorkflowConfig:
    """Test class for the WorkflowConfig class"""

    def _get_valid_config_dict(self, **overrides) -> dict:
        """Creates a fresh valid config dict to override."""
        config_dict = {
            "random_state": None,
            "preprocessing": {
                "preprocess": True,
                "operations": [
                    {
                        "name": "OrdinalEncoder",
                        "columns": ["SEX", "ETHNICITY"],
                    },
                    {
                        "name": "StandarScaler",
                        "columns": ["SEX", "ETHNICITY"],
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
                    "metrics": ["roc_auc", "ici"],
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
        """Pass valid configuration to WorkflowConfig."""
        raw_config = self._get_valid_config_dict()
        config = WorkflowConfig.model_validate(raw_config)

        assert config.model_dump() == raw_config


class TestRecalibrationConfig:
    """Test class for the RecalibrationConfig class"""

    def _get_valid_config_dict(self, **overrides) -> dict:
        """Creates a fresh valid config dict to override."""
        config_dict = {
            "recalibrate": True,
            "method": "isotonic",
            "hyperparameters": {},
        }
        config_dict.update(overrides)

        return config_dict

    def test_valid_config(self) -> None:
        """Pass valid configuration to RecalibrationConfig."""
        raw_config = self._get_valid_config_dict()
        config = RecalibrationConfig.model_validate(raw_config)

        assert config.model_dump() == raw_config

    @pytest.mark.parametrize("recalibrate_val", [True, False])
    def test_recalibration_flag_boolean_values(self, recalibrate_val: bool) -> None:
        """Pass True and False recalibrate flags to RecalibrationConfig."""
        config = RecalibrationConfig(recalibrate=recalibrate_val, method="isotonic")
        assert config.recalibrate is recalibrate_val


class TestModelSetupConfig:
    """Test class for the ModelSetup class"""

    def test_model_setup_valid(self):
        """Test valid model setup with defaults."""
        model = ModelSetup(algorithm="LogisticRegression")
        assert model.algorithm == "LogisticRegression"
        assert model.hyperparameters == {}
        assert model.recalibration is None

    def test_model_setup_with_recalibration(self):
        """Test valid model setup including recalibration parameters."""
        model = ModelSetup(
            algorithm="XGBClassifier",
            hyperparameters={"learning_rate": 0.1},
            recalibration={  # type: ignore
                "recalibrate": True,
                "method": "isotonic",
                "hyperparameters": {"out_of_bounds": "clip"},
            },
        )
        assert model.recalibration
        assert model.recalibration.recalibrate == True
        assert model.recalibration.method == "isotonic"
        assert model.recalibration.hyperparameters["out_of_bounds"] == "clip"

    def test_model_setup_extra_forbidden(self):
        """Test that extra fields at the model setup level are strictly forbidden."""
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            ModelSetup(algorithm="RF", unexpected_flag=True)  # type: ignore


# ==============================================================================
# DISPLAY CONFIGURATION TESTS
# ==============================================================================


class TestDisplayDefaultsConfig:
    """Test class for the DisplayDefaultsConfig schema."""

    def test_default_values(self) -> None:
        """Test default values when no parameters are provided."""
        config = DisplayDefaultsConfig()

        assert config.n_bootstraps == 1000
        assert config.save is True
        assert config.show is False
        assert config.n_bins == 10
        assert config.strategy == "uniform"

    def test_custom_values(self) -> None:
        """Test instantiating with custom values."""
        config = DisplayDefaultsConfig(
            n_bootstraps=200,
            save=False,
            show=True,
            n_bins=20,
            strategy="spline",
        )

        assert config.n_bootstraps == 200
        assert config.save is False
        assert config.show is True
        assert config.n_bins == 20
        assert config.strategy == "spline"

    def test_extra_fields_allowed(self) -> None:
        """Test that extra fields are allowed for flexibility in plot parameters."""
        config = DisplayDefaultsConfig.model_validate(
            {"n_bootstraps": 500, "dpi": 300, "custom_color": "blue"}
        )

        assert config.n_bootstraps == 500
        assert getattr(config, "dpi", None) == 300
        assert getattr(config, "custom_color", None) == "blue"


class TestDisplayConfig:
    """Test class for the DisplayConfig schema."""

    def test_valid_config(self) -> None:
        """Pass a fully defined valid configuration to DisplayConfig."""
        raw_config = {
            "defaults": {
                "n_bootstraps": 1000,
                "save": True,
                "show": False,
                "n_bins": 10,
                "strategy": "uniform",
            },
            "overrides": {
                "calibration": {
                    "n_bootstraps": 200,
                    "strategy": "spline",
                },
                "distribution": {"n_bins": 25},
            },
            "outcome_overrides": {
                "MORTALITY_30D": {
                    "calibration": {
                        "n_bootstraps": 50,
                        "strategy": "uniform",
                    }
                }
            },
            "theme": {"primary_color": "#2D90D8", "show_grid": False},
        }

        config = DisplayConfig.model_validate(raw_config)

        assert config.defaults.n_bootstraps == 1000
        assert config.overrides["calibration"]["n_bootstraps"] == 200
        assert (
            config.outcome_overrides["MORTALITY_30D"]["calibration"]["strategy"]
            == "uniform"
        )
        assert config.theme["primary_color"] == "#2D90D8"

    def test_flat_and_legacy_keys_transformed_to_defaults(self) -> None:
        """Test that legacy flat keys get mapped into defaults during pre-validation."""
        raw_config = {
            "calibration_strategy": "spline",
            "n_bootstraps": 500,
            "save": False,
        }

        config = DisplayConfig.model_validate(raw_config)

        assert config.defaults.strategy == "spline"
        assert config.defaults.n_bootstraps == 500
        assert config.defaults.save is False

    @pytest.mark.parametrize(
        "invalid_plot_key",
        ["invalid_plot_name", "unknown_curve", "scatter_plot"],
    )
    def test_invalid_override_plot_key(self, invalid_plot_key: str) -> None:
        """Test that unknown plot override types raise a ValidationError."""
        raw_config = {
            "overrides": {
                invalid_plot_key: {"n_bootstraps": 100},
            }
        }

        with pytest.raises(
            ValidationError, match=f"Unknown plot override type '{invalid_plot_key}'"
        ):
            DisplayConfig.model_validate(raw_config)

    def test_invalid_outcome_override_plot_key(self) -> None:
        """Test that unknown plot types inside outcome_overrides raise a ValidationError."""
        raw_config = {
            "outcome_overrides": {
                "MORTALITY_30D": {
                    "unsupported_diagram": {"n_bins": 5},
                }
            }
        }

        with pytest.raises(
            ValidationError, match="Unknown plot override type 'unsupported_diagram'"
        ):
            DisplayConfig.model_validate(raw_config)


class TestMedpipeConfig:
    """Test class for the MedpipeConfig class."""

    def _get_valid_config_dict(self, tmp_path: Path, **overrides) -> dict:
        """Creates a fresh valid config dict to override."""
        config_dict = {
            "meta": {
                "project_name": "medpipe-test",
                "run_mode": "audit",
                "verbose": "compact",
            },
            "data": {
                "path": str(tmp_path / "path/to/data.csv"),
                "predictors": ["AGE", "SEX", "OP_SEVERITY"],
                "outcomes": ["MORTALITY_30D", "ANY_COMP"],
            },
            "default_model": {
                "algorithm": "HistGradientBoostingClassifier",
                "hyperparameters": {"learning_rate": 0.1},
                "recalibration": {
                    "recalibrate": True,
                    "method": "isotonic",
                    "hyperparameters": {},
                },
            },
            "workflow": {
                "random_state": 42,
                "preprocessing": None,
                "validation": {
                    "cross_validation": {
                        "strategy": "random",
                        "grid_search": None,
                        "n_splits": 5,
                        "shuffle": True,
                        "group_column": None,
                    },
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
                },
                "evaluation": {
                    "metrics": {
                        "metrics": ["roc_auc", "ici"],
                        "n_bootstraps": 200,
                        "ci_level": 0.95,
                    },
                    "fairness": {
                        "strata": ["AGE", "SEX"],
                        "groups": {"AGE": [[18, 50], [51, 120]]},
                    },
                },
            },
            "display": {
                "defaults": {
                    "n_bootstraps": 1000,
                    "save": True,
                    "show": False,
                    "n_bins": 10,
                    "strategy": "uniform",
                },
                "overrides": {
                    "calibration": {
                        "n_bootstraps": 200,
                        "strategy": "spline",
                    }
                },
            },
        }
        config_dict.update(overrides)
        return config_dict

    def test_valid_config(self, tmp_path: Path) -> None:
        """Pass valid configuration to MedpipeConfig."""
        raw_config = self._get_valid_config_dict(tmp_path)
        config = MedpipeConfig.model_validate(raw_config)

        dumped = config.model_dump(exclude={"resolved_models"})
        revalidated = MedpipeConfig.model_validate(dumped)

        assert revalidated == config

    def test_recalibration_split(self, tmp_path: Path) -> None:
        """Test case when recalibration method and split mismatch."""
        match_expr = (
            "Recalibration validation split must be "
            "specified when a recalibration method is used"
        )

        config = self._get_valid_config_dict(tmp_path)
        config["workflow"]["validation"]["recalibration_split"] = None

        with pytest.raises(ValueError, match=match_expr):
            MedpipeConfig.model_validate(config)

    @pytest.mark.parametrize("run_mode", ["audit", "cv"])
    def test_cross_validation_run_mode(
        self, tmp_path: Path, run_mode: Literal["audit", "cv"]
    ) -> None:
        """Test case when run_mode and cross_validation mismatch."""
        match_expr = (
            "Cross-validation parameters must be specified "
            "when run_mode is not 'fast'"
        )
        config = self._get_valid_config_dict(tmp_path)
        config["meta"]["run_mode"] = run_mode
        config["workflow"]["validation"]["cross_validation"] = None

        with pytest.raises(ValueError, match=match_expr):
            MedpipeConfig.model_validate(config)

    @pytest.mark.parametrize(
        "run_mode",
        ["audit", "eval"],
    )
    def test_evaluation(self, tmp_path: Path, run_mode: str) -> None:
        """Test case when run_mode and evaluation mismatch."""
        match_expr = (
            f"Evaluation fairness parameters must be specified "
            "when run_mode is 'audit' or 'eval'"
        )
        config = self._get_valid_config_dict(tmp_path)
        config["meta"]["run_mode"] = run_mode
        config["workflow"]["evaluation"]["fairness"] = None

        with pytest.raises(ValueError, match=match_expr):
            MedpipeConfig.model_validate(config)

    @pytest.mark.parametrize("run_mode", ["audit", "eval"])
    def test_display_config_value_error(self, tmp_path: Path, run_mode: str) -> None:
        """Test case when run_mode is audit or eval and display config is missing."""
        match_expr = (
            "Display parameters must be specified " "when run_mode is 'audit' or 'eval'"
        )
        config = self._get_valid_config_dict(tmp_path)
        config["meta"]["run_mode"] = run_mode
        config["display"] = None

        with pytest.raises(ValueError, match=match_expr):
            MedpipeConfig.model_validate(config)

    def test_cascade_no_overrides(self, tmp_path: Path) -> None:
        """Test that default models map to all outcomes when there are no overrides."""
        raw_config = self._get_valid_config_dict(tmp_path)
        config = MedpipeConfig.model_validate(raw_config)

        assert "MORTALITY_30D" in config.resolved_models
        assert "ANY_COMP" in config.resolved_models

        assert (
            config.resolved_models["MORTALITY_30D"].algorithm
            == "HistGradientBoostingClassifier"
        )
        assert (
            config.resolved_models["ANY_COMP"].algorithm
            == "HistGradientBoostingClassifier"
        )
        assert (
            config.resolved_models["MORTALITY_30D"].hyperparameters["learning_rate"]
            == 0.1
        )

    def test_cascade_algorithm_and_hyperparameters_merge(self, tmp_path: Path) -> None:
        """Test that hyperparameters correctly deep merge when overridden."""
        raw_config = self._get_valid_config_dict(tmp_path)

        raw_config["outcome_overrides"] = {
            "ANY_COMP": {
                "algorithm": "RandomForestClassifier",
                "hyperparameters": {"n_estimators": 200, "max_depth": 5},
            }
        }

        config = MedpipeConfig.model_validate(raw_config)

        model_mortality = config.resolved_models["MORTALITY_30D"]
        assert model_mortality.algorithm == "HistGradientBoostingClassifier"
        assert model_mortality.hyperparameters == {"learning_rate": 0.1}

        model_comp = config.resolved_models["ANY_COMP"]
        assert model_comp.algorithm == "RandomForestClassifier"
        assert model_comp.hyperparameters == {
            "n_estimators": 200,
            "max_depth": 5,
        }

    def test_cascade_recalibration_deep_merge(self, tmp_path: Path) -> None:
        """Test that recalibration methods and kwargs merge correctly."""
        raw_config = self._get_valid_config_dict(tmp_path)

        raw_config["default_model"]["recalibration"]["hyperparameters"]["y_min"] = 0

        raw_config["outcome_overrides"] = {
            "MORTALITY_30D": {
                "algorithm": "HistGradientBoostingClassifier",
                "recalibration": {
                    "recalibrate": True,
                    "method": "isotonic",
                    "hyperparameters": {"cv": 5},
                },
            }
        }

        config = MedpipeConfig.model_validate(raw_config)

        res_recal = config.resolved_models["MORTALITY_30D"].recalibration
        assert res_recal is not None
        assert res_recal.method == "isotonic"
        assert res_recal.hyperparameters == {"y_min": 0, "cv": 5}

    def test_cascade_recalibration_no_deep_merge_with_different_methods(
        self, tmp_path: Path
    ) -> None:
        """Test that recalibration methods and kwargs do not merge when method is
        different."""
        raw_config = self._get_valid_config_dict(tmp_path)

        raw_config["default_model"]["recalibration"]["hyperparameters"]["y_min"] = 0

        raw_config["outcome_overrides"] = {
            "MORTALITY_30D": {
                "algorithm": "HistGradientBoostingClassifier",
                "recalibration": {
                    "recalibrate": True,
                    "method": "sigmoid",
                    "hyperparameters": {"cv": 5},
                },
            }
        }

        config = MedpipeConfig.model_validate(raw_config)

        res_recal = config.resolved_models["MORTALITY_30D"].recalibration
        assert res_recal is not None
        assert res_recal.method == "sigmoid"
        assert res_recal.hyperparameters == {"cv": 5}

    def test_cascade_same_algorithm_deep_merges_model_hyperparameters(
        self, tmp_path: Path
    ) -> None:
        """Test that model hyperparameters deep-merge when the algorithm
        is unchanged or omitted."""
        raw_config = self._get_valid_config_dict(tmp_path)
        raw_config["default_model"] = {
            "algorithm": "HistGradientBoostingClassifier",
            "hyperparameters": {"learning_rate": 0.1, "max_iter": 100},
        }
        raw_config["outcome_overrides"] = {
            "ANY_COMP": {
                # Algorithm is same, only overriding specific model hyperparameters
                "algorithm": "HistGradientBoostingClassifier",
                "hyperparameters": {"learning_rate": 0.05, "max_depth": 5},
            }
        }

        config = MedpipeConfig.model_validate(raw_config)

        comp_model = config.resolved_models["ANY_COMP"]
        assert comp_model.algorithm == "HistGradientBoostingClassifier"
        # learning_rate is updated, max_iter is retained from base, max_depth is added
        assert comp_model.hyperparameters == {
            "learning_rate": 0.05,
            "max_iter": 100,
            "max_depth": 5,
        }

    def test_cascade_algorithm_change_replaces_hyperparameters(
        self, tmp_path: Path
    ) -> None:
        """Test changing algorithm replaces hyperparameters instead of
        merging incompatible ones."""
        raw_config = self._get_valid_config_dict(tmp_path)
        raw_config["default_model"] = {
            "algorithm": "HistGradientBoostingClassifier",
            "hyperparameters": {"learning_rate": 0.1, "max_iter": 100},
        }
        raw_config["outcome_overrides"] = {
            "ANY_COMP": {
                "algorithm": "RandomForestClassifier",
                "hyperparameters": {"n_estimators": 50},
            }
        }

        config = MedpipeConfig.model_validate(raw_config)

        # learning_rate and max_iter should NOT leak into RandomForestClassifier
        comp_model = config.resolved_models["ANY_COMP"]
        assert comp_model.algorithm == "RandomForestClassifier"
        assert comp_model.hyperparameters == {"n_estimators": 50}

    def test_cascade_explicitly_set_recalibration_none(self, tmp_path: Path) -> None:
        """Test setting recalibration to None in outcome_overrides disables
        recalibration."""
        raw_config = self._get_valid_config_dict(tmp_path)
        raw_config["default_model"]["recalibration"] = {
            "recalibrate": True,
            "method": "isotonic",
            "hyperparameters": {"out_of_bounds": "clip"},
        }
        raw_config["outcome_overrides"] = {
            "ANY_COMP": {
                "algorithm": "HistGradientBoostingClassifier",
                "recalibration": None,
            }
        }

        config = MedpipeConfig.model_validate(raw_config)

        assert config.resolved_models["MORTALITY_30D"].recalibration is not None
        assert config.resolved_models["ANY_COMP"].recalibration is None

    def test_cascade_recalibration_different_method_replaces_entirely(
        self, tmp_path: Path
    ) -> None:
        """Test switching recalibration method replaces base hyperparameters completely."""
        raw_config = self._get_valid_config_dict(tmp_path)
        raw_config["default_model"]["recalibration"] = {
            "recalibrate": True,
            "method": "isotonic",
            "hyperparameters": {"out_of_bounds": "clip"},
        }
        raw_config["outcome_overrides"] = {
            "ANY_COMP": {
                "algorithm": "HistGradientBoostingClassifier",
                "recalibration": {
                    "recalibrate": True,
                    "method": "sigmoid",
                    "hyperparameters": {"cv": 3},
                },
            }
        }

        config = MedpipeConfig.model_validate(raw_config)

        res_recal = config.resolved_models["ANY_COMP"].recalibration
        assert res_recal is not None
        assert res_recal.method == "sigmoid"
        # out_of_bounds should NOT leak from isotonic into sigmoid
        assert res_recal.hyperparameters == {"cv": 3}

    def test_cascade_recalibration_same_method_deep_merges(
        self, tmp_path: Path
    ) -> None:
        """Test that same recalibration method updates fields and
        deep-merges hyperparameters."""
        raw_config = self._get_valid_config_dict(tmp_path)
        raw_config["default_model"]["recalibration"] = {
            "recalibrate": True,
            "method": "isotonic",
            "hyperparameters": {"out_of_bounds": "clip"},
        }
        raw_config["outcome_overrides"] = {
            "ANY_COMP": {
                "algorithm": "HistGradientBoostingClassifier",
                "recalibration": {
                    "recalibrate": False,
                    "method": "isotonic",
                    "hyperparameters": {"n_jobs": 2},
                },
            }
        }

        config = MedpipeConfig.model_validate(raw_config)

        res_recal = config.resolved_models["ANY_COMP"].recalibration
        assert res_recal is not None
        assert res_recal.recalibrate is False
        assert res_recal.method == "isotonic"
        assert res_recal.hyperparameters == {"out_of_bounds": "clip", "n_jobs": 2}

    def test_cascade_adds_recalibration_when_base_has_none(
        self, tmp_path: Path
    ) -> None:
        """Test that an override can introduce recalibration to a
        base model that lacks it."""
        raw_config = self._get_valid_config_dict(tmp_path)

        raw_config["default_model"]["recalibration"] = None

        raw_config["outcome_overrides"] = {
            "ANY_COMP": {
                "algorithm": "HistGradientBoostingClassifier",
                "recalibration": {"recalibrate": True, "method": "sigmoid"},
            }
        }

        config = MedpipeConfig.model_validate(raw_config)

        assert config.default_model.recalibration is None
        assert config.resolved_models["MORTALITY_30D"].recalibration is None

        assert config.resolved_models["ANY_COMP"].recalibration is not None
        assert config.resolved_models["ANY_COMP"].recalibration.method == "sigmoid"

    def test_validate_search_cv_succeeds_with_list_hyperparameters(
        self, tmp_path: Path
    ) -> None:
        """Test that grid_search=true passes validation when at least
        one hyperparameter is a list."""
        raw_config = self._get_valid_config_dict(tmp_path)
        raw_config["workflow"]["validation"]["cross_validation"]["grid_search"] = True
        raw_config["default_model"]["hyperparameters"] = {
            "learning_rate": [0.01, 0.1],
            "max_depth": 3,
        }

        config = MedpipeConfig.model_validate(raw_config)

        assert config.workflow.validation.cross_validation
        assert config.workflow.validation.cross_validation.strategy == "random"
        assert config.workflow.validation.cross_validation.grid_search
        assert config.resolved_models["MORTALITY_30D"].hyperparameters[
            "learning_rate"
        ] == [0.01, 0.1]

    def test_outcome_override_not_in_outcomes_list(self, tmp_path: Path) -> None:
        """Test that specifying a model override for an outcome not
        listed in data.outcomes raises ValidationError."""
        raw_config = self._get_valid_config_dict(tmp_path)

        raw_config["outcome_overrides"] = {
            "UNLISTED_OUTCOME": {
                "algorithm": "RandomForestClassifier",
            }
        }

        match_expr = (
            "Outcome override 'UNLISTED_OUTCOME' is not present in data.outcomes"
        )

        with pytest.raises(ValidationError, match=escape(match_expr)):
            MedpipeConfig.model_validate(raw_config)

    def test_display_outcome_override_not_in_outcomes_list(
        self, tmp_path: Path
    ) -> None:
        """Test that specifying a display override for an outcome not
        listed in data.outcomes raises ValidationError."""
        raw_config = self._get_valid_config_dict(tmp_path)

        raw_config["display"]["outcome_overrides"] = {
            "UNLISTED_OUTCOME": {
                "calibration": {"n_bootstraps": 10},
            }
        }

        match_expr = "Display outcome override 'UNLISTED_OUTCOME' is not present in data.outcomes"

        with pytest.raises(ValidationError, match=escape(match_expr)):
            MedpipeConfig.model_validate(raw_config)

    def test_cascade_disable_recalibration_via_flag(self, tmp_path: Path) -> None:
        """Test overriding recalibrate=False turns off recalibration for
        a specific outcome."""
        raw_config = self._get_valid_config_dict(tmp_path)
        raw_config["default_model"]["recalibration"] = {
            "recalibrate": True,
            "method": "isotonic",
            "hyperparameters": {},
        }
        raw_config["outcome_overrides"] = {
            "ANY_COMP": {
                "algorithm": "HistGradientBoostingClassifier",
                "recalibration": {"recalibrate": False, "method": "isotonic"},
            }
        }

        config = MedpipeConfig.model_validate(raw_config)

        # MORTALITY_30D inherits default True, ANY_COMP overrides to False
        assert config.resolved_models["MORTALITY_30D"].recalibration
        assert config.resolved_models["ANY_COMP"].recalibration
        assert config.resolved_models["MORTALITY_30D"].recalibration.recalibrate is True
        assert config.resolved_models["ANY_COMP"].recalibration.recalibrate is False
