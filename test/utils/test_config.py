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
        raw_config = self._get_valid_config_dict()
        config = MetaConfig.model_validate(raw_config)

        assert config.model_dump() == raw_config

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
        raw_config = self._get_valid_config_dict(tmp_path)
        config = PathsConfig.model_validate(raw_config)

        assert config.model_dump() == raw_config

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
        raw_config = self._get_valid_config_dict()
        config = ModelConfig.model_validate(raw_config)

        assert config.model_dump() == raw_config


class TestRecalibrationConfig:
    """Test class for the RecalibrationConfig class"""

    def _get_valid_config_dict(self, **overrides) -> dict:
        """Creates a fresh valid config dict to override."""
        config_dict = {
            "method": "isotonic",
        }
        config_dict.update(overrides)

        return config_dict

    def test_valid_config(self) -> None:
        """Pass valid configuration to RecalibrationConfig."""
        raw_config = self._get_valid_config_dict()
        config = RecalibrationConfig.model_validate(raw_config)

        assert config.model_dump() == raw_config


class TestTopLevelConfig:
    """Test class for the TopLevelConfig class"""

    def _get_valid_config_dict(self, tmp_path: Path, **overrides) -> dict:
        """Creates a fresh valid config dict to override."""
        config_dict = {
            "meta": {
                "version": "v0.0.1",
                "project_name": "mepipe-test",
                "run_mode": "audit",
            },
            "paths": {
                "config_dir": str(tmp_path / "path/to/config"),
                "model_dir": str(tmp_path / "path/to/models"),
                "figure_dir": str(tmp_path / "path/to/figures"),
            },
            "model": {"algorithm": "HistGradientBoostingClassifier"},
            "recalibration": {"method": "isotonic"},
        }
        config_dict.update(overrides)

        return config_dict

    def test_valid_config(self, tmp_path: Path) -> None:
        """Pass valid configuration to TopLevelConfig."""
        raw_config = self._get_valid_config_dict(tmp_path)
        config = TopLevelConfig.model_validate(raw_config)

        assert config.model_dump() == raw_config


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
            "group_column": None,
            "n_splits": 2,
            "shuffle": True,
            "random_state": 0,
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
                "group_column": "DHB_NAME",
                "n_splits": 2,
                "shuffle": True,
                "random_state": 0,
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
                "group_column": None,
                "n_splits": 2,
                "shuffle": True,
                "random_state": 0,
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
        config_dict = {"metrics": ["roc_auc", "log_loss", "ici"]}
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


class TestCalibrationConfig:
    """Test class for the CalibrationConfig class"""

    def _get_valid_config_dict(self, **overrides) -> dict:
        """Creates a fresh valid config dict to override."""
        config_dict = {"strategy": "uniform", "n_bootstraps": 200}
        config_dict.update(overrides)

        return config_dict

    @pytest.mark.parametrize("strategy", ["uniform", "quantile", "spline"])
    def test_valid_config(self, strategy: str) -> None:
        """Pass valid configuration to CalibrationConfig."""
        raw_config = self._get_valid_config_dict(strategy=strategy)
        config = CalibrationConfig.model_validate(raw_config)

        assert config.model_dump() == raw_config


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
            "metrics": {"metrics": ["roc_auc", "ici"]},
            "calibration": {"strategy": "uniform", "n_bootstraps": 200},
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
                    "group_column": "DHB_NAME",
                    "n_splits": 2,
                    "shuffle": True,
                    "random_state": 0,
                },
            },
            "evaluation": {
                "metrics": {"metrics": ["roc_auc", "ici"]},
                "calibration": {"strategy": "uniform", "n_bootstraps": 200},
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


class TestPredictorConfig:
    """Test class for the PredictorConfig class"""

    def _get_valid_config_dict(self, **overrides) -> dict:
        """Creates a fresh valid config dict to override."""
        config_dict = {"learning_rate": 0.1}
        config_dict.update(overrides)

        return config_dict

    def test_valid_config(self) -> None:
        """Pass valid configuration to PredictorConfig."""
        raw_config = self._get_valid_config_dict()
        config = PredictorConfig.model_validate(raw_config)

        assert config.model_dump() == raw_config

    @pytest.mark.parametrize("learning_rate", [-0.5, 0.0])
    def test_learning_rate_limits(self, learning_rate: float) -> None:
        """Test the learning rate value limits."""
        with pytest.raises(ValidationError, match="Input should be greater than 0"):
            PredictorConfig.model_validate(
                self._get_valid_config_dict(learning_rate=learning_rate)
            )


class TestRecalibratorConfig:
    """Test class for the RecalibratorConfig class"""

    def _get_valid_config_dict(self, **overrides) -> dict:
        """Creates a fresh valid config dict to override."""
        config_dict = {"out_of_bounds": "clip"}
        config_dict.update(overrides)

        return config_dict

    def test_valid_config(self) -> None:
        """Pass valid configuration to RecalibratorConfig."""
        raw_config = self._get_valid_config_dict()
        config = RecalibratorConfig.model_validate(raw_config)

        assert config.model_dump() == raw_config

    def test_valid_config_None(self) -> None:
        """Test case when configuration is empty dictionary."""
        config = RecalibratorConfig.model_validate({})

        for value in config.model_dump().values():
            assert value == None


class TestModelHyperparamsSubConfigConfig:
    """Test class for the ModelHyperparamsSubConfigConfig class"""

    def _get_valid_config_dict(self, **overrides) -> dict:
        """Creates a fresh valid config dict to override."""
        config_dict = {
            "predictor": {"learning_rate": 0.1},
            "recalibrator": {"out_of_bounds": "clip"},
        }
        config_dict.update(overrides)

        return config_dict

    def test_valid_config(self) -> None:
        """Pass valid configuration to ModelHyperparamsSubConfigConfig."""
        raw_config = self._get_valid_config_dict()
        config = ModelHyperparamSubConfig.model_validate(raw_config)

        assert config.model_dump() == raw_config


class TestHyperparameterConfig:
    """Test class for the HyperparameterConfig class"""

    def _get_valid_config_dict(self, **overrides) -> dict:
        """Creates a fresh valid config dict to override."""
        config_dict = {
            "hyperparameters": {
                "predictor": {"learning_rate": 0.1},
                "recalibrator": {"out_of_bounds": "clip"},
            },
        }
        config_dict.update(overrides)

        return config_dict

    def test_valid_config(self) -> None:
        """Pass valid configuration to HyperparameterConfig."""
        raw_config = self._get_valid_config_dict()
        config = HyperparameterConfig.model_validate(raw_config)

        assert config.model_dump() == raw_config


class TestMedpipeConfig:
    """Test class for the MedpipeConfig class"""

    def _get_valid_config_dict(self, tmp_path: Path, **overrides) -> dict:
        """Creates a fresh valid config dict to override."""
        config_dict = {
            "top_level": {
                "meta": {
                    "version": "v0.0.1",
                    "project_name": "mepipe-test",
                    "run_mode": "audit",
                },
                "paths": {
                    "config_dir": str(tmp_path / "path/to/config"),
                    "model_dir": str(tmp_path / "path/to/models"),
                    "figure_dir": str(tmp_path / "path/to/figures"),
                },
                "model": {"algorithm": "HistGradientBoostingClassifier"},
                "recalibration": {"method": "isotonic"},
            },
            "data": {
                "path": str(tmp_path / "path/to/data.csv"),
                "predictors": ["AGE", "SEX", "OP_SEVERITY"],
                "outcomes": ["MORTALITY_30D"],
            },
            "workflow": {
                "preprocessing": None,
                "validation": {
                    "cross_validation": {
                        "strategy": "random",
                        "n_splits": 5,
                        "shuffle": True,
                        "random_state": 42,
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
                    "metrics": {"metrics": ["roc_auc", "ici"]},
                    "calibration": {
                        "strategy": "uniform",
                        "n_bootstraps": 200,
                    },
                    "fairness": {
                        "strata": ["AGE", "SEX"],
                        "groups": {"AGE": [[18, 50], [51, 120]]},
                    },
                },
            },
            "hyperparameters": {
                "hyperparameters": {
                    "predictor": {"learning_rate": 0.1},
                    "recalibrator": {"out_of_bounds": "clip"},
                },
            },
        }
        config_dict.update(overrides)

        return config_dict

    def test_valid_config(self, tmp_path: Path) -> None:
        """Pass valid configuration to MedpipeConfig."""
        raw_config = self._get_valid_config_dict(tmp_path)
        config = MedpipeConfig.model_validate(raw_config)

        assert config.model_dump() == raw_config

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
        config["top_level"]["meta"]["run_mode"] = run_mode
        config["workflow"]["validation"]["cross_validation"] = None

        with pytest.raises(ValueError, match=match_expr):
            MedpipeConfig.model_validate(config)

    @pytest.mark.parametrize(
        "run_mode,params",
        [
            ("audit", "calibration"),
            ("audit", "fairness"),
            ("eval", "calibration"),
            ("eval", "fairness"),
        ],
    )
    def test_evaluation(self, tmp_path: Path, run_mode: str, params: str) -> None:
        """Test case when run_mode and evaluation mismatch."""
        match_expr = (
            f"Evaluation {params} parameters must be specified "
            "when run_mode is 'audit' or 'eval'"
        )
        config = self._get_valid_config_dict(tmp_path)
        config["top_level"]["meta"]["run_mode"] = run_mode
        config["workflow"]["evaluation"][params] = None

        with pytest.raises(ValueError, match=match_expr):
            MedpipeConfig.model_validate(config)


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
        """Test successful function call."""
        config_file = example_config_dir / subtype / (subtype + "_v0.toml")
        read_subconfiguration_file(config_file, subtype)

    def test_incorrect_subtype(self) -> None:
        """Test safety check on wrong subtype."""
        match_expr = f"Unexpected subtype invalid_subtype, expecting one "
        f"of {list(SUBCONFIG_REGISTRY.keys())}"
        with pytest.raises(ValueError, match=match_expr):
            read_subconfiguration_file("data.toml", "invalid_subtype")  # type: ignore


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
        """Test successful function call."""
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
