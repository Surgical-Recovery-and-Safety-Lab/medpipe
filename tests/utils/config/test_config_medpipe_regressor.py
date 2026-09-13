"""
Test functions for the MedpipeRegressorConfig schema of the config module.
"""

from pathlib import Path
from re import escape
from typing import Literal

import pytest
from pydantic import ValidationError

from medpipe.utils.config import MedpipeRegressorConfig


class TestMedpipeRegressorConfig:
    """Test class for the MedpipeRegressorConfig class."""

    def _get_valid_config_dict(self, tmp_path: Path, **overrides) -> dict:
        """Creates a fresh valid config dict to override."""
        config_dict = {
            "meta": {
                "project_name": "medpipe-regressor-test",
                "run_mode": "audit",
                "verbose": "compact",
            },
            "data": {
                "path": str(tmp_path / "path/to/data.csv"),
                "predictors": ["AGE", "SEX", "OP_SEVERITY"],
                "outcomes": ["LOS_DAYS", "BLOOD_LOSS_ML"],
            },
            "default_model": {
                "algorithm": "NGBRegressor",
                "hyperparameters": {"n_estimators": 300},
            },
            "workflow": {
                "random_state": 42,
                "n_jobs": 1,
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
                },
                "evaluation": {
                    "metrics": {
                        "metrics": ["rmse", "mae"],
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
                },
                "overrides": {
                    "residuals": {"n_bootstraps": 200},
                },
            },
        }
        config_dict.update(overrides)
        return config_dict

    def test_valid_config(self, tmp_path: Path) -> None:
        """Pass valid configuration to MedpipeRegressorConfig."""
        raw_config = self._get_valid_config_dict(tmp_path)
        config = MedpipeRegressorConfig.model_validate(raw_config)

        dumped = config.model_dump(exclude={"resolved_models"})
        revalidated = MedpipeRegressorConfig.model_validate(dumped)

        assert revalidated == config

    def test_extra_fields_forbidden(self, tmp_path: Path) -> None:
        """Test that unknown top-level fields raise a ValidationError."""
        raw_config = self._get_valid_config_dict(tmp_path, unexpected_flag=True)

        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            MedpipeRegressorConfig.model_validate(raw_config)

    def test_fast_run_mode_minimal_config(self, tmp_path: Path) -> None:
        """Test that 'fast' run_mode does not require cross-validation,
        fairness, or display parameters."""
        raw_config = self._get_valid_config_dict(tmp_path)
        raw_config["meta"]["run_mode"] = "fast"
        raw_config["workflow"]["validation"]["cross_validation"] = None
        raw_config["workflow"]["evaluation"]["fairness"] = None
        raw_config["display"] = None

        config = MedpipeRegressorConfig.model_validate(raw_config)

        assert config.workflow.validation.cross_validation is None
        assert config.display is None

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
            MedpipeRegressorConfig.model_validate(config)

    @pytest.mark.parametrize("run_mode", ["audit", "eval"])
    def test_evaluation(self, tmp_path: Path, run_mode: str) -> None:
        """Test case when run_mode and evaluation mismatch."""
        match_expr = (
            "Evaluation fairness parameters must be specified "
            "when run_mode is 'audit' or 'eval'"
        )
        config = self._get_valid_config_dict(tmp_path)
        config["meta"]["run_mode"] = run_mode
        config["workflow"]["evaluation"]["fairness"] = None

        with pytest.raises(ValueError, match=match_expr):
            MedpipeRegressorConfig.model_validate(config)

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
            MedpipeRegressorConfig.model_validate(config)

    def test_cascade_no_overrides(self, tmp_path: Path) -> None:
        """Test that default models map to all outcomes when there are no overrides."""
        raw_config = self._get_valid_config_dict(tmp_path)
        config = MedpipeRegressorConfig.model_validate(raw_config)

        assert "LOS_DAYS" in config.resolved_models
        assert "BLOOD_LOSS_ML" in config.resolved_models

        assert config.resolved_models["LOS_DAYS"].algorithm == "NGBRegressor"
        assert config.resolved_models["LOS_DAYS"].hyperparameters["n_estimators"] == 300

    def test_cascade_algorithm_and_hyperparameters_merge(self, tmp_path: Path) -> None:
        """Test that hyperparameters correctly deep merge when overridden."""
        raw_config = self._get_valid_config_dict(tmp_path)

        raw_config["outcome_overrides"] = {
            "BLOOD_LOSS_ML": {
                "algorithm": "LinearRegression",
                "hyperparameters": {"fit_intercept": False},
            }
        }

        config = MedpipeRegressorConfig.model_validate(raw_config)

        model_los = config.resolved_models["LOS_DAYS"]
        assert model_los.algorithm == "NGBRegressor"
        assert model_los.hyperparameters == {"n_estimators": 300}

        model_blood = config.resolved_models["BLOOD_LOSS_ML"]
        assert model_blood.algorithm == "LinearRegression"
        assert model_blood.hyperparameters == {"fit_intercept": False}

    def test_cascade_same_algorithm_deep_merges_model_hyperparameters(
        self, tmp_path: Path
    ) -> None:
        """Test that model hyperparameters deep-merge when the algorithm
        is unchanged or omitted."""
        raw_config = self._get_valid_config_dict(tmp_path)
        raw_config["default_model"] = {
            "algorithm": "NGBRegressor",
            "hyperparameters": {"n_estimators": 300, "learning_rate": 0.01},
        }
        raw_config["outcome_overrides"] = {
            "BLOOD_LOSS_ML": {
                "algorithm": "NGBRegressor",
                "hyperparameters": {"learning_rate": 0.1, "verbose": False},
            }
        }

        config = MedpipeRegressorConfig.model_validate(raw_config)

        comp_model = config.resolved_models["BLOOD_LOSS_ML"]
        assert comp_model.algorithm == "NGBRegressor"
        assert comp_model.hyperparameters == {
            "n_estimators": 300,
            "learning_rate": 0.1,
            "verbose": False,
        }

    def test_cascade_algorithm_change_replaces_hyperparameters(
        self, tmp_path: Path
    ) -> None:
        """Test changing algorithm replaces hyperparameters instead of
        merging incompatible ones."""
        raw_config = self._get_valid_config_dict(tmp_path)
        raw_config["default_model"] = {
            "algorithm": "NGBRegressor",
            "hyperparameters": {"n_estimators": 300},
        }
        raw_config["outcome_overrides"] = {
            "BLOOD_LOSS_ML": {
                "algorithm": "LinearRegression",
                "hyperparameters": {"fit_intercept": False},
            }
        }

        config = MedpipeRegressorConfig.model_validate(raw_config)

        comp_model = config.resolved_models["BLOOD_LOSS_ML"]
        assert comp_model.algorithm == "LinearRegression"
        assert comp_model.hyperparameters == {"fit_intercept": False}

    def test_outcome_override_not_in_outcomes_list(self, tmp_path: Path) -> None:
        """Test that specifying a model override for an outcome not
        listed in data.outcomes raises ValidationError."""
        raw_config = self._get_valid_config_dict(tmp_path)

        raw_config["outcome_overrides"] = {
            "UNLISTED_OUTCOME": {
                "algorithm": "LinearRegression",
            }
        }

        match_expr = (
            "Outcome override 'UNLISTED_OUTCOME' is not present in data.outcomes"
        )

        with pytest.raises(ValidationError, match=escape(match_expr)):
            MedpipeRegressorConfig.model_validate(raw_config)

    def test_display_outcome_overrides_valid_outcomes_pass(
        self, tmp_path: Path
    ) -> None:
        """Test that display outcome overrides for outcomes present in
        data.outcomes validate successfully."""
        raw_config = self._get_valid_config_dict(tmp_path)

        raw_config["display"]["outcome_overrides"] = {
            "LOS_DAYS": {"residuals": {"n_bootstraps": 10}},
            "BLOOD_LOSS_ML": {"residuals": {"n_bootstraps": 20}},
        }

        config = MedpipeRegressorConfig.model_validate(raw_config)

        assert set(config.display.outcome_overrides) == {
            "LOS_DAYS",
            "BLOOD_LOSS_ML",
        }

    def test_display_outcome_override_not_in_outcomes_list(
        self, tmp_path: Path
    ) -> None:
        """Test that specifying a display override for an outcome not
        listed in data.outcomes raises ValidationError."""
        raw_config = self._get_valid_config_dict(tmp_path)

        raw_config["display"]["outcome_overrides"] = {
            "UNLISTED_OUTCOME": {
                "residuals": {"n_bootstraps": 10},
            }
        }

        match_expr = (
            "Display outcome override 'UNLISTED_OUTCOME' is not present in "
            "data.outcomes"
        )

        with pytest.raises(ValidationError, match=escape(match_expr)):
            MedpipeRegressorConfig.model_validate(raw_config)

    def test_no_recalibration_field_on_default_model(self, tmp_path: Path) -> None:
        """Test that specifying recalibration on default_model is rejected,
        since post-hoc recalibration is not supported on the regression
        track."""
        raw_config = self._get_valid_config_dict(tmp_path)
        raw_config["default_model"]["recalibration"] = {
            "recalibrate": True,
            "method": "isotonic",
        }

        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            MedpipeRegressorConfig.model_validate(raw_config)
