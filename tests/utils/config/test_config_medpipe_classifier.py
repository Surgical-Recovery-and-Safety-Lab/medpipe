"""
Test functions for the MedpipeClassifierConfig schema of the config module.
"""

from pathlib import Path
from re import escape
from typing import Literal

import pytest
from pydantic import ValidationError

from medpipe.utils.config import MedpipeClassifierConfig


class TestMedpipeClassifierConfig:
    """Test class for the MedpipeClassifierConfig class."""

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
        """Pass valid configuration to MedpipeClassifierConfig."""
        raw_config = self._get_valid_config_dict(tmp_path)
        config = MedpipeClassifierConfig.model_validate(raw_config)

        dumped = config.model_dump(exclude={"resolved_models"})
        revalidated = MedpipeClassifierConfig.model_validate(dumped)

        assert revalidated == config

    def test_extra_fields_forbidden(self, tmp_path: Path) -> None:
        """Test that unknown top-level fields raise a ValidationError."""
        raw_config = self._get_valid_config_dict(tmp_path, unexpected_flag=True)

        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            MedpipeClassifierConfig.model_validate(raw_config)

    def test_fast_run_mode_minimal_config(self, tmp_path: Path) -> None:
        """Test that 'fast' run_mode does not require cross-validation,
        fairness, or display parameters."""
        raw_config = self._get_valid_config_dict(tmp_path)
        raw_config["meta"]["run_mode"] = "fast"
        raw_config["default_model"]["recalibration"] = None
        raw_config["workflow"]["validation"]["cross_validation"] = None
        raw_config["workflow"]["validation"]["recalibration_split"] = None
        raw_config["workflow"]["evaluation"]["fairness"] = None
        raw_config["display"] = None

        config = MedpipeClassifierConfig.model_validate(raw_config)

        assert config.workflow.validation.cross_validation is None
        assert config.display is None

    def test_recalibration_split(self, tmp_path: Path) -> None:
        """Test case when recalibration method and split mismatch."""
        match_expr = (
            "Recalibration validation split must be "
            "specified when a recalibration method is used"
        )

        config = self._get_valid_config_dict(tmp_path)
        config["workflow"]["validation"]["recalibration_split"] = None

        with pytest.raises(ValueError, match=match_expr):
            MedpipeClassifierConfig.model_validate(config)

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
            MedpipeClassifierConfig.model_validate(config)

    @pytest.mark.parametrize(
        "run_mode",
        ["audit", "eval"],
    )
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
            MedpipeClassifierConfig.model_validate(config)

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
            MedpipeClassifierConfig.model_validate(config)

    def test_cascade_no_overrides(self, tmp_path: Path) -> None:
        """Test that default models map to all outcomes when there are no overrides."""
        raw_config = self._get_valid_config_dict(tmp_path)
        config = MedpipeClassifierConfig.model_validate(raw_config)

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

        config = MedpipeClassifierConfig.model_validate(raw_config)

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

        config = MedpipeClassifierConfig.model_validate(raw_config)

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

        config = MedpipeClassifierConfig.model_validate(raw_config)

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

        config = MedpipeClassifierConfig.model_validate(raw_config)

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

        config = MedpipeClassifierConfig.model_validate(raw_config)

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

        config = MedpipeClassifierConfig.model_validate(raw_config)

        assert config.resolved_models["MORTALITY_30D"].recalibration is not None
        assert config.resolved_models["ANY_COMP"].recalibration is None

    def test_cascade_recalibration_different_method_replaces_entirely(
        self, tmp_path: Path
    ) -> None:
        """Test switching recalibration method replaces base hyperparameters
        completely."""
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

        config = MedpipeClassifierConfig.model_validate(raw_config)

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

        config = MedpipeClassifierConfig.model_validate(raw_config)

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

        config = MedpipeClassifierConfig.model_validate(raw_config)

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

        config = MedpipeClassifierConfig.model_validate(raw_config)

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
            MedpipeClassifierConfig.model_validate(raw_config)

    def test_cascade_recalibration_omitted_method_keeps_base_method(
        self, tmp_path: Path
    ) -> None:
        """Test that omitting 'method' from an override's recalibration dict
        leaves the base method untouched while other fields still update."""
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
                    # 'method' intentionally omitted so it is not part of the
                    # explicitly-set fields used during cascading.
                    "recalibrate": False,
                    "hyperparameters": {"n_jobs": 2},
                },
            }
        }

        config = MedpipeClassifierConfig.model_validate(raw_config)

        res_recal = config.resolved_models["ANY_COMP"].recalibration
        assert res_recal is not None
        assert res_recal.method == "isotonic"
        assert res_recal.recalibrate is False
        assert res_recal.hyperparameters == {"out_of_bounds": "clip", "n_jobs": 2}

    def test_display_outcome_overrides_valid_outcomes_pass(
        self, tmp_path: Path
    ) -> None:
        """Test that display outcome overrides for outcomes present in
        data.outcomes validate successfully."""
        raw_config = self._get_valid_config_dict(tmp_path)

        raw_config["display"]["outcome_overrides"] = {
            "MORTALITY_30D": {"calibration": {"n_bootstraps": 10}},
            "ANY_COMP": {"calibration": {"n_bootstraps": 20}},
        }

        config = MedpipeClassifierConfig.model_validate(raw_config)

        assert set(config.display.outcome_overrides) == {"MORTALITY_30D", "ANY_COMP"}

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

        match_expr = (
            "Display outcome override 'UNLISTED_OUTCOME' is not present in "
            "data.outcomes"
        )

        with pytest.raises(ValidationError, match=escape(match_expr)):
            MedpipeClassifierConfig.model_validate(raw_config)

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

        config = MedpipeClassifierConfig.model_validate(raw_config)

        # MORTALITY_30D inherits default True, ANY_COMP overrides to False
        assert config.resolved_models["MORTALITY_30D"].recalibration
        assert config.resolved_models["ANY_COMP"].recalibration
        assert config.resolved_models["MORTALITY_30D"].recalibration.recalibrate is True
        assert config.resolved_models["ANY_COMP"].recalibration.recalibrate is False
