import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import joblib
import numpy as np
import pandas as pd
import pytest

from medpipe.pipeline.pipeline import Medpipe
from medpipe.utils.config import MedpipeConfig
from medpipe.utils.io import read_toml_configuration

# ==============================================================================
# 1. UNIT TESTS (Mocked Sub-components & Routing Validation)
# ==============================================================================


class TestMedpipeUnit:
    """Unit tests verifying orchestration delegation and argument routing."""

    @patch("medpipe.pipeline.pipeline.MedpipeOrchestrator")
    @patch("medpipe.pipeline.pipeline.MedpipeRunner")
    @patch("medpipe.pipeline.pipeline.MedpipeEvaluator")
    def test_medpipe_initialization(
        self, mock_eval_cls, mock_runner_cls, mock_orch_cls
    ):
        """Verify Medpipe initializes sub-orchestrators correctly."""
        mock_config = MagicMock(spec=MedpipeConfig)
        mock_orch_instance = mock_orch_cls.return_value

        mp = Medpipe(config=mock_config)

        mock_orch_cls.assert_called_once_with(mock_config, "artifacts", None)
        mock_runner_cls.assert_called_once_with(orchestrator=mock_orch_instance)
        mock_eval_cls.assert_called_once_with(
            orchestrator=mock_orch_instance, runner=mock_runner_cls.return_value
        )
        assert mp.mp_config == mock_orch_instance.config

    @patch("medpipe.pipeline.pipeline.MedpipeOrchestrator")
    @patch("medpipe.pipeline.pipeline.MedpipeRunner")
    @patch("medpipe.pipeline.pipeline.MedpipeEvaluator")
    def test_inference_delegation(self, mock_eval_cls, mock_runner_cls, mock_orch_cls):
        """Verify predict, predict_proba, and decision_function delegate to evaluator."""
        mp = Medpipe(config=MagicMock())
        X = pd.DataFrame({"A": [1, 2]})

        mp.predict(X, outcome="MORTALITY")
        mp.evaluator.predict.assert_called_once_with(
            X=X, model=None, outcome="MORTALITY"
        )

        mp.predict_proba(X, outcome="MORTALITY")
        mp.evaluator.predict_proba.assert_called_once_with(
            X=X, model=None, outcome="MORTALITY"
        )

        mp.decision_function(X, outcome="MORTALITY")
        mp.evaluator.decision_function.assert_called_once_with(
            X=X, model=None, outcome="MORTALITY"
        )

    @patch("medpipe.pipeline.pipeline.MedpipeOrchestrator")
    @patch("medpipe.pipeline.pipeline.MedpipeRunner")
    @patch("medpipe.pipeline.pipeline.MedpipeEvaluator")
    def test_evaluate_y_dataframe_resolution(
        self, mock_eval_cls, mock_runner_cls, mock_orch_cls
    ):
        """Verify y DataFrame slicing resolution logic in evaluate()."""
        mp = Medpipe(config=MagicMock())
        X = pd.DataFrame({"AGE": [50, 60]})
        y_df = pd.DataFrame(
            {
                "MORTALITY_30D": [0, 1],
                "READMISSION_90D": [1, 0],
            }
        )

        # When outcome matches a column in y_df
        mp.evaluate(X, y_df, outcome="MORTALITY_30D")
        _, kwargs = mp.evaluator.evaluate.call_args
        pd.testing.assert_series_equal(kwargs["y"], y_df["MORTALITY_30D"])

        # When y is a single-column DataFrame without outcome specified
        y_single = pd.DataFrame({"TARGET": [0, 1]})
        mp.evaluate(X, y_single)
        _, kwargs = mp.evaluator.evaluate.call_args
        pd.testing.assert_series_equal(kwargs["y"], y_single.iloc[:, 0])


class TestMedpipeFit:
    """Unit tests verifying orchestration delegation and argument routing in Medpipe.fit."""

    @patch("medpipe.pipeline.pipeline.MedpipeDisplayer")
    @patch("medpipe.pipeline.pipeline.MedpipeEvaluator")
    @patch("medpipe.pipeline.pipeline.MedpipeRunner")
    @patch("medpipe.pipeline.pipeline.MedpipeOrchestrator")
    def test_fit_delegation_with_all_arguments(
        self, mock_orch_cls, mock_runner_cls, mock_eval_cls, mock_displayer_cls
    ):
        """Verify fit() delegates to runner.run() with complete parameter mappings."""
        mp = Medpipe(config=MagicMock())

        # Sample datasets
        X_train = pd.DataFrame({"AGE": [50, 60], "BMI": [22.5, 28.1]})
        y_train = pd.DataFrame({"MORTALITY_30D": [0, 1]})
        X_recal = pd.DataFrame({"AGE": [55], "BMI": [25.0]})
        y_recal = pd.DataFrame({"MORTALITY_30D": [0]})
        groups_train = np.array([1, 2])

        # Mock runner return value
        expected_fitted_models = {"MORTALITY_30D": MagicMock()}
        mp.runner.run.return_value = expected_fitted_models

        # Execute fit
        result = mp.fit(
            X_train=X_train,
            y_train=y_train,
            X_recal=X_recal,
            y_recal=y_recal,
            groups_train=groups_train,
        )

        # Assert delegation and argument translation (e.g. y_train -> y_train_df)
        mp.runner.run.assert_called_once_with(
            X_train=X_train,
            y_train_df=y_train,
            X_recal=X_recal,
            y_recal_df=y_recal,
            groups_train=groups_train,
        )
        assert result == expected_fitted_models

    @patch("medpipe.pipeline.pipeline.MedpipeDisplayer")
    @patch("medpipe.pipeline.pipeline.MedpipeEvaluator")
    @patch("medpipe.pipeline.pipeline.MedpipeRunner")
    @patch("medpipe.pipeline.pipeline.MedpipeOrchestrator")
    def test_fit_delegation_with_defaults(
        self, mock_orch_cls, mock_runner_cls, mock_eval_cls, mock_displayer_cls
    ):
        """Verify fit() passes None defaults for optional recalibration and group parameters."""
        mp = Medpipe(config=MagicMock())

        X_train = pd.DataFrame({"AGE": [50, 60]})
        y_train = pd.DataFrame({"MORTALITY_30D": [0, 1]})

        mp.runner.run.return_value = {}

        result = mp.fit(X_train=X_train, y_train=y_train)

        mp.runner.run.assert_called_once_with(
            X_train=X_train,
            y_train_df=y_train,
            X_recal=None,
            y_recal_df=None,
            groups_train=None,
        )
        assert result == {}


class TestMedpipePredict:
    """Unit tests verifying delegation and parameter passing in Medpipe.predict."""

    @patch("medpipe.pipeline.pipeline.MedpipeDisplayer")
    @patch("medpipe.pipeline.pipeline.MedpipeEvaluator")
    @patch("medpipe.pipeline.pipeline.MedpipeRunner")
    @patch("medpipe.pipeline.pipeline.MedpipeOrchestrator")
    def test_predict_delegation_with_dataframe(
        self, mock_orch_cls, mock_runner_cls, mock_eval_cls, mock_displayer_cls
    ):
        """Verify predict passes DataFrame inputs and outcome kwargs to evaluator."""
        mp = Medpipe(config=MagicMock())
        X = pd.DataFrame({"AGE": [50, 60], "BMI": [22.5, 28.1]})
        expected_preds = np.array([0, 1])
        mp.evaluator.predict.return_value = expected_preds

        preds = mp.predict(X, outcome="MORTALITY_30D")

        mp.evaluator.predict.assert_called_once_with(
            X=X, model=None, outcome="MORTALITY_30D"
        )
        np.testing.assert_array_equal(preds, expected_preds)

    @patch("medpipe.pipeline.pipeline.MedpipeDisplayer")
    @patch("medpipe.pipeline.pipeline.MedpipeEvaluator")
    @patch("medpipe.pipeline.pipeline.MedpipeRunner")
    @patch("medpipe.pipeline.pipeline.MedpipeOrchestrator")
    def test_predict_delegation_with_explicit_model_and_ndarray(
        self, mock_orch_cls, mock_runner_cls, mock_eval_cls, mock_displayer_cls
    ):
        """Verify predict forwards explicit model instances and numpy arrays."""
        mp = Medpipe(config=MagicMock())
        X = np.array([[50, 22.5], [60, 28.1]])
        mock_model = MagicMock()
        expected_preds = np.array([1, 0])
        mp.evaluator.predict.return_value = expected_preds

        preds = mp.predict(X, model=mock_model)

        mp.evaluator.predict.assert_called_once_with(
            X=X, model=mock_model, outcome=None
        )
        np.testing.assert_array_equal(preds, expected_preds)


class TestMedpipePredictProba:
    """Unit tests verifying delegation and parameter passing in Medpipe.predict_proba."""

    @patch("medpipe.pipeline.pipeline.MedpipeDisplayer")
    @patch("medpipe.pipeline.pipeline.MedpipeEvaluator")
    @patch("medpipe.pipeline.pipeline.MedpipeRunner")
    @patch("medpipe.pipeline.pipeline.MedpipeOrchestrator")
    def test_predict_proba_delegation_with_outcome(
        self, mock_orch_cls, mock_runner_cls, mock_eval_cls, mock_displayer_cls
    ):
        """Verify predict_proba delegates correctly when outcome target is specified."""
        mp = Medpipe(config=MagicMock())
        X = pd.DataFrame({"AGE": [50, 60]})
        expected_probas = np.array([[0.8, 0.2], [0.3, 0.7]])
        mp.evaluator.predict_proba.return_value = expected_probas

        probas = mp.predict_proba(X, outcome="READMISSION_90D")

        mp.evaluator.predict_proba.assert_called_once_with(
            X=X, model=None, outcome="READMISSION_90D"
        )
        np.testing.assert_array_equal(probas, expected_probas)

    @patch("medpipe.pipeline.pipeline.MedpipeDisplayer")
    @patch("medpipe.pipeline.pipeline.MedpipeEvaluator")
    @patch("medpipe.pipeline.pipeline.MedpipeRunner")
    @patch("medpipe.pipeline.pipeline.MedpipeOrchestrator")
    def test_predict_proba_delegation_with_defaults(
        self, mock_orch_cls, mock_runner_cls, mock_eval_cls, mock_displayer_cls
    ):
        """Verify predict_proba passes None defaults when model and outcome are omitted."""
        mp = Medpipe(config=MagicMock())
        X = pd.DataFrame({"AGE": [50]})
        expected_probas = np.array([0.15])
        mp.evaluator.predict_proba.return_value = expected_probas

        probas = mp.predict_proba(X)

        mp.evaluator.predict_proba.assert_called_once_with(
            X=X, model=None, outcome=None
        )
        np.testing.assert_array_equal(probas, expected_probas)


class TestMedpipeDecisionFunction:
    """Unit tests verifying delegation and parameter passing in Medpipe.decision_function."""

    @patch("medpipe.pipeline.pipeline.MedpipeDisplayer")
    @patch("medpipe.pipeline.pipeline.MedpipeEvaluator")
    @patch("medpipe.pipeline.pipeline.MedpipeRunner")
    @patch("medpipe.pipeline.pipeline.MedpipeOrchestrator")
    def test_decision_function_delegation_with_outcome(
        self, mock_orch_cls, mock_runner_cls, mock_eval_cls, mock_displayer_cls
    ):
        """Verify decision_function forwards target outcome identifier to evaluator."""
        mp = Medpipe(config=MagicMock())
        X = pd.DataFrame({"AGE": [50, 60]})
        expected_scores = np.array([-1.2, 2.4])
        mp.evaluator.decision_function.return_value = expected_scores

        scores = mp.decision_function(X, outcome="MORTALITY_30D")

        mp.evaluator.decision_function.assert_called_once_with(
            X=X, model=None, outcome="MORTALITY_30D"
        )
        np.testing.assert_array_equal(scores, expected_scores)

    @patch("medpipe.pipeline.pipeline.MedpipeDisplayer")
    @patch("medpipe.pipeline.pipeline.MedpipeEvaluator")
    @patch("medpipe.pipeline.pipeline.MedpipeRunner")
    @patch("medpipe.pipeline.pipeline.MedpipeOrchestrator")
    def test_decision_function_delegation_with_explicit_model(
        self, mock_orch_cls, mock_runner_cls, mock_eval_cls, mock_displayer_cls
    ):
        """Verify decision_function passes explicit model override."""
        mp = Medpipe(config=MagicMock())
        X = pd.DataFrame({"AGE": [50]})
        mock_model = MagicMock()
        expected_scores = np.array([0.85])
        mp.evaluator.decision_function.return_value = expected_scores

        scores = mp.decision_function(X, model=mock_model)

        mp.evaluator.decision_function.assert_called_once_with(
            X=X, model=mock_model, outcome=None
        )
        np.testing.assert_array_equal(scores, expected_scores)


class TestMedpipeEvaluate:
    """Unit tests verifying dataset resolution and delegation in Medpipe.evaluate."""

    @patch("medpipe.pipeline.pipeline.MedpipeDisplayer")
    @patch("medpipe.pipeline.pipeline.MedpipeEvaluator")
    @patch("medpipe.pipeline.pipeline.MedpipeRunner")
    @patch("medpipe.pipeline.pipeline.MedpipeOrchestrator")
    def test_evaluate_delegation_basic_series(
        self, mock_orch_cls, mock_runner_cls, mock_eval_cls, mock_displayer_cls
    ):
        """Verify evaluate forwards Series target and default kwargs to evaluator."""
        mp = Medpipe(config=MagicMock())
        X = pd.DataFrame({"AGE": [50, 60]})
        y = pd.Series([0, 1], name="MORTALITY_30D")
        expected_eval = {"overall": {"roc_auc": 0.88}}
        mp.evaluator.evaluate.return_value = expected_eval

        results = mp.evaluate(X, y, outcome="MORTALITY_30D")

        mp.evaluator.evaluate.assert_called_once_with(
            X=X,
            y=y,
            outcome="MORTALITY_30D",
            model=None,
            metrics=None,
            subgroup_specs=None,
            save_artifacts=True,
        )
        assert results == expected_eval

    @patch("medpipe.pipeline.pipeline.MedpipeDisplayer")
    @patch("medpipe.pipeline.pipeline.MedpipeEvaluator")
    @patch("medpipe.pipeline.pipeline.MedpipeRunner")
    @patch("medpipe.pipeline.pipeline.MedpipeOrchestrator")
    def test_evaluate_y_dataframe_outcome_column_resolution(
        self, mock_orch_cls, mock_runner_cls, mock_eval_cls, mock_displayer_cls
    ):
        """Verify evaluate slices y DataFrame when outcome matches a column name."""
        mp = Medpipe(config=MagicMock())
        X = pd.DataFrame({"AGE": [50, 60]})
        y_df = pd.DataFrame({"MORTALITY_30D": [0, 1], "READMISSION_90D": [1, 0]})
        mp.evaluator.evaluate.return_value = {}

        mp.evaluate(X, y_df, outcome="READMISSION_90D")

        _, kwargs = mp.evaluator.evaluate.call_args
        pd.testing.assert_series_equal(kwargs["y"], y_df["READMISSION_90D"])

    @patch("medpipe.pipeline.pipeline.MedpipeDisplayer")
    @patch("medpipe.pipeline.pipeline.MedpipeEvaluator")
    @patch("medpipe.pipeline.pipeline.MedpipeRunner")
    @patch("medpipe.pipeline.pipeline.MedpipeOrchestrator")
    def test_evaluate_y_single_column_dataframe_fallback(
        self, mock_orch_cls, mock_runner_cls, mock_eval_cls, mock_displayer_cls
    ):
        """Verify evaluate extracts the first column if y is 1-col DataFrame without matching outcome name."""
        mp = Medpipe(config=MagicMock())
        X = pd.DataFrame({"AGE": [50, 60]})
        y_df = pd.DataFrame({"GENERIC_LABEL": [1, 0]})
        mp.evaluator.evaluate.return_value = {}

        mp.evaluate(X, y_df, outcome="CUSTOM_NAME")

        _, kwargs = mp.evaluator.evaluate.call_args
        pd.testing.assert_series_equal(kwargs["y"], y_df.iloc[:, 0])

    @patch("medpipe.pipeline.pipeline.MedpipeDisplayer")
    @patch("medpipe.pipeline.pipeline.MedpipeEvaluator")
    @patch("medpipe.pipeline.pipeline.MedpipeRunner")
    @patch("medpipe.pipeline.pipeline.MedpipeOrchestrator")
    def test_evaluate_explicit_parameters(
        self, mock_orch_cls, mock_runner_cls, mock_eval_cls, mock_displayer_cls
    ):
        """Verify evaluate forwards custom metrics, subgroup_specs, model, and save_artifacts flag."""
        mp = Medpipe(config=MagicMock())
        X = pd.DataFrame({"AGE": [50]})
        y = np.array([1])
        mock_model = MagicMock()
        custom_metrics = ["brier_score"]
        subgroups = {"age_65": "AGE >= 65"}

        mp.evaluate(
            X=X,
            y=y,
            outcome="MORTALITY",
            model=mock_model,
            metrics=custom_metrics,
            subgroup_specs=subgroups,
            save_artifacts=False,
        )

        mp.evaluator.evaluate.assert_called_once_with(
            X=X,
            y=y,
            outcome="MORTALITY",
            model=mock_model,
            metrics=custom_metrics,
            subgroup_specs=subgroups,
            save_artifacts=False,
        )


class TestMedpipeRun:
    """Unit tests verifying full workflow sequence and branching in Medpipe.run."""

    @patch("medpipe.pipeline.pipeline.MedpipeDisplayer")
    @patch("medpipe.pipeline.pipeline.MedpipeEvaluator")
    @patch("medpipe.pipeline.pipeline.MedpipeRunner")
    @patch("medpipe.pipeline.pipeline.MedpipeOrchestrator")
    def test_run_fast_mode_execution_flow(
        self, mock_orch_cls, mock_runner_cls, mock_eval_cls, mock_displayer_cls
    ):
        """Verify run executes data prep, fit, and test evaluation without generating plots in fast mode."""
        mp = Medpipe(config=MagicMock())
        mp.mp_config.meta.run_mode = "fast"
        mp.mp_config.data.kwargs = {"extra_arg": 0.2}
        mp.orchestrator.config.data.outcomes = ["MORTALITY_30D"]
        mp.orchestrator.get_subgroup_specs.return_value = {"age_gt_60": "AGE > 60"}

        # Mock split outputs
        X_tr, y_tr = pd.DataFrame({"A": [1, 2]}), pd.DataFrame(
            {"MORTALITY_30D": [0, 1]}
        )
        X_te, y_te = pd.DataFrame({"A": [3]}), pd.DataFrame({"MORTALITY_30D": [1]})
        mp.orchestrator.prepare_data.return_value = (
            X_tr,
            y_tr,
            None,
            None,
            X_te,
            y_te,
            None,
        )

        mp.fit = MagicMock(return_value={"MORTALITY_30D": "fitted_model"})
        mp.evaluate = MagicMock(return_value={"overall": {"roc_auc": 0.9}})

        results = mp.run(groups_train=None)

        mp.orchestrator.prepare_data.assert_called_once_with(extra_arg=0.2)
        mp.fit.assert_called_once_with(
            X_train=X_tr,
            y_train=y_tr,
            X_recal=None,
            y_recal=None,
            groups_train=None,
        )
        mp.evaluate.assert_called_once_with(
            X=X_te,
            y=y_te["MORTALITY_30D"].to_numpy(),
            outcome="MORTALITY_30D",
            subgroup_specs={"age_gt_60": "AGE > 60"},
            save_artifacts=True,
        )

        assert results["fitted_models"] == {"MORTALITY_30D": "fitted_model"}
        assert results["evaluations"] == {
            "MORTALITY_30D": {"overall": {"roc_auc": 0.9}}
        }
        assert "plots" not in results

    @patch("medpipe.pipeline.pipeline.MedpipeDisplayer")
    @patch("medpipe.pipeline.pipeline.MedpipeEvaluator")
    @patch("medpipe.pipeline.pipeline.MedpipeRunner")
    @patch("medpipe.pipeline.pipeline.MedpipeOrchestrator")
    def test_run_audit_mode_triggers_visualization_and_heatmaps(
        self, mock_orch_cls, mock_runner_cls, mock_eval_cls, mock_displayer_cls
    ):
        """Verify run executes plotting routines and generates strata heatmaps in audit/eval mode."""
        mp = Medpipe(config=MagicMock())
        mp.mp_config.meta.run_mode = "audit"
        mp.mp_config.data.kwargs = {}
        mp.orchestrator.config.data.outcomes = ["MORTALITY_30D"]
        mp.orchestrator.get_subgroup_specs.return_value = {}

        X_tr, y_tr = pd.DataFrame({"A": [1]}), pd.DataFrame({"MORTALITY_30D": [0]})
        X_te, y_te = pd.DataFrame({"A": [2]}), pd.DataFrame({"MORTALITY_30D": [1]})
        mp.orchestrator.prepare_data.return_value = (
            X_tr,
            y_tr,
            None,
            None,
            X_te,
            y_te,
            None,
        )

        mp.fit = MagicMock(return_value={"MORTALITY_30D": "model"})
        mp.evaluate = MagicMock(return_value={"overall": {}})
        mp.predict_proba = MagicMock(return_value=np.array([0.85]))
        mp.plot_all = MagicMock(return_value={"roc": ("fig_obj", "ax_obj")})
        mp.displayer.plot_all_heatmaps.return_value = {
            "auc_heatmap": ("fig_hm", "ax_hm")
        }

        results = mp.run()

        mp.predict_proba.assert_called_once_with(X=X_te, outcome="MORTALITY_30D")
        mp.plot_all.assert_called_once_with(
            y_true=y_te["MORTALITY_30D"].to_numpy(),
            probas=np.array([0.85]),
            outcome="MORTALITY_30D",
        )
        mp.displayer.plot_all_heatmaps.assert_called_once_with(
            evaluations={"MORTALITY_30D": {"overall": {}}}
        )

        assert "plots" in results
        assert results["plots"]["MORTALITY_30D"] == {"roc": ("fig_obj", "ax_obj")}
        assert results["plots"]["strata_heatmaps"] == {
            "auc_heatmap": ("fig_hm", "ax_hm")
        }


class TestMedpipePlotAll:
    """Unit tests verifying parameter forwarding in Medpipe.plot_all."""

    @patch("medpipe.pipeline.pipeline.MedpipeDisplayer")
    @patch("medpipe.pipeline.pipeline.MedpipeEvaluator")
    @patch("medpipe.pipeline.pipeline.MedpipeRunner")
    @patch("medpipe.pipeline.pipeline.MedpipeOrchestrator")
    def test_plot_all_delegation(
        self, mock_orch_cls, mock_runner_cls, mock_eval_cls, mock_displayer_cls
    ):
        """Verify plot_all forwards all inputs and style kwargs to Displayer."""
        mp = Medpipe(config=MagicMock())
        y_true = np.array([0, 1, 0, 1])
        probas = np.array([0.1, 0.8, 0.2, 0.9])
        expected_plots = {"roc": (MagicMock(), MagicMock())}
        mp.displayer.plot_all.return_value = expected_plots

        plots = mp.plot_all(
            y_true=y_true,
            probas=probas,
            outcome="READMISSION_90D",
            n_bootstraps=500,
            save=False,
            show=True,
            dpi=300,
            color="red",
        )

        mp.displayer.plot_all.assert_called_once_with(
            y_true=y_true,
            probas=probas,
            outcome="READMISSION_90D",
            n_bootstraps=500,
            save=False,
            show=True,
            dpi=300,
            color="red",
        )
        assert plots == expected_plots


class TestMedpipeLoad:
    """Test suite for the Medpipe.load class factory method."""

    @pytest.fixture
    def valid_config_dict(self, tmp_path: Path) -> dict:
        """Provides a minimal valid raw configuration dictionary."""
        return {
            "meta": {
                "project_name": "demo_project",
                "run_mode": "fast",
                "verbose": "compact",
            },
            "data": {
                "path": str(tmp_path / "data.csv"),
                "predictors": ["AGE", "SEX"],
                "outcomes": ["MORTALITY_30D"],
            },
            "default_model": {
                "algorithm": "HistGradientBoostingClassifier",
                "hyperparameters": {},
            },
            "workflow": {
                "validation": {
                    "test_split": {"strategy": "random", "test_size": 0.2},
                },
                "evaluation": {
                    "metrics": {"metrics": ["roc_auc"]},
                },
            },
        }

    def test_load_missing_config_raises_file_not_found(self, tmp_path: Path) -> None:
        """Test that FileNotFoundError is raised if resolved_config.json is missing."""
        empty_run_dir = tmp_path / "empty_run"
        empty_run_dir.mkdir()

        with pytest.raises(
            FileNotFoundError,
            match="Cannot load Medpipe instance: Configuration JSON missing",
        ):
            Medpipe.load(empty_run_dir)

    def test_load_successful_without_models_directory(
        self, tmp_path: Path, valid_config_dict: dict
    ) -> None:
        """Test successful reconstruction of Medpipe when no models directory is present."""
        run_dir = tmp_path / "run_2026_08_10"
        run_dir.mkdir()

        config_path = run_dir / "resolved_config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(valid_config_dict, f)

        pipe = Medpipe.load(run_dir)

        assert isinstance(pipe, Medpipe)
        assert pipe.orchestrator.run_dir == run_dir
        assert pipe.displayer.run_dir == run_dir
        assert pipe.mp_config.meta.project_name == "demo_project"

    def test_load_successful_with_fitted_models(
        self, tmp_path: Path, valid_config_dict: dict
    ) -> None:
        """Test loading and restoring serialized fitted models into runner."""
        run_dir = tmp_path / "run_2026_08_10"
        models_dir = run_dir / "models"
        models_dir.mkdir(parents=True)

        # Write resolved JSON configuration
        config_path = run_dir / "resolved_config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(valid_config_dict, f)

        # Serialize fitted models dictionary matching project_name
        mock_fitted_models = {"MORTALITY_30D": "fitted_model_placeholder"}
        model_artifact = models_dir / "demo_project_fitted.joblib"
        joblib.dump(mock_fitted_models, model_artifact)

        # Pass run_dir as str to test string path resolution
        pipe = Medpipe.load(str(run_dir))

        assert isinstance(pipe, Medpipe)
        assert pipe.orchestrator.run_dir == run_dir
        assert pipe.displayer.run_dir == run_dir
        assert pipe.runner.fitted_models == mock_fitted_models


# ==============================================================================
# 2. END-TO-END STRESS TESTS (Minimal Synthetic Dataset & Configuration Branches)
# ==============================================================================


@pytest.fixture
def repo_root(request: pytest.FixtureRequest) -> Path:
    """Resolve repository root directory from pytest config."""
    return Path(request.config.rootpath)


@pytest.fixture
def test_data_path(repo_root: Path) -> Path:
    """Locate tests/test_data.csv relative to repo root."""
    path = repo_root / "tests" / "test_data.csv"
    if not path.exists():
        path = repo_root / "test" / "test_data.csv"
    return path


@pytest.fixture
def base_config_path(repo_root: Path) -> Path:
    """Locate examples/default_config.toml relative to repo root."""
    return repo_root / "examples" / "default_config.toml"


@pytest.fixture
def build_medpipe_config(test_data_path: Path, base_config_path: Path):
    """Factory fixture loading default_config.toml and updating data.path dynamically."""

    def _factory(run_mode: str = "fast", disable_recal: bool = False) -> MedpipeConfig:
        config = read_toml_configuration(base_config_path)
        config.data.path = str(test_data_path)
        config.meta.run_mode = run_mode

        if disable_recal:
            config.default_model.recalibration = None
            for override in config.outcome_overrides.values():
                override.recalibration = None
            config.workflow.validation.recalibration_split = None

        return config

    return _factory


class TestMedpipeStressIntegration:
    """Integration stress tests executing Medpipe and verifying artifact lifecycle."""

    @pytest.mark.parametrize("run_mode", ["fast", "cv", "eval", "audit"])
    @pytest.mark.parametrize("disable_recal", [False, True])
    def test_run_mode_and_recalibration_branches(
        self, build_medpipe_config, tmp_path: Path, run_mode: str, disable_recal: bool
    ) -> None:
        """Executes pipeline end-to-end, verifies artifact creation, and deletes artifacts."""
        try:
            config = build_medpipe_config(
                run_mode=run_mode, disable_recal=disable_recal
            )
        except Exception as err:
            pytest.skip(
                f"Config combination incompatible with test dataset state: {err}"
            )

        # Direct artifacts to an isolated temporary directory
        artifact_dir = tmp_path / "artifacts"
        pipeline = Medpipe(config=config, base_artifact_dir=artifact_dir)
        run_dir = pipeline.orchestrator.run_dir

        try:
            results = pipeline.run()

            # 1. Validate returned pipeline output structure
            assert "fitted_models" in results
            assert "evaluations" in results
            assert len(results["fitted_models"]) == len(config.data.outcomes)
            assert len(results["evaluations"]) == len(config.data.outcomes)

            # 2. Verify expected artifacts were created on disk
            assert run_dir.exists()
            assert (run_dir / "env_state.json").exists()

            models_dir = run_dir / "models"
            assert models_dir.exists()
            for outcome in config.data.outcomes:
                assert (models_dir / f"{outcome}_model.joblib").exists()
                assert (run_dir / f"{outcome}_evaluation_results.json").exists()

        finally:
            # 3. Clean up/delete the run directory after verification
            if run_dir.exists():
                # Close any active logging handlers attached to the run file to release file locks
                for handler in list(pipeline.logger.handlers):
                    handler.close()
                    pipeline.logger.removeHandler(handler)
                shutil.rmtree(run_dir, ignore_errors=True)

            assert not run_dir.exists()

    def test_manual_workflow_fit_predict_evaluate_cleanup(
        self, build_medpipe_config, tmp_path: Path
    ) -> None:
        """Validates manual step-by-step API execution and artifact cleanup."""
        config = build_medpipe_config(run_mode="fast", disable_recal=True)
        artifact_dir = tmp_path / "artifacts"
        pipeline = Medpipe(config=config, base_artifact_dir=artifact_dir)
        run_dir = pipeline.orchestrator.run_dir

        try:
            X_train, y_train, X_recal, y_recal, X_test, y_test, groups_train = (
                pipeline.orchestrator.prepare_data()
            )

            fitted = pipeline.fit(X_train, y_train, X_recal, y_recal, groups_train)
            assert set(fitted.keys()) == set(config.data.outcomes)

            for outcome in config.data.outcomes:
                eval_dict = pipeline.evaluate(
                    X=X_test,
                    y=y_test[outcome],
                    outcome=outcome,
                    save_artifacts=True,
                )
                assert "overall" in eval_dict
                assert (run_dir / f"{outcome}_evaluation_results.json").exists()

        finally:
            if run_dir.exists():
                for handler in list(pipeline.logger.handlers):
                    handler.close()
                    pipeline.logger.removeHandler(handler)
                shutil.rmtree(run_dir, ignore_errors=True)

            assert not run_dir.exists()
