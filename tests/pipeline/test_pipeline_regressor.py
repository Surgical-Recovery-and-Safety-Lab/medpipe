"""
Tests for MedpipeRegressor, mirroring tests/pipeline/test_pipeline.py's
structure for MedpipeClassifier (minus plotting, which isn't implemented
for the regression track yet).
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import joblib
import numpy as np
import pandas as pd
import pytest

from medpipe.pipeline.pipeline import MedpipeRegressor
from medpipe.utils.config import MedpipeRegressorConfig


class TestMedpipeRegressorUnit:
    """Unit tests verifying orchestration delegation and argument routing."""

    @patch("medpipe.pipeline.pipeline.MedpipeOrchestrator")
    @patch("medpipe.pipeline.pipeline.MedpipeRegressorRunner")
    @patch("medpipe.pipeline.pipeline.MedpipeRegressorEvaluator")
    def test_medpipe_regressor_initialization(
        self, mock_eval_cls, mock_runner_cls, mock_orch_cls
    ):
        """Verify MedpipeRegressor initializes sub-orchestrators correctly."""
        mock_config = MagicMock(spec=MedpipeRegressorConfig)
        mock_orch_instance = mock_orch_cls.return_value

        mp = MedpipeRegressor(config=mock_config)

        mock_orch_cls.assert_called_once_with(mock_config, "artifacts", None)
        mock_runner_cls.assert_called_once_with(orchestrator=mock_orch_instance)
        mock_eval_cls.assert_called_once_with(
            orchestrator=mock_orch_instance, runner=mock_runner_cls.return_value
        )
        assert mp.mp_config == mock_orch_instance.config

    @patch("medpipe.pipeline.pipeline.read_regressor_toml_configuration")
    @patch("medpipe.pipeline.pipeline.MedpipeOrchestrator")
    @patch("medpipe.pipeline.pipeline.MedpipeRegressorRunner")
    @patch("medpipe.pipeline.pipeline.MedpipeRegressorEvaluator")
    def test_medpipe_regressor_parses_string_path_as_regressor_config(
        self, mock_eval_cls, mock_runner_cls, mock_orch_cls, mock_read_regressor_toml
    ):
        """Verify a string/Path config is parsed via
        read_regressor_toml_configuration (not the classifier loader),
        since MedpipeOrchestrator always treats a bare path as a classifier
        config."""
        mock_parsed_config = MagicMock(spec=MedpipeRegressorConfig)
        mock_read_regressor_toml.return_value = mock_parsed_config

        MedpipeRegressor(config="path/to/regressor_config.toml")

        mock_read_regressor_toml.assert_called_once_with(
            "path/to/regressor_config.toml"
        )
        mock_orch_cls.assert_called_once_with(mock_parsed_config, "artifacts", None)

    @patch("medpipe.pipeline.pipeline.MedpipeOrchestrator")
    @patch("medpipe.pipeline.pipeline.MedpipeRegressorRunner")
    @patch("medpipe.pipeline.pipeline.MedpipeRegressorEvaluator")
    def test_inference_delegation(self, mock_eval_cls, mock_runner_cls, mock_orch_cls):
        """Verify predict and predict_dist delegate to evaluator."""
        mp = MedpipeRegressor(config=MagicMock())
        X = pd.DataFrame({"A": [1, 2]})

        mp.predict(X, outcome="LOS_DAYS")
        mp._evaluator.predict.assert_called_once_with(
            X=X, model=None, outcome="LOS_DAYS"
        )

        mp.predict_dist(X, outcome="LOS_DAYS")
        mp._evaluator.predict_dist.assert_called_once_with(
            X=X, model=None, outcome="LOS_DAYS"
        )

    @patch("medpipe.pipeline.pipeline.MedpipeOrchestrator")
    @patch("medpipe.pipeline.pipeline.MedpipeRegressorRunner")
    @patch("medpipe.pipeline.pipeline.MedpipeRegressorEvaluator")
    def test_evaluate_y_dataframe_resolution(
        self, mock_eval_cls, mock_runner_cls, mock_orch_cls
    ):
        """Verify y DataFrame slicing resolution logic in evaluate()."""
        mp = MedpipeRegressor(config=MagicMock())
        X = pd.DataFrame({"AGE": [50, 60]})
        y_df = pd.DataFrame(
            {
                "LOS_DAYS": [3.5, 7.2],
                "BLOOD_LOSS_ML": [120.0, 300.0],
            }
        )

        mp.evaluate(X, y_df, outcome="LOS_DAYS")
        _, kwargs = mp._evaluator.evaluate.call_args
        pd.testing.assert_series_equal(kwargs["y"], y_df["LOS_DAYS"])

        y_single = pd.DataFrame({"TARGET": [1.0, 2.0]})
        mp.evaluate(X, y_single)
        _, kwargs = mp._evaluator.evaluate.call_args
        pd.testing.assert_series_equal(kwargs["y"], y_single.iloc[:, 0])

        # Multi-column DataFrame with no outcome match: passed through as-is
        mp.evaluate(X, y_df)
        _, kwargs = mp._evaluator.evaluate.call_args
        pd.testing.assert_frame_equal(kwargs["y"], y_df)


class TestMedpipeRegressorProperties:
    """Unit tests for MedpipeRegressor's thin delegating properties."""

    @patch("medpipe.pipeline.pipeline.MedpipeRegressorEvaluator")
    @patch("medpipe.pipeline.pipeline.MedpipeRegressorRunner")
    @patch("medpipe.pipeline.pipeline.MedpipeOrchestrator")
    def test_models_returns_runner_fitted_models(
        self, mock_orch_cls, mock_runner_cls, mock_eval_cls
    ):
        """Test that models delegates to runner.fitted_models."""
        mp = MedpipeRegressor(config=MagicMock())
        mp._runner.fitted_models = {"LOS_DAYS": "a_model"}

        assert mp.models == {"LOS_DAYS": "a_model"}

    @patch("medpipe.pipeline.pipeline.MedpipeRegressorEvaluator")
    @patch("medpipe.pipeline.pipeline.MedpipeRegressorRunner")
    @patch("medpipe.pipeline.pipeline.MedpipeOrchestrator")
    def test_is_fitted_false_when_no_models(
        self, mock_orch_cls, mock_runner_cls, mock_eval_cls
    ):
        """Test that is_fitted is False when fitted_models is empty."""
        mp = MedpipeRegressor(config=MagicMock())
        mp._runner.fitted_models = {}

        assert mp.is_fitted is False

    @patch("medpipe.pipeline.pipeline.MedpipeRegressorEvaluator")
    @patch("medpipe.pipeline.pipeline.MedpipeRegressorRunner")
    @patch("medpipe.pipeline.pipeline.MedpipeOrchestrator")
    def test_run_dir_returns_orchestrator_run_dir(
        self, mock_orch_cls, mock_runner_cls, mock_eval_cls
    ):
        """Test that run_dir delegates to orchestrator.run_dir."""
        mp = MedpipeRegressor(config=MagicMock())
        mp._orchestrator.run_dir = Path("/fake/run/dir")

        assert mp.run_dir == Path("/fake/run/dir")

    @patch("medpipe.pipeline.pipeline.MedpipeRegressorEvaluator")
    @patch("medpipe.pipeline.pipeline.MedpipeRegressorRunner")
    @patch("medpipe.pipeline.pipeline.MedpipeOrchestrator")
    def test_metrics_returns_evaluator_metrics(
        self, mock_orch_cls, mock_runner_cls, mock_eval_cls
    ):
        """Test that metrics delegates to evaluator.metrics."""
        mp = MedpipeRegressor(config=MagicMock())
        mp._evaluator.metrics = ["rmse", "mae", "crps"]

        assert mp.metrics == ["rmse", "mae", "crps"]

    @patch("medpipe.pipeline.pipeline.MedpipeRegressorEvaluator")
    @patch("medpipe.pipeline.pipeline.MedpipeRegressorRunner")
    @patch("medpipe.pipeline.pipeline.MedpipeOrchestrator")
    def test_data_split_returns_orchestrator_splits(
        self, mock_orch_cls, mock_runner_cls, mock_eval_cls
    ):
        """Test that data_split delegates to orchestrator.splits."""
        mp = MedpipeRegressor(config=MagicMock())
        mp._orchestrator.splits = "fake_splits"

        assert mp.data_split == "fake_splits"


class TestMedpipeRegressorFit:
    """Unit tests for MedpipeRegressor.fit."""

    @patch("medpipe.pipeline.pipeline.MedpipeRegressorEvaluator")
    @patch("medpipe.pipeline.pipeline.MedpipeRegressorRunner")
    @patch("medpipe.pipeline.pipeline.MedpipeOrchestrator")
    def test_fit_delegates_to_runner_run(
        self, mock_orch_cls, mock_runner_cls, mock_eval_cls
    ):
        """Test that fit delegates to runner.run with the expected arguments."""
        mp = MedpipeRegressor(config=MagicMock())
        X_train = pd.DataFrame({"A": [1, 2]})
        y_train = pd.DataFrame({"LOS_DAYS": [3.0, 5.0]})
        mp._runner.run.return_value = {"LOS_DAYS": "fitted_model"}

        result = mp.fit(X_train=X_train, y_train=y_train, groups_train=None)

        mp._runner.run.assert_called_once_with(
            X_train=X_train,
            y_train_df=y_train,
            X_recal=None,
            y_recal_df=None,
            groups_train=None,
        )
        assert result == {"LOS_DAYS": "fitted_model"}


class TestMedpipeRegressorRun:
    """Unit tests verifying full workflow sequence in MedpipeRegressor.run."""

    @patch("medpipe.pipeline.pipeline.MedpipeRegressorEvaluator")
    @patch("medpipe.pipeline.pipeline.MedpipeRegressorRunner")
    @patch("medpipe.pipeline.pipeline.MedpipeOrchestrator")
    def test_run_execution_flow_never_generates_plots(
        self, mock_orch_cls, mock_runner_cls, mock_eval_cls
    ):
        """Verify run executes data prep, fit, and test evaluation, and
        never attempts any plotting regardless of run_mode."""
        mp = MedpipeRegressor(config=MagicMock())
        mp.mp_config.meta.run_mode = "audit"
        mp.mp_config.data.kwargs = {"extra_arg": 0.2}
        mp._orchestrator.config.data.outcomes = ["LOS_DAYS"]
        mp._orchestrator.get_subgroup_specs.return_value = {"site": "site"}
        mp._orchestrator.fairness_splits = None

        X_tr, y_tr = pd.DataFrame({"A": [1, 2]}), pd.DataFrame({"LOS_DAYS": [3.0, 5.0]})
        X_te, y_te = pd.DataFrame({"A": [3]}), pd.DataFrame({"LOS_DAYS": [4.0]})
        mp._orchestrator.prepare_data.return_value = (
            X_tr,
            y_tr,
            None,
            None,
            X_te,
            y_te,
            None,
        )

        mp.fit = MagicMock(return_value={"LOS_DAYS": "fitted_model"})
        mp.evaluate = MagicMock(return_value={"overall": {"rmse": 1.2}})

        results = mp.run(groups_train=None)

        mp._orchestrator.prepare_data.assert_called_once_with(extra_arg=0.2)
        mp.fit.assert_called_once_with(
            X_train=X_tr,
            y_train=y_tr,
            X_recal=None,
            y_recal=None,
            groups_train=None,
        )
        mp.evaluate.assert_called_once_with(
            X=X_te,
            y=y_te["LOS_DAYS"].to_numpy(),
            outcome="LOS_DAYS",
            subgroup_specs={"site": "site"},
            fairness_data=None,
            save_artifacts=True,
        )

        assert results["fitted_models"] == {"LOS_DAYS": "fitted_model"}
        assert results["evaluations"] == {"LOS_DAYS": {"overall": {"rmse": 1.2}}}
        assert "plots" not in results

    @patch("medpipe.pipeline.pipeline.MedpipeRegressorEvaluator")
    @patch("medpipe.pipeline.pipeline.MedpipeRegressorRunner")
    @patch("medpipe.pipeline.pipeline.MedpipeOrchestrator")
    def test_run_logs_fit_and_total_duration_at_debug_level(
        self, mock_orch_cls, mock_runner_cls, mock_eval_cls
    ):
        """Verify run logs the model fitting duration and the total run
        duration at the debug log level."""
        mp = MedpipeRegressor(config=MagicMock())
        mp.mp_config.meta.run_mode = "fast"
        mp.mp_config.data.kwargs = {}
        mp._orchestrator.config.data.outcomes = ["LOS_DAYS"]
        mp._orchestrator.get_subgroup_specs.return_value = {}

        X_tr, y_tr = pd.DataFrame({"A": [1, 2]}), pd.DataFrame({"LOS_DAYS": [3.0, 5.0]})
        X_te, y_te = pd.DataFrame({"A": [3]}), pd.DataFrame({"LOS_DAYS": [4.0]})
        mp._orchestrator.prepare_data.return_value = (
            X_tr,
            y_tr,
            None,
            None,
            X_te,
            y_te,
            None,
        )

        mp.fit = MagicMock(return_value={"LOS_DAYS": "fitted_model"})
        mp.evaluate = MagicMock(return_value={"overall": {"rmse": 1.2}})
        mp._logger = MagicMock()

        mp.run(groups_train=None)

        debug_messages = [call.args[0] for call in mp._logger.debug.call_args_list]
        assert any(
            "Model fitting completed in" in msg and "seconds" in msg
            for msg in debug_messages
        )
        assert any(
            "Full pipeline run completed in" in msg and "seconds" in msg
            for msg in debug_messages
        )


class TestMedpipeRegressorLoad:
    """Test suite for the MedpipeRegressor.load class factory method."""

    @pytest.fixture
    def valid_config_dict(self, tmp_path: Path) -> dict:
        """Provides a minimal valid raw regressor configuration dictionary."""
        return {
            "meta": {
                "project_name": "demo_regressor_project",
                "run_mode": "fast",
                "verbose": "compact",
            },
            "data": {
                "path": str(tmp_path / "data.csv"),
                "predictors": ["AGE", "SEX"],
                "outcomes": ["LOS_DAYS"],
            },
            "default_model": {
                "algorithm": "OrdBoostRegressor",
                "hyperparameters": {},
            },
            "workflow": {
                "validation": {
                    "test_split": {"strategy": "random", "test_size": 0.2},
                },
                "evaluation": {
                    "metrics": {"metrics": ["rmse"]},
                },
            },
        }

    def test_load_missing_config_raises_file_not_found(self, tmp_path: Path) -> None:
        """Test that FileNotFoundError is raised if resolved_config.json is missing."""
        empty_run_dir = tmp_path / "empty_run"
        empty_run_dir.mkdir()

        with pytest.raises(
            FileNotFoundError,
            match="Cannot load MedpipeRegressor instance: Configuration JSON missing",
        ):
            MedpipeRegressor.load(empty_run_dir)

    def test_load_successful_without_models_directory(
        self, tmp_path: Path, valid_config_dict: dict
    ) -> None:
        """Test successful reconstruction of MedpipeRegressor when no models
        directory is present."""
        run_dir = tmp_path / "run_2026_09_13"
        config_dir = run_dir / "env"
        config_dir.mkdir(parents=True)

        config_path = config_dir / "resolved_config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(valid_config_dict, f)

        pipe = MedpipeRegressor.load(run_dir)

        assert isinstance(pipe, MedpipeRegressor)
        assert pipe.run_dir == run_dir / "eval"
        assert pipe.mp_config.meta.project_name == "demo_regressor_project"

    def test_load_successful_with_fitted_models(
        self, tmp_path: Path, valid_config_dict: dict
    ) -> None:
        """Test loading and restoring serialized fitted models into runner."""
        run_dir = tmp_path / "run_2026_09_13"
        models_dir = run_dir / "models"
        config_dir = run_dir / "env"
        models_dir.mkdir(parents=True)
        config_dir.mkdir(parents=True)

        config_path = config_dir / "resolved_config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(valid_config_dict, f)

        mock_fitted_models = {"LOS_DAYS": "fitted_model_placeholder"}
        model_artifact = models_dir / "demo_regressor_project_fitted.joblib"
        joblib.dump(mock_fitted_models, model_artifact)

        pipe = MedpipeRegressor.load(str(run_dir))

        assert isinstance(pipe, MedpipeRegressor)
        assert pipe.run_dir == run_dir / "eval"
        assert pipe.models == mock_fitted_models


class TestMedpipeRegressorStressIntegration:
    """End-to-end integration test executing MedpipeRegressor against a
    real, small synthetic dataset with a real OrdBoostRegressor - no mocking
    of the orchestrator/runner/evaluator stack."""

    def test_full_run_with_real_ordboost_regressor(self, tmp_path: Path) -> None:
        """Test that a full run() completes end-to-end: config parsing,
        data splitting, real OrdBoostRegressor training with CRPS-scored
        CV, and held-out evaluation with rmse/mae/crps."""
        rng = np.random.default_rng(42)
        n = 120
        df = pd.DataFrame(
            {
                "feature1": rng.standard_normal(n),
                "feature2": rng.standard_normal(n),
                "LOS_DAYS": np.exp(rng.standard_normal(n) * 0.5)
                + rng.normal(0.0, 0.5, n),
            }
        )
        data_path = tmp_path / "data.csv"
        df.to_csv(data_path, index=False)

        config_path = tmp_path / "config.toml"
        config_path.write_text(f"""
[meta]
project_name = "stress_test"
run_mode = "cv"

[data]
path = "{data_path}"
predictors = ["feature1", "feature2"]
outcomes = ["LOS_DAYS"]

[default_model]
algorithm = "OrdBoostRegressor"

[default_model.hyperparameters]
n_bins = 10
mapper = "quantile"
learning_rate = 0.1
max_iter = 15
random_state = 42

[workflow.validation.test_split]
strategy = "random"
test_size = 0.2

[workflow.validation.cross_validation]
strategy = "random"
n_splits = 3
grid_search = false

[workflow.evaluation.metrics]
metrics = ["rmse", "crps"]
""")

        mp = MedpipeRegressor(
            config=str(config_path), base_artifact_dir=str(tmp_path / "artifacts")
        )
        results = mp.run()

        assert mp.is_fitted
        assert "LOS_DAYS" in results["fitted_models"]

        overall = results["evaluations"]["LOS_DAYS"]["overall"]
        assert np.isfinite(overall["rmse"]["point_estimate"])
        assert np.isfinite(overall["crps"]["point_estimate"])
