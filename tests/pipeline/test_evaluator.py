"""
Unit tests for medpipe.evaluator.MedpipeClassifierEvaluator.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from medpipe.pipeline.evaluator import (
    BaseEvaluator,
    MedpipeClassifierEvaluator,
    MedpipeRegressorEvaluator,
)

# --- Fixtures ---


@pytest.fixture
def mock_orchestrator(tmp_path):
    """Fixture providing a mock MedpipeOrchestrator instance."""
    orchestrator = MagicMock()
    orchestrator.run_dir = tmp_path / "runs" / "v1"
    orchestrator.run_dir.mkdir(parents=True, exist_ok=True)

    orchestrator.config = MagicMock()
    orchestrator.config.workflow.random_state = 50
    orchestrator.config.workflow.evaluation.metrics.metrics = ["accuracy", "ap"]
    orchestrator.config.workflow.evaluation.metrics.n_bootstraps = 500
    orchestrator.config.workflow.evaluation.metrics.ci_level = 0.95

    mock_artifact_manager = MagicMock()
    mock_artifact_manager.save_json.return_value = (
        orchestrator.run_dir / "artifacts/results" / "test_evaluation_results.json"
    )
    orchestrator.artifact_manager = mock_artifact_manager

    return orchestrator


@pytest.fixture
def mock_model():
    """Fixture providing a mock estimator implementing standard scikit-learn methods."""
    model = MagicMock()
    model.predict.return_value = np.array([0, 1, 1, 0])
    model.predict_proba.return_value = np.array([0.1, 0.9, 0.8, 0.2])
    model.decision_function.return_value = np.array([-1.5, 2.1, 1.2, -0.8])
    return model


@pytest.fixture
def mock_runner(mock_model):
    """Fixture providing a mock MedpipeClassifierRunner containing fitted models."""
    runner = MagicMock()
    runner.fitted_models = {"MORTALITY_30D": mock_model}
    return runner


@pytest.fixture
def sample_data():
    """Fixture providing feature DataFrame X and label Series y."""
    X = pd.DataFrame(
        {
            "age": [65, 45, 72, 50],
            "sex": ["M", "F", "M", "F"],
            "bmi": [28.5, 22.0, 31.2, 24.1],
        },
        index=pd.Index([101, 102, 103, 104]),
    )
    y = pd.Series([0, 1, 1, 0], index=[101, 102, 103, 104], name="target")
    return X, y


# --- Test Classes per Function ---


class TestMedpipeEvaluatorInit:
    """Tests for MedpipeClassifierEvaluator.__init__."""

    def test_init_success(self, mock_orchestrator, mock_runner):
        """Test initialization when explicit metrics list is supplied."""
        evaluator = MedpipeClassifierEvaluator(
            orchestrator=mock_orchestrator,
            runner=mock_runner,
        )

        assert evaluator.n_bootstraps == 500
        assert evaluator.ci_level == 0.95
        assert evaluator.random_state == 50
        assert evaluator.metrics == ["accuracy", "ap"]
        assert evaluator.fitted_models == mock_runner.fitted_models


class TestMedpipeEvaluatorGetModel:
    """Tests for MedpipeClassifierEvaluator._get_model."""

    def test_get_model_explicit_instance(
        self, mock_orchestrator, mock_runner, mock_model
    ):
        """Test resolving model when an explicit model instance is passed."""
        evaluator = MedpipeClassifierEvaluator(mock_orchestrator, mock_runner)
        explicit_model = MagicMock()

        resolved = evaluator._get_model(model=explicit_model, outcome="ignored")
        assert resolved == explicit_model

    def test_get_model_by_outcome_key_success(
        self, mock_orchestrator, mock_runner, mock_model
    ):
        """Test resolving model via outcome key lookup in runner.fitted_models."""
        evaluator = MedpipeClassifierEvaluator(mock_orchestrator, mock_runner)

        resolved = evaluator._get_model(outcome="MORTALITY_30D")
        assert resolved == mock_model

    def test_get_model_by_outcome_key_error(self, mock_orchestrator, mock_runner):
        """Test KeyError raised when specified outcome key is absent in
        fitted_models."""
        evaluator = MedpipeClassifierEvaluator(mock_orchestrator, mock_runner)

        with pytest.raises(KeyError, match="Outcome 'non_existent' not found"):
            evaluator._get_model(outcome="non_existent")

    def test_get_model_single_fitted_model_implicit(
        self, mock_orchestrator, mock_runner, mock_model
    ):
        """Test resolving single fitted model implicitly when outcome and
        model are None."""
        evaluator = MedpipeClassifierEvaluator(mock_orchestrator, mock_runner)

        resolved = evaluator._get_model()
        assert resolved == mock_model

    def test_get_model_multiple_fitted_models_ambiguous_value_error(
        self, mock_orchestrator, mock_runner
    ):
        """Test ValueError raised when multiple fitted models exist and
        choice is ambiguous."""
        mock_runner.fitted_models = {
            "MORTALITY_30D": MagicMock(),
            "mortality_90d": MagicMock(),
        }
        evaluator = MedpipeClassifierEvaluator(mock_orchestrator, mock_runner)

        with pytest.raises(ValueError, match="Multiple models found"):
            evaluator._get_model()


class TestMedpipeEvaluatorPredict:
    """Tests for MedpipeClassifierEvaluator.predict."""

    def test_predict_success(
        self, mock_orchestrator, mock_runner, mock_model, sample_data
    ):
        """Test predict method success returning ndarray."""
        X, _ = sample_data
        evaluator = MedpipeClassifierEvaluator(mock_orchestrator, mock_runner)

        preds = evaluator.predict(X, outcome="MORTALITY_30D")

        assert isinstance(preds, np.ndarray)
        np.testing.assert_array_equal(preds, [0, 1, 1, 0])
        mock_model.predict.assert_called_once_with(X)

    def test_predict_missing_method_attribute_error(
        self, mock_orchestrator, mock_runner, sample_data
    ):
        """Test AttributeError raised when target model lacks predict method."""
        X, _ = sample_data
        bad_model = object()  # Lacks predict method
        evaluator = MedpipeClassifierEvaluator(mock_orchestrator, mock_runner)

        with pytest.raises(AttributeError, match="model does not implement 'predict'"):
            evaluator.predict(X, model=bad_model)


class TestMedpipeEvaluatorPredictProba:
    """Tests for MedpipeClassifierEvaluator.predict_proba."""

    def test_predict_proba_success(
        self, mock_orchestrator, mock_runner, mock_model, sample_data
    ):
        """Test predict_proba method success returning ndarray."""
        X, _ = sample_data
        evaluator = MedpipeClassifierEvaluator(mock_orchestrator, mock_runner)

        probas = evaluator.predict_proba(X, outcome="MORTALITY_30D")

        assert isinstance(probas, np.ndarray)
        np.testing.assert_array_equal(probas, [0.1, 0.9, 0.8, 0.2])
        mock_model.predict_proba.assert_called_once_with(X)

    def test_predict_proba_missing_method_attribute_error(
        self, mock_orchestrator, mock_runner, sample_data
    ):
        """Test AttributeError raised when target model lacks predict_proba method."""
        X, _ = sample_data
        bad_model = object()
        evaluator = MedpipeClassifierEvaluator(mock_orchestrator, mock_runner)

        with pytest.raises(
            AttributeError, match="model does not implement 'predict_proba'"
        ):
            evaluator.predict_proba(X, model=bad_model)


class TestMedpipeEvaluatorDecisionFunction:
    """Tests for MedpipeClassifierEvaluator.decision_function."""

    def test_decision_function_success(
        self, mock_orchestrator, mock_runner, mock_model, sample_data
    ):
        """Test decision_function method success returning ndarray."""
        X, _ = sample_data
        evaluator = MedpipeClassifierEvaluator(mock_orchestrator, mock_runner)

        scores = evaluator.decision_function(X, outcome="MORTALITY_30D")

        assert isinstance(scores, np.ndarray)
        np.testing.assert_array_equal(scores, [-1.5, 2.1, 1.2, -0.8])
        mock_model.decision_function.assert_called_once_with(X)

    def test_decision_function_missing_method_attribute_error(
        self, mock_orchestrator, mock_runner, sample_data
    ):
        """Test AttributeError raised when target model lacks
        decision_function method."""
        X, _ = sample_data
        bad_model = object()
        evaluator = MedpipeClassifierEvaluator(mock_orchestrator, mock_runner)

        with pytest.raises(
            AttributeError, match="model does not implement 'decision_function'"
        ):
            evaluator.decision_function(X, model=bad_model)


class TestMedpipeEvaluatorExtractSubgroups:
    """Tests for MedpipeClassifierEvaluator.extract_subgroups."""

    def test_extract_subgroups_string_spec_success(
        self, mock_orchestrator, mock_runner, sample_data
    ):
        """Test subgroup extraction using column string categorical groupby."""
        X, _ = sample_data
        evaluator = MedpipeClassifierEvaluator(mock_orchestrator, mock_runner)

        specs = {"sex_group": "sex"}
        subgroups = evaluator.extract_subgroups(X, specs)  # type: ignore

        assert "sex_group" in subgroups
        assert set(subgroups["sex_group"].keys()) == {"M", "F"}
        pd.testing.assert_index_equal(subgroups["sex_group"]["M"], pd.Index([101, 103]))
        pd.testing.assert_index_equal(subgroups["sex_group"]["F"], pd.Index([102, 104]))

    def test_extract_subgroups_list_of_ranges_success(
        self, mock_orchestrator, mock_runner, sample_data
    ):
        """Test subgroup extraction using a list of numerical range bounds."""
        X, _ = sample_data
        evaluator = MedpipeClassifierEvaluator(mock_orchestrator, mock_runner)

        specs = {"age": [[18, 50], [51, 120]]}
        subgroups = evaluator.extract_subgroups(X, specs)  # type: ignore

        assert "age" in subgroups
        assert "[18, 50]" in subgroups["age"]
        assert "[51, 120]" in subgroups["age"]

        pd.testing.assert_index_equal(
            subgroups["age"]["[18, 50]"], pd.Index([102, 104])
        )
        pd.testing.assert_index_equal(
            subgroups["age"]["[51, 120]"], pd.Index([101, 103])
        )

    def test_extract_subgroups_callable_spec_success(
        self, mock_orchestrator, mock_runner, sample_data
    ):
        """Test subgroup extraction using predicate callable grouping."""
        X, _ = sample_data
        evaluator = MedpipeClassifierEvaluator(mock_orchestrator, mock_runner)

        specs = {"elderly": lambda df: df["age"] >= 65}
        subgroups = evaluator.extract_subgroups(X, specs)

        assert "elderly" in subgroups
        assert set(subgroups["elderly"].keys()) == {"true", "false"}
        pd.testing.assert_index_equal(
            subgroups["elderly"]["true"], pd.Index([101, 103])
        )
        pd.testing.assert_index_equal(
            subgroups["elderly"]["false"], pd.Index([102, 104])
        )

    def test_extract_subgroups_callable_spec_returns_non_series_mask(
        self, mock_orchestrator, mock_runner, sample_data
    ):
        """Test that a callable returning a plain numpy array (not a
        pd.Series) is wrapped correctly before indexing."""
        X, _ = sample_data
        evaluator = MedpipeClassifierEvaluator(mock_orchestrator, mock_runner)

        specs = {"elderly": lambda df: (df["age"] >= 65).to_numpy()}
        subgroups = evaluator.extract_subgroups(X, specs)

        assert set(subgroups["elderly"].keys()) == {"true", "false"}
        pd.testing.assert_index_equal(
            subgroups["elderly"]["true"], pd.Index([101, 103])
        )
        pd.testing.assert_index_equal(
            subgroups["elderly"]["false"], pd.Index([102, 104])
        )

    def test_extract_subgroups_list_of_discrete_values(
        self, mock_orchestrator, mock_runner
    ):
        """Test subgroup extraction using a list of discrete category
        values (not numeric range pairs) — each entry falls back to
        resolve_subgroup_mask's scalar-equality path, and the resulting key
        is the plain value itself rather than a "[min, max]" range label."""
        X = pd.DataFrame(
            {"ethnicity": ["Maori", "Pacific", "European", "Maori"]},
            index=pd.Index([101, 102, 103, 104]),
        )
        evaluator = MedpipeClassifierEvaluator(mock_orchestrator, mock_runner)

        specs = {"ethnicity": ["Maori", "Pacific"]}
        subgroups = evaluator.extract_subgroups(X, specs)  # type: ignore

        assert set(subgroups["ethnicity"].keys()) == {"Maori", "Pacific"}
        pd.testing.assert_index_equal(
            subgroups["ethnicity"]["Maori"], pd.Index([101, 104])
        )
        pd.testing.assert_index_equal(
            subgroups["ethnicity"]["Pacific"], pd.Index([102])
        )

    def test_extract_subgroups_fallback_scalar_spec(
        self, mock_orchestrator, mock_runner, sample_data
    ):
        """Test subgroup extraction falling back to resolve_subgroup_mask for
        non-column specs."""
        X, _ = sample_data
        evaluator = MedpipeClassifierEvaluator(mock_orchestrator, mock_runner)

        specs = {"sex": "M"}
        subgroups = evaluator.extract_subgroups(X, specs)  # type: ignore

        assert "sex" in subgroups
        assert "M" in subgroups["sex"]
        pd.testing.assert_index_equal(subgroups["sex"]["M"], pd.Index([101, 103]))


class TestMedpipeEvaluatorEvaluateSlice:
    """Tests for MedpipeClassifierEvaluator._evaluate_slice."""

    @patch("medpipe.pipeline.evaluator.bootstrap_confidence_intervals")
    def test_evaluate_slice_bootstrap_success(
        self, mock_bootstrap, mock_orchestrator, mock_runner
    ):
        """Test slice evaluation when bootstrap CI calculation succeeds."""
        expected_results = {
            "accuracy": {
                "point_estimate": 0.85,
                "ci_lower": 0.70,
                "ci_upper": 0.95,
            }
        }
        mock_bootstrap.return_value = expected_results

        evaluator = MedpipeClassifierEvaluator(mock_orchestrator, mock_runner)
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0.1, 0.8, 0.9, 0.2])

        results = evaluator._evaluate_slice(y_true, y_pred, metrics=["accuracy"])
        assert results == expected_results

    @patch("medpipe.pipeline.evaluator.compute_metrics")
    @patch("medpipe.pipeline.evaluator.bootstrap_confidence_intervals")
    def test_evaluate_slice_bootstrap_fallback_to_point_estimates(
        self, mock_bootstrap, mock_compute, mock_orchestrator, mock_runner
    ):
        """Test slice evaluation fallback to point estimates when bootstrap fails."""
        mock_bootstrap.side_effect = RuntimeError("Resampling failed")
        mock_compute.return_value = [0.85]

        evaluator = MedpipeClassifierEvaluator(mock_orchestrator, mock_runner)
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0.1, 0.8, 0.9, 0.2])

        results = evaluator._evaluate_slice(y_true, y_pred, metrics=["accuracy"])

        assert "accuracy" in results
        assert results["accuracy"]["point_estimate"] == 0.85
        assert np.isnan(results["accuracy"]["ci_lower"])
        assert np.isnan(results["accuracy"]["ci_upper"])

    @patch("medpipe.pipeline.evaluator.compute_metrics")
    @patch("medpipe.pipeline.evaluator.bootstrap_confidence_intervals")
    def test_evaluate_slice_double_failure_fallback_to_nans(
        self, mock_bootstrap, mock_compute, mock_orchestrator, mock_runner
    ):
        """Test slice evaluation complete fallback to NaNs when both
        bootstrap and point computation fail."""
        mock_bootstrap.side_effect = RuntimeError("Resampling failed")
        mock_compute.side_effect = ValueError("Calculation error")

        evaluator = MedpipeClassifierEvaluator(mock_orchestrator, mock_runner)
        results = evaluator._evaluate_slice(
            np.array([0]), np.array([0.1]), metrics=["roc_auc"]
        )

        assert "roc_auc" in results
        assert np.isnan(results["roc_auc"]["point_estimate"])
        assert np.isnan(results["roc_auc"]["ci_lower"])
        assert np.isnan(results["roc_auc"]["ci_upper"])


class TestMedpipeEvaluatorEvaluate:
    """Tests for MedpipeClassifierEvaluator.evaluate."""

    @patch.object(MedpipeClassifierEvaluator, "_evaluate_slice")
    @patch.object(MedpipeClassifierEvaluator, "_save_evaluation_artifacts")
    def test_evaluate_overall_only_proba_model(
        self,
        mock_save,
        mock_eval_slice,
        mock_orchestrator,
        mock_runner,
        mock_model,
        sample_data,
    ):
        """Test full evaluate call using predict_proba model without subgroups."""
        X, y = sample_data
        mock_eval_slice.return_value = {
            "accuracy": {"point_estimate": 1.0, "ci_lower": 1.0, "ci_upper": 1.0}
        }

        evaluator = MedpipeClassifierEvaluator(mock_orchestrator, mock_runner)
        res = evaluator.evaluate(X, y, outcome="MORTALITY_30D", save_artifacts=False)

        assert res["outcome"] == "MORTALITY_30D"
        assert "overall" in res
        assert "strata" not in res
        mock_model.predict_proba.assert_called_once()
        mock_save.assert_not_called()

    @patch.object(MedpipeClassifierEvaluator, "_evaluate_slice")
    def test_evaluate_fallback_to_decision_function(
        self, mock_eval_slice, mock_orchestrator, mock_runner, sample_data
    ):
        """Test evaluate falls back to decision_function when predict_proba is
        absent."""
        X, y = sample_data
        df_model = MagicMock(spec=["decision_function"])
        df_model.decision_function.return_value = np.array([-1.0, 1.0, 1.0, -1.0])

        mock_eval_slice.return_value = {}
        evaluator = MedpipeClassifierEvaluator(mock_orchestrator, mock_runner)

        res = evaluator.evaluate(
            X, y, model=df_model, outcome="custom", save_artifacts=False
        )

        assert res["outcome"] == "custom"
        df_model.decision_function.assert_called_once_with(X)

    @patch.object(MedpipeClassifierEvaluator, "_evaluate_slice")
    def test_evaluate_fallback_to_predict(
        self, mock_eval_slice, mock_orchestrator, mock_runner, sample_data
    ):
        """Test evaluate falls back to predict when both predict_proba
        and decision_function are absent."""
        X, y = sample_data
        predict_model = MagicMock(spec=["predict"])
        predict_model.predict.return_value = np.array([0, 1, 1, 0])

        mock_eval_slice.return_value = {}
        evaluator = MedpipeClassifierEvaluator(mock_orchestrator, mock_runner)

        res = evaluator.evaluate(
            X, y, model=predict_model, outcome="custom", save_artifacts=False
        )

        assert res["outcome"] == "custom"
        predict_model.predict.assert_called_once_with(X)

    @patch.object(MedpipeClassifierEvaluator, "_evaluate_slice")
    @patch.object(MedpipeClassifierEvaluator, "_save_evaluation_artifacts")
    def test_evaluate_with_subgroups_and_empty_group_handling(
        self,
        mock_save,
        mock_eval_slice,
        mock_orchestrator,
        mock_runner,
        sample_data,
    ):
        """Test subgroup evaluation handling, including skipping empty
        subgroup slices."""
        X, y = sample_data
        mock_eval_slice.return_value = {"accuracy": {"point_estimate": 0.8}}

        evaluator = MedpipeClassifierEvaluator(mock_orchestrator, mock_runner)

        subgroup_specs = {
            "sex": "sex",
            # Subgroup predicate that matches zero samples in sample_data (age > 100)
            "centenarians": lambda df: df["age"] > 100,
        }

        results = evaluator.evaluate(
            X,
            y,
            outcome="MORTALITY_30D",
            subgroup_specs=subgroup_specs,
            save_artifacts=True,
        )

        assert "strata" in results
        assert "sex" in results["strata"]
        assert "M" in results["strata"]["sex"]
        assert "F" in results["strata"]["sex"]

        # Centenarians true group is empty and should be skipped
        assert "true" not in results["strata"]["centenarians"]
        assert "false" in results["strata"]["centenarians"]

        mock_save.assert_called_once_with(results, outcome="MORTALITY_30D")


class TestMedpipeEvaluatorSaveEvaluationArtifacts:
    """Tests for MedpipeClassifierEvaluator._save_evaluation_artifacts."""

    def test_save_evaluation_artifacts_success(self, mock_orchestrator, mock_runner):
        """Test persisting evaluation results to disk via ArtifactManager."""
        evaluator = MedpipeClassifierEvaluator(mock_orchestrator, mock_runner)
        results = {"outcome": "MORTALITY_30D", "overall": {}}

        saved_path = evaluator._save_evaluation_artifacts(
            results, outcome="MORTALITY_30D"
        )

        expected_artifacts_dir = mock_orchestrator.run_dir / "results"
        expected_filename = "MORTALITY_30D_evaluation_results.json"

        mock_orchestrator.artifact_manager.save_json.assert_called_once_with(
            results, expected_artifacts_dir, expected_filename
        )

        assert saved_path == (
            mock_orchestrator.run_dir / "artifacts/results/test_evaluation_results.json"
        )


class TestRegressorEvaluator:
    """Tests for MedpipeRegressorEvaluator against continuous synthetic
    data, confirming the compute_metrics rounding fix (Phase 0) is
    correctly exercised end-to-end through evaluate()."""

    @pytest.fixture
    def mock_regressor_model(self):
        """Fixture providing a mock estimator with continuous point
        predictions (not integer-rounded probabilities)."""
        model = MagicMock()
        model.predict.return_value = np.array([2.5, 7.1, 4.3, 9.9])
        del model.predict_proba
        del model.decision_function
        return model

    @pytest.fixture
    def mock_regressor_runner(self, mock_regressor_model):
        """Fixture providing a mock MedpipeRegressorRunner containing a
        fitted model."""
        runner = MagicMock()
        runner.fitted_models = {"LOS_DAYS": mock_regressor_model}
        return runner

    @pytest.fixture
    def continuous_sample_data(self):
        """Fixture providing feature DataFrame X and a continuous target y."""
        X = pd.DataFrame(
            {"age": [65, 45, 72, 50]},
            index=pd.Index([101, 102, 103, 104]),
        )
        y = pd.Series([2.0, 7.0, 4.0, 10.0], index=[101, 102, 103, 104], name="los")
        return X, y

    def test_get_predictions_returns_raw_point_predictions(
        self, mock_orchestrator, mock_regressor_runner, mock_regressor_model
    ):
        """Test that _get_predictions delegates to predict() for point
        predictions, and does not call predict_dist when no distributional
        metric is requested."""
        evaluator = MedpipeRegressorEvaluator(mock_orchestrator, mock_regressor_runner)

        result = evaluator._get_predictions(
            X=pd.DataFrame(), target_model=mock_regressor_model, metrics=["rmse"]
        )

        np.testing.assert_array_equal(result.point, [2.5, 7.1, 4.3, 9.9])
        assert result.dist is None
        mock_regressor_model.predict_dist.assert_not_called()

    def test_get_predictions_includes_dist_when_crps_requested(
        self, mock_orchestrator, mock_regressor_runner, mock_regressor_model
    ):
        """Test that _get_predictions also calls predict_dist when a
        predict_dist-based metric (e.g. crps) is requested."""
        mock_regressor_model.predict_dist.return_value = "fake_dist"
        evaluator = MedpipeRegressorEvaluator(mock_orchestrator, mock_regressor_runner)

        X = pd.DataFrame({"a": [1, 2, 3, 4]})
        result = evaluator._get_predictions(
            X=X, target_model=mock_regressor_model, metrics=["rmse", "crps"]
        )

        np.testing.assert_array_equal(result.point, [2.5, 7.1, 4.3, 9.9])
        assert result.dist == "fake_dist"
        mock_regressor_model.predict_dist.assert_called_once_with(X)

    def test_evaluate_rmse_mae_not_rounded(
        self,
        mock_orchestrator,
        mock_regressor_runner,
        continuous_sample_data,
    ):
        """Test that evaluate() with rmse/mae uses unrounded continuous
        predictions end-to-end (regression test for the Phase 0 bug where
        compute_metrics rounded every non-predict_proba metric's input)."""
        mock_orchestrator.config.workflow.evaluation.metrics.metrics = [
            "rmse",
            "mae",
        ]
        evaluator = MedpipeRegressorEvaluator(mock_orchestrator, mock_regressor_runner)
        X, y = continuous_sample_data

        results = evaluator.evaluate(
            X, y, outcome="LOS_DAYS", metrics=["rmse", "mae"], save_artifacts=False
        )

        # Predictions [2.5, 7.1, 4.3, 9.9] vs truth [2.0, 7.0, 4.0, 10.0]:
        # errors are [0.5, 0.1, 0.3, -0.1] -> MAE = 0.25 exactly.
        # If predictions were rounded to [2, 7, 4, 10] first (the old bug),
        # MAE would be 0.5 instead.
        mae_point_estimate = results["overall"]["mae"]["point_estimate"]
        assert mae_point_estimate == pytest.approx(0.25)


class TestBaseEvaluatorHooks:
    """Unit tests for BaseEvaluator's default hook implementations."""

    def test_get_predictions_not_implemented(
        self, mock_orchestrator, mock_runner, mock_model
    ):
        """Test that the base class requires subclasses to resolve
        outcome-type-specific predictions."""
        evaluator = BaseEvaluator(mock_orchestrator, mock_runner)

        with pytest.raises(NotImplementedError):
            evaluator._get_predictions(
                X=pd.DataFrame(), target_model=mock_model, metrics=["accuracy"]
            )
