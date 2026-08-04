"""
Unit tests for medpipe.evaluator.MedpipeEvaluator.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from medpipe.pipeline.evaluator import MedpipeEvaluator

# --- Fixtures ---


@pytest.fixture
def mock_orchestrator(tmp_path):
    """Fixture providing a mock MedpipeOrchestrator instance."""
    orchestrator = MagicMock()
    orchestrator.run_dir = tmp_path / "runs" / "v1"
    orchestrator.run_dir.mkdir(parents=True, exist_ok=True)

    mock_artifact_manager = MagicMock()
    mock_artifact_manager.save_json.return_value = (
        orchestrator.run_dir / "artifacts" / "test_evaluation_results.json"
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
    """Fixture providing a mock MedpipeRunner containing fitted models."""
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
        index=[101, 102, 103, 104],
    )
    y = pd.Series([0, 1, 1, 0], index=[101, 102, 103, 104], name="target")
    return X, y


# --- Test Classes per Function ---


class TestMedpipeEvaluatorInit:
    """Tests for MedpipeEvaluator.__init__."""

    def test_init_explicit_metrics(self, mock_orchestrator, mock_runner):
        """Test initialization when explicit metrics list is supplied."""
        metrics = ["roc_auc", "accuracy"]
        evaluator = MedpipeEvaluator(
            orchestrator=mock_orchestrator,
            runner=mock_runner,
            metrics=metrics,
            n_bootstraps=500,
            ci_level=0.90,
            random_state=42,
        )

        assert evaluator.metrics == metrics
        assert evaluator.n_bootstraps == 500
        assert evaluator.ci_level == 0.90
        assert evaluator.random_state == 42
        assert evaluator.fitted_models == mock_runner.fitted_models

    def test_init_metrics_from_orchestrator_config(
        self, mock_orchestrator, mock_runner
    ):
        """Test retrieving metrics from nested orchestrator config when metrics arg is None."""
        config_metrics = ["brier_score", "roc_auc"]

        # Build nested mock config matching evaluator path check
        mock_config = MagicMock()
        mock_config.workflow.evaluation.metrics.metrics = config_metrics
        mock_orchestrator.config = mock_config

        evaluator = MedpipeEvaluator(
            orchestrator=mock_orchestrator, runner=mock_runner, metrics=None
        )
        assert evaluator.metrics == config_metrics

    def test_init_fallback_default_metrics(self, mock_orchestrator, mock_runner):
        """Test fallback default metrics when no metrics supplied or configured in orchestrator."""
        del mock_orchestrator.config  # Ensure no config attribute exists

        evaluator = MedpipeEvaluator(
            orchestrator=mock_orchestrator, runner=mock_runner, metrics=None
        )
        assert evaluator.metrics == ["accuracy", "roc_auc", "brier_score"]


class TestMedpipeEvaluatorGetModel:
    """Tests for MedpipeEvaluator._get_model."""

    def test_get_model_explicit_instance(
        self, mock_orchestrator, mock_runner, mock_model
    ):
        """Test resolving model when an explicit model instance is passed."""
        evaluator = MedpipeEvaluator(mock_orchestrator, mock_runner)
        explicit_model = MagicMock()

        resolved = evaluator._get_model(model=explicit_model, outcome="ignored")
        assert resolved == explicit_model

    def test_get_model_by_outcome_key_success(
        self, mock_orchestrator, mock_runner, mock_model
    ):
        """Test resolving model via outcome key lookup in runner.fitted_models."""
        evaluator = MedpipeEvaluator(mock_orchestrator, mock_runner)

        resolved = evaluator._get_model(outcome="MORTALITY_30D")
        assert resolved == mock_model

    def test_get_model_by_outcome_key_error(self, mock_orchestrator, mock_runner):
        """Test KeyError raised when specified outcome key is absent in fitted_models."""
        evaluator = MedpipeEvaluator(mock_orchestrator, mock_runner)

        with pytest.raises(KeyError, match="Outcome 'non_existent' not found"):
            evaluator._get_model(outcome="non_existent")

    def test_get_model_single_fitted_model_implicit(
        self, mock_orchestrator, mock_runner, mock_model
    ):
        """Test resolving single fitted model implicitly when outcome and model are None."""
        evaluator = MedpipeEvaluator(mock_orchestrator, mock_runner)

        resolved = evaluator._get_model()
        assert resolved == mock_model

    def test_get_model_multiple_fitted_models_ambiguous_value_error(
        self, mock_orchestrator, mock_runner
    ):
        """Test ValueError raised when multiple fitted models exist and choice is ambiguous."""
        mock_runner.fitted_models = {
            "MORTALITY_30D": MagicMock(),
            "mortality_90d": MagicMock(),
        }
        evaluator = MedpipeEvaluator(mock_orchestrator, mock_runner)

        with pytest.raises(ValueError, match="Multiple models found"):
            evaluator._get_model()


class TestMedpipeEvaluatorPredict:
    """Tests for MedpipeEvaluator.predict."""

    def test_predict_success(
        self, mock_orchestrator, mock_runner, mock_model, sample_data
    ):
        """Test predict method success returning ndarray."""
        X, _ = sample_data
        evaluator = MedpipeEvaluator(mock_orchestrator, mock_runner)

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
        evaluator = MedpipeEvaluator(mock_orchestrator, mock_runner)

        with pytest.raises(AttributeError, match="model does not implement 'predict'"):
            evaluator.predict(X, model=bad_model)


class TestMedpipeEvaluatorPredictProba:
    """Tests for MedpipeEvaluator.predict_proba."""

    def test_predict_proba_success(
        self, mock_orchestrator, mock_runner, mock_model, sample_data
    ):
        """Test predict_proba method success returning ndarray."""
        X, _ = sample_data
        evaluator = MedpipeEvaluator(mock_orchestrator, mock_runner)

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
        evaluator = MedpipeEvaluator(mock_orchestrator, mock_runner)

        with pytest.raises(
            AttributeError, match="model does not implement 'predict_proba'"
        ):
            evaluator.predict_proba(X, model=bad_model)


class TestMedpipeEvaluatorDecisionFunction:
    """Tests for MedpipeEvaluator.decision_function."""

    def test_decision_function_success(
        self, mock_orchestrator, mock_runner, mock_model, sample_data
    ):
        """Test decision_function method success returning ndarray."""
        X, _ = sample_data
        evaluator = MedpipeEvaluator(mock_orchestrator, mock_runner)

        scores = evaluator.decision_function(X, outcome="MORTALITY_30D")

        assert isinstance(scores, np.ndarray)
        np.testing.assert_array_equal(scores, [-1.5, 2.1, 1.2, -0.8])
        mock_model.decision_function.assert_called_once_with(X)

    def test_decision_function_missing_method_attribute_error(
        self, mock_orchestrator, mock_runner, sample_data
    ):
        """Test AttributeError raised when target model lacks decision_function method."""
        X, _ = sample_data
        bad_model = object()
        evaluator = MedpipeEvaluator(mock_orchestrator, mock_runner)

        with pytest.raises(
            AttributeError, match="model does not implement 'decision_function'"
        ):
            evaluator.decision_function(X, model=bad_model)


class TestMedpipeEvaluatorExtractSubgroups:
    """Tests for MedpipeEvaluator.extract_subgroups."""

    def test_extract_subgroups_string_spec_success(
        self, mock_orchestrator, mock_runner, sample_data
    ):
        """Test subgroup extraction using column string grouping."""
        X, _ = sample_data
        evaluator = MedpipeEvaluator(mock_orchestrator, mock_runner)

        specs = {"sex_group": "sex"}
        subgroups = evaluator.extract_subgroups(X, specs)

        assert "sex_group" in subgroups
        assert set(subgroups["sex_group"].keys()) == {"M", "F"}
        pd.testing.assert_index_equal(subgroups["sex_group"]["M"], pd.Index([101, 103]))
        pd.testing.assert_index_equal(subgroups["sex_group"]["F"], pd.Index([102, 104]))

    def test_extract_subgroups_callable_spec_success(
        self, mock_orchestrator, mock_runner, sample_data
    ):
        """Test subgroup extraction using predicate callable grouping."""
        X, _ = sample_data
        evaluator = MedpipeEvaluator(mock_orchestrator, mock_runner)

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

    def test_extract_subgroups_missing_column_key_error(
        self, mock_orchestrator, mock_runner, sample_data
    ):
        """Test KeyError raised when column string spec is missing from DataFrame X."""
        X, _ = sample_data
        evaluator = MedpipeEvaluator(mock_orchestrator, mock_runner)

        specs = {"invalid": "non_existent_col"}
        with pytest.raises(KeyError, match="Column 'non_existent_col' not found"):
            evaluator.extract_subgroups(X, specs)

    def test_extract_subgroups_invalid_spec_type_error(
        self, mock_orchestrator, mock_runner, sample_data
    ):
        """Test TypeError raised when specification is neither string nor callable."""
        X, _ = sample_data
        evaluator = MedpipeEvaluator(mock_orchestrator, mock_runner)

        specs = {"invalid_spec": 12345}  # Integer spec
        with pytest.raises(TypeError, match="must be a column name string or callable"):
            evaluator.extract_subgroups(X, specs)


class TestMedpipeEvaluatorEvaluateSlice:
    """Tests for MedpipeEvaluator._evaluate_slice."""

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

        evaluator = MedpipeEvaluator(mock_orchestrator, mock_runner)
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

        evaluator = MedpipeEvaluator(mock_orchestrator, mock_runner)
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

        evaluator = MedpipeEvaluator(mock_orchestrator, mock_runner)
        results = evaluator._evaluate_slice(
            np.array([0]), np.array([0.1]), metrics=["roc_auc"]
        )

        assert "roc_auc" in results
        assert np.isnan(results["roc_auc"]["point_estimate"])
        assert np.isnan(results["roc_auc"]["ci_lower"])
        assert np.isnan(results["roc_auc"]["ci_upper"])


class TestMedpipeEvaluatorEvaluate:
    """Tests for MedpipeEvaluator.evaluate."""

    @patch.object(MedpipeEvaluator, "_evaluate_slice")
    @patch.object(MedpipeEvaluator, "_save_evaluation_artifacts")
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

        evaluator = MedpipeEvaluator(mock_orchestrator, mock_runner)
        res = evaluator.evaluate(X, y, outcome="MORTALITY_30D", save_artifacts=False)

        assert res["outcome"] == "MORTALITY_30D"
        assert "overall" in res
        assert "subgroups" not in res
        mock_model.predict_proba.assert_called_once()
        mock_save.assert_not_called()

    @patch.object(MedpipeEvaluator, "_evaluate_slice")
    def test_evaluate_fallback_to_decision_function(
        self, mock_eval_slice, mock_orchestrator, mock_runner, sample_data
    ):
        """Test evaluate falls back to decision_function when predict_proba is absent."""
        X, y = sample_data
        df_model = MagicMock(spec=["decision_function"])
        df_model.decision_function.return_value = np.array([-1.0, 1.0, 1.0, -1.0])

        mock_eval_slice.return_value = {}
        evaluator = MedpipeEvaluator(mock_orchestrator, mock_runner)

        res = evaluator.evaluate(
            X, y, model=df_model, outcome="custom", save_artifacts=False
        )

        assert res["outcome"] == "custom"
        df_model.decision_function.assert_called_once_with(X)

    @patch.object(MedpipeEvaluator, "_evaluate_slice")
    def test_evaluate_fallback_to_predict(
        self, mock_eval_slice, mock_orchestrator, mock_runner, sample_data
    ):
        """Test evaluate falls back to predict when both predict_proba
        and decision_function are absent."""
        X, y = sample_data
        predict_model = MagicMock(spec=["predict"])
        predict_model.predict.return_value = np.array([0, 1, 1, 0])

        mock_eval_slice.return_value = {}
        evaluator = MedpipeEvaluator(mock_orchestrator, mock_runner)

        res = evaluator.evaluate(
            X, y, model=predict_model, outcome="custom", save_artifacts=False
        )

        assert res["outcome"] == "custom"
        predict_model.predict.assert_called_once_with(X)

    @patch.object(MedpipeEvaluator, "_evaluate_slice")
    @patch.object(MedpipeEvaluator, "_save_evaluation_artifacts")
    def test_evaluate_with_subgroups_and_empty_group_handling(
        self,
        mock_save,
        mock_eval_slice,
        mock_orchestrator,
        mock_runner,
        sample_data,
    ):
        """Test subgroup evaluation handling, including skipping empty subgroup slices."""
        X, y = sample_data
        mock_eval_slice.return_value = {"accuracy": {"point_estimate": 0.8}}

        evaluator = MedpipeEvaluator(mock_orchestrator, mock_runner)

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

        assert "subgroups" in results
        assert "sex" in results["subgroups"]
        assert "M" in results["subgroups"]["sex"]
        assert "F" in results["subgroups"]["sex"]

        # Centenarians true group is empty and should be skipped
        assert "true" not in results["subgroups"]["centenarians"]
        assert "false" in results["subgroups"]["centenarians"]

        mock_save.assert_called_once_with(results, outcome="MORTALITY_30D")


class TestMedpipeEvaluatorSaveEvaluationArtifacts:
    """Tests for MedpipeEvaluator._save_evaluation_artifacts."""

    def test_save_evaluation_artifacts_success(self, mock_orchestrator, mock_runner):
        """Test persisting evaluation results to disk via ArtifactManager."""
        evaluator = MedpipeEvaluator(mock_orchestrator, mock_runner)
        results = {"outcome": "MORTALITY_30D", "overall": {}}

        saved_path = evaluator._save_evaluation_artifacts(
            results, outcome="MORTALITY_30D"
        )

        expected_artifacts_dir = mock_orchestrator.run_dir / "artifacts"
        expected_filename = "MORTALITY_30D_evaluation_results.json"

        mock_orchestrator.artifact_manager.save_json.assert_called_once_with(
            results, expected_artifacts_dir, expected_filename
        )
        assert saved_path == (expected_artifacts_dir / "test_evaluation_results.json")
