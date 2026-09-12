"""
Tests for MedpipeRunner._train_model_cv.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from medpipe.metrics.registry import MetricRegistry, MetricSpec
from medpipe.pipeline.runner import MedpipeRunner


class TestTrainModelCv:
    """Unit tests for MedpipeRunner._train_model_cv."""

    @patch("medpipe.pipeline.runner.MedpipeRunner._save_cv_results")
    @patch("medpipe.pipeline.runner.cross_validate")
    def test_train_model_cv_standard_cv(
        self, mock_cv, mock_save_cv_results, mock_orchestrator, dummy_data
    ):
        """Test _train_model_cv runs cross_validate and calls _save_cv_results."""
        runner = MedpipeRunner(orchestrator=mock_orchestrator)

        X_train, y_train, _, _ = dummy_data
        mock_pipeline = MagicMock(spec=Pipeline)
        cv_splitter = MagicMock()

        mock_cv.return_value = {
            "fit_time": [0.1],
            "test_accuracy": [0.85],
        }

        hyperparams = {"max_depth": 3, "n_estimators": 100}

        result = runner._train_model_cv(
            outcome="MORTALITY_30D",
            pipeline=mock_pipeline,
            hyperparams=hyperparams,
            X_train=X_train,
            y_train=y_train,
            groups_train=None,
            cv_splitter=cv_splitter,
        )

        mock_cv.assert_called_once()
        call_kwargs = mock_cv.call_args[1]
        assert call_kwargs["estimator"] == mock_pipeline
        assert call_kwargs["X"].equals(X_train)
        assert np.array_equal(call_kwargs["y"], y_train)
        assert call_kwargs["groups"] is None
        assert call_kwargs["cv"] == cv_splitter
        assert call_kwargs["n_jobs"] == 4
        assert isinstance(call_kwargs["scoring"], dict)
        assert "accuracy" in call_kwargs["scoring"]

        mock_save_cv_results.assert_called_once()
        saved_df = mock_save_cv_results.call_args[0][1]
        assert isinstance(saved_df, pd.DataFrame)
        assert "test_accuracy" in saved_df.columns

        mock_pipeline.fit.assert_called_once_with(X_train, y_train)
        assert result == mock_pipeline.fit.return_value

    @patch("medpipe.pipeline.runner.MedpipeRunner._save_cv_results")
    @patch("medpipe.pipeline.runner.cross_validate")
    def test_train_model_cv_with_groups_train(
        self, mock_cv, mock_save_cv_results, mock_orchestrator, dummy_data
    ):
        """Test _train_model_cv with a non-None groups_train array, which
        both forwards the groups to cross_validate and exercises the
        group-count debug logging branch."""
        mock_orchestrator.config.workflow.validation.cross_validation.strategy = (
            "group"
        )
        runner = MedpipeRunner(orchestrator=mock_orchestrator)

        X_train, y_train, _, _ = dummy_data
        mock_pipeline = MagicMock(spec=Pipeline)
        cv_splitter = MagicMock()
        groups_train = np.array(["A", "A", "B", "B", "C", "C"])

        mock_cv.return_value = {"fit_time": [0.1], "test_accuracy": [0.85]}

        runner._train_model_cv(
            outcome="MORTALITY_30D",
            pipeline=mock_pipeline,
            hyperparams={"max_depth": 3},
            X_train=X_train,
            y_train=y_train,
            groups_train=groups_train,
            cv_splitter=cv_splitter,
        )

        mock_cv.assert_called_once()
        call_kwargs = mock_cv.call_args[1]
        np.testing.assert_array_equal(call_kwargs["groups"], groups_train)

    @patch("medpipe.pipeline.runner.MedpipeRunner._save_cv_results")
    @patch("medpipe.pipeline.runner.GridSearchCV")
    def test_train_model_cv_grid_search(
        self, mock_grid_search, mock_save_cv_results, mock_orchestrator, dummy_data
    ):
        """Test _train_model_cv triggers GridSearchCV and calls
        _save_cv_results when strategy is 'search'."""
        mock_orchestrator.config.workflow.validation.cross_validation.strategy = (
            "random"
        )

        mock_orchestrator.config.workflow.validation.cross_validation.grid_search = True
        runner = MedpipeRunner(orchestrator=mock_orchestrator)
        runner.orchestrator.config.workflow.evaluation.metrics.metrics = [
            "accuracy",
            "roc_auc",
        ]

        X_train, y_train, _, _ = dummy_data
        mock_pipeline = MagicMock(spec=Pipeline)
        cv_splitter = MagicMock()

        hyperparams = {"max_depth": [3, 5], "n_estimators": 10}

        mock_search_instance = MagicMock()
        mock_grid_search.return_value = mock_search_instance
        mock_search_instance.best_estimator_ = "best_model"
        mock_search_instance.best_score_ = 0.9102
        mock_search_instance.cv_results_ = {
            "params": [{"classifier__max_depth": 3}],
            "mean_test_accuracy": [0.85],
        }

        result = runner._train_model_cv(
            outcome="MORTALITY_30D",
            pipeline=mock_pipeline,
            hyperparams=hyperparams,
            X_train=X_train,
            y_train=y_train,
            groups_train=None,
            cv_splitter=cv_splitter,
        )

        expected_params = {
            "classifier__max_depth": [3, 5],
            "classifier__n_estimators": [10],
        }

        mock_grid_search.assert_called_once()
        call_kwargs = mock_grid_search.call_args[1]
        assert call_kwargs["estimator"] == mock_pipeline
        assert call_kwargs["param_grid"] == expected_params
        assert call_kwargs["cv"] == cv_splitter
        assert call_kwargs["n_jobs"] == 4
        assert isinstance(call_kwargs["scoring"], dict)
        assert call_kwargs["refit"] == "accuracy"

        mock_save_cv_results.assert_called_once()
        saved_df = mock_save_cv_results.call_args[0][1]
        assert isinstance(saved_df, pd.DataFrame)

        mock_search_instance.fit.assert_called_once_with(X_train, y_train, groups=None)
        assert result == "best_model"

    @patch("medpipe.pipeline.runner.MedpipeRunner._save_cv_results")
    @patch("medpipe.pipeline.runner.cross_validate")
    def test_train_model_cv_standard_cv_with_custom_metrics(
        self, mock_cv, mock_save_cv_results, mock_orchestrator, dummy_data
    ):
        """Test _train_model_cv converts built-in & custom metrics (e.g. ici)
        to scorers."""
        mock_orchestrator.config.workflow.evaluation.metrics.metrics = [
            "accuracy",
            "ici",
        ]
        mock_orchestrator.config.workflow.cross_validation.grid_search = False
        runner = MedpipeRunner(orchestrator=mock_orchestrator)

        X_train, y_train, _, _ = dummy_data
        mock_pipeline = MagicMock(spec=Pipeline)
        cv_splitter = MagicMock()

        hyperparams = {"max_depth": 3, "n_estimators": 100}

        result = runner._train_model_cv(
            outcome="MORTALITY_30D",
            pipeline=mock_pipeline,
            hyperparams=hyperparams,
            X_train=X_train,
            y_train=y_train,
            groups_train=None,
            cv_splitter=cv_splitter,
        )

        mock_cv.assert_called_once()
        passed_scoring = mock_cv.call_args[1]["scoring"]

        assert isinstance(passed_scoring, dict)
        assert "accuracy" in passed_scoring
        assert "ici" in passed_scoring
        assert callable(passed_scoring["accuracy"])
        assert callable(passed_scoring["ici"])

        mock_save_cv_results.assert_called_once()
        mock_pipeline.fit.assert_called_once_with(X_train, y_train)
        assert result == mock_pipeline.fit.return_value

    @patch("medpipe.pipeline.runner.MedpipeRunner._save_cv_results")
    @patch("medpipe.pipeline.runner.GridSearchCV")
    def test_train_model_cv_grid_search_with_custom_registry_metric(
        self, mock_grid_search, mock_save_cv_results, mock_orchestrator, dummy_data
    ):
        """Test _train_model_cv handles custom metrics registered via
        MetricRegistry in GridSearchCV."""
        custom_spec = MetricSpec(
            name="dummy_custom_score",
            func=lambda y, y_pred: 0.95,
            response_method="predict",
            display_name="Dummy Score",
        )
        MetricRegistry.register_spec(custom_spec)

        mock_orchestrator.config.workflow.validation.cross_validation.strategy = (
            "random"
        )
        mock_orchestrator.config.workflow.validation.cross_validation.grid_search = True
        mock_orchestrator.config.workflow.evaluation.metrics.metrics = [
            "dummy_custom_score",
            "ici",
        ]

        runner = MedpipeRunner(orchestrator=mock_orchestrator)

        X_train, y_train, _, _ = dummy_data
        mock_pipeline = MagicMock(spec=Pipeline)
        cv_splitter = MagicMock()

        hyperparams = {"max_depth": [3, 5], "n_estimators": 10}

        mock_search_instance = MagicMock()
        mock_grid_search.return_value = mock_search_instance
        mock_search_instance.best_estimator_ = "best_model"
        mock_search_instance.best_score_ = 0.9102
        mock_search_instance.cv_results_ = {"mean_test_dummy_custom_score": [0.95]}

        result = runner._train_model_cv(
            outcome="MORTALITY_30D",
            pipeline=mock_pipeline,
            hyperparams=hyperparams,
            X_train=X_train,
            y_train=y_train,
            groups_train=None,
            cv_splitter=cv_splitter,
        )

        mock_grid_search.assert_called_once()
        call_kwargs = mock_grid_search.call_args[1]

        assert call_kwargs["refit"] == "dummy_custom_score"
        passed_scoring = call_kwargs["scoring"]
        assert isinstance(passed_scoring, dict)
        assert "dummy_custom_score" in passed_scoring
        assert "ici" in passed_scoring

        mock_save_cv_results.assert_called_once()
        mock_search_instance.fit.assert_called_once_with(X_train, y_train, groups=None)
        assert result == "best_model"

    @patch("medpipe.pipeline.runner.MedpipeRunner._save_cv_results")
    @patch("medpipe.pipeline.runner.cross_validate")
    def test_train_model_cv_missing_metrics_config(
        self, mock_cv, mock_save_cv_results, mock_orchestrator, dummy_data
    ):
        """Test _train_model_cv falls back to 'roc_auc' if metrics config is missing."""
        runner = MedpipeRunner(orchestrator=mock_orchestrator)
        del runner.orchestrator.config.workflow.evaluation.metrics

        X_train, y_train, _, _ = dummy_data
        mock_pipeline = MagicMock(spec=Pipeline)

        runner._train_model_cv(
            outcome="MORTALITY_30D",
            pipeline=mock_pipeline,
            hyperparams={"depth": 3},
            X_train=X_train,
            y_train=y_train,
            groups_train=None,
            cv_splitter=MagicMock(),
        )

        passed_scoring = mock_cv.call_args[1]["scoring"]
        assert isinstance(passed_scoring, dict)
        assert "roc_auc" in passed_scoring
        mock_save_cv_results.assert_called_once()

    @patch("medpipe.pipeline.runner.MedpipeRunner._save_cv_results")
    def test_train_model_cv_group_strategy_without_groups_raises_error(
        self, mock_save_cv_results, mock_orchestrator, dummy_data
    ):
        """Test that cross-validation with a group strategy raises an error
        if groups_train is None."""
        mock_orchestrator.config.workflow.validation.cross_validation.strategy = "group"
        runner = MedpipeRunner(orchestrator=mock_orchestrator)

        X_train, y_train, _, _ = dummy_data
        mock_pipeline = MagicMock(spec=Pipeline)
        cv_splitter = runner._create_cv_splitter("group", 2, 42)

        hyperparams = {"max_depth": 3}

        with pytest.raises(
            ValueError, match="The 'groups' parameter should not be None"
        ):
            runner._train_model_cv(
                outcome="MORTALITY_30D",
                pipeline=mock_pipeline,
                hyperparams=hyperparams,
                X_train=X_train,
                y_train=y_train,
                groups_train=None,
                cv_splitter=cv_splitter,
            )

    @pytest.mark.parametrize("n_jobs", [1, 4, -1])
    @patch("medpipe.pipeline.runner.MedpipeRunner._save_cv_results")
    @patch("medpipe.pipeline.runner.cross_validate")
    def test_train_model_cv_standard_cv_passes_n_jobs(
        self, mock_cv, mock_save_cv_results, mock_orchestrator, dummy_data, n_jobs
    ):
        """Test _train_model_cv propagates configured n_jobs to sklearn
        cross_validate."""
        mock_orchestrator.config.workflow.n_jobs = n_jobs
        runner = MedpipeRunner(orchestrator=mock_orchestrator)

        X_train, y_train, _, _ = dummy_data

        runner._train_model_cv(
            outcome="MORTALITY_30D",
            pipeline=MagicMock(spec=Pipeline),
            hyperparams={"max_depth": 3},
            X_train=X_train,
            y_train=y_train,
            groups_train=None,
            cv_splitter=MagicMock(),
        )

        mock_cv.assert_called_once()
        assert mock_cv.call_args[1]["n_jobs"] == n_jobs

    @pytest.mark.parametrize("n_jobs", [1, 4, -1])
    @patch("medpipe.pipeline.runner.MedpipeRunner._save_cv_results")
    @patch("medpipe.pipeline.runner.GridSearchCV")
    def test_train_model_cv_grid_search_passes_n_jobs(
        self,
        mock_grid_search,
        mock_save_cv_results,
        mock_orchestrator,
        dummy_data,
        n_jobs,
    ):
        """Test _train_model_cv propagates configured n_jobs to sklearn GridSearchCV."""
        mock_orchestrator.config.workflow.n_jobs = n_jobs
        mock_orchestrator.config.workflow.validation.cross_validation.grid_search = True
        mock_orchestrator.config.workflow.evaluation.metrics.metrics = ["accuracy"]

        runner = MedpipeRunner(orchestrator=mock_orchestrator)
        X_train, y_train, _, _ = dummy_data

        mock_search_instance = MagicMock()
        mock_grid_search.return_value = mock_search_instance
        mock_search_instance.best_estimator_ = "best_model"
        mock_search_instance.best_score_ = 0.9102
        mock_search_instance.cv_results_ = {
            "params": [{"classifier__max_depth": 3}],
            "mean_test_accuracy": [0.85],
        }

        runner._train_model_cv(
            outcome="MORTALITY_30D",
            pipeline=MagicMock(spec=Pipeline),
            hyperparams={"max_depth": [3, 5]},
            X_train=X_train,
            y_train=y_train,
            groups_train=None,
            cv_splitter=MagicMock(),
        )

        mock_grid_search.assert_called_once()
        assert mock_grid_search.call_args[1]["n_jobs"] == n_jobs
