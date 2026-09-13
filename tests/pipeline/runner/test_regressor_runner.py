"""
Tests for MedpipeRegressorRunner.
"""

from unittest.mock import MagicMock, patch

import pytest
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline

from medpipe.pipeline.estimator import DistributionalPipeline
from medpipe.pipeline.runner import MedpipeRegressorRunner
from medpipe.utils.config import RegressorModelSetup


class TestCreateCvSplitter:
    """Unit tests for MedpipeRegressorRunner._create_cv_splitter."""

    def test_create_cv_splitter(self, mock_orchestrator):
        runner = MedpipeRegressorRunner(orchestrator=mock_orchestrator)

        random_cv = runner._create_cv_splitter("random", 3, 42)
        assert isinstance(random_cv, KFold)
        assert random_cv.n_splits == 3

        group_cv = runner._create_cv_splitter("group", 5, 42)
        assert isinstance(group_cv, GroupKFold)

        with pytest.raises(ValueError, match="Strategy must be 'random' or 'group'"):
            runner._create_cv_splitter("invalid_strategy", 5, 42)


class TestFitOutcomeBuildsDistributionalPipeline:
    """Tests confirming MedpipeRegressorRunner wires up the 'regressor'
    step name and DistributionalPipeline class."""

    @patch("medpipe.pipeline.runner.MedpipeRegressorRunner._save_model")
    def test_fit_outcome_uses_regressor_step_name_and_pipeline_class(
        self, mock_save, mock_orchestrator, dummy_data
    ):
        """Test that fit_outcome builds a DistributionalPipeline with a
        'regressor' final step, and that no post-hoc adjustment occurs
        (recalibration is not supported on the regression track)."""
        mock_orchestrator.config.meta.run_mode = "fast"

        mock_model_setup = MagicMock(spec=RegressorModelSetup)
        mock_model_setup.algorithm = "LinearRegression"
        mock_model_setup.hyperparameters = {}
        mock_model_setup.model_dump.return_value = {}
        mock_orchestrator.config.resolved_models = {"LOS_DAYS": mock_model_setup}

        runner = MedpipeRegressorRunner(orchestrator=mock_orchestrator)
        X_train, y_train, _, _ = dummy_data

        captured_pipelines = []
        original_init = Pipeline.__init__

        def capturing_init(self, steps, **kwargs):
            captured_pipelines.append(steps)
            original_init(self, steps, **kwargs)

        with (
            patch.object(Pipeline, "__init__", capturing_init),
            patch.object(Pipeline, "fit", return_value=MagicMock()),
        ):
            model = runner.fit_outcome("LOS_DAYS", X_train, y_train)

        step_names = [name for name, _ in captured_pipelines[0]]
        assert step_names == ["regressor"]

        mock_save.assert_called_once_with(model, "LOS_DAYS")

    def test_fit_outcome_pipeline_is_distributional(
        self, mock_orchestrator, dummy_data
    ):
        """Test that the fitted pipeline returned is a DistributionalPipeline
        instance, not a plain sklearn Pipeline."""
        mock_orchestrator.config.meta.run_mode = "fast"

        mock_model_setup = MagicMock(spec=RegressorModelSetup)
        mock_model_setup.algorithm = "LinearRegression"
        mock_model_setup.hyperparameters = {}
        mock_model_setup.model_dump.return_value = {}
        mock_orchestrator.config.resolved_models = {"LOS_DAYS": mock_model_setup}

        runner = MedpipeRegressorRunner(orchestrator=mock_orchestrator)
        X_train, y_train, _, _ = dummy_data

        with patch(
            "medpipe.pipeline.runner.MedpipeRegressorRunner._save_model"
        ):
            model = runner.fit_outcome("LOS_DAYS", X_train, y_train)

        assert isinstance(model, DistributionalPipeline)
