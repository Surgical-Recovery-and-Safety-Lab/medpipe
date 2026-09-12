"""
Tests for MedpipeRunner.fit_outcome.
"""

from unittest.mock import MagicMock, patch

import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from medpipe.pipeline.runner import MedpipeRunner
from medpipe.utils.config import ModelSetup


class TestFitOutcome:
    """Unit tests for MedpipeRunner.fit_outcome."""

    @pytest.mark.parametrize(
        "run_mode, should_call_cv",
        [
            ("fast", False),
            ("eval", False),
            ("cv", True),
            ("audit", True),
        ],
    )
    @patch("medpipe.pipeline.runner.MedpipeRunner._save_model")
    @patch("medpipe.pipeline.runner.MedpipeRunner._train_model_cv")
    def test_fit_outcome_run_modes(
        self,
        mock_train_cv,
        mock_save,
        mock_orchestrator,
        dummy_data,
        run_mode,
        should_call_cv,
    ):
        """Verify run_mode logic properly controls routing to CV vs
        direct pipeline fitting."""
        mock_orchestrator.config.meta.run_mode = run_mode

        # Configure cross-validation mock so _create_cv_splitter succeeds
        cv_cfg = MagicMock()
        cv_cfg.strategy = "random"
        cv_cfg.n_splits = 3
        mock_orchestrator.config.workflow.validation.cross_validation = cv_cfg
        mock_orchestrator.config.workflow.random_state = 42

        # Set up resolved_models on mock_orchestrator.config
        mock_model_setup = MagicMock(spec=ModelSetup)
        mock_model_setup.algorithm = "HistGradientBoostingClassifier"
        mock_model_setup.hyperparameters = {}
        mock_model_setup.model_dump.return_value = {}
        mock_orchestrator.config.resolved_models = {"MORTALITY_30D": mock_model_setup}

        runner = MedpipeRunner(orchestrator=mock_orchestrator)
        X_train, y_train, _, _ = dummy_data

        with patch.object(
            Pipeline, "fit", return_value=MagicMock()
        ) as mock_pipeline_fit:
            runner.fit_outcome("MORTALITY_30D", X_train, y_train)

            if should_call_cv:
                mock_train_cv.assert_called_once()
                mock_pipeline_fit.assert_not_called()
            else:
                mock_train_cv.assert_not_called()
                mock_pipeline_fit.assert_called_once_with(X_train, y_train)

    def test_fit_outcome_no_algorithm_raises_error(
        self,
        mock_orchestrator,
        dummy_data,
    ):
        """Test missing algorithm configuration fails gracefully."""
        mock_model_setup = MagicMock(spec=ModelSetup)
        mock_model_setup.algorithm = None
        mock_orchestrator.config.resolved_models = {"MORTALITY_30D": mock_model_setup}

        runner = MedpipeRunner(orchestrator=mock_orchestrator)

        with pytest.raises(
            ValueError, match="No algorithm specified for outcome: MORTALITY_30D"
        ):
            runner.fit_outcome("MORTALITY_30D", dummy_data[0], dummy_data[1])

    @patch("medpipe.pipeline.runner.MedpipeRunner._train_model_cv")
    @patch("medpipe.pipeline.runner.MedpipeRunner._save_model")
    def test_fit_outcome_with_recalibration(
        self, mock_save, mock_train_cv, mock_orchestrator, dummy_data
    ):
        """Test fit_outcome utilizes CalibratedClassifierCV when recal
        data is provided."""
        mock_orchestrator.config.meta.run_mode = "cv"

        # Configure cross-validation mock
        cv_cfg = MagicMock()
        cv_cfg.strategy = "random"
        cv_cfg.n_splits = 3
        mock_orchestrator.config.workflow.validation.cross_validation = cv_cfg
        mock_orchestrator.config.workflow.random_state = 42

        # Configure model setup with recalibration dict
        mock_model_setup = MagicMock(spec=ModelSetup)
        mock_model_setup.algorithm = "RandomForestClassifier"
        mock_model_setup.hyperparameters = {}
        mock_model_setup.model_dump.return_value = {
            "recalibration": {"recalibrate": True, "method": "isotonic"}
        }
        mock_orchestrator.config.resolved_models = {"MORTALITY_30D": mock_model_setup}

        runner = MedpipeRunner(orchestrator=mock_orchestrator)
        X_train, y_train, X_recal, y_recal = dummy_data

        real_pipeline = Pipeline(
            [("clf", RandomForestClassifier(n_estimators=2, max_depth=2))]
        )
        real_pipeline.fit(X_train, y_train)
        mock_train_cv.return_value = real_pipeline

        model = runner.fit_outcome(
            "MORTALITY_30D", X_train, y_train, X_recal=X_recal, y_recal=y_recal
        )

        assert isinstance(model, CalibratedClassifierCV)
        mock_save.assert_called_once_with(model, "MORTALITY_30D")
        mock_train_cv.assert_called_once()

    def test_fit_outcome_group_cv_missing_groups_raises_error(
        self, mock_orchestrator, dummy_data
    ):
        """Test that group cross-validation strategy raises ValueError
        when groups_train is None."""
        mock_orchestrator.config.meta.run_mode = "cv"

        cv_cfg = MagicMock()
        cv_cfg.strategy = "group"
        cv_cfg.n_splits = 3
        mock_orchestrator.config.workflow.validation.cross_validation = cv_cfg
        mock_orchestrator.config.workflow.random_state = 42

        mock_model_setup = MagicMock(spec=ModelSetup)
        mock_model_setup.algorithm = "RandomForestClassifier"
        mock_model_setup.hyperparameters = {}
        mock_orchestrator.config.resolved_models = {"MORTALITY_30D": mock_model_setup}

        runner = MedpipeRunner(orchestrator=mock_orchestrator)
        X_train, y_train, _, _ = dummy_data

        with pytest.raises(
            ValueError, match="The 'groups' parameter should not be None"
        ):
            runner.fit_outcome(
                "MORTALITY_30D",
                X_train,
                y_train,
                groups_train=None,
            )

    @patch("medpipe.pipeline.runner.MedpipeRunner._save_model")
    def test_fit_outcome_includes_preprocessor_step_when_present(
        self, mock_save, mock_orchestrator, dummy_data
    ):
        """Test that when the orchestrator provides a preprocessor, it is
        prepended as a 'preprocessor' step ahead of the classifier — the
        branch skipped whenever build_preprocessor() returns None (the
        fixture's default)."""
        mock_orchestrator.config.meta.run_mode = "fast"

        mock_preprocessor = MagicMock(spec=Pipeline)
        mock_orchestrator.build_preprocessor.return_value = mock_preprocessor

        mock_model_setup = MagicMock(spec=ModelSetup)
        mock_model_setup.algorithm = "RandomForestClassifier"
        mock_model_setup.hyperparameters = {}
        mock_model_setup.model_dump.return_value = {}
        mock_orchestrator.config.resolved_models = {"MORTALITY_30D": mock_model_setup}

        runner = MedpipeRunner(orchestrator=mock_orchestrator)
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
            runner.fit_outcome("MORTALITY_30D", X_train, y_train)

        step_names = [name for name, _ in captured_pipelines[0]]
        assert step_names == ["preprocessor", "classifier"]
        assert captured_pipelines[0][0][1] is mock_preprocessor
