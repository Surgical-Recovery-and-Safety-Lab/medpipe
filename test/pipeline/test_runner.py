from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline

from medpipe.models.registry import ModelRegistry
from medpipe.pipeline.orchestrator import MedpipeOrchestrator
from medpipe.pipeline.runner import MedpipeRunner


class TestMedpipeRunner:
    """
    Comprehensive test suite for the MedpipeRunner class.
    """

    @pytest.fixture
    def mock_orchestrator(self):
        """Creates a mocked MedpipeOrchestrator with standard configurations."""
        orchestrator = MagicMock(spec=MedpipeOrchestrator)
        orchestrator.run_dir = Path("/fake/run/dir")
        orchestrator.build_preprocessor.return_value = None

        # Explicitly mock the configuration object first, replicating your mock_config fixture
        mock_config = MagicMock()
        mock_config.data = MagicMock()
        mock_config.workflow = MagicMock()

        # Apply the required runner configurations
        mock_config.data.outcomes = ["MORTALITY_30D"]
        mock_config.workflow.validation.cross_validation.strategy = "random"
        mock_config.workflow.validation.cross_validation.n_splits = 2
        mock_config.workflow.validation.cross_validation.random_state = 42
        mock_config.workflow.evaluation.metrics = ["accuracy"]

        # Attach the configuration mock to the orchestrator mock
        orchestrator.config = mock_config

        # Standard resolved configuration
        orchestrator.resolved_model_configs = {
            "MORTALITY_30D": {
                "algorithm": "RandomForestClassifier",
                "hyperparameters": {"n_estimators": 5, "max_depth": 3},
                "recalibration": {"method": "temperature"},
            }
        }
        return orchestrator

    @pytest.fixture
    def dummy_data(self):
        """Provides small, consistent dummy data for fitting."""
        X_train = pd.DataFrame(
            {"feature1": [1, 2, 3, 4, 5, 6], "feature2": [6, 5, 4, 3, 2, 1]}
        )
        y_train = np.array([0, 1, 0, 1, 0, 1])

        X_recal = pd.DataFrame(
            {"feature1": [2, 3, 4, 5, 6, 7], "feature2": [7, 6, 5, 4, 3, 2]}
        )
        y_recal = np.array([1, 0, 1, 0, 1, 0])

        return X_train, y_train, X_recal, y_recal

    # --- Unit Tests for Internal Methods ---

    def test_instantiate_estimator_classifier(self, mock_orchestrator):
        runner = MedpipeRunner(orchestrator=mock_orchestrator)
        estimator = runner._instantiate_estimator(
            "RandomForestClassifier", {"n_estimators": 10}
        )

        assert isinstance(estimator, RandomForestClassifier)
        assert estimator.n_estimators == 10

    def test_instantiate_estimator_regressor_wrapped(self, mock_orchestrator):
        """Test that regressors are automatically wrapped in TransformedTargetRegressor."""
        # Ensure LinearRegression is available via the fallback modules
        runner = MedpipeRunner(orchestrator=mock_orchestrator)
        estimator = runner._instantiate_estimator("LinearRegression", {})

        assert isinstance(estimator, TransformedTargetRegressor)
        assert isinstance(estimator.regressor, LinearRegression)

    def test_instantiate_estimator_list_params_filtered(self, mock_orchestrator):
        """Test that list hyperparameters are reduced to scalars for initial instantiation."""
        runner = MedpipeRunner(orchestrator=mock_orchestrator)
        params = {"n_estimators": [10, 50, 100], "max_depth": 5}

        estimator = runner._instantiate_estimator("RandomForestClassifier", params)
        assert isinstance(estimator, RandomForestClassifier)
        assert estimator.n_estimators == 10  # Should take the first element of the list
        assert estimator.max_depth == 5

    def test_create_cv_splitter(self, mock_orchestrator):
        runner = MedpipeRunner(orchestrator=mock_orchestrator)

        random_cv = runner._create_cv_splitter("random", 3, 42)
        assert isinstance(random_cv, StratifiedKFold)
        assert random_cv.n_splits == 3

        group_cv = runner._create_cv_splitter("group", 5, 42)
        assert isinstance(group_cv, StratifiedGroupKFold)

        with pytest.raises(ValueError, match="Strategy must be 'random' or 'group'"):
            runner._create_cv_splitter("invalid_strategy", 5, 42)

    @patch("medpipe.pipeline.runner.joblib.dump")
    def test_save_model(self, mock_dump, mock_orchestrator):
        runner = MedpipeRunner(orchestrator=mock_orchestrator)
        mock_model = MagicMock(spec=Pipeline)

        # Setup mock run_dir to avoid actual disk writes if parents don't exist
        with patch.object(Path, "mkdir") as mock_mkdir:
            runner._save_model(mock_model, "MORTALITY_30D")

            mock_mkdir.assert_called_once_with(exist_ok=True, parents=True)
            mock_dump.assert_called_once()

            # Verify the filepath constructed
            expected_path = Path("/fake/run/dir/models/MORTALITY_30D_model.joblib")
            assert mock_dump.call_args[0][1] == expected_path

    # --- Unit Tests for Training and Calibration Sub-routines ---

    @patch("medpipe.pipeline.runner.cross_validate")
    def test_train_model_standard_cv(self, mock_cv, mock_orchestrator, dummy_data):
        """Test _train_model uses standard cross_validate when no list hyperparameters exist."""
        runner = MedpipeRunner(orchestrator=mock_orchestrator)

        X_train, y_train, _, _ = dummy_data
        mock_pipeline = MagicMock(spec=Pipeline)
        cv_splitter = MagicMock()

        # Scalar hyperparameters trigger standard CV
        hyperparams = {"max_depth": 3, "n_estimators": 100}

        result = runner._train_model(
            outcome="MORTALITY_30D",
            pipeline=mock_pipeline,
            hyperparams=hyperparams,
            X_train=X_train,
            y_train=y_train,
            groups_train=None,
            cv_splitter=cv_splitter,
        )

        # Verify cross_validate was called with correct arguments
        mock_cv.assert_called_once_with(
            estimator=mock_pipeline,
            X=X_train,
            y=y_train,
            groups=None,
            cv=cv_splitter,
            scoring=["accuracy"],  # Drawn from the mock_orchestrator fixture
            n_jobs=-1,
        )

        # Verify the base pipeline was fitted on the full training data and returned
        mock_pipeline.fit.assert_called_once_with(X_train, y_train)
        assert result == mock_pipeline.fit.return_value

    @patch("medpipe.pipeline.runner.GridSearchCV")
    def test_train_model_grid_search(
        self, mock_grid_search, mock_orchestrator, dummy_data
    ):
        """Test _train_model triggers GridSearchCV and prefixes parameters correctly."""
        runner = MedpipeRunner(orchestrator=mock_orchestrator)
        # Adding a second metric to verify refit behavior
        runner.orchestrator.config.workflow.evaluation.metrics = ["accuracy", "roc_auc"]

        X_train, y_train, _, _ = dummy_data
        mock_pipeline = MagicMock(spec=Pipeline)
        cv_splitter = MagicMock()

        # List hyperparameter triggers GridSearchCV
        hyperparams = {"max_depth": [3, 5], "n_estimators": 10}

        # Setup mock return for GridSearchCV
        mock_search_instance = MagicMock()
        mock_grid_search.return_value = mock_search_instance
        mock_search_instance.best_estimator_ = "best_model"

        result = runner._train_model(
            outcome="MORTALITY_30D",
            pipeline=mock_pipeline,
            hyperparams=hyperparams,
            X_train=X_train,
            y_train=y_train,
            groups_train=None,
            cv_splitter=cv_splitter,
        )

        # Verify parameters were prefixed with 'classifier__' and
        # scalars converted to lists
        expected_params = {
            "classifier__max_depth": [3, 5],
            "classifier__n_estimators": [10],
        }

        mock_grid_search.assert_called_once_with(
            estimator=mock_pipeline,
            param_grid=expected_params,
            cv=cv_splitter,
            scoring=["accuracy", "roc_auc"],
            refit="accuracy",  # Should map to the first configured metric
            n_jobs=-1,
        )

        # Verify search was fitted and best estimator returned
        mock_search_instance.fit.assert_called_once_with(X_train, y_train, groups=None)
        assert result == "best_model"

    @patch("medpipe.pipeline.runner.cross_validate")
    def test_train_model_missing_metrics_config(
        self, mock_cv, mock_orchestrator, dummy_data
    ):
        """Test _train_model falls back to 'roc_auc' if metrics config is missing."""
        runner = MedpipeRunner(orchestrator=mock_orchestrator)
        # Explicitly break the metrics config to trigger the AttributeError fallback
        del runner.orchestrator.config.workflow.evaluation.metrics

        X_train, y_train, _, _ = dummy_data
        mock_pipeline = MagicMock(spec=Pipeline)

        runner._train_model(
            outcome="MORTALITY_30D",
            pipeline=mock_pipeline,
            hyperparams={"depth": 3},
            X_train=X_train,
            y_train=y_train,
            groups_train=None,
            cv_splitter=MagicMock(),
        )

        # Assert fallback metric was used
        assert mock_cv.call_args[1]["scoring"] == ["roc_auc"]

    @patch("medpipe.pipeline.runner.CalibratedClassifierCV")
    @patch("medpipe.pipeline.runner.FrozenEstimator")
    def test_calibrate_model_success(
        self, mock_frozen, mock_calibrated, mock_orchestrator, dummy_data
    ):
        """Test successful calibration with holdout data."""
        runner = MedpipeRunner(orchestrator=mock_orchestrator)
        _, _, X_recal, y_recal = dummy_data

        mock_pipeline = MagicMock(spec=Pipeline)
        model_config = {"recalibration": {"method": "sigmoid"}}

        # Setup mock returns
        mock_calibrator_instance = MagicMock()
        mock_calibrated.return_value = mock_calibrator_instance
        mock_calibrator_instance.fit.return_value = "final_calibrated_model"
        mock_frozen.return_value = "frozen_pipeline"

        result = runner._calibrate_model(
            outcome="MORTALITY_30D",
            best_pipeline=mock_pipeline,
            model_config=model_config,
            X_recal=X_recal,
            y_recal=y_recal,
        )

        # Verify the pipeline was frozen
        mock_frozen.assert_called_once_with(mock_pipeline)

        # Verify calibrator was built with the frozen model and correct method
        mock_calibrated.assert_called_once_with(
            estimator="frozen_pipeline", cv=2, method="sigmoid"
        )

        # Verify calibrator was fitted on recalibration data
        mock_calibrator_instance.fit.assert_called_once_with(X_recal, y_recal)
        assert result == "final_calibrated_model"

    def test_calibrate_model_skip_none_data(self, mock_orchestrator):
        """Test calibration is skipped when X_recal is None."""
        runner = MedpipeRunner(orchestrator=mock_orchestrator)

        mock_pipeline = MagicMock(spec=Pipeline)
        model_config = {"recalibration": {"method": "sigmoid"}}

        result = runner._calibrate_model(
            outcome="MORTALITY_30D",
            best_pipeline=mock_pipeline,
            model_config=model_config,
            X_recal=None,
            y_recal=None,
        )

        # Should return original pipeline untouched
        assert result == mock_pipeline

    def test_calibrate_model_skip_empty_dataframe(self, mock_orchestrator):
        """Test calibration is skipped when X_recal is an empty DataFrame."""
        runner = MedpipeRunner(orchestrator=mock_orchestrator)

        mock_pipeline = MagicMock(spec=Pipeline)
        model_config = {"recalibration": {"method": "sigmoid"}}

        result = runner._calibrate_model(
            outcome="MORTALITY_30D",
            best_pipeline=mock_pipeline,
            model_config=model_config,
            X_recal=pd.DataFrame(),  # Empty DataFrame
            y_recal=np.array([]),
        )

        # Should return original pipeline untouched
        assert result == mock_pipeline

    def test_calibrate_model_skip_missing_config(self, mock_orchestrator, dummy_data):
        """Test calibration is skipped when model config lacks recalibration settings."""
        runner = MedpipeRunner(orchestrator=mock_orchestrator)
        _, _, X_recal, y_recal = dummy_data

        mock_pipeline = MagicMock(spec=Pipeline)
        model_config = {}  # Empty config, no recalibration key

        result = runner._calibrate_model(
            outcome="MORTALITY_30D",
            best_pipeline=mock_pipeline,
            model_config=model_config,
            X_recal=X_recal,
            y_recal=y_recal,
        )

        # Should return original pipeline untouched
        assert result == mock_pipeline

    # --- Tests for the Execution Loop ---

    def test_fit_outcome_no_algorithm_raises_error(self, mock_orchestrator, dummy_data):
        """Test missing algorithm configuration fails gracefully."""
        mock_orchestrator.resolved_model_configs = {"MORTALITY_30D": {}}
        runner = MedpipeRunner(orchestrator=mock_orchestrator)

        with pytest.raises(
            ValueError, match="No algorithm specified for outcome: MORTALITY_30D"
        ):
            runner.fit_outcome("MORTALITY_30D", dummy_data[0], dummy_data[1])

    @patch("medpipe.pipeline.runner.MedpipeRunner._save_model")
    def test_fit_outcome_standard_cv(self, mock_save, mock_orchestrator, dummy_data):
        """Test fit_outcome using standard cross-validation (no list hyperparams)."""
        runner = MedpipeRunner(orchestrator=mock_orchestrator)
        X_train, y_train, _, _ = dummy_data

        # Patch cross_validate so it doesn't actually run, but returns dummy metrics
        with patch("medpipe.pipeline.runner.cross_validate") as mock_cv:
            mock_cv.return_value = {"test_score": [0.8, 0.85]}

            model = runner.fit_outcome("MORTALITY_30D", X_train, y_train)

            # Assert cross_validate was called, not GridSearchCV
            mock_cv.assert_called_once()

            # Assert returning a Pipeline
            assert isinstance(model, Pipeline)
            assert "classifier" in model.named_steps
            mock_save.assert_called_once_with(model, "MORTALITY_30D")

    @patch("medpipe.pipeline.runner.MedpipeRunner._save_model")
    def test_fit_outcome_gridsearch(self, mock_save, mock_orchestrator, dummy_data):
        """Test fit_outcome triggers GridSearchCV when lists are present in params."""
        # Inject list parameter to trigger GridSearch
        mock_orchestrator.resolved_model_configs["MORTALITY_30D"]["hyperparameters"] = {
            "n_estimators": [5, 10],
            "max_depth": 3,
        }

        runner = MedpipeRunner(orchestrator=mock_orchestrator)
        X_train, y_train, _, _ = dummy_data

        with patch("medpipe.pipeline.runner.GridSearchCV") as MockGridSearch:
            mock_gs_instance = MockGridSearch.return_value
            mock_gs_instance.best_estimator_ = Pipeline(
                [("classifier", RandomForestClassifier())]
            )
            mock_gs_instance.best_params_ = {
                "classifier__n_estimators": 5,
                "classifier__max_depth": [3],
            }

            model = runner.fit_outcome("MORTALITY_30D", X_train, y_train)

            MockGridSearch.assert_called_once()
            mock_gs_instance.fit.assert_called_once()
            assert isinstance(model, Pipeline)
            mock_save.assert_called_once_with(model, "MORTALITY_30D")

    @patch("medpipe.pipeline.runner.MedpipeRunner._save_model")
    def test_fit_outcome_with_recalibration(
        self, mock_save, mock_orchestrator, dummy_data
    ):
        """Test fit_outcome utilizes CalibratedClassifierCV when recal data is provided."""
        runner = MedpipeRunner(orchestrator=mock_orchestrator)
        X_train, y_train, X_recal, y_recal = dummy_data

        with patch("medpipe.pipeline.runner.cross_validate"):
            model = runner.fit_outcome(
                "MORTALITY_30D", X_train, y_train, X_recal=X_recal, y_recal=y_recal
            )

            # Should be wrapped in calibrator
            assert isinstance(model, CalibratedClassifierCV)
            mock_save.assert_called_once_with(model, "MORTALITY_30D")

    @patch("medpipe.pipeline.runner.MedpipeRunner.fit_outcome")
    def test_run_orchestrates_outcomes(
        self, mock_fit_outcome, mock_orchestrator, dummy_data
    ):
        """Test that run method correctly iterates outcomes and unpacks dataframes."""
        # Setup multiple outcomes
        mock_orchestrator.config.data.outcomes = ["OUTCOME_1", "OUTCOME_2"]
        runner = MedpipeRunner(orchestrator=mock_orchestrator)

        X_train = dummy_data[0]
        y_train_df = pd.DataFrame(
            {"OUTCOME_1": [0, 1, 0, 1], "OUTCOME_2": [1, 1, 0, 0]}
        )
        X_recal = dummy_data[2]
        y_recal_df = pd.DataFrame({"OUTCOME_1": [1, 0], "OUTCOME_2": [0, 1]})

        mock_fit_outcome.return_value = MagicMock(spec=Pipeline)

        fitted_models = runner.run(X_train, y_train_df, X_recal, y_recal_df)

        # Verify fit_outcome was called twice
        assert mock_fit_outcome.call_count == 2
        assert "OUTCOME_1" in fitted_models
        assert "OUTCOME_2" in fitted_models

        # Verify the 1D extraction for the first call
        first_call_kwargs = mock_fit_outcome.call_args_list[0].kwargs
        assert first_call_kwargs["outcome"] == "OUTCOME_1"
        assert np.array_equal(first_call_kwargs["y_train"], np.array([0, 1, 0, 1]))
        assert np.array_equal(first_call_kwargs["y_recal"], np.array([1, 0]))
