"""
Tests for MedpipeClassifierRunner.run.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from medpipe.pipeline.runner import MedpipeClassifierRunner


class TestRun:
    """Unit tests for MedpipeClassifierRunner.run."""

    @patch("medpipe.pipeline.runner.MedpipeClassifierRunner._save_final_models")
    @patch("medpipe.pipeline.runner.MedpipeClassifierRunner.fit_outcome")
    def test_run_orchestrates_outcomes_and_saves_final_models(
        self, mock_fit_outcome, mock_save_final_models, mock_orchestrator, dummy_data
    ):
        """Test that run method iterates outcomes, stores fitted models, and
        saves final model dictionary."""
        mock_orchestrator.config.data.outcomes = ["OUTCOME_1", "OUTCOME_2"]
        runner = MedpipeClassifierRunner(orchestrator=mock_orchestrator)

        X_train = dummy_data[0]
        y_train_df = pd.DataFrame(
            {"OUTCOME_1": [0, 1, 0, 1], "OUTCOME_2": [1, 1, 0, 0]}
        )
        X_recal = dummy_data[2]
        y_recal_df = pd.DataFrame({"OUTCOME_1": [1, 0], "OUTCOME_2": [0, 1]})

        mock_fit_outcome.return_value = MagicMock(spec=Pipeline)

        fitted_models = runner.run(X_train, y_train_df, X_recal, y_recal_df)

        assert mock_fit_outcome.call_count == 2
        assert "OUTCOME_1" in fitted_models
        assert "OUTCOME_2" in fitted_models
        mock_save_final_models.assert_called_once()

        first_call_kwargs = mock_fit_outcome.call_args_list[0].kwargs
        assert first_call_kwargs["outcome"] == "OUTCOME_1"
        assert np.array_equal(first_call_kwargs["y_train"], np.array([0, 1, 0, 1]))
        assert np.array_equal(first_call_kwargs["y_recal"], np.array([1, 0]))

    def test_run_missing_outcome_in_y_train_raises_keyerror(
        self, mock_orchestrator, dummy_data
    ):
        """Test that run raises KeyError if configured outcome column is
        absent in y_train_df."""
        mock_orchestrator.config.data.outcomes = ["MISSING_OUTCOME"]
        runner = MedpipeClassifierRunner(orchestrator=mock_orchestrator)

        X_train = dummy_data[0]
        y_train_df = pd.DataFrame({"OTHER_OUTCOME": [0, 1, 0, 1]})

        with pytest.raises(KeyError, match="MISSING_OUTCOME"):
            runner.run(X_train, y_train_df)

    @patch("medpipe.pipeline.runner.MedpipeClassifierRunner._save_final_models")
    @patch("medpipe.pipeline.runner.MedpipeClassifierRunner.fit_outcome")
    def test_run_without_recalibration_data(
        self, mock_fit_outcome, mock_save_final_models, mock_orchestrator, dummy_data
    ):
        """Test that run completes successfully when y_recal_df is omitted
        entirely (the default None), passing y_recal=None through to
        fit_outcome for every outcome."""
        mock_orchestrator.config.data.outcomes = ["OUTCOME_1"]
        runner = MedpipeClassifierRunner(orchestrator=mock_orchestrator)

        X_train = dummy_data[0]
        y_train_df = pd.DataFrame({"OUTCOME_1": [0, 1, 0, 1, 0, 1]})

        mock_fit_outcome.return_value = MagicMock(spec=Pipeline)

        runner.run(X_train, y_train_df)

        call_kwargs = mock_fit_outcome.call_args_list[0].kwargs
        assert call_kwargs["y_recal"] is None
        assert call_kwargs["X_recal"] is None

    @patch("medpipe.pipeline.runner.MedpipeClassifierRunner._save_final_models")
    @patch("medpipe.pipeline.runner.MedpipeClassifierRunner.fit_outcome")
    def test_run_recalibration_data_missing_outcome_column(
        self, mock_fit_outcome, mock_save_final_models, mock_orchestrator, dummy_data
    ):
        """Test that an outcome absent from y_recal_df's columns falls back
        to y_recal=None for that outcome specifically, rather than raising
        — recalibration data isn't required to cover every outcome."""
        mock_orchestrator.config.data.outcomes = ["OUTCOME_1", "OUTCOME_2"]
        runner = MedpipeClassifierRunner(orchestrator=mock_orchestrator)

        X_train = dummy_data[0]
        y_train_df = pd.DataFrame(
            {"OUTCOME_1": [0, 1, 0, 1], "OUTCOME_2": [1, 1, 0, 0]}
        )
        # Only OUTCOME_1 has recalibration data available.
        y_recal_df = pd.DataFrame({"OUTCOME_1": [1, 0]})

        mock_fit_outcome.return_value = MagicMock(spec=Pipeline)

        runner.run(X_train, y_train_df, y_recal_df=y_recal_df)

        calls_by_outcome = {
            call.kwargs["outcome"]: call.kwargs
            for call in mock_fit_outcome.call_args_list
        }
        assert np.array_equal(
            calls_by_outcome["OUTCOME_1"]["y_recal"], np.array([1, 0])
        )
        assert calls_by_outcome["OUTCOME_2"]["y_recal"] is None
