"""
Tests for MedpipeRunner._calibrate_model.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from medpipe.pipeline.runner import MedpipeRunner


class TestCalibrateModel:
    """Unit tests for MedpipeRunner._calibrate_model."""

    @patch("medpipe.pipeline.runner.CalibratedClassifierCV")
    @patch("medpipe.pipeline.runner.FrozenEstimator")
    def test_calibrate_model_success(
        self, mock_frozen, mock_calibrated, mock_orchestrator, dummy_data
    ):
        """Test successful calibration with holdout data."""
        runner = MedpipeRunner(orchestrator=mock_orchestrator)
        _, _, X_recal, y_recal = dummy_data

        mock_pipeline = MagicMock(spec=Pipeline)
        model_config = {"recalibration": {"recalibrate": True, "method": "sigmoid"}}

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

        mock_frozen.assert_called_once_with(mock_pipeline)
        mock_calibrated.assert_called_once_with(
            estimator="frozen_pipeline", cv=2, method="sigmoid"
        )
        mock_calibrator_instance.fit.assert_called_once_with(X_recal, y_recal)
        assert result == "final_calibrated_model"

    def test_calibrate_model_skip_none_data(self, mock_orchestrator):
        """Test calibration is skipped when X_recal is None."""
        runner = MedpipeRunner(orchestrator=mock_orchestrator)
        mock_pipeline = MagicMock(spec=Pipeline)
        model_config = {"recalibration": {"recalibrate": True, "method": "sigmoid"}}

        result = runner._calibrate_model(
            outcome="MORTALITY_30D",
            best_pipeline=mock_pipeline,
            model_config=model_config,
            X_recal=None,
            y_recal=None,
        )

        assert result == mock_pipeline

    def test_calibrate_model_skip_empty_dataframe(self, mock_orchestrator):
        """Test calibration is skipped when X_recal is an empty DataFrame."""
        runner = MedpipeRunner(orchestrator=mock_orchestrator)
        mock_pipeline = MagicMock(spec=Pipeline)
        model_config = {"recalibration": {"recalibrate": True, "method": "sigmoid"}}

        result = runner._calibrate_model(
            outcome="MORTALITY_30D",
            best_pipeline=mock_pipeline,
            model_config=model_config,
            X_recal=pd.DataFrame(),
            y_recal=np.array([]),
        )

        assert result == mock_pipeline

    def test_calibrate_model_skip_missing_config(self, mock_orchestrator, dummy_data):
        """Test calibration is skipped when model config lacks recalibration settings."""
        runner = MedpipeRunner(orchestrator=mock_orchestrator)
        _, _, X_recal, y_recal = dummy_data

        mock_pipeline = MagicMock(spec=Pipeline)
        model_config = {}

        result = runner._calibrate_model(
            outcome="MORTALITY_30D",
            best_pipeline=mock_pipeline,
            model_config=model_config,
            X_recal=X_recal,
            y_recal=y_recal,
        )

        assert result == mock_pipeline

    def test_calibrate_model_skip_recalibrate_flag_false(
        self, mock_orchestrator, dummy_data
    ):
        """Test calibration is skipped when recalibrate is set to False in config."""
        runner = MedpipeRunner(orchestrator=mock_orchestrator)
        _, _, X_recal, y_recal = dummy_data

        mock_pipeline = MagicMock(spec=Pipeline)
        model_config = {"recalibration": {"recalibrate": False, "method": "sigmoid"}}

        result = runner._calibrate_model(
            outcome="MORTALITY_30D",
            best_pipeline=mock_pipeline,
            model_config=model_config,
            X_recal=X_recal,
            y_recal=y_recal,
        )

        assert result == mock_pipeline

    def test_calibrate_model_raises_assertion_on_none_y_recal(
        self, mock_orchestrator, dummy_data
    ):
        """Test assertion error is raised if y_recal is None when
        recalibration is enabled."""
        runner = MedpipeRunner(orchestrator=mock_orchestrator)
        _, _, X_recal, _ = dummy_data

        mock_pipeline = MagicMock(spec=Pipeline)
        model_config = {"recalibration": {"recalibrate": True, "method": "sigmoid"}}

        with pytest.raises(AssertionError):
            runner._calibrate_model(
                outcome="MORTALITY_30D",
                best_pipeline=mock_pipeline,
                model_config=model_config,
                X_recal=X_recal,
                y_recal=None,
            )
