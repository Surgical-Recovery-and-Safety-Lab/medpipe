"""
Tests for MedpipeClassifierRunner's artifact-persistence helpers: _save_model,
_save_final_models, and _save_cv_results.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from medpipe.pipeline.runner import MedpipeClassifierRunner


class TestSaveModel:
    """Unit tests for MedpipeClassifierRunner._save_model."""

    @patch("medpipe.pipeline.runner.joblib.dump")
    def test_save_model(self, mock_dump, mock_orchestrator):
        runner = MedpipeClassifierRunner(orchestrator=mock_orchestrator)
        mock_model = MagicMock(spec=Pipeline)

        with patch.object(Path, "mkdir") as mock_mkdir:
            runner._save_model(mock_model, "MORTALITY_30D")

            mock_mkdir.assert_called_once_with(exist_ok=True, parents=True)
            mock_dump.assert_called_once()

            expected_path = Path("/fake/run/dir/models/MORTALITY_30D_model.joblib")
            assert mock_dump.call_args[0][1] == expected_path


class TestSaveFinalModels:
    """Unit tests for MedpipeClassifierRunner._save_final_models."""

    @patch("medpipe.pipeline.runner.joblib.dump")
    def test_save_final_models(self, mock_dump, mock_orchestrator):
        """Test that _save_final_models saves the complete model bundle using
        project_name."""
        runner = MedpipeClassifierRunner(orchestrator=mock_orchestrator)
        mock_model = MagicMock(spec=Pipeline)
        runner.fitted_models = {"MORTALITY_30D": mock_model}

        with patch.object(Path, "mkdir") as mock_mkdir:
            runner._save_final_models()

            mock_mkdir.assert_called_once_with(exist_ok=True, parents=True)
            mock_dump.assert_called_once()

            expected_path = Path("/fake/run/dir/models/test_project_fitted.joblib")
            assert mock_dump.call_args[0][0] == {"MORTALITY_30D": mock_model}
            assert mock_dump.call_args[0][1] == expected_path


class TestSaveCvResults:
    """Unit tests for MedpipeClassifierRunner._save_cv_results."""

    @patch.object(pd.DataFrame, "to_csv")
    def test_save_cv_results(self, mock_to_csv, mock_orchestrator):
        """Test that _save_cv_results writes fold results to CSV and summary
        statistics to JSON."""
        mock_artifact_manager = MagicMock()
        mock_orchestrator.artifact_manager = mock_artifact_manager
        runner = MedpipeClassifierRunner(orchestrator=mock_orchestrator)

        cv_data = {
            "fit_time": [0.1, 0.2],
            "test_accuracy": [0.8, 0.9],
            "test_ici": [0.02, 0.04],
        }
        cv_results_df = pd.DataFrame(cv_data)

        with patch.object(Path, "mkdir") as mock_mkdir:
            runner._save_cv_results("MORTALITY_30D", cv_results_df)

            mock_mkdir.assert_called_once_with(exist_ok=True, parents=True)
            mock_to_csv.assert_called_once_with(
                Path("/fake/run/dir/CV/MORTALITY_30D_cv_results.csv"),
                index=False,
            )
            mock_artifact_manager.save_json.assert_called_once()

            # Extract arguments passed to
            # save_json(obj, destination_dir, filename)
            args, kwargs = mock_artifact_manager.save_json.call_args
            saved_obj = args[0] if args else kwargs.get("obj")
            filename = args[2] if len(args) > 2 else kwargs.get("filename")

            # Verify the summary contents and filename
            assert "fit_time" not in saved_obj
            assert "accuracy" in saved_obj
            assert "ici" in saved_obj
            assert saved_obj["accuracy"]["mean"] == pytest.approx(0.85)
            assert saved_obj["ici"]["std"] == pytest.approx(
                cv_results_df["test_ici"].std()
            )
            assert filename == "MORTALITY_30D_cv_summary.json"
