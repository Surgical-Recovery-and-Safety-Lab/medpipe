"""
Tests for MedpipeOrchestrator.ingest_data and its
_get_validation_columns helper.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from medpipe.pipeline.orchestrator import MedpipeOrchestrator


class TestGetValidationColumns:
    """Unit tests for MedpipeOrchestrator._get_validation_columns."""

    def test_get_validation_columns_all_splits_with_deduplication(self):
        """Test retrieving validation columns across test, recalibration,
        and CV with duplicate columns."""
        orchestrator = object.__new__(MedpipeOrchestrator)

        val_cfg = MagicMock()
        val_cfg.test_split.group_column = "OP_YEAR"
        val_cfg.recalibration_split.group_column = "OP_YEAR"  # Duplicate
        val_cfg.cross_validation.group_column = "DHB_NAME"

        orchestrator.config = MagicMock()
        orchestrator.config.workflow.validation = val_cfg

        cols = orchestrator._get_validation_columns()

        assert cols == ["OP_YEAR", "DHB_NAME"]

    def test_get_validation_columns_partial_splits(self):
        """Test retrieving validation columns when only some splits
        specify a group column."""
        orchestrator = object.__new__(MedpipeOrchestrator)

        val_cfg = MagicMock()
        val_cfg.test_split.group_column = "OP_YEAR"
        val_cfg.recalibration_split.group_column = None
        val_cfg.cross_validation = None

        orchestrator.config = MagicMock()
        orchestrator.config.workflow.validation = val_cfg

        cols = orchestrator._get_validation_columns()

        assert cols == ["OP_YEAR"]

    def test_get_validation_columns_missing_validation_config(self):
        """Test returning an empty list when validation config or
        workflow block is missing."""
        orchestrator = object.__new__(MedpipeOrchestrator)
        orchestrator.config = MagicMock()
        orchestrator.config.workflow.validation = None

        cols = orchestrator._get_validation_columns()

        assert cols == []


@patch("medpipe.pipeline.orchestrator.ArtifactManager")
@patch("medpipe.pipeline.orchestrator.get_console_logger")
@patch("medpipe.pipeline.orchestrator.add_file_handler")
class TestIngestData:
    """Unit tests for MedpipeOrchestrator.ingest_data."""

    @patch("medpipe.pipeline.orchestrator.load_data")
    def test_ingest_data_filters_unneeded_columns_successfully(
        self,
        mock_load_data,
        mock_add_handler,
        mock_get_logger,
        mock_artifact_mgr,
        mock_config,
    ):
        """Test successful ingestion of predictors, outcomes, and group columns while filtering extra ones."""
        # 1. Configure predictors, outcomes, and group splits
        mock_config.data.predictors = ["AGE", "BMI"]
        mock_config.data.outcomes = ["MORTALITY_30D"]

        val_config = MagicMock()
        val_config.test_split.group_column = "OP_YEAR"
        val_config.recalibration_split.group_column = (
            "OP_YEAR"  # Tests deduplication logic
        )
        val_config.cross_validation.group_column = "DHB_NAME"
        mock_config.workflow.validation = val_config

        # 2. Raw DataFrame containing required columns + an unneeded column
        raw_df = pd.DataFrame(
            {
                "AGE": [25, 30],
                "BMI": [22.5, 24.1],
                "MORTALITY_30D": [0, 1],
                "OP_YEAR": [2023, 2024],
                "DHB_NAME": ["Auckland", "Wellington"],
                "UNNEEDED_COLUMN": ["X", "Y"],
            }
        )
        mock_load_data.return_value = raw_df

        orchestrator = MedpipeOrchestrator(config=mock_config)
        result = orchestrator.ingest_data()

        mock_load_data.assert_called_once_with("dummy/path/data.csv")
        assert isinstance(result, pd.DataFrame)

        # 3. Verify 'UNNEEDED_COLUMN' was filtered out
        expected_df = pd.DataFrame(
            {
                "AGE": [25, 30],
                "BMI": [22.5, 24.1],
                "MORTALITY_30D": [0, 1],
                "OP_YEAR": [2023, 2024],
                "DHB_NAME": ["Auckland", "Wellington"],
            }
        )
        pd.testing.assert_frame_equal(result, expected_df)

    @patch("medpipe.pipeline.orchestrator.load_data")
    def test_ingest_data_passes_keyword_args(
        self,
        mock_load_data,
        mock_add_handler,
        mock_get_logger,
        mock_artifact_mgr,
        mock_config,
    ):
        """Test successful ingestion of keyword arguments."""
        # 1. Configure predictors, outcomes, and group splits
        mock_config.data.predictors = ["AGE", "BMI"]
        mock_config.data.outcomes = ["MORTALITY_30D"]

        val_config = MagicMock()
        val_config.test_split.group_column = "OP_YEAR"
        val_config.recalibration_split.group_column = (
            "OP_YEAR"  # Tests deduplication logic
        )
        val_config.cross_validation.group_column = "DHB_NAME"
        mock_config.workflow.validation = val_config

        # 2. Raw DataFrame containing required columns + an unneeded column
        raw_df = pd.DataFrame(
            {
                "AGE": [25, 30],
                "BMI": [22.5, 24.1],
                "MORTALITY_30D": [0, 1],
                "OP_YEAR": [2023, 2024],
                "DHB_NAME": ["Auckland", "Wellington"],
                "UNNEEDED_COLUMN": ["X", "Y"],
            }
        )
        mock_load_data.return_value = raw_df

        orchestrator = MedpipeOrchestrator(config=mock_config)
        orchestrator.ingest_data(**{"extra_arg": 1})

        mock_load_data.assert_called_once_with("dummy/path/data.csv", extra_arg=1)

    @patch("medpipe.pipeline.orchestrator.load_data")
    def test_ingest_data_missing_required_column_raises_key_error(
        self,
        mock_load_data,
        mock_add_handler,
        mock_get_logger,
        mock_artifact_mgr,
        mock_config,
    ):
        """Test data ingestion raises KeyError if any configured column is missing from the raw data."""
        mock_config.data.predictors = ["AGE", "BMI"]
        mock_config.data.outcomes = ["MORTALITY_30D"]
        mock_config.workflow.validation = None

        # Input DataFrame missing required 'BMI' column
        incomplete_df = pd.DataFrame(
            {
                "AGE": [25, 30],
                "MORTALITY_30D": [0, 1],
            }
        )
        mock_load_data.return_value = incomplete_df

        orchestrator = MedpipeOrchestrator(config=mock_config)

        with pytest.raises(
            KeyError,
            match="required columns were missing from the dataset: \\['BMI'\\]",
        ):
            orchestrator.ingest_data()

    @patch("medpipe.pipeline.orchestrator.load_data")
    def test_ingest_data_failure_not_dataframe(
        self,
        mock_load_data,
        mock_add_handler,
        mock_get_logger,
        mock_artifact_mgr,
        mock_config,
    ):
        """Test data ingestion raises TypeError if loaded object is not a pandas DataFrame."""
        mock_load_data.return_value = {"AGE": [25, 30]}

        orchestrator = MedpipeOrchestrator(config=mock_config)

        with pytest.raises(TypeError, match="Input data should be a pd.DataFrame"):
            orchestrator.ingest_data()
