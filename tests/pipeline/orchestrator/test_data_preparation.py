"""
Tests for MedpipeOrchestrator.prepare_data.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from medpipe.pipeline.orchestrator import DataSplits, MedpipeOrchestrator


@patch("medpipe.pipeline.orchestrator.ArtifactManager")
@patch("medpipe.pipeline.orchestrator.get_console_logger")
@patch("medpipe.pipeline.orchestrator.add_file_handler")
class TestPrepareData:
    """Unit tests for MedpipeOrchestrator.prepare_data."""

    @patch("medpipe.pipeline.orchestrator.split_data")
    @patch("medpipe.pipeline.orchestrator.extract_labels")
    def test_prepare_data_with_recalibration_and_cv_groups(
        self,
        mock_extract_labels,
        mock_split_data,
        mock_add_handler,
        mock_get_logger,
        mock_artifact_mgr,
        mock_config,
    ):
        """Test full data preparation pipeline with test split, recalibration split,

        CV group extraction, and validation column dropping.
        """
        mock_config.data.outcomes = ["MORTALITY_30D"]

        # 1. Configure validation splits and CV group column
        val_config = MagicMock()
        val_config.test_split.strategy = "group"
        val_config.test_split.group_column = "OP_YEAR"
        val_config.recalibration_split.strategy = "group"
        val_config.recalibration_split.group_column = "OP_YEAR"
        val_config.cross_validation.group_column = "DHB_NAME"
        mock_config.workflow.validation = val_config

        orchestrator = MedpipeOrchestrator(config=mock_config)

        # 2. Mock ingest_data return value
        raw_data = pd.DataFrame(
            {
                "AGE": [25, 30, 45, 60],
                "DHB_NAME": ["Auckland", "Wellington", "Auckland", "Wellington"],
                "OP_YEAR": [2023, 2023, 2024, 2024],
                "MORTALITY_30D": [0, 1, 0, 1],
            }
        )
        orchestrator.ingest_data = MagicMock(return_value=raw_data)

        # 3. Setup mock side effects for extract_labels
        X_all = raw_data[["AGE", "DHB_NAME", "OP_YEAR"]]
        y_all_arr = np.array([[0], [1], [0], [1]])

        X_train_no_cv = pd.DataFrame(
            {"AGE": [25, 30], "OP_YEAR": [2023, 2023]}, index=pd.Index([0, 1])
        )
        expected_groups = np.array(["Auckland", "Wellington"])

        mock_extract_labels.side_effect = [
            (X_all, y_all_arr),  # Initial outcome extraction call
            (X_train_no_cv, expected_groups),  # CV Group extraction call
        ]

        # 4. Setup mock side effects for split_data
        X_temp = pd.DataFrame(
            {
                "AGE": [25, 30, 45],
                "DHB_NAME": ["Auckland", "Wellington", "Auckland"],
                "OP_YEAR": [2023, 2023, 2023],
            },
            index=pd.Index([0, 1, 2]),
        )
        y_temp_arr = np.array([[0], [1], [0]])
        X_test_df = pd.DataFrame(
            {"AGE": [60], "DHB_NAME": ["Wellington"], "OP_YEAR": [2024]},
            index=pd.Index([3]),
        )
        y_test_arr = np.array([[1]])

        X_train_df = pd.DataFrame(
            {
                "AGE": [25, 30],
                "DHB_NAME": ["Auckland", "Wellington"],
                "OP_YEAR": [2023, 2023],
            },
            index=pd.Index([0, 1]),
        )
        y_train_arr = np.array([[0], [1]])
        X_recal_df = pd.DataFrame(
            {"AGE": [45], "DHB_NAME": ["Auckland"], "OP_YEAR": [2023]},
            index=pd.Index([2]),
        )
        y_recal_arr = np.array([[0]])

        mock_split_data.side_effect = [
            (X_temp, y_temp_arr, X_test_df, y_test_arr),  # Test split call
            (
                X_train_df,
                y_train_arr,
                X_recal_df,
                y_recal_arr,
            ),  # Recalibration split call
        ]

        # Execute
        X_train, y_train, X_recal, y_recal, X_test, y_test, groups = (
            orchestrator.prepare_data()
        )

        # Assertions
        assert mock_split_data.call_count == 2
        assert mock_extract_labels.call_count == 2
        assert y_recal is not None
        assert X_recal is not None

        # Verify outcome column alignment
        assert list(y_train.columns) == ["MORTALITY_30D"]
        assert list(y_recal.columns) == ["MORTALITY_30D"]
        assert list(y_test.columns) == ["MORTALITY_30D"]

        # Verify group columns ("OP_YEAR", "DHB_NAME") were dropped from
        # feature matrices
        assert "OP_YEAR" not in X_train.columns
        assert "DHB_NAME" not in X_train.columns
        assert "OP_YEAR" not in X_test.columns
        assert "DHB_NAME" not in X_test.columns
        assert "OP_YEAR" not in X_recal.columns
        assert "DHB_NAME" not in X_recal.columns

        # Verify predictors are retained
        assert "AGE" in X_train.columns
        assert "AGE" in X_test.columns
        assert "AGE" in X_recal.columns

        # Verify CV groups array extraction
        np.testing.assert_array_equal(groups, expected_groups)

    @patch("medpipe.pipeline.orchestrator.split_data")
    @patch("medpipe.pipeline.orchestrator.extract_labels")
    def test_prepare_data_without_recalibration(
        self,
        mock_extract_labels,
        mock_split_data,
        mock_add_handler,
        mock_get_logger,
        mock_artifact_mgr,
        mock_config,
    ):
        """Test data preparation handles missing recalibration and
        cross-validation configs gracefully."""
        mock_config.data.outcomes = ["MORTALITY_30D"]

        val_config = MagicMock()
        val_config.test_split.strategy = "random"
        val_config.test_split.group_column = None
        val_config.recalibration_split = None
        val_config.cross_validation = None
        mock_config.workflow.validation = val_config

        orchestrator = MedpipeOrchestrator(config=mock_config)

        raw_data = pd.DataFrame({"AGE": [25, 30, 45], "MORTALITY_30D": [0, 1, 0]})
        orchestrator.ingest_data = MagicMock(return_value=raw_data)

        X_all = pd.DataFrame({"AGE": [25, 30, 45]})
        y_all_arr = np.array([[0], [1], [0]])
        mock_extract_labels.return_value = (X_all, y_all_arr)

        X_temp = pd.DataFrame({"AGE": [25, 30]}, index=pd.Index([0, 1]))
        y_temp_arr = np.array([[0], [1]])
        X_test_df = pd.DataFrame({"AGE": [45]}, index=pd.Index([2]))
        y_test_arr = np.array([[0]])

        mock_split_data.return_value = (X_temp, y_temp_arr, X_test_df, y_test_arr)

        X_train, _y_train, X_recal, y_recal, X_test, _y_test, groups = (
            orchestrator.prepare_data()
        )

        assert mock_split_data.call_count == 1
        assert mock_extract_labels.call_count == 1

        assert X_recal is None
        assert y_recal is None
        assert groups is None

        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_test, pd.DataFrame)
        assert list(X_train.columns) == ["AGE"]
        assert list(X_test.columns) == ["AGE"]

    @patch("medpipe.pipeline.orchestrator.split_data")
    @patch("medpipe.pipeline.orchestrator.extract_labels")
    def test_prepare_data_drops_group_column_without_recalibration(
        self,
        mock_extract_labels,
        mock_split_data,
        mock_add_handler,
        mock_get_logger,
        mock_artifact_mgr,
        mock_config,
    ):
        """Test that the test-split's group column is dropped from the
        train/test features even when there is no recalibration split
        (X_recal stays None) — the column-dropping block's "no recal"
        branch, which the other dropping test doesn't exercise since it
        always has a recalibration split active."""
        mock_config.data.outcomes = ["MORTALITY_30D"]

        val_config = MagicMock()
        val_config.test_split.strategy = "group"
        val_config.test_split.group_column = "OP_YEAR"
        val_config.recalibration_split = None
        val_config.cross_validation = None
        mock_config.workflow.validation = val_config

        orchestrator = MedpipeOrchestrator(config=mock_config)

        raw_data = pd.DataFrame(
            {
                "AGE": [25, 30, 45],
                "OP_YEAR": [2023, 2023, 2024],
                "MORTALITY_30D": [0, 1, 0],
            }
        )
        orchestrator.ingest_data = MagicMock(return_value=raw_data)

        X_all = pd.DataFrame({"AGE": [25, 30, 45], "OP_YEAR": [2023, 2023, 2024]})
        y_all_arr = np.array([[0], [1], [0]])
        mock_extract_labels.return_value = (X_all, y_all_arr)

        X_temp = pd.DataFrame(
            {"AGE": [25, 30], "OP_YEAR": [2023, 2023]}, index=pd.Index([0, 1])
        )
        y_temp_arr = np.array([[0], [1]])
        X_test_df = pd.DataFrame({"AGE": [45], "OP_YEAR": [2024]}, index=pd.Index([2]))
        y_test_arr = np.array([[0]])

        mock_split_data.return_value = (X_temp, y_temp_arr, X_test_df, y_test_arr)

        X_train, _y_train, X_recal, _y_recal, X_test, _y_test, _groups = (
            orchestrator.prepare_data()
        )

        assert X_recal is None
        assert "OP_YEAR" not in X_train.columns
        assert "OP_YEAR" not in X_test.columns
        assert "AGE" in X_train.columns
        assert "AGE" in X_test.columns

    @patch("medpipe.pipeline.orchestrator.split_data")
    @patch("medpipe.pipeline.orchestrator.extract_labels")
    def test_prepare_data_passes_random_state_to_split_data(
        self,
        mock_extract_labels,
        mock_split_data,
        mock_add_handler,
        mock_get_logger,
        mock_artifact_mgr,
        mock_config,
    ):
        """Test that both the test-split and recalibration-split calls to
        split_data receive workflow.random_state, so the resolved seed
        actually reaches the underlying random splitting (regression test
        for a prior bug where random_state was never threaded through)."""
        mock_config.data.outcomes = ["MORTALITY_30D"]
        mock_config.workflow.random_state = 7

        val_config = MagicMock()
        val_config.test_split.strategy = "random"
        val_config.test_split.group_column = None
        val_config.test_split.test_size = 0.25
        val_config.recalibration_split.strategy = "random"
        val_config.recalibration_split.group_column = None
        val_config.recalibration_split.recalibration_size = 0.1
        val_config.cross_validation = None
        mock_config.workflow.validation = val_config

        orchestrator = MedpipeOrchestrator(config=mock_config)

        raw_data = pd.DataFrame(
            {"AGE": [25, 30, 45, 60], "MORTALITY_30D": [0, 1, 0, 1]}
        )
        orchestrator.ingest_data = MagicMock(return_value=raw_data)

        X_all = pd.DataFrame({"AGE": [25, 30, 45, 60]})
        y_all_arr = np.array([[0], [1], [0], [1]])
        mock_extract_labels.return_value = (X_all, y_all_arr)

        X_temp = pd.DataFrame({"AGE": [25, 30, 45]}, index=pd.Index([0, 1, 2]))
        y_temp_arr = np.array([[0], [1], [0]])
        X_test_df = pd.DataFrame({"AGE": [60]}, index=pd.Index([3]))
        y_test_arr = np.array([[1]])

        X_train_df = pd.DataFrame({"AGE": [25, 30]}, index=pd.Index([0, 1]))
        y_train_arr = np.array([[0], [1]])
        X_recal_df = pd.DataFrame({"AGE": [45]}, index=pd.Index([2]))
        y_recal_arr = np.array([[0]])

        mock_split_data.side_effect = [
            (X_temp, y_temp_arr, X_test_df, y_test_arr),
            (X_train_df, y_train_arr, X_recal_df, y_recal_arr),
        ]

        orchestrator.prepare_data()

        assert mock_split_data.call_count == 2
        for actual_call in mock_split_data.call_args_list:
            assert actual_call.kwargs["random_state"] == 7

    @patch("medpipe.pipeline.orchestrator.split_data")
    @patch("medpipe.pipeline.orchestrator.extract_labels")
    def test_prepare_data_missing_cv_group_column_raises_key_error(
        self,
        mock_extract_labels,
        mock_split_data,
        mock_add_handler,
        mock_get_logger,
        mock_artifact_mgr,
        mock_config,
    ):
        """Test that KeyError is raised when the cross_validation group
        column is missing from training features."""
        mock_config.data.outcomes = ["MORTALITY_30D"]

        val_config = MagicMock()
        val_config.test_split.strategy = "random"
        val_config.test_split.group_column = None
        val_config.recalibration_split = None
        val_config.cross_validation.group_column = "NON_EXISTENT_COLUMN"
        mock_config.workflow.validation = val_config

        orchestrator = MedpipeOrchestrator(config=mock_config)
        orchestrator.ingest_data = MagicMock(
            return_value=pd.DataFrame({"AGE": [25, 30]})
        )

        mock_extract_labels.return_value = (
            pd.DataFrame({"AGE": [25, 30]}),
            np.array([[0], [1]]),
        )
        mock_split_data.return_value = (
            pd.DataFrame({"AGE": [25, 30]}),
            np.array([[0], [1]]),
            pd.DataFrame(),
            np.array([]),
        )

        with pytest.raises(
            KeyError,
            match=(
                r"Cross-validation group column 'NON_EXISTENT_COLUMN' "
                r"was not found in dataset columns\."
            ),
        ):
            orchestrator.prepare_data()

    def test_prepare_data_missing_validation_raises_value_error(
        self, mock_add_handler, mock_get_logger, mock_artifact_mgr, mock_config
    ):
        """Test that missing validation configuration raises a ValueError."""
        mock_config.workflow.validation = None
        mock_config.data.outcomes = ["MORTALITY_30D"]
        orchestrator = MedpipeOrchestrator(config=mock_config)
        orchestrator.ingest_data = MagicMock(
            return_value=pd.DataFrame({"MORTALITY_30D": [0]})
        )

        with pytest.raises(
            ValueError, match="Validation configuration is missing from workflow"
        ):
            orchestrator.prepare_data()

    @patch("medpipe.pipeline.orchestrator.split_data")
    @patch("medpipe.pipeline.orchestrator.extract_labels")
    def test_prepare_data_populates_splits_property(
        self,
        mock_extract_labels,
        mock_split_data,
        mock_add_handler,
        mock_get_logger,
        mock_artifact_mgr,
        mock_config,
    ):
        """Test that prepare_data properly creates and exposes the
        DataSplits dataclass."""
        mock_config.data.outcomes = ["MORTALITY_30D"]

        val_config = MagicMock()
        val_config.test_split.strategy = "random"
        val_config.test_split.group_column = None
        val_config.recalibration_split = None
        val_config.cross_validation = None
        mock_config.workflow.validation = val_config

        orchestrator = MedpipeOrchestrator(config=mock_config)

        raw_data = pd.DataFrame({"AGE": [25, 30, 45], "MORTALITY_30D": [0, 1, 0]})
        orchestrator.ingest_data = MagicMock(return_value=raw_data)

        X_all = pd.DataFrame({"AGE": [25, 30, 45]})
        y_all_arr = np.array([[0], [1], [0]])
        mock_extract_labels.return_value = (X_all, y_all_arr)

        X_temp = pd.DataFrame({"AGE": [25, 30]}, index=pd.Index([0, 1]))
        y_temp_arr = np.array([[0], [1]])
        X_test_df = pd.DataFrame({"AGE": [45]}, index=pd.Index([2]))
        y_test_arr = np.array([[0]])

        mock_split_data.return_value = (X_temp, y_temp_arr, X_test_df, y_test_arr)

        # Call prepare_data to trigger _splits population
        orchestrator.prepare_data()

        # Verify splits property returns a populated DataSplits instance
        splits = orchestrator.splits
        assert isinstance(splits, DataSplits)
        pd.testing.assert_frame_equal(splits.X_train, X_temp)
        pd.testing.assert_frame_equal(splits.X_test, X_test_df)
        assert splits.X_recal is None
        assert splits.y_recal is None
        assert splits.groups_train is None
