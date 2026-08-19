from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from medpipe.pipeline.orchestrator import MedpipeOrchestrator
from medpipe.utils.config import MedpipeConfig

# =============================================================================
# Shared Fixtures
# =============================================================================


@pytest.fixture
def mock_config():
    """Creates a mock MedpipeConfig with necessary attributes."""
    config = MagicMock(spec=MedpipeConfig)

    # Mock Meta section
    config.meta = MagicMock()
    config.meta.project_name = "demo"
    config.meta.verbose = 0

    # Mock Data section
    config.data = MagicMock()
    config.data.path = "dummy/path/data.csv"
    config.data.predictors = ["AGE", "BMI"]

    # Mock Workflow section
    config.workflow = MagicMock()
    config.workflow.preprocessing = MagicMock()
    config.workflow.preprocessing.preprocess = True

    # Mock a single operation
    op1 = MagicMock()
    op1.name = "StandardScaler"
    op1.columns = ["AGE"]
    op1.model_extra = {"with_mean": True}
    config.workflow.preprocessing.operations = [op1]

    # Mock model_dump for reproducibility artifacts
    config.model_dump.return_value = {"mocked": "config"}

    return config


# =============================================================================
# Test Classes per Orchestrator Function / Method
# =============================================================================


@patch("medpipe.pipeline.orchestrator.ArtifactManager")
@patch("medpipe.pipeline.orchestrator.get_console_logger")
@patch("medpipe.pipeline.orchestrator.add_file_handler")
class TestInit:
    """Unit tests for MedpipeOrchestrator.__init__."""

    def test_init_with_config_object(
        self, mock_add_handler, mock_get_logger, mock_artifact_mgr, mock_config
    ):
        """Test initialization when passed a MedpipeConfig object directly."""
        mock_artifact_mgr_instance = mock_artifact_mgr.return_value
        mock_artifact_mgr_instance.create_run_directory.return_value = Path(
            "artifacts/run_1"
        )

        orchestrator = MedpipeOrchestrator(config=mock_config)

        assert orchestrator.config == mock_config
        assert orchestrator.run_dir == Path("artifacts/run_1")
        mock_artifact_mgr_instance.save_env_state.assert_called_once()

    @patch("medpipe.pipeline.orchestrator.read_toml_configuration")
    def test_init_with_string_path(
        self,
        mock_read_toml,
        mock_add_handler,
        mock_get_logger,
        mock_artifact_mgr,
        mock_config,
    ):
        """Test initialization when passed a string path."""
        mock_read_toml.return_value = mock_config

        orchestrator = MedpipeOrchestrator(config="path/to/config.toml")

        mock_read_toml.assert_called_once_with("path/to/config.toml")
        assert orchestrator.config == mock_config

    def test_init_invalid_type_raises_error(
        self, mock_add_handler, mock_get_logger, mock_artifact_mgr
    ):
        """Test initialization fails when passed an invalid config type."""
        with pytest.raises(
            ValueError,
            match="A configuration file or a MedpipeConfig must be specified",
        ):
            MedpipeOrchestrator(config=12345)  # type: ignore


@patch("medpipe.pipeline.orchestrator.ArtifactManager")
@patch("medpipe.pipeline.orchestrator.get_console_logger")
@patch("medpipe.pipeline.orchestrator.add_file_handler")
class TestSaveReproducibilityArtifacts:
    """Unit tests for MedpipeOrchestrator._save_reproducibility_artifacts."""

    def test_save_artifacts_success(
        self, mock_add_handler, mock_get_logger, mock_artifact_mgr, mock_config
    ):
        """Test artifact saving when a valid data path and configuration are provided."""
        mock_artifact_mgr_instance = mock_artifact_mgr.return_value
        mock_artifact_mgr_instance.create_run_directory.return_value = Path(
            "/tmp/run_1"
        )

        # Configure mock_config properties
        mock_config.data.path = "dummy/path/data.csv"
        mock_config.resolved_models = {}

        expected_config_dict = {
            "meta": {"project_name": "demo"},
            "data": {"path": "dummy/path/data.csv"},
        }
        mock_config.model_dump.return_value = expected_config_dict

        # Initializing Orchestrator triggers _save_reproducibility_artifacts() in __init__
        orchestrator = MedpipeOrchestrator(config=mock_config)

        # Verify save_env_state call
        mock_artifact_mgr_instance.save_env_state.assert_called_once_with(
            destination_dir=orchestrator.run_dir,
            config=expected_config_dict,
            dataset_path="dummy/path/data.csv",
        )

    def test_save_artifacts_missing_data_attribute(
        self, mock_add_handler, mock_get_logger, mock_artifact_mgr
    ):
        """Test artifact saving handles a config entirely missing the data block."""
        mock_artifact_mgr_instance = mock_artifact_mgr.return_value
        mock_artifact_mgr_instance.create_run_directory.return_value = Path(
            "/tmp/run_1"
        )

        mock_config_no_data = MagicMock(spec=MedpipeConfig)
        mock_config_no_data.resolved_models = {}
        mock_config_no_data.meta = MagicMock()
        mock_config_no_data.meta.verbose = 0
        # Explicitly remove the data attribute to trigger dataset_path=None fallback
        del mock_config_no_data.data
        mock_config_no_data.model_dump.return_value = {"workflow": {}}

        # Initializing Orchestrator triggers _save_reproducibility_artifacts() in __init__
        orchestrator = MedpipeOrchestrator(config=mock_config_no_data)

        # Verify save_env_state was called with dataset_path=None
        mock_artifact_mgr_instance.save_env_state.assert_called_once_with(
            destination_dir=orchestrator.run_dir,
            config={"workflow": {}},
            dataset_path=None,
        )


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

        # Verify group columns ("OP_YEAR", "DHB_NAME") were dropped from feature matrices
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
        """Test data preparation handles missing recalibration and cross-validation configs gracefully."""
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

        X_train, y_train, X_recal, y_recal, X_test, y_test, groups = (
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
    def test_prepare_data_missing_cv_group_column_raises_key_error(
        self,
        mock_extract_labels,
        mock_split_data,
        mock_add_handler,
        mock_get_logger,
        mock_artifact_mgr,
        mock_config,
    ):
        """Test that KeyError is raised when the cross_validation group column is missing from training features."""
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
            match="Cross-validation group column 'NON_EXISTENT_COLUMN' was not found in dataset columns.",
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


@patch("medpipe.pipeline.orchestrator.ArtifactManager")
@patch("medpipe.pipeline.orchestrator.get_console_logger")
@patch("medpipe.pipeline.orchestrator.add_file_handler")
class TestBuildPreprocessor:
    """Unit tests for MedpipeOrchestrator.build_preprocessor."""

    def test_build_preprocessor_success(
        self, mock_add_handler, mock_get_logger, mock_artifact_mgr, mock_config
    ):
        """Test that a valid config builds an sklearn Pipeline."""
        orchestrator = MedpipeOrchestrator(config=mock_config)

        pipeline = orchestrator.build_preprocessor()

        assert pipeline is not None
        assert isinstance(pipeline, Pipeline)
        assert len(pipeline.steps) == 1
        assert pipeline.steps[0][0] == "transformer_1"

    def test_build_preprocessor_disabled(
        self, mock_add_handler, mock_get_logger, mock_artifact_mgr, mock_config
    ):
        """Test that returning None happens when preprocess is False."""
        mock_config.workflow.preprocessing.preprocess = False

        orchestrator = MedpipeOrchestrator(config=mock_config)
        pipeline = orchestrator.build_preprocessor()

        assert pipeline is None

    def test_build_preprocessor_no_operations(
        self, mock_add_handler, mock_get_logger, mock_artifact_mgr, mock_config
    ):
        """Test that an empty operations list but preprocess=True returns an empty pipeline."""
        mock_config.workflow.preprocessing.preprocess = True
        mock_config.workflow.preprocessing.operations = []

        orchestrator = MedpipeOrchestrator(config=mock_config)
        pipeline = orchestrator.build_preprocessor()

        assert pipeline is not None
        assert isinstance(pipeline, Pipeline)
        assert len(pipeline.steps) == 0

    def test_build_preprocessor_dict_is_none(
        self, mock_add_handler, mock_get_logger, mock_artifact_mgr, mock_config
    ):
        """Test pipeline building returns None if the preprocessing config block is missing."""
        mock_config.workflow.preprocessing = None

        orchestrator = MedpipeOrchestrator(config=mock_config)
        pipeline = orchestrator.build_preprocessor()

        assert pipeline is None

    def test_build_preprocessor_invalid_operation_raises_error(
        self, mock_add_handler, mock_get_logger, mock_artifact_mgr, mock_config
    ):
        """Test that an invalid operation name in the config bubbles up a ValueError."""
        mock_config.workflow.preprocessing.operations[0].name = "FakeMagicalTransformer"

        orchestrator = MedpipeOrchestrator(config=mock_config)

        with pytest.raises(
            ValueError,
            match="was not found in the custom registry or fallback modules.",
        ):
            orchestrator.build_preprocessor()


class TestExtractStratumSubgroup:
    """Unit test suite for MedpipeOrchestrator.extract_stratum_subgroup."""

    @pytest.fixture
    def mock_orchestrator(self) -> MedpipeOrchestrator:
        """Create a lightweight MedpipeOrchestrator instance with mocked dependencies."""
        orchestrator = object.__new__(MedpipeOrchestrator)
        orchestrator.logger = MagicMock()
        return orchestrator

    @pytest.fixture
    def sample_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Provide aligned feature (X) and label (y) DataFrames."""
        indices = pd.Index([101, 102, 103, 104, 105])
        X = pd.DataFrame(
            {
                "AGE": [20, 45, 70, 35, 80],
                "SEX": ["F", "M", "F", "M", "F"],
                "FEAT1": [1.1, 2.2, 3.3, 4.4, 5.5],
            },
            index=indices,
        )
        y = pd.DataFrame(
            {
                "readmission": [0, 1, 0, 1, 1],
                "mortality": [0, 0, 1, 0, 1],
            },
            index=indices,
        )
        return X, y

    def test_extract_subgroup_with_x_and_y_success(
        self,
        mock_orchestrator: MedpipeOrchestrator,
        sample_data: tuple[pd.DataFrame, pd.DataFrame],
    ) -> None:
        """Verify slicing both features (X) and target labels (y) preserves alignment and copy independence."""
        X, y = sample_data

        X_sub, y_sub = mock_orchestrator.extract_stratum_subgroup(
            X=X, column="SEX", group="M", y=y
        )

        assert isinstance(X_sub, pd.DataFrame)
        assert isinstance(y_sub, pd.DataFrame)
        assert len(X_sub) == 2
        assert len(y_sub) == 2

        assert list(X_sub.index) == [102, 104]
        assert list(y_sub.index) == [102, 104]

        # Ensure returned objects are deep copies
        X_sub.loc[102, "FEAT1"] = 999.0
        assert X.loc[102, "FEAT1"] == 2.2

    def test_extract_subgroup_without_y_returns_none(
        self,
        mock_orchestrator: MedpipeOrchestrator,
        sample_data: tuple[pd.DataFrame, pd.DataFrame],
    ) -> None:
        """Verify call with y=None returns (X_subgroup, None)."""
        X, _ = sample_data

        X_sub, y_sub = mock_orchestrator.extract_stratum_subgroup(
            X=X, column="AGE", group=(18, 50), y=None
        )

        assert len(X_sub) == 3
        assert y_sub is None

    def test_extract_subgroup_range_tuple(
        self,
        mock_orchestrator: MedpipeOrchestrator,
        sample_data: tuple[pd.DataFrame, pd.DataFrame],
    ) -> None:
        """Verify continuous numerical group resolution using tuple bounds."""
        X, y = sample_data

        X_sub, y_sub = mock_orchestrator.extract_stratum_subgroup(
            X=X, column="AGE", group=(60, 90), y=y
        )

        assert y_sub is not None
        assert len(X_sub) == 2
        assert list(X_sub.index) == [103, 105]
        assert list(y_sub.index) == [103, 105]

    def test_zero_matches_logs_warning_and_returns_empty_dataframes(
        self,
        mock_orchestrator: MedpipeOrchestrator,
        sample_data: tuple[pd.DataFrame, pd.DataFrame],
    ) -> None:
        """Verify zero matched rows triggers logger warning and returns empty DataFrames."""
        X, y = sample_data

        X_sub, y_sub = mock_orchestrator.extract_stratum_subgroup(
            X=X, column="SEX", group="UNKNOWN", y=y
        )

        assert y_sub is not None
        assert len(X_sub) == 0
        assert len(y_sub) == 0
        assert list(X_sub.columns) == list(X.columns)
        assert list(y_sub.columns) == list(y.columns)

        mock_orchestrator.logger.warning.assert_called_once()  # type: ignore
        log_msg = mock_orchestrator.logger.warning.call_args[0][0]  # type: ignore
        assert "returned 0 samples" in log_msg

    def test_missing_column_raises_key_error(
        self,
        mock_orchestrator: MedpipeOrchestrator,
        sample_data: tuple[pd.DataFrame, pd.DataFrame],
    ) -> None:
        """Verify KeyError is raised when the stratification column is missing from X."""
        X, y = sample_data

        with pytest.raises(KeyError, match="Stratum column 'INVALID_COL' not found"):
            mock_orchestrator.extract_stratum_subgroup(
                X=X, column="INVALID_COL", group="M", y=y
            )


class TestGetSubgroupSpecs:
    """Unit test suite for MedpipeOrchestrator.get_subgroup_specs."""

    def test_get_subgroup_specs_with_strata_and_custom_groups(self):
        """Test resolving subgroup specs when both strata and custom range
        groups are defined."""
        orchestrator = object.__new__(MedpipeOrchestrator)

        fairness_cfg = MagicMock()
        fairness_cfg.strata = ["SEX", "AGE"]
        fairness_cfg.groups = {"AGE": [[18, 50], [51, 120]]}

        orchestrator.config = MagicMock()
        orchestrator.config.workflow.evaluation.fairness = fairness_cfg

        specs = orchestrator.get_subgroup_specs()

        assert specs == {
            "SEX": "SEX",
            "AGE": [[18, 50], [51, 120]],
        }

    def test_get_subgroup_specs_strata_only(self):
        """Test resolving subgroup specs when strata are present without
        custom groups."""
        orchestrator = object.__new__(MedpipeOrchestrator)

        fairness_cfg = MagicMock()
        fairness_cfg.strata = ["SEX", "ETHNICITY"]
        fairness_cfg.groups = None

        orchestrator.config = MagicMock()
        orchestrator.config.workflow.evaluation.fairness = fairness_cfg

        specs = orchestrator.get_subgroup_specs()

        assert specs == {
            "SEX": "SEX",
            "ETHNICITY": "ETHNICITY",
        }

    def test_get_subgroup_specs_missing_fairness_config(self):
        """Test returning empty dict when fairness config or strata list
        is missing."""
        orchestrator = object.__new__(MedpipeOrchestrator)
        orchestrator.config = MagicMock()
        orchestrator.config.workflow.evaluation.fairness = None

        specs = orchestrator.get_subgroup_specs()

        assert specs == {}

    def test_get_subgroup_specs_missing_workflow_or_evaluation(self):
        """Test returning empty dict when workflow or evaluation section is
        missing entirely."""
        orchestrator = object.__new__(MedpipeOrchestrator)
        orchestrator.config = MagicMock()
        orchestrator.config.workflow = None

        specs = orchestrator.get_subgroup_specs()

        assert specs == {}


class TestCheckOperation:
    """Unit test suite for MedpipeOrchestrator._check_operation."""

    def test_check_operation_registered_class(self):
        """Test resolving valid transformer classes from registry."""
        orchestrator = object.__new__(MedpipeOrchestrator)

        op_cls = orchestrator._check_operation("StandardScaler")
        assert op_cls.__name__ == "StandardScaler"

    def test_check_operation_invalid_class_raises_value_error(self):
        """Test invalid operation name raises ValueError."""
        orchestrator = object.__new__(MedpipeOrchestrator)

        with pytest.raises(
            ValueError, match="was not found in the custom registry or fallback modules"
        ):
            orchestrator._check_operation("NonExistentTransformer")
