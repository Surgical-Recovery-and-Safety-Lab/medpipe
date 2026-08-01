from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from medpipe.pipeline.orchestrator import MedpipeOrchestrator
from medpipe.utils.config import MedpipeConfig


@pytest.fixture
def mock_config():
    """Creates a mock MedpipeConfig with necessary attributes."""
    config = MagicMock(spec=MedpipeConfig)

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


@patch("medpipe.pipeline.orchestrator.ArtifactManager")
@patch("medpipe.pipeline.orchestrator.get_console_logger")
@patch("medpipe.pipeline.orchestrator.add_file_handler")
class TestMedpipeOrchestrator:

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
        mock_artifact_mgr_instance.save_resolved_config.assert_called_once()
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
            MedpipeOrchestrator(config=12345)

    @patch("medpipe.pipeline.orchestrator.load_data")
    def test_ingest_data_success(
        self,
        mock_load_data,
        mock_add_handler,
        mock_get_logger,
        mock_artifact_mgr,
        mock_config,
    ):
        """Test successful data ingestion returning a DataFrame."""
        mock_df = pd.DataFrame({"AGE": [25, 30], "BMI": [22.5, 24.1]})
        mock_load_data.return_value = mock_df

        orchestrator = MedpipeOrchestrator(config=mock_config)
        result = orchestrator.ingest_data()

        mock_load_data.assert_called_once_with("dummy/path/data.csv")
        assert isinstance(result, pd.DataFrame)
        pd.testing.assert_frame_equal(result, mock_df)

    @patch("medpipe.pipeline.orchestrator.load_data")
    def test_ingest_data_failure_not_dataframe(
        self,
        mock_load_data,
        mock_add_handler,
        mock_get_logger,
        mock_artifact_mgr,
        mock_config,
    ):
        """Test data ingestion raises TypeError if data is not a DataFrame."""
        mock_load_data.return_value = {
            "AGE": [25, 30]
        }  # Returns dict instead of DataFrame

        orchestrator = MedpipeOrchestrator(config=mock_config)

        with pytest.raises(TypeError, match="Input data should be a pd.DataFrame"):
            orchestrator.ingest_data()

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

    @patch.object(MedpipeOrchestrator, "resolve_model_configurations")
    def test_save_artifacts_missing_data_attribute(
        self, mock_resolve, mock_add_handler, mock_get_logger, mock_artifact_mgr
    ):
        """Test artifact saving handles a config entirely missing the data block."""
        mock_config_no_data = MagicMock(spec=MedpipeConfig)
        # Explicitly remove the data attribute to trigger the artifact manager fallback
        del mock_config_no_data.data
        mock_config_no_data.model_dump.return_value = {"workflow": {}}

        mock_artifact_mgr_instance = mock_artifact_mgr.return_value

        # Initialize orchestrator (mock_resolve prevents it from crashing on missing data)
        orchestrator = MedpipeOrchestrator(config=mock_config_no_data)

        # Verify save_env_state was called with dataset_path=None
        mock_artifact_mgr_instance.save_env_state.assert_called_once_with(
            destination_dir=orchestrator.run_dir,
            config={"workflow": {}},
            dataset_path=None,
        )

    def test_build_preprocessor_dict_is_none(
        self, mock_add_handler, mock_get_logger, mock_artifact_mgr, mock_config
    ):
        """Test pipeline building returns None if the preprocessing config
        block is missing."""
        mock_config.workflow.preprocessing = None

        orchestrator = MedpipeOrchestrator(config=mock_config)
        pipeline = orchestrator.build_preprocessor()

        assert pipeline is None

    def test_build_preprocessor_invalid_operation_raises_error(
        self, mock_add_handler, mock_get_logger, mock_artifact_mgr, mock_config
    ):
        """Test that an invalid operation name in the config bubbles up
        the Registry's ValueError."""
        # Inject a bad operation name into the mock configuration
        mock_config.workflow.preprocessing.operations[0].name = "FakeMagicalTransformer"

        orchestrator = MedpipeOrchestrator(config=mock_config)

        with pytest.raises(
            ValueError,
            match="was not found in the custom registry or fallback modules.",
        ):
            orchestrator.build_preprocessor()

    def test_resolve_model_configurations_success(
        self, mock_add_handler, mock_get_logger, mock_artifact_mgr, mock_config
    ):
        """
        Test that default models and outcome overrides are correctly merged.
        """
        # Set up outcomes
        mock_config.data.outcomes = ["MORTALITY_30D", "ANY_COMP"]

        # Mock the configuration dictionary structure based on medpipe.toml
        mock_config.model_dump.return_value = {
            "default_model": {
                "algorithm": "HistGradientBoostingClassifier",
                "hyperparameters": {"learning_rate": 0.1, "loss": "log_loss"},
            },
            "outcome_overrides": {
                "ANY_COMP": {
                    "algorithm": "RandomForestClassifier",
                    "hyperparameters": {"n_estimators": 200, "max_depth": 10},
                }
            },
        }

        orchestrator = MedpipeOrchestrator(config=mock_config)
        resolved = orchestrator.resolve_model_configurations()

        assert "MORTALITY_30D" in resolved
        assert "ANY_COMP" in resolved

        # MORTALITY_30D should use pure defaults
        assert (
            resolved["MORTALITY_30D"]["algorithm"] == "HistGradientBoostingClassifier"
        )
        assert resolved["MORTALITY_30D"]["hyperparameters"]["learning_rate"] == 0.1

        # ANY_COMP should use overridden algorithm and hyperparameters
        assert resolved["ANY_COMP"]["algorithm"] == "RandomForestClassifier"
        assert resolved["ANY_COMP"]["hyperparameters"]["n_estimators"] == 200

    def test_resolve_model_configurations_no_overrides(
        self, mock_add_handler, mock_get_logger, mock_artifact_mgr, mock_config
    ):
        """
        Test configuration resolution when no overrides are provided.
        """
        mock_config.data.outcomes = ["MORTALITY_30D"]
        mock_config.model_dump.return_value = {
            "default_model": {"algorithm": "HistGradientBoostingClassifier"}
        }

        orchestrator = MedpipeOrchestrator(config=mock_config)
        resolved = orchestrator.resolve_model_configurations()

        assert (
            resolved["MORTALITY_30D"]["algorithm"] == "HistGradientBoostingClassifier"
        )

    @patch("medpipe.pipeline.orchestrator.split_data")
    @patch("medpipe.pipeline.orchestrator.extract_labels")
    def test_prepare_data_with_recalibration(
        self,
        mock_extract,
        mock_split,
        mock_add_handler,
        mock_get_logger,
        mock_artifact_mgr,
        mock_config,
    ):
        """
        Test data preparation when both test and recalibration splits are configured.
        """
        # Setup mock configuration
        mock_config.data.outcomes = ["MORTALITY_30D"]
        val_config = MagicMock()
        val_config.test_split.strategy = "group"
        val_config.recalibration_split.strategy = "group"
        mock_config.workflow.validation = val_config

        orchestrator = MedpipeOrchestrator(config=mock_config)

        # Mocking ingest_data manually on the instance to avoid patching load_data again
        orchestrator.ingest_data = MagicMock(
            return_value=pd.DataFrame({"A": [1, 2, 3, 4]})
        )

        # Mock extract_labels return
        mock_extract.return_value = (
            pd.DataFrame({"A": [1, 2, 3, 4]}),
            np.array([[0], [1], [0], [1]]),
        )

        # Mock split_data returns. It is called twice:
        # 1st call (Test Split): returns (X_temp, y_temp_arr, X_test, y_test_arr)
        # 2nd call (Recal Split): returns (X_train, y_train_arr, X_recal, y_recal_arr)
        mock_split.side_effect = [
            (
                pd.DataFrame(index=[0, 1, 2]),
                np.array([[0], [1], [0]]),
                pd.DataFrame(index=[3]),
                np.array([[1]]),
            ),
            (
                pd.DataFrame(index=[0, 1]),
                np.array([[0], [1]]),
                pd.DataFrame(index=[2]),
                np.array([[0]]),
            ),
        ]

        X_train, y_train, X_recal, y_recal, X_test, y_test = orchestrator.prepare_data()

        # Assertions
        assert mock_split.call_count == 2
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(y_train, pd.DataFrame)
        assert isinstance(X_recal, pd.DataFrame)
        assert isinstance(y_recal, pd.DataFrame)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_test, pd.DataFrame)

        # Ensure labels are rebuilt with correct outcome columns
        assert list(y_train.columns) == ["MORTALITY_30D"]
        assert list(y_recal.columns) == ["MORTALITY_30D"]
        assert list(y_test.columns) == ["MORTALITY_30D"]

    @patch("medpipe.pipeline.orchestrator.split_data")
    @patch("medpipe.pipeline.orchestrator.extract_labels")
    def test_prepare_data_without_recalibration(
        self,
        mock_extract,
        mock_split,
        mock_add_handler,
        mock_get_logger,
        mock_artifact_mgr,
        mock_config,
    ):
        """
        Test data preparation handles missing recalibration configuration gracefully.
        """
        # Setup mock configuration without recalibration split
        mock_config.data.outcomes = ["MORTALITY_30D"]
        val_config = MagicMock()
        val_config.test_split.strategy = "group"
        val_config.recalibration_split = None  # No recalibration
        mock_config.workflow.validation = val_config

        orchestrator = MedpipeOrchestrator(config=mock_config)
        orchestrator.ingest_data = MagicMock(return_value=pd.DataFrame())

        mock_extract.return_value = (pd.DataFrame(), np.array([]))

        # split_data will only be called once
        mock_split.return_value = (
            pd.DataFrame(index=[0, 1]),
            np.array([[0], [1]]),
            pd.DataFrame(index=[2]),
            np.array([[0]]),
        )

        X_train, y_train, X_recal, y_recal, X_test, y_test = orchestrator.prepare_data()

        assert mock_split.call_count == 1
        assert X_recal is None
        assert y_recal is None
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_test, pd.DataFrame)

    def test_prepare_data_missing_validation_raises_error(
        self, mock_add_handler, mock_get_logger, mock_artifact_mgr, mock_config
    ):
        """
        Test that missing validation configuration raises a ValueError.
        """
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

    def test_resolve_model_configurations_missing_default_model(
        self, mock_add_handler, mock_get_logger, mock_artifact_mgr, mock_config
    ):
        """
        Test configuration resolution when default_model is completely absent.
        """
        mock_config.data.outcomes = ["ANY_COMP"]

        # Missing 'default_model' entirely
        mock_config.model_dump.return_value = {
            "outcome_overrides": {
                "ANY_COMP": {
                    "algorithm": "RandomForestClassifier",
                    "hyperparameters": {"n_estimators": 100},
                }
            }
        }

        orchestrator = MedpipeOrchestrator(config=mock_config)
        resolved = orchestrator.resolve_model_configurations()

        assert "ANY_COMP" in resolved
        assert resolved["ANY_COMP"]["algorithm"] == "RandomForestClassifier"
        assert "default_model" not in resolved["ANY_COMP"]

    def test_resolve_model_configurations_completely_empty(
        self, mock_add_handler, mock_get_logger, mock_artifact_mgr, mock_config
    ):
        """
        Test configuration resolution when neither defaults nor overrides exist.
        """
        mock_config.data.outcomes = ["MORTALITY_30D"]

        # Completely empty model configuration
        mock_config.model_dump.return_value = {}

        orchestrator = MedpipeOrchestrator(config=mock_config)
        resolved = orchestrator.resolve_model_configurations()

        assert "MORTALITY_30D" in resolved
        assert resolved["MORTALITY_30D"] == {}  # Should resolve to an empty dictionary

    def test_resolve_model_configurations_extra_override_ignored(
        self, mock_add_handler, mock_get_logger, mock_artifact_mgr, mock_config
    ):
        """
        Test that overrides for outcomes not listed in data.outcomes are ignored.
        """
        mock_config.data.outcomes = ["MORTALITY_30D"]

        mock_config.model_dump.return_value = {
            "default_model": {"algorithm": "LogisticRegression"},
            "outcome_overrides": {
                "MORTALITY_30D": {"algorithm": "RandomForestClassifier"},
                "GHOST_OUTCOME": {"algorithm": "SVC"},  # Not in data.outcomes
            },
        }

        orchestrator = MedpipeOrchestrator(config=mock_config)
        resolved = orchestrator.resolve_model_configurations()

        assert "MORTALITY_30D" in resolved
        assert "GHOST_OUTCOME" not in resolved
