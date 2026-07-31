from pathlib import Path
from unittest.mock import MagicMock, patch

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

    def test_save_artifacts_missing_data_attribute(
        self, mock_add_handler, mock_get_logger, mock_artifact_mgr
    ):
        """Test artifact saving handles a config entirely missing the data block."""
        mock_config_no_data = MagicMock(spec=MedpipeConfig)
        # Explicitly remove the data attribute to trigger the fallback
        del mock_config_no_data.data
        mock_config_no_data.model_dump.return_value = {"workflow": {}}

        mock_artifact_mgr_instance = mock_artifact_mgr.return_value

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

        with pytest.raises(ValueError, match="is not registered and was not found"):
            orchestrator.build_preprocessor()
