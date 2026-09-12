"""
Tests for MedpipeOrchestrator.__init__, _save_reproducibility_artifacts,
and the `splits` property guardrail.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from medpipe.pipeline.orchestrator import MedpipeOrchestrator
from medpipe.utils.config import MedpipeConfig


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

    @patch("medpipe.pipeline.orchestrator.set_verbosity")
    def test_init_uses_config_verbose_when_no_override(
        self,
        mock_set_verbosity,
        mock_add_handler,
        mock_get_logger,
        mock_artifact_mgr,
        mock_config,
    ):
        """Test that verbosity falls back to config.meta.verbose when
        verbose_override is not provided."""
        MedpipeOrchestrator(config=mock_config)

        mock_set_verbosity.assert_called_once_with(mock_config.meta.verbose)

    @patch("medpipe.pipeline.orchestrator.set_verbosity")
    def test_init_verbose_override_takes_precedence(
        self,
        mock_set_verbosity,
        mock_add_handler,
        mock_get_logger,
        mock_artifact_mgr,
        mock_config,
    ):
        """Test that an explicit verbose_override is used instead of the
        configuration's own verbosity setting."""
        MedpipeOrchestrator(config=mock_config, verbose_override="debug")

        mock_set_verbosity.assert_called_once_with("debug")


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
            destination_dir=orchestrator.run_dir / "env",
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
            destination_dir=orchestrator.run_dir / "env",
            config={"workflow": {}},
            dataset_path=None,
        )


@patch("medpipe.pipeline.orchestrator.ArtifactManager")
@patch("medpipe.pipeline.orchestrator.get_console_logger")
@patch("medpipe.pipeline.orchestrator.add_file_handler")
class TestSplitsProperty:
    """Unit tests for the MedpipeOrchestrator.splits property guardrail."""

    def test_splits_uninitialized_raises_runtime_error(
        self, mock_add_handler, mock_get_logger, mock_artifact_mgr, mock_config
    ):
        """Test that accessing .splits before prepare_data() raises RuntimeError."""
        orchestrator = MedpipeOrchestrator(config=mock_config)

        with pytest.raises(
            RuntimeError,
            match="Data has not been prepared yet. Call 'prepare_data\\(\\)'",
        ):
            _ = orchestrator.splits
