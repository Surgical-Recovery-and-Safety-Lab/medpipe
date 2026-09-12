"""
Tests for MedpipeOrchestrator.build_preprocessor and its
_check_operation helper.
"""

from unittest.mock import MagicMock, patch

import pytest
from sklearn.pipeline import Pipeline

from medpipe.pipeline.orchestrator import MedpipeOrchestrator


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

    def test_build_preprocessor_multiple_operations(
        self, mock_add_handler, mock_get_logger, mock_artifact_mgr, mock_config
    ):
        """Test that multiple operations are chained into successive
        ColumnTransformer steps, exercising the "not the last operation"
        branch (which keeps pandas output for intermediate steps) as well
        as the "last operation" branch already covered by the single-op test."""
        mock_config.data.predictors = ["AGE", "BMI"]

        op1 = MagicMock()
        op1.name = "StandardScaler"
        op1.columns = ["AGE"]
        op1.model_extra = {}

        op2 = MagicMock()
        op2.name = "MinMaxScaler"
        op2.columns = ["BMI"]
        op2.model_extra = {}

        mock_config.workflow.preprocessing.operations = [op1, op2]

        orchestrator = MedpipeOrchestrator(config=mock_config)
        pipeline = orchestrator.build_preprocessor()

        assert pipeline is not None
        assert len(pipeline.steps) == 2
        assert pipeline.steps[0][0] == "transformer_1"
        assert pipeline.steps[1][0] == "transformer_2"

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
