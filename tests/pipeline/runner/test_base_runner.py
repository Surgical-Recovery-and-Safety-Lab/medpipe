"""
Tests for BaseRunner's outcome-type-agnostic hook default behavior.
"""

from unittest.mock import MagicMock

import pytest

from medpipe.pipeline.runner import BaseRunner


class TestBaseRunnerHooks:
    """Unit tests for BaseRunner's default hook implementations."""

    def test_create_cv_splitter_not_implemented(self, mock_orchestrator):
        """Test that the base class requires subclasses to select a
        CV splitter."""
        runner = BaseRunner(orchestrator=mock_orchestrator)

        with pytest.raises(NotImplementedError):
            runner._create_cv_splitter("random", 3, 42)

    def test_postfit_is_a_no_op_passthrough(self, mock_orchestrator):
        """Test that the default post-fit hook returns the fitted pipeline
        unchanged, with no post-hoc adjustment."""
        runner = BaseRunner(orchestrator=mock_orchestrator)
        best_pipeline = MagicMock()

        result = runner._postfit(
            outcome="MORTALITY_30D",
            best_pipeline=best_pipeline,
            model_config={},
            X_recal=None,
            y_recal=None,
        )

        assert result is best_pipeline
