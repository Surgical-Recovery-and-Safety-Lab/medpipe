import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from medpipe.pipeline.pipeline import Medpipe
from medpipe.utils.config import MedpipeConfig

# ==============================================================================
# 1. UNIT TESTS (Mocked Sub-components & Routing Validation)
# ==============================================================================


class TestMedpipeUnit:
    """Unit tests verifying orchestration delegation and argument routing."""

    @patch("medpipe.pipeline.pipeline.MedpipeOrchestrator")
    @patch("medpipe.pipeline.pipeline.MedpipeRunner")
    @patch("medpipe.pipeline.pipeline.MedpipeEvaluator")
    def test_medpipe_initialization(
        self, mock_eval_cls, mock_runner_cls, mock_orch_cls
    ):
        """Verify Medpipe initializes sub-orchestrators correctly."""
        mock_config = MagicMock(spec=MedpipeConfig)
        mock_orch_instance = mock_orch_cls.return_value

        mp = Medpipe(config=mock_config)

        mock_orch_cls.assert_called_once_with(mock_config, "artifacts")
        mock_runner_cls.assert_called_once_with(orchestrator=mock_orch_instance)
        mock_eval_cls.assert_called_once_with(
            orchestrator=mock_orch_instance, runner=mock_runner_cls.return_value
        )
        assert mp.mp_config == mock_orch_instance.config

    @patch("medpipe.pipeline.pipeline.MedpipeOrchestrator")
    @patch("medpipe.pipeline.pipeline.MedpipeRunner")
    @patch("medpipe.pipeline.pipeline.MedpipeEvaluator")
    def test_inference_delegation(self, mock_eval_cls, mock_runner_cls, mock_orch_cls):
        """Verify predict, predict_proba, and decision_function delegate to evaluator."""
        mp = Medpipe(config=MagicMock())
        X = pd.DataFrame({"A": [1, 2]})

        mp.predict(X, outcome="MORTALITY")
        mp.evaluator.predict.assert_called_once_with(
            X=X, model=None, outcome="MORTALITY"
        )

        mp.predict_proba(X, outcome="MORTALITY")
        mp.evaluator.predict_proba.assert_called_once_with(
            X=X, model=None, outcome="MORTALITY"
        )

        mp.decision_function(X, outcome="MORTALITY")
        mp.evaluator.decision_function.assert_called_once_with(
            X=X, model=None, outcome="MORTALITY"
        )

    @patch("medpipe.pipeline.pipeline.MedpipeOrchestrator")
    @patch("medpipe.pipeline.pipeline.MedpipeRunner")
    @patch("medpipe.pipeline.pipeline.MedpipeEvaluator")
    def test_evaluate_y_dataframe_resolution(
        self, mock_eval_cls, mock_runner_cls, mock_orch_cls
    ):
        """Verify y DataFrame slicing resolution logic in evaluate()."""
        mp = Medpipe(config=MagicMock())
        X = pd.DataFrame({"AGE": [50, 60]})
        y_df = pd.DataFrame(
            {
                "MORTALITY_30D": [0, 1],
                "READMISSION_90D": [1, 0],
            }
        )

        # When outcome matches a column in y_df
        mp.evaluate(X, y_df, outcome="MORTALITY_30D")
        _, kwargs = mp.evaluator.evaluate.call_args
        pd.testing.assert_series_equal(kwargs["y"], y_df["MORTALITY_30D"])

        # When y is a single-column DataFrame without outcome specified
        y_single = pd.DataFrame({"TARGET": [0, 1]})
        mp.evaluate(X, y_single)
        _, kwargs = mp.evaluator.evaluate.call_args
        pd.testing.assert_series_equal(kwargs["y"], y_single.iloc[:, 0])
