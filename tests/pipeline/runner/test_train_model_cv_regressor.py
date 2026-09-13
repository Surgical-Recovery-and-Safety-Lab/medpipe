"""
Integration tests for MedpipeRegressorRunner._train_model_cv with CRPS
scoring, exercising a real DistributionalPipeline wrapping a real
OrdBoostRegressor through scikit-learn's actual cross_validate/GridSearchCV
machinery (not mocked), since this is exactly the scenario the CRPS scorer
path (bypassing sklearn's make_scorer) needs to prove out.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from ordboost import OrdBoostRegressor
from sklearn.model_selection import KFold

from medpipe.metrics.core import build_scorers
from medpipe.pipeline.estimator import DistributionalPipeline
from medpipe.pipeline.runner import MedpipeRegressorRunner


@pytest.fixture
def continuous_regression_data() -> tuple[pd.DataFrame, np.ndarray]:
    """Synthetic continuous regression data, large enough for a few CV folds
    and for OrdBoostRegressor's ordinal binning to have something to bin."""
    rng = np.random.default_rng(42)
    n_samples = 80
    X = pd.DataFrame(
        {
            "feature1": rng.standard_normal(n_samples),
            "feature2": rng.standard_normal(n_samples),
        }
    )
    y = np.exp(X["feature1"].to_numpy() * 0.5) + rng.normal(0.0, 0.5, n_samples)
    return X, y


class TestTrainModelCvRegressorCrps:
    """Integration tests for CRPS-based cross-validation scoring against a
    real OrdBoostRegressor wrapped in DistributionalPipeline."""

    @patch("medpipe.pipeline.runner.MedpipeRegressorRunner._save_cv_results")
    def test_cross_validate_with_crps_scoring_produces_finite_scores(
        self, mock_save_cv_results, mock_orchestrator, continuous_regression_data
    ):
        """Test that running real cross_validate with scoring={'crps': ...}
        against a DistributionalPipeline(OrdBoostRegressor) produces finite
        scores, proving the predict_dist-based scorer works end-to-end
        through sklearn's actual CV machinery (not just unit-tested in
        isolation)."""
        mock_orchestrator.config.workflow.evaluation.metrics.metrics = ["crps"]
        mock_orchestrator.config.workflow.n_jobs = 1
        mock_orchestrator.config.workflow.validation.cross_validation.grid_search = (
            False
        )

        runner = MedpipeRegressorRunner(orchestrator=mock_orchestrator)
        X, y = continuous_regression_data

        pipeline = DistributionalPipeline(
            [
                (
                    "regressor",
                    OrdBoostRegressor(
                        n_bins=10,
                        mapper="quantile",
                        learning_rate=0.1,
                        max_iter=20,
                        random_state=42,
                    ),
                )
            ]
        )
        cv_splitter = KFold(n_splits=3, shuffle=True, random_state=42)

        result = runner._train_model_cv(
            outcome="LOS_DAYS",
            pipeline=pipeline,
            hyperparams={},
            X_train=X,
            y_train=y,
            groups_train=None,
            cv_splitter=cv_splitter,
        )

        assert isinstance(result, DistributionalPipeline)
        mock_save_cv_results.assert_called_once()

        saved_cv_results_df = mock_save_cv_results.call_args[0][1]
        assert "test_crps" in saved_cv_results_df.columns
        assert np.all(np.isfinite(saved_cv_results_df["test_crps"]))
        # sklearn's "greater is better" convention: since CRPS scorer
        # negates the raw (positive) CRPS loss, fold scores should be <= 0.
        assert np.all(saved_cv_results_df["test_crps"] <= 0)

    def test_build_scorers_crps_scorer_direct_call(
        self, continuous_regression_data
    ):
        """Test build_scorers(['crps']) produces a scorer that, called
        directly against a fitted DistributionalPipeline(OrdBoostRegressor),
        returns a finite, non-positive value."""
        X, y = continuous_regression_data

        pipeline = DistributionalPipeline(
            [
                (
                    "regressor",
                    OrdBoostRegressor(
                        n_bins=10,
                        mapper="quantile",
                        learning_rate=0.1,
                        max_iter=20,
                        random_state=42,
                    ),
                )
            ]
        )
        pipeline.fit(X, y)

        scorers = build_scorers(["crps"])
        score = scorers["crps"](pipeline, X, y)

        assert np.isfinite(score)
        assert score <= 0
