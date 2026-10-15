"""
Integration tests for MedpipeRegressorEvaluator.evaluate() with CRPS,
exercising a real fitted OrdBoostRegressor end-to-end: point predictions,
distributional predictions, bootstrap confidence intervals, and per-subgroup
re-evaluation (since OrdBoost's distribution object does not support index
slicing, subgroup CRPS must be recomputed via predict_dist on the subgroup's
rows rather than sliced from the full-dataset distribution).
"""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from ordboost import OrdBoostRegressor

from medpipe.pipeline.estimator import DistributionalPipeline
from medpipe.pipeline.evaluator import MedpipeRegressorEvaluator


@pytest.fixture
def fitted_ordboost_pipeline_and_data():
    """A real, fitted DistributionalPipeline(OrdBoostRegressor) plus a
    held-out test set. The 'site' column doubles as both a model feature and
    a fairness stratification variable (a realistic setup, e.g. 'SEX' is
    often both), so the model can predict directly on the full evaluate() X
    without any column filtering."""
    rng = np.random.default_rng(7)
    n_train = 150
    X_train = pd.DataFrame(
        {
            "feature1": rng.standard_normal(n_train),
            "feature2": rng.standard_normal(n_train),
            "site": rng.integers(0, 2, size=n_train),
        }
    )
    y_train = np.exp(X_train["feature1"].to_numpy() * 0.5) + rng.normal(
        0.0, 0.5, n_train
    )

    n_test = 40
    X_test = pd.DataFrame(
        {
            "feature1": rng.standard_normal(n_test),
            "feature2": rng.standard_normal(n_test),
            "site": rng.integers(0, 2, size=n_test),
        },
        index=pd.RangeIndex(n_test),
    )
    y_test = pd.Series(
        np.exp(X_test["feature1"].to_numpy() * 0.5) + rng.normal(0.0, 0.5, n_test),
        index=X_test.index,
    )

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
    pipeline.fit(X_train, y_train)

    return pipeline, X_test, y_test


class TestRegressorEvaluatorCrpsIntegration:
    """End-to-end tests for CRPS flowing through
    MedpipeRegressorEvaluator.evaluate()."""

    def test_evaluate_with_crps_produces_finite_bootstrap_ci(
        self, fitted_ordboost_pipeline_and_data
    ):
        """Test that evaluate() with metrics=['rmse', 'crps'] against a real
        fitted OrdBoostRegressor produces finite point estimates and
        bootstrap CIs for both, end-to-end."""
        pipeline, X_test, y_test = fitted_ordboost_pipeline_and_data

        mock_orchestrator = _make_mock_orchestrator()
        mock_runner = _make_mock_runner(pipeline)

        evaluator = MedpipeRegressorEvaluator(mock_orchestrator, mock_runner)

        results = evaluator.evaluate(
            X_test,
            y_test,
            outcome="LOS_DAYS",
            metrics=["rmse", "crps"],
            save_artifacts=False,
        )

        overall = results["overall"]
        assert set(overall) == {"rmse", "crps"}
        for metric_name in ("rmse", "crps"):
            entry = overall[metric_name]
            assert np.isfinite(entry["point_estimate"])
            assert np.isfinite(entry["ci_lower"])
            assert np.isfinite(entry["ci_upper"])
            assert entry["ci_lower"] <= entry["point_estimate"] <= entry["ci_upper"]

    def test_evaluate_with_crps_and_subgroups_recomputes_distribution(
        self, fitted_ordboost_pipeline_and_data
    ):
        """Test that requesting crps alongside subgroup_specs correctly
        recomputes predict_dist per-stratum (rather than failing on
        OrdBoost's non-sliceable distribution object), producing finite
        per-subgroup CRPS results."""
        pipeline, X_test, y_test = fitted_ordboost_pipeline_and_data

        mock_orchestrator = _make_mock_orchestrator()
        mock_runner = _make_mock_runner(pipeline)

        evaluator = MedpipeRegressorEvaluator(mock_orchestrator, mock_runner)

        results = evaluator.evaluate(
            X_test,
            y_test,
            outcome="LOS_DAYS",
            metrics=["rmse", "crps"],
            subgroup_specs={"site": "site"},
            save_artifacts=False,
        )

        strata = results["strata"]["site"]
        assert set(strata) <= {"0", "1"}
        for _group_val, group_results in strata.items():
            assert np.isfinite(group_results["crps"]["point_estimate"])
            assert np.isfinite(group_results["rmse"]["point_estimate"])


def _make_mock_orchestrator():
    orchestrator = MagicMock()
    orchestrator.run_dir = "/tmp/does-not-matter"
    orchestrator.config.workflow.random_state = 7
    orchestrator.config.workflow.evaluation.metrics.metrics = ["rmse", "mae"]
    orchestrator.config.workflow.evaluation.metrics.n_bootstraps = 50
    orchestrator.config.workflow.evaluation.metrics.ci_level = 0.95
    return orchestrator


def _make_mock_runner(fitted_pipeline):
    runner = MagicMock()
    runner.fitted_models = {"LOS_DAYS": fitted_pipeline}
    return runner
