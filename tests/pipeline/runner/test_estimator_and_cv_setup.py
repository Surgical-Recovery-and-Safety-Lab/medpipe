"""
Tests for MedpipeRunner._instantiate_estimator and _create_cv_splitter.
"""

import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from medpipe.pipeline.runner import MedpipeRunner


class TestInstantiateEstimator:
    """Unit tests for MedpipeRunner._instantiate_estimator."""

    def test_instantiate_estimator_classifier(self, mock_orchestrator):
        runner = MedpipeRunner(orchestrator=mock_orchestrator)
        estimator = runner._instantiate_estimator(
            "RandomForestClassifier", {"n_estimators": 10}
        )

        assert isinstance(estimator, RandomForestClassifier)
        assert estimator.n_estimators == 10

    def test_instantiate_estimator_regressor_returned_unwrapped(
        self, mock_orchestrator
    ):
        """Test that regressors are returned as plain estimators, with no
        target-transformation wrapping applied."""
        runner = MedpipeRunner(orchestrator=mock_orchestrator)
        estimator = runner._instantiate_estimator("LinearRegression", {})

        assert isinstance(estimator, LinearRegression)

    def test_instantiate_estimator_list_params_filtered(self, mock_orchestrator):
        """Test that list hyperparameters are reduced to scalars for initial
        instantiation."""
        runner = MedpipeRunner(orchestrator=mock_orchestrator)
        params = {"n_estimators": [10, 50, 100], "max_depth": 5}

        estimator = runner._instantiate_estimator("RandomForestClassifier", params)
        assert isinstance(estimator, RandomForestClassifier)
        assert estimator.n_estimators == 10
        assert estimator.max_depth == 5


class TestCreateCvSplitter:
    """Unit tests for MedpipeRunner._create_cv_splitter."""

    def test_create_cv_splitter(self, mock_orchestrator):
        runner = MedpipeRunner(orchestrator=mock_orchestrator)

        random_cv = runner._create_cv_splitter("random", 3, 42)
        assert isinstance(random_cv, StratifiedKFold)
        assert random_cv.n_splits == 3

        group_cv = runner._create_cv_splitter("group", 5, 42)
        assert isinstance(group_cv, StratifiedGroupKFold)

        with pytest.raises(ValueError, match="Strategy must be 'random' or 'group'"):
            runner._create_cv_splitter("invalid_strategy", 5, 42)
