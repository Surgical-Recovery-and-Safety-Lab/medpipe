"""
Test functions for the metrics.registry module.
"""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from medpipe.metrics.registry import MetricRegistry, MetricSpec


@pytest.fixture
def fitted_estimator() -> LogisticRegression:
    """A tiny, deterministically fitted binary classifier for exercising
    scorers end-to-end without needing to mock sklearn's estimator protocol."""
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])
    return LogisticRegression().fit(X, y)


class TestMetricRegistry:
    """Test class for the MetricRegistry class."""

    def test_get_returns_registered_spec(self) -> None:
        """Test retrieving a default-registered spec by name."""
        spec = MetricRegistry.get("roc_auc")

        assert isinstance(spec, MetricSpec)
        assert spec.name == "roc_auc"
        assert spec.display_name == "AUROC"

    def test_get_missing_metric_raises_value_error(self) -> None:
        """Test that looking up an unregistered metric name raises with a
        message naming the missing key."""
        with pytest.raises(ValueError, match="'non_existent_metric' was not found"):
            MetricRegistry.get("non_existent_metric")

    def test_register_spec_adds_and_returns_the_same_instance(self) -> None:
        """Test that register_spec both stores the spec under its name and
        returns that exact same instance back to the caller."""
        custom_spec = MetricSpec(
            name="custom_test_metric",
            func=lambda y, y_p: 0.5,
            response_method="predict",
            display_name="Custom Metric",
        )

        returned = MetricRegistry.register_spec(custom_spec)

        assert returned is custom_spec
        assert MetricRegistry.get("custom_test_metric") is custom_spec

    def test_register_spec_overwrites_existing_name(self) -> None:
        """Test that registering a new spec under an already-used name
        replaces the previous entry rather than raising or duplicating it."""
        first = MetricSpec(
            name="overwrite_me",
            func=lambda y, y_p: 0.1,
            response_method="predict",
            display_name="First",
        )
        second = MetricSpec(
            name="overwrite_me",
            func=lambda y, y_p: 0.2,
            response_method="predict",
            display_name="Second",
        )

        MetricRegistry.register_spec(first)
        MetricRegistry.register_spec(second)

        assert MetricRegistry.get("overwrite_me") is second


class TestMetricSpec:
    """Test class for the MetricSpec class, in particular get_scorer's
    dispatch between a pre-registered sklearn scorer and a custom
    make_scorer-built one, and that scorer's choice of estimator response
    method (predict vs predict_proba)."""

    def test_get_scorer_uses_sklearn_scorer_name_when_provided(
        self, fitted_estimator: LogisticRegression
    ) -> None:
        """Test that a spec with sklearn_scorer_name set builds its scorer
        via sklearn's own registry rather than make_scorer, and that the
        resulting scorer computes the expected value."""
        spec = MetricSpec(
            name="accuracy_like",
            func=accuracy_score,
            response_method="predict",
            display_name="Accuracy",
            sklearn_scorer_name="accuracy",
        )
        X = np.array([[0.0], [1.0], [2.0], [3.0]])
        y = np.array([0, 0, 1, 1])

        scorer = spec.get_scorer()
        result = scorer(fitted_estimator, X, y)

        expected = accuracy_score(y, fitted_estimator.predict(X))
        assert result == pytest.approx(expected)

    def test_get_scorer_string_predict_proba_calls_predict_proba(
        self, fitted_estimator: LogisticRegression
    ) -> None:
        """Test that response_method='predict_proba' (a plain string, no
        sklearn_scorer_name) builds a scorer that feeds the positive-class
        probability column to the metric function."""
        X = np.array([[0.0], [1.0], [2.0], [3.0]])
        y = np.array([0, 0, 1, 1])
        spec = MetricSpec(
            name="dummy_proba",
            func=lambda y_true, y_score: float(np.sum(y_score)),
            response_method="predict_proba",
            display_name="Dummy",
        )

        scorer = spec.get_scorer()
        result = scorer(fitted_estimator, X, y)

        expected = float(np.sum(fitted_estimator.predict_proba(X)[:, 1]))
        assert result == pytest.approx(expected)

    def test_get_scorer_string_predict_calls_predict(
        self, fitted_estimator: LogisticRegression
    ) -> None:
        """Test that a plain 'predict' response_method (no sklearn_scorer_name)
        builds a scorer that feeds the estimator's hard predictions to the
        metric function, not probabilities."""
        X = np.array([[0.0], [1.0], [2.0], [3.0]])
        y = np.array([0, 0, 1, 1])
        spec = MetricSpec(
            name="dummy_predict",
            func=lambda y_true, y_score: float(np.sum(y_score)),
            response_method="predict",
            display_name="Dummy",
        )

        scorer = spec.get_scorer()
        result = scorer(fitted_estimator, X, y)

        expected = float(np.sum(fitted_estimator.predict(X)))
        assert result == pytest.approx(expected)

    def test_get_scorer_tuple_with_predict_proba_calls_predict_proba(
        self, fitted_estimator: LogisticRegression
    ) -> None:
        """Test that a tuple response_method containing 'predict_proba'
        (e.g. as used by roc_auc/ap) is detected via the isinstance(tuple)
        branch and still routes to predict_proba."""
        X = np.array([[0.0], [1.0], [2.0], [3.0]])
        y = np.array([0, 0, 1, 1])
        spec = MetricSpec(
            name="dummy_tuple_proba",
            func=lambda y_true, y_score: float(np.sum(y_score)),
            response_method=("decision_function", "predict_proba"),
            display_name="Dummy",
        )

        scorer = spec.get_scorer()
        result = scorer(fitted_estimator, X, y)

        expected = float(np.sum(fitted_estimator.predict_proba(X)[:, 1]))
        assert result == pytest.approx(expected)

    def test_get_scorer_tuple_without_predict_proba_calls_predict(
        self, fitted_estimator: LogisticRegression
    ) -> None:
        """Test that a tuple response_method NOT containing 'predict_proba'
        falls back to 'predict' via the isinstance(tuple) branch."""
        X = np.array([[0.0], [1.0], [2.0], [3.0]])
        y = np.array([0, 0, 1, 1])
        spec = MetricSpec(
            name="dummy_tuple_predict",
            func=lambda y_true, y_score: float(np.sum(y_score)),
            response_method=("decision_function",),
            display_name="Dummy",
        )

        scorer = spec.get_scorer()
        result = scorer(fitted_estimator, X, y)

        expected = float(np.sum(fitted_estimator.predict(X)))
        assert result == pytest.approx(expected)

    def test_get_scorer_predict_dist_bypasses_make_scorer(self) -> None:
        """Test that response_method='predict_dist' produces a plain raw
        callable (sklearn's scorer(estimator, X, y) convention) rather than
        a sklearn make_scorer-built _Scorer object, since make_scorer has no
        concept of a distributional response method."""
        captured: dict[str, object] = {}

        def fake_dist_metric(y_true, dist):
            captured["y_true"] = y_true
            captured["dist"] = dist
            return 3.5

        spec = MetricSpec(
            name="dummy_dist",
            func=fake_dist_metric,
            response_method="predict_dist",
            display_name="Dummy Dist",
        )

        class _FakeEstimator:
            def predict_dist(self, X):
                return f"dist_for:{list(X)}"

        scorer = spec.get_scorer()
        assert not hasattr(scorer, "_score_func")  # not a sklearn _Scorer

        result = scorer(_FakeEstimator(), [1, 2, 3], np.array([0.1, 0.2, 0.3]))

        # Negated, since CRPS-like distributional metrics are losses
        # (lower is better) but sklearn scorers are "greater is better".
        assert result == -3.5
        assert captured["dist"] == "dist_for:[1, 2, 3]"
        np.testing.assert_array_equal(captured["y_true"], [0.1, 0.2, 0.3])
