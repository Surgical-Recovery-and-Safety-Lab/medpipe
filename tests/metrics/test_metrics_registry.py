import pytest

from medpipe.metrics.registry import MetricRegistry, MetricSpec


class TestMetricRegistryAndSpec:
    """Test suite covering MetricRegistry and MetricSpec mechanics."""

    def test_registry_get_valid(self) -> None:
        """Test retrieving registered specs."""
        spec = MetricRegistry.get("roc_auc")
        assert isinstance(spec, MetricSpec)
        assert spec.name == "roc_auc"
        assert spec.display_name == "AUROC"

    def test_registry_get_invalid_raises_value_error(self) -> None:
        """Test looking up an unregistered metric name."""
        with pytest.raises(ValueError, match="non_existent_metric"):
            MetricRegistry.get("non_existent_metric")

    def test_register_custom_spec(self) -> None:
        """Test registering a custom metric spec dynamically."""
        dummy_func = lambda y, y_p: 0.5
        custom_spec = MetricSpec(
            name="custom_test_metric",
            func=dummy_func,
            response_method="predict",
            display_name="Custom Metric",
        )
        MetricRegistry.register_spec(custom_spec)

        retrieved = MetricRegistry.get("custom_test_metric")
        assert retrieved.name == "custom_test_metric"
        assert retrieved.func(None, None) == 0.5

    def test_metric_spec_get_scorer_with_sklearn_name(self) -> None:
        """Test get_scorer when a pre-defined sklearn_scorer_name is provided."""
        spec = MetricRegistry.get("accuracy")
        scorer = spec.get_scorer()
        assert callable(scorer)

    def test_metric_spec_get_scorer_custom_make_scorer(self) -> None:
        """Test get_scorer fallback when sklearn_scorer_name is None."""
        dummy_func = lambda y, y_pred: 1.0
        spec = MetricSpec(
            name="dummy_custom",
            func=dummy_func,
            response_method="predict_proba",
            display_name="Dummy",
            sklearn_scorer_name=None,
        )
        scorer = spec.get_scorer()
        assert callable(scorer)
