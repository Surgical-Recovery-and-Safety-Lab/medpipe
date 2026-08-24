import ngboost
import pytest
import sklearn.ensemble

from medpipe.models.registry import ModelRegistry


class TestModelRegistry:
    """
    Test suite for the domain-specific ModelRegistry.
    """

    @pytest.fixture(autouse=True)
    def clean_registry(self):
        """Clears the ModelRegistry custom dictionary before each test."""
        ModelRegistry._registry.clear()
        yield

    def test_fallback_sklearn_ensemble(self):
        """Test retrieving a model from the sklearn.ensemble fallback."""
        model_cls = ModelRegistry.get("RandomForestClassifier")
        assert model_cls is sklearn.ensemble.RandomForestClassifier

    def test_fallback_sklearn_linear_model(self):
        """Test retrieving a model from the sklearn.linear_model fallback."""
        model_cls = ModelRegistry.get("LogisticRegression")
        assert model_cls is sklearn.linear_model.LogisticRegression

    def test_fallback_ngboost(self):
        """Test retrieving a model from the ngboost fallback."""
        model_cls = ModelRegistry.get("NGBClassifier")
        assert model_cls is ngboost.NGBClassifier

    def test_custom_model_registration(self):
        """Test that a custom model can be injected into the ModelRegistry."""

        @ModelRegistry.register(name="MyCustomModel")
        class MyCustomModel:
            pass

        assert ModelRegistry.get("MyCustomModel") is MyCustomModel
