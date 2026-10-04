import ngboost
import ordboost
import pytest
import sklearn.ensemble
import sklearn.isotonic
import sklearn.linear_model

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

    def test_fallback_sklearn_isotonic(self):
        """Test retrieving a model from the sklearn.isotonic fallback."""
        model_cls = ModelRegistry.get("IsotonicRegression")
        assert model_cls is sklearn.isotonic.IsotonicRegression

    def test_fallback_ngboost(self):
        """Test retrieving a model from the ngboost fallback."""
        model_cls = ModelRegistry.get("NGBClassifier")
        assert model_cls is ngboost.NGBClassifier

    def test_fallback_ordboost(self):
        """Test retrieving a model from the ordboost fallback."""
        model_cls = ModelRegistry.get("OrdBoostRegressor")
        assert model_cls is ordboost.OrdBoostRegressor

    def test_missing_model_raises_value_error(self):
        """Test that a name absent from both the registry and its fallback
        modules raises a ValueError."""
        with pytest.raises(ValueError, match="'NotARealModel' was not found"):
            ModelRegistry.get("NotARealModel")

    def test_custom_model_registration(self):
        """Test that a custom model can be injected into the ModelRegistry."""

        @ModelRegistry.register(name="MyCustomModel")
        class MyCustomModel:
            pass

        assert ModelRegistry.get("MyCustomModel") is MyCustomModel
