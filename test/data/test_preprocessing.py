import pytest
import sklearn.impute
import sklearn.preprocessing
from sklearn.base import BaseEstimator, TransformerMixin

from medpipe.data.preprocessing import PreprocessorRegistry


class DummyCustomTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X


class TestPreprocessorRegistry:
    """Test suite for the PreprocessorRegistry class."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        """Reset the registry before each test to ensure isolation."""
        PreprocessorRegistry._registry = {}

    def test_register_with_custom_name(self):
        """Test registering a transformer with a specified name."""

        @PreprocessorRegistry.register(name="MyBinner")
        class MyBinner(DummyCustomTransformer):
            pass

        assert "MyBinner" in PreprocessorRegistry.list_registered()
        assert PreprocessorRegistry.get("MyBinner") is MyBinner

    def test_register_with_default_name(self):
        """Test registering a transformer without a name defaults to class name."""

        @PreprocessorRegistry.register()
        class DefaultNameTransformer(DummyCustomTransformer):
            pass

        assert "DefaultNameTransformer" in PreprocessorRegistry.list_registered()
        assert (
            PreprocessorRegistry.get("DefaultNameTransformer") is DefaultNameTransformer
        )

    def test_get_sklearn_preprocessing_operation(self):
        """Test retrieving a standard sklearn.preprocessing operation."""
        op_class = PreprocessorRegistry.get("StandardScaler")
        assert op_class is sklearn.preprocessing.StandardScaler

    def test_get_sklearn_impute_operation(self):
        """Test retrieving a standard sklearn.impute operation."""
        op_class = PreprocessorRegistry.get("SimpleImputer")
        assert op_class is sklearn.impute.SimpleImputer

    def test_get_nonexistent_operation_raises_error(self):
        """Test that requesting an unknown operation raises a ValueError."""
        with pytest.raises(ValueError, match="is not registered and was not found"):
            PreprocessorRegistry.get("NonExistentMagicTransformer")

    def test_list_registered(self):
        """Test that list_registered returns all custom keys."""
        PreprocessorRegistry.register(name="Op1")(DummyCustomTransformer)
        PreprocessorRegistry.register(name="Op2")(DummyCustomTransformer)

        registered = PreprocessorRegistry.list_registered()
        assert len(registered) == 2
        assert "Op1" in registered
        assert "Op2" in registered

    def test_register_override_existing(self):
        """Test that registering a new transformer with an existing name
        overrides it."""

        @PreprocessorRegistry.register(name="DuplicateName")
        class FirstTransformer(DummyCustomTransformer):
            pass

        @PreprocessorRegistry.register(name="DuplicateName")
        class SecondTransformer(DummyCustomTransformer):
            pass

        # The registry should hold the most recently registered class
        assert PreprocessorRegistry.get("DuplicateName") is SecondTransformer
