import pytest
import sklearn.impute
import sklearn.preprocessing

from medpipe.data.registry import PreprocessorRegistry


class TestPreprocessorRegistry:
    """
    Test suite for the domain-specific PreprocessorRegistry.
    """

    @pytest.fixture(autouse=True)
    def clean_registry(self):
        """Clears the PreprocessorRegistry custom dictionary before each test."""
        PreprocessorRegistry._registry.clear()
        yield

    def test_fallback_sklearn_preprocessing(self):
        """Test retrieving a transformer from sklearn.preprocessing fallback."""
        preprocessor_cls = PreprocessorRegistry.get("StandardScaler")
        assert preprocessor_cls is sklearn.preprocessing.StandardScaler

    def test_fallback_sklearn_impute(self):
        """Test retrieving an imputer from sklearn.impute fallback."""
        imputer_cls = PreprocessorRegistry.get("SimpleImputer")
        assert imputer_cls is sklearn.impute.SimpleImputer

    def test_custom_preprocessor_registration(self):
        """Test that a custom preprocessor can be injected."""

        @PreprocessorRegistry.register()
        class CustomScaler:
            pass

        assert PreprocessorRegistry.get("CustomScaler") is CustomScaler
