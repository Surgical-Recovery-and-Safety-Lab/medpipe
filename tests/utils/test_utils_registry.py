import pytest

from medpipe.data.registry import PreprocessorRegistry
from medpipe.models.registry import ModelRegistry
from medpipe.utils.registry import BaseRegistry


class TestBaseRegistry:
    """
    Test suite for the BaseRegistry utility class.
    """

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """
        Fixture to create a fresh dummy registry before each test
        and clean up to prevent state leakage.
        """

        class DummyRegistry(BaseRegistry):
            _registry = {}
            _fallback_modules = []

        self.DummyRegistry = DummyRegistry
        yield
        self.DummyRegistry._registry.clear()

    def test_register_default_name(self):
        """Test registering a class without a custom name defaults to the class name."""

        @self.DummyRegistry.register()
        class MyComponent:
            pass

        assert "MyComponent" in self.DummyRegistry.list_registered()
        assert self.DummyRegistry.get("MyComponent") is MyComponent

    def test_register_custom_name(self):
        """Test registering a class with a specific custom string name."""

        @self.DummyRegistry.register(name="CustomName")
        class AnotherComponent:
            pass

        assert "CustomName" in self.DummyRegistry.list_registered()
        assert "AnotherComponent" not in self.DummyRegistry.list_registered()
        assert self.DummyRegistry.get("CustomName") is AnotherComponent

    def test_get_missing_component_raises_error(self):
        """Test that requesting an unregistered component raises a ValueError."""
        with pytest.raises(ValueError, match="'MissingComponent' was not found"):
            self.DummyRegistry.get("MissingComponent")


class TestRegistryIndependence:
    """
    Test suite to ensure registries do not leak state into one another.
    """

    @pytest.fixture(autouse=True)
    def clean_registries(self):
        """Clears both registries before testing."""
        ModelRegistry._registry.clear()
        PreprocessorRegistry._registry.clear()
        yield
        ModelRegistry._registry.clear()
        PreprocessorRegistry._registry.clear()

    def test_registries_do_not_share_state(self):
        """
        Ensure that registering a component in the ModelRegistry does not
        make it available in the PreprocessorRegistry, verifying the DRY
        base class implementation works perfectly.
        """

        @ModelRegistry.register(name="IsolatedModel")
        class IsolatedModel:
            pass

        # It should exist in ModelRegistry
        assert ModelRegistry.get("IsolatedModel") is IsolatedModel

        # It should NOT exist in PreprocessorRegistry
        with pytest.raises(ValueError, match="'IsolatedModel' was not found"):
            PreprocessorRegistry.get("IsolatedModel")
