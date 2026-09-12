"""
Tests for the BaseRegistry class of the medpipe.utils.registry module.
"""

import types
from collections.abc import Generator
from typing import ClassVar

import pytest

from medpipe.data.registry import PreprocessorRegistry
from medpipe.models.registry import ModelRegistry
from medpipe.utils.registry import BaseRegistry


@pytest.fixture
def dummy_registry() -> type[BaseRegistry]:
    """A fresh BaseRegistry subclass, isolated by __init_subclass__.

    Deliberately does NOT redeclare _registry or _fallback_modules itself,
    so tests exercise the base class's own isolation behavior rather than
    a manually-reset one.
    """

    class DummyRegistry(BaseRegistry):
        pass

    return DummyRegistry


class TestSubclassIsolation:
    """Tests for BaseRegistry.__init_subclass__ and the isolation contract
    described in the class docstring."""

    def test_registry_starts_empty_without_manual_declaration(
        self, dummy_registry: type[BaseRegistry]
    ) -> None:
        """Test that __init_subclass__ alone gives a subclass a fresh,
        empty _registry, even when the subclass body never declares one."""
        assert dummy_registry._registry == {}

    def test_subclasses_do_not_share_registry_state(self) -> None:
        """Test that two independent subclasses each get their own
        _registry dict, with no cross-contamination."""

        class SubA(BaseRegistry):
            pass

        class SubB(BaseRegistry):
            pass

        SubA.register("shared_name")(int)

        assert SubA.get("shared_name") is int
        with pytest.raises(ValueError, match="'shared_name' was not found"):
            SubB.get("shared_name")

    def test_registry_dict_is_a_distinct_object_per_subclass(self) -> None:
        """Test that each subclass's _registry is a distinct dict object,
        not the same dict shared by reference."""

        class SubA(BaseRegistry):
            pass

        class SubB(BaseRegistry):
            pass

        assert SubA._registry is not SubB._registry

    def test_using_base_registry_directly_raises_attribute_error(self) -> None:
        """Test that BaseRegistry itself has no _registry until subclassed,
        since __init_subclass__ never runs for the base class itself."""
        with pytest.raises(AttributeError):
            BaseRegistry.register()(int)

        with pytest.raises(AttributeError):
            BaseRegistry.get("anything")


class TestRegister:
    """Tests for BaseRegistry.register."""

    def test_register_default_name_uses_class_name(
        self, dummy_registry: type[BaseRegistry]
    ) -> None:
        """Test registering a class without a custom name defaults to the class name."""

        @dummy_registry.register()
        class MyComponent:
            pass

        assert "MyComponent" in dummy_registry.list_registered()
        assert dummy_registry.get("MyComponent") is MyComponent

    def test_register_custom_name(self, dummy_registry: type[BaseRegistry]) -> None:
        """Test registering a class with a specific custom string name."""

        @dummy_registry.register(name="CustomName")
        class AnotherComponent:
            pass

        assert "CustomName" in dummy_registry.list_registered()
        assert "AnotherComponent" not in dummy_registry.list_registered()
        assert dummy_registry.get("CustomName") is AnotherComponent

    def test_register_returns_the_original_item_unchanged(
        self, dummy_registry: type[BaseRegistry]
    ) -> None:
        """Test that the decorator returns the exact same object it was
        given, so registration has no side effect on the item itself."""

        class Original:
            pass

        returned = dummy_registry.register()(Original)

        assert returned is Original

    def test_register_overwrites_existing_name(
        self, dummy_registry: type[BaseRegistry]
    ) -> None:
        """Test that re-registering an already-used name replaces the
        previous entry rather than raising or duplicating it."""

        @dummy_registry.register(name="Slot")
        class First:
            pass

        @dummy_registry.register(name="Slot")
        class Second:
            pass

        assert dummy_registry.get("Slot") is Second
        assert dummy_registry.list_registered().count("Slot") == 1

    def test_register_function_without_custom_name(
        self, dummy_registry: type[BaseRegistry]
    ) -> None:
        """Test that plain functions (not just classes) can be registered,
        since BaseRegistry is generic over any item type."""

        @dummy_registry.register()
        def my_function():
            return "value"

        assert dummy_registry.get("my_function") is my_function

    def test_register_object_without_dunder_name_falls_back_to_str(
        self, dummy_registry: type[BaseRegistry]
    ) -> None:
        """Test that an item with no __name__ attribute (e.g. a plain
        instance) is keyed by str(item) when no explicit name is given."""
        instance = object()

        dummy_registry.register()(instance)

        assert dummy_registry.get(str(instance)) is instance


class TestGet:
    """Tests for BaseRegistry.get."""

    def test_get_missing_component_raises_value_error(
        self, dummy_registry: type[BaseRegistry]
    ) -> None:
        """Test that requesting an unregistered component raises a ValueError."""
        with pytest.raises(ValueError, match="'MissingComponent' was not found"):
            dummy_registry.get("MissingComponent")

    def test_get_error_message_lists_available_options(
        self, dummy_registry: type[BaseRegistry]
    ) -> None:
        """Test that the ValueError message reports the currently
        registered names to help the caller pick a valid one."""
        dummy_registry.register(name="Known")(int)

        with pytest.raises(ValueError, match=r"Available options: \['Known'\]"):
            dummy_registry.get("Unknown")

    def test_get_prefers_registry_over_fallback_modules(
        self, dummy_registry: type[BaseRegistry]
    ) -> None:
        """Test that a name present in both _registry and a fallback
        module resolves to the _registry entry."""
        fallback_module = types.SimpleNamespace(Shared="from_fallback")
        dummy_registry._fallback_modules = [fallback_module]
        dummy_registry.register(name="Shared")("from_registry")

        assert dummy_registry.get("Shared") == "from_registry"

    def test_get_falls_back_to_module_when_not_registered(
        self, dummy_registry: type[BaseRegistry]
    ) -> None:
        """Test that a name absent from _registry is resolved from a
        fallback module instead of raising."""
        fallback_module = types.SimpleNamespace(Helper="helper_value")
        dummy_registry._fallback_modules = [fallback_module]

        assert dummy_registry.get("Helper") == "helper_value"

    def test_get_checks_fallback_modules_in_order(
        self, dummy_registry: type[BaseRegistry]
    ) -> None:
        """Test that fallback modules are checked in list order, so the
        first module defining the name wins."""
        first_module = types.SimpleNamespace(Thing="from_first")
        second_module = types.SimpleNamespace(Thing="from_second")
        dummy_registry._fallback_modules = [first_module, second_module]

        assert dummy_registry.get("Thing") == "from_first"

    def test_get_skips_fallback_modules_lacking_the_name(
        self, dummy_registry: type[BaseRegistry]
    ) -> None:
        """Test that a fallback module without the requested attribute is
        skipped in favor of a later one that has it."""
        empty_module = types.SimpleNamespace()
        real_module = types.SimpleNamespace(Thing="from_real_module")
        dummy_registry._fallback_modules = [empty_module, real_module]

        assert dummy_registry.get("Thing") == "from_real_module"

    def test_get_raises_when_absent_from_registry_and_all_fallbacks(
        self, dummy_registry: type[BaseRegistry]
    ) -> None:
        """Test that a name missing from _registry and every fallback
        module still raises ValueError."""
        dummy_registry._fallback_modules = [types.SimpleNamespace()]

        with pytest.raises(ValueError, match="'Missing' was not found"):
            dummy_registry.get("Missing")


class TestFallbackModulesIsolation:
    """Tests for the _fallback_modules isolation contract described in the
    BaseRegistry docstring: __init_subclass__ gives every subclass its own
    independent list, whether or not the subclass redeclares it itself."""

    def test_declaring_a_dedicated_list_isolates_it_from_siblings(self) -> None:
        """Test that a subclass which declares its own _fallback_modules
        does not leak mutations into a sibling subclass."""

        class SubA(BaseRegistry):
            _fallback_modules: ClassVar[list] = []

        class SubB(BaseRegistry):
            _fallback_modules: ClassVar[list] = []

        SubA._fallback_modules.append(types.SimpleNamespace(Thing="a"))

        assert SubA._fallback_modules != SubB._fallback_modules

    def test_omitting_the_declaration_still_isolates_fallback_modules(self) -> None:
        """Test that subclasses which do NOT redeclare _fallback_modules
        still each get their own independent list, so a module registered
        via one subclass's fallback list never becomes visible through a
        sibling subclass that also skipped the declaration."""

        class SubA(BaseRegistry):
            pass

        class SubB(BaseRegistry):
            pass

        assert SubA._fallback_modules is not SubB._fallback_modules

        SubA._fallback_modules.append(types.SimpleNamespace(Leaked="leaked_value"))

        assert SubB._fallback_modules == []
        with pytest.raises(ValueError, match="'Leaked' was not found"):
            SubB.get("Leaked")

    def test_deeper_subclass_inherits_parent_fallback_modules_as_a_copy(
        self,
    ) -> None:
        """Test that a subclass-of-a-subclass which doesn't declare its own
        _fallback_modules still sees its parent's fallback modules, but as
        an independent copy rather than the same list object."""
        parent_module = types.SimpleNamespace(Inherited="inherited_value")

        class Parent(BaseRegistry):
            _fallback_modules: ClassVar[list] = [parent_module]

        class Child(Parent):
            pass

        assert Child._fallback_modules == [parent_module]
        assert Child._fallback_modules is not Parent._fallback_modules

        Child._fallback_modules.append(types.SimpleNamespace(ChildOnly="child_value"))

        assert Child.get("ChildOnly") == "child_value"
        with pytest.raises(ValueError, match="'ChildOnly' was not found"):
            Parent.get("ChildOnly")


class TestListRegistered:
    """Tests for BaseRegistry.list_registered."""

    def test_list_registered_empty_by_default(
        self, dummy_registry: type[BaseRegistry]
    ) -> None:
        """Test that a freshly created subclass reports no registered names."""
        assert dummy_registry.list_registered() == []

    def test_list_registered_returns_all_registered_names(
        self, dummy_registry: type[BaseRegistry]
    ) -> None:
        """Test that every registered name appears in list_registered."""
        dummy_registry.register(name="First")(int)
        dummy_registry.register(name="Second")(str)

        assert set(dummy_registry.list_registered()) == {"First", "Second"}


class TestRegistryIndependenceAcrossConcreteSubclasses:
    """
    Integration-style check that real registries shipped in medpipe
    (ModelRegistry, PreprocessorRegistry) do not share state, confirming
    the isolation contract holds beyond synthetic dummy subclasses too.
    """

    @pytest.fixture(autouse=True)
    def clean_registries(self) -> Generator:
        """Clears both registries before and after each test."""
        ModelRegistry._registry.clear()
        PreprocessorRegistry._registry.clear()
        yield
        ModelRegistry._registry.clear()
        PreprocessorRegistry._registry.clear()

    def test_registries_do_not_share_state(self) -> None:
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
