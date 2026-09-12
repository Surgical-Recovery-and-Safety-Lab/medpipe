from collections.abc import Callable
from typing import Any, ClassVar


class BaseRegistry[T]:
    """
    Abstract base class for creating component registries.

    Every subclass automatically gets its own isolated `_registry` dict
    and `_fallback_modules` list via `__init_subclass__`, so state never
    leaks between subclasses even if they don't redeclare either
    attribute themselves.

    Parameters
    ----------
    T
        The type of item stored in the registry (e.g., instances or class types).

    """

    _registry: dict[str, T]
    _fallback_modules: ClassVar[list[Any]] = []

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Automatically isolate dictionary state for each subclass
        cls._registry = {}

        # Isolate fallback module state too: give each subclass its own
        # copy of whatever list it would otherwise inherit (empty for a
        # direct BaseRegistry subclass, or the parent's contents for a
        # deeper subclass), so mutating one subclass's list can never leak
        # into another. Subclasses that declare their own _fallback_modules
        # in their class body already own an isolated list and are left
        # untouched.
        if "_fallback_modules" not in cls.__dict__:
            cls._fallback_modules = list(cls._fallback_modules)

    @classmethod
    def register(cls, name: str | None = None) -> Callable[[T], T]:
        """Decorator to register a custom item into the specific subclass registry.

        Parameters
        ----------
        name : str or None, default=None
            The lookup key for the item. Defaults to the item's __name__ attribute.

        Returns
        -------
        decorator : Callable[[T], T]
            Decorator function registering the item.

        """

        def decorator(item: T) -> T:
            reg_name = name if name else getattr(item, "__name__", str(item))
            cls._registry[reg_name] = item
            return item

        return decorator

    @classmethod
    def get(cls, name: str) -> T:
        """Retrieve a registered item, checking fallback modules if necessary.

        Parameters
        ----------
        name : str
            The name identifier of the registered item.

        Returns
        -------
        T
            The registered instance or class.

        Raises
        ------
        ValueError
            If the requested item is not found in the registry or fallbacks.

        """
        if name in cls._registry:
            return cls._registry[name]

        for module in cls._fallback_modules:
            if hasattr(module, name):
                return getattr(module, name)

        raise ValueError(
            f"'{name}' was not found in the custom registry or fallback modules. "
            f"Available options: {cls.list_registered()}"
        )

    @classmethod
    def list_registered(cls) -> list[str]:
        """Return a list of string keys registered in this custom registry.

        Returns
        -------
        List[str]
            List of registered item lookup keys.

        """
        return list(cls._registry.keys())
