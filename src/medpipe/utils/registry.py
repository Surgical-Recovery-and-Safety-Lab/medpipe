from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar

T = TypeVar("T")


class BaseRegistry(Generic[T]):
    """
    Abstract base class for creating component registries.

    Subclasses must explicitly define their own `_registry` dictionary
    and `_fallback_modules` list to prevent cross-contamination.

    Parameters
    ----------
    Generic[T]
        The type of item stored in the registry (e.g., instances or class types).

    """

    _registry: Dict[str, T]
    _fallback_modules: List[Any] = []

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Automatically isolate dictionary state for each subclass
        cls._registry = {}

    @classmethod
    def register(cls, name: Optional[str] = None) -> Callable[[T], T]:
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
    def list_registered(cls) -> List[str]:
        """Return a list of string keys registered in this custom registry.

        Returns
        -------
        List[str]
            List of registered item lookup keys.

        """
        return list(cls._registry.keys())
