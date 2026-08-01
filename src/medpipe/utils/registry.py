from typing import Any, Callable, Dict, List, Optional, Type


class BaseRegistry:
    """
    Abstract base class for creating component registries.

    Subclasses must explicitly define their own `_registry` dictionary
    and `_fallback_modules` list to prevent cross-contamination.
    """

    _registry: Dict[str, Type] = {}
    _fallback_modules: List[Any] = []

    @classmethod
    def register(cls, name: Optional[str] = None) -> Callable[[Type], Type]:
        """
        Decorator to register a custom class into the specific subclass registry.

        Parameters
        ----------
        name : Optional[str]
            The lookup name for the component. Defaults to the class name.

        """

        def decorator(subclass: Type) -> Type:
            reg_name = name if name else subclass.__name__
            cls._registry[reg_name] = subclass
            return subclass

        return decorator

    @classmethod
    def get(cls, name: str) -> Type:
        """
        Retrieve a registered class, checking fallback modules if necessary.

        Parameters
        ----------
        name : str
            The name of the class to retrieve.

        Returns
        -------
        Type
            The uninstantiated class.

        Raises
        ------
        ValueError
            If the component is not found in the custom registry or fallbacks.

        """
        if name in cls._registry:
            return cls._registry[name]

        for module in cls._fallback_modules:
            if hasattr(module, name):
                return getattr(module, name)

        raise ValueError(
            f"'{name}' was not found in the custom registry or fallback modules."
        )

    @classmethod
    def list_registered(cls) -> List[str]:
        """
        Return a list of custom registered operations.

        Returns
        -------
        List[str]
            A list of string keys representing the registered custom operations.

        """
        return list(cls._registry.keys())
