from typing import Callable, Dict, List, Optional, Type

import sklearn.impute
import sklearn.preprocessing


class PreprocessorRegistry:
    """
    Registry for managing and resolving preprocessing operation classes.

    This class provides a dynamic registry to decouple operation lookup from
    specific modules. It allows users to register custom transformer classes
    while falling back to standard scikit-learn operations when needed.

    Attributes
    ----------
    _registry : Dict[str, Type]
        Internal dictionary storing the registered custom transformer classes,
        keyed by their assigned registry names.

    Methods
    -------
    register(name=None)
        Decorator to register a custom transformer class.
    get(name)
        Retrieve a registered transformer class by name.
    list_registered()
        Return a list of all custom registered operations.

    """

    _registry: Dict[str, Type] = {}

    @classmethod
    def register(cls, name: Optional[str] = None) -> Callable[[Type], Type]:
        """
        Decorator to register a custom transformer class.

        Parameters
        ----------
        name : str, optional
            The lookup name for the transformer. If not provided, the class
            name will be used as the registry key.

        Returns
        -------
        Callable[[Type], Type]
            A decorator function that registers the provided class and returns it.

        """

        def decorator(subclass: Type) -> Type:
            reg_name = name if name else subclass.__name__
            cls._registry[reg_name] = subclass
            return subclass

        return decorator

    @classmethod
    def get(cls, name: str) -> Type:
        """
        Retrieve a registered transformer class by name.

        The method first checks the internal custom registry. If the operation
        is not found, it falls back to searching `sklearn.preprocessing` and
        `sklearn.impute`.

        Parameters
        ----------
        name : str
            The name of the transformer operation to retrieve.

        Returns
        -------
        Type
            The uninstantiated transformer class.

        Raises
        ------
        ValueError
            If the operation is not registered and cannot be found in the
            standard scikit-learn modules.

        """
        # 1. Check custom registry first
        if name in cls._registry:
            return cls._registry[name]

        # 2. Fall back to standard sklearn modules
        if hasattr(sklearn.preprocessing, name):
            return getattr(sklearn.preprocessing, name)

        if hasattr(sklearn.impute, name):
            return getattr(sklearn.impute, name)

        raise ValueError(
            f"Operation '{name}' is not registered and was not found in "
            "sklearn.preprocessing or sklearn.impute."
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
