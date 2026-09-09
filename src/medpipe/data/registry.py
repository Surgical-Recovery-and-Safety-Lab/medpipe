from typing import Callable, Type

import sklearn.impute
import sklearn.preprocessing

from medpipe.utils.registry import BaseRegistry


class PreprocessorRegistry(BaseRegistry[Type[Callable]]):
    """
    Registry for managing and resolving data preprocessing operations.
    """

    _fallback_modules = [
        sklearn.preprocessing,
        sklearn.impute,
    ]
