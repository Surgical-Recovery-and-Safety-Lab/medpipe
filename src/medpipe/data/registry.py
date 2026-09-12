from collections.abc import Callable
from types import ModuleType
from typing import ClassVar

import sklearn.impute
import sklearn.preprocessing

from medpipe.utils.registry import BaseRegistry


class PreprocessorRegistry(BaseRegistry[type[Callable]]):
    """
    Registry for managing and resolving data preprocessing operations.
    """

    _fallback_modules: ClassVar[list[ModuleType]] = [
        sklearn.preprocessing,
        sklearn.impute,
    ]
