from types import ModuleType
from typing import ClassVar

import ngboost
import sklearn.ensemble
import sklearn.isotonic
import sklearn.linear_model
from sklearn.base import BaseEstimator

from medpipe.utils.registry import BaseRegistry


class ModelRegistry(BaseRegistry[type[BaseEstimator]]):
    """
    Registry for managing and resolving machine learning estimators.
    """

    _fallback_modules: ClassVar[list[ModuleType]] = [
        sklearn.ensemble,
        sklearn.linear_model,
        sklearn.isotonic,
        ngboost,
    ]
