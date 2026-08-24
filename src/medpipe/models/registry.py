from typing import Type

import ngboost
import sklearn.ensemble
import sklearn.isotonic
import sklearn.linear_model
from sklearn.base import BaseEstimator

from medpipe.utils.registry import BaseRegistry


class ModelRegistry(BaseRegistry[Type[BaseEstimator]]):
    """
    Registry for managing and resolving machine learning estimators.
    """

    _fallback_modules = [
        sklearn.ensemble,
        sklearn.linear_model,
        sklearn.isotonic,
        ngboost,
    ]
