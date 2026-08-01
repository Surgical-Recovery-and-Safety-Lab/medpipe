import ngboost
import sklearn.ensemble
import sklearn.isotonic
import sklearn.linear_model

from medpipe.utils.registry import BaseRegistry


class ModelRegistry(BaseRegistry):
    """
    Registry for managing and resolving machine learning estimators.
    """

    # Explicitly create a new dictionary for models
    _registry = {}

    _fallback_modules = [
        sklearn.ensemble,
        sklearn.linear_model,
        sklearn.isotonic,
        ngboost,
    ]
