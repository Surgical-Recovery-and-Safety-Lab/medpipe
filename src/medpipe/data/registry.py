import sklearn.impute
import sklearn.preprocessing

from medpipe.utils.registry import BaseRegistry


class PreprocessorRegistry(BaseRegistry):
    """
    Registry for managing and resolving data preprocessing operations.
    """

    # Explicitly create a new dictionary for preprocessing operations.
    _registry = {}

    _fallback_modules = [
        sklearn.preprocessing,
        sklearn.impute,
    ]
