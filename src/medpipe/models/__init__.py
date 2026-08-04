"""
medpipe.models
--------------
Model registry and estimator lookup utilities for the Medpipe package[cite: 15].

Provides a centralized registry for registering, resolving, and instantiating
machine learning estimators across scikit-learn, NGBoost, and custom models.
"""

from medpipe.models.registry import ModelRegistry

__all__ = [
    "ModelRegistry",
]
