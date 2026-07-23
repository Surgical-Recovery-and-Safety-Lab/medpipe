"""
Transformers functions module.

This module provides transformers for regressors.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

if TYPE_CHECKING:
    import numpy.typing as npt


class BoundedLogitTransformer(BaseEstimator, TransformerMixin):
    """
    Class that transforms data to safely bound it in the logit space.

    Attributes
    ----------
    max_value : float, default: 1.0
        Max value in the data, needed for back-transformation.
    epsilon : float, default: 1e-7
        Value for clipping to avoid log(0).

    Methods
    -------
    fit(y)
        Needed for compatibility.
    transform(y)
        Transform data from the original space into the logit space.
    inverse_transform(y)
        Transforms data from the logit space back to the original space.
    """

    def __init__(self, max_value: float = 1.0, epsilon: float = 1e-7) -> None:
        self.max_value = max_value
        self.epsilon = epsilon  # To prevent log(0) or division by zero

    def fit(self, y: npt.NDArray) -> "BoundedLogitTransformer":
        """Needed for compatibility."""
        return self

    def transform(self, y: npt.NDArray) -> npt.NDArray:
        """
        Transform data from the original space into the
        logit space.

        The data is clipped to avoid log(0).

        Parameters
        ----------
        y : npt.NDArray
            Data to transform into the logit space.

        Returns
        -------
        y_transformed : npt.NDArray
            Transformed data in the logit space.

        """
        self.max_value = np.max(y)  # Save max value
        y_scaled = np.clip(y / self.max_value, self.epsilon, 1 - self.epsilon)
        return np.log(y_scaled / (1 - y_scaled))

    def inverse_transform(self, y: npt.NDArray) -> npt.NDArray:
        """
        Transforms data from the logit space back to the
        original space.

        Parameters
        ----------
        y : npt.NDArray
            Values in the logit space to be transformed.

        Returns
        -------
        y_transformed : npt.NDArray
            Back-transformed y into the original space.

        """
        exp_y = np.exp(y)
        return (exp_y / (1 + exp_y)) * self.max_value
