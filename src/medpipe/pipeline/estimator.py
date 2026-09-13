"""
Pipeline subclass exposing a unified distributional prediction interface.
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


class DistributionalPipeline(Pipeline):
    """
    A scikit-learn Pipeline that exposes a unified `predict_dist` method,
    delegating to whichever of `predict_dist` or `pred_dist` the final
    estimator implements.

    Different probabilistic boosting libraries name their distributional
    prediction method differently (e.g. NGBoost's `NGBRegressor.pred_dist`,
    OrdBoost's presumed `predict_dist`). This subclass lets the rest of
    medpipe call a single, consistent method regardless of which library
    produced the final estimator.

    Adds no constructor parameters or instance state beyond `Pipeline`
    itself, so `sklearn.base.clone()` (used internally by `GridSearchCV`
    and `cross_validate`) preserves this subclass unchanged.

    """

    def predict_dist(self, X: pd.DataFrame | np.ndarray) -> Any:
        """
        Predict the full predictive distribution for samples in X.

        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            Features dataset of shape (n_samples, n_features).

        Returns
        -------
        Any
            The distribution object returned by the final estimator's
            `predict_dist` or `pred_dist` method (e.g. an
            `ngboost.distns.Normal` instance).

        Raises
        ------
        AttributeError
            If the final estimator implements neither `predict_dist` nor
            `pred_dist`.

        """
        Xt = X
        if len(self.steps) > 1:
            Xt = self[:-1].transform(Xt)

        final_estimator = self.steps[-1][1]

        if hasattr(final_estimator, "predict_dist"):
            return final_estimator.predict_dist(Xt)
        elif hasattr(final_estimator, "pred_dist"):
            return final_estimator.pred_dist(Xt)

        raise AttributeError(
            f"{type(final_estimator).__name__} implements neither "
            "'predict_dist' nor 'pred_dist'."
        )
