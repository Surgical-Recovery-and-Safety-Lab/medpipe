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

    def __sklearn_is_fitted__(self) -> bool:
        """
        Indicate whether the pipeline's final estimator has been fitted.

        Falls back to checking NGBoost's `base_models` list when the
        default scikit-learn check reports "not fitted". NGBoost estimators
        (e.g. `NGBRegressor`) never set any scikit-learn-conventional
        fitted attribute (a name ending in an underscore, not starting with
        `__`), which is exactly what `sklearn.utils.validation
        .check_is_fitted`'s default heuristic looks for on the final
        estimator. That makes `Pipeline.__sklearn_is_fitted__()` report
        "not fitted" even immediately after a fully successful `fit()`
        call, which in turn makes every `Pipeline` method that starts with
        `check_is_fitted(self)` (`predict`, `predict_proba`, `score`, ...)
        raise `NotFittedError` unconditionally for a fitted NGBoost model.
        `NGBoost.fit()` always initializes `base_models = []` and appends
        one entry per completed boosting round, so a non-empty list is a
        reliable "has been fitted" signal across the NGBoost model family.

        Returns
        -------
        bool
            True if the pipeline's final estimator has been fitted.

        """
        if super().__sklearn_is_fitted__():
            return True

        final_estimator = self.steps[-1][1] if self.steps else None
        return bool(getattr(final_estimator, "base_models", None))

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
