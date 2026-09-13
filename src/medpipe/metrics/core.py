"""
Core metric functions module.

"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import scores.probability as scores_probability
import xarray as xr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    mean_absolute_error,
    precision_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)
from splinecalib import SplineCalib

from medpipe.metrics.registry import MetricRegistry, MetricSpec

if TYPE_CHECKING:
    from typing import Any

    import numpy.typing as npt

# ------------------------------------------------------------------------------
# STANDALONE METRIC FUNCTIONS
# ------------------------------------------------------------------------------

# Grid resolution and tail coverage used to build the CDF threshold grid that
# `crps_score` integrates over. Both NGBoost's and OrdBoost's distribution
# objects implement `.ppf(q)`/`.cdf(x)` for a scalar argument, returning an
# (n_samples,) array, regardless of the distribution family/library that
# produced them (e.g. NGBoost's parametric Normal, OrdBoost's non-parametric
# binned distribution) - so this grid-based approach works uniformly across
# libraries without special-casing any specific distribution type.
_CRPS_THRESHOLD_GRID_SIZE = 200
_CRPS_EXTREME_QUANTILES = (0.001, 0.999)


def ici_score(y: npt.NDArray, y_pred: npt.NDArray) -> float:
    """Computes the integrated calibration index using a spline-based curve."""
    if y_pred.ndim == 2:
        y_pred = y_pred[:, 1]

    spline = SplineCalib(logodds_scale=True)
    spline.fit(y_pred, y)
    smoothed_outputs = spline.predict(y_pred)

    if smoothed_outputs is not None:
        return float(np.mean(np.abs(smoothed_outputs - y_pred)))
    else:
        raise ValueError("Error predicting probabilities with spline")


def crps_score(y: npt.NDArray, dist: Any) -> float:
    """Computes the mean Continuous Ranked Probability Score (CRPS) for a
    predictive distribution, via `scores.probability.crps_cdf`'s exact
    piecewise-linear integration over a shared CDF threshold grid.

    Works with any distribution object exposing scipy-style `.ppf(q)` and
    `.cdf(x)` methods for a scalar argument (e.g. NGBoost's or OrdBoost's
    distribution objects), regardless of the underlying distribution family.

    Parameters
    ----------
    y : npt.NDArray
        Ground truth continuous target values of shape (n_samples,).
    dist : Any
        A predictive distribution object exposing `.ppf(q)` and `.cdf(x)`.

    Returns
    -------
    float
        The mean CRPS across samples (lower is better).

    """
    y_arr = np.asarray(y)
    low_q, high_q = _CRPS_EXTREME_QUANTILES

    lower = float(min(np.min(dist.ppf(low_q)), np.min(y_arr)))
    upper = float(max(np.max(dist.ppf(high_q)), np.max(y_arr)))
    if upper <= lower:
        upper = lower + 1e-6

    thresholds = np.linspace(lower, upper, _CRPS_THRESHOLD_GRID_SIZE)
    forecast_cdf = np.stack([dist.cdf(x) for x in thresholds], axis=-1)

    fcst = xr.DataArray(
        forecast_cdf,
        dims=["sample", "threshold"],
        coords={"threshold": thresholds},
    )
    obs = xr.DataArray(y_arr, dims=["sample"])

    result = scores_probability.crps_cdf(fcst, obs, threshold_dim="threshold")
    return float(result["total"].values)


# ------------------------------------------------------------------------------
# DEFAULT METRIC REGISTRATIONS
# ------------------------------------------------------------------------------

_DEFAULT_METRICS = [
    MetricSpec(
        "accuracy",
        accuracy_score,
        "predict",
        "Accuracy",
        "accuracy",
        needs_threshold=True,
    ),
    MetricSpec(
        "precision",
        precision_score,
        "predict",
        "Precision",
        "precision",
        needs_threshold=True,
    ),
    MetricSpec(
        "recall",
        recall_score,
        "predict",
        "Recall",
        "recall",
        needs_threshold=True,
    ),
    MetricSpec("f1", f1_score, "predict", "F1", "f1", needs_threshold=True),
    MetricSpec("log_loss", log_loss, "predict_proba", "Log loss", "neg_log_loss"),
    MetricSpec(
        "brier_score",
        brier_score_loss,
        "predict_proba",
        "Brier score",
        "neg_brier_score",
    ),
    MetricSpec(
        "roc_auc",
        roc_auc_score,
        ("decision_function", "predict_proba"),
        "AUROC",
        "roc_auc",
    ),
    MetricSpec(
        "auroc",
        roc_auc_score,
        ("decision_function", "predict_proba"),
        "AUROC",
        "roc_auc",
    ),
    MetricSpec(
        "ap",
        average_precision_score,
        ("decision_function", "predict_proba"),
        "AP",
        "average_precision",
    ),
    MetricSpec(
        "rmse",
        root_mean_squared_error,
        "predict",
        "RMSE",
        "neg_root_mean_squared_error",
    ),
    MetricSpec("mae", mean_absolute_error, "predict", "MAE", "neg_mean_absolute_error"),
    MetricSpec("ici", ici_score, "predict_proba", "ICI"),
    MetricSpec("crps", crps_score, "predict_dist", "CRPS"),
]

for _spec in _DEFAULT_METRICS:
    MetricRegistry.register_spec(_spec)

METRICS = MetricRegistry.list_registered()


def build_scorers(metrics: list[str] | npt.NDArray) -> dict[str, Callable]:
    """
    Build the dictionary of scorers for cross-validation.

    Parameters
    ----------
    metrics : list[str]
        List of metrics to use.

    Returns
    -------
    scorers : dict[str, Callable]
        Dictionary of scorers to pass to cross_validate.

    Raises
    ------
    TypeError
        If metrics is not a list of strings.
    ValueError
        If a metric is not a valid option.

    """
    if not isinstance(metrics, (list, np.ndarray)):
        raise TypeError("Input metrics should be a list of strings")
    if len(metrics) == 0 or not isinstance(metrics[0], str):
        raise TypeError("Input metrics should be a list of strings")

    scorers = {}
    for metric_name in metrics:
        spec = MetricRegistry.get(metric_name)
        scorers[metric_name] = spec.get_scorer()

    return scorers


def compute_metrics(
    metrics: list[str] | npt.NDArray, y: npt.NDArray, y_pred: npt.NDArray
) -> npt.NDArray:
    """
    Computes metrics based on predicted data.

    Scores are located at the same index as in the metrics array.

    Parameters
    ----------
    metrics : list[str]
        List of metrics to use.
    y : npt.NDArray
        Ground truth labels.
    y_pred : npt.NDArray
        Predictions from the model of shape
        (n_samples,) or (n_samples, 2).

    Returns
    -------
    scores : npt.NDArray
        Score array of shape (n_metrics,).

    Raises
    ------
    TypeError
        If metrics is not a list of strings.
        If y is not a np.ndarray.
        If y_pred is not a np.ndarray
    ValueError
        If there is only one class for AUROC and AP calculations.

    """
    if not isinstance(metrics, (list, np.ndarray)):
        raise TypeError("Input metrics should be a list of strings")
    if len(metrics) == 0 or not isinstance(metrics[0], str):
        raise TypeError("Input metrics should be a list of strings")

    if not isinstance(y_pred, np.ndarray):
        raise TypeError(f"Input y_pred should be a np.ndarray, but got {type(y_pred)}")

    if not isinstance(y, np.ndarray):
        raise TypeError(f"Input y should be a np.ndarray, but got {type(y)}")

    # Ensure y contains at least 2 classes for binary ranking/precision metrics
    if len(np.unique(y)) < 2:
        binary_metrics = {"roc_auc", "auroc", "ap"}
        if any(m in binary_metrics for m in metrics):
            raise ValueError(
                "Only one class present in y_true. "
                "AUROC and AP score are not defined in that case."
            )
    if y_pred.ndim == 2:
        y_pred = y_pred[:, 1]

    scores = np.zeros(len(metrics))

    for i, metric_name in enumerate(metrics):
        spec = MetricRegistry.get(metric_name)

        if spec.response_method == "predict_dist":
            raise ValueError(
                f"'{metric_name}' requires a full predictive distribution "
                "and cannot be computed from a flat prediction array via "
                "compute_metrics(); it is only usable as a cross-validation "
                "scorer via build_scorers()."
            )

        if spec.needs_threshold:
            scores[i] = float(spec.func(y, np.round(y_pred)))
        else:
            scores[i] = float(spec.func(y, y_pred))

    return scores


def bootstrap_confidence_intervals(
    metrics: list[str],
    y_true: npt.NDArray,
    y_pred: npt.NDArray,
    n_bootstraps: int = 1000,
    ci_level: float = 0.95,
    random_state: int | np.random.Generator | None = None,
) -> dict[str, dict[str, float]]:
    """
    Compute non-parametric bootstrap confidence intervals for evaluation metrics.

    Resamples paired target labels and predictions with replacement, evaluating
    all specified metrics across bootstrap iterations. Edge cases where resamples
    lack sufficient class diversity (e.g., single-class bootstrap draws) are safely
    caught and skipped.

    Parameters
    ----------
    metrics : list[str]
        Metric identifier keys registered in `MetricRegistry`
        (e.g., `["roc_auc", "log_loss"]`).
    y_true : npt.NDArray
        Ground truth binary target labels of shape (n_samples,).
    y_pred : npt.NDArray
        Predicted probabilities or decision values of shape (n_samples,) or
        (n_samples, 2).
    n_bootstraps : int, default=1000
        Number of bootstrap resampling iterations.
    ci_level : float, default=0.95
        Confidence level for the calculated interval bounds (e.g., 0.95 for 95% CI).
    random_state : int, np.random.Generator, or None, default=None
        Seed or random generator instance to ensure reproducible resampling.

    Returns
    -------
    ci_results : dict[str, dict[str, float]]
        Dictionary mapping each metric key to a dictionary containing:
        - ``"point_estimate"``: Metric score computed on the original dataset.
        - ``"ci_lower"``: Lower confidence boundary.
        - ``"ci_upper"``: Upper confidence boundary.

    Raises
    ------
    ValueError
        If ``ci_level`` is not strictly between 0.0 and 1.0, or if all bootstrap
        iterations fail due to severe class imbalance.

    """
    if not (0.0 < ci_level < 1.0):
        raise ValueError(f"ci_level must be between 0.0 and 1.0, got {ci_level}")

    y_true_arr = np.asarray(y_true).ravel()
    y_pred_arr = np.asarray(y_pred)

    y_pred_arr = y_pred_arr[:, 1] if y_pred_arr.ndim == 2 else y_pred_arr.ravel()

    n_samples = len(y_true_arr)
    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )

    # 1. Point estimates on the unresampled original data
    point_estimates = compute_metrics(metrics, y_true_arr, y_pred_arr)

    # 2. Resampling loop
    bootstrapped_scores: list[npt.NDArray] = []

    for _ in range(n_bootstraps):
        boot_idx = rng.integers(0, n_samples, size=n_samples)
        y_boot = y_true_arr[boot_idx]
        p_boot = y_pred_arr[boot_idx]

        # Ensure resample contains at least two classes for binary metrics
        # (e.g., ROC AUC / log loss)
        if len(np.unique(y_boot)) < 2:
            continue

        try:
            scores = compute_metrics(metrics, y_boot, p_boot)
            bootstrapped_scores.append(scores)
        except Exception:
            # Skip invalid iterations
            # (e.g., divide-by-zero or numerical errors in spline fits)
            continue

    if not bootstrapped_scores:
        raise ValueError(
            "All bootstrap iterations failed to compute valid metrics. "
            "Check dataset sample size or severe class imbalance."
        )

    # Matrix shape: (n_valid_bootstraps, n_metrics)
    scores_matrix = np.array(bootstrapped_scores)

    # 3. Compute Percentile Boundaries
    alpha = (1.0 - ci_level) / 2.0
    lower_percentile = alpha * 100.0
    upper_percentile = (1.0 - alpha) * 100.0

    lower_bounds = np.percentile(scores_matrix, lower_percentile, axis=0)
    upper_bounds = np.percentile(scores_matrix, upper_percentile, axis=0)

    # 4. Format Output Structure
    ci_results: dict[str, dict[str, float]] = {}
    for idx, metric_name in enumerate(metrics):
        ci_results[metric_name] = {
            "point_estimate": float(point_estimates[idx]),
            "ci_lower": float(lower_bounds[idx]),
            "ci_upper": float(upper_bounds[idx]),
        }

    return ci_results
