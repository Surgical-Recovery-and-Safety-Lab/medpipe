"""
Core metric functions module.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
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

from medpipe._types import Labels
from medpipe.metrics.registry import MetricRegistry, MetricSpec

if TYPE_CHECKING:
    import numpy.typing as npt

# ------------------------------------------------------------------------------
# STANDALONE METRIC FUNCTIONS
# ------------------------------------------------------------------------------


def ici_score(y: Labels, y_pred: npt.NDArray) -> float:
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


# ------------------------------------------------------------------------------
# DEFAULT METRIC REGISTRATIONS
# ------------------------------------------------------------------------------

_DEFAULT_METRICS = [
    MetricSpec("accuracy", accuracy_score, "predict", "Accuracy", "accuracy"),
    MetricSpec("precision", precision_score, "predict", "Precision", "precision"),
    MetricSpec("recall", recall_score, "predict", "Recall", "recall"),
    MetricSpec("f1", f1_score, "predict", "F1", "f1"),
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
    metrics: list[str] | npt.NDArray, y: Labels, y_pred: npt.NDArray
) -> npt.NDArray:
    """
    Computes metrics based on predicted data.

    Scores are located at the same index as in the metrics array.

    Parameters
    ----------
    metrics : list[str]
        List of metrics to use.
    y : Labels
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
    y_labels = np.round(y_pred)

    for i, metric_name in enumerate(metrics):
        spec = MetricRegistry.get(metric_name)

        if "predict_proba" in spec.response_method:
            scores[i] = float(spec.func(y, y_pred))
        else:
            scores[i] = float(spec.func(y, y_labels))

    return scores


def bootstrap_confidence_intervals(
    metrics: list[str],
    y_true: Labels,
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
    y_true : Labels
        Ground truth binary target labels of shape (n_samples,).
    y_pred : npt.NDArray
        Predicted probabilities or decision values of shape (n_samples,) or (n_samples, 2).
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

    if y_pred_arr.ndim == 2:
        y_pred_arr = y_pred_arr[:, 1]
    else:
        y_pred_arr = y_pred_arr.ravel()

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
