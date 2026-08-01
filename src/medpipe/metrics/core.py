"""
Core metric functions module.

This module provides functions to compute and print metrics.

Functions:
- ici_score: Computes the integrated calibration index using a
    spline-based calibration curve.
- build_scorers: Build the dictionary of scorers for cross-validation.
- compute_metrics: Computes metrics based on predicted data.
- print_metrics: prints the numerical metrics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
from ml_insights import SplineCalib
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
    if (
        not isinstance(metrics, (list, np.ndarray))
        or not metrics
        or not isinstance(metrics[0], str)
    ):
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

    """
    if (
        not isinstance(metrics, (list, np.ndarray))
        or not metrics
        or not isinstance(metrics[0], str)
    ):
        raise TypeError("Input metrics should be a list of strings")

    if not isinstance(y_pred, np.ndarray):
        raise TypeError(f"Input y_pred should be a np.ndarray, but got {type(y_pred)}")

    if not isinstance(y, np.ndarray):
        raise TypeError(f"Input y should be a np.ndarray, but got {type(y)}")

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


def print_metrics(
    results: npt.NDArray,
    metrics: list[str],
) -> None:
    """
    Prints the metrics on the terminal.

    Metric names are read in order and the index should
    correspond to the value at the same index in results.

    Parameters
    ----------
    results : npt.NDArray
        Array of metric values to print.
    metrics : list[str]
        List of metric names to print.

    Returns
    -------
    None
        Nothing is returned.

    Raises
    ------
    TypeError
        If results is not a np.ndarray.
        If metrics is not a list.
    ValueError
        If metrics and results are not the same length.
        If metrics contains an invalid metric.

    """
    if not isinstance(results, np.ndarray):
        raise TypeError(
            f"Input results should be a np.ndarray, but got {type(results)}"
        )

    if not isinstance(metrics, list):
        raise TypeError(
            f"Input metrics should be a list of strings, but got {type(metrics)}"
        )

    if len(metrics) != len(results):
        raise ValueError(
            "Input results and metrics should have the same length, "
            f"but got {len(results)} and {len(metrics)}"
        )

    for i, metric_name in enumerate(metrics):
        spec = MetricRegistry.get(metric_name)
        print(f"{spec.display_name}: {results[i]:.3f}")


def compute_strata_metrics(
    metrics: list[str],
    strata_idx: list[npt.NDArray],
    y: Labels,
    y_pred: npt.NDArray,
) -> list[npt.NDArray]:
    """
    Computes metrics for the different strata.

    Scores for each strata are ordered as the strata_idx.
    Scores are located at the same index as in the metrics array.

    Parameters
    ----------
    metrics : list[str]
        List of metrics to use.
    strata_idx : list[npt.NDArray]
        List of indices for each strata.
    y : Labels
        Ground truth labels.
    y_pred : npt.NDArray
        Predictions from the model of shape
        (n_samples,) or (n_samples, 2).

    Returns
    -------
    scores : list[npt.NDArray]
        List of score arrays of shape (n_metrics,)
        for each strata.

    Raises
    ------
    TypeError
        If strata_idx is not a list[np.ndarray] of integers.

    """
    if not isinstance(strata_idx, list):
        raise TypeError(
            f"Input strata_idx must be a list, got {type(strata_idx).__name__}"
        )

    for idx, arr in enumerate(strata_idx):
        if not isinstance(arr, np.ndarray):
            raise TypeError(
                f"Element at index {idx} must be a np.ndarray, got {type(arr).__name__}"
            )
        if arr.dtype.kind not in ("i", "u"):
            raise TypeError(
                f"Array at index {idx} must contain integers, got dtype '{arr.dtype}'"
            )

    scores = []
    for idx in strata_idx:
        scores.append(compute_metrics(metrics, y[idx], y_pred[idx]))

    return scores
