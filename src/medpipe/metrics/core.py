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
    brier_score_loss,
    f1_score,
    get_scorer,
    log_loss,
    make_scorer,
    mean_absolute_error,
    roc_auc_score,
    root_mean_squared_error,
)

from medpipe._types import Labels
from medpipe.utils.exceptions import array_dim_check

if TYPE_CHECKING:
    import numpy.typing as npt

SCRIPT_NAME = "metrics/core"


def ici_score(
    y: Labels,
    y_pred: npt.NDArray,
) -> float:
    """
    Computes the integrated calibration index using a spline-based
    calibration curve.

    Parameters
    ----------
    y : Labels
        Ground truth labels.
    y_pred : npt.NDArray
        Predictions from the model of shape
        (n_samples,) or (n_samples, 2).

    Returns
    -------
    ici : float
        Integrated calibration index value.

    Raises
    ------
    ValueError
        If spline predicted probabilities are None.

    """
    if y_pred.ndim == 2:
        y_pred = y_pred[:, 1]  # Get only positive probabilities

    # Create and fit spline
    spline = SplineCalib(logodds_scale=True)
    spline.fit(y_pred, y)
    smoothed_outputs = spline.predict(y_pred)

    if smoothed_outputs is not None:
        return float(np.mean(np.abs(smoothed_outputs - y_pred)))
    else:
        raise ValueError("Error predicting probabilities with spline")


# Define metric registery
METRIC_MAPPING = {  #  metric name, scorer, function to use, print name
    "accuracy": ("accuracy", accuracy_score, "predict", "Accuracy"),
    "log_loss": ("neg_log_loss", log_loss, "predict_proba", "Log loss"),
    "brier_score": (
        "neg_brier_score",
        brier_score_loss,
        "predict_proba",
        "Brier score",
    ),
    "f1": ("f1", f1_score, "predict", "F1"),
    "roc_auc": (
        "roc_auc",
        roc_auc_score,
        ("decision_function", "predict_proba"),
        "AUROC",
    ),
    "auroc": (
        "roc_auc",
        roc_auc_score,
        ("decision_function", "predict_proba"),
        "AUROC",
    ),
    "ici": ("ici", ici_score, "predict_proba", "ICI"),
    "rmse": ("root_mean_squared_error", root_mean_squared_error, ("predict"), "RMSE"),
    "mae": ("mean_absolute_error", mean_absolute_error, ("predict"), "MAE"),
}

METRICS = [key for key in METRIC_MAPPING.keys()]


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
        # Ensure metrics is not an empty list and is a list of strings
        raise TypeError("Input metrics should be a list of strings")

    scorers = {}  # Empty dict to contain scorers
    for metric in metrics:
        if metric == "ici":
            scorers[metric] = make_scorer(ici_score, response_method="predict_proba")

        elif metric in METRICS:
            scorers[metric] = get_scorer(METRIC_MAPPING[metric][0])
        else:
            expr = (
                f"{metric} was not found in available metric "
                f"list. Available metrics are {METRICS}"
            )
            raise ValueError(expr)
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
        # Ensure metrics is not an empty list and is a list of strings
        raise TypeError("Input metrics should be a list of strings")

    if not isinstance(y_pred, np.ndarray):
        raise TypeError(f"Input y_pred should be a np.ndarray, but got {type(y_pred)}")

    if not isinstance(y, np.ndarray):
        raise TypeError(f"Input y should be a np.ndarray, but got {type(y)}")

    if y_pred.ndim == 2:
        y_pred = y_pred[:, 1]  # Get only positive probabilities

    array_dim_check(y, y_pred)

    scores = np.zeros(len(metrics))  # Empty array to hold scores
    y_labels = np.round(y_pred)  # Get labels based on probabilities

    for i, metric in enumerate(metrics):
        try:
            method = METRIC_MAPPING[metric][2]
        except KeyError:
            expr = (
                f"{metric} was not found in available metric "
                f"list. Available metrics are {METRICS}"
            )
            raise ValueError(expr)

        if "predict_proba" in method:
            # Use the predictions to compute the score
            scores[i] = METRIC_MAPPING[metric][1](y, y_pred)

        else:
            scores[i] = METRIC_MAPPING[metric][1](y, y_labels)

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
        expr = f"Input results should be a np.ndarray, but got {type(results)}"
        raise TypeError(expr)

    if not isinstance(metrics, list):
        expr = f"Input metrics should be a list of strings, but got {type(metrics)}"
        raise TypeError(expr)

    if len(metrics) != len(results):
        expr = (
            "Input results and metrics should have the same length, "
            f"but got {len(results)} and {len(metrics)}"
        )
        raise ValueError(expr)

    for i, metric in enumerate(metrics):
        if metric not in METRICS:
            expr = (
                f"{metric} was not found in available metric "
                f"list. Available metrics are {METRICS}"
            )
            raise ValueError(expr)

        print(f"{METRIC_MAPPING[metric][3]}: {results[i]:.3f}")


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
    if not isinstance(strata_idx, list) and not isinstance(strata_idx[0], np.ndarray):
        raise TypeError("Input strata_idx should be a list[np.ndarray]")

    if not isinstance(strata_idx[0][0], int):
        raise TypeError("Input strata_idx should be a list[np.ndarray] of integers")

    scores = []
    for idx in strata_idx:
        scores.append(compute_metrics(metrics, y[idx], y_pred[idx]))

    return scores
