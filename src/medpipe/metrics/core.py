"""
Core metric functions module.

This module provides functions to compute and print metrics.

Functions:
- print_metrics: prints the numerical metrics.
- print_metrics_CI: prints numerical metrics with their confidence intervals.
- compute_all_CI: computes the confidence interval for all metrics.
- compute_CI: computes the confidence interval.
- extract_metric : extracts a metric for each fold.
- compute_pred_metrics : computes the metrics that require the prediction labels.
- compute_score_metrics : computes the metrics that require the score.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import sklearn as skl
from scipy.stats import sem, t

from medpipe._types import CI, CIDict, FullProba, Labels, MetricDict, ModelMetrics
from medpipe.utils.exceptions import array_check
from medpipe.utils.logger import print_message

if TYPE_CHECKING:
    import logging

    import numpy.typing as npt

SCRIPT_NAME = "metrics/core"


def print_metrics(
    metric_dict: dict[str, list[float]],
    label_list: list[str],
    logger: logging.Logger | None = None,
) -> None:
    """
    Prints the metrics on the terminal.

    Parameters
    ----------
    metric_dict : dict[str, list[float]]
        Dictionary of the model performance for one fold.
        Keys are the metric name and values are the metric value.
        The test metrics used are:
         - accuracy
         - f1
         - precision
         - recall
         - log_loss
         - auroc (Area Under Receiver Operator Characteristic)
         - ap (Average Precision)
    label_list : list[str]
        List of predicted labels.
    logger : logging.Logger | None, default: None
        Logger object to log prints. If None print to terminal.

    Returns
    -------
    None
        Nothing is returned.

    """
    # Get all keys except ones we don't want to print (like curves)
    printable_metrics = [k for k in metric_dict.keys() if k not in ["roc", "prc"]]

    for i, label in enumerate(label_list):
        print_message(f"  {label} metrics:", logger, SCRIPT_NAME)
        for metric in printable_metrics:
            val = metric_dict[metric][i]
            # Dynamic spacing and capitalization
            name = metric.replace("_", " ").capitalize()
            print_message(f"    {name}: {val:.3f}", logger, SCRIPT_NAME)


def print_metrics_CI(
    ci_dict: CIDict, label_list: list[str], logger: logging.Logger | None = None
) -> None:
    """
    Prints the metrics with their confidence intervals.

    Parameters
    ----------
    ci_dict : CIDict
        Dictionary containing the metric value and confidence intervals.
        The keys are the name of the metrics and the values are a tuple with
        first element the metric value, second the lower bound, and third the
        upper bound.
    label_list : list[str]
        List of predicted labels.
    logger : logging.Logger, default: None
        Logger object to log prints. If None print to terminal.

    Returns
    -------
    None
        Nothing is returned.

    """
    n_it = len(label_list)  # Number of print iterations

    for i in range(n_it):
        # If label_list is a list
        if i < len(label_list):
            print_message(f"  {label_list[i]} metrics:", logger, SCRIPT_NAME)
        else:
            print_message("  Global metrics:", logger, SCRIPT_NAME)

        for metric in ci_dict.keys():
            stat, lb, ub = ci_dict[metric]
            print_message(
                f"    {metric.capitalize()}: {stat[i]:.3f} CI [{lb[i]:.3f}, {ub[i]:.3f}]",
                logger,
                SCRIPT_NAME,
            )


def compute_all_CI(model_metrics: ModelMetrics, metric_list: list[str] = []) -> CIDict:
    """
    Computes the confidence intervals for all metrics.

    Parameters
    ----------
    model_metrics : ModelMetrics
        Model metrics for different folds.
    metric_list : list[str], default: []
        List of metrics to calculate confidence interval.

    Returns
    -------
    ci_dict : CIDict
        Dictionary containing the metric value and confidence intervals.
        The keys are the name of the metrics and the values are a tuple with
        first element the metric value, second the lower bound, and third the
        upper bound.

    """
    ci_dict = {}
    # Get all available metric keys from the first fold
    all_metrics = list(next(iter(model_metrics.values())).keys())

    if not metric_list:
        metric_list = all_metrics

    for metric in metric_list:
        # Skip curve-based metrics which aren't scalar values
        if metric in ["roc", "prc"]:
            continue

        # Convert to a 2D array: (n_folds, n_labels)
        metric_values = np.array(extract_metric(model_metrics, metric))

        # compute_CI now handles the (n_folds, n_labels) shape entirely in one go
        ci_dict[metric] = compute_CI(metric_values)

    return ci_dict


def compute_CI(data: list[float] | npt.NDArray) -> CI:
    """
    Computes the confidence interval of the data.

    The CI is calculated using the Student's t-distribution.

    Parameters
    ----------
    data : list[float] | npt.NDArray
        Data on which to compute the confidence interval of shape (n_samples, n_sets).

    Returns
    -------
    mean_arr : npt.NDArray
        Mean values of shape (n_sets,).
    lower_b_arr : npt.NDArray
        Lower bound of the confidence intervals of shape (n_sets,).
    upper_b_arr : npt.NDArray
        Upper bound of the confidence intervals of shape (n_sets,).

    Raises
    ------
    TypeError
        If data is not array-like

    """
    array_check(data)
    arr_data = np.atleast_2d(data)  # Handles (n,) or (n, m) automatically

    # Vectorized mean and standard error
    means = np.mean(arr_data, axis=0)
    std_errs = sem(arr_data, axis=0)
    n = arr_data.shape[0]

    # t.interval supports array-like inputs for loc and scale
    lower, upper = t.interval(0.95, n - 1, loc=means, scale=std_errs)

    return means, lower, upper


def extract_metric(model_metrics: ModelMetrics, metric_name: str) -> list[float]:
    """
    Extracts the desired metric from each fold in the metric dictionary.

    Parameters
    ----------
    model_metrics : ModelMetrics
        Model metrics for different folds.
    metric_name : str
        Name of the metric to extract.

    Returns
    -------
    metric_list : list[float]
        List containing the metric values for each fold.

    """
    metric_list = []

    for metrics in model_metrics.values():
        # Loop through values directly to get the desired metric
        metric_list.append(metrics[metric_name])

    return metric_list


def compute_pred_metrics(
    metric_list: list[str], y_true: Labels, y_pred: Labels
) -> MetricDict:
    """
    Computes the metrics that require the prediction labels.

    Parameters
    ----------
    metric_list : list[str]
        List of metrics. Possible values:
         - accuracy
         - f1
         - precision
         - recall
    y_true : Labels
        Ground truth labels of shape (n_samples, n_classes).
    y_pred : Labels
        Predicted labels of shape (n_samples, n_classes).

    Returns
    -------
    metric_dict : MetricDict
        Dictionary of the metrics. The keys are the name of the metric
        and the values are the computed metric value.
        If multilabel then the list contains the value for each class and
        the last value is the average value.

    Raises
    ------
    ValueError
        If the metric is not recognised.

    """
    metric_dict = {}
    is_multilabel = y_true.ndim > 1

    for metric in metric_list:
        if metric == "accuracy":
            if is_multilabel:
                # Vectorized accuracy per label
                acc_per_label = (y_true == y_pred).mean(axis=0)
                overall_acc = skl.metrics.accuracy_score(y_true, y_pred)
                metric_dict["accuracy"] = np.append(acc_per_label, overall_acc)
            else:
                metric_dict["accuracy"] = [skl.metrics.accuracy_score(y_true, y_pred)]
            continue

        # Map string names to skl functions
        func = getattr(skl.metrics, f"{metric}_score")

        if is_multilabel:
            # Returns array of scores for each label
            scores = func(y_true, y_pred, average=None, zero_division=0.0)
            # Append the weighted average as the last element
            weighted = func(y_true, y_pred, average="weighted", zero_division=0.0)
            metric_dict[metric] = np.append(scores, weighted)
        else:
            metric_dict[metric] = [
                func(y_true, y_pred, average="binary", zero_division=0.0)
            ]

    return metric_dict


def compute_score_metrics(
    metric_list: list[str], y_true: Labels, y_pred_proba: FullProba
) -> MetricDict:
    """
    Computes the metrics that require the score.

    Parameters
    ----------
    metric_list : list[str]
        List of metrics. Possible values:
         - roc
         - auroc (area under the curve)
         - prc (precision-recall curve)
         - ap (average precision)
         - log_loss
    y_true : Labels
        Ground truth labels of shape (n_samples, n_classes).
    y_pred_proba : FullProba
        Full predicted probabilities.

    Returns
    -------
    metric_dict : MetricDict
        Dictionary of the metrics. The keys are the name of the metric
        and the values are the computed metric values.
        If multilabel then the list contains the value for each class.

    Raises
    ------
    ValueError
        If the metric is not recognised.

    """
    metric_dict = {}
    multilabel = True

    if len(y_true.shape) == 1:
        # Make into a list
        y_true = np.expand_dims(y_true, 1)
        if len(y_pred_proba.shape) == 2:
            y_pred_proba = np.expand_dims(y_pred_proba, 0)
        multilabel = False

    for metric in metric_list:
        values = []  # Create empty list to hold the metrics for each label
        for i, scores in enumerate(y_pred_proba):
            match metric:
                case "roc":
                    values.append(skl.metrics.roc_curve(y_true[:, i], scores[:, 1]))
                case "auroc":
                    values.append(skl.metrics.roc_auc_score(y_true[:, i], scores[:, 1]))
                case "prc":
                    values.append(
                        skl.metrics.precision_recall_curve(y_true[:, i], scores[:, 1])
                    )
                case "ap":
                    values.append(
                        skl.metrics.average_precision_score(y_true[:, i], scores[:, 1])
                    )
                case "log_loss":
                    values.append(skl.metrics.log_loss(y_true[:, i], scores[:, 1]))
                case _:
                    raise ValueError(f"{metric} is an unrecognised metric")

            metric_dict.update({metric: values})

        if multilabel:
            if metric == "ap" or metric == "auroc" or metric == "log_loss":
                # Add the average log loss, AUROC, and AP score
                metric_dict[metric].append(np.mean(metric_dict[metric]))
    return metric_dict
