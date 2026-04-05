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

from medpipe._types import CI, CIDict, FProbas, Labels, MetricDict
from medpipe.utils.exceptions import array_check
from medpipe.utils.logger import print_message

if TYPE_CHECKING:
    import logging

    import numpy.typing as npt

SCRIPT_NAME = "metrics/core"


def print_metrics(
    metric_dict: MetricDict,
    label_list: list[str],
    logger: logging.Logger | None = None,
) -> None:
    """
    Prints the metrics on the terminal.

    Parameters
    ----------
    metric_dict : MetricDict
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
    n_it = len(label_list)  # Number of print iterations

    for i in range(n_it):
        # If label_list is a list
        print_message(f"  {label_list[i]} metrics:", logger, SCRIPT_NAME)

        print_message(
            f"    Accuracy: {metric_dict["accuracy"][i]:.3f}", logger, SCRIPT_NAME
        )
        print_message(f"    F1: {metric_dict["f1"][i]:.3f}", logger, SCRIPT_NAME)
        print_message(
            f"    Precision: {metric_dict["precision"][i]:.3f}", logger, SCRIPT_NAME
        )
        print_message(
            f"    Recall: {metric_dict["recall"][i]:.3f}", logger, SCRIPT_NAME
        )
        print_message(
            f"    Log loss: {metric_dict["log_loss"][i]:.3f}", logger, SCRIPT_NAME
        )
        print_message(f"    AUROC: {metric_dict["auroc"][i]:.3f}", logger, SCRIPT_NAME)
        print_message(f"    AP: {metric_dict["ap"][i]:.3f}", logger, SCRIPT_NAME)


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


def compute_all_CI(model_metrics: MetricDict, metric_list: list[str] = []) -> CIDict:
    """
    Computes the confidence intervals for all metrics.

    Parameters
    ----------
    model_metrics : MetricDict
        Model metrics for different folds.
    metric_list : list[str], default: []
        List of metrics to calculate confidence interval.
    **kwargs
        Extra arguments for the compute_CI function.

    Returns
    -------
    ci_dict : CIDict
        Dictionary containing the metric value and confidence intervals.
        The keys are the name of the metrics and the values are a tuple with
        first element the metric value, second the lower bound, and third the
        upper bound.

    """
    ci_dict = {}  # Empty dict to contain the confidence intervals for metrics
    metrics = next(iter(model_metrics.values())).keys()

    if metric_list == []:
        # Default to all metrics if not specified
        metric_list = list(metrics)

    for metric in metrics:
        if metric == "roc" or metric == "prc" or metric not in metric_list:
            # Skip ROC, PRC, and metrics not in the given list
            continue
        metric_values = extract_metric(model_metrics, metric)
        ci_dict.update({metric: compute_CI(metric_values)})

    return ci_dict


def compute_CI(data: npt.ArrayLike | npt.NDArray) -> CI:
    """
    Computes the confidence interval of the data.

    The CI is calculated using the Student's t-distribution.

    Parameters
    ----------
    data : npt.ArrayLike
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
    arr_data = np.asarray(data)

    if len(arr_data.shape) == 1:
        # Make sure there are 2 dimensions
        arr_data = np.expand_dims(arr_data, 1)

    mean_arr = np.zeros(arr_data.shape[1])
    lower_b_arr = np.zeros(arr_data.shape[1])
    upper_b_arr = np.zeros(arr_data.shape[1])

    for i in range(arr_data.shape[1]):
        mean_arr[i] = np.mean(arr_data[:, i])
        std_err = sem(arr_data[:, i])

        lower_b_arr[i], upper_b_arr[i] = t.interval(
            0.95, len(arr_data[:, i]) - 1, loc=mean_arr[i], scale=std_err
        )

    return mean_arr, lower_b_arr, upper_b_arr


def extract_metric(model_metrics: MetricDict, metric_name: str) -> list[float]:
    """
    Extracts the desired metric from each fold in the metric dictionary.

    Parameters
    ----------
    model_metrics : MetricDict
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
    multilabel = False
    average = "binary"

    if len(y_true.shape) > 1:
        # Multilabel situation
        multilabel = True
        average = None

    if "accuracy" in metric_list:
        # Deal with accuracy separately to get accuracy for each label
        values = []  # Store values

        if multilabel:
            # Iterate over each label and add individual label accuracy
            for i in range(y_true.shape[1]):
                values.append(skl.metrics.accuracy_score(y_true[:, i], y_pred[:, i]))

        values.append(skl.metrics.accuracy_score(y_true, y_pred))
        metric_dict.update({"accuracy": values})
        metric_list.remove("accuracy")

    for metric in metric_list:
        values = []  # Create empty list to hold the metrics for each label
        match metric:
            case "f1":
                values.append(skl.metrics.f1_score(y_true, y_pred, average=average))
                if multilabel:
                    values = np.append(
                        values,
                        skl.metrics.f1_score(y_true, y_pred, average="weighted"),
                    )
                metric_dict.update({metric: values})

            case "precision":
                values.append(
                    skl.metrics.precision_score(
                        y_true, y_pred, average=average, zero_division=0.0
                    )
                )
                if multilabel:
                    values = np.append(
                        values,
                        skl.metrics.precision_score(
                            y_true, y_pred, average="weighted", zero_division=0.0
                        ),
                    )
                metric_dict.update({metric: values})

            case "recall":
                values.append(skl.metrics.recall_score(y_true, y_pred, average=average))
                if multilabel:
                    values = np.append(
                        values,
                        skl.metrics.recall_score(y_true, y_pred, average="weighted"),
                    )
                metric_dict.update({metric: values})

            case _:
                raise ValueError(f"{metric} is an unrecognised metric")

    return metric_dict


def compute_score_metrics(
    metric_list: list[str], y_true: Labels, y_pred_proba: FProbas
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
    y_pred_proba : FProbas
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
