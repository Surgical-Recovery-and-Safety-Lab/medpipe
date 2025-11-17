"""
Core metric functions module.

This module provides functions to compute and print metrics.

Functions:
- print_metrics: prints the numerical metrics.
- print_metrics_CI: prints numerical metrics with their confidence intervals.
- compute_all_CI: computes the confidence interval for all metrics.
- compute_CI: computes the confidence interval.
- extract_metric : extracts a metric for each fold.
"""

import numpy as np
from scipy.stats import bootstrap

from pyrisk.utils.exceptions import array_check


def print_metrics(metric_dict) -> None:
    """
    Prints the metrics on the terminal.

    Parameters
    ----------
    metric_dict : dict[str, float or tuple(array-like)]
        Dictionary of the model performance for one fold.
        Keys are the metric name and values are the metric value.
        The test metrics used are:
         - accuracy
         - f1
         - precision
         - recall
         - auroc (Area Under Receiver Operator Characteristic)

    Returns
    -------
    None
        Nothing is returned.

    """
    print(f"    Accuracy: {metric_dict["accuracy"]:.3f}")
    print(f"    F1: {metric_dict["f1"]:.3f}")
    print(f"    Precision: {metric_dict["precision"]:.3f}")
    print(f"    Recall: {metric_dict["recall"]:.3f}")
    print(f"    AUROC: {metric_dict["auroc"]:.3f}")


def print_metrics_CI(ci_dict):
    """
    Prints the metrics with their confidence intervals.

    Parameters
    ----------
    ci_dict : dict[str, tuple(float, float, float)]
        Dictionary containing the metric value and confidence intervals.
        The keys are the name of the metrics and the values are a tuple with
        first element the metric value, second the lower bound, and third the
        upper bound.

    Returns
    -------
    None
        Nothing is returned.

    """
    for metric in ci_dict.keys():
        stat, lower_b, upper_b = ci_dict[metric]
        print(f"  {metric.capitalize()}: {stat:.3f} CI [{lower_b:.3f}, {upper_b:.3f}]")


def compute_all_CI(model_metrics, **kwargs):
    """
    Computes the confidence intervals for all metrics.

    Parameters
    ----------
    model_metrics : dict[int, dict[str, float or tuple(array-like)]]
        Model metrics for different folds.
    **kwargs
        Extra arguments for the compute_CI function.

    Returns
    -------
    ci_dict : dict[str, tuple(float, float, float)]
        Dictionary containing the metric value and confidence intervals.
        The keys are the name of the metrics and the values are a tuple with
        first element the metric value, second the lower bound, and third the
        upper bound.

    """
    ci_dict = {}  # Empty dict to contain the confidence intervals for metrics
    metrics = next(iter(model_metrics.values())).keys()

    for metric in metrics:
        if metric == "roc" or metric == "prc":
            # Skip ROC and PRC metrics
            continue
        metric_list = extract_metric(model_metrics, metric)
        ci_dict.update({metric: compute_CI(metric_list, **kwargs)})

    return ci_dict


def compute_CI(data, statistic=np.mean, **kwargs):
    """
    Computes the confidence interval using the bootstrap method.

    Parameters
    ----------
    data : array-like
        Data on which to compute the confidence interval.
    statistic : callable, default: np.mean
        Statistic for which the confidence interval is calculated.
    **kwargs
        Extra arguments for the scipy.stats.bootstrap method.

    Returns
    -------
    stat : float
        Value of the statistic that has been calculated.
    lower_b : float
        Lower bound of the confidence interval.
    upper_b : float
        Upper bound of the confidence interval.

    Raises
    ------
    TypeError
        If data is not array-like

    """
    array_check(data)

    bootstrap_res = bootstrap((data,), statistic=statistic, **kwargs)

    lower_b, upper_b = bootstrap_res.confidence_interval

    return statistic(data), lower_b, upper_b


def extract_metric(model_metrics, metric_name):
    """
    Extracts the desired metric from each fold in the metric dictionary.

    Parameters
    ----------
    model_metrics : dict[int, dict[str, float or tuple(array-like)]]
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
