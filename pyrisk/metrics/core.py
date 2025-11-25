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

import numpy as np
import sklearn as skl
from scipy.stats import bootstrap

from pyrisk.utils.exceptions import array_check
from pyrisk.utils.logger import print_message

SCRIPT_NAME = "metrics/core"


def print_metrics(metric_dict, label_list, logger=None) -> None:
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
         - ap (Average Precision)
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
    if n_it > 1:
        n_it += 1  # Add one for the global values if multilabel

    for i in range(n_it):
        # If label_list is a list
        if i < len(label_list):
            print_message(f"  {label_list[i]} metrics:", logger, SCRIPT_NAME)
        else:
            print_message("  Global metrics:", logger, SCRIPT_NAME)

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
        print_message(f"    AUROC: {metric_dict["auroc"][i]:.3f}", logger, SCRIPT_NAME)
        print_message(f"    AP: {metric_dict["ap"][i]:.3f}", logger, SCRIPT_NAME)


def print_metrics_CI(ci_dict, label_list, logger=None):
    """
    Prints the metrics with their confidence intervals.

    Parameters
    ----------
    ci_dict : dict[str, tuple(float, float, float)]
        Dictionary containing the metric value and confidence intervals.
        The keys are the name of the metrics and the values are a tuple with
        first element the metric value, second the lower bound, and third the
        upper bound.
    logger : logging.Logger, default: None
        Logger object to log prints. If None print to terminal.

    Returns
    -------
    None
        Nothing is returned.

    """
    n_it = len(label_list)  # Number of print iterations
    if n_it > 1:
        n_it += 1  # Add one for the global values if multilabel

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

    return statistic(data, axis=0), lower_b, upper_b


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


def compute_pred_metrics(metric_list, y_true, y_pred):
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
    y_true : array-like of shape (n_samples, n_classes)
        Ground truth labels.
    y_pred : array-like of shape (n_samples, n_classes)
        Predicted labels.

    Returns
    -------
    metric_dict : dict[str, list[float]]
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
            for i in range(len(y_true.shape)):
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
                        skl.metrics.f1_score(y_true, y_pred, average="macro"),
                    )
                metric_dict.update({metric: values})

            case "precision":
                values.append(
                    skl.metrics.precision_score(y_true, y_pred, average=average)
                )
                if multilabel:
                    values = np.append(
                        values,
                        skl.metrics.precision_score(y_true, y_pred, average="macro"),
                    )
                metric_dict.update({metric: values})

            case "recall":
                values.append(skl.metrics.recall_score(y_true, y_pred, average=average))
                if multilabel:
                    values = np.append(
                        values,
                        skl.metrics.recall_score(y_true, y_pred, average="macro"),
                    )
                metric_dict.update({metric: values})

            case _:
                raise ValueError(f"{metric} is an unrecognised metric")

    return metric_dict


def compute_score_metrics(metric_list, y_true, y_pred_proba):
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
    y_true : array-like of shape (n_samples, n_classes)
        Ground truth labels.
    y_pred_proba : np.array or list[np.array]
        Predicted scores for each label.

    Returns
    -------
    metric_dict : dict[str, list[float or tuple]]
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

    if type(y_pred_proba) is not type(list()):
        # Make into a list
        y_pred_proba = [y_pred_proba]
        y_true = np.expand_dims(y_true, 1)
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
                case _:
                    raise ValueError(f"{metric} is an unrecognised metric")

            metric_dict.update({metric: values})

        if multilabel:
            if metric == "ap" or metric == "auroc":
                # Add the average AUROC of AP score
                metric_dict[metric].append(np.mean(metric_dict[metric]))
    return metric_dict
