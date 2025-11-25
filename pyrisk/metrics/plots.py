"""
Plot functions module.

This module provides functions to plot results.

Functions:
- plot_from_display: Plot results from a Display class.
"""

import matplotlib.pyplot as plt
import numpy as np
import sklearn as skl

from pyrisk.utils.exceptions import array_check, array_dim_check


def plot_from_display(y_true, y_pred, display, **kwargs) -> None:
    """
    Plot results from a Display class.

    See sklearn.metrics for the Display classes and the from_predictions method.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth labels.
    y_pred : array-like of shape (n_samples,)
        Predicted labels.
    display : str {"roc", "confusion", "precision-recall"}
        Type of display class to use.
    **kwargs :
        Extra arguments for the display classes.

    Returns
    -------
    ValueError
        If display is not a valid option.

    """
    # Check that inputs are correct
    array_check(y_pred)
    array_check(y_true)

    if display not in ["roc", "confusion", "precision-recall"]:
        raise ValueError(f"Invalid display, got {display}")

    match display:
        case "roc":
            print("[INFO] Plotting ROC curve")
            display_fct = skl.metrics.RocCurveDisplay.from_predictions
        case "confusion":
            print("[INFO] Plotting confusion matrix")
            display_fct = skl.metrics.ConfusionMatrixDisplay.from_predictions
        case "precision-recall":
            print("[INFO] Plotting precision-recall curve")
            display_fct = skl.metrics.PrecisionRecallDisplay.from_predictions
        case _:
            # Default to ROC curve just in case
            display_fct = skl.metrics.RocCurveDisplay.from_predictions

    display_fct(y_true, y_pred, **kwargs)
    plt.show()


def plot_mean_ROC_curve(model_metrics, label_list, nb_points=500):
    """
    Plots the ROC curve of each fold and the mean ROC curve.

    Parameters
    ----------
    model_metrics : dict[int, dict[str, float or tuple(array-like)]]
        Model metrics for different folds.
    label_list : list[str]
        List of predicted labels.
    nb_points : int, default: 500
        Number of points for the mean ROC curve.

    Returns
    -------
    None
        Nothing is returned.

    Raises
    ------
    TypeError
        If nb_points is not an int.

    """

    if type(nb_points) is not type(0):
        raise TypeError(f"nb_points should be an int, but got {type(nb_points)}")

    n_it = len(label_list)  # Number of print iterations
    if n_it > 1:
        n_it += 1

    mean_fpr = np.linspace(0, 1, nb_points)
    global_tprs = []
    global_aucs = []

    for i in range(n_it):
        tprs = []
        aucs = []
        metric = ""

        for k, fold in enumerate(model_metrics.keys()):
            if i < len(label_list):
                # Setup for a single label
                metric = label_list[i]
                fpr, tpr, _ = model_metrics[fold]["roc"][i]
                tprs.append(np.interp(mean_fpr, fpr, tpr))
                aucs.append(model_metrics[fold]["auroc"][i])

            else:
                # Setup for the global label
                metric = "global"
                tprs = np.mean(global_tprs, axis=0)
                aucs = np.mean(global_aucs, axis=0)

            plt.plot(
                mean_fpr,
                tprs[k],
                lw=1,
                alpha=0.3,
                label=f"Fold number {fold} (AUC {aucs[k]:.3f})",
            )

        global_tprs.append(tprs)
        global_aucs.append(aucs)

        # Compute mean and std for all folds
        mean_tpr = np.mean(tprs, axis=0)
        mean_auroc = np.mean(aucs, axis=0)

        std_tprs = np.std(tprs, axis=0)
        std_auroc = np.std(aucs, axis=0)

        upper_std = np.minimum(mean_tpr + std_tprs, 1)
        lower_std = np.maximum(mean_tpr - std_tprs, 0)

        # Plot mean curve with shaded std
        plt.plot(
            mean_fpr,
            mean_tpr,
            color="b",
            alpha=0.8,
            lw=2,
            label=rf"Mean {metric} ROC curve (AUC {mean_auroc:.3f} $\pm$ {std_auroc:.3f})",
        )
        plt.fill_between(
            mean_fpr,
            lower_std,
            upper_std,
            color="grey",
            alpha=0.2,
            label=r"Mean $\pm$ 1 SD",
        )

        plt.xlim([-0.01, 1.01])
        plt.ylim([-0.01, 1.01])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"Cross-Validated {metric} ROC")
        plt.legend(loc="lower right")
        plt.show()


def plot_mean_PR_curve(model_metrics, label_list, nb_points=500):
    """
    Plots the PR curve of each fold and the mean PR curve.

    Parameters
    ----------
    model_metrics : dict[int, dict[str, float or tuple(array-like)]]
        Model metrics for different folds.
    label_list : list[str]
        List of predicted labels.
    nb_points : int, default: 500
        Number of points for the mean curve.

    Returns
    -------
    None
        Nothing is returned.

    Raises
    ------
    TypeError
        If nb_points is not an int.

    """

    if type(nb_points) is not type(0):
        raise TypeError(f"nb_points should be an int, but got {type(nb_points)}")

    n_it = len(label_list)  # Number of print iterations
    if n_it > 1:
        n_it += 1  # Add one for the global values if multilabel

    mean_precision = np.linspace(0, 1, nb_points)
    global_recalls = []
    global_aps = []

    for i in range(n_it):
        recalls = []
        aps = []
        metric = ""

        for k, fold in enumerate(model_metrics.keys()):
            if i < len(label_list):
                # Setup for a single label
                metric = label_list[i]
                precision, recall, _ = model_metrics[fold]["prc"][i]
                recalls.append(np.interp(mean_precision, precision, recall))
                aps.append(model_metrics[fold]["ap"][i])

            else:
                # Setup for the global label
                metric = "global"
                recalls = np.mean(global_recalls, axis=0)
                aps = np.mean(global_aps, axis=0)

            plt.plot(
                recalls[k],
                mean_precision,
                lw=1,
                alpha=0.5,
                label=f"Fold number {fold} (AP {aps[k]:.3f})",
            )

        global_recalls.append(recalls)
        global_aps.append(aps)

        # Compute mean and std for all folds
        mean_recall = np.mean(recalls, axis=0)
        mean_ap = np.mean(aps, axis=0)

        std_recall = np.std(recalls, axis=0)
        std_ap = np.std(aps, axis=0)

        upper_std = np.minimum(mean_recall + std_recall, 1)
        lower_std = np.maximum(mean_recall - std_recall, 0)

        # Plot mean curve with shaded std
        plt.plot(
            mean_recall,
            mean_precision,
            color="b",
            alpha=0.8,
            lw=2,
            label=rf"Mean {metric} PRC curve (AP {mean_ap:.3f} $\pm$ {std_ap:.3f})",
        )
        plt.fill_betweenx(
            mean_precision,
            lower_std,
            upper_std,
            color="grey",
            alpha=0.3,
            label=r"Mean $\pm$ 1 SD",
        )

        plt.xlim([-0.01, 1.01])
        plt.ylim([-0.01, 1.01])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Cross-Validated {metric} PRC")
        plt.legend(loc="upper right")
        plt.show()


def plot_metrics_CI(ci_dict, label_list):
    """
    Plots the metrics with confidence intrevals for each fold.

    Parameters
    ----------
    ci_dict : dict[str, tuple(float, float, float)]
        Dictionary containing the metric value and confidence intervals.
        The keys are the name of the metrics and the values are a tuple with
        first element the metric value, second the lower bound, and third the
        upper bound.
    label_list : list[str]
        List of predicted labels.

    Returns
    -------
    None
        Nothing is returned.

    """
    n_it = len(label_list)  # Number of iterations
    if n_it > 1:
        n_it += 1  # Add one for the global values if multilabel

    # Set up the figure and axis
    _, ax = plt.subplots(dpi=300)
    bar_width = 0.15
    colours = plt.get_cmap("Pastel1")
    index = np.arange(len(ci_dict.keys()))

    # Loop through each metric
    for i, values in enumerate(ci_dict.values()):
        for j in range(n_it):
            if j < len(label_list):
                label = label_list[j]
            else:
                label = "Global"

            value = values[0][j]
            lower_b = values[1][j]
            upper_b = values[2][j]

            # Plot the metric for each label with error bars (CI bounds)
            ax.bar(
                index[i] + (j * bar_width),
                value,
                bar_width,
                color=colours(j),
                label=label if i == 0 else "",
                zorder=3,
            )
            # Error bars for confidence interval
            ax.errorbar(
                index[i] + (j * bar_width),
                value,
                yerr=[[value - lower_b], [upper_b - value]],
                fmt="none",
                color="black",
                capsize=2,
                zorder=2,
            )

    # Customize the chart
    ax.set_xlabel("Metrics")
    ax.set_ylabel("Values")
    ax.set_title("Metrics Comparison with Confidence Intervals")
    ax.set_ylim(ymin=0, ymax=1)

    # Set the x-ticks to be at the center of each group of bars
    ax.set_xticks(index + bar_width * (n_it / 2))
    ax.set_xticklabels(ci_dict.keys(), rotation=45)

    # Adding a legend and space for labels
    ax.legend(title="Labels", loc="best", bbox_to_anchor=(1.05, 1))

    # Tight layout to avoid overlapping
    plt.tight_layout()
    plt.show()
