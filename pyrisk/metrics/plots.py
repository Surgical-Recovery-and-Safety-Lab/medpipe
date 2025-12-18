"""
Plot functions module.

This module provides functions to plot results.

Functions:
- plot_from_display: Plot results from a Display class.
- plot_mean_ROC_curve: Plots the ROC curve of each fold and the mean ROC curve.
- plot_mean_PR_curve: Plots the precision-recall curve of each fold and the mean PRC.
- plot_metrics_CI: Plots the metrics with confidence intrevals for each fold.
"""

import matplotlib.pyplot as plt
import numpy as np
import sklearn as skl
from matplotlib.axes._axes import Axes
from matplotlib.figure import Figure

from pyrisk.utils.exceptions import array_check


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
        case "calibration":
            display_fct = skl.calibration.CalibrationDisplay.from_predictions
        case _:
            raise ValueError(f"Invalid display, got {display}")

    display_fct(y_true, y_pred, **kwargs)
    plt.show()


def plot_mean_ROC_curve(
    model_metrics,
    label_list,
    save_path="",
    extension=".png",
    show_fig=True,
    nb_points=500,
):
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
    save_path : str, default: []
        Path to the save file.
    extension : str, default: ".png"
        Extension to save figure in.
    show_fig : bool, default: True
        Flag to show the figure.

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
        # Set up figure and colours
        _, ax = plt.subplots(dpi=300)
        colours = plt.get_cmap("tab20b")

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

            ax.plot(
                mean_fpr,
                tprs[k],
                lw=1,
                label=f"Fold number {fold} (AUC {aucs[k]:.3f})",
                color=colours(k),
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
        ax.plot(
            mean_fpr,
            mean_tpr,
            color="k",
            lw=2,
            label=rf"Mean {metric} ROC curve (AUC {mean_auroc:.3f} $\pm$ {std_auroc:.3f})",
        )
        ax.fill_between(
            mean_fpr,
            lower_std,
            upper_std,
            color="grey",
            alpha=0.5,
            label=r"Mean $\pm$ 1 SD",
        )

        ax.set_xlim(xmin=0, xmax=1)
        ax.set_ylim(ymin=0, ymax=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"Cross-Validated {metric} ROC")
        ax.legend(ncol=2, fontsize=5, loc="lower right")

        # Tight layout to avoid overlapping
        plt.tight_layout()
        if save_path:
            save_file = save_path + f"_{label_list[i]}" + extension
            file_checks(save_file, extension=extension, exists=False)
            plt.savefig(save_file)
        if show_fig:
            plt.show()


def plot_mean_PR_curve(
    model_metrics,
    label_list,
    save_path="",
    extension=".png",
    show_fig=True,
    nb_points=500,
):
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
    save_path : str, default: []
        Path to the save file.
    extension : str, default: ".png"
        Extension to save figure in.
    show_fig : bool, default: True
        Flag to show the figure.

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
        # Set up figure and colours
        _, ax = plt.subplots(dpi=300)
        colours = plt.get_cmap("tab20b")

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

            ax.plot(
                recalls[k],
                mean_precision,
                lw=1,
                label=f"Fold number {fold} (AP {aps[k]:.3f})",
                color=colours(k),
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
        ax.plot(
            mean_recall,
            mean_precision,
            color="k",
            lw=2,
            label=rf"Mean {metric} PRC curve (AP {mean_ap:.3f} $\pm$ {std_ap:.3f})",
            zorder=3,
        )
        ax.fill_betweenx(
            mean_precision,
            lower_std,
            upper_std,
            color="grey",
            alpha=0.5,
            label=r"Mean $\pm$ 1 SD",
            zorder=2,
        )

        ax.set_xlim(xmin=0, xmax=1)
        ax.set_ylim(ymin=0, ymax=1)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"Cross-Validated {metric} PRC")
        ax.legend(ncol=2, fontsize=5, loc="best", bbox_to_anchor=(1, 1))

        # Tight layout to avoid overlapping
        plt.tight_layout()

        if save_path:
            save_file = save_path + f"_{label_list[i]}" + extension
            file_checks(save_file, extension=extension, exists=False)
            plt.savefig(save_file)
        if show_fig:
            plt.show()


def plot_metrics_CI(
    ci_dict, label_list, save_path="", extension=".png", show_fig=True, **kwargs
):
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
    save_path : str, default: []
        Path to the save file.
    extension : str, default: ".png"
        Extension to save figure in.
    show_fig : bool, default: True
        Flag to show the figure.
    **kwargs
        Extra arguments for the figure or axes objects.

    Returns
    -------
    None
        Nothing is returned.

    """
    n_it = len(label_list)  # Number of iterations
    if n_it > 1:
        n_it += 1  # Add one for the global values if multilabel

    # Split arguments based on where they should be sent
    ax_kwargs = {key: value for key, value in kwargs.items() if key in dir(Axes)}
    fig_kwargs = {key: value for key, value in kwargs.items() if key in dir(Figure)}

    # Set up the figure and axis
    fig, ax = plt.subplots(**fig_kwargs)
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

    for key, val in ax_kwargs.items():
        getattr(ax, key)(val)

    ax.legend(title="Labels", loc="upper right", bbox_to_anchor=(1.5, 0.9))
    plt.tight_layout()
    fig.subplots_adjust(right=0.68)

    if save_path:
        save_file = save_path + extension
        file_checks(save_file, extension=extension, exists=False)
        plt.savefig(save_file)
    if show_fig:
        plt.show()


def plot_prediction_distribution(
    y_pred_proba=[],
    y_pred_proba_calib=[],
    label_list=[],
    n_bins=10,
    save_path="",
    extension=".png",
    show_fig=True,
    **kwargs,
):
    """
    Plots the prediction probabilities.

    Parameters
    ----------
    y_pred_proba : array-like of shape (n_samples, n_classes), default: []
        Predicted probabilities from the predictor.
    y_pred_proba_calib : array-like of shape (n_samples, n_classes), default: []
        Predicted probabilities from the calibrator.
    label_list : list[str], default: []
        List of predicted labels.
    n_bins : int, default: 10
        Number of bins for the histogram.
    save_path : str, default: []
        Path to the save file.
    extension : str, default: ".png"
        Extension to save figure in.
    show_fig : bool, default: True
        Flag to show the figure.
    **kwargs
        Extra arguments for the figure or axes objects.

    Returns
    -------
    None
        Nothing is returned.

    Raises
    ------
    ValueError
        If y_pred_proba and y_pred_proba_calib are both empty.

    """
    array_check(y_pred_proba)
    array_check(y_pred_proba_calib)

    n_classes = 1  # Default number of classes
    title = kwargs["set_title"] if "set_title" in kwargs.keys() else ""

    if len(y_pred_proba) > 0 and len(y_pred_proba_calib) > 0:
        if len(y_pred_proba) > 1:
            # If multiple labels, check that n_classes agree
            array_dim_check(y_pred_proba, y_pred_proba_calib, dim=1)
            n_classes = y_pred_proba.shape[1]
        else:
            y_pred_proba = y_pred_proba.reshape(-1, 1)
            y_pred_proba_calib = y_pred_proba_calib(-1, 1)

    elif len(y_pred_proba) > 0:
        if len(y_pred_proba) > 1:
            n_classes = y_pred_proba.shape[1]
        else:
            y_pred_proba = y_pred_proba.reshape(-1, 1)

    elif len(y_pred_proba_calib) > 0:
        if len(y_pred_proba_calib) > 1:
            n_classes = y_pred_proba_calib.shape[1]
        else:
            y_pred_proba_calib = y_pred_proba_calib(-1, 1)

    else:
        raise ValueError("At least one set of predicted probabilities is needed")

    colours = plt.get_cmap("Pastel1")

    # Split arguments based on where they should be sent
    ax_kwargs = {key: value for key, value in kwargs.items() if key in dir(Axes)}
    fig_kwargs = {key: value for key, value in kwargs.items() if key in dir(Figure)}

    for i in range(n_classes):
        # Set figure and axes properties
        fig, ax = plt.subplots(**fig_kwargs)  # Create a new figure
        colour_val = 0

        # Default settings that are overwritten if kwargs are passed
        ax.set_xlabel("Predicted probabilities")
        ax.set_ylabel("Count")
        ax.set_yscale("log")
        ax.set_title(f"Predicted probabilities distribution for {label_list[i]}")

        if "set_title" in kwargs.keys() and len(label_list) > 0:
            kwargs["set_title"] = title + label_list[i]

        if len(y_pred_proba) > 0 and len(y_pred_proba_calib) > 0:
            ax.hist(
                y_pred_proba[:, i],
                color=colours(colour_val),
                alpha=0.5,
                label="Uncalibrated",
                bins=n_bins,
            )

            colour_val += 1

            ax.hist(
                y_pred_proba_calib[:, i],
                color=colours(colour_val),
                alpha=0.5,
                label="Calibrated",
                bins=n_bins,
            )

        elif len(y_pred_proba) > 0:
            ax.hist(
                y_pred_proba[:, i],
                color=colours(colour_val),
                label="Uncalibrated",
                bins=n_bins,
            )

        elif len(y_pred_proba_calib) > 0:
            ax.hist(
                y_pred_proba_calib[:, i],
                color=colours(colour_val),
                label="Calibrated",
                bins=n_bins,
            )

        colour_val += 1

        for key, val in ax_kwargs.items():
            getattr(ax, key)(val)

        ax.legend(loc="upper right", bbox_to_anchor=(1.4, 0.9), borderaxespad=0.0)
        plt.tight_layout()
        fig.subplots_adjust(right=0.7, bottom=0.14)

        if save_path:
            save_file = save_path + f"_{label_list[i]}" + extension
            file_checks(save_file, extension=extension, exists=False)
            plt.savefig(save_file)
        if show_fig:
            plt.show()
