"""
Plot functions module.

This module provides functions to plot results.

Functions:
- plot_from_display: Plot results from a Display class.
- plot_mean_ROC_curve: Plots the ROC curve of each fold and the mean ROC curve.
- plot_mean_PR_curve: Plots the precision-recall curve of each fold and the mean PRC.
- plot_metrics_CI: Plots the metrics with confidence intrevals for each fold.
- plot_prediction_distribution: Plots the prediction probabilities.
- plot_reliability_diagrams: Plots the reliability diagrams.
"""

import matplotlib.pyplot as plt
import numpy as np
import sklearn as skl
from matplotlib.axes._axes import Axes
from matplotlib.figure import Figure
from sklearn.calibration import calibration_curve

from pyrisk.utils.exceptions import array_check, array_dim_check, file_checks


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
    display : str {"roc", "confusion", "precision-recall", "calibration"}
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
    **kwargs,
):
    """
    Plots the ROC curve of each fold and the mean ROC curve.

    Parameters
    ----------
    model_metrics : dict[int, dict[str, float or tuple(array-like)]]
        Model metrics for different folds.
    label_list : list[str]
        List of predicted labels.
    save_path : str, default: []
        Path to the save file.
    extension : str, default: ".png"
        Extension to save figure in.
    show_fig : bool, default: True
        Flag to show the figure.
    nb_points : int, default: 500
        Number of points for the mean ROC curve.
    **kwargs
        Extra arguments for the figure or axes objects.

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
        n_it = 1

    mean_fpr = np.linspace(0, 1, nb_points)
    global_tprs = []
    global_aucs = []

    # Split arguments based on where they should be sent
    ax_kwargs = {key: value for key, value in kwargs.items() if key in dir(Axes)}
    fig_kwargs = {key: value for key, value in kwargs.items() if key in dir(Figure)}

    for i in range(n_it):
        # Set up figure and colours
        _, ax = plt.subplots(**fig_kwargs)
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

        # Set ax_kwargs to override if needed
        for key, val in ax_kwargs.items():
            getattr(ax, key)(val)

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
    **kwargs,
):
    """
    Plots the PR curve of each fold and the mean PR curve.

    Parameters
    ----------
    model_metrics : dict[int, dict[str, float or tuple(array-like)]]
        Model metrics for different folds.
    label_list : list[str]
        List of predicted labels.
    save_path : str, default: []
        Path to the save file.
    extension : str, default: ".png"
        Extension to save figure in.
    show_fig : bool, default: True
        Flag to show the figure.
    nb_points : int, default: 500
        Number of points for the mean curve.
    **kwargs
        Extra arguments for the figure or axes objects.

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
        n_it = 1  # Add one for the global values if multilabel

    mean_precision = np.linspace(0, 1, nb_points)
    global_recalls = []
    global_aps = []

    # Split arguments based on where they should be sent
    ax_kwargs = {key: value for key, value in kwargs.items() if key in dir(Axes)}
    fig_kwargs = {key: value for key, value in kwargs.items() if key in dir(Figure)}

    for i in range(n_it):
        # Set up figure and colours
        _, ax = plt.subplots(**fig_kwargs)
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

        # Set ax_kwargs to override if needed
        for key, val in ax_kwargs.items():
            getattr(ax, key)(val)

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
    ci_dict : dict[str, list[tuple(float, float, float)]]
        Dictionary containing the metric value and confidence intervals.
        The keys are the name of the metrics and the values are a list of tuple
        with first element the metric value, second the lower bound, and third
        the upper bound. One list elements per model. One list elements per model
    label_list : list[str]
        List of labels for the legend.
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
    # Split arguments based on where they should be sent
    ax_kwargs = {key: value for key, value in kwargs.items() if key in dir(Axes)}
    fig_kwargs = {key: value for key, value in kwargs.items() if key in dir(Figure)}

    # Set up some variables
    colours = [
        "#2D90D8",
        "#33367A",
        "#96690E",
        "#CDB4DB",
        "#F2CC8F",
    ]
    y_labels = {
        "auroc": "AUROC",
        "ap": "AUPRC",
        "log_loss": "Log loss",
        "accuracy": "Accuracy",
        "recall": "Recall",
        "precision": "Precision",
        "f1": "F1",
    }
    bar_width = 0.3
    x = np.arange(len(label_list)) * bar_width

    # Loop through each metric
    for key, values in ci_dict.items():
        # Set up the figure and axis
        fig, ax = plt.subplots(**fig_kwargs)  # One figure per metric

        for j in range(len(values[0])):
            value = values[0][j]
            lower_b = values[1][j]

            ax.bar(
                x[j],
                value,
                width=bar_width,
                color=colours[j],
                edgecolor=(0, 0, 0, 1),
                label=label_list[j],
            )

            ax.errorbar(
                x[j],
                value,
                yerr=value - lower_b,
                fmt="none",
                color="black",
                capsize=5,
            )

        # Customize the chart
        ax.set_ylabel(y_labels[key], fontweight="bold")
        if key != "log_loss":
            ax.set_ylim([0, 1.05])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Remove x ticks
        ax.set_xticks([])
        ax.set_xticklabels([])

        # Place legend for the figure
        fig.legend(loc="center right", title="Models")

        # Set ax_kwargs to override if needed
        for key, val in ax_kwargs.items():
            getattr(ax, key)(val)

        plt.tight_layout()
        fig.subplots_adjust(right=0.7, bottom=0.14)
        if save_path:
            save_file = save_path + key + extension
            file_checks(save_file, extension=extension, exists=False)
            plt.savefig(save_file)
        if show_fig:
            plt.show()


def plot_prediction_distribution(
    dist_list,
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
    dist_list : list[array]
        List of the predicted probability distributions.
    label_list : list[str]
        List of labels for the legend.
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

    """
    title = kwargs["set_title"] if "set_title" in kwargs.keys() else ""

    # Split arguments based on where they should be sent
    ax_kwargs = {key: value for key, value in kwargs.items() if key in dir(Axes)}
    fig_kwargs = {key: value for key, value in kwargs.items() if key in dir(Figure)}

    # Set up variables
    colour_list = ["#2D90D8", "#33367A", "#96690E", "#CDB4DB", "#F2CC8F"]
    bins = np.linspace(0, 1, n_bins + 1)

    # Set figure and axes properties
    fig, ax = plt.subplots(**fig_kwargs)  # Create a new figure

    # Set labels and scale
    ax.set_xlabel("Predicted probabilities", fontweight="bold")
    ax.set_ylabel("Count", fontweight="bold")
    ax.set_yscale("log")

    # Set ax_kwargs to override if needed
    for key, val in ax_kwargs.items():
        getattr(ax, key)(val)

    ax.hist(
        dist_list,
        color=colour_list[: len(dist_list)],
        stacked=True,
        edgecolor="black",
        bins=bins,
        label=label_list,
    )
    # Remove spines for aesthetics
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    # Add legend
    ax.legend(loc="upper right", bbox_to_anchor=(1.45, 0.9), title="Models")
    ax.title(title)
    ax.set_xlim([-0.05, 1.05])  # Set x limits

    # Adjust layout
    plt.tight_layout()
    fig.subplots_adjust(right=0.7, bottom=0.14)

    if save_path:
        save_file = save_path + extension
        file_checks(save_file, extension=extension, exists=False)
        plt.savefig(save_file)
    if show_fig:
        plt.show()


def plot_reliability_diagrams(
    y_test,
    y_pred_proba=[],
    y_pred_proba_calib=[],
    label_list=[],
    save_path="",
    extension=".png",
    show_fig=True,
    display_kwargs={},
    **kwargs,
):
    """
    Plots the reliability diagrams using CalibrationDisplay from
    sklearn.calibration.

    Parameters
    ----------
    y_test : array-like of shape (n_samples, n_classes)
        Ground truth labels.
    y_pred_proba : array-like of shape (n_samples, n_classes), default: []
        Predicted probabilities from the predictor.
    y_pred_proba_calib : array-like of shape (n_samples, n_classes), default: []
        Predicted probabilities from the calibrator.
    label_list : list[str], default: []
    save_path : str, default: []
        Path to the save file.
    extension : str, default: ".png"
        Extension to save figure in.
    show_fig : bool, default: True
        Flag to show the figure.
    display_kwargs : dict[str, value], default: {}
        Extra arguments for the CalibrationDisplay.
    **kwargs
        Extra arguments for the figure or axes objects.

    Returns
    -------
    None
        Nothing is returned.

    """
    n_classes = 1  # Default number of classes

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
        max_val = 0

        # Plot perfect calibration
        ax.plot(
            np.linspace(0, 1, 100),
            np.linspace(0, 1, 100),
            "k--",
            label="Perfectly calibrated",
        )
        if len(y_pred_proba) > 0:
            prob_true, prob_pred = calibration_curve(
                y_test[:, i],
                y_pred_proba[:, i],
                **display_kwargs,
            )
            max_val = _get_max_calibration_value(
                max_val, prob_pred.max(), prob_true.max()
            )
            ax.plot(
                prob_pred,
                prob_true,
                marker="s",
                color=colours(colour_val),
                label="Uncalibrated",
            )

            colour_val += 1

        if len(y_pred_proba_calib) > 0:
            prob_true, prob_pred = calibration_curve(
                y_test[:, i],
                y_pred_proba_calib[:, i],
                **display_kwargs,
            )
            max_val = _get_max_calibration_value(
                max_val, prob_pred.max(), prob_true.max()
            )
            ax.plot(
                prob_pred,
                prob_true,
                marker="s",
                color=colours(colour_val),
                label="Calibrated",
            )

        colour_val += 1

        ax.set_title(f"Reliability diagram for {label_list[i]}")
        ax.set_xlabel("Predicted probabilities")
        ax.set_ylabel("Observed proportion")
        ax.set_xlim((-0.01, max_val + 0.05))
        ax.set_ylim((-0.01, max_val + 0.05))

        # Set ax_kwargs to override if needed
        for key, val in ax_kwargs.items():
            getattr(ax, key)(val)

        ax.legend(loc="upper right", bbox_to_anchor=(1.6, 0.9))
        plt.tight_layout()
        fig.subplots_adjust(right=0.66, bottom=0.14)

        if save_path:
            save_file = save_path + f"_{label_list[i]}" + extension
            file_checks(save_file, extension=extension, exists=False)
            plt.savefig(save_file)
        if show_fig:
            plt.show()


def _get_max_calibration_value(cur_max, prob_pred_max, prob_true_max):
    """
    Returns the maximum between inputs.

    Parameters
    ----------
    cur_max : float
        Current maximum value.
    prob_pred_max : float
        Predicted probabilities maximum.
    prob_true_max : float
        True probabilities maximum.

    Returns
    -------
    cur_max : float
        New current maximum.

    """
    if prob_pred_max > cur_max:
        cur_max = prob_pred_max
    if prob_true_max > cur_max:
        cur_max = prob_true_max
    return cur_max
