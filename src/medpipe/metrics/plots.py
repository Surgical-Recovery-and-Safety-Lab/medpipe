"""
Plot functions module.

This module provides functions to plot results.

Functions:
- plot_metrics_CI: Plots the metrics with confidence intrevals for each fold.
- plot_prediction_distribution: Plots the prediction probabilities.
- plot_reliability_diagrams: Plots the reliability diagrams.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes._axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.calibration import calibration_curve

from medpipe.utils.exceptions import file_checks


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

        plt.close()


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
    ax.set_title(title)
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

    plt.close()


def plot_reliability_diagrams(
    y_test,
    proba_list,
    label_list=[],
    distribution=True,
    save_path="",
    extension=".png",
    show_fig=True,
    display_kwargs={},
    **kwargs,
):
    """
    Plots the reliability diagrams for the given probabilities.

    Parameters
    ----------
    y_test : array-like of shape (n_samples, n_classes)
        Ground truth labels.
    proba_list : list[array]
        List of predicted probabilities.
    label_list : list[str], default: []
        List of labels for the legend.
    distribution : bool, default: False
        Flag to plot the probability distribution as well.
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
    colours = ["#2D90D8", "#33367A", "#96690E", "#CDB4DB", "#F2CC8F"]

    # Split arguments based on where they should be sent
    ax_kwargs = {key: value for key, value in kwargs.items() if key in dir(Axes)}
    fig_kwargs = {key: value for key, value in kwargs.items() if key in dir(Figure)}

    # Set figure properties
    fig, ax = plt.subplots(**fig_kwargs)

    # Plot perfect calibration
    ax.plot(
        np.linspace(0, 1, 100),
        np.linspace(0, 1, 100),
        "k--",
        label="Perfectly calibrated",
    )

    for i in range(len(proba_list)):
        prob_true, prob_pred = calibration_curve(
            y_test,
            proba_list[i],
            **display_kwargs,
        )
        ax.plot(
            prob_pred,
            prob_true,
            marker=".",
            color=colours[i],
            label=label_list[i],
        )

    if distribution:
        # Create new plot for distribution
        divider = make_axes_locatable(ax)
        ax_dist = divider.append_axes("bottom", 0.5, pad=0.1, sharex=ax)

        if "n_bins" in display_kwargs.keys():
            bins = np.linspace(0, 1, display_kwargs["n_bins"] + 1)
        else:
            bins = np.linspace(0, 1, 21)

        ax_dist.hist(
            proba_list,
            stacked=True,
            color=colours[: len(proba_list)],
            edgecolor="black",
            bins=bins,
            label=label_list,
        )
        ax_dist.set_yscale("log")

    title = kwargs["set_title"] if "set_title" in kwargs.keys() else ""
    ax.set_title(title)
    ax.set_xlabel("Predicted probabilities", fontweight="bold")
    ax.set_ylabel("Observed proportion", fontweight="bold")

    # Set ax_kwargs to override if needed
    for key, val in ax_kwargs.items():
        getattr(ax, key)(val)

    ax.legend(loc="upper right", bbox_to_anchor=(1.6, 0.9))
    plt.tight_layout()
    fig.subplots_adjust(right=0.66, bottom=0.14)

    if save_path:
        save_file = save_path + extension
        file_checks(save_file, extension=extension, exists=False)
        plt.savefig(save_file)
    if show_fig:
        plt.show()

    plt.close()
