"""
Plot functions module.

This module provides functions to plot results.

Functions:
- plot_prediction_distribution: Plots the prediction probabilities.
- plot_reliability_diagrams: Plots the reliability diagrams.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes._axes import Axes
from matplotlib.figure import Figure
from ml_insights import SplineCalib
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.calibration import calibration_curve

from medpipe._types import Labels
from medpipe.utils.exceptions import file_checks

if TYPE_CHECKING:
    import numpy.typing as npt

COLOURS = ["#2D90D8", "#33367A", "#96690E", "#CDB4DB", "#F2CC8F"]


def plot_probability_distribution(
    probas: npt.NDArray,
    label: str = "",
    n_bins: int = 10,
    save_path: str = "",
    extension: str = ".png",
    show_fig: bool = True,
    **kwargs: Any,
):
    """
    Plots the predicted probability distribution as a histogram.

    By default, the yscale is log, but can be overwritten by
    specifying it as a keyword argument.

    Parameters
    ----------
    probas : npt.NDArray
        Predicted probabilities of shape (n_samples, 2) or
        the positive class of shape (n_samples,).
    label : str
         Label for the legend.
    n_bins : int, default: 10
        Number of bins for the histogram.
    save_path : str, default: []
        Path to the save file.
    extension : str, default: ".png"
        Extension to save figure in.
    show_fig : bool, default: True
        Flag to show the figure.
    **kwargs : Any
        Extra arguments for the figure or axes objects.

    Returns
    -------
    None
        Nothing is returned.

    """
    # Split arguments based on where they should be sent
    ax_kwargs = {key: value for key, value in kwargs.items() if key in dir(Axes)}
    fig_kwargs = {key: value for key, value in kwargs.items() if key in dir(Figure)}

    # Set up variables
    bins = np.linspace(0, 1, n_bins + 1)

    # Set figure and axes properties
    fig, ax = plt.subplots(**fig_kwargs)  # Create a new figure

    # Set labels and scale
    ax.set_xlabel("Predicted probabilities", fontweight="bold")
    ax.set_ylabel("Count", fontweight="bold")
    ax.set_yscale("log")

    if probas.ndim == 2:
        probas = probas[:, 1]

    ax.hist(
        probas,
        color=COLOURS[0],
        stacked=True,
        edgecolor="black",
        bins=bins,
        label=label,
    )
    # Remove spines for aesthetics
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    # Add legend and title
    ax.legend(loc="upper right", bbox_to_anchor=(1.45, 0.9), frameon=False)
    title = kwargs["set_title"] if "set_title" in kwargs.keys() else ""
    ax.set_title(title, fontweight="bold")

    ax.set_xlim([-0.05, 1.05])  # Set x limits

    # Adjust layout
    plt.tight_layout()
    fig.subplots_adjust(right=0.7, bottom=0.14)

    # Set ax_kwargs to override if needed
    if "set_title" in kwargs.keys():
        ax_kwargs.pop("set_title")
    for key, val in ax_kwargs.items():
        getattr(ax, key)(val)

    if save_path:
        save_file = save_path + extension
        file_checks(save_file, extension=extension, exists=False)
        plt.savefig(save_file)
    if show_fig:
        plt.show()

    plt.close()


def plot_reliability_diagrams(
    y_test: Labels,
    probas: npt.NDArray,
    label: str = "",
    strategy: Literal["quantile", "spline"] = "spline",
    n_bootstraps: int = 200,
    save_path: str = "",
    extension: str = ".png",
    show_fig: bool = True,
    calibration_kwargs: dict[str, Any] = {},
    **kwargs: Any,
):
    """
    Plots the reliability diagrams for the given probabilities.

    The 95% confidence interval is calculated using the bootstrap method for
    the calibration curve.
    The probability distribution can also be plotted under the calibration
    curve.

    Parameters
    ----------
    y_test : Labels
        Ground truth labels of shape (n_samples,).
    probas : npt.NDArray
        Predicted probabilities of shape (n_samples, 2) or
        positive class probabilities of shape (n_samples,).
    label : str, default: ""
        List of labels for the legend.
    strategy : Literal["quantile", "spline"]
        Strategy to use for the calibration curve.
    n_bootstraps : int, default: 200
        Number of iteration for the bootstrap.
    save_path : str, default: []
        Path to the save file.
    extension : str, default: ".png"
        Extension to save figure in.
    show_fig : bool, default: True
        Flag to show the figure.
    calibration_kwargs : dict[str, Any], default: {}
        Extra arguments for the calibration function.
    **kwargs : Any
        Extra arguments for the figure or axes objects.

    Returns
    -------
    None
        Nothing is returned.

    """
    # Split arguments based on where they should be sent
    ax_kwargs = {key: value for key, value in kwargs.items() if key in dir(Axes)}
    fig_kwargs = {key: value for key, value in kwargs.items() if key in dir(Figure)}

    # Set figure properties
    fig, ax = plt.subplots(**fig_kwargs)
    grid_resolution = 100  # Number of points to plot with
    grid = np.linspace(0, 1, grid_resolution)

    # Remove spines for aesthetics
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    # Plot perfect calibration line
    ax.plot(
        np.linspace(0, 1, grid_resolution),
        np.linspace(0, 1, grid_resolution),
        "k--",
        label="Perfectly calibrated",
    )

    if probas.ndim == 2:
        probas = probas[:, 1]  # Get only positive class

    prob_true, prob_pred = _get_calibration_data(
        y_test, probas, strategy, grid=grid, **calibration_kwargs
    )

    boots = []
    rng = np.random.default_rng()  # Set a default random number generator

    for _ in range(n_bootstraps):
        idx = rng.choice(len(y_test), len(y_test), replace=True)
        prob_true_boot, prob_pred_boot = _get_calibration_data(
            y_test[idx], probas[idx], strategy, grid=grid, **calibration_kwargs
        )
        boots.append(np.interp(prob_pred, prob_pred_boot, prob_true_boot))

    lower = np.percentile(boots, 2.5, axis=0)
    upper = np.percentile(boots, 97.5, axis=0)

    _plot_calibration(
        ax, prob_pred, prob_true, lower, upper, strategy, COLOURS[0], label
    )

    if max(prob_pred) < 0.4:
        # Add inset if calibration curve does not cover enough of the graph
        # Remove spines for aesthetics
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)

        # Create inset
        ax_ins = ax.inset_axes(
            [0.5, 0.1, 0.5, 0.5],
            yticklabels=[],
        )
        # Connect the inset to the zoomed area in the main plot
        ax.indicate_inset_zoom(ax_ins, edgecolor="black")

        ax_ins.plot(  # Reference line
            np.linspace(0, max(prob_pred), 100),
            np.linspace(0, max(prob_pred), 100),
            "k--",
        )

        # Plot inset
        _plot_calibration(
            ax_ins, prob_pred, prob_true, lower, upper, strategy, COLOURS[0]
        )

    # Remove spines for aesthetics
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    # Create new plot for distribution
    divider = make_axes_locatable(ax)
    ax_dist = divider.append_axes("bottom", 0.5, pad=0.1, sharex=ax)

    bins = np.linspace(0, 1, 21)

    ax_dist.hist(
        probas,
        color=COLOURS[0],
        edgecolor="black",
        bins=bins,
        label=label,
    )
    ax_dist.set_yscale("log")
    ax_dist.set_xlabel("Predicted probabilities", fontweight="bold")

    # Set title and labels
    title = kwargs["set_title"] if "set_title" in kwargs.keys() else ""
    ax.set_title(title, fontweight="bold")
    if "set_title" in kwargs.keys():
        ax_kwargs.pop("set_title")
    ax.set_ylabel("Observed proportion", fontweight="bold")

    # Set ax_kwargs to override if needed
    for key, val in ax_kwargs.items():
        getattr(ax, key)(val)

    ax.legend(loc="upper right", bbox_to_anchor=(1.6, 0.9), frameon=False)
    plt.tight_layout()

    # Remove spines for aesthetics for distribution
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    fig.subplots_adjust(right=0.66, bottom=0.14)

    if save_path:
        save_file = save_path + extension
        file_checks(save_file, extension=extension, exists=False)
        plt.savefig(save_file)
    if show_fig:
        plt.show()

    plt.close()


def _get_calibration_data(
    y_test: Labels,
    probas: npt.NDArray,
    strategy: Literal["quantile", "spline"],
    grid: npt.NDArray | None = None,
    **calibration_kwargs,
) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Gets the calibration data based on the strategy.

    Parameters
    ----------
    y_test : Labels
        Ground truth labels of shape (n_samples,).
    probas : npt.NDArray
        Predicted probabilities of the positive class
        of shape (n_samples,).
    strategy : Literal["quantile", "spline"]
        Strategy to use for the calibration curve.
    grid : npt.NDArray | None, default: None
        Grid used for predictions with spline strategy.
    calibration_kwargs
        Keyword arguments for the calibration function.

    Returns
    -------
    prob_true : npt.NDArray
        Expected calibration points.
    prob_pred : npt.NDArray
        Actual calibration points.

    Raises
    ------
    ValueError
        If grid is None with spline strategy.
        If strategy is not quantile or spline.

    """
    if strategy == "quantile":
        prob_true, prob_pred = calibration_curve(
            y_test,
            probas,
            **calibration_kwargs,
        )

    elif strategy == "spline":
        if grid is None:
            raise ValueError("Input grid should not be None with spline strategy")
        spline = SplineCalib()
        spline.fit(probas, y_test)

        prob_true = grid
        prob_pred = spline.predict(grid)
        assert prob_pred is not None  # Avoids type error

    else:
        raise ValueError(f"Unknown strategy {strategy}, should be quantile or spline")

    return prob_true, prob_pred


def _plot_calibration(
    ax: Axes,
    prob_pred: npt.NDArray,
    prob_true: npt.NDArray,
    lower: npt.NDArray,
    upper: npt.NDArray,
    strategy: Literal["quantile", "spline"],
    colour: str,
    label: str | None = None,
):
    """
    Helper function to plot the calibration and 95% CI on a Axes object.

    Parameters
    ----------
    ax : plt.Axes
        Axes on which to plot the data.
    prob_pred : npt.NDArray
        Predicted probabilities.
    prob_true : npt.NDArray
        True probabilities.
    lower : npt.NDArray
        Lower bounds of the confidence interval.
    upper : npt.NDArray
        Upper bounds of the confidence interval.
    strategy : Literal["quantile", "spline"]
        Strategy to use to select the marker for plotting.
    colour : str
        Colour to plot in.
    label : str or None, default: None
        Label for the plotted curves.

    Returns
    -------
    None
        Nothing is returned.

    """
    if strategy == "quantile":
        marker = "."
    elif strategy == "spline":
        marker = "-"

    ax.plot(
        prob_pred,
        prob_true,
        marker=marker,
        color=colour,
        label=label,
    )
    ax.fill_between(
        prob_pred,
        lower,
        upper,
        color=colour,
        alpha=0.5,
        label=f"{label} 95% CI",
    )
