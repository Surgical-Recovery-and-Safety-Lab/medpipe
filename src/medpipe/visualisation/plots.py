"""
Stateless drawing primitives for Medpipe visualizations.
"""

from typing import Any, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure
from mpl_toolkits.axes_grid1 import make_axes_locatable

from medpipe.visualisation.themes import MedpipeTheme

# Medpipe Default Palette Defaults
_DEFAULT_THEME = MedpipeTheme()


def draw_probability_distribution(
    probas: np.ndarray,
    n_bins: int = 10,
    label: str = "Predicted probabilities",
    ax: Optional[Axes] = None,
    yscale: str = "linear",
    color: str = _DEFAULT_THEME.primary_color,
    edgecolor: str = "black",
    title: Optional[str] = None,
    show_spines: bool = _DEFAULT_THEME.show_spines,
    **hist_kwargs: Any,
) -> Tuple[Figure | SubFigure, Axes]:
    """Render a predicted probability distribution histogram.

    Parameters
    ----------
    probas : np.ndarray
        Predicted probabilities of shape (n_samples, 2) or (n_samples,).
    n_bins : int, default=10
        Number of equal-width bins across the [0, 1] probability range.
    label : str, default="Predicted Probabilities"
        Legend label for the histogram series.
    ax : matplotlib.axes.Axes, optional
        Pre-existing Matplotlib axes instance. If None, a new figure and axes are created.
    yscale : str, default="linear"
        Scale to use for the y-axis (e.g. linear, log, etc.)
    color : str, default=_DEFAULT_THEME.primary_color
        Fill color for histogram bars.
    edgecolor : str, default="black"
        Border color for histogram bars.
    title : str, optional
        Axes title text.
    show_spines : bool, default=_DEFAULT_THEME.show_spines
        Whether to keep the top and right border spines visible.
    **hist_kwargs : Any
        Additional Matplotlib keyword arguments forwarded to `ax.hist`.

    Returns
    -------
    fig : matplotlib.figure.Figure or matplotlib.figure.SubFigure
        Parent Matplotlib figure containing the axes.
    ax : matplotlib.axes.Axes
        Matplotlib axes containing the rendered histogram.

    Raises
    ------
    ValueError
        If ax is provided but not attached to a Figure.

    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.get_figure()
        if fig is None:
            raise ValueError("The provided Axes instance is not attached to a Figure")

    if probas.ndim == 2:
        probas = probas[:, 1]

    bins = np.linspace(0.0, 1.0, n_bins + 1).tolist()

    # Clean hist_kwargs to prevent parameter collisions
    for key in ("color", "edgecolor", "bins", "label"):
        hist_kwargs.pop(key, None)

    ax.hist(
        probas,
        bins=bins,
        color=color,
        edgecolor=edgecolor,
        label=label,
        **hist_kwargs,
    )

    ax.set_xlabel("Predicted probabilities", fontweight="bold")
    ax.set_ylabel("Count", fontweight="bold")
    ax.set_yscale(yscale)
    ax.set_xlim(xmin=-0.05, xmax=1.05)

    if title:
        ax.set_title(title, fontweight="bold")

    if not show_spines:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    ax.legend(loc="upper right", frameon=False)

    return fig, ax


def draw_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    lower_ci: Optional[np.ndarray] = None,
    upper_ci: Optional[np.ndarray] = None,
    label: str = "Model",
    ax: Optional[Axes] = None,
    color: str = _DEFAULT_THEME.primary_color,
    ci_color: Optional[str] = None,
    ci_alpha: float = _DEFAULT_THEME.ci_alpha,
    linestyle: str = "-",
    linewidth: float = _DEFAULT_THEME.linewidth,
    chance_linestyle: str = "--",
    chance_color: str = "black",
    title: Optional[str] = None,
    show_spines: bool = _DEFAULT_THEME.show_spines,
    **line_kwargs: Any,
) -> Tuple[Figure | SubFigure, Axes]:
    """Render a Receiver Operating Characteristic (ROC) curve and optional CI bounds.

    Parameters
    ----------
    fpr : np.ndarray
        False positive rates across decision thresholds.
    tpr : np.ndarray
        True positive rates across decision thresholds.
    lower_ci : np.ndarray, optional
        Lower bound array of TPR values for 95% confidence interval shading.
    upper_ci : np.ndarray, optional
        Upper bound array of TPR values for 95% confidence interval shading.
    label : str, default="Model"
        Legend label for the plotted ROC curve.
    ax : matplotlib.axes.Axes, optional
        Pre-existing Matplotlib axes instance. If None, a new figure and axes are created.
    color : str, default=_DEFAULT_THEME.primary_color
        Color specifier for the main ROC curve and default confidence interval fill.
    ci_color : str, optional
        Custom color specifier for confidence interval shading. Defaults to `color`.
    ci_alpha : float, default=_DEFAULT_THEME.ci_alpha
        Opacity level for confidence interval shaded region [0.0, 1.0].
    linestyle : str, default="-"
        Line style specification for the ROC curve.
    linewidth : float, default=_DEFAULT_THEME.linewidth
        Width in points for the ROC curve line.
    chance_linestyle : str, default="--"
        Line style specification for the diagonal chance reference line.
    chance_color : str, default="black"
        Color specifier for the diagonal chance reference line.
    title : str, optional
        Axes title text.
    show_spines : bool, default=_DEFAULT_THEME.show_spines
        Whether to keep the top and right border spines visible.
    **line_kwargs : Any
        Additional Matplotlib keyword arguments forwarded to `ax.plot` for the ROC curve.

    Returns
    -------
    fig : Figure | SubFigure
        Parent Matplotlib figure containing the axes.
    ax : Axes
        Matplotlib axes containing all rendered ROC elements.

    Raises
    ------
    ValueError
        If ax is not attached to a Figure.

    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.get_figure()
        if fig is None:
            raise ValueError("The provided Axes instance is not attached to a Figure")

    # Draw diagonal reference line (chance level)
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle=chance_linestyle,
        color=chance_color,
        label="Chance level",
    )

    # Clean line_kwargs to prevent parameter collision
    for key in ("color", "linestyle", "linewidth", "label"):
        line_kwargs.pop(key, None)

    # Draw main ROC curve line
    ax.plot(
        fpr,
        tpr,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        label=label,
        **line_kwargs,
    )

    # Draw confidence interval shading if bounds are supplied
    if lower_ci is not None and upper_ci is not None:
        ax.fill_between(
            fpr,
            lower_ci,
            upper_ci,
            color=ci_color or color,
            alpha=ci_alpha,
            label=f"{label} 95% CI",
        )

    # Format axis bounds and labels
    ax.set_xlabel("FPR (1 - Specificity)", fontweight="bold")
    ax.set_ylabel("TPR (Sensitivity)", fontweight="bold")
    ax.set_xlim(xmin=-0.02, xmax=1.02)
    ax.set_ylim(ymin=-0.02, ymax=1.02)

    if title:
        ax.set_title(title, fontweight="bold")

    if not show_spines:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    ax.legend(loc="lower right", frameon=False)

    return fig, ax


def draw_precision_recall_curve(
    precision: np.ndarray,
    recall: np.ndarray,
    lower_ci: Optional[np.ndarray] = None,
    upper_ci: Optional[np.ndarray] = None,
    baseline: Optional[float] = None,
    label: str = "Model",
    ax: Optional[Axes] = None,
    color: str = _DEFAULT_THEME.primary_color,
    ci_color: Optional[str] = None,
    ci_alpha: float = _DEFAULT_THEME.ci_alpha,
    linestyle: str = "-",
    linewidth: float = _DEFAULT_THEME.linewidth,
    baseline_linestyle: str = "--",
    baseline_color: str = "black",
    title: Optional[str] = None,
    show_spines: bool = _DEFAULT_THEME.show_spines,
    **line_kwargs: Any,
) -> Tuple[Figure | SubFigure, Axes]:
    """Render a Precision-Recall (PR) curve with optional CI bounds and baseline.

    Parameters
    ----------
    precision : np.ndarray
        Precision values across thresholds.
    recall : np.ndarray
        Recall values across thresholds.
    lower_ci : np.ndarray, optional
        Lower bound array of precision values for 95% confidence interval shading.
    upper_ci : np.ndarray, optional
        Upper bound array of precision values for 95% confidence interval shading.
    baseline : float, optional
        Horizontal baseline value representing chance performance (positive class ratio).
    label : str, default="Model"
        Legend label for the plotted PR curve.
    ax : matplotlib.axes.Axes, optional
        Pre-existing Matplotlib axes instance. If None, a new figure and axes are created.
    color : str, default=_DEFAULT_THEME.primary_color
        Color specifier for the main PR curve and default confidence interval fill.
    ci_color : str, optional
        Custom color specifier for confidence interval shading. Defaults to `color`.
    ci_alpha : float, default=_DEFAULT_THEME.ci_alpha
        Opacity level for confidence interval shaded region [0.0, 1.0].
    linestyle : str, default="-"
        Line style specification for the PR curve.
    linewidth : float, default=_DEFAULT_THEME.linewidth
        Width in points for the PR curve line.
    baseline_linestyle : str, default="--"
        Line style specification for the baseline reference line.
    baseline_color : str, default="black"
        Color specifier for the baseline reference line.
    title : str, optional
        Axes title text.
    show_spines : bool, default=_DEFAULT_THEME.show_spines
        Whether to keep the top and right border spines visible.
    **line_kwargs : Any
        Additional Matplotlib keyword arguments forwarded to `ax.plot` for the PR curve.

    Returns
    -------
    fig : Figure or SubFigure
        Parent Matplotlib figure containing the axes.
    ax : Axes
        Matplotlib axes containing all rendered PR elements.

    Raises
    ------
    ValueError
        If ax is provided but not attached to a Figure.

    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.get_figure()
        if fig is None:
            raise ValueError("The provided Axes instance is not attached to a Figure")

    # Draw horizontal baseline (prevalence / chance level) if provided
    if baseline is not None:
        ax.axhline(
            y=baseline,
            linestyle=baseline_linestyle,
            color=baseline_color,
            label=f"Baseline ({baseline:.2f})",
        )

    # Clean line_kwargs to prevent parameter collisions
    for key in ("color", "linestyle", "linewidth", "label"):
        line_kwargs.pop(key, None)

    # Draw main PR curve line (Recall on X-axis, Precision on Y-axis)
    ax.plot(
        recall,
        precision,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        label=label,
        **line_kwargs,
    )

    # Draw confidence interval shading if bounds are supplied
    if lower_ci is not None and upper_ci is not None:
        ax.fill_between(
            recall,
            lower_ci,
            upper_ci,
            color=ci_color or color,
            alpha=ci_alpha,
            label=f"{label} 95% CI",
        )

    # Format axis bounds and labels
    ax.set_xlabel("Recall (Sensitivity)", fontweight="bold")
    ax.set_ylabel("Precision (PPV)", fontweight="bold")
    ax.set_xlim(xmin=-0.02, xmax=1.02)
    ax.set_ylim(ymin=-0.02, ymax=1.02)

    if title:
        ax.set_title(title, fontweight="bold")

    if not show_spines:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    ax.legend(loc="lower left", frameon=False)

    return fig, ax


def draw_reliability_diagram(
    prob_true: np.ndarray,
    prob_pred: np.ndarray,
    probas: Optional[np.ndarray] = None,
    lower_ci: Optional[np.ndarray] = None,
    upper_ci: Optional[np.ndarray] = None,
    label: str = "Model",
    dist_n_bins: int = 20,
    dist_yscale: str = "linear",
    auto_inset: bool = True,
    ax: Optional[Axes] = None,
    color: str = _DEFAULT_THEME.primary_color,
    ci_color: Optional[str] = None,
    ci_alpha: float = _DEFAULT_THEME.ci_alpha,
    linestyle: str = "-",
    linewidth: float = _DEFAULT_THEME.linewidth,
    marker: Optional[str] = "o",
    ref_linestyle: str = "--",
    ref_color: str = "black",
    title: Optional[str] = None,
    show_spines: bool = _DEFAULT_THEME.show_spines,
    **line_kwargs: Any,
) -> Tuple[Figure | SubFigure, Axes]:
    """Render a reliability diagram with optional prediction distribution.

    Parameters
    ----------
    prob_true : np.ndarray
        Fraction of positives or spline-calibrated probabilities.
    prob_pred : np.ndarray
        Mean predicted probabilities or evaluation grid points.
    probas : np.ndarray, optional
        Raw predicted probabilities of shape (n_samples, 2) or (n_samples,).
        If provided, renders a histogram of predicted probabilities underneath the graph.
    lower_ci : np.ndarray, optional
        Lower bound array for 95% confidence interval shading.
    upper_ci : np.ndarray, optional
        Upper bound array for 95% confidence interval shading.
    label : str, default="Model"
        Legend label for the plotted curve.
    dist_n_bins : int, default=20
        Number of bins for the underlying probability distribution histogram.
    dist_yscale : str, default="linear"
        Scale to use for the y-axis (e.g. linear, log, etc.)
    auto_inset : bool, default=True
        Whether to automatically render a zoomed inset box when maximum predicted probability < 0.4.
    ax : matplotlib.axes.Axes, optional
        Pre-existing Matplotlib axes instance. If None, a new figure and axes are created.
    color : str, default=_DEFAULT_THEME.primary_color
        Color specifier for main calibration elements and distribution bars.
    ci_color : str, optional
        Custom color specifier for confidence interval shading. Defaults to `color`.
    ci_alpha : float, default=_DEFAULT_THEME.ci_alpha
        Opacity level for confidence interval shaded region [0.0, 1.0].
    linestyle : str, default="-"
        Line style for the calibration curve.
    linewidth : float, default=_DEFAULT_THEME.linewidth
        Width in points for the calibration line.
    marker : str or None, default="o"
        Marker symbol for calibration points.
    ref_linestyle : str, default="--"
        Line style for the perfect calibration reference line.
    ref_color : str, default="black"
        Color specifier for the reference line.
    title : str, optional
        Axes title text.
    show_spines : bool, default=_DEFAULT_THEME.show_spines
        Whether to keep top and right border spines visible.
    **line_kwargs : Any
        Additional Matplotlib keyword arguments forwarded to `ax.plot`.

    Returns
    -------
    fig : matplotlib.figure.Figure or matplotlib.figure.SubFigure
        Parent Matplotlib figure containing the axes.
    ax : matplotlib.axes.Axes
        Matplotlib axes containing the main calibration diagram.

    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.get_figure()
        if fig is None:
            raise ValueError("The provided Axes instance is not attached to a Figure")

    # Clean line_kwargs to prevent parameter collisions
    for key in ("color", "linestyle", "linewidth", "marker", "label"):
        line_kwargs.pop(key, None)

    # 1. Reference Line
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle=ref_linestyle,
        color=ref_color,
        label="Perfectly calibrated",
    )

    # 2. Calibration Line & CI Fills
    plot_kwargs: dict[str, Any] = {
        "color": color,
        "linestyle": linestyle,
        "linewidth": linewidth,
        "label": label,
        **line_kwargs,
    }
    if marker:
        plot_kwargs["marker"] = marker

    ax.plot(prob_pred, prob_true, **plot_kwargs)

    if lower_ci is not None and upper_ci is not None:
        ax.fill_between(
            prob_pred,
            lower_ci,
            upper_ci,
            color=ci_color or color,
            alpha=ci_alpha,
            label=f"{label} 95% CI",
        )

    # 3. Axis Formatting
    ax.set_ylabel("Fraction of positives", fontweight="bold")
    ax.set_xlim(xmin=-0.02, xmax=1.02)
    ax.set_ylim(ymin=-0.02, ymax=1.02)

    if title:
        ax.set_title(title, fontweight="bold")

    if not show_spines:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # 4. Bottom Distribution Subplot
    if probas is not None:
        if probas.ndim == 2:
            probas = probas[:, 1]

        ax.tick_params(labelbottom=False)
        ax.set_xlabel("")

        divider = make_axes_locatable(ax)
        ax_dist = divider.append_axes("bottom", size="25%", pad=0.15, sharex=ax)

        bins = np.linspace(0.0, 1.0, dist_n_bins + 1)
        ax_dist.hist(
            probas,
            bins=bins,
            color=color,
            edgecolor="black",
        )
        ax_dist.set_xlabel("Mean predicted probability", fontweight="bold")
        ax_dist.set_ylabel("Count", fontweight="bold")
        ax_dist.set_yscale(dist_yscale)

        if not show_spines:
            ax_dist.spines["top"].set_visible(False)
            ax_dist.spines["right"].set_visible(False)
    else:
        ax.set_xlabel("Mean predicted probability", fontweight="bold")

    ax.legend(loc="lower right", frameon=False)

    return fig, ax


def draw_strata_heatmap(
    plot_data: np.ndarray,
    text_data: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    colorbar_label: str = r"|$\Delta$ Score|",
    vmax: float = 0.1,
    ax: Optional[Axes] = None,
    cmap: str = "cividis",
    title: Optional[str] = None,
    **heatmap_kwargs: Any,
) -> Tuple[Figure | SubFigure, Axes]:
    """Render a pre-computed strata heatmap matrix onto an Axes.

    Parameters
    ----------
    plot_data : np.ndarray
        2D matrix of delta values driving heatmap cell colors.
    text_data : np.ndarray
        2D matrix of absolute score values displayed as cell text.
    row_labels : list of str
        Y-axis tick labels (e.g., ['All strata', 'Male', 'Female']).
    col_labels : list of str
        X-axis tick labels (e.g., ['Mortality', 'Readmission']).
    colorbar_label : str, default=r"|$\\Delta$ Score|"
        Label text for the colorbar legend.
    vmax : float, default=0.1
        Maximum color scale limit for imshow.
    ax : matplotlib.axes.Axes, optional
        Pre-existing Matplotlib axes instance. If None, a new figure and axes are created.
    cmap : str, default="cividis"
        Matplotlib colormap identifier.
    title : str, optional
        Axes title text.
    **heatmap_kwargs : Any
        Additional keyword arguments forwarded to `ax.imshow`.

    Returns
    -------
    fig : matplotlib.figure.Figure or matplotlib.figure.SubFigure
        Parent Matplotlib figure containing the axes.
    ax : matplotlib.axes.Axes
        Matplotlib axes containing the rendered heatmap.

    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.get_figure()
        if fig is None:
            raise ValueError("The provided Axes instance is not attached to a Figure")

    n_rows, n_cols = plot_data.shape

    # Clean kwargs
    for key in ("cmap", "aspect", "vmax"):
        heatmap_kwargs.pop(key, None)

    im = ax.imshow(plot_data, cmap=cmap, aspect="equal", vmax=vmax, **heatmap_kwargs)

    fig.colorbar(im, ax=ax, shrink=0.8, extend="max", label=colorbar_label)

    ax.set_yticks(np.arange(n_rows), labels=row_labels)
    ax.set_xticks(
        np.arange(n_cols),
        labels=col_labels,
        rotation=-30,
        rotation_mode="anchor",
        ha="left",
    )
    ax.set_ylabel("Strata", fontweight="bold")
    ax.set_xlabel("Outcomes", fontweight="bold")

    # Render cell text annotations
    for i in range(n_cols):
        for j in range(n_rows):
            colour = "k" if np.abs(plot_data[j, i]) >= 0.8 * vmax else "w"
            ax.text(
                i,
                j,
                f"{text_data[j, i]:.2f}",
                ha="center",
                va="center",
                color=colour,
                fontsize=6,
                fontweight="bold",
            )

    ax.spines[:].set_visible(False)

    # Grid divider lines between cells
    ax.set_yticks(np.arange(n_rows + 1) - 0.5, minor=True)
    ax.set_xticks(np.arange(n_cols + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)

    if title:
        ax.set_title(title, fontweight="bold")

    return fig, ax


def draw_dca_curve(
    thresholds: np.ndarray,
    net_benefit_model: np.ndarray,
    net_benefit_all: np.ndarray,
    label: str = "Model",
    ax: Optional[Axes] = None,
    color: str = _DEFAULT_THEME.primary_color,
    linestyle: str = "-",
    linewidth: float = _DEFAULT_THEME.linewidth,
    all_linestyle: str = ":",
    all_color: str = "gray",
    none_linestyle: str = "--",
    none_color: str = "black",
    title: Optional[str] = None,
    show_spines: bool = _DEFAULT_THEME.show_spines,
    **line_kwargs: Any,
) -> Tuple[Figure | SubFigure, Axes]:
    """Render a Decision Curve Analysis (DCA) plot comparing net benefit across thresholds.

    Parameters
    ----------
    thresholds : np.ndarray
        Array of threshold probabilities along the X-axis.
    net_benefit_model : np.ndarray
        Computed Net Benefit values for the evaluated model.
    net_benefit_all : np.ndarray
        Computed Net Benefit values for the 'Treat All' strategy.
    label : str, default="Model"
        Legend label for the model net benefit curve.
    ax : matplotlib.axes.Axes, optional
        Pre-existing Matplotlib axes instance. If None, a new figure and axes are created.
    color : str, default=_DEFAULT_THEME.primary_color
        Color specifier for the model net benefit line.
    linestyle : str, default="-"
        Line style for the model curve.
    linewidth : float, default=_DEFAULT_THEME.linewidth
        Width in points for the model curve line.
    all_linestyle : str, default=":"
        Line style for the 'Treat All' reference curve.
    all_color : str, default="gray"
        Color specifier for the 'Treat All' reference curve.
    none_linestyle : str, default="--"
        Line style for the 'Treat None' reference baseline.
    none_color : str, default="black"
        Color specifier for the 'Treat None' reference baseline.
    title : str, optional
        Axes title text.
    show_spines : bool, default=_DEFAULT_THEME.show_spines
        Whether to keep top and right border spines visible.
    **line_kwargs : Any
        Additional Matplotlib keyword arguments forwarded to `ax.plot` for the model curve.

    Returns
    -------
    fig : matplotlib.figure.Figure or matplotlib.figure.SubFigure
        Parent Matplotlib figure containing the axes.
    ax : matplotlib.axes.Axes
        Matplotlib axes containing the rendered decision curves.

    Raises
    ------
    ValueError
        If ax is provided but not attached to a Figure.

    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.get_figure()
        if fig is None:
            raise ValueError("The provided Axes instance is not attached to a Figure")

    # Draw reference strategy curves
    ax.plot(
        thresholds,
        np.zeros_like(thresholds),
        linestyle=none_linestyle,
        color=none_color,
        label="Treat None",
    )
    ax.plot(
        thresholds,
        net_benefit_all,
        linestyle=all_linestyle,
        color=all_color,
        label="Treat All",
    )

    # Clean line_kwargs to prevent parameter collisions
    for key in ("color", "linestyle", "linewidth", "label"):
        line_kwargs.pop(key, None)

    # Draw main model Net Benefit curve
    ax.plot(
        thresholds,
        net_benefit_model,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        label=label,
        **line_kwargs,
    )

    # Format axis bounds and labels
    ax.set_xlabel("Threshold probability", fontweight="bold")
    ax.set_ylabel("Net benefit", fontweight="bold")
    ax.set_xlim(xmin=-0.02, xmax=1.02)

    # Dynamically set y limits based on data range
    y_min = max(-0.05, min(np.min(net_benefit_model), np.min(net_benefit_all)) - 0.02)
    y_max = max(np.max(net_benefit_model), np.max(net_benefit_all)) + 0.05
    ax.set_ylim(ymin=y_min, ymax=y_max)

    if title:
        ax.set_title(title, fontweight="bold")

    if not show_spines:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    ax.legend(loc="upper right", frameon=False)

    return fig, ax
