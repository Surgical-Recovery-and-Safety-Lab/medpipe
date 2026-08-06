"""
Stateless drawing primitives for Medpipe visualizations.
"""

from typing import Any, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure

from medpipe.visualisation.themes import MedpipeTheme

# Medpipe Default Palette Defaults
_DEFAULT_THEME = MedpipeTheme()


def draw_probability_distribution(
    probas: np.ndarray,
    n_bins: int = 10,
    label: str = "Predicted Probabilities",
    ax: Optional[Axes] = None,
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

    ax.set_xlabel("Predicted Probabilities", fontweight="bold")
    ax.set_ylabel("Count", fontweight="bold")
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
    fig : matplotlib.figure.Figure
        Parent Matplotlib figure containing the axes.
    ax : matplotlib.axes.Axes
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
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontweight="bold")
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontweight="bold")
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
    fig : matplotlib.figure.Figure or matplotlib.figure.SubFigure
        Parent Matplotlib figure containing the axes.
    ax : matplotlib.axes.Axes
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
    lower_ci: Optional[np.ndarray] = None,
    upper_ci: Optional[np.ndarray] = None,
    label: str = "Model",
    ax: Optional[Axes] = None,
    color: str = _DEFAULT_THEME.primary_color,
    ci_color: Optional[str] = None,
    ci_alpha: float = _DEFAULT_THEME.ci_alpha,
    linestyle: str = "-",
    linewidth: float = _DEFAULT_THEME.linewidth,
    marker: str = "o",
    ref_linestyle: str = "--",
    ref_color: str = "black",
    title: Optional[str] = None,
    show_spines: bool = _DEFAULT_THEME.show_spines,
    **line_kwargs: Any,
) -> Tuple[Figure | SubFigure, Axes]:
    """Render a reliability diagram (calibration curve) with optional CI bounds.

    Parameters
    ----------
    prob_true : np.ndarray
        Fraction of positives in each calibration bin.
    prob_pred : np.ndarray
        Mean predicted probability in each calibration bin.
    lower_ci : np.ndarray, optional
        Lower bound array of true probabilities for 95% confidence interval shading.
    upper_ci : np.ndarray, optional
        Upper bound array of true probabilities for 95% confidence interval shading.
    label : str, default="Model"
        Legend label for the plotted calibration curve.
    ax : matplotlib.axes.Axes, optional
        Pre-existing Matplotlib axes instance. If None, a new figure and axes are created.
    color : str, default=_DEFAULT_THEME.primary_color
        Color specifier for the main calibration line and default CI fill.
    ci_color : str, optional
        Custom color specifier for confidence interval shading. Defaults to `color`.
    ci_alpha : float, default=_DEFAULT_THEME.ci_alpha
        Opacity level for confidence interval shaded region [0.0, 1.0].
    linestyle : str, default="-"
        Line style specification for the calibration curve.
    linewidth : float, default=_DEFAULT_THEME.linewidth
        Width in points for the calibration line.
    marker : str, default="o"
        Marker symbol for binned probability points.
    ref_linestyle : str, default="--"
        Line style specification for the diagonal perfect calibration reference line.
    ref_color : str, default="black"
        Color specifier for the perfect calibration reference line.
    title : str, optional
        Axes title text.
    show_spines : bool, default=_DEFAULT_THEME.show_spines
        Whether to keep top and right border spines visible.
    **line_kwargs : Any
        Additional Matplotlib keyword arguments forwarded to `ax.plot` for the calibration curve.

    Returns
    -------
    fig : matplotlib.figure.Figure or matplotlib.figure.SubFigure
        Parent Matplotlib figure containing the axes.
    ax : matplotlib.axes.Axes
        Matplotlib axes containing all rendered calibration elements.

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

    # Draw perfect calibration diagonal reference line
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle=ref_linestyle,
        color=ref_color,
        label="Perfectly calibrated",
    )

    # Clean line_kwargs to prevent parameter collisions
    for key in ("color", "linestyle", "linewidth", "marker", "label"):
        line_kwargs.pop(key, None)

    # Draw main calibration curve
    ax.plot(
        prob_pred,
        prob_true,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        marker=marker,
        label=label,
        **line_kwargs,
    )

    # Draw confidence interval shading if bounds are supplied
    if lower_ci is not None and upper_ci is not None:
        ax.fill_between(
            prob_pred,
            lower_ci,
            upper_ci,
            color=ci_color or color,
            alpha=ci_alpha,
            label=f"{label} 95% CI",
        )

    # Format axis bounds and labels
    ax.set_xlabel("Mean Predicted Probability", fontweight="bold")
    ax.set_ylabel("Fraction of Positives", fontweight="bold")
    ax.set_xlim(xmin=-0.02, xmax=1.02)
    ax.set_ylim(ymin=-0.02, ymax=1.02)

    if title:
        ax.set_title(title, fontweight="bold")

    if not show_spines:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    ax.legend(loc="lower right", frameon=False)

    return fig, ax
