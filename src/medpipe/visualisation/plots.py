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
