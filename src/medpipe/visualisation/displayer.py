"""
High-level display and visualisation manager module.
"""

from pathlib import Path
from typing import Any, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.metrics import auc, roc_curve

from medpipe.pipeline.orchestrator import MedpipeOrchestrator
from medpipe.utils.logger import get_console_logger
from medpipe.visualisation.plots import draw_roc_curve
from medpipe.visualisation.themes import MedpipeTheme


class MedpipeDisplayer:
    """High-level visualisation and display manager for Medpipe pipeline runs.

    This class handles statistical calculations (such as bootstrap confidence intervals
    and calibration metrics), applies theme aesthetics, delegates rendering to stateless
    drawing functions, and manages artifact persistence.

    Parameters
    ----------
    orchestrator : MedpipeOrchestrator
        Active pipeline orchestrator instance containing execution context and run directory.
    theme : MedpipeTheme, optional
        Aesthetic theme configuration. If None, defaults to `MedpipeTheme()`.

    Attributes
    ----------
    orchestrator : MedpipeOrchestrator
        Associated pipeline orchestrator instance.
    run_dir : Path
        Path to the output directory for storing generated figures.
    theme : MedpipeTheme
        Active visual theme specification.
    logger : logging.Logger
        Console logger instance for displayer operations.

    Methods
    -------
    plot_roc_curve(y_true, probas, outcome="default", label=None, n_bootstraps=1000, save=True, show=False, **style_kwargs)
        Compute ROC statistics, render curve with optional bootstrap CIs, and save output.
    """

    def __init__(
        self,
        orchestrator: MedpipeOrchestrator,
        theme: Optional[MedpipeTheme] = None,
    ) -> None:
        self.orchestrator = orchestrator
        self.run_dir = orchestrator.run_dir
        self.theme = theme or MedpipeTheme()
        self.logger = get_console_logger("medpipe.displayer")

    # --- Internal Helpers ---

    def _compute_roc_data(
        self,
        y_true: np.ndarray,
        probas: np.ndarray,
        n_bootstraps: int = 1000,
        random_state: Optional[int] = 42,
    ) -> Tuple[
        np.ndarray, np.ndarray, float, Optional[np.ndarray], Optional[np.ndarray]
    ]:
        """Compute Receiver Operating Characteristic metrics and bootstrap CIs.

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth binary target labels of shape (n_samples,).
        probas : np.ndarray
            Predicted probabilities of shape (n_samples, 2) or (n_samples,).
        n_bootstraps : int, default=1000
            Number of bootstrap iterations for 95% confidence interval estimation.
        random_state : int, optional, default=42
            Random seed for bootstrap resampling reproducibility.

        Returns
        -------
        fpr : np.ndarray
            False positive rates across decision thresholds.
        tpr : np.ndarray
            True positive rates across decision thresholds.
        roc_auc : float
            Area Under the ROC Curve.
        lower_ci : np.ndarray or None
            Lower 2.5% percentile bound of TPR across bootstraps, if n_bootstraps > 0.
        upper_ci : np.ndarray or None
            Upper 97.5% percentile bound of TPR across bootstraps, if n_bootstraps > 0.

        """
        if probas.ndim == 2:
            probas = probas[:, 1]

        y_true = np.asarray(y_true).squeeze()

        fpr, tpr, _ = roc_curve(y_true, probas)
        roc_auc = float(auc(fpr, tpr))

        if n_bootstraps <= 0:
            return fpr, tpr, roc_auc, None, None

        rng = np.random.default_rng(random_state)
        n_samples = len(y_true)
        boots = []

        for _ in range(n_bootstraps):
            idx = rng.choice(n_samples, size=n_samples, replace=True)
            # Ensure bootstrap resample includes both binary classes
            if len(np.unique(y_true[idx])) < 2:
                continue
            fpr_b, tpr_b, _ = roc_curve(y_true[idx], probas[idx])
            boots.append(np.interp(fpr, fpr_b, tpr_b))

        if not boots:
            return fpr, tpr, roc_auc, None, None

        lower_ci = np.percentile(boots, 2.5, axis=0)
        upper_ci = np.percentile(boots, 97.5, axis=0)

        return fpr, tpr, roc_auc, lower_ci, upper_ci

    def _save_figure(
        self, fig: Figure, filename: str, outcome: Optional[str] = None
    ) -> Path:
        """Persist figure artifact to disk in the run directory structure.

        Parameters
        ----------
        fig : plt.Figure
            Matplotlib figure object to be saved.
        filename : str
            Base filename for the saved image file (without extension).
        outcome : str, optional
            Subdirectory name corresponding to a specific clinical outcome.

        Returns
        -------
        Path
            Absolute path to the created plot file.
        """
        plot_dir = self.run_dir / "plots"
        if outcome:
            plot_dir = plot_dir / outcome
        plot_dir.mkdir(parents=True, exist_ok=True)

        save_path = plot_dir / f"{filename}.png"
        fig.savefig(save_path, dpi=self.theme.dpi, bbox_inches="tight")
        self.logger.info(f"Saved plot artifact to {save_path}")
        return save_path

    # --- High-Level Plotting Methods ---

    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        probas: np.ndarray,
        outcome: str = "default",
        label: Optional[str] = None,
        n_bootstraps: int = 1000,
        save: bool = True,
        show: bool = False,
        **style_kwargs: Any,
    ) -> Tuple[Figure, Axes]:
        """Compute ROC metrics, render curve with confidence intervals, and save figure.

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth binary target labels of shape (n_samples,).
        probas : np.ndarray
            Predicted probabilities of shape (n_samples, 2) or (n_samples,).
        outcome : str, default="default"
            Outcome identifier used for figure titles and folder structuring.
        label : str, optional
            Legend label for the model. If None, defaults to 'Model (AUC = X.XX)'.
        n_bootstraps : int, default=1000
            Number of bootstrap iterations for confidence intervals. Set to 0 to disable.
        save : bool, default=True
            Automatically save the generated plot to the run directory.
        show : bool, default=False
            Whether to display the plot interactively before closing.
        **style_kwargs : Any
            Additional style parameters forwarded to `draw_roc_curve`.

        Returns
        -------
        fig : Figure
            Rendered Matplotlib figure object.
        ax : Axes
            Matplotlib axes containing the plotted elements.

        """
        fpr, tpr, roc_auc, lower_ci, upper_ci = self._compute_roc_data(
            y_true=y_true,
            probas=probas,
            n_bootstraps=n_bootstraps,
        )

        display_label = label or f"Model (AUC = {roc_auc:.3f})"

        # Apply global theme context
        with (
            plt.style.context(self.theme.style_sheet),
            plt.rc_context(self.theme.to_rc_params()),
        ):
            fig, ax = draw_roc_curve(
                fpr=fpr,
                tpr=tpr,
                lower_ci=lower_ci,
                upper_ci=upper_ci,
                label=display_label,
                color=style_kwargs.pop("color", self.theme.primary_color),
                ci_alpha=style_kwargs.pop("ci_alpha", self.theme.ci_alpha),
                linewidth=style_kwargs.pop("linewidth", self.theme.linewidth),
                show_spines=style_kwargs.pop("show_spines", self.theme.show_spines),
                title=f"ROC Curve - {outcome.capitalize()}",
                **style_kwargs,
            )

        if save:
            self._save_figure(fig=fig, filename=f"{outcome}_roc_curve", outcome=outcome)

        if show:
            plt.show()
        elif save:
            plt.close(fig)

        return fig, ax
