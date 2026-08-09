"""
High-level display and visualisation manager module.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)

from medpipe.pipeline.orchestrator import MedpipeOrchestrator
from medpipe.utils.logger import get_console_logger
from medpipe.visualisation.plots import (
    draw_dca_curve,
    draw_precision_recall_curve,
    draw_probability_distribution,
    draw_reliability_diagram,
    draw_roc_curve,
    draw_strata_heatmap,
)
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
    plot_roc_curve(y_true, probas, outcome="default", label=None,
    n_bootstraps=1000, save=True, show=False, **style_kwargs)
        Compute ROC statistics, render curve with optional bootstrap CIs,
        and save output.
    plot_precision_recall_curve(y_true, probas, outcome="default",
    label=None, n_bootstraps=1000, save=True, show=False, **style_kwargs)
        Compute PR statistics, render curve with optional bootstrap CIs,
        and save output.
    plot_probability_distribution(probas, outcome="default", n_bins=10,
    label=None, save=True, show=False, **style_kwargs)
        Render predicted probability distribution histogram and save output.
    plot_reliability_diagram(y_true, probas, outcome="default", n_bins=10,
    strategy="uniform", label=None, n_bootstraps=1000, save=True,
    show=False, **style_kwargs)
        Compute calibration metrics (binned or spline), render reliability
        diagram with optional CIs, and save output.
    plot_strata_heatmap(outcomes, metric, strata, scores, strata_scores,
    save=True, show=False, **style_kwargs)
        Validate subgroup inputs, compute delta matrix, render strata heatmap,
        and save output.
    plot_dca_curve(y_true, probas, outcome="default", thresholds=None,
    label=None, save=True, show=False, **style_kwargs)
        Compute Net Benefit across decision thresholds, render DCA plot,
        and save output.
    plot_all(y_true, probas, outcome="default", n_bootstraps=1000, save=True,
    show=False, **style_kwargs)
        Execute all core model evaluation visualization routines
        (ROC, PR, distribution, reliability, DCA) for a given outcome.
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

    # --- Config Resolution ---

    @staticmethod
    def _normalize_plot_type(plot_type: str) -> str:
        """Normalize plot aliases to canonical names.

        Parameters
        ----------
        plot_type : str
            Raw identifier or alias for a specific plot type
            (e.g., 'calibration', 'pr_curve').

        Returns
        -------
        str
            Canonical plot type identifier used for consistent configuration lookup.

        """
        mapping = {
            "calibration": "reliability",
            "reliability_diagram": "reliability",
            "pr": "precision_recall",
            "pr_curve": "precision_recall",
            "roc_curve": "roc",
            "distribution": "probability_distribution",
            "dist": "probability_distribution",
            "dca_curve": "dca",
        }
        return mapping.get(plot_type.lower(), plot_type.lower())

    def _resolve_plot_config(
        self,
        plot_type: str,
        outcome: Optional[str] = None,
        **runtime_kwargs: Any,
    ) -> Dict[str, Any]:
        """Resolve plot parameters hierarchically across configuration levels.

        Applies parameter precedence in the following order (lowest to highest):
        1. Default global display settings (`display_cfg.defaults`)
        2. Plot-type overrides (`display_cfg.overrides`)
        3. Outcome-specific plot overrides (`display_cfg.outcome_overrides`)
        4. Explicit non-None runtime arguments (`runtime_kwargs`)

        Parameters
        ----------
        plot_type : str
            Plot identifier or alias (e.g., 'calibration', 'roc', 'distribution').
        outcome : str, optional
            Outcome key used to retrieve outcome-specific plot overrides.
        **runtime_kwargs : Any
            Runtime keyword arguments passed directly to the calling plot method.

        Returns
        -------
        dict of {str : Any}
            Fully resolved dictionary of parameters for the specified plot.

        """
        display_cfg = getattr(self.orchestrator.config, "display", None)

        if display_cfg is None:
            config_params: Dict[str, Any] = {
                "n_bootstraps": 1000,
                "save": True,
                "show": False,
                "n_bins": 10,
                "strategy": "uniform",
            }
            overrides: Dict[str, Any] = {}
            outcome_overrides: Dict[str, Any] = {}
        else:
            config_params = display_cfg.defaults.model_dump()
            overrides = display_cfg.overrides
            outcome_overrides = display_cfg.outcome_overrides

        canonical_type = self._normalize_plot_type(plot_type)

        # 1. Apply global plot-type overrides
        for key in (plot_type.lower(), canonical_type):
            if key in overrides:
                config_params.update(overrides[key])

        # 2. Apply outcome-specific plot overrides
        if outcome and outcome in outcome_overrides:
            out_cfg = outcome_overrides[outcome]
            for key in (plot_type.lower(), canonical_type):
                if key in out_cfg:
                    config_params.update(out_cfg[key])

        # 3. Apply explicit non-None runtime kwargs overrides
        for k, v in runtime_kwargs.items():
            if v is not None:
                config_params[k] = v

        return config_params

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

    def _compute_precision_recall_data(
        self,
        y_true: np.ndarray,
        probas: np.ndarray,
        n_bootstraps: int = 1000,
        random_state: Optional[int] = 42,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        float,
        float,
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        """Compute Precision-Recall metrics, Average Precision, and bootstrap CIs.

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
        precision : np.ndarray
            Precision values across decision thresholds.
        recall : np.ndarray
            Recall values across decision thresholds.
        ap_score : float
            Average Precision (AP) score.
        baseline : float
            Prevalence / fraction of positive ground truth samples.
        lower_ci : np.ndarray or None
            Lower 2.5% percentile bound of precision across bootstraps, if n_bootstraps > 0.
        upper_ci : np.ndarray or None
            Upper 97.5% percentile bound of precision across bootstraps, if n_bootstraps > 0.

        """
        if probas.ndim == 2:
            probas = probas[:, 1]

        y_true = np.asarray(y_true).squeeze()

        precision, recall, _ = precision_recall_curve(y_true, probas)
        ap_score = float(average_precision_score(y_true, probas))
        baseline = float(np.mean(y_true))

        if n_bootstraps <= 0:
            return precision, recall, ap_score, baseline, None, None

        rng = np.random.default_rng(random_state)
        n_samples = len(y_true)
        boots = []

        # Scikit-learn precision_recall_curve outputs recall in descending order
        rev_recall = recall[::-1]

        for _ in range(n_bootstraps):
            idx = rng.choice(n_samples, size=n_samples, replace=True)
            if len(np.unique(y_true[idx])) < 2:
                continue

            prec_b, rec_b, _ = precision_recall_curve(y_true[idx], probas[idx])
            interp_prec = np.interp(rev_recall, rec_b[::-1], prec_b[::-1])
            boots.append(interp_prec[::-1])

        if not boots:
            return precision, recall, ap_score, baseline, None, None

        lower_ci = np.percentile(boots, 2.5, axis=0)
        upper_ci = np.percentile(boots, 97.5, axis=0)

        return precision, recall, ap_score, baseline, lower_ci, upper_ci

    def _compute_reliability_data(
        self,
        y_true: np.ndarray,
        probas: np.ndarray,
        n_bins: int = 10,
        strategy: str = "uniform",
        n_bootstraps: int = 1000,
        random_state: Optional[int] = 42,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        """Compute calibration curve data (reliability diagram) and bootstrap CIs.

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth binary target labels of shape (n_samples,).
        probas : np.ndarray
            Predicted probabilities of shape (n_samples, 2) or (n_samples,).
        n_bins : int, default=10
            Number of bins used for 'uniform' or 'quantile' binning strategies.
        strategy : {'uniform', 'quantile', 'spline'}, default='uniform'
            Calibration curve estimation strategy.
        n_bootstraps : int, default=1000
            Number of bootstrap iterations for 95% confidence interval estimation.
        random_state : int, optional, default=42
            Random seed for bootstrap resampling reproducibility.

        Returns
        -------
        prob_true : np.ndarray
            Fraction of positives or spline-calibrated probabilities.
        prob_pred : np.ndarray
            Mean predicted probabilities or evaluation grid points.
        lower_ci : np.ndarray or None
            Lower 2.5% percentile bound of calibration curve across bootstraps.
        upper_ci : np.ndarray or None
            Upper 97.5% percentile bound of calibration curve across bootstraps.

        """
        if probas.ndim == 2:
            probas = probas[:, 1]

        y_true = np.asarray(y_true).squeeze()

        if strategy == "spline":
            from splinecalib import SplineCalib

            sc = SplineCalib()
            sc.fit(probas, y_true)
            prob_pred = np.linspace(0.0, 1.0, 100)
            prob_true = sc.calibrate(prob_pred)

            assert prob_true is not None
            if prob_true.ndim == 2:
                prob_true = prob_true[:, 1]
        else:
            prob_true, prob_pred = calibration_curve(
                y_true, probas, n_bins=n_bins, strategy=strategy
            )

        if n_bootstraps <= 0 or len(prob_pred) == 0:
            return prob_true, prob_pred, None, None

        rng = np.random.default_rng(random_state)
        n_samples = len(y_true)
        boots = []

        for _ in range(n_bootstraps):
            idx = rng.choice(n_samples, size=n_samples, replace=True)
            if len(np.unique(y_true[idx])) < 2:
                continue

            if strategy == "spline":
                sc_b = SplineCalib()  # type: ignore
                sc_b.fit(probas[idx], y_true[idx])
                b_true = sc_b.calibrate(prob_pred)

                assert b_true is not None
                if b_true.ndim == 2:
                    b_true = b_true[:, 1]
                boots.append(b_true)
            else:
                b_true, b_pred = calibration_curve(
                    y_true[idx], probas[idx], n_bins=n_bins, strategy=strategy
                )
                if len(b_pred) > 1:
                    interp_true = np.interp(prob_pred, b_pred, b_true)
                    boots.append(interp_true)

        if not boots:
            return prob_true, prob_pred, None, None

        lower_ci = np.percentile(boots, 2.5, axis=0)
        upper_ci = np.percentile(boots, 97.5, axis=0)

        return prob_true, prob_pred, lower_ci, upper_ci

    def _save_figure(
        self, fig: Figure | SubFigure, filename: str, outcome: Optional[str] = None
    ) -> Path:
        """Persist figure artifact to disk in the run directory structure.

        Parameters
        ----------
        fig : Figure | SubFigure
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

    def _compute_dca_data(
        self,
        y_true: np.ndarray,
        probas: np.ndarray,
        thresholds: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute Net Benefit for Model, Treat All, and Treat None.

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth binary target labels of shape (n_samples,).
        probas : np.ndarray
            Predicted probabilities of shape (n_samples, 2) or (n_samples,).
        thresholds : np.ndarray, optional
            Array of threshold probabilities. Defaults to `np.linspace(0.01, 0.99, 99)`.

        Returns
        -------
        thresholds : np.ndarray
            Evaluated threshold probabilities.
        net_benefit_model : np.ndarray
            Model Net Benefit across thresholds.
        net_benefit_all : np.ndarray
            Treat All Net Benefit across thresholds.
        """
        if probas.ndim == 2:
            probas = probas[:, 1]

        y_true = np.asarray(y_true).squeeze()
        n_samples = len(y_true)

        if thresholds is None:
            thresholds = np.linspace(0.01, 0.99, 99)

        positives = np.sum(y_true == 1)
        negatives = n_samples - positives

        # Calculate Treat All Net Benefit
        net_benefit_all = (positives / n_samples) - (negatives / n_samples) * (
            thresholds / (1.0 - thresholds)
        )

        # Calculate Model Net Benefit
        nb_model_list = []
        for p_t in thresholds:
            y_pred = probas >= p_t
            tp = np.sum((y_pred == 1) & (y_true == 1))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            nb = (tp / n_samples) - (fp / n_samples) * (p_t / (1.0 - p_t))
            nb_model_list.append(nb)

        net_benefit_model = np.array(nb_model_list)

        return thresholds, net_benefit_model, net_benefit_all

    # --- High-Level Plotting Methods ---

    def plot_probability_distribution(
        self,
        probas: np.ndarray,
        outcome: str = "default",
        n_bins: Optional[int] = None,
        label: Optional[str] = None,
        save: Optional[bool] = None,
        show: Optional[bool] = None,
        **style_kwargs: Any,
    ) -> Tuple[Figure | SubFigure, Axes]:
        """Render prediction probability histogram and save figure artifact.

        Parameters
        ----------
        probas : np.ndarray
            Predicted probabilities of shape (n_samples, 2) or (n_samples,).
        outcome : str, default="default"
            Outcome identifier used for figure titles and directory structuring.
        n_bins : int, default=10
            Number of equal-width bins for the histogram.
        label : str, optional
            Legend label. Defaults to "Predicted Probabilities".
        save : bool, optional
            Automatically save the generated plot to the run directory.
        show : bool, optional
            Whether to display the plot interactively before closing.
        **style_kwargs : Any
            Additional style parameters forwarded to `draw_probability_distribution`.

        Returns
        -------
        fig : Figure | SubFigure
            Rendered Matplotlib figure object.
        ax : Axes
            Matplotlib axes containing the plotted histogram.

        """
        cfg = self._resolve_plot_config(
            plot_type="distribution",
            outcome=outcome,
            n_bins=n_bins,
            save=save,
            show=show,
            **style_kwargs,
        )

        n_bins_val = cfg["n_bins"]
        save_val = cfg["save"]
        show_val = cfg["show"]

        self.logger.info(
            f"[{outcome}] Starting predicted probability distribution plotting."
        )
        self.logger.debug(
            f"[{outcome}] Plotting predicted probability distribution with "
            f"probabilities: {probas.shape} and {n_bins_val} bins."
        )
        display_label = label or "Predicted Probabilities"

        with (plt.rc_context(self.theme.to_rc_params()),):
            fig, ax = draw_probability_distribution(
                probas=probas,
                n_bins=n_bins_val,
                label=display_label,
                color=style_kwargs.pop("color", self.theme.primary_color),
                show_spines=style_kwargs.pop("show_spines", self.theme.show_spines),
                title=f"Probability Distribution - {outcome.capitalize()}",
                **style_kwargs,
            )

        if save_val:
            self._save_figure(
                fig=fig,
                filename=f"{outcome}_probability_distribution",
                outcome=outcome,
            )

        if show_val:
            plt.show()
        elif save_val:
            plt.close(fig)

        return fig, ax

    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        probas: np.ndarray,
        outcome: str = "default",
        label: Optional[str] = None,
        n_bootstraps: Optional[int] = None,
        save: Optional[bool] = None,
        show: Optional[bool] = None,
        **style_kwargs: Any,
    ) -> Tuple[Figure | SubFigure, Axes]:
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
        n_bootstraps : int, optional
            Number of bootstrap iterations for confidence intervals. Set to 0 to disable.
        save : bool, optional
            Automatically save the generated plot to the run directory.
        show : bool, optional
            Whether to display the plot interactively before closing.
        **style_kwargs : Any
            Additional style parameters forwarded to `draw_roc_curve`.

        Returns
        -------
        fig : Figure | SubFigure
            Rendered Matplotlib figure object.
        ax : Axes
            Matplotlib axes containing the plotted elements.

        """
        cfg = self._resolve_plot_config(
            plot_type="roc",
            outcome=outcome,
            n_bootstraps=n_bootstraps,
            save=save,
            show=show,
            **style_kwargs,
        )

        n_bootstraps_val = cfg["n_bootstraps"]
        save_val = cfg["save"]
        show_val = cfg["show"]

        self.logger.info(f"[{outcome}] Starting ROC curve plotting.")
        self.logger.debug(
            f"[{outcome}] Plotting ROC curve with "
            f"y: {y_true.shape} and {n_bootstraps_val} bootstrap iterations."
        )
        fpr, tpr, roc_auc, lower_ci, upper_ci = self._compute_roc_data(
            y_true=y_true,
            probas=probas,
            n_bootstraps=n_bootstraps_val,
        )

        display_label = label or f"Model (AUC = {roc_auc:.3f})"

        # Apply global theme context
        with (plt.rc_context(self.theme.to_rc_params()),):
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

        if save_val:
            self._save_figure(fig=fig, filename=f"{outcome}_roc_curve", outcome=outcome)

        if show_val:
            plt.show()
        elif save_val:
            plt.close(fig)

        return fig, ax

    def plot_precision_recall_curve(
        self,
        y_true: np.ndarray,
        probas: np.ndarray,
        outcome: str = "default",
        label: Optional[str] = None,
        n_bootstraps: Optional[int] = None,
        save: Optional[bool] = None,
        show: Optional[bool] = None,
        **style_kwargs: Any,
    ) -> Tuple[Figure | SubFigure, Axes]:
        """Compute PR metrics, render curve with confidence intervals, and save figure.

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth binary target labels of shape (n_samples,).
        probas : np.ndarray
            Predicted probabilities of shape (n_samples, 2) or (n_samples,).
        outcome : str, default="default"
            Outcome identifier used for figure titles and folder structuring.
        label : str, optional
            Legend label for the model. If None, defaults to 'Model (AP = X.XX)'.
        n_bootstraps : int, optional
            Number of bootstrap iterations for confidence intervals. Set to 0 to disable.
        save : bool, optional
            Automatically save the generated plot to the run directory.
        show : bool, optional
            Whether to display the plot interactively before closing.
        **style_kwargs : Any
            Additional style parameters forwarded to `draw_precision_recall_curve`.

        Returns
        -------
        fig : Figure | SubFigure
            Rendered Matplotlib figure object.
        ax : Axes
            Matplotlib axes containing the plotted elements.

        """
        cfg = self._resolve_plot_config(
            plot_type="precision_recall",
            outcome=outcome,
            n_bootstraps=n_bootstraps,
            save=save,
            show=show,
            **style_kwargs,
        )

        n_bootstraps_val = cfg["n_bootstraps"]
        save_val = cfg["save"]
        show_val = cfg["show"]

        self.logger.info(f"[{outcome}] Starting PR curve plotting.")
        self.logger.debug(
            f"[{outcome}] Plotting PR curve with "
            f"y: {y_true.shape} and {n_bootstraps_val} bootstrap iterations."
        )
        (
            precision,
            recall,
            ap_score,
            baseline,
            lower_ci,
            upper_ci,
        ) = self._compute_precision_recall_data(
            y_true=y_true,
            probas=probas,
            n_bootstraps=n_bootstraps_val,
        )

        display_label = label or f"Model (AP = {ap_score:.3f})"

        with (plt.rc_context(self.theme.to_rc_params()),):
            fig, ax = draw_precision_recall_curve(
                precision=precision,
                recall=recall,
                lower_ci=lower_ci,
                upper_ci=upper_ci,
                baseline=baseline,
                label=display_label,
                color=style_kwargs.pop("color", self.theme.primary_color),
                ci_alpha=style_kwargs.pop("ci_alpha", self.theme.ci_alpha),
                linewidth=style_kwargs.pop("linewidth", self.theme.linewidth),
                show_spines=style_kwargs.pop("show_spines", self.theme.show_spines),
                title=f"Precision-Recall Curve - {outcome.capitalize()}",
                **style_kwargs,
            )

        if save_val:
            self._save_figure(fig=fig, filename=f"{outcome}_pr_curve", outcome=outcome)

        if show_val:
            plt.show()
        elif save_val:
            plt.close(fig)

        return fig, ax

    def plot_reliability_diagram(
        self,
        y_true: np.ndarray,
        probas: np.ndarray,
        outcome: str = "default",
        n_bins: int = 10,
        strategy: Optional[str] = None,
        label: Optional[str] = None,
        n_bootstraps: Optional[int] = None,
        save: Optional[bool] = None,
        show: Optional[bool] = None,
        **style_kwargs: Any,
    ) -> Tuple[Figure | SubFigure, Axes]:
        """Compute calibration data, render a reliability diagram with a
        probability distribution subplot, and save figure.

        Parameters
        ----------
        y_true : numpy.ndarray
            Ground truth binary target labels of shape (n_samples,).
        probas : numpy.ndarray
            Predicted probabilities of shape (n_samples, 2) or (n_samples,).
        outcome : str, default="default"
            Outcome identifier used for figure titles and directory structuring.
        n_bins : int, default=10
            Number of calibration bins (ignored if strategy='spline').
        strategy : {'uniform', 'quantile', 'spline'}, optional
            Binning or smoothing strategy for calibration calculation.
        label : str, optional
            Legend label for the model curve. Defaults to 'Model'.
        n_bootstraps : int, optional
            Number of bootstrap iterations for confidence intervals. Set to 0 to disable.
        save : bool, optional
            Automatically save the generated plot to the run directory.
        show : bool, optional
            Whether to display the plot interactively before closing.
        **style_kwargs : Any
            Additional style parameters forwarded to `draw_reliability_diagram`.

        Returns
        -------
        fig : matplotlib.figure.Figure or matplotlib.figure.SubFigure
            Rendered Matplotlib figure object.
        ax : matplotlib.axes.Axes
            Matplotlib axes containing the main reliability diagram.

        """
        cfg = self._resolve_plot_config(
            plot_type="reliability",
            outcome=outcome,
            n_bins=n_bins,
            strategy=strategy,
            n_bootstraps=n_bootstraps,
            save=save,
            show=show,
            **style_kwargs,
        )

        n_bins_val = cfg["n_bins"]
        strategy_val = cfg["strategy"]
        n_bootstraps_val = cfg["n_bootstraps"]
        save_val = cfg["save"]
        show_val = cfg["show"]
        breakpoint()
        self.logger.info(f"[{outcome}] Starting reliability diagram plotting.")
        self.logger.debug(
            f"[{outcome}] Plotting reliability diagram with "
            f"y: {y_true.shape}, {n_bootstraps_val} bootstrap iterations, "
            f"{n_bins_val} bins, and {strategy_val} strategy."
        )
        prob_true, prob_pred, lower_ci, upper_ci = self._compute_reliability_data(
            y_true=y_true,
            probas=probas,
            n_bins=n_bins_val,
            strategy=strategy_val,
            n_bootstraps=n_bootstraps_val,
        )

        display_label = label or "Model"

        # Suppress scatter points for smooth continuous spline curves
        if strategy == "spline":
            style_kwargs.setdefault("marker", None)

        with (plt.rc_context(self.theme.to_rc_params()),):
            fig, ax = draw_reliability_diagram(
                prob_true=prob_true,
                prob_pred=prob_pred,
                probas=probas,  # Pass raw probabilities for bottom histogram
                lower_ci=lower_ci,
                upper_ci=upper_ci,
                label=display_label,
                color=style_kwargs.pop("color", self.theme.primary_color),
                ci_alpha=style_kwargs.pop("ci_alpha", self.theme.ci_alpha),
                linewidth=style_kwargs.pop("linewidth", self.theme.linewidth),
                show_spines=style_kwargs.pop("show_spines", self.theme.show_spines),
                title=f"Reliability Diagram - {outcome.capitalize()}",
                **style_kwargs,
            )

        if save_val:
            self._save_figure(
                fig=fig, filename=f"{outcome}_reliability_diagram", outcome=outcome
            )

        if show_val:
            plt.show()
        elif save_val:
            plt.close(fig)

        return fig, ax

    def plot_strata_heatmap(
        self,
        outcomes: list[str],
        metric: str,
        strata: list[str],
        scores: np.ndarray,
        strata_scores: np.ndarray,
        save: Optional[bool] = None,
        show: Optional[bool] = None,
        **style_kwargs: Any,
    ) -> Tuple[Figure | SubFigure, Axes]:
        """Validate strata data, compute delta matrix, and render heatmap.

        Parameters
        ----------
        outcomes : list of str
            List of outcome names.
        metric : str
            Metric identifier being evaluated (e.g., 'auc', 'ici').
        strata : list of str
            List of subgroup strata names.
        scores : np.ndarray
            Baseline metric scores for unstratified models of shape (n_outcomes,).
        strata_scores : np.ndarray
            Metric scores per stratum and outcome of shape (n_strata, n_outcomes).
        save : bool, optional
            Automatically save the generated plot to the run directory.
        show : bool, optional
            Whether to display the plot interactively before closing.
        **style_kwargs : Any
            Additional style parameters forwarded to `draw_strata_heatmap`.

        Returns
        -------
        fig : Figure | SubFigure
            Rendered Matplotlib figure object.
        ax : Axes
            Matplotlib axes containing the heatmap plot.

        Raises
        ------
        ValueError
            If matrix dimensions do not match the provided strata, outcomes, or scores.

        """
        cfg = self._resolve_plot_config(
            plot_type="strata_heatmap",
            save=save,
            show=show,
            **style_kwargs,
        )
        save_val = cfg["save"]
        show_val = cfg["show"]

        scores_arr = np.asarray(scores)
        strata_scores_arr = np.asarray(strata_scores)

        # 1. Validation Logic
        if strata_scores_arr.ndim != 2:
            raise ValueError(
                f"strata_scores must be a 2D array, got shape {strata_scores_arr.shape}"
            )
        if len(strata) != strata_scores_arr.shape[0]:
            raise ValueError(
                f"Inputs strata and strata_scores must have matching row count, "
                f"got {len(strata)} and {strata_scores_arr.shape[0]}"
            )
        if len(outcomes) != strata_scores_arr.shape[1]:
            raise ValueError(
                f"Inputs outcomes and strata_scores must have matching column count, "
                f"got {len(outcomes)} and {strata_scores_arr.shape[1]}"
            )
        if len(scores_arr) != strata_scores_arr.shape[1]:
            raise ValueError(
                f"Inputs scores and strata_scores must have matching column count, "
                f"got {len(scores_arr)} and {strata_scores_arr.shape[1]}"
            )

        # 2. Data Preparation
        strata_matrix = np.vstack((scores_arr, strata_scores_arr))
        plot_data = np.abs(strata_matrix - scores_arr)
        text_data = strata_matrix.copy()

        is_ici = metric.lower() == "ici"
        vmax = 0.5 if is_ici else 0.1
        percent = " (%)" if is_ici else ""

        if is_ici:
            plot_data *= 100
            text_data *= 100

        colorbar_label = rf"|$\Delta$ {metric.upper()}|" + percent
        title = style_kwargs.pop("title", f"Strata Delta - {metric.upper()}{percent}")
        row_labels = ["All strata"] + list(strata)

        # 3. Stateless Drawing Delegate
        with (plt.rc_context(self.theme.to_rc_params()),):
            fig, ax = draw_strata_heatmap(
                plot_data=plot_data,
                text_data=text_data,
                row_labels=row_labels,
                col_labels=outcomes,
                colorbar_label=colorbar_label,
                vmax=vmax,
                title=title,
                **style_kwargs,
            )

        if save_val:
            self._save_figure(fig=fig, filename=f"{metric}_strata_heatmap")

        if show_val:
            plt.show()
        elif save_val:
            plt.close(fig)

        return fig, ax

    def plot_dca_curve(
        self,
        y_true: np.ndarray,
        probas: np.ndarray,
        outcome: str = "default",
        thresholds: Optional[np.ndarray] = None,
        label: Optional[str] = None,
        save: Optional[bool] = None,
        show: Optional[bool] = None,
        **style_kwargs: Any,
    ) -> Tuple[Figure | SubFigure, Axes]:
        """Compute Decision Curve Analysis metrics, render plot, and save figure artifact.

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth binary target labels of shape (n_samples,).
        probas : np.ndarray
            Predicted probabilities of shape (n_samples, 2) or (n_samples,).
        outcome : str, default="default"
            Outcome identifier used for figure titles and directory structuring.
        thresholds : np.ndarray, optional
            Array of threshold probabilities.
        label : str, optional
            Legend label for the model curve. Defaults to 'Model'.
        save : bool, optional
            Automatically save the generated plot to the run directory.
        show : bool, optional
            Whether to display the plot interactively before closing.
        **style_kwargs : Any
            Additional style parameters forwarded to `draw_dca_curve`.

        Returns
        -------
        fig : Figure | SubFigure
            Rendered Matplotlib figure object.
        ax : Axes
            Matplotlib axes containing the DCA plot.

        """
        cfg = self._resolve_plot_config(
            plot_type="dca",
            save=save,
            show=show,
            **style_kwargs,
        )
        save_val = cfg["save"]
        show_val = cfg["show"]

        self.logger.info(f"[{outcome}] Starting DCA graph plotting.")
        self.logger.debug(f"[{outcome}] Plotting DCA graph with y: {y_true.shape}.")
        thresh, nb_model, nb_all = self._compute_dca_data(
            y_true=y_true,
            probas=probas,
            thresholds=thresholds,
        )

        display_label = label or "Model"

        with (plt.rc_context(self.theme.to_rc_params()),):
            fig, ax = draw_dca_curve(
                thresholds=thresh,
                net_benefit_model=nb_model,
                net_benefit_all=nb_all,
                label=display_label,
                color=style_kwargs.pop("color", self.theme.primary_color),
                linewidth=style_kwargs.pop("linewidth", self.theme.linewidth),
                show_spines=style_kwargs.pop("show_spines", self.theme.show_spines),
                title=f"Decision Curve Analysis - {outcome.capitalize()}",
                **style_kwargs,
            )

        if save_val:
            self._save_figure(fig=fig, filename=f"{outcome}_dca_curve", outcome=outcome)

        if show_val:
            plt.show()
        elif save_val:
            plt.close(fig)

        return fig, ax

    def plot_all(
        self,
        y_true: np.ndarray,
        probas: np.ndarray,
        outcome: str = "default",
        n_bootstraps: Optional[int] = None,
        save: Optional[bool] = None,
        show: Optional[bool] = None,
        **style_kwargs: Any,
    ) -> Dict[str, Tuple[Figure | SubFigure, Axes]]:
        """Execute all core evaluation visualization routines for a given outcome.

        Generates and optionally persists the ROC curve, Precision-Recall curve,
        Probability Distribution histogram, Reliability Diagram, and Decision
        Curve Analysis (DCA).

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth binary target labels of shape (n_samples,).
        probas : np.ndarray
            Predicted probabilities of shape (n_samples, 2) or (n_samples,).
        outcome : str, default="default"
            Outcome identifier used for figure titles and output folder structuring.
        n_bootstraps : int, optional
            Number of bootstrap iterations for ROC, PR, and reliability curves.
        save : bool, optional
            Automatically save all generated plot artifacts to the run directory.
        show : bool, optional
            Whether to display figures interactively before closing.
        **style_kwargs : Any
            Additional style parameters forwarded to underlying drawing functions.

        Returns
        -------
        Dict[str, Tuple[Figure | SubFigure, Axes]]
            Dictionary mapping plot identifiers ('roc', 'pr', 'distribution',
            'reliability', 'dca') to their rendered (Figure, Axes) tuples.

        """
        plots: Dict[str, Tuple[Figure | SubFigure, Axes]] = {}

        self.logger.info(f"--- Starting graphical display for outcome: {outcome} ---")

        # Pass copies of style_kwargs to prevent parameter mutation across calls
        plots["roc"] = self.plot_roc_curve(
            y_true=y_true,
            probas=probas,
            outcome=outcome,
            n_bootstraps=n_bootstraps,
            save=save,
            show=show,
            **style_kwargs.copy(),
        )

        plots["pr"] = self.plot_precision_recall_curve(
            y_true=y_true,
            probas=probas,
            outcome=outcome,
            n_bootstraps=n_bootstraps,
            save=save,
            show=show,
            **style_kwargs.copy(),
        )

        plots["distribution"] = self.plot_probability_distribution(
            probas=probas,
            outcome=outcome,
            save=save,
            show=show,
            **style_kwargs.copy(),
        )

        plots["reliability"] = self.plot_reliability_diagram(
            y_true=y_true,
            probas=probas,
            outcome=outcome,
            n_bootstraps=n_bootstraps,
            save=save,
            show=show,
            **style_kwargs.copy(),
        )

        plots["dca"] = self.plot_dca_curve(
            y_true=y_true,
            probas=probas,
            outcome=outcome,
            save=save,
            show=show,
            **style_kwargs.copy(),
        )

        self.logger.info(f"--- Finished graphical display for outcome: {outcome} ---")
        return plots
