"""
High-level display and visualisation manager module.
"""

import ast
from pathlib import Path
from typing import Any, ClassVar

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure
from ordboost.distributions import ContinuousPredictiveDistribution
from ordboost.mappers import BaseBinMapper
from ordboost.metrics import (
    interval_coverage_rate,
    marginal_calibration_curve,
    pit_diagnostics,
    sharpness,
    winkler_score,
)
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)

from medpipe.metrics.registry import MetricRegistry
from medpipe.pipeline.orchestrator import MedpipeOrchestrator
from medpipe.utils.logger import get_console_logger
from medpipe.visualisation.plots import (
    draw_coverage_curve,
    draw_dca_curve,
    draw_marginal_calibration,
    draw_pit_histogram,
    draw_precision_recall_curve,
    draw_probability_distribution,
    draw_reliability_diagram,
    draw_roc_curve,
    draw_sharpness_curve,
    draw_strata_heatmap,
    draw_winkler_curve,
)
from medpipe.visualisation.themes import MedpipeTheme


class BaseDisplayer:
    """Shared visualisation engine providing config resolution, subgroup
    heatmap plotting, and artifact persistence, independent of outcome type.

    Config-resolution and heatmap plotting are outcome-type-agnostic: plot
    parameter precedence (defaults -> overrides -> outcome overrides ->
    runtime kwargs) and strata delta heatmaps operate purely on plot-type
    names, metric names, and evaluation dictionaries produced identically by
    both the classifier and regressor evaluators. Outcome-type-specific
    metric computation and high-level plotting methods (e.g. ROC/PR curves
    for classification, coverage/sharpness for regression) are provided by
    subclasses, which also override `_PLOT_TYPE_ALIASES` and
    `_FALLBACK_DISPLAY_DEFAULTS` for their own plot types.

    Parameters
    ----------
    orchestrator : MedpipeOrchestrator
        Active pipeline orchestrator instance containing execution context and
        run directory.
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
    plot_strata_heatmap(outcomes, metric, strata, scores, strata_scores,
    save=None, show=None, **style_kwargs)
        Validate subgroup inputs, compute delta matrix, render strata heatmap,
        and save output.
    plot_all_heatmaps(evaluations, metrics=None, save=None, show=None, **style_kwargs)
        Generate subgroup delta heatmaps across outcomes for each evaluated metric.

    """

    _PLOT_TYPE_ALIASES: ClassVar[dict[str, str]] = {}
    _FALLBACK_DISPLAY_DEFAULTS: ClassVar[dict[str, Any]] = {
        "n_bootstraps": 1000,
        "save": True,
        "show": False,
        "n_jobs": 1,
    }

    def __init__(
        self,
        orchestrator: MedpipeOrchestrator,
        theme: MedpipeTheme | None = None,
    ) -> None:
        self.orchestrator = orchestrator
        self.run_dir = orchestrator.run_dir
        self.theme = theme or MedpipeTheme()
        self.logger = get_console_logger("medpipe.displayer")

    # --- Config Resolution ---

    @classmethod
    def _normalize_plot_type(cls, plot_type: str) -> str:
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
        return cls._PLOT_TYPE_ALIASES.get(plot_type.lower(), plot_type.lower())

    def _resolve_plot_config(
        self,
        plot_type: str,
        outcome: str | None = None,
        **runtime_kwargs: Any,
    ) -> dict[str, Any]:
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
            config_params: dict[str, Any] = dict(self._FALLBACK_DISPLAY_DEFAULTS)
            overrides: dict[str, Any] = {}
            outcome_overrides: dict[str, Any] = {}
        else:
            config_params = display_cfg.defaults.model_dump()
            overrides = display_cfg.overrides
            outcome_overrides = display_cfg.outcome_overrides

        canonical_type = self._normalize_plot_type(plot_type)

        # 1. Apply global plot-type overrides (matching on any alias that
        # normalizes to the same canonical plot type as `plot_type`)
        for key, value in overrides.items():
            if self._normalize_plot_type(key) == canonical_type:
                config_params.update(value)

        # 2. Apply outcome-specific plot overrides
        if outcome and outcome in outcome_overrides:
            out_cfg = outcome_overrides[outcome]
            for key, value in out_cfg.items():
                if self._normalize_plot_type(key) == canonical_type:
                    config_params.update(value)

        # 3. Apply explicit non-None runtime kwargs overrides
        for k, v in runtime_kwargs.items():
            if v is not None:
                config_params[k] = v

        return config_params

    # --- Internal Helpers ---
    @staticmethod
    def _format_stratum_label(stratum_var: str, cat_key: Any) -> str:
        """Format raw stratum variable names and category keys into clean
        display labels.

        Parses stringified numerical bounds (e.g., '[18, 50]' or '[51, 120]') into
        readable ranges (e.g., '18-50' or open-ended '≥ 51'). Non-interval keys
        (e.g., 'F', 'M') are returned with standard variable prefixing.

        Parameters
        ----------
        stratum_var : str
            Name of the subgroup stratum variable (e.g., 'AGE', 'SEX').
        cat_key : Any
            Category identifier or raw bin string (e.g., '[18, 50]', 'F').

        Returns
        -------
        str
            Formatted stratum label suitable for heatmap visual displays.

        Examples
        --------
        >>> BaseDisplayer._format_stratum_label("AGE", "[18, 50]")
        'AGE: 18-50'
        >>> BaseDisplayer._format_stratum_label("AGE", "[51, 120]")
        'AGE: ≥ 51'
        >>> BaseDisplayer._format_stratum_label("SEX", "F")
        'SEX: F'

        """
        cat_str = str(cat_key).strip()

        if cat_str.startswith("[") and cat_str.endswith("]"):
            try:
                parsed = ast.literal_eval(cat_str)
                if isinstance(parsed, (list, tuple)) and len(parsed) == 2:
                    low, high = parsed[0], parsed[1]

                    if isinstance(high, (int, float)) and high >= 100:
                        return f"{stratum_var}: ≥ {low}"

                    return f"{stratum_var}: {low}-{high}"
            except (ValueError, SyntaxError):
                pass

        return f"{stratum_var}: {cat_str}"

    def _save_figure(
        self, fig: Figure | SubFigure, filename: str, outcome: str | None = None
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

    # --- Subgroup Heatmap Plotting ---

    def plot_strata_heatmap(
        self,
        outcomes: list[str],
        metric: str,
        strata: list[str],
        scores: np.ndarray,
        strata_scores: np.ndarray,
        save: bool | None = None,
        show: bool | None = None,
        **style_kwargs: Any,
    ) -> tuple[Figure | SubFigure, Axes]:
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

        try:
            spec = MetricRegistry.get(metric)
            display_name = (
                getattr(spec, "display_name", None) or metric.replace("_", " ").upper()
            )
        except Exception:
            display_name = metric.replace("_", " ").upper()

        is_ici = metric.lower() == "ici"
        vmax = 0.5 if is_ici else 0.1
        percent = " (%)" if is_ici else ""

        if is_ici:
            plot_data *= 100
            text_data *= 100

        colorbar_label = rf"|$\Delta$ {display_name}|" + percent
        title = style_kwargs.pop("title", f"Strata delta - {display_name}{percent}")
        row_labels = ["All strata", *list(strata)]

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

    def plot_all_heatmaps(
        self,
        evaluations: dict[str, Any],
        metrics: list[str] | None = None,
        save: bool | None = None,
        show: bool | None = None,
        **style_kwargs: Any,
    ) -> dict[str, tuple[Figure | SubFigure, Axes]]:
        """Generate subgroup delta heatmaps across outcomes for each evaluated metric.

        Parameters
        ----------
        evaluations : dict of str to Any
            Nested evaluations dictionary mapping outcome keys to their overall and
            subgroup performance evaluation results.
        metrics : list of str, optional
            List of metric names to render heatmaps for (e.g., ['auc', 'ici']).
            If None, inferred from the first outcome's overall metrics.
        save : bool, optional
            Automatically save all generated heatmap figures to disk.
        show : bool, optional
            Whether to display figures interactively before closing.
        **style_kwargs : Any
            Additional style parameters forwarded to `plot_strata_heatmap`.

        Returns
        -------
        heatmap_plots : dict of str to (Figure, Axes)
            Dictionary mapping metric names to their rendered (Figure, Axes) tuples.

        """
        outcomes = list(evaluations.keys())
        if not outcomes:
            return {}

        first_eval = evaluations[outcomes[0]]
        overall_dict = first_eval.get("overall", {})
        strata_dict = first_eval.get("strata", {})

        target_metrics = metrics or list(overall_dict.keys())

        # Flatten nested strata dict structure: stratum_variable -> category -> metric
        # Example row labels: "SEX: F", "SEX: M", "AGE: [18, 50]"
        strata_tuples: list[tuple[str, str]] = []
        strata_row_labels: list[str] = []

        for stratum_var, cat_dict in strata_dict.items():
            for cat_key in cat_dict:
                strata_tuples.append((stratum_var, cat_key))
                # Use helper to transform raw cat_key into clean label
                strata_row_labels.append(
                    self._format_stratum_label(stratum_var, cat_key)
                )

        if not strata_tuples:
            self.logger.warning(
                "No subgroup strata found in evaluation dict; skipping heatmaps."
            )
            return {}

        heatmap_plots: dict[str, tuple[Figure | SubFigure, Axes]] = {}

        for metric in target_metrics:
            # Look up MetricSpec display_name
            try:
                spec = MetricRegistry.get(metric)
                disp_name = (
                    getattr(spec, "display_name", None)
                    or metric.replace("_", " ").upper()
                )
            except Exception:
                disp_name = metric.replace("_", " ").upper()

            self.logger.info(
                f"--- Starting heatmap plotting for metric: {disp_name} ---"
            )
            # Extract unstratified baseline point estimates across outcomes
            scores_list = []
            for out in outcomes:
                entry = evaluations[out].get("overall", {}).get(metric, {})
                val = (
                    entry.get("point_estimate", np.nan)
                    if isinstance(entry, dict)
                    else entry
                )
                scores_list.append(val)
            scores = np.array(scores_list)

            # Extract stratum point estimates matrix (shape: n_strata_rows, n_outcomes)
            strata_scores_list = []
            for stratum_var, cat_key in strata_tuples:
                row = []
                for out in outcomes:
                    entry = (
                        evaluations[out]
                        .get("strata", {})
                        .get(stratum_var, {})
                        .get(cat_key, {})
                        .get(metric, {})
                    )
                    val = (
                        entry.get("point_estimate", np.nan)
                        if isinstance(entry, dict)
                        else entry
                    )
                    row.append(val)
                strata_scores_list.append(row)

            strata_scores = np.array(strata_scores_list)

            # Render heatmap for current metric
            heatmap_plots[metric] = self.plot_strata_heatmap(
                outcomes=outcomes,
                metric=metric,
                strata=strata_row_labels,
                scores=scores,
                strata_scores=strata_scores,
                save=save,
                show=show,
                **style_kwargs.copy(),
            )

            self.logger.info(
                f"--- Finished heatmap plotting for metric: {disp_name} ---"
            )
        return heatmap_plots


def _reliability_bootstrap_iteration(
    idx: np.ndarray,
    y_true: np.ndarray,
    probas: np.ndarray,
    prob_pred: np.ndarray,
    strategy: str,
    n_bins: int,
) -> np.ndarray | None:
    """Compute one bootstrap resample's calibration curve for
    `MedpipeClassifierDisplayer._compute_reliability_data`.

    Isolated to a module-level function so each resample's (potentially
    expensive, e.g. `SplineCalib`-fitting) computation can be dispatched to
    a `joblib.Parallel` worker independently of the others.

    Parameters
    ----------
    idx : numpy.ndarray
        Bootstrap resample indices (with replacement) into `y_true`/`probas`.
    y_true : numpy.ndarray
        Full ground truth binary target labels.
    probas : numpy.ndarray
        Full predicted probabilities (already reduced to 1D).
    prob_pred : numpy.ndarray
        Evaluation grid (spline) or original bin centers (uniform/quantile)
        the resampled curve is interpolated/evaluated onto.
    strategy : {'uniform', 'quantile', 'spline'}
        Calibration curve estimation strategy.
    n_bins : int
        Number of bins used for 'uniform' or 'quantile' binning strategies.

    Returns
    -------
    numpy.ndarray or None
        The resampled calibration curve evaluated at `prob_pred`, or `None`
        if the resample lacked class diversity or was too sparse to use.

    """
    if len(np.unique(y_true[idx])) < 2:
        return None

    if strategy == "spline":
        from splinecalib import SplineCalib

        sc_b = SplineCalib()  # type: ignore
        sc_b.fit(probas[idx], y_true[idx])
        b_true = sc_b.calibrate(prob_pred)

        assert b_true is not None
        if b_true.ndim == 2:
            b_true = b_true[:, 1]
        return b_true

    b_true, b_pred = calibration_curve(
        y_true[idx], probas[idx], n_bins=n_bins, strategy=strategy
    )
    if len(b_pred) > 1:
        return np.interp(prob_pred, b_pred, b_true)
    return None


class MedpipeClassifierDisplayer(BaseDisplayer):
    """High-level visualisation and display manager for MedpipeClassifier pipeline runs.

    This class handles statistical calculations (such as bootstrap confidence intervals
    and calibration metrics), applies theme aesthetics, delegates rendering to stateless
    drawing functions, and manages artifact persistence.

    Parameters
    ----------
    orchestrator : MedpipeOrchestrator
        Active pipeline orchestrator instance containing execution context and
        run directory.
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
    n_bootstraps=None, save=None, show=None, **style_kwargs)
        Compute ROC statistics, render curve with optional bootstrap CIs,
        and save output.
    plot_precision_recall_curve(y_true, probas, outcome="default",
    label=None, n_bootstraps=None, save=None, show=None, **style_kwargs)
        Compute PR statistics, render curve with optional bootstrap CIs,
        and save output.
    plot_probability_distribution(probas, outcome="default", n_bins=None,
    label=None, save=None, show=None, **style_kwargs)
        Render predicted probability distribution histogram and save output.
    plot_reliability_diagram(y_true, probas, outcome="default", n_bins=None,
    strategy=None, label=None, n_bootstraps=None, save=None,
    show=False, **style_kwargs)
        Compute calibration metrics (binned or spline), render reliability
        diagram with optional CIs, and save output.
    plot_strata_heatmap(outcomes, metric, strata, scores, strata_scores,
    save=None, show=None, **style_kwargs)
        Validate subgroup inputs, compute delta matrix, render strata heatmap,
        and save output.
    plot_dca_curve(y_true, probas, outcome="default", thresholds=None,
    label=None, save=None, show=None, **style_kwargs)
        Compute Net Benefit across decision thresholds, render DCA plot,
        and save output.
    plot_all_heatmaps(evaluations, metrics=None, save=None, show=None, **style_kwargs)
        Generate subgroup delta heatmaps across outcomes for each evaluated metric.
    plot_all(y_true, probas, outcome="default", n_bootstraps=None, save=None,
    show=None, **style_kwargs)
        Execute all core model evaluation visualization routines
        (ROC, PR, distribution, reliability, DCA) for a given outcome.
    """

    _PLOT_TYPE_ALIASES: ClassVar[dict[str, str]] = {
        "calibration": "reliability",
        "reliability_diagram": "reliability",
        "pr": "precision_recall",
        "pr_curve": "precision_recall",
        "roc_curve": "roc",
        "distribution": "probability_distribution",
        "dist": "probability_distribution",
        "dca_curve": "dca",
    }
    _FALLBACK_DISPLAY_DEFAULTS: ClassVar[dict[str, Any]] = {
        **BaseDisplayer._FALLBACK_DISPLAY_DEFAULTS,
        "n_bins": 10,
        "dist_n_bins": 10,
        "dist_yscale": "linear",
        "strategy": "uniform",
        "n_jobs": 1,
    }

    def _compute_roc_data(
        self,
        y_true: np.ndarray,
        probas: np.ndarray,
        n_bootstraps: int = 1000,
        random_state: int | None = 42,
    ) -> tuple[
        np.ndarray, np.ndarray, float, np.ndarray | None, np.ndarray | None
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
        random_state: int | None = 42,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        float,
        float,
        np.ndarray | None,
        np.ndarray | None,
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
            Lower 2.5% percentile bound of precision across bootstraps, if
            n_bootstraps > 0.
        upper_ci : np.ndarray or None
            Upper 97.5% percentile bound of precision across bootstraps, if
            n_bootstraps > 0.

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
        random_state: int | None = 42,
        n_jobs: int | None = 1,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray | None,
        np.ndarray | None,
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
        n_jobs : int or None, default=1
            Number of parallel jobs used to compute bootstrap resamples,
            each of which independently fits its own calibration curve
            (e.g. a fresh `SplineCalib` under `strategy='spline'`).

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
        # Draw all resample indices up front, sequentially, so the random
        # draw sequence (and thus reproducibility for a given random_state)
        # is unaffected by dispatching the actual curve-fitting work below
        # to parallel workers.
        bootstrap_indices = [
            rng.choice(n_samples, size=n_samples, replace=True)
            for _ in range(n_bootstraps)
        ]

        # Process-based (the joblib default): each resample's fit (e.g. a
        # SplineCalib fit via scipy's L-BFGS-B) spends its time in a
        # Python-level optimizer loop that holds the GIL, so a threading
        # backend adds contention instead of speedup here.
        boot_results = Parallel(n_jobs=n_jobs)(
            delayed(_reliability_bootstrap_iteration)(
                idx, y_true, probas, prob_pred, strategy, n_bins
            )
            for idx in bootstrap_indices
        )
        boots = [b for b in boot_results if b is not None]

        if not boots:
            return prob_true, prob_pred, None, None

        lower_ci = np.percentile(boots, 2.5, axis=0)
        upper_ci = np.percentile(boots, 97.5, axis=0)

        return prob_true, prob_pred, lower_ci, upper_ci

    def _compute_dca_data(
        self,
        y_true: np.ndarray,
        probas: np.ndarray,
        thresholds: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        n_bins: int | None = None,
        yscale: str | None = None,
        label: str | None = None,
        save: bool | None = None,
        show: bool | None = None,
        **style_kwargs: Any,
    ) -> tuple[Figure | SubFigure, Axes]:
        """Render prediction probability histogram and save figure artifact.

        Parameters
        ----------
        probas : np.ndarray
            Predicted probabilities of shape (n_samples, 2) or (n_samples,).
        outcome : str, default="default"
            Outcome identifier used for figure titles and directory structuring.
        n_bins : int, optional
            Number of equal-width bins for the histogram.
        yscale : str, optional
            Scale to use for the y-axis (e.g. linear, log, etc.)
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
            dist_n_bins=n_bins,
            dist_yscale=yscale,
            save=save,
            show=show,
            **style_kwargs,
        )

        n_bins_val = cfg["dist_n_bins"]
        save_val = cfg["save"]
        show_val = cfg["show"]
        dist_yscale_val = cfg["dist_yscale"]

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
                yscale=dist_yscale_val,
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
        label: str | None = None,
        n_bootstraps: int | None = None,
        save: bool | None = None,
        show: bool | None = None,
        **style_kwargs: Any,
    ) -> tuple[Figure | SubFigure, Axes]:
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
            Number of bootstrap iterations for confidence intervals. Set to 0
            to disable.
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
        label: str | None = None,
        n_bootstraps: int | None = None,
        save: bool | None = None,
        show: bool | None = None,
        **style_kwargs: Any,
    ) -> tuple[Figure | SubFigure, Axes]:
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
            Number of bootstrap iterations for confidence intervals. Set to 0
            to disable.
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
        n_bins: int | None = None,
        dist_n_bins: int | None = None,
        dist_yscale: str | None = None,
        strategy: str | None = None,
        label: str | None = None,
        n_bootstraps: int | None = None,
        save: bool | None = None,
        show: bool | None = None,
        **style_kwargs: Any,
    ) -> tuple[Figure | SubFigure, Axes]:
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
        n_bins : int, optional
            Number of calibration bins (ignored if strategy='spline').
        dist_n_bins : int, optional
            Number of bins for the distribution.
        dist_yscale : str, optional
            Scale to use for the y-axis (e.g. linear, log, etc.)
        strategy : {'uniform', 'quantile', 'spline'}, optional
            Binning or smoothing strategy for calibration calculation.
        label : str, optional
            Legend label for the model curve. Defaults to 'Model'.
        n_bootstraps : int, optional
            Number of bootstrap iterations for confidence intervals. Set to 0
            to disable.
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
            dist_n_bins=dist_n_bins,
            dist_yscale=dist_yscale,
            strategy=strategy,
            n_bootstraps=n_bootstraps,
            save=save,
            show=show,
            **style_kwargs,
        )

        n_bins_val = cfg["n_bins"]
        strategy_val = cfg["strategy"]
        n_bootstraps_val = cfg["n_bootstraps"]
        n_jobs_val = cfg["n_jobs"]
        dist_yscale_val = cfg["dist_yscale"]
        dist_n_bins_val = cfg["dist_n_bins"]
        save_val = cfg["save"]
        show_val = cfg["show"]

        self.logger.info(f"[{outcome}] Starting reliability diagram plotting.")
        self.logger.debug(
            f"[{outcome}] Plotting reliability diagram with "
            f"y: {y_true.shape}, {n_bootstraps_val} bootstrap iterations, "
            f"{n_bins_val} bins, {strategy_val} strategy, and "
            f"{n_jobs_val} parallel job(s)."
        )
        prob_true, prob_pred, lower_ci, upper_ci = self._compute_reliability_data(
            y_true=y_true,
            probas=probas,
            n_bins=n_bins_val,
            strategy=strategy_val,
            n_bootstraps=n_bootstraps_val,
            n_jobs=n_jobs_val,
        )

        display_label = label or "Model"

        # Suppress scatter points for smooth continuous spline curves
        if strategy_val == "spline":
            style_kwargs.setdefault("marker", None)

        with (plt.rc_context(self.theme.to_rc_params()),):
            fig, ax = draw_reliability_diagram(
                prob_true=prob_true,
                prob_pred=prob_pred,
                probas=probas,  # Pass raw probabilities for bottom histogram
                dist_yscale=dist_yscale_val,
                dist_n_bins=dist_n_bins_val,
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

    def plot_dca_curve(
        self,
        y_true: np.ndarray,
        probas: np.ndarray,
        outcome: str = "default",
        thresholds: np.ndarray | None = None,
        label: str | None = None,
        save: bool | None = None,
        show: bool | None = None,
        **style_kwargs: Any,
    ) -> tuple[Figure | SubFigure, Axes]:
        """Compute Decision Curve Analysis metrics, render plot, and save
        figure artifact.

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
        n_bootstraps: int | None = None,
        save: bool | None = None,
        show: bool | None = None,
        **style_kwargs: Any,
    ) -> dict[str, tuple[Figure | SubFigure, Axes]]:
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
        plots: dict[str, tuple[Figure | SubFigure, Axes]] = {}

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


class MedpipeRegressorDisplayer(BaseDisplayer):
    """High-level visualisation and display manager for MedpipeRegressor pipeline runs.

    Renders distributional diagnostic figures (coverage reliability, sharpness,
    Winkler score, marginal calibration, and PIT histograms) for regression
    models that produce a predictive CDF via `predict_dist`. Coverage,
    sharpness, and Winkler score are bootstrapped for confidence intervals,
    mirroring `MedpipeClassifierDisplayer`'s ROC/PR/reliability curves.

    Currently only OrdBoost's `ContinuousPredictiveDistribution` is
    supported (a `mapper` is required for PIT histograms, e.g. the fitted
    `OrdBoostRegressor`'s `mapper_` attribute); NGBoost-wrapped
    distributions are not yet supported and will raise a clear error if
    passed to `plot_pit_histogram`.

    Parameters
    ----------
    orchestrator : MedpipeOrchestrator
        Active pipeline orchestrator instance containing execution context and
        run directory.
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
    plot_coverage(y_true, dist, outcome="default", coverage_levels=None,
    n_bootstraps=None, label=None, save=None, show=None, **style_kwargs)
        Compute empirical coverage across nominal coverage levels with
        bootstrap CIs, render curve, and save output.
    plot_sharpness(y_true, dist, outcome="default", coverage_levels=None,
    n_bootstraps=None, label=None, save=None, show=None, **style_kwargs)
        Compute mean interval width across nominal coverage levels with
        bootstrap CIs, render curve, and save output.
    plot_winkler(y_true, dist, outcome="default", coverage_levels=None,
    n_bootstraps=None, label=None, save=None, show=None, **style_kwargs)
        Compute Winkler score across nominal coverage levels with bootstrap
        CIs, render curve, and save output.
    plot_marginal_calibration(y_true, dist, outcome="default", label=None,
    save=None, show=None, **style_kwargs)
        Compute marginal calibration curve, render plot, and save output.
    plot_pit_histogram(y_true, dist, mapper, outcome="default", n_bins=None,
    label=None, save=None, show=None, **style_kwargs)
        Compute PIT diagnostics, render histogram, and save output.
    plot_strata_heatmap(outcomes, metric, strata, scores, strata_scores,
    save=None, show=None, **style_kwargs)
        Validate subgroup inputs, compute delta matrix, render strata heatmap,
        and save output.
    plot_all_heatmaps(evaluations, metrics=None, save=None, show=None, **style_kwargs)
        Generate subgroup delta heatmaps across outcomes for each evaluated metric.
    plot_all(y_true, dist, mapper, outcome="default", coverage_levels=None,
    n_bootstraps=None, save=None, show=None, **style_kwargs)
        Execute all core distributional diagnostic visualization routines
        for a given outcome.

    """

    _PLOT_TYPE_ALIASES: ClassVar[dict[str, str]] = {
        "coverage_curve": "coverage",
        "sharpness_curve": "sharpness",
        "winkler_curve": "winkler",
        "winkler_score": "winkler",
        "marginal_calib": "marginal_calibration",
        "pit": "pit_histogram",
    }
    # Hardcoded defaults (not yet exposed via DisplayDefaultsConfig): nominal
    # coverage grid (percent) for coverage/sharpness/Winkler, and PIT
    # histogram bin count.
    _DEFAULT_COVERAGE_LEVELS: ClassVar[np.ndarray] = np.arange(10, 100, 10)
    _DEFAULT_PIT_N_BINS: ClassVar[int] = 20

    def _compute_coverage_sharpness_winkler_data(
        self,
        y_true: np.ndarray,
        dist: ContinuousPredictiveDistribution,
        coverage_levels: np.ndarray,
        n_bootstraps: int = 1000,
        random_state: int | None = 42,
    ) -> dict[str, Any]:
        """Compute empirical coverage, sharpness, and Winkler score across
        nominal coverage levels, with bootstrap CIs.

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth continuous target values of shape (n_samples,).
        dist : ContinuousPredictiveDistribution
            Predictive distribution for the same samples.
        coverage_levels : np.ndarray
            Nominal central-interval coverage levels in percent (e.g. 90 for
            a 90% interval).
        n_bootstraps : int, default=1000
            Number of bootstrap iterations for 95% confidence interval estimation.
        random_state : int, optional, default=42
            Random seed for bootstrap resampling reproducibility.

        Returns
        -------
        dict of str to Any
            `"coverage_levels"` (np.ndarray), and `"coverage"`, `"sharpness"`,
            `"winkler"`, each a dict with `"point"`, `"lower_ci"`, `"upper_ci"`
            (the latter two None when `n_bootstraps <= 0`).

        """
        y_true_arr = np.asarray(y_true, dtype=float)
        coverage_levels_arr = np.asarray(coverage_levels, dtype=float)
        alphas = 1.0 - coverage_levels_arr / 100.0

        empirical_coverage = np.empty_like(alphas)
        sharp = np.empty_like(alphas)
        winkler = np.empty_like(alphas)

        for i, alpha in enumerate(alphas):
            empirical_coverage[i] = (
                interval_coverage_rate(y_true_arr, dist, alpha=alpha) * 100.0
            )
            sharp[i] = sharpness(dist, alpha=alpha)
            winkler[i] = winkler_score(y_true_arr, dist, alpha=alpha)

        result: dict[str, Any] = {
            "coverage_levels": coverage_levels_arr,
            "coverage": {
                "point": empirical_coverage,
                "lower_ci": None,
                "upper_ci": None,
            },
            "sharpness": {"point": sharp, "lower_ci": None, "upper_ci": None},
            "winkler": {"point": winkler, "lower_ci": None, "upper_ci": None},
        }

        if n_bootstraps <= 0:
            return result

        rng = np.random.default_rng(random_state)
        n_samples = len(y_true_arr)
        n_levels = len(alphas)
        coverage_boot = np.empty((n_bootstraps, n_levels))
        sharp_boot = np.empty((n_bootstraps, n_levels))
        winkler_boot = np.empty((n_bootstraps, n_levels))

        for b in range(n_bootstraps):
            idx = rng.choice(n_samples, size=n_samples, replace=True)
            y_b = y_true_arr[idx]
            dist_b = ContinuousPredictiveDistribution(
                grid_y=dist.grid_y, grid_cdf=dist.grid_cdf[idx]
            )
            for j, alpha in enumerate(alphas):
                coverage_boot[b, j] = (
                    interval_coverage_rate(y_b, dist_b, alpha=alpha) * 100.0
                )
                sharp_boot[b, j] = sharpness(dist_b, alpha=alpha)
                winkler_boot[b, j] = winkler_score(y_b, dist_b, alpha=alpha)

        result["coverage"]["lower_ci"] = np.percentile(coverage_boot, 2.5, axis=0)
        result["coverage"]["upper_ci"] = np.percentile(coverage_boot, 97.5, axis=0)
        result["sharpness"]["lower_ci"] = np.percentile(sharp_boot, 2.5, axis=0)
        result["sharpness"]["upper_ci"] = np.percentile(sharp_boot, 97.5, axis=0)
        result["winkler"]["lower_ci"] = np.percentile(winkler_boot, 2.5, axis=0)
        result["winkler"]["upper_ci"] = np.percentile(winkler_boot, 97.5, axis=0)

        return result

    def _compute_marginal_calibration_data(
        self,
        y_true: np.ndarray,
        dist: ContinuousPredictiveDistribution,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute the marginal calibration curve for a predictive distribution.

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth continuous target values of shape (n_samples,).
        dist : ContinuousPredictiveDistribution
            Predictive distribution for the same samples.

        Returns
        -------
        grid_y : np.ndarray
            Target-value grid points.
        diff : np.ndarray
            Empirical CDF minus mean predicted CDF at each grid point.

        """
        y_true_arr = np.asarray(y_true, dtype=float)
        grid_y, diff = marginal_calibration_curve(y_true_arr, dist)
        return grid_y, diff

    def _compute_pit_histogram_data(
        self,
        y_true: np.ndarray,
        dist: ContinuousPredictiveDistribution,
        mapper: BaseBinMapper | None,
        n_bins: int = 20,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Compute PIT (Probability Integral Transform) histogram diagnostics.

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth continuous target values of shape (n_samples,).
        dist : ContinuousPredictiveDistribution
            Predictive distribution for the same samples.
        mapper : BaseBinMapper or None
            The fitted OrdBoost model's bin mapper (e.g. its `mapper_`
            attribute), required to compute exact PIT diagnostics.
        n_bins : int, default=20
            Number of PIT histogram bins.

        Returns
        -------
        bin_centres : np.ndarray
            Bin centre positions in [0, 1].
        hist_values : np.ndarray
            Density values for each bin.
        alpha_score : float
            PIT alpha (uniformity) score.

        Raises
        ------
        ValueError
            If `mapper` is None, since NGBoost-wrapped distributions (which
            have no OrdBoost bin mapper) are not yet supported.

        """
        if mapper is None:
            raise ValueError(
                "A `mapper` (e.g. the fitted OrdBoostRegressor's `mapper_` "
                "attribute) is required to compute a PIT histogram. "
                "Distributions without an OrdBoost bin mapper (e.g. "
                "NGBoost-wrapped distributions) are not yet supported."
            )

        y_true_arr = np.asarray(y_true, dtype=float)
        pit = pit_diagnostics(y_true_arr, dist, mapper)
        alpha_score = float(np.asarray(pit.alpha_score()))
        hist = pit.hist_values(n_bins)
        bin_centres = np.asarray(hist["bin_centre"].values)
        hist_values = np.asarray(hist.values)

        return bin_centres, hist_values, alpha_score

    # --- High-Level Plotting Methods ---

    def plot_coverage(
        self,
        y_true: np.ndarray,
        dist: ContinuousPredictiveDistribution,
        outcome: str = "default",
        coverage_levels: np.ndarray | None = None,
        n_bootstraps: int | None = None,
        label: str | None = None,
        save: bool | None = None,
        show: bool | None = None,
        **style_kwargs: Any,
    ) -> tuple[Figure | SubFigure, Axes]:
        """Compute coverage reliability, render curve with confidence
        intervals, and save figure.

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth continuous target values of shape (n_samples,).
        dist : ContinuousPredictiveDistribution
            Predictive distribution for the same samples.
        outcome : str, default="default"
            Outcome identifier used for figure titles and folder structuring.
        coverage_levels : np.ndarray, optional
            Nominal central-interval coverage levels in percent. Defaults to
            `np.arange(10, 100, 10)`.
        n_bootstraps : int, optional
            Number of bootstrap iterations for confidence intervals. Set to 0
            to disable.
        label : str, optional
            Legend label for the model. Defaults to 'Model'.
        save : bool, optional
            Automatically save the generated plot to the run directory.
        show : bool, optional
            Whether to display the plot interactively before closing.
        **style_kwargs : Any
            Additional style parameters forwarded to `draw_coverage_curve`.

        Returns
        -------
        fig : Figure | SubFigure
            Rendered Matplotlib figure object.
        ax : Axes
            Matplotlib axes containing the plotted elements.

        """
        cfg = self._resolve_plot_config(
            plot_type="coverage",
            outcome=outcome,
            n_bootstraps=n_bootstraps,
            save=save,
            show=show,
            **style_kwargs,
        )
        n_bootstraps_val = cfg["n_bootstraps"]
        save_val = cfg["save"]
        show_val = cfg["show"]
        levels = (
            coverage_levels
            if coverage_levels is not None
            else self._DEFAULT_COVERAGE_LEVELS
        )

        self.logger.info(f"[{outcome}] Starting coverage curve plotting.")
        data = self._compute_coverage_sharpness_winkler_data(
            y_true=y_true,
            dist=dist,
            coverage_levels=levels,
            n_bootstraps=n_bootstraps_val,
        )

        display_label = label or "Model"

        with (plt.rc_context(self.theme.to_rc_params()),):
            fig, ax = draw_coverage_curve(
                coverage_levels=data["coverage_levels"],
                empirical_coverage=data["coverage"]["point"],
                lower_ci=data["coverage"]["lower_ci"],
                upper_ci=data["coverage"]["upper_ci"],
                label=display_label,
                color=style_kwargs.pop("color", self.theme.primary_color),
                ci_alpha=style_kwargs.pop("ci_alpha", self.theme.ci_alpha),
                linewidth=style_kwargs.pop("linewidth", self.theme.linewidth),
                show_spines=style_kwargs.pop("show_spines", self.theme.show_spines),
                title=f"Coverage Reliability - {outcome.capitalize()}",
                **style_kwargs,
            )

        if save_val:
            self._save_figure(fig=fig, filename=f"{outcome}_coverage", outcome=outcome)

        if show_val:
            plt.show()
        elif save_val:
            plt.close(fig)

        return fig, ax

    def plot_sharpness(
        self,
        y_true: np.ndarray,
        dist: ContinuousPredictiveDistribution,
        outcome: str = "default",
        coverage_levels: np.ndarray | None = None,
        n_bootstraps: int | None = None,
        label: str | None = None,
        save: bool | None = None,
        show: bool | None = None,
        **style_kwargs: Any,
    ) -> tuple[Figure | SubFigure, Axes]:
        """Compute sharpness, render curve with confidence intervals, and
        save figure.

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth continuous target values of shape (n_samples,).
        dist : ContinuousPredictiveDistribution
            Predictive distribution for the same samples.
        outcome : str, default="default"
            Outcome identifier used for figure titles and folder structuring.
        coverage_levels : np.ndarray, optional
            Nominal central-interval coverage levels in percent. Defaults to
            `np.arange(10, 100, 10)`.
        n_bootstraps : int, optional
            Number of bootstrap iterations for confidence intervals. Set to 0
            to disable.
        label : str, optional
            Legend label for the model. Defaults to 'Model'.
        save : bool, optional
            Automatically save the generated plot to the run directory.
        show : bool, optional
            Whether to display the plot interactively before closing.
        **style_kwargs : Any
            Additional style parameters forwarded to `draw_sharpness_curve`.

        Returns
        -------
        fig : Figure | SubFigure
            Rendered Matplotlib figure object.
        ax : Axes
            Matplotlib axes containing the plotted elements.

        """
        cfg = self._resolve_plot_config(
            plot_type="sharpness",
            outcome=outcome,
            n_bootstraps=n_bootstraps,
            save=save,
            show=show,
            **style_kwargs,
        )
        n_bootstraps_val = cfg["n_bootstraps"]
        save_val = cfg["save"]
        show_val = cfg["show"]
        levels = (
            coverage_levels
            if coverage_levels is not None
            else self._DEFAULT_COVERAGE_LEVELS
        )

        self.logger.info(f"[{outcome}] Starting sharpness curve plotting.")
        data = self._compute_coverage_sharpness_winkler_data(
            y_true=y_true,
            dist=dist,
            coverage_levels=levels,
            n_bootstraps=n_bootstraps_val,
        )

        display_label = label or "Model"

        with (plt.rc_context(self.theme.to_rc_params()),):
            fig, ax = draw_sharpness_curve(
                coverage_levels=data["coverage_levels"],
                sharpness_values=data["sharpness"]["point"],
                lower_ci=data["sharpness"]["lower_ci"],
                upper_ci=data["sharpness"]["upper_ci"],
                label=display_label,
                color=style_kwargs.pop("color", self.theme.primary_color),
                ci_alpha=style_kwargs.pop("ci_alpha", self.theme.ci_alpha),
                linewidth=style_kwargs.pop("linewidth", self.theme.linewidth),
                show_spines=style_kwargs.pop("show_spines", self.theme.show_spines),
                title=f"Sharpness - {outcome.capitalize()}",
                **style_kwargs,
            )

        if save_val:
            self._save_figure(fig=fig, filename=f"{outcome}_sharpness", outcome=outcome)

        if show_val:
            plt.show()
        elif save_val:
            plt.close(fig)

        return fig, ax

    def plot_winkler(
        self,
        y_true: np.ndarray,
        dist: ContinuousPredictiveDistribution,
        outcome: str = "default",
        coverage_levels: np.ndarray | None = None,
        n_bootstraps: int | None = None,
        label: str | None = None,
        save: bool | None = None,
        show: bool | None = None,
        **style_kwargs: Any,
    ) -> tuple[Figure | SubFigure, Axes]:
        """Compute Winkler score, render curve with confidence intervals,
        and save figure.

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth continuous target values of shape (n_samples,).
        dist : ContinuousPredictiveDistribution
            Predictive distribution for the same samples.
        outcome : str, default="default"
            Outcome identifier used for figure titles and folder structuring.
        coverage_levels : np.ndarray, optional
            Nominal central-interval coverage levels in percent. Defaults to
            `np.arange(10, 100, 10)`.
        n_bootstraps : int, optional
            Number of bootstrap iterations for confidence intervals. Set to 0
            to disable.
        label : str, optional
            Legend label for the model. Defaults to 'Model'.
        save : bool, optional
            Automatically save the generated plot to the run directory.
        show : bool, optional
            Whether to display the plot interactively before closing.
        **style_kwargs : Any
            Additional style parameters forwarded to `draw_winkler_curve`.

        Returns
        -------
        fig : Figure | SubFigure
            Rendered Matplotlib figure object.
        ax : Axes
            Matplotlib axes containing the plotted elements.

        """
        cfg = self._resolve_plot_config(
            plot_type="winkler",
            outcome=outcome,
            n_bootstraps=n_bootstraps,
            save=save,
            show=show,
            **style_kwargs,
        )
        n_bootstraps_val = cfg["n_bootstraps"]
        save_val = cfg["save"]
        show_val = cfg["show"]
        levels = (
            coverage_levels
            if coverage_levels is not None
            else self._DEFAULT_COVERAGE_LEVELS
        )

        self.logger.info(f"[{outcome}] Starting Winkler score curve plotting.")
        data = self._compute_coverage_sharpness_winkler_data(
            y_true=y_true,
            dist=dist,
            coverage_levels=levels,
            n_bootstraps=n_bootstraps_val,
        )

        display_label = label or "Model"

        with (plt.rc_context(self.theme.to_rc_params()),):
            fig, ax = draw_winkler_curve(
                coverage_levels=data["coverage_levels"],
                winkler_values=data["winkler"]["point"],
                lower_ci=data["winkler"]["lower_ci"],
                upper_ci=data["winkler"]["upper_ci"],
                label=display_label,
                color=style_kwargs.pop("color", self.theme.primary_color),
                ci_alpha=style_kwargs.pop("ci_alpha", self.theme.ci_alpha),
                linewidth=style_kwargs.pop("linewidth", self.theme.linewidth),
                show_spines=style_kwargs.pop("show_spines", self.theme.show_spines),
                title=f"Winkler Score - {outcome.capitalize()}",
                **style_kwargs,
            )

        if save_val:
            self._save_figure(fig=fig, filename=f"{outcome}_winkler", outcome=outcome)

        if show_val:
            plt.show()
        elif save_val:
            plt.close(fig)

        return fig, ax

    def plot_marginal_calibration(
        self,
        y_true: np.ndarray,
        dist: ContinuousPredictiveDistribution,
        outcome: str = "default",
        label: str | None = None,
        save: bool | None = None,
        show: bool | None = None,
        **style_kwargs: Any,
    ) -> tuple[Figure | SubFigure, Axes]:
        """Compute the marginal calibration curve, render plot, and save figure.

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth continuous target values of shape (n_samples,).
        dist : ContinuousPredictiveDistribution
            Predictive distribution for the same samples.
        outcome : str, default="default"
            Outcome identifier used for figure titles and folder structuring.
        label : str, optional
            Legend label for the model. Defaults to 'Model'.
        save : bool, optional
            Automatically save the generated plot to the run directory.
        show : bool, optional
            Whether to display the plot interactively before closing.
        **style_kwargs : Any
            Additional style parameters forwarded to `draw_marginal_calibration`.

        Returns
        -------
        fig : Figure | SubFigure
            Rendered Matplotlib figure object.
        ax : Axes
            Matplotlib axes containing the plotted elements.

        """
        cfg = self._resolve_plot_config(
            plot_type="marginal_calibration",
            outcome=outcome,
            save=save,
            show=show,
            **style_kwargs,
        )
        save_val = cfg["save"]
        show_val = cfg["show"]

        self.logger.info(f"[{outcome}] Starting marginal calibration plotting.")
        grid_y, diff = self._compute_marginal_calibration_data(y_true=y_true, dist=dist)

        display_label = label or "Model"

        with (plt.rc_context(self.theme.to_rc_params()),):
            fig, ax = draw_marginal_calibration(
                grid_y=grid_y,
                diff=diff,
                label=display_label,
                color=style_kwargs.pop("color", self.theme.primary_color),
                linewidth=style_kwargs.pop("linewidth", self.theme.linewidth),
                show_spines=style_kwargs.pop("show_spines", self.theme.show_spines),
                title=f"Marginal Calibration - {outcome.capitalize()}",
                **style_kwargs,
            )

        if save_val:
            self._save_figure(
                fig=fig, filename=f"{outcome}_marginal_calibration", outcome=outcome
            )

        if show_val:
            plt.show()
        elif save_val:
            plt.close(fig)

        return fig, ax

    def plot_pit_histogram(
        self,
        y_true: np.ndarray,
        dist: ContinuousPredictiveDistribution,
        mapper: BaseBinMapper | None = None,
        outcome: str = "default",
        n_bins: int | None = None,
        label: str | None = None,
        save: bool | None = None,
        show: bool | None = None,
        **style_kwargs: Any,
    ) -> tuple[Figure | SubFigure, Axes]:
        """Compute PIT diagnostics, render histogram, and save figure.

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth continuous target values of shape (n_samples,).
        dist : ContinuousPredictiveDistribution
            Predictive distribution for the same samples.
        mapper : BaseBinMapper, optional
            The fitted OrdBoost model's bin mapper (e.g. its `mapper_`
            attribute). Required; NGBoost-wrapped distributions are not yet
            supported and will raise a clear error.
        outcome : str, default="default"
            Outcome identifier used for figure titles and folder structuring.
        n_bins : int, optional
            Number of PIT histogram bins. Defaults to 20.
        label : str, optional
            Legend label prefix for the histogram. Defaults to 'Model'.
        save : bool, optional
            Automatically save the generated plot to the run directory.
        show : bool, optional
            Whether to display the plot interactively before closing.
        **style_kwargs : Any
            Additional style parameters forwarded to `draw_pit_histogram`.

        Returns
        -------
        fig : Figure | SubFigure
            Rendered Matplotlib figure object.
        ax : Axes
            Matplotlib axes containing the plotted elements.

        Raises
        ------
        ValueError
            If `mapper` is None.

        """
        cfg = self._resolve_plot_config(
            plot_type="pit_histogram",
            outcome=outcome,
            save=save,
            show=show,
            **style_kwargs,
        )
        # Hardcoded for now rather than routed through `_resolve_plot_config`,
        # since DisplayDefaultsConfig.n_bins is shared with the classifier's
        # calibration-bin default (10); revisit once regressor-specific
        # display defaults are added to the config schema.
        n_bins_val = n_bins if n_bins is not None else self._DEFAULT_PIT_N_BINS
        save_val = cfg["save"]
        show_val = cfg["show"]

        self.logger.info(f"[{outcome}] Starting PIT histogram plotting.")
        bin_centres, hist_values, alpha_score = self._compute_pit_histogram_data(
            y_true=y_true, dist=dist, mapper=mapper, n_bins=n_bins_val
        )

        display_label = label or "Model"

        with (plt.rc_context(self.theme.to_rc_params()),):
            fig, ax = draw_pit_histogram(
                bin_centres=bin_centres,
                hist_values=hist_values,
                alpha_score=alpha_score,
                n_bins=n_bins_val,
                label=display_label,
                color=style_kwargs.pop("color", self.theme.primary_color),
                show_spines=style_kwargs.pop("show_spines", self.theme.show_spines),
                title=f"PIT Histogram - {outcome.capitalize()}",
                **style_kwargs,
            )

        if save_val:
            self._save_figure(
                fig=fig, filename=f"{outcome}_pit_histogram", outcome=outcome
            )

        if show_val:
            plt.show()
        elif save_val:
            plt.close(fig)

        return fig, ax

    def plot_all(
        self,
        y_true: np.ndarray,
        dist: ContinuousPredictiveDistribution,
        mapper: BaseBinMapper | None = None,
        outcome: str = "default",
        coverage_levels: np.ndarray | None = None,
        n_bootstraps: int | None = None,
        save: bool | None = None,
        show: bool | None = None,
        **style_kwargs: Any,
    ) -> dict[str, tuple[Figure | SubFigure, Axes]]:
        """Execute all core distributional diagnostic visualization routines
        for a given outcome.

        Generates and optionally persists the coverage reliability curve,
        sharpness curve, Winkler score curve, marginal calibration curve,
        and PIT histogram.

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth continuous target values of shape (n_samples,).
        dist : ContinuousPredictiveDistribution
            Predictive distribution for the same samples.
        mapper : BaseBinMapper, optional
            The fitted OrdBoost model's bin mapper, required for the PIT
            histogram (e.g. its `mapper_` attribute).
        outcome : str, default="default"
            Outcome identifier used for figure titles and output folder structuring.
        coverage_levels : np.ndarray, optional
            Nominal central-interval coverage levels in percent, used for the
            coverage, sharpness, and Winkler score curves.
        n_bootstraps : int, optional
            Number of bootstrap iterations for coverage, sharpness, and
            Winkler score curves.
        save : bool, optional
            Automatically save all generated plot artifacts to the run directory.
        show : bool, optional
            Whether to display figures interactively before closing.
        **style_kwargs : Any
            Additional style parameters forwarded to underlying drawing functions.

        Returns
        -------
        Dict[str, Tuple[Figure | SubFigure, Axes]]
            Dictionary mapping plot identifiers ('coverage', 'sharpness',
            'winkler', 'marginal_calibration', 'pit_histogram') to their
            rendered (Figure, Axes) tuples.

        """
        plots: dict[str, tuple[Figure | SubFigure, Axes]] = {}

        self.logger.info(f"--- Starting graphical display for outcome: {outcome} ---")

        plots["coverage"] = self.plot_coverage(
            y_true=y_true,
            dist=dist,
            outcome=outcome,
            coverage_levels=coverage_levels,
            n_bootstraps=n_bootstraps,
            save=save,
            show=show,
            **style_kwargs.copy(),
        )

        plots["sharpness"] = self.plot_sharpness(
            y_true=y_true,
            dist=dist,
            outcome=outcome,
            coverage_levels=coverage_levels,
            n_bootstraps=n_bootstraps,
            save=save,
            show=show,
            **style_kwargs.copy(),
        )

        plots["winkler"] = self.plot_winkler(
            y_true=y_true,
            dist=dist,
            outcome=outcome,
            coverage_levels=coverage_levels,
            n_bootstraps=n_bootstraps,
            save=save,
            show=show,
            **style_kwargs.copy(),
        )

        plots["marginal_calibration"] = self.plot_marginal_calibration(
            y_true=y_true,
            dist=dist,
            outcome=outcome,
            save=save,
            show=show,
            **style_kwargs.copy(),
        )

        plots["pit_histogram"] = self.plot_pit_histogram(
            y_true=y_true,
            dist=dist,
            mapper=mapper,
            outcome=outcome,
            save=save,
            show=show,
            **style_kwargs.copy(),
        )

        self.logger.info(f"--- Finished graphical display for outcome: {outcome} ---")
        return plots
