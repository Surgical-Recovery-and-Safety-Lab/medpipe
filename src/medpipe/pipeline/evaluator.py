"""
Evaluator module for Medpipe machine learning models and pipelines.

Provides TRIPOD+AI compliant model evaluation, subgroup performance analysis,
bootstrap confidence interval estimation, logging, and artifact management.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

import numpy as np
import numpy.typing as npt
import pandas as pd

from medpipe.data.utils import resolve_subgroup_mask
from medpipe.metrics.core import bootstrap_confidence_intervals, compute_metrics
from medpipe.utils.logger import get_console_logger

if TYPE_CHECKING:
    from medpipe.pipeline.orchestrator import MedpipeOrchestrator
    from medpipe.pipeline.runner import MedpipeRunner


class MedpipeEvaluator:
    """
    Evaluation engine for Medpipe machine learning models and pipelines.

    Provides standard inference interfaces (`predict`, `predict_proba`, `decision_function`)
    and structured performance evaluation across full datasets and extracted data subgroups.
    In compliance with TRIPOD+AI reporting guidelines, evaluation metrics include bootstrap
    confidence intervals. Results are automatically logged and saved to disk using the
    orchestrator's `ArtifactManager`.

    Parameters
    ----------
    orchestrator : MedpipeOrchestrator
        The pipeline orchestrator instance containing workflow configuration,
        run directories, and the `ArtifactManager`.
    runner : MedpipeRunner
        The pipeline runner instance containing the dictionary of fitted models
        (`fitted_models`).

    Attributes
    ----------
    orchestrator : MedpipeOrchestrator
        Pipeline orchestrator instance.
    runner : MedpipeRunner
        Pipeline runner instance.
    fitted_models : dict of str to object
        Dictionary mapping outcome names to fitted estimators or pipelines.
    metrics : list of str
        List of metric names used during evaluation.
    n_bootstraps : int
        Number of bootstrap iterations.
    ci_level : float
        Target confidence interval level.
    random_state : int, np.random.Generator, or None
        Random state instance for resampling.
    logger : logging.Logger
        Logger instance configured under `"medpipe.evaluator"`.

    Methods
    -------
    predict(X, model=None, outcome=None)
        Predict class labels for samples in X.
    predict_proba(X, model=None, outcome=None)
        Predict class probabilities for samples in X.
    decision_function(X, model=None, outcome=None)
        Compute decision function scores for samples in X.
    extract_subgroups(X, subgroup_specs)
        Extract index subsets for specified data subgroups.
    evaluate(X, y, outcome=None, metrics=None, subgroup_specs=None, save_artifacts=True)
        Evaluate model performance with confidence intervals across full data and subgroups.

    """

    def __init__(
        self,
        orchestrator: MedpipeOrchestrator,
        runner: MedpipeRunner,
    ) -> None:
        self.orchestrator = orchestrator
        self.runner = runner
        eval_config = self.orchestrator.config.workflow.evaluation

        self.fitted_models: Dict[str, Any] = getattr(runner, "fitted_models", {})
        self.n_bootstraps = eval_config.metrics.n_bootstraps
        self.ci_level = eval_config.metrics.ci_level
        self.random_state = self.orchestrator.config.workflow.random_state
        self.logger = get_console_logger("medpipe.evaluator")

        self.metrics = eval_config.metrics.metrics

    def _get_model(
        self, model: Optional[Any] = None, outcome: Optional[str] = None
    ) -> Any:
        """
        Resolve estimator model from argument, outcome key, or fitted dictionary.

        Parameters
        ----------
        model : object, optional
            Explicit fitted model instance.
        outcome : str, optional
            Outcome name key corresponding to an entry in `self.fitted_models`.

        Returns
        -------
        resolved_model : object
            Fitted estimator or pipeline instance.

        Raises
        ------
        KeyError
            If specified outcome is not found in `self.fitted_models`.
        ValueError
            If model is unspecified and `self.fitted_models` contains multiple models.

        """
        if model is not None:
            return model
        if outcome is not None:
            if outcome in self.fitted_models:
                return self.fitted_models[outcome]
            raise KeyError(
                f"Outcome '{outcome}' not found in runner.fitted_models dictionary."
            )
        if len(self.fitted_models) == 1:
            return next(iter(self.fitted_models.values()))
        raise ValueError(
            "Multiple models found in runner.fitted_models. "
            "Please explicitly specify 'outcome' or pass 'model'."
        )

    def predict(
        self,
        X: Union[pd.DataFrame, npt.NDArray],
        model: Optional[Any] = None,
        outcome: Optional[str] = None,
    ) -> npt.NDArray:
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            Features dataset of shape (n_samples, n_features).
        model : object, optional
            Fitted model instance. If None, resolved via `outcome` or `fitted_models`.
        outcome : str, optional
            Outcome key to look up in `self.fitted_models`.

        Returns
        -------
        y_pred : numpy.ndarray
            Predicted class labels of shape (n_samples,).

        Raises
        ------
        AttributeError
            If the resolved model does not implement a `predict` method.

        """
        target_model = self._get_model(model, outcome)
        if not hasattr(target_model, "predict"):
            raise AttributeError("The underlying model does not implement 'predict'.")
        return np.asarray(target_model.predict(X))

    def predict_proba(
        self,
        X: Union[pd.DataFrame, npt.NDArray],
        model: Optional[Any] = None,
        outcome: Optional[str] = None,
    ) -> npt.NDArray:
        """
        Predict class probabilities for samples in X.

        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            Features dataset of shape (n_samples, n_features).
        model : object, optional
            Fitted model instance. If None, resolved via `outcome` or `fitted_models`.
        outcome : str, optional
            Outcome key to look up in `self.fitted_models`.

        Returns
        -------
        y_proba : numpy.ndarray
            Predicted class probabilities of shape (n_samples, n_classes) or (n_samples,).

        Raises
        ------
        AttributeError
            If the resolved model does not implement a `predict_proba` method.

        """
        target_model = self._get_model(model, outcome)
        if not hasattr(target_model, "predict_proba"):
            raise AttributeError(
                "The underlying model does not implement 'predict_proba'."
            )
        return np.asarray(target_model.predict_proba(X))

    def decision_function(
        self,
        X: Union[pd.DataFrame, npt.NDArray],
        model: Optional[Any] = None,
        outcome: Optional[str] = None,
    ) -> npt.NDArray:
        """
        Compute decision function scores for samples in X.

        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            Features dataset of shape (n_samples, n_features).
        model : object, optional
            Fitted model instance. If None, resolved via `outcome` or `fitted_models`.
        outcome : str, optional
            Outcome key to look up in `self.fitted_models`.

        Returns
        -------
        scores : numpy.ndarray
            Confidence scores or decision function values of shape (n_samples,).

        Raises
        ------
        AttributeError
            If the resolved model does not implement a `decision_function` method.

        """
        target_model = self._get_model(model, outcome)
        if not hasattr(target_model, "decision_function"):
            raise AttributeError(
                "The underlying model does not implement 'decision_function'."
            )
        return np.asarray(target_model.decision_function(X))

    def extract_subgroups(
        self,
        X: pd.DataFrame,
        subgroup_specs: Dict[str, Union[str, Callable[[pd.DataFrame], pd.Series]]],
    ) -> Dict[str, Dict[str, pd.Index]]:
        """
        Extract sample index subsets for specified data subgroups.

        Parameters
        ----------
        X : pandas.DataFrame
            Input feature dataset.
        subgroup_specs : dict of str to (str or callable)
            Mapping where keys are subgroup category names (e.g., `'age_group'`) and values are:
            - Column name (`str`): Groups samples by unique column values.
            - Predicate (`callable`): Function taking `X` and returning a boolean `pd.Series` mask.

        Returns
        -------
        subgroups : dict of str to dict of str to pandas.Index
            Nested dictionary structured as `{category_name: {subgroup_val: row_indices}}`.

        Raises
        ------
        KeyError
            If a specified column string is missing from `X`.
        TypeError
            If a specification is neither a string nor a callable.

        """
        self.logger.debug(
            f"Extracting subgroups for {len(subgroup_specs)} categories.",
        )
        subgroups: Dict[str, Dict[str, pd.Index]] = {}

        for cat_name, spec in subgroup_specs.items():
            cat_subgroups: Dict[str, pd.Index] = {}

            if isinstance(spec, str) and spec in X.columns:
                # Column string: discrete categorical groupby
                for val, group_df in X.groupby(spec):
                    cat_subgroups[str(val)] = group_df.index

            elif isinstance(spec, (list, tuple)):
                # List of custom ranges e.g. [[18, 50], [51, 120]]
                for grp in spec:
                    mask = resolve_subgroup_mask(df=X, column=cat_name, group=grp)
                    grp_key = (
                        f"[{grp[0]}, {grp[1]}]"
                        if isinstance(grp, (list, tuple)) and len(grp) == 2
                        else str(grp)
                    )
                    cat_subgroups[grp_key] = X.index[mask]

            elif callable(spec):
                mask = spec(X)
                if not isinstance(mask, pd.Series):
                    mask = pd.Series(mask, index=X.index)
                cat_subgroups["true"] = X.index[mask]
                cat_subgroups["false"] = X.index[~mask]
            else:
                mask = resolve_subgroup_mask(df=X, column=cat_name, group=spec)
                cat_subgroups[str(spec)] = X.index[mask]

            subgroups[cat_name] = cat_subgroups
            self.logger.debug(
                f"Extracted stratum '{cat_name}' matching groups: {spec}",
            )

        return subgroups

    def _evaluate_slice(
        self,
        y_true: npt.NDArray,
        y_pred: npt.NDArray,
        metrics: List[str],
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate a single slice of data with TRIPOD+AI compliant confidence intervals.

        Parameters
        ----------
        y_true : numpy.ndarray
            Ground truth target array for the data slice.
        y_pred : numpy.ndarray
            Predicted probabilities or scores for the data slice.
        metrics : list of str
            List of metric names to evaluate.

        Returns
        -------
        scores : dict of str to dict of str to float
            Nested dictionary mapping metric names to dictionaries containing:
            - `"point_estimate"`: Metric score computed on original dataset slice.
            - `"ci_lower"`: Lower bound of the bootstrap confidence interval.
            - `"ci_upper"`: Upper bound of the bootstrap confidence interval.

        """
        try:
            return bootstrap_confidence_intervals(
                metrics=metrics,
                y_true=y_true,
                y_pred=y_pred,
                n_bootstraps=self.n_bootstraps,
                ci_level=self.ci_level,
                random_state=self.random_state,
            )
        except Exception as err:
            self.logger.warning(
                "Bootstrap CI calculation failed for slice (%s). Falling back to point estimates.",
                err,
            )
            try:
                point_scores = compute_metrics(metrics, y_true, y_pred)
                return {
                    metric: {
                        "point_estimate": float(score),
                        "ci_lower": np.nan,
                        "ci_upper": np.nan,
                    }
                    for metric, score in zip(metrics, point_scores)
                }
            except Exception:
                return {
                    metric: {
                        "point_estimate": np.nan,
                        "ci_lower": np.nan,
                        "ci_upper": np.nan,
                    }
                    for metric in metrics
                }

    def evaluate(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, npt.NDArray],
        outcome: Optional[str] = None,
        model: Optional[Any] = None,
        metrics: Optional[List[str]] = None,
        subgroup_specs: Optional[
            Dict[str, Union[str, Callable[[pd.DataFrame], pd.Series]]]
        ] = None,
        save_artifacts: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate model performance across the full dataset and optional subgroups.

        Computes point estimates and bootstrap confidence intervals for all requested
        metrics in accordance with TRIPOD+AI guidelines. Results are optionally saved
        to disk via the orchestrator's `ArtifactManager`.

        Parameters
        ----------
        X : pandas.DataFrame
            Feature dataset of shape (n_samples, n_features).
        y : pandas.Series or numpy.ndarray
            Ground truth target values of shape (n_samples,).
        outcome : str, optional
            Outcome key name. Used for logging, model resolution, and artifact saving.
        model : object, optional
            Explicit fitted model instance. If None, resolved from `self.fitted_models`.
        metrics : list of str, optional
            List of metrics to evaluate. If None, defaults to `self.metrics`.
        subgroup_specs : dict of str to (str or callable), optional
            Specifications for extracting demographic/clinical subgroups.
        save_artifacts : bool, default=True
            Whether to write evaluation summary results to disk via `ArtifactManager`.

        Returns
        -------
        results : dict of str to Any
            Nested dictionary containing:
            - `"outcome"`: Name of outcome evaluated.
            - `"overall"`: Dict mapping metric names to dicts of point estimates and CIs.
            - `"subgroups"`: Dict mapping subgroup strata to slice metric results.

        """
        outcome_name = outcome or "default_outcome"
        self.logger.info(f"[{outcome_name}] Starting model evaluation")

        eval_metrics = metrics if metrics is not None else self.metrics
        target_model = self._get_model(model, outcome)

        y_arr = np.asarray(y)

        # Retrieve model predictions
        if hasattr(target_model, "predict_proba"):
            y_pred = self.predict_proba(X, model=target_model)
        elif hasattr(target_model, "decision_function"):
            y_pred = self.decision_function(X, model=target_model)
        else:
            y_pred = self.predict(X, model=target_model)

        # 1. Compute overall evaluation with confidence intervals
        self.logger.info(
            "[%s] Computing overall metrics with %d bootstrap iterations (CI=%.2f)",
            outcome_name,
            self.n_bootstraps,
            self.ci_level,
        )
        results: Dict[str, Any] = {
            "outcome": outcome_name,
            "overall": self._evaluate_slice(y_arr, y_pred, eval_metrics),
        }

        # 2. Compute subgroup metrics using identical slice evaluation logic
        if subgroup_specs:
            self.logger.info(
                f"[{outcome_name}] Evaluating performance across extracted subgroups"
            )
            subgroup_results: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
            subgroups = self.extract_subgroups(X, subgroup_specs)

            for cat_name, cat_groups in subgroups.items():
                subgroup_results[cat_name] = {}
                for group_val, indices in cat_groups.items():
                    if len(indices) == 0:
                        self.logger.warning(
                            "Subgroup '%s=%s' is empty. Skipping.", cat_name, group_val
                        )
                        continue

                    pos_idx = X.index.get_indexer(indices)
                    y_sub = y_arr[pos_idx]
                    y_pred_sub = y_pred[pos_idx]

                    subgroup_results[cat_name][group_val] = self._evaluate_slice(
                        y_sub, y_pred_sub, eval_metrics
                    )

            results["subgroups"] = subgroup_results

        self.logger.info(f"[{outcome_name}] Completed evaluation for outcome")

        # 3. Save artifacts using ArtifactManager
        if save_artifacts:
            self._save_evaluation_artifacts(results, outcome=outcome_name)

        return results

    def _save_evaluation_artifacts(
        self,
        results: Dict[str, Any],
        outcome: str,
    ) -> Path:
        """
        Save evaluation outcomes to disk using the ArtifactManager or run directory.

        Parameters
        ----------
        results : dict of str to Any
            Evaluation results dictionary containing overall and subgroup metrics.
        outcome : str
            Name of the outcome evaluated.

        Returns
        -------
        saved_path : pathlib.Path
            Path to the saved artifact file on disk.

        """
        filename = f"{outcome}_evaluation_results.json"
        saved_path: Optional[Path] = None

        artifacts_dir = self.orchestrator.run_dir
        artifact_mgr = self.orchestrator.artifact_manager

        saved_path = artifact_mgr.save_json(results, artifacts_dir, filename)

        self.logger.info(
            f"[{outcome}] Successfully saved evaluation artifacts to {saved_path}",
        )
        return saved_path
