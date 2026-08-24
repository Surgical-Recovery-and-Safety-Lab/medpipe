from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Union

from sklearn.metrics import get_scorer, make_scorer

from medpipe.utils.registry import BaseRegistry


@dataclass(frozen=True)
class MetricSpec:
    """Encapsulates execution details and metadata for an evaluation metric.

    Parameters
    ----------
    name : str
        Unique lookup identifier for the metric.
    func : Callable
        Underlying metric evaluation function.
    response_method : str or tuple of str
        Scikit-learn model prediction method required by the metric
        (e.g., 'predict', 'predict_proba').
    display_name : str
        Human-readable name used for visual displays and reports.
    sklearn_scorer_name : str or None, default=None
        Optional pre-registered scikit-learn scorer string key.

    """

    name: str
    func: Callable
    response_method: Union[str, tuple[str, ...]]
    display_name: str
    sklearn_scorer_name: str | None = None

    def get_scorer(self) -> Callable:
        """Construct a scikit-learn compatible scorer function for cross-validation.

        Returns
        -------
        scorer : Callable
            Scikit-learn scorer object suitable for model evaluation or tuning.
        """
        if self.sklearn_scorer_name:
            return get_scorer(self.sklearn_scorer_name)

        need_proba = "predict_proba" in (
            self.response_method
            if isinstance(self.response_method, tuple)
            else (self.response_method,)
        )
        return make_scorer(
            self.func,
            response_method="predict_proba" if need_proba else "predict",
        )


class MetricRegistry(BaseRegistry[MetricSpec]):
    """Registry managing available and custom MetricSpec definitions."""

    @classmethod
    def register_spec(cls, spec: MetricSpec) -> MetricSpec:
        """Register a MetricSpec instance directly into the registry.

        Parameters
        ----------
        spec : MetricSpec
            The metric specification instance to register.

        Returns
        -------
        MetricSpec
            The registered metric specification instance.
        """
        cls._registry[spec.name] = spec
        return spec

    @classmethod
    def get(cls, name: str) -> MetricSpec:
        """Retrieve a registered MetricSpec instance by name.

        Parameters
        ----------
        name : str
            The name identifier of the metric to retrieve.

        Returns
        -------
        MetricSpec
            The matching metric specification instance.

        Raises
        ------
        ValueError
            If the requested metric name is not found in the registry.
        """
        if name in cls._registry:
            return cls._registry[name]

        raise ValueError(
            f"'{name}' was not found in available metrics. "
            f"Available metrics are {cls.list_registered()}"
        )
