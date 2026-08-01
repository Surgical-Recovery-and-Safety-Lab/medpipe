from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Union

from sklearn.metrics import get_scorer, make_scorer

from medpipe.utils.registry import BaseRegistry


@dataclass(frozen=True)
class MetricSpec:
    """Encapsulates execution details for a metric."""

    name: str
    func: Callable
    response_method: Union[str, tuple[str, ...]]
    display_name: str
    sklearn_scorer_name: str | None = None

    def get_scorer(self) -> Callable:
        """Returns a scikit-learn compatible scorer for cross-validation."""
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


class MetricRegistry(BaseRegistry):
    """Registry managing available and custom MetricSpec definitions."""

    _registry: dict[str, MetricSpec] = {}  # Overridden to hold MetricSpec instances
    _fallback_modules: list = []

    @classmethod
    def register_spec(cls, spec: MetricSpec) -> MetricSpec:
        """Registers a MetricSpec instance directly into the registry."""
        cls._registry[spec.name] = spec
        return spec

    @classmethod
    def get(cls, name: str) -> MetricSpec:
        """Retrieve a MetricSpec by name."""
        if name in cls._registry:
            return cls._registry[name]

        raise ValueError(
            f"'{name}' was not found in available metrics. "
            f"Available metrics are {cls.list_registered()}"
        )
