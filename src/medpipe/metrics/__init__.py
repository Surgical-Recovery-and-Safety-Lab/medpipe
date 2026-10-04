"""
medpipe.metrics
---------------
Metrics, scoring registries, and plotting utilities for the medpipe package.

Provides core metric computation functions, bootstrap confidence interval
estimation, cross-validation scorer building, and metric registries.
"""

from medpipe.metrics.core import (
    PredictionBundle,
    bootstrap_confidence_intervals,
    build_scorers,
    compute_metrics,
    crps_score,
    ici_score,
)
from medpipe.metrics.registry import MetricRegistry, MetricSpec

__all__ = [  # noqa: RUF022 (grouped by category, not alphabetical)
    # Core metric functions
    "compute_metrics",
    "build_scorers",
    "bootstrap_confidence_intervals",
    "ici_score",
    "crps_score",
    "PredictionBundle",
    # Metric registry and specifications
    "MetricRegistry",
    "MetricSpec",
]
