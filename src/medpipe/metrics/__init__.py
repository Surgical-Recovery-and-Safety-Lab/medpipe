"""
medpipe.metrics
---------------
Metrics, scoring registries, and plotting utilities for the Medpipe package.

Provides core metric computation functions, bootstrap confidence interval
estimation, cross-validation scorer building, metric registries,
and visualization tools.
"""

from medpipe.metrics.core import (
    bootstrap_confidence_intervals,
    build_scorers,
    compute_metrics,
    ici_score,
)
from medpipe.metrics.plots import (
    plot_probability_distribution,
    plot_reliability_diagram,
    plot_ROC_curve,
    plot_strata_heatmap,
)
from medpipe.metrics.registry import MetricRegistry, MetricSpec

__all__ = [
    # Core metric functions
    "compute_metrics",
    "build_scorers",
    "bootstrap_confidence_intervals",
    "ici_score",
    # Metric registry and specifications
    "MetricRegistry",
    "MetricSpec",
    # Plotting routines
    "plot_probability_distribution",
    "plot_reliability_diagram",
    "plot_ROC_curve",
    "plot_strata_heatmap",
]
