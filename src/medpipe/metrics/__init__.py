"""
medpipe.metrics module

submodules:
- core: contains functions to compute and print metrics.
- plots: contains functions to plot results.
"""

from . import core, plots
from .core import (
    compute_all_CI,
    compute_pred_metrics,
    compute_score_metrics,
    print_metrics,
    print_metrics_CI,
)
from .plots import (
    plot_metrics_CI,
    plot_prediction_distribution,
    plot_reliability_diagrams,
)
