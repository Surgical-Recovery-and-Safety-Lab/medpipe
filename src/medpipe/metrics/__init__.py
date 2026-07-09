"""
medpipe.metrics module

submodules:
- core: contains functions to compute and print metrics.
- plots: contains functions to plot results.
"""

from . import core, plots
from .core import compute_metrics, print_metrics
from .plots import plot_probability_distribution, plot_reliability_diagram
