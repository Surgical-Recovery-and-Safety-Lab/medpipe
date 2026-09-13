"""Visualisation and plotting module for MedpipeClassifier.

Provides the high-level MedpipeClassifierDisplayer manager, stateless drawing
primitives, and aesthetic theme configurations for pipeline evaluation graphics.
"""

from medpipe.visualisation.displayer import MedpipeClassifierDisplayer
from medpipe.visualisation.plots import (
    draw_dca_curve,
    draw_precision_recall_curve,
    draw_probability_distribution,
    draw_reliability_diagram,
    draw_roc_curve,
    draw_strata_heatmap,
)
from medpipe.visualisation.themes import MedpipeTheme

__all__ = [
    "MedpipeClassifierDisplayer",
    "MedpipeTheme",
    "draw_dca_curve",
    "draw_precision_recall_curve",
    "draw_probability_distribution",
    "draw_reliability_diagram",
    "draw_roc_curve",
    "draw_strata_heatmap",
]
