"""
medpipe module

submodules:
- data: contains data related functions
- metrics: contains model metric functions
- models: contains model creation functions
- pipeline: contains pipeline functions
- utils: contains utility functions
"""

from .data import extract_labels
from .metrics import (
    plot_metrics_CI,
    plot_prediction_distribution,
    plot_reliability_diagrams,
    print_metrics,
)
from .models import load_pipeline, save_pipeline
from .pipeline.pipeline import MedpipePipeline
from .utils import (
    exception_handler,
    load_data,
    print_message,
    read_toml_configuration,
    setup_logger,
)

__all__ = [
    "MedpipePipeline",
    "exception_handler",
    "load_data",
    "print_message",
    "read_toml_configuration",
    "setup_logger",
    "load_pipeline",
    "save_pipeline",
    "plot_metrics_CI",
    "plot_prediction_distribution",
    "plot_reliability_diagrams",
    "print_metrics",
    "extract_labels",
]
__version__ = "v0.3.0"
