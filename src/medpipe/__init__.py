"""
Medpipe: A Python framework for clinical machine learning pipeline orchestration.

Provides unified high-level interfaces for data preparation, model fitting,
post-hoc calibration, TRIPOD+AI compliant evaluation, and reproducibility tracking.
"""

from medpipe.metrics import MetricRegistry
from medpipe.models import ModelRegistry
from medpipe.pipeline import (
    Medpipe,
    MedpipeEvaluator,
    MedpipeOrchestrator,
    MedpipeRunner,
)
from medpipe.utils import MedpipeConfig

__version__ = "0.4.0"

__all__ = [
    # Primary API Entry Point
    "Medpipe",
    "MedpipeConfig",
    # Sub-Orchestrators (for custom/modular workflows)
    "MedpipeOrchestrator",
    "MedpipeRunner",
    "MedpipeEvaluator",
    # Component Registries (for custom models & metrics)
    "ModelRegistry",
    "MetricRegistry",
    # Package Metadata
    "__version__",
]
