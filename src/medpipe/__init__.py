"""
Medpipe: A Python framework for clinical machine learning pipeline
orchestration.

Provides unified high-level interfaces for data preparation, model fitting,
post-hoc calibration, TRIPOD+AI compliant evaluation, and reproducibility tracking.
"""

from medpipe.metrics import MetricRegistry
from medpipe.models import ModelRegistry
from medpipe.pipeline import (
    MedpipeClassifier,
    MedpipeClassifierEvaluator,
    MedpipeClassifierRunner,
    MedpipeOrchestrator,
    MedpipeRegressor,
    MedpipeRegressorEvaluator,
    MedpipeRegressorRunner,
)
from medpipe.utils import MedpipeClassifierConfig, MedpipeRegressorConfig

__version__ = "0.4.0.dev1"

__all__ = [  # noqa: RUF022 (grouped by category, not alphabetical)
    # Primary API Entry Points
    "MedpipeClassifier",
    "MedpipeClassifierConfig",
    "MedpipeRegressor",
    "MedpipeRegressorConfig",
    # Sub-Orchestrators (for custom/modular workflows)
    "MedpipeOrchestrator",
    "MedpipeClassifierRunner",
    "MedpipeClassifierEvaluator",
    "MedpipeRegressorRunner",
    "MedpipeRegressorEvaluator",
    # Component Registries (for custom models & metrics)
    "ModelRegistry",
    "MetricRegistry",
    # Package Metadata
    "__version__",
]
