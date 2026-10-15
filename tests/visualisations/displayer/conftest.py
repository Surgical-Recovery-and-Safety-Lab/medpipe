"""
Shared fixtures for the MedpipeClassifierDisplayer test suite.
"""

from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock

import matplotlib.pyplot as plt
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _close_figures() -> Generator:
    """Automatically close all Matplotlib figures after each test."""
    yield
    plt.close("all")


@pytest.fixture
def mock_orchestrator(tmp_path: Path) -> MagicMock:
    """Provides a mock MedpipeOrchestrator with a temporary run_dir."""
    orchestrator = MagicMock()
    orchestrator.run_dir = tmp_path
    orchestrator.config.display = None
    return orchestrator


@pytest.fixture
def sample_binary_data() -> tuple[np.ndarray, np.ndarray]:
    """Provides synthetic ground truth labels and predicted probabilities."""
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=100)
    probas = rng.uniform(0.0, 1.0, size=100)
    return y_true, probas


@pytest.fixture
def sample_evaluations() -> dict:
    """Provides a realistic nested evaluation result dictionary across
    outcomes and strata."""
    return {
        "ANY_COMP": {
            "outcome": "ANY_COMP",
            "overall": {
                "roc_auc": {
                    "point_estimate": 0.5666666666666667,
                    "ci_lower": 0.44575071660019944,
                    "ci_upper": 0.6805549568965518,
                },
                "log_loss": {
                    "point_estimate": 1.6449383564955808,
                    "ci_lower": 1.157229903186509,
                    "ci_upper": 2.0489328647437963,
                },
            },
            "strata": {
                "SEX": {
                    "F": {
                        "roc_auc": {
                            "point_estimate": 0.5,
                            "ci_lower": 0.2906,
                            "ci_upper": 0.6702,
                        },
                        "log_loss": {
                            "point_estimate": 1.2839,
                            "ci_lower": 0.7626,
                            "ci_upper": 1.9761,
                        },
                    },
                    "M": {
                        "roc_auc": {
                            "point_estimate": 0.5833,
                            "ci_lower": 0.3902,
                            "ci_upper": 0.8017,
                        },
                        "log_loss": {
                            "point_estimate": 2.1413,
                            "ci_lower": 1.3700,
                            "ci_upper": 3.0628,
                        },
                    },
                },
                "AGE": {
                    "[18, 50]": {
                        "roc_auc": {
                            "point_estimate": 0.6783,
                            "ci_lower": 0.5380,
                            "ci_upper": 0.8192,
                        },
                        "log_loss": {
                            "point_estimate": 1.8763,
                            "ci_lower": 1.0491,
                            "ci_upper": 2.5909,
                        },
                    },
                    "[51, 120]": {
                        "roc_auc": {
                            "point_estimate": 0.2470,
                            "ci_lower": 0.0713,
                            "ci_upper": 0.5702,
                        },
                        "log_loss": {
                            "point_estimate": 1.4253,
                            "ci_lower": 0.8195,
                            "ci_upper": 2.1574,
                        },
                    },
                },
            },
        },
        "MORTALITY_30D": {
            "outcome": "MORTALITY_30D",
            "overall": {
                "roc_auc": {
                    "point_estimate": 0.720,
                    "ci_lower": 0.600,
                    "ci_upper": 0.840,
                },
                "log_loss": {
                    "point_estimate": 0.450,
                    "ci_lower": 0.300,
                    "ci_upper": 0.600,
                },
            },
            "strata": {
                "SEX": {
                    "F": {
                        "roc_auc": {
                            "point_estimate": 0.700,
                            "ci_lower": 0.550,
                            "ci_upper": 0.820,
                        },
                        "log_loss": {
                            "point_estimate": 0.480,
                            "ci_lower": 0.320,
                            "ci_upper": 0.620,
                        },
                    },
                    "M": {
                        "roc_auc": {
                            "point_estimate": 0.740,
                            "ci_lower": 0.610,
                            "ci_upper": 0.850,
                        },
                        "log_loss": {
                            "point_estimate": 0.420,
                            "ci_lower": 0.280,
                            "ci_upper": 0.580,
                        },
                    },
                },
                "AGE": {
                    "[18, 50]": {
                        "roc_auc": {
                            "point_estimate": 0.750,
                            "ci_lower": 0.620,
                            "ci_upper": 0.870,
                        },
                        "log_loss": {
                            "point_estimate": 0.400,
                            "ci_lower": 0.250,
                            "ci_upper": 0.550,
                        },
                    },
                    "[51, 120]": {
                        "roc_auc": {
                            "point_estimate": 0.680,
                            "ci_lower": 0.520,
                            "ci_upper": 0.810,
                        },
                        "log_loss": {
                            "point_estimate": 0.500,
                            "ci_lower": 0.350,
                            "ci_upper": 0.650,
                        },
                    },
                },
            },
        },
    }
