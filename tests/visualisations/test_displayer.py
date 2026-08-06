"""Tests for the MedpipeDisplayer class."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

matplotlib.use("Agg")  # Non-interactive backend for headless testing

from medpipe.visualisation.displayer import MedpipeDisplayer
from medpipe.visualisation.themes import MedpipeTheme


@pytest.fixture(autouse=True)
def _close_figures():
    """Automatically close all Matplotlib figures after each test."""
    yield
    plt.close("all")


@pytest.fixture
def mock_orchestrator(tmp_path: Path):
    """Provides a mock MedpipeOrchestrator with a temporary run_dir."""
    orchestrator = MagicMock()
    orchestrator.run_dir = tmp_path
    return orchestrator


@pytest.fixture
def sample_binary_data():
    """Provides synthetic ground truth labels and predicted probabilities."""
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=100)
    probas = rng.uniform(0.0, 1.0, size=100)
    return y_true, probas


class TestMedpipeDisplayerInit:
    """Tests for MedpipeDisplayer initialization."""

    def test_init_default_theme(self, mock_orchestrator) -> None:
        """Test initialization with default MedpipeTheme."""
        displayer = MedpipeDisplayer(orchestrator=mock_orchestrator)

        assert displayer.orchestrator == mock_orchestrator
        assert displayer.run_dir == mock_orchestrator.run_dir
        assert isinstance(displayer.theme, MedpipeTheme)
        assert displayer.logger is not None

    def test_init_custom_theme(self, mock_orchestrator) -> None:
        """Test initialization with a custom MedpipeTheme."""
        custom_theme = MedpipeTheme(primary_color="#FF0000", dpi=150)
        displayer = MedpipeDisplayer(orchestrator=mock_orchestrator, theme=custom_theme)

        assert displayer.theme.primary_color == "#FF0000"
        assert displayer.theme.dpi == 150


class TestComputeRocData:
    """Tests for the internal `_compute_roc_data` statistical helper method."""

    def test_compute_roc_data_success_1d_probas(
        self, mock_orchestrator, sample_binary_data
    ) -> None:
        """Test ROC data calculation with 1D predicted probabilities and bootstrapping."""
        y_true, probas = sample_binary_data
        displayer = MedpipeDisplayer(orchestrator=mock_orchestrator)

        fpr, tpr, roc_auc, lower_ci, upper_ci = displayer._compute_roc_data(
            y_true=y_true, probas=probas, n_bootstraps=50
        )

        assert isinstance(fpr, np.ndarray)
        assert isinstance(tpr, np.ndarray)
        assert isinstance(roc_auc, float)
        assert 0.0 <= roc_auc <= 1.0
        assert isinstance(lower_ci, np.ndarray)
        assert isinstance(upper_ci, np.ndarray)
        assert len(lower_ci) == len(fpr)
        assert len(upper_ci) == len(fpr)

    def test_compute_roc_data_2d_probas(
        self, mock_orchestrator, sample_binary_data
    ) -> None:
        """Test ROC data calculation when probabilities are 2D (n_samples, 2)."""
        y_true, probas_1d = sample_binary_data
        probas_2d = np.column_stack((1 - probas_1d, probas_1d))
        displayer = MedpipeDisplayer(orchestrator=mock_orchestrator)

        fpr, tpr, roc_auc, lower_ci, upper_ci = displayer._compute_roc_data(
            y_true=y_true, probas=probas_2d, n_bootstraps=10
        )

        assert isinstance(roc_auc, float)
        assert lower_ci is not None
        assert upper_ci is not None

    def test_compute_roc_data_disabled_bootstraps(
        self, mock_orchestrator, sample_binary_data
    ) -> None:
        """Test that n_bootstraps <= 0 returns None for confidence interval arrays."""
        y_true, probas = sample_binary_data
        displayer = MedpipeDisplayer(orchestrator=mock_orchestrator)

        fpr, tpr, roc_auc, lower_ci, upper_ci = displayer._compute_roc_data(
            y_true=y_true, probas=probas, n_bootstraps=0
        )

        assert isinstance(fpr, np.ndarray)
        assert isinstance(tpr, np.ndarray)
        assert isinstance(roc_auc, float)
        assert lower_ci is None
        assert upper_ci is None

    def test_compute_roc_data_single_class_resamples_ignored(
        self, mock_orchestrator
    ) -> None:
        """Edge case: ensure bootstraps skipping single-class resamples do not crash execution."""
        # Highly imbalanced dataset where bootstrap resamples frequently lack positive class
        y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        probas = np.linspace(0.1, 0.9, 10)
        displayer = MedpipeDisplayer(orchestrator=mock_orchestrator)

        fpr, tpr, roc_auc, lower_ci, upper_ci = displayer._compute_roc_data(
            y_true=y_true, probas=probas, n_bootstraps=20, random_state=1
        )

        assert isinstance(roc_auc, float)


class TestSaveFigure:
    """Tests for the internal `_save_figure` method."""

    def test_save_figure_default_directory(
        self, mock_orchestrator, tmp_path: Path
    ) -> None:
        """Test saving a figure to the default run directory path."""
        displayer = MedpipeDisplayer(orchestrator=mock_orchestrator)
        fig, _ = plt.subplots()

        saved_path = displayer._save_figure(fig=fig, filename="test_plot")

        expected_path = tmp_path / "plots" / "test_plot.png"
        assert saved_path == expected_path
        assert saved_path.exists()

    def test_save_figure_with_outcome_subdirectory(
        self, mock_orchestrator, tmp_path: Path
    ) -> None:
        """Test saving a figure inside an outcome-specific subdirectory."""
        displayer = MedpipeDisplayer(orchestrator=mock_orchestrator)
        fig, _ = plt.subplots()

        saved_path = displayer._save_figure(
            fig=fig, filename="roc_curve", outcome="mortality"
        )

        expected_path = tmp_path / "plots" / "mortality" / "roc_curve.png"
        assert saved_path == expected_path
        assert saved_path.exists()


class TestPlotRocCurve:
    """Tests for the high-level `plot_roc_curve` method."""

    def test_plot_roc_curve_success_and_saves(
        self, mock_orchestrator, sample_binary_data, tmp_path: Path
    ) -> None:
        """Test successful ROC plot generation with default artifact saving."""
        y_true, probas = sample_binary_data
        displayer = MedpipeDisplayer(orchestrator=mock_orchestrator)

        fig, ax = displayer.plot_roc_curve(
            y_true=y_true,
            probas=probas,
            outcome="diabetes",
            n_bootstraps=10,
            save=True,
            show=False,
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert (tmp_path / "plots" / "diabetes" / "diabetes_roc_curve.png").exists()

    def test_plot_roc_curve_without_saving(
        self, mock_orchestrator, sample_binary_data, tmp_path: Path
    ) -> None:
        """Test ROC plot generation when save=False."""
        y_true, probas = sample_binary_data
        displayer = MedpipeDisplayer(orchestrator=mock_orchestrator)

        fig, ax = displayer.plot_roc_curve(
            y_true=y_true,
            probas=probas,
            outcome="stroke",
            n_bootstraps=0,
            save=False,
            show=False,
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert not (tmp_path / "plots" / "stroke" / "stroke_roc_curve.png").exists()

    def test_plot_roc_curve_custom_label_and_styles(
        self, mock_orchestrator, sample_binary_data
    ) -> None:
        """Test passing custom legend labels and style override parameters."""
        y_true, probas = sample_binary_data
        displayer = MedpipeDisplayer(orchestrator=mock_orchestrator)

        fig, ax = displayer.plot_roc_curve(
            y_true=y_true,
            probas=probas,
            outcome="cardiac",
            label="XGBoost Classifier",
            n_bootstraps=5,
            save=False,
            show=False,
            color="#00FF00",
            ci_alpha=0.1,
            show_spines=True,
        )

        labels = [text.get_text() for text in ax.get_legend().get_texts()]
        assert "XGBoost Classifier" in labels
        assert "ROC Curve - Diabetes" not in ax.get_title()
        assert "ROC Curve - Cardiac" in ax.get_title()

    @patch("matplotlib.pyplot.show")
    def test_plot_roc_curve_show_flag(
        self, mock_show, mock_orchestrator, sample_binary_data
    ) -> None:
        """Test interactive figure display when show=True."""
        y_true, probas = sample_binary_data
        displayer = MedpipeDisplayer(orchestrator=mock_orchestrator)

        displayer.plot_roc_curve(
            y_true=y_true,
            probas=probas,
            n_bootstraps=0,
            save=False,
            show=True,
        )

        mock_show.assert_called_once()
