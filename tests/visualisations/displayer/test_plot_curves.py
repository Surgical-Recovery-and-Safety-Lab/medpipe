"""
Tests for MedpipeClassifierDisplayer's high-level curve-plotting methods:
plot_data_distribution, plot_roc_curve, plot_precision_recall_curve,
plot_reliability_diagram, and plot_dca_curve.
"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from medpipe.visualisation.displayer import MedpipeClassifierDisplayer


class TestPlotDataDistribution:
    """Tests for the high-level `plot_data_distribution` method."""

    def test_plot_data_distribution_success_and_saves(
        self, mock_orchestrator, sample_binary_data, tmp_path: Path
    ) -> None:
        """Test successful distribution plot generation with artifact saving."""
        _, probas = sample_binary_data
        displayer = MedpipeClassifierDisplayer(orchestrator=mock_orchestrator)

        fig, ax = displayer.plot_data_distribution(
            data=probas,
            outcome="mortality",
            n_bins=12,
            save=True,
            show=False,
        )

        expected_file = (
            tmp_path / "plots" / "mortality" / "mortality_data_distribution.png"
        )
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert expected_file.exists()

    def test_plot_data_distribution_no_save(
        self, mock_orchestrator, sample_binary_data, tmp_path: Path
    ) -> None:
        """Test that save=False skips artifact persistence."""
        _, probas = sample_binary_data
        displayer = MedpipeClassifierDisplayer(orchestrator=mock_orchestrator)

        displayer.plot_data_distribution(
            data=probas, outcome="mortality", save=False, show=False
        )

        assert not (tmp_path / "plots").exists()

    @patch("matplotlib.pyplot.show")
    def test_plot_data_distribution_show_flag(
        self, mock_show, mock_orchestrator, sample_binary_data
    ) -> None:
        """Test interactive display when show=True."""
        _, probas = sample_binary_data
        displayer = MedpipeClassifierDisplayer(orchestrator=mock_orchestrator)

        displayer.plot_data_distribution(data=probas, save=False, show=True)

        mock_show.assert_called_once()


class TestPlotRocCurve:
    """Tests for the high-level `plot_roc_curve` method."""

    def test_plot_roc_curve_success_and_saves(
        self, mock_orchestrator, sample_binary_data, tmp_path: Path
    ) -> None:
        """Test successful ROC plot generation with default artifact saving."""
        y_true, probas = sample_binary_data
        displayer = MedpipeClassifierDisplayer(orchestrator=mock_orchestrator)

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

    def test_plot_roc_curve_no_save(
        self, mock_orchestrator, sample_binary_data, tmp_path: Path
    ) -> None:
        """Test that save=False skips artifact persistence."""
        y_true, probas = sample_binary_data
        displayer = MedpipeClassifierDisplayer(orchestrator=mock_orchestrator)

        displayer.plot_roc_curve(
            y_true=y_true, probas=probas, n_bootstraps=10, save=False, show=False
        )

        assert not (tmp_path / "plots").exists()

    @patch("matplotlib.pyplot.show")
    def test_plot_roc_curve_show_flag(
        self, mock_show, mock_orchestrator, sample_binary_data
    ) -> None:
        """Test interactive display when show=True."""
        y_true, probas = sample_binary_data
        displayer = MedpipeClassifierDisplayer(orchestrator=mock_orchestrator)

        displayer.plot_roc_curve(
            y_true=y_true, probas=probas, n_bootstraps=10, save=False, show=True
        )

        mock_show.assert_called_once()


class TestPlotPrecisionRecallCurve:
    """Tests for the high-level `plot_precision_recall_curve` method."""

    def test_plot_pr_curve_success_and_saves(
        self, mock_orchestrator, sample_binary_data, tmp_path: Path
    ) -> None:
        """Test successful PR curve generation with figure artifact saving."""
        y_true, probas = sample_binary_data
        displayer = MedpipeClassifierDisplayer(orchestrator=mock_orchestrator)

        fig, ax = displayer.plot_precision_recall_curve(
            y_true=y_true,
            probas=probas,
            outcome="sepsis",
            n_bootstraps=10,
            save=True,
            show=False,
        )

        expected_file = tmp_path / "plots" / "sepsis" / "sepsis_pr_curve.png"
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert expected_file.exists()

    def test_plot_pr_curve_no_save(
        self, mock_orchestrator, sample_binary_data, tmp_path: Path
    ) -> None:
        """Test that save=False skips artifact persistence."""
        y_true, probas = sample_binary_data
        displayer = MedpipeClassifierDisplayer(orchestrator=mock_orchestrator)

        displayer.plot_precision_recall_curve(
            y_true=y_true, probas=probas, n_bootstraps=10, save=False, show=False
        )

        assert not (tmp_path / "plots").exists()

    @patch("matplotlib.pyplot.show")
    def test_plot_pr_curve_show_flag(
        self, mock_show, mock_orchestrator, sample_binary_data
    ) -> None:
        """Test interactive display when show=True."""
        y_true, probas = sample_binary_data
        displayer = MedpipeClassifierDisplayer(orchestrator=mock_orchestrator)

        displayer.plot_precision_recall_curve(
            y_true=y_true, probas=probas, n_bootstraps=10, save=False, show=True
        )

        mock_show.assert_called_once()


class TestPlotReliabilityDiagram:
    """Tests for the high-level `plot_reliability_diagram` method."""

    def test_plot_reliability_diagram_success_and_saves(
        self, mock_orchestrator, sample_binary_data, tmp_path: Path
    ) -> None:
        """Test successful reliability diagram generation with figure artifact
        saving."""
        y_true, probas = sample_binary_data
        displayer = MedpipeClassifierDisplayer(orchestrator=mock_orchestrator)

        fig, ax = displayer.plot_reliability_diagram(
            y_true=y_true,
            probas=probas,
            outcome="mortality",
            n_bins=5,
            n_bootstraps=10,
            save=True,
            show=False,
        )

        expected_file = (
            tmp_path / "plots" / "mortality" / "mortality_reliability_diagram.png"
        )
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert expected_file.exists()

    def test_plot_reliability_diagram_no_save(
        self, mock_orchestrator, sample_binary_data, tmp_path: Path
    ) -> None:
        """Test that save=False skips artifact persistence."""
        y_true, probas = sample_binary_data
        displayer = MedpipeClassifierDisplayer(orchestrator=mock_orchestrator)

        displayer.plot_reliability_diagram(
            y_true=y_true, probas=probas, n_bootstraps=10, save=False, show=False
        )

        assert not (tmp_path / "plots").exists()

    @patch("matplotlib.pyplot.show")
    def test_plot_reliability_diagram_show_flag(
        self, mock_show, mock_orchestrator, sample_binary_data
    ) -> None:
        """Test interactive display when show=True."""
        y_true, probas = sample_binary_data
        displayer = MedpipeClassifierDisplayer(orchestrator=mock_orchestrator)

        displayer.plot_reliability_diagram(
            y_true=y_true, probas=probas, n_bootstraps=10, save=False, show=True
        )

        mock_show.assert_called_once()

    def test_plot_reliability_diagram_explicit_spline_strategy_suppresses_marker(
        self, mock_orchestrator, sample_binary_data
    ) -> None:
        """Test that passing strategy='spline' explicitly as a runtime
        argument suppresses the calibration point marker, as intended for
        smooth continuous spline curves."""
        y_true, probas = sample_binary_data
        displayer = MedpipeClassifierDisplayer(orchestrator=mock_orchestrator)

        _, ax = displayer.plot_reliability_diagram(
            y_true=y_true,
            probas=probas,
            strategy="spline",
            n_bootstraps=5,
            save=False,
            show=False,
        )

        model_line = next(
            line for line in ax.get_lines() if line.get_label() == "Model"
        )
        assert model_line.get_marker() == "None"

    def test_plot_reliability_diagram_config_resolved_spline_suppresses_marker(
        self, mock_orchestrator, sample_binary_data
    ) -> None:
        """Test that strategy='spline' resolved from display config (not
        passed as an explicit runtime keyword) also suppresses the marker.

        Regression test for a bug where the marker-suppression check
        compared against the raw `strategy` parameter (None when resolved
        purely from config) instead of the resolved config value, so
        config-driven spline plots kept their scatter markers even though
        the curve itself was correctly computed via spline.
        """
        from medpipe.utils.config import DisplayConfig, DisplayDefaultsConfig

        mock_orchestrator.config.display = DisplayConfig(
            defaults=DisplayDefaultsConfig(
                strategy="spline", dist_n_bins=10, dist_yscale="linear"
            )
        )
        y_true, probas = sample_binary_data
        displayer = MedpipeClassifierDisplayer(orchestrator=mock_orchestrator)

        _, ax = displayer.plot_reliability_diagram(
            y_true=y_true, probas=probas, n_bootstraps=5, save=False, show=False
        )

        model_line = next(
            line for line in ax.get_lines() if line.get_label() == "Model"
        )
        assert model_line.get_marker() == "None"


class TestPlotDcaCurve:
    """Tests for the high-level `plot_dca_curve` method."""

    def test_plot_dca_curve_success_and_saves(
        self, mock_orchestrator, sample_binary_data, tmp_path: Path
    ) -> None:
        """Test successful DCA plot generation with figure artifact saving."""
        y_true, probas = sample_binary_data
        displayer = MedpipeClassifierDisplayer(orchestrator=mock_orchestrator)

        fig, ax = displayer.plot_dca_curve(
            y_true=y_true,
            probas=probas,
            outcome="stroke",
            save=True,
            show=False,
        )

        expected_file = tmp_path / "plots" / "stroke" / "stroke_dca_curve.png"
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert expected_file.exists()

    def test_plot_dca_curve_no_save(
        self, mock_orchestrator, sample_binary_data, tmp_path: Path
    ) -> None:
        """Test that save=False skips artifact persistence."""
        y_true, probas = sample_binary_data
        displayer = MedpipeClassifierDisplayer(orchestrator=mock_orchestrator)

        displayer.plot_dca_curve(
            y_true=y_true, probas=probas, save=False, show=False
        )

        assert not (tmp_path / "plots").exists()

    @patch("matplotlib.pyplot.show")
    def test_plot_dca_curve_show_flag(
        self, mock_show, mock_orchestrator, sample_binary_data
    ) -> None:
        """Test interactive display when show=True."""
        y_true, probas = sample_binary_data
        displayer = MedpipeClassifierDisplayer(orchestrator=mock_orchestrator)

        displayer.plot_dca_curve(
            y_true=y_true, probas=probas, save=False, show=True
        )

        mock_show.assert_called_once()

    def test_plot_dca_curve_custom_thresholds(
        self, mock_orchestrator, sample_binary_data
    ) -> None:
        """Test that explicit thresholds are forwarded through to the
        underlying computation and rendering."""
        y_true, probas = sample_binary_data
        displayer = MedpipeClassifierDisplayer(orchestrator=mock_orchestrator)
        custom_thresholds = np.linspace(0.1, 0.5, 10)

        _, ax = displayer.plot_dca_curve(
            y_true=y_true,
            probas=probas,
            thresholds=custom_thresholds,
            save=False,
            show=False,
        )

        model_line = next(
            line for line in ax.get_lines() if line.get_label() == "Model"
        )
        assert len(model_line.get_xdata()) == 10
