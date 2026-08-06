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

        _, _, roc_auc, lower_ci, upper_ci = displayer._compute_roc_data(
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

        _, _, roc_auc, _, _ = displayer._compute_roc_data(
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


class TestPlotProbabilityDistribution:
    """Tests for the high-level `plot_probability_distribution` method."""

    def test_plot_probability_distribution_success_and_saves(
        self, mock_orchestrator, sample_binary_data, tmp_path: Path
    ) -> None:
        """Test successful distribution plot generation with artifact saving."""
        _, probas = sample_binary_data
        displayer = MedpipeDisplayer(orchestrator=mock_orchestrator)

        fig, ax = displayer.plot_probability_distribution(
            probas=probas,
            outcome="mortality",
            n_bins=12,
            save=True,
            show=False,
        )

        expected_file = (
            tmp_path / "plots" / "mortality" / "mortality_probability_distribution.png"
        )
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert expected_file.exists()

    def test_plot_probability_distribution_without_saving(
        self, mock_orchestrator, sample_binary_data, tmp_path: Path
    ) -> None:
        """Test distribution plotting when save=False."""
        _, probas = sample_binary_data
        displayer = MedpipeDisplayer(orchestrator=mock_orchestrator)

        fig, _ = displayer.plot_probability_distribution(
            probas=probas,
            outcome="readmission",
            save=False,
            show=False,
        )

        expected_file = (
            tmp_path
            / "plots"
            / "readmission"
            / "readmission_probability_distribution.png"
        )
        assert isinstance(fig, Figure)
        assert not expected_file.exists()

    def test_plot_probability_distribution_2d_probas(
        self, mock_orchestrator, sample_binary_data
    ) -> None:
        """Test handling of 2D probability matrices passed to displayer method."""
        _, probas_1d = sample_binary_data
        probas_2d = np.column_stack((1 - probas_1d, probas_1d))
        displayer = MedpipeDisplayer(orchestrator=mock_orchestrator)

        fig, ax = displayer.plot_probability_distribution(
            probas=probas_2d,
            outcome="icu_admission",
            save=False,
            show=False,
        )

        assert isinstance(fig, Figure)
        assert "icu_admission" in ax.get_title().lower()

    @patch("matplotlib.pyplot.show")
    def test_plot_probability_distribution_show_flag(
        self, mock_show, mock_orchestrator, sample_binary_data
    ) -> None:
        """Test interactive plot display when show=True."""
        _, probas = sample_binary_data
        displayer = MedpipeDisplayer(orchestrator=mock_orchestrator)

        displayer.plot_probability_distribution(
            probas=probas,
            save=False,
            show=True,
        )

        mock_show.assert_called_once()


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

        _, ax = displayer.plot_roc_curve(
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


class TestComputePrecisionRecallData:
    """Tests for the internal `_compute_precision_recall_data` helper method."""

    def test_compute_pr_data_success_and_bootstraps(
        self, mock_orchestrator, sample_binary_data
    ) -> None:
        """Test calculation of PR curves, AP score, baseline, and bootstrap CIs."""
        y_true, probas = sample_binary_data
        displayer = MedpipeDisplayer(orchestrator=mock_orchestrator)

        (
            precision,
            recall,
            ap_score,
            baseline,
            lower_ci,
            upper_ci,
        ) = displayer._compute_precision_recall_data(
            y_true=y_true, probas=probas, n_bootstraps=20
        )

        assert isinstance(precision, np.ndarray)
        assert isinstance(recall, np.ndarray)
        assert isinstance(ap_score, float)
        assert 0.0 <= ap_score <= 1.0
        assert isinstance(baseline, float)
        assert isinstance(lower_ci, np.ndarray)
        assert isinstance(upper_ci, np.ndarray)

    def test_compute_pr_data_disabled_bootstraps(
        self, mock_orchestrator, sample_binary_data
    ) -> None:
        """Test that n_bootstraps <= 0 returns None for confidence interval arrays."""
        y_true, probas = sample_binary_data
        displayer = MedpipeDisplayer(orchestrator=mock_orchestrator)

        _, _, _, _, lower_ci, upper_ci = displayer._compute_precision_recall_data(
            y_true=y_true, probas=probas, n_bootstraps=0
        )

        assert lower_ci is None
        assert upper_ci is None


class TestPlotPrecisionRecallCurve:
    """Tests for the high-level `plot_precision_recall_curve` method."""

    def test_plot_pr_curve_success_and_saves(
        self, mock_orchestrator, sample_binary_data, tmp_path: Path
    ) -> None:
        """Test successful PR curve generation with figure artifact saving."""
        y_true, probas = sample_binary_data
        displayer = MedpipeDisplayer(orchestrator=mock_orchestrator)

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

    def test_plot_pr_curve_without_saving(
        self, mock_orchestrator, sample_binary_data, tmp_path: Path
    ) -> None:
        """Test PR curve rendering when save=False."""
        y_true, probas = sample_binary_data
        displayer = MedpipeDisplayer(orchestrator=mock_orchestrator)

        fig, _ = displayer.plot_precision_recall_curve(
            y_true=y_true,
            probas=probas,
            outcome="aki",
            save=False,
            show=False,
        )

        expected_file = tmp_path / "plots" / "aki" / "aki_pr_curve.png"
        assert isinstance(fig, Figure)
        assert not expected_file.exists()

    @patch("matplotlib.pyplot.show")
    def test_plot_pr_curve_show_flag(
        self, mock_show, mock_orchestrator, sample_binary_data
    ) -> None:
        """Test interactive figure display when show=True."""
        y_true, probas = sample_binary_data
        displayer = MedpipeDisplayer(orchestrator=mock_orchestrator)

        displayer.plot_precision_recall_curve(
            y_true=y_true,
            probas=probas,
            save=False,
            show=True,
        )

        mock_show.assert_called_once()


class TestComputeReliabilityData:
    """Tests for the internal `_compute_reliability_data` helper method."""

    def test_compute_reliability_data_success_and_bootstraps(
        self, mock_orchestrator, sample_binary_data
    ) -> None:
        """Test calculation of calibration points and bootstrap CIs."""
        y_true, probas = sample_binary_data
        displayer = MedpipeDisplayer(orchestrator=mock_orchestrator)

        (
            prob_true,
            prob_pred,
            lower_ci,
            upper_ci,
        ) = displayer._compute_reliability_data(
            y_true=y_true, probas=probas, n_bins=5, n_bootstraps=20
        )

        assert isinstance(prob_true, np.ndarray)
        assert isinstance(prob_pred, np.ndarray)
        assert len(prob_true) == len(prob_pred)
        assert isinstance(lower_ci, np.ndarray)
        assert isinstance(upper_ci, np.ndarray)

    def test_compute_reliability_data_disabled_bootstraps(
        self, mock_orchestrator, sample_binary_data
    ) -> None:
        """Test that n_bootstraps <= 0 returns None for confidence interval arrays."""
        y_true, probas = sample_binary_data
        displayer = MedpipeDisplayer(orchestrator=mock_orchestrator)

        _, _, lower_ci, upper_ci = displayer._compute_reliability_data(
            y_true=y_true, probas=probas, n_bins=5, n_bootstraps=0
        )

        assert lower_ci is None
        assert upper_ci is None


class TestPlotReliabilityDiagram:
    """Tests for the high-level `plot_reliability_diagram` method."""

    def test_plot_reliability_diagram_success_and_saves(
        self, mock_orchestrator, sample_binary_data, tmp_path: Path
    ) -> None:
        """Test successful reliability diagram generation with figure artifact saving."""
        y_true, probas = sample_binary_data
        displayer = MedpipeDisplayer(orchestrator=mock_orchestrator)

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

    def test_plot_reliability_diagram_without_saving(
        self, mock_orchestrator, sample_binary_data, tmp_path: Path
    ) -> None:
        """Test reliability diagram rendering when save=False."""
        y_true, probas = sample_binary_data
        displayer = MedpipeDisplayer(orchestrator=mock_orchestrator)

        fig, _ = displayer.plot_reliability_diagram(
            y_true=y_true,
            probas=probas,
            outcome="readmission",
            save=False,
            show=False,
        )

        expected_file = (
            tmp_path / "plots" / "readmission" / "readmission_reliability_diagram.png"
        )
        assert isinstance(fig, Figure)
        assert not expected_file.exists()

    @patch("matplotlib.pyplot.show")
    def test_plot_reliability_diagram_show_flag(
        self, mock_show, mock_orchestrator, sample_binary_data
    ) -> None:
        """Test interactive figure display when show=True."""
        y_true, probas = sample_binary_data
        displayer = MedpipeDisplayer(orchestrator=mock_orchestrator)

        displayer.plot_reliability_diagram(
            y_true=y_true,
            probas=probas,
            save=False,
            show=True,
        )

        mock_show.assert_called_once()

    @patch("splinecalib.SplineCalib")
    def test_compute_reliability_data_spline_strategy(
        self, mock_spline_cls, mock_orchestrator, sample_binary_data
    ) -> None:
        """Test reliability calculation using the splinecalib SplineCalib strategy."""
        y_true, probas = sample_binary_data
        displayer = MedpipeDisplayer(orchestrator=mock_orchestrator)

        mock_sc = MagicMock()
        mock_sc.calibrate.side_effect = lambda x: x * 0.95
        mock_spline_cls.return_value = mock_sc

        prob_true, prob_pred, lower_ci, upper_ci = displayer._compute_reliability_data(
            y_true=y_true, probas=probas, strategy="spline", n_bootstraps=10
        )

        assert mock_sc.fit.called
        assert len(prob_pred) == 100
        assert len(prob_true) == 100
        assert lower_ci is not None
        assert upper_ci is not None

    @patch("splinecalib.SplineCalib")
    def test_plot_reliability_diagram_spline_strategy(
        self, mock_spline_cls, mock_orchestrator, sample_binary_data
    ) -> None:
        """Test plot generation with spline calibration strategy."""
        y_true, probas = sample_binary_data
        displayer = MedpipeDisplayer(orchestrator=mock_orchestrator)

        mock_sc = MagicMock()
        mock_sc.calibrate.side_effect = lambda x: x * 0.95
        mock_spline_cls.return_value = mock_sc

        fig, ax = displayer.plot_reliability_diagram(
            y_true=y_true,
            probas=probas,
            outcome="icu_admission",
            strategy="spline",
            n_bootstraps=0,
            save=False,
            show=False,
        )

        assert isinstance(fig, Figure)
        # Line for model should not have marker points when strategy='spline'
        model_line = ax.get_lines()[1]
        assert model_line.get_marker() in ("", "None", None)


class TestPlotStrataHeatmap:
    """Tests for the high-level `plot_strata_heatmap` method."""

    def test_plot_strata_heatmap_success_and_saves(
        self, mock_orchestrator, tmp_path: Path
    ) -> None:
        """Test successful strata heatmap generation with figure artifact saving."""
        displayer = MedpipeDisplayer(orchestrator=mock_orchestrator)
        outcomes = ["Mortality", "Readmission"]
        strata = ["Male", "Female"]
        scores = np.array([0.85, 0.78])
        strata_scores = np.array([[0.83, 0.80], [0.87, 0.76]])

        fig, ax = displayer.plot_strata_heatmap(
            outcomes=outcomes,
            metric="auc",
            strata=strata,
            scores=scores,
            strata_scores=strata_scores,
            save=True,
            show=False,
        )

        expected_file = tmp_path / "plots" / "auc_strata_heatmap.png"
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert expected_file.exists()

    def test_plot_strata_heatmap_without_saving(
        self, mock_orchestrator, tmp_path: Path
    ) -> None:
        """Test heatmap plot rendering when save=False."""
        displayer = MedpipeDisplayer(orchestrator=mock_orchestrator)

        fig, _ = displayer.plot_strata_heatmap(
            outcomes=["Outcome1"],
            metric="brier",
            strata=["Stratum1"],
            scores=np.array([0.1]),
            strata_scores=np.array([[0.12]]),
            save=False,
            show=False,
        )

        expected_file = tmp_path / "plots" / "brier_strata_heatmap.png"
        assert isinstance(fig, Figure)
        assert not expected_file.exists()

    @patch("matplotlib.pyplot.show")
    def test_plot_strata_heatmap_show_flag(self, mock_show, mock_orchestrator) -> None:
        """Test interactive plot display when show=True."""
        displayer = MedpipeDisplayer(orchestrator=mock_orchestrator)

        displayer.plot_strata_heatmap(
            outcomes=["Outcome1"],
            metric="auc",
            strata=["Stratum1"],
            scores=np.array([0.8]),
            strata_scores=np.array([[0.85]]),
            save=False,
            show=True,
        )

        mock_show.assert_called_once()

    def test_plot_strata_heatmap_dimension_mismatch_raises(
        self, mock_orchestrator
    ) -> None:
        """Test that mismatched input matrix dimensions raise ValueError before plotting."""
        displayer = MedpipeDisplayer(orchestrator=mock_orchestrator)

        with pytest.raises(ValueError, match="matching column count"):
            displayer.plot_strata_heatmap(
                outcomes=["Mortality", "Readmission"],
                metric="auc",
                strata=["Male"],
                scores=np.array([0.80, 0.75]),
                strata_scores=np.array([[0.82]]),  # Missing 2nd column
                save=False,
            )


class TestComputeDcaData:
    """Tests for the internal `_compute_dca_data` helper method."""

    def test_compute_dca_data_success(
        self, mock_orchestrator, sample_binary_data
    ) -> None:
        """Test calculation of Net Benefit metrics without bootstrapping."""
        y_true, probas = sample_binary_data
        displayer = MedpipeDisplayer(orchestrator=mock_orchestrator)

        thresh, nb_model, nb_all = displayer._compute_dca_data(
            y_true=y_true, probas=probas
        )

        assert len(thresh) == 99
        assert len(nb_model) == 99
        assert len(nb_all) == 99


class TestPlotDcaCurve:
    """Tests for the high-level `plot_dca_curve` method."""

    def test_plot_dca_curve_success_and_saves(
        self, mock_orchestrator, sample_binary_data, tmp_path: Path
    ) -> None:
        """Test successful DCA plot generation with figure artifact saving."""
        y_true, probas = sample_binary_data
        displayer = MedpipeDisplayer(orchestrator=mock_orchestrator)

        fig, ax = displayer.plot_dca_curve(
            y_true=y_true,
            probas=probas,
            outcome="mortality",
            save=True,
            show=False,
        )

        expected_file = tmp_path / "plots" / "mortality" / "mortality_dca_curve.png"
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert expected_file.exists()

    def test_plot_dca_curve_without_saving(
        self, mock_orchestrator, sample_binary_data, tmp_path: Path
    ) -> None:
        """Test DCA plot rendering when save=False."""
        y_true, probas = sample_binary_data
        displayer = MedpipeDisplayer(orchestrator=mock_orchestrator)

        fig, _ = displayer.plot_dca_curve(
            y_true=y_true,
            probas=probas,
            outcome="readmission",
            save=False,
            show=False,
        )

        expected_file = tmp_path / "plots" / "readmission" / "readmission_dca_curve.png"
        assert isinstance(fig, Figure)
        assert not expected_file.exists()

    @patch("matplotlib.pyplot.show")
    def test_plot_dca_curve_show_flag(
        self, mock_show, mock_orchestrator, sample_binary_data
    ) -> None:
        """Test interactive figure display when show=True."""
        y_true, probas = sample_binary_data
        displayer = MedpipeDisplayer(orchestrator=mock_orchestrator)

        displayer.plot_dca_curve(
            y_true=y_true,
            probas=probas,
            save=False,
            show=True,
        )

        mock_show.assert_called_once()


class TestPlotAll:
    """Tests for the combined `plot_all` method."""

    def test_plot_all_success_and_saves_all_artifacts(
        self, mock_orchestrator, sample_binary_data, tmp_path: Path
    ) -> None:
        """Test that plot_all executes all 5 plotting methods and persists artifacts."""
        y_true, probas = sample_binary_data
        displayer = MedpipeDisplayer(orchestrator=mock_orchestrator)

        plots = displayer.plot_all(
            y_true=y_true,
            probas=probas,
            outcome="mortality",
            n_bootstraps=10,
            save=True,
            show=False,
        )

        # Check dictionary structure
        expected_keys = {"roc", "pr", "distribution", "reliability", "dca"}
        assert set(plots.keys()) == expected_keys

        for fig, ax in plots.values():
            assert isinstance(fig, Figure)
            assert isinstance(ax, Axes)

        # Check that all 5 artifact files were created on disk
        plot_dir = tmp_path / "plots" / "mortality"
        expected_files = [
            plot_dir / "mortality_roc_curve.png",
            plot_dir / "mortality_pr_curve.png",
            plot_dir / "mortality_probability_distribution.png",
            plot_dir / "mortality_reliability_diagram.png",
            plot_dir / "mortality_dca_curve.png",
        ]
        for file_path in expected_files:
            assert file_path.exists()

    def test_plot_all_without_saving(
        self, mock_orchestrator, sample_binary_data, tmp_path: Path
    ) -> None:
        """Test plot_all execution when save=False."""
        y_true, probas = sample_binary_data
        displayer = MedpipeDisplayer(orchestrator=mock_orchestrator)

        plots = displayer.plot_all(
            y_true=y_true,
            probas=probas,
            outcome="readmission",
            n_bootstraps=0,
            save=False,
            show=False,
        )

        assert len(plots) == 5
        plot_dir = tmp_path / "plots" / "readmission"
        assert not plot_dir.exists()
