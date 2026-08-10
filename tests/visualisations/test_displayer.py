"""Tests for the MedpipeDisplayer class."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from medpipe.metrics.registry import MetricRegistry, MetricSpec
from medpipe.utils.config import DisplayConfig, DisplayDefaultsConfig
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
    orchestrator.config.display = None
    return orchestrator


@pytest.fixture
def sample_binary_data():
    """Provides synthetic ground truth labels and predicted probabilities."""
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=100)
    probas = rng.uniform(0.0, 1.0, size=100)
    return y_true, probas


@pytest.fixture
def sample_evaluations():
    """Provides a realistic nested evaluation result dictionary across outcomes and strata."""
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


class TestNormalizePlotType:
    """Tests for MedpipeDisplayer._normalize_plot_type static helper."""

    @pytest.mark.parametrize(
        "input_type, expected",
        [
            ("calibration", "reliability"),
            ("reliability_diagram", "reliability"),
            ("pr", "precision_recall"),
            ("pr_curve", "precision_recall"),
            ("roc_curve", "roc"),
            ("distribution", "probability_distribution"),
            ("dist", "probability_distribution"),
            ("dca_curve", "dca"),
            ("roc", "roc"),
            ("dca", "dca"),
            ("custom_plot", "custom_plot"),
        ],
    )
    def test_normalize_plot_type_mappings(self, input_type: str, expected: str) -> None:
        """Test canonical resolution of plot names and aliases."""
        assert MedpipeDisplayer._normalize_plot_type(input_type) == expected

    def test_normalize_plot_type_case_insensitive(self) -> None:
        """Test that normalization handles uppercase inputs."""
        assert MedpipeDisplayer._normalize_plot_type("CALIBRATION") == "reliability"
        assert MedpipeDisplayer._normalize_plot_type("PR_CURVE") == "precision_recall"


class TestResolvePlotConfig:
    """Tests for MedpipeDisplayer._resolve_plot_config hierarchical resolution."""

    @pytest.fixture
    def mock_displayer(self) -> MedpipeDisplayer:
        """Create a displayer instance with a mock orchestrator configuration."""
        mock_orchestrator = MagicMock()
        mock_orchestrator.run_dir = MagicMock()

        display_config = DisplayConfig(
            defaults=DisplayDefaultsConfig(
                n_bootstraps=1000,
                save=True,
                show=False,
                n_bins=10,
                strategy="uniform",
            ),
            overrides={
                "reliability": {"n_bootstraps": 200, "strategy": "spline"},
                "probability_distribution": {"n_bins": 25},
            },
            outcome_overrides={
                "MORTALITY_30D": {
                    "calibration": {"n_bootstraps": 50, "strategy": "uniform"},
                }
            },
        )
        mock_orchestrator.config.display = display_config
        return MedpipeDisplayer(orchestrator=mock_orchestrator)

    def test_resolve_default_values(self, mock_displayer: MedpipeDisplayer) -> None:
        """Test fallback to global defaults when no overrides exist for plot type."""
        resolved = mock_displayer._resolve_plot_config(plot_type="roc")

        assert resolved["n_bootstraps"] == 1000
        assert resolved["save"] is True
        assert resolved["show"] is False

    def test_resolve_global_plot_override(
        self, mock_displayer: MedpipeDisplayer
    ) -> None:
        """Test that plot-level overrides take precedence over global defaults."""
        resolved = mock_displayer._resolve_plot_config(plot_type="reliability")

        assert resolved["n_bootstraps"] == 200
        assert resolved["strategy"] == "spline"
        assert resolved["save"] is True

    def test_resolve_alias_plot_override(
        self, mock_displayer: MedpipeDisplayer
    ) -> None:
        """Test that canonical alias normalization resolves plot-level overrides."""
        resolved = mock_displayer._resolve_plot_config(plot_type="calibration")

        assert resolved["n_bootstraps"] == 200
        assert resolved["strategy"] == "spline"

    def test_resolve_outcome_override(self, mock_displayer: MedpipeDisplayer) -> None:
        """Test that outcome-specific overrides take precedence over global plot overrides."""
        resolved = mock_displayer._resolve_plot_config(
            plot_type="calibration", outcome="MORTALITY_30D"
        )

        assert resolved["n_bootstraps"] == 50
        assert resolved["strategy"] == "uniform"

    def test_resolve_runtime_kwargs_precedence(
        self, mock_displayer: MedpipeDisplayer
    ) -> None:
        """Test that explicit runtime kwargs override all configuration levels."""
        resolved = mock_displayer._resolve_plot_config(
            plot_type="calibration",
            outcome="MORTALITY_30D",
            n_bootstraps=10,
            show=True,
        )

        assert resolved["n_bootstraps"] == 10
        assert resolved["strategy"] == "uniform"
        assert resolved["show"] is True

    def test_resolve_ignores_none_runtime_kwargs(
        self, mock_displayer: MedpipeDisplayer
    ) -> None:
        """Test that None values passed as runtime kwargs do not overwrite configured values."""
        resolved = mock_displayer._resolve_plot_config(
            plot_type="calibration",
            n_bootstraps=None,
        )

        assert resolved["n_bootstraps"] == 200

    def test_resolve_when_display_config_is_none(self, mock_orchestrator) -> None:
        """Test that _resolve_plot_config falls back to system defaults when display config is None."""
        displayer = MedpipeDisplayer(orchestrator=mock_orchestrator)
        resolved = displayer._resolve_plot_config(plot_type="roc")

        assert resolved["n_bootstraps"] == 1000
        assert resolved["save"] is True
        assert resolved["show"] is False
        assert resolved["n_bins"] == 10
        assert resolved["strategy"] == "uniform"


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


class TestFormatStratumLabel:
    """Tests for MedpipeDisplayer._format_stratum_label static helper."""

    @pytest.mark.parametrize(
        "stratum_var, cat_key, expected",
        [
            ("AGE", "[18, 50]", "AGE: 18–50"),
            ("AGE", "[51, 120]", "AGE: ≥ 51"),
            ("AGE", "[100, 150]", "AGE: ≥ 100"),
            ("SEX", "F", "SEX: F"),
            ("SEX", "M", "SEX: M"),
            ("ETHNICITY", "European", "ETHNICITY: European"),
            (
                "AGE",
                "[18, invalid]",
                "AGE: [18, invalid]",
            ),  # Fallback gracefully on syntax error
            ("STAGE", "Stage 1", "STAGE: Stage 1"),
        ],
    )
    def test_format_stratum_label_cases(
        self, stratum_var: str, cat_key: str, expected: str
    ) -> None:
        """Test formatting of interval strings, open-ended bounds, and categorical keys."""
        result = MedpipeDisplayer._format_stratum_label(stratum_var, cat_key)
        assert result == expected


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

    def test_plot_strata_heatmap_uses_metric_registry_display_name(
        self, mock_orchestrator
    ) -> None:
        """Test that plot_strata_heatmap looks up display_name from MetricRegistry."""
        custom_spec = MetricSpec(
            name="custom_metric",
            func=lambda y, p: 0.8,
            response_method="predict",
            display_name="Custom Metric Name",
        )
        MetricRegistry.register_spec(custom_spec)

        displayer = MedpipeDisplayer(orchestrator=mock_orchestrator)
        fig, ax = displayer.plot_strata_heatmap(
            outcomes=["ANY_COMP"],
            metric="custom_metric",
            strata=["SEX: F"],
            scores=np.array([0.80]),
            strata_scores=np.array([[0.82]]),
            save=False,
            show=False,
        )

        assert isinstance(fig, Figure)
        assert "Custom Metric Name" in ax.get_title()

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
                strata_scores=np.array([[0.82]]),
                save=False,
            )


class TestPlotAllHeatmaps:
    """Tests for the cross-outcome `plot_all_heatmaps` method."""

    def test_plot_all_heatmaps_success_and_saves(
        self, mock_orchestrator, sample_evaluations, tmp_path: Path
    ) -> None:
        """Test plot_all_heatmaps correctly parses nested evaluations and creates heatmap figures."""
        displayer = MedpipeDisplayer(orchestrator=mock_orchestrator)

        heatmap_plots = displayer.plot_all_heatmaps(
            evaluations=sample_evaluations,
            save=True,
            show=False,
        )

        assert "roc_auc" in heatmap_plots
        assert "log_loss" in heatmap_plots

        for fig, ax in heatmap_plots.values():
            assert isinstance(fig, Figure)
            assert isinstance(ax, Axes)

        # Verify saved artifacts on disk
        assert (tmp_path / "plots" / "roc_auc_strata_heatmap.png").exists()
        assert (tmp_path / "plots" / "log_loss_strata_heatmap.png").exists()

    def test_plot_all_heatmaps_without_saving(
        self, mock_orchestrator, sample_evaluations, tmp_path: Path
    ) -> None:
        """Test plot_all_heatmaps execution when save=False."""
        displayer = MedpipeDisplayer(orchestrator=mock_orchestrator)

        heatmap_plots = displayer.plot_all_heatmaps(
            evaluations=sample_evaluations,
            save=False,
            show=False,
        )

        assert len(heatmap_plots) == 2
        assert not (tmp_path / "plots" / "roc_auc_strata_heatmap.png").exists()

    def test_plot_all_heatmaps_custom_metrics_list(
        self, mock_orchestrator, sample_evaluations
    ) -> None:
        """Test filtering plot_all_heatmaps to a specified subset of metrics."""
        displayer = MedpipeDisplayer(orchestrator=mock_orchestrator)

        heatmap_plots = displayer.plot_all_heatmaps(
            evaluations=sample_evaluations,
            metrics=["roc_auc"],
            save=False,
            show=False,
        )

        assert list(heatmap_plots.keys()) == ["roc_auc"]

    def test_plot_all_heatmaps_empty_evaluations(self, mock_orchestrator) -> None:
        """Test plot_all_heatmaps gracefully returns an empty dictionary when passed empty evaluations."""
        displayer = MedpipeDisplayer(orchestrator=mock_orchestrator)
        heatmap_plots = displayer.plot_all_heatmaps(evaluations={}, save=False)

        assert heatmap_plots == {}

    def test_plot_all_heatmaps_no_strata_found(self, mock_orchestrator) -> None:
        """Test plot_all_heatmaps returns empty dict if evaluation contains no subgroup strata."""
        evals_without_strata = {
            "MORTALITY_30D": {
                "overall": {"roc_auc": {"point_estimate": 0.8}},
                "strata": {},
            }
        }
        displayer = MedpipeDisplayer(orchestrator=mock_orchestrator)

        heatmap_plots = displayer.plot_all_heatmaps(
            evaluations=evals_without_strata, save=False
        )
        assert heatmap_plots == {}

    @patch("matplotlib.pyplot.show")
    def test_plot_all_heatmaps_show_flag(
        self, mock_show, mock_orchestrator, sample_evaluations
    ) -> None:
        """Test interactive display when show=True."""
        displayer = MedpipeDisplayer(orchestrator=mock_orchestrator)

        displayer.plot_all_heatmaps(
            evaluations=sample_evaluations,
            save=False,
            show=True,
        )

        # Expected 2 show calls (one per metric)
        assert mock_show.call_count == 2


class TestPlotAll:
    """Tests for the combined `plot_all` method."""

    def test_plot_all_success_and_saves_all_artifacts(
        self, mock_orchestrator, sample_binary_data, tmp_path: Path
    ) -> None:
        """Test that plot_all executes all 5 outcome plotting methods and persists artifacts."""
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

        expected_keys = {"roc", "pr", "distribution", "reliability", "dca"}
        assert set(plots.keys()) == expected_keys

        for fig, ax in plots.values():
            assert isinstance(fig, Figure)
            assert isinstance(ax, Axes)

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

    def test_plot_all_heatmaps_uses_formatted_stratum_labels(
        self, mock_orchestrator, sample_evaluations
    ) -> None:
        """Verify that rendered heatmaps contain formatted row labels
        (e.g. 'AGE: 18–50')."""
        displayer = MedpipeDisplayer(orchestrator=mock_orchestrator)

        heatmap_plots = displayer.plot_all_heatmaps(
            evaluations=sample_evaluations,
            metrics=["roc_auc"],
            save=False,
            show=False,
        )

        _, ax = heatmap_plots["roc_auc"]

        # Y-axis labels include "All strata" followed by the formatted row labels
        rendered_yticklabels = [label.get_text() for label in ax.get_yticklabels()]

        assert "All strata" in rendered_yticklabels
        assert "SEX: F" in rendered_yticklabels
        assert "SEX: M" in rendered_yticklabels
        assert "AGE: 18–50" in rendered_yticklabels
        assert "AGE: ≥ 51" in rendered_yticklabels
        assert "AGE: [51, 120]" not in rendered_yticklabels  # Raw string transformed
