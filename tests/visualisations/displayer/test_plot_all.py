"""
Tests for MedpipeDisplayer's combined `plot_all` method.
"""

from pathlib import Path

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from medpipe.visualisation.displayer import MedpipeDisplayer


class TestPlotAll:
    """Tests for the combined `plot_all` method."""

    def test_plot_all_success_and_saves_all_artifacts(
        self, mock_orchestrator, sample_binary_data, tmp_path: Path
    ) -> None:
        """Test that plot_all executes all 5 outcome plotting methods and
        persists artifacts."""
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

    def test_plot_all_no_save(
        self, mock_orchestrator, sample_binary_data, tmp_path: Path
    ) -> None:
        """Test that save=False skips artifact persistence for every plot."""
        y_true, probas = sample_binary_data
        displayer = MedpipeDisplayer(orchestrator=mock_orchestrator)

        displayer.plot_all(
            y_true=y_true,
            probas=probas,
            outcome="mortality",
            n_bootstraps=10,
            save=False,
            show=False,
        )

        assert not (tmp_path / "plots").exists()

    def test_plot_all_style_kwargs_applied_to_every_sub_plot(
        self, mock_orchestrator, sample_binary_data
    ) -> None:
        """Test that a style kwarg passed to plot_all (e.g. color) is
        forwarded consistently to every underlying sub-plot call, not just
        the first one."""
        y_true, probas = sample_binary_data
        displayer = MedpipeDisplayer(orchestrator=mock_orchestrator)

        plots = displayer.plot_all(
            y_true=y_true,
            probas=probas,
            n_bootstraps=5,
            save=False,
            show=False,
            color="#123456",
        )

        roc_ax = plots["roc"][1]
        pr_ax = plots["pr"][1]
        # Index 0 on both axes is a reference line (chance level / baseline)
        # drawn in a fixed color; index 1 is the main model curve.
        roc_color = roc_ax.get_lines()[1].get_color()
        pr_color = pr_ax.get_lines()[1].get_color()

        assert roc_color == "#123456"
        assert pr_color == "#123456"

    def test_plot_all_heatmaps_uses_formatted_stratum_labels(
        self, mock_orchestrator, sample_evaluations
    ) -> None:
        """Verify that rendered heatmaps contain formatted row labels
        (e.g. 'AGE: 18-50')."""
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
        assert "AGE: 18-50" in rendered_yticklabels
        assert "AGE: ≥ 51" in rendered_yticklabels
        assert "AGE: [51, 120]" not in rendered_yticklabels  # Raw string transformed
