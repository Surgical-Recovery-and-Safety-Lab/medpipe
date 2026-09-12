"""
Tests for MedpipeDisplayer's heatmap-plotting methods: plot_strata_heatmap
and plot_all_heatmaps.
"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from medpipe.metrics.registry import MetricRegistry, MetricSpec
from medpipe.visualisation.displayer import MedpipeDisplayer


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

    def test_plot_strata_heatmap_unregistered_metric_falls_back_to_upper(
        self, mock_orchestrator
    ) -> None:
        """Test that a metric name absent from MetricRegistry falls back to
        an upper-cased, underscore-replaced display name rather than raising."""
        displayer = MedpipeDisplayer(orchestrator=mock_orchestrator)

        fig, ax = displayer.plot_strata_heatmap(
            outcomes=["ANY_COMP"],
            metric="totally_unregistered_metric",
            strata=["SEX: F"],
            scores=np.array([0.80]),
            strata_scores=np.array([[0.82]]),
            save=False,
            show=False,
        )

        assert "TOTALLY UNREGISTERED METRIC" in ax.get_title()

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

    def test_plot_strata_heatmap_ici_metric_scales_to_percent(
        self, mock_orchestrator
    ) -> None:
        """Test that the 'ici' metric is scaled by 100 and labeled as a
        percentage, unlike other metrics."""
        displayer = MedpipeDisplayer(orchestrator=mock_orchestrator)

        _, ax = displayer.plot_strata_heatmap(
            outcomes=["Outcome1"],
            metric="ici",
            strata=["Stratum1"],
            scores=np.array([0.1]),
            strata_scores=np.array([[0.12]]),
            save=False,
            show=False,
        )

        assert "(%)" in ax.get_title()

    def test_plot_strata_heatmap_not_2d_strata_scores_raises(
        self, mock_orchestrator
    ) -> None:
        """Test that a non-2D strata_scores array raises ValueError before
        any plotting is attempted."""
        displayer = MedpipeDisplayer(orchestrator=mock_orchestrator)

        with pytest.raises(ValueError, match="must be a 2D array"):
            displayer.plot_strata_heatmap(
                outcomes=["Mortality"],
                metric="auc",
                strata=["Male"],
                scores=np.array([0.80]),
                strata_scores=np.array([0.82]),  # 1D, not 2D
                save=False,
            )

    def test_plot_strata_heatmap_strata_row_mismatch_raises(
        self, mock_orchestrator
    ) -> None:
        """Test that a strata list whose length doesn't match
        strata_scores' row count raises ValueError."""
        displayer = MedpipeDisplayer(orchestrator=mock_orchestrator)

        with pytest.raises(ValueError, match="matching row count"):
            displayer.plot_strata_heatmap(
                outcomes=["Mortality"],
                metric="auc",
                strata=["Male", "Female"],  # 2 strata
                scores=np.array([0.80]),
                strata_scores=np.array([[0.82]]),  # Only 1 row
                save=False,
            )

    def test_plot_strata_heatmap_outcomes_column_mismatch_raises(
        self, mock_orchestrator
    ) -> None:
        """Test that an outcomes list whose length doesn't match
        strata_scores' column count raises ValueError."""
        displayer = MedpipeDisplayer(orchestrator=mock_orchestrator)

        with pytest.raises(ValueError, match="matching column count"):
            displayer.plot_strata_heatmap(
                outcomes=["Mortality", "Readmission"],
                metric="auc",
                strata=["Male"],
                scores=np.array([0.80, 0.75]),
                strata_scores=np.array([[0.82]]),  # Only 1 column
                save=False,
            )

    def test_plot_strata_heatmap_scores_column_mismatch_raises(
        self, mock_orchestrator
    ) -> None:
        """Test that a scores array whose length doesn't match
        strata_scores' column count raises ValueError, once the outcomes
        and row checks have already passed."""
        displayer = MedpipeDisplayer(orchestrator=mock_orchestrator)

        with pytest.raises(ValueError, match="matching column count"):
            displayer.plot_strata_heatmap(
                outcomes=["Mortality"],
                metric="auc",
                strata=["Male"],
                scores=np.array([0.80, 0.75]),  # 2 scores
                strata_scores=np.array([[0.82]]),  # Only 1 column
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

    def test_plot_all_heatmaps_unregistered_metric_falls_back_to_upper(
        self, mock_orchestrator, sample_evaluations
    ) -> None:
        """Test that requesting a metric absent from MetricRegistry falls
        back to an upper-cased display name rather than raising, mirroring
        plot_strata_heatmap's own fallback."""
        displayer = MedpipeDisplayer(orchestrator=mock_orchestrator)

        heatmap_plots = displayer.plot_all_heatmaps(
            evaluations=sample_evaluations,
            metrics=["totally_unregistered_metric"],
            save=False,
            show=False,
        )

        # The metric is still processed even though it isn't in the
        # evaluations' "overall" dict — scores default to NaN but the
        # heatmap still renders without raising.
        assert list(heatmap_plots.keys()) == ["totally_unregistered_metric"]
        _, ax = heatmap_plots["totally_unregistered_metric"]
        assert "TOTALLY UNREGISTERED METRIC" in ax.get_title()
