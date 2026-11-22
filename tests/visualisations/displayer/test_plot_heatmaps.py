"""
Tests for MedpipeClassifierDisplayer's heatmap-plotting methods: plot_strata_heatmap
and plot_all_heatmaps.
"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from medpipe.metrics.registry import MetricRegistry, MetricSpec
from medpipe.visualisation.displayer import MedpipeClassifierDisplayer


class TestStrataHeatmapScale:
    """Tests of the colour scale of `plot_strata_heatmap`."""

    @staticmethod
    def _color_limit(displayer, metric: str, scores, strata_scores, **kwargs) -> float:
        strata_scores = np.array(strata_scores)
        _, ax = displayer.plot_strata_heatmap(
            outcomes=[f"O{i}" for i in range(len(scores))],
            metric=metric,
            strata=[f"s{i}" for i in range(strata_scores.shape[0])],
            scores=np.array(scores),
            strata_scores=strata_scores,
            save=False,
            show=False,
            **kwargs,
        )
        return float(ax.images[0].get_clim()[1])

    @pytest.mark.parametrize("metric", ["auc", "rmse"])
    def test_scale_is_95th_percentile_of_strata_deltas(
        self, mock_orchestrator, metric: str
    ) -> None:
        """Test that the colour max is the 95th percentile of the strata deltas,
        whatever the metric's units."""
        displayer = MedpipeClassifierDisplayer(orchestrator=mock_orchestrator)

        limit = self._color_limit(
            displayer, metric, [10.0, 12.0], [[16.0, 11.0], [9.0, 20.0]]
        )

        assert limit == pytest.approx(np.percentile([6.0, 1.0, 1.0, 8.0], 95))

    def test_outlier_does_not_set_scale(self, mock_orchestrator) -> None:
        """Test that a single outlying delta is clipped instead of setting the
        colour max."""
        displayer = MedpipeClassifierDisplayer(orchestrator=mock_orchestrator)
        deltas = np.full((10, 10), 0.01)
        deltas[0, 0] = 1.0

        limit = self._color_limit(displayer, "auc", np.zeros(10), deltas)

        assert limit == pytest.approx(0.01)

    def test_custom_percentile(self, mock_orchestrator) -> None:
        """Test that `vmax_percentile` selects the percentile used."""
        displayer = MedpipeClassifierDisplayer(orchestrator=mock_orchestrator)

        limit = self._color_limit(
            displayer,
            "auc",
            [0.0, 0.0],
            [[0.1, 0.2], [0.3, 0.4]],
            vmax_percentile=50,
        )

        assert limit == pytest.approx(0.25)

    def test_ici_scale_uses_percent_values(self, mock_orchestrator) -> None:
        """Test that the ICI colour max is computed on the display-scaled
        (x100) values."""
        displayer = MedpipeClassifierDisplayer(orchestrator=mock_orchestrator)

        limit = self._color_limit(displayer, "ici", [0.01], [[0.012], [0.012]])

        assert limit == pytest.approx(0.2)

    def test_mostly_zero_deltas_fall_back_to_max(self, mock_orchestrator) -> None:
        """Test that a zero percentile falls back to the largest delta."""
        displayer = MedpipeClassifierDisplayer(orchestrator=mock_orchestrator)
        strata_scores = np.zeros((30, 1))
        strata_scores[0, 0] = 0.3

        limit = self._color_limit(displayer, "auc", [0.0], strata_scores)

        assert limit == pytest.approx(0.3)

    def test_no_differences_do_not_produce_zero_width_scale(
        self, mock_orchestrator
    ) -> None:
        """Test that identical scores do not produce a zero-width scale."""
        displayer = MedpipeClassifierDisplayer(orchestrator=mock_orchestrator)

        limit = self._color_limit(
            displayer, "rmse", [10.0, 12.0], [[10.0, 12.0], [10.0, 12.0]]
        )

        assert limit == pytest.approx(1.0)


class TestPlotStrataHeatmap:
    """Tests for the high-level `plot_strata_heatmap` method."""

    def test_plot_strata_heatmap_success_and_saves(
        self, mock_orchestrator, tmp_path: Path
    ) -> None:
        """Test successful strata heatmap generation with figure artifact saving."""
        displayer = MedpipeClassifierDisplayer(orchestrator=mock_orchestrator)
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

        # patch.dict restores MetricRegistry._registry to its prior state on
        # exit, so this registration doesn't leak into other tests.
        with patch.dict(MetricRegistry._registry, {}, clear=False):
            MetricRegistry.register_spec(custom_spec)

            displayer = MedpipeClassifierDisplayer(orchestrator=mock_orchestrator)
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

    def test_plot_strata_heatmap_applies_spec_display_scale(
        self, mock_orchestrator
    ) -> None:
        """Test that values are multiplied by the spec's display_scale and the
        factor is noted in the title and colour-bar label."""
        custom_spec = MetricSpec(
            name="tiny_metric",
            func=lambda y, p: 0.001,
            response_method="predict",
            display_name="Tiny",
            display_scale=1000,
        )

        with patch.dict(MetricRegistry._registry, {}, clear=False):
            MetricRegistry.register_spec(custom_spec)

            displayer = MedpipeClassifierDisplayer(orchestrator=mock_orchestrator)
            fig, ax = displayer.plot_strata_heatmap(
                outcomes=["A"],
                metric="tiny_metric",
                strata=["x"],
                scores=np.array([0.001]),
                strata_scores=np.array([[0.003]]),
                save=False,
                show=False,
            )

        assert r"Tiny $\times$ 1000" in ax.get_title()
        assert r"$\times$ 1000" in fig.axes[-1].get_ylabel()
        np.testing.assert_allclose(ax.images[1].get_array(), [[2.0]])
        assert [t.get_text() for t in ax.texts] == ["1.00", "3.00"]

    def test_plot_strata_heatmap_unscaled_metric_has_no_factor_in_title(
        self, mock_orchestrator
    ) -> None:
        """Test that a metric with display_scale 1 gets no scale factor label."""
        displayer = MedpipeClassifierDisplayer(orchestrator=mock_orchestrator)

        _, ax = displayer.plot_strata_heatmap(
            outcomes=["A"],
            metric="roc_auc",
            strata=["x"],
            scores=np.array([0.8]),
            strata_scores=np.array([[0.7]]),
            save=False,
            show=False,
        )

        assert r"$\times$" not in ax.get_title()

    def test_plot_strata_heatmap_unregistered_metric_falls_back_to_upper(
        self, mock_orchestrator
    ) -> None:
        """Test that a metric name absent from MetricRegistry falls back to
        an upper-cased, underscore-replaced display name rather than raising."""
        displayer = MedpipeClassifierDisplayer(orchestrator=mock_orchestrator)

        _fig, ax = displayer.plot_strata_heatmap(
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
        displayer = MedpipeClassifierDisplayer(orchestrator=mock_orchestrator)

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
        displayer = MedpipeClassifierDisplayer(orchestrator=mock_orchestrator)

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
        displayer = MedpipeClassifierDisplayer(orchestrator=mock_orchestrator)

        _, ax = displayer.plot_strata_heatmap(
            outcomes=["Outcome1"],
            metric="ici",
            strata=["Stratum1"],
            scores=np.array([0.1]),
            strata_scores=np.array([[0.12]]),
            save=False,
            show=False,
        )

        assert r"$\times$ 100" in ax.get_title()

    def test_plot_strata_heatmap_not_2d_strata_scores_raises(
        self, mock_orchestrator
    ) -> None:
        """Test that a non-2D strata_scores array raises ValueError before
        any plotting is attempted."""
        displayer = MedpipeClassifierDisplayer(orchestrator=mock_orchestrator)

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
        displayer = MedpipeClassifierDisplayer(orchestrator=mock_orchestrator)

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
        displayer = MedpipeClassifierDisplayer(orchestrator=mock_orchestrator)

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
        displayer = MedpipeClassifierDisplayer(orchestrator=mock_orchestrator)

        with pytest.raises(ValueError, match="matching column count"):
            displayer.plot_strata_heatmap(
                outcomes=["Mortality"],
                metric="auc",
                strata=["Male"],
                scores=np.array([0.80, 0.75]),  # 2 scores
                strata_scores=np.array([[0.82]]),  # Only 1 column
                save=False,
            )

    def test_plot_strata_heatmap_strata_groups_mismatch_raises(
        self, mock_orchestrator
    ) -> None:
        """Edge case: raise ValueError when strata_groups do not sum to the
        number of strata."""
        displayer = MedpipeClassifierDisplayer(orchestrator=mock_orchestrator)

        with pytest.raises(ValueError, match="strata_groups must sum"):
            displayer.plot_strata_heatmap(
                outcomes=["A"],
                metric="auc",
                strata=["x", "y", "z"],
                scores=np.array([0.8]),
                strata_scores=np.array([[0.7], [0.9], [0.8]]),
                strata_groups=[1, 1],
                save=False,
                show=False,
            )


class TestPlotAllHeatmaps:
    """Tests for the cross-outcome `plot_all_heatmaps` method."""

    def test_plot_all_heatmaps_success_and_saves(
        self, mock_orchestrator, sample_evaluations, tmp_path: Path
    ) -> None:
        """Test plot_all_heatmaps correctly parses nested evaluations and
        creates heatmap figures."""
        displayer = MedpipeClassifierDisplayer(orchestrator=mock_orchestrator)

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

    def test_plot_all_heatmaps_groups_rows_by_stratum_variable(
        self, mock_orchestrator, sample_evaluations
    ) -> None:
        """Test that each stratification variable is drawn as its own image
        block, plus one block for the 'All strata' row."""
        displayer = MedpipeClassifierDisplayer(orchestrator=mock_orchestrator)
        strata = sample_evaluations[next(iter(sample_evaluations))]["strata"]

        heatmap_plots = displayer.plot_all_heatmaps(
            evaluations=sample_evaluations, save=False, show=False
        )

        _, ax = heatmap_plots["roc_auc"]
        assert len(ax.images) == 1 + len(strata)

    def test_plot_all_heatmaps_without_saving(
        self, mock_orchestrator, sample_evaluations, tmp_path: Path
    ) -> None:
        """Test plot_all_heatmaps execution when save=False."""
        displayer = MedpipeClassifierDisplayer(orchestrator=mock_orchestrator)

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
        displayer = MedpipeClassifierDisplayer(orchestrator=mock_orchestrator)

        heatmap_plots = displayer.plot_all_heatmaps(
            evaluations=sample_evaluations,
            metrics=["roc_auc"],
            save=False,
            show=False,
        )

        assert list(heatmap_plots.keys()) == ["roc_auc"]

    def test_plot_all_heatmaps_empty_evaluations(self, mock_orchestrator) -> None:
        """Test plot_all_heatmaps gracefully returns an empty dictionary when
        passed empty evaluations."""
        displayer = MedpipeClassifierDisplayer(orchestrator=mock_orchestrator)
        heatmap_plots = displayer.plot_all_heatmaps(evaluations={}, save=False)

        assert heatmap_plots == {}

    def test_plot_all_heatmaps_no_strata_found(self, mock_orchestrator) -> None:
        """Test plot_all_heatmaps returns empty dict if evaluation contains no
        subgroup strata."""
        evals_without_strata = {
            "MORTALITY_30D": {
                "overall": {"roc_auc": {"point_estimate": 0.8}},
                "strata": {},
            }
        }
        displayer = MedpipeClassifierDisplayer(orchestrator=mock_orchestrator)

        heatmap_plots = displayer.plot_all_heatmaps(
            evaluations=evals_without_strata, save=False
        )
        assert heatmap_plots == {}

    @patch("matplotlib.pyplot.show")
    def test_plot_all_heatmaps_show_flag(
        self, mock_show, mock_orchestrator, sample_evaluations
    ) -> None:
        """Test interactive display when show=True."""
        displayer = MedpipeClassifierDisplayer(orchestrator=mock_orchestrator)

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
        displayer = MedpipeClassifierDisplayer(orchestrator=mock_orchestrator)

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
