"""
Integration tests for MedpipeRegressorDisplayer, exercising a real fitted
OrdBoostRegressor end-to-end: coverage, sharpness, Winkler score (all
bootstrapped), marginal calibration, and PIT histogram diagnostics.
"""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from ordboost import OrdBoostRegressor

from medpipe.pipeline.estimator import DistributionalPipeline
from medpipe.visualisation.displayer import MedpipeRegressorDisplayer


@pytest.fixture
def fitted_ordboost_dist_and_data():
    """A real, fitted OrdBoostRegressor's predictive distribution, its
    fitted bin mapper, and held-out ground truth values."""
    rng = np.random.default_rng(7)
    n_train = 150
    X_train = pd.DataFrame(
        {
            "feature1": rng.standard_normal(n_train),
            "feature2": rng.standard_normal(n_train),
        }
    )
    y_train = np.exp(X_train["feature1"].to_numpy() * 0.5) + rng.normal(
        0.0, 0.5, n_train
    )

    n_test = 60
    X_test = pd.DataFrame(
        {
            "feature1": rng.standard_normal(n_test),
            "feature2": rng.standard_normal(n_test),
        }
    )
    y_test = np.exp(X_test["feature1"].to_numpy() * 0.5) + rng.normal(
        0.0, 0.5, n_test
    )

    pipeline = DistributionalPipeline(
        [
            (
                "regressor",
                OrdBoostRegressor(
                    n_bins=10,
                    mapper="quantile",
                    learning_rate=0.1,
                    max_iter=20,
                    random_state=42,
                ),
            )
        ]
    )
    pipeline.fit(X_train, y_train)

    dist = pipeline.predict_dist(X_test)
    mapper = pipeline.named_steps["regressor"].mapper_

    return y_test, dist, mapper


@pytest.fixture
def mock_orchestrator(tmp_path: Path) -> MagicMock:
    """Provides a mock MedpipeOrchestrator with a temporary run_dir and no
    display configuration (fallback defaults path)."""
    orchestrator = MagicMock()
    orchestrator.run_dir = tmp_path
    orchestrator.config.display = None
    return orchestrator


class TestPlotCoverage:
    """Tests for MedpipeRegressorDisplayer.plot_coverage."""

    def test_plot_coverage_returns_finite_bootstrapped_curve(
        self, mock_orchestrator, fitted_ordboost_dist_and_data
    ) -> None:
        """Test that coverage is computed with finite bootstrap CIs, and
        empirical coverage stays within [0, 100]."""
        y_true, dist, _mapper = fitted_ordboost_dist_and_data
        displayer = MedpipeRegressorDisplayer(orchestrator=mock_orchestrator)

        fig, ax = displayer.plot_coverage(
            y_true=y_true, dist=dist, n_bootstraps=20, save=False, show=False
        )

        assert fig is not None
        assert ax is not None
        lines = [line for line in ax.get_lines() if len(line.get_ydata()) == 10]
        assert len(lines) >= 1
        for line in lines:
            ydata = np.asarray(line.get_ydata())
            assert np.all(np.isfinite(ydata))
            assert np.all((ydata >= 0) & (ydata <= 100))

    def test_plot_coverage_disables_bootstrap_with_zero(
        self, mock_orchestrator, fitted_ordboost_dist_and_data
    ) -> None:
        """Test that n_bootstraps=0 skips CI computation without erroring."""
        y_true, dist, _mapper = fitted_ordboost_dist_and_data
        displayer = MedpipeRegressorDisplayer(orchestrator=mock_orchestrator)

        data = displayer._compute_coverage_sharpness_winkler_data(
            y_true=y_true, dist=dist, coverage_levels=np.arange(5, 100, 10),
            n_bootstraps=0,
        )

        assert data["coverage"]["lower_ci"] is None
        assert data["coverage"]["upper_ci"] is None


class TestPlotSharpness:
    """Tests for MedpipeRegressorDisplayer.plot_sharpness."""

    def test_plot_sharpness_values_are_positive_and_finite(
        self, mock_orchestrator, fitted_ordboost_dist_and_data
    ) -> None:
        """Test that sharpness (mean interval width) is positive and finite
        across coverage levels."""
        y_true, dist, _mapper = fitted_ordboost_dist_and_data
        displayer = MedpipeRegressorDisplayer(orchestrator=mock_orchestrator)

        data = displayer._compute_coverage_sharpness_winkler_data(
            y_true=y_true, dist=dist, coverage_levels=np.arange(5, 100, 10),
            n_bootstraps=20,
        )

        sharp = data["sharpness"]["point"]
        assert np.all(np.isfinite(sharp))
        assert np.all(sharp > 0)

        fig, ax = displayer.plot_sharpness(
            y_true=y_true, dist=dist, n_bootstraps=20, save=False, show=False
        )
        assert fig is not None
        assert ax is not None

    def test_sharpness_increases_with_nominal_coverage(
        self, mock_orchestrator, fitted_ordboost_dist_and_data
    ) -> None:
        """Test that wider nominal coverage requires wider (or equal)
        intervals, a basic sanity check on the coverage/alpha convention."""
        y_true, dist, _mapper = fitted_ordboost_dist_and_data
        displayer = MedpipeRegressorDisplayer(orchestrator=mock_orchestrator)

        data = displayer._compute_coverage_sharpness_winkler_data(
            y_true=y_true, dist=dist, coverage_levels=np.array([10.0, 90.0]),
            n_bootstraps=0,
        )

        sharp = data["sharpness"]["point"]
        assert sharp[1] >= sharp[0]


class TestPlotWinkler:
    """Tests for MedpipeRegressorDisplayer.plot_winkler."""

    def test_plot_winkler_returns_finite_curve(
        self, mock_orchestrator, fitted_ordboost_dist_and_data
    ) -> None:
        """Test that the Winkler score curve is finite across coverage
        levels and the figure renders."""
        y_true, dist, _mapper = fitted_ordboost_dist_and_data
        displayer = MedpipeRegressorDisplayer(orchestrator=mock_orchestrator)

        fig, ax = displayer.plot_winkler(
            y_true=y_true, dist=dist, n_bootstraps=20, save=False, show=False
        )

        assert fig is not None
        assert ax is not None


class TestPlotMarginalCalibration:
    """Tests for MedpipeRegressorDisplayer.plot_marginal_calibration."""

    def test_plot_marginal_calibration_curve_is_bounded(
        self, mock_orchestrator, fitted_ordboost_dist_and_data
    ) -> None:
        """Test that the marginal calibration difference stays within
        [-1, 1] (difference of two CDFs) and the figure renders."""
        y_true, dist, _mapper = fitted_ordboost_dist_and_data
        displayer = MedpipeRegressorDisplayer(orchestrator=mock_orchestrator)

        _grid_y, diff = displayer._compute_marginal_calibration_data(
            y_true=y_true, dist=dist
        )

        assert np.all(np.isfinite(diff))
        assert np.all(diff >= -1.0) and np.all(diff <= 1.0)

        fig, ax = displayer.plot_marginal_calibration(
            y_true=y_true, dist=dist, save=False, show=False
        )
        assert fig is not None
        assert ax is not None


class TestPlotPitHistogram:
    """Tests for MedpipeRegressorDisplayer.plot_pit_histogram."""

    def test_plot_pit_histogram_with_mapper_succeeds(
        self, mock_orchestrator, fitted_ordboost_dist_and_data
    ) -> None:
        """Test that PIT diagnostics compute successfully when the fitted
        OrdBoost mapper is provided, and the alpha score is finite."""
        y_true, dist, mapper = fitted_ordboost_dist_and_data
        displayer = MedpipeRegressorDisplayer(orchestrator=mock_orchestrator)

        bin_centres, hist_values, alpha_score = displayer._compute_pit_histogram_data(
            y_true=y_true, dist=dist, mapper=mapper, n_bins=20
        )

        assert len(bin_centres) == len(hist_values) == 20
        assert np.isfinite(alpha_score)

        fig, ax = displayer.plot_pit_histogram(
            y_true=y_true, dist=dist, mapper=mapper, save=False, show=False
        )
        assert fig is not None
        assert ax is not None

    def test_plot_pit_histogram_without_mapper_raises_clear_error(
        self, mock_orchestrator, fitted_ordboost_dist_and_data
    ) -> None:
        """Test that omitting the mapper (e.g. an NGBoost-wrapped
        distribution, not yet supported) raises a clear ValueError instead
        of crashing deep inside ordboost.metrics."""
        y_true, dist, _mapper = fitted_ordboost_dist_and_data
        displayer = MedpipeRegressorDisplayer(orchestrator=mock_orchestrator)

        with pytest.raises(ValueError, match=r"mapper.*required"):
            displayer.plot_pit_histogram(
                y_true=y_true, dist=dist, mapper=None, save=False, show=False
            )


class TestPlotAll:
    """Tests for MedpipeRegressorDisplayer.plot_all."""

    def test_plot_all_renders_all_five_figures(
        self, mock_orchestrator, fitted_ordboost_dist_and_data
    ) -> None:
        """Test that plot_all produces all five distributional diagnostic
        figures in one call."""
        y_true, dist, mapper = fitted_ordboost_dist_and_data
        displayer = MedpipeRegressorDisplayer(orchestrator=mock_orchestrator)

        plots = displayer.plot_all(
            y_true=y_true,
            dist=dist,
            mapper=mapper,
            n_bootstraps=10,
            save=False,
            show=False,
        )

        assert set(plots.keys()) == {
            "coverage",
            "sharpness",
            "winkler",
            "marginal_calibration",
            "pit_histogram",
        }
        for fig, ax in plots.values():
            assert fig is not None
            assert ax is not None

    def test_plot_all_saves_five_files_to_run_dir(
        self, mock_orchestrator, fitted_ordboost_dist_and_data, tmp_path: Path
    ) -> None:
        """Test that plot_all with save=True writes all five figures to the
        run directory's plots/ folder."""
        y_true, dist, mapper = fitted_ordboost_dist_and_data
        displayer = MedpipeRegressorDisplayer(orchestrator=mock_orchestrator)

        displayer.plot_all(
            y_true=y_true,
            dist=dist,
            mapper=mapper,
            n_bootstraps=5,
            outcome="LOS_DAYS",
            save=True,
            show=False,
        )

        plot_dir = tmp_path / "plots" / "LOS_DAYS"
        expected_files = {
            "LOS_DAYS_coverage.png",
            "LOS_DAYS_sharpness.png",
            "LOS_DAYS_winkler.png",
            "LOS_DAYS_marginal_calibration.png",
            "LOS_DAYS_pit_histogram.png",
        }
        actual_files = {p.name for p in plot_dir.iterdir()}
        assert expected_files <= actual_files
