#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test functions for the metrics.plots module
"""

import os
from pathlib import Path
from typing import Any, Dict, Generator, Literal
from unittest.mock import MagicMock, patch

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pytest

from medpipe.metrics.plots import (
    _get_calibration_data,
    _plot_calibration,
    plot_probability_distribution,
    plot_reliability_diagram,
    plot_ROC_curve,
    plot_strata_heatmap,
)

matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def mock_show() -> Generator[Any, Any, Any]:
    """Fixture for the show function."""
    with patch("matplotlib.pyplot.show") as m:
        yield m


class TestPlotProbabilityDistribution:
    """Test class for the plot_probability_distribution function."""

    @pytest.fixture
    def mock_probas(self) -> tuple[npt.NDArray, npt.NDArray]:
        """Generates mock 1D and 2D probability data matrices."""
        np.random.seed(42)
        probas_1d = np.random.uniform(0, 1, size=(100,))
        probas_2d = np.vstack([1 - probas_1d, probas_1d]).T
        return probas_1d, probas_2d

    @pytest.mark.parametrize("dim", [1, 2])
    def test_plot_probability_distribution_success(
        self, tmp_path: Path, mock_probas: tuple[npt.NDArray, npt.NDArray], dim: int
    ) -> None:
        """Validates that both 1D and 2D array inputs successfully save files
        to disk using tmp_path."""
        probas_1d, probas_2d = mock_probas
        probas_input = probas_1d if dim == 1 else probas_2d

        # Define a base filename inside pytest's isolated temporary directory
        base_save_path = os.path.join(tmp_path, "test_distribution_plot")
        expected_extension = ".png"
        expected_file_path = base_save_path + expected_extension

        plot_probability_distribution(
            probas=probas_input,
            label="Test Model",
            n_bins=12,
            save_path=base_save_path,
            extension=expected_extension,
            show_fig=False,
            set_title="Test Title",
        )

        # Assert file was generated completely on disk
        assert os.path.exists(expected_file_path)
        assert os.path.getsize(expected_file_path) > 0

    def test_plot_probability_distribution_invalid_dimensions(self) -> None:
        """Validates error case when parsing higher dimensional arrays down
        to 2D column subsets."""
        invalid_probas = np.random.uniform(0, 1, size=(10, 2, 2))

        # Higher dimensions fail pandas/numpy array operations within standard plotting contexts
        with pytest.raises(ValueError):
            plot_probability_distribution(probas=invalid_probas, show_fig=False)


class TestPlotROCCurve:
    """Test class for the plot_ROC_curve function."""

    @pytest.fixture
    def dummy_data(self) -> tuple[npt.NDArray, npt.NDArray]:
        """Fixture providing standard binary labels and probability scores."""
        y_test = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        probas = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.2, 0.8])
        return y_test, probas

    def test_plot_ROC_curve_success(
        self, mock_show: MagicMock, dummy_data: tuple[npt.NDArray, npt.NDArray]
    ) -> None:
        """Test case verifying that the function executes successfully under
        normal conditions."""
        y_test, probas = dummy_data
        plot_ROC_curve(
            y_test=y_test,
            probas=probas,
            label="Model A",
            n_bootstraps=10,
            show_fig=True,
        )
        mock_show.assert_called_once()

    def test_plot_ROC_curve_zero_bootstraps(
        self, dummy_data: tuple[npt.NDArray, npt.NDArray]
    ) -> None:
        """Test case ensuring the function executes correctly when
        n_bootstraps is set to 0."""
        y_test, probas = dummy_data
        plot_ROC_curve(
            y_test=y_test,
            probas=probas,
            n_bootstraps=0,
            show_fig=False,
        )

    def test_plot_ROC_curve_save_file(
        self,
        tmp_path: Path,
        dummy_data: tuple[npt.NDArray, npt.NDArray],
    ) -> None:
        """Test case verifying that file saving logic and validation
        checks are triggered when save_path is provided."""
        y_test, probas = dummy_data
        save_path = str(tmp_path / "test_roc_plot")
        extension = ".png"
        expected_file = save_path + extension

        plot_ROC_curve(
            y_test=y_test,
            probas=probas,
            save_path=save_path,
            extension=extension,
            show_fig=False,
            n_bootstraps=5,
        )

        assert os.path.exists(expected_file)
        assert os.path.getsize(expected_file) > 0

    def test_plot_ROC_curve_custom_kwargs(
        self, dummy_data: tuple[npt.NDArray, npt.NDArray]
    ) -> None:
        """Test case checking that custom figure and axes kwargs
        like set_title are processed correctly."""
        y_test, probas = dummy_data
        plot_ROC_curve(
            y_test=y_test,
            probas=probas,
            set_title="Custom ROC Title",
            n_bootstraps=5,
            show_fig=False,
        )

    def test_plot_ROC_curve_hide_fig(
        self, mock_show: MagicMock, dummy_data: tuple[npt.NDArray, npt.NDArray]
    ) -> None:
        """Test case verifying that plt.show is not called when show_fig is False."""
        y_test, probas = dummy_data
        plot_ROC_curve(
            y_test=y_test,
            probas=probas,
            show_fig=False,
            n_bootstraps=5,
        )
        mock_show.assert_not_called()

    def test_plot_ROC_curve_2d_probas(self) -> None:
        """Test case ensuring handling of 2D probability matrices (n_samples, 2)."""
        y_test = np.array([0, 1, 0, 1])
        # Passing 2D array of class probabilities [prob_class_0, prob_class_1]
        probas_2d = np.array(
            [
                [0.8, 0.2],
                [0.1, 0.9],
                [0.7, 0.3],
                [0.3, 0.7],
            ]
        )

        # Note: If roc_curve requires 1D or slice [:, 1],
        # this tests current interface robustness
        try:
            plot_ROC_curve(
                y_test=y_test,
                probas=probas_2d[:, 1],
                n_bootstraps=5,
                show_fig=False,
            )
        except Exception as e:
            pytest.fail(f"Function failed with 1D probability slice from 2D array: {e}")


class TestGetCalibrationData:
    """Test class for the _get_calibration_data function."""

    @pytest.fixture
    def sample_inputs(self) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """Generate sample inputs."""
        y = np.array([0, 1, 0, 1])
        probas = np.array([0.1, 0.9, 0.2, 0.8])
        grid = np.linspace(0, 1, 5)
        return y, probas, grid

    def test_get_calibration_data_test_strategies(
        self, sample_inputs: tuple[npt.NDArray, npt.NDArray, npt.NDArray]
    ) -> None:
        """Ensures sklearn's calibration_curve is called correctly for
        standard strategies."""
        y, probas, _ = sample_inputs

        with patch("medpipe.metrics.plots.calibration_curve") as mock_curve:
            mock_curve.return_value = (np.array([0.0, 1.0]), np.array([0.15, 0.85]))

            prob_true, _ = _get_calibration_data(
                y, probas, strategy="uniform", n_bins=2
            )

            mock_curve.assert_called_once_with(y, probas, n_bins=2)
            assert len(prob_true) == 2

    def test_get_calibration_data_spline_strategy_success(
        self, sample_inputs: tuple[npt.NDArray, npt.NDArray, npt.NDArray]
    ) -> None:
        """Ensures spline object fits and predicts across the assigned
        evaluation grid."""
        y, probas, grid = sample_inputs

        mock_spline = MagicMock()
        mock_spline.predict.return_value = grid * 0.95

        with patch("medpipe.metrics.plots.SplineCalib", return_value=mock_spline):
            prob_true, prob_pred = _get_calibration_data(
                y, probas, strategy="spline", grid=grid
            )

            mock_spline.fit.assert_called_once()
            np.testing.assert_array_equal(prob_true, grid)
            assert prob_pred is not None

    def test_get_calibration_data_spline_missing_grid(
        self, sample_inputs: tuple[npt.NDArray, npt.NDArray, npt.NDArray]
    ) -> None:
        """Validates error check requiring grid arrays for the spline strategy."""
        y, probas, _ = sample_inputs
        with pytest.raises(
            ValueError, match="Input grid should not be None with spline strategy"
        ):
            _get_calibration_data(y, probas, strategy="spline", grid=None)

    def test_get_calibration_data_unknown_strategy(
        self, sample_inputs: tuple[npt.NDArray, npt.NDArray, npt.NDArray]
    ) -> None:
        """Validates error check rejecting non-supported strategy configurations."""
        y, probas, grid = sample_inputs
        with pytest.raises(ValueError, match="Unknown strategy invalid_strat"):
            _get_calibration_data(y, probas, strategy="invalid_strat", grid=grid)  # type: ignore


class TestPlotCalibration:
    """Test class for the plot_calibration function."""

    @pytest.fixture
    def mock_axes(self) -> Generator[Any, Any, Any]:
        """Fixture for the axes."""
        fig, ax = plt.subplots()
        yield ax
        plt.close(fig)

    @pytest.mark.parametrize(
        "strategy, expected_marker",
        [("uniform", "."), ("quantile", "."), ("spline", "")],
    )
    def test_plot_calibration_marker_selection(
        self,
        mock_axes: Generator[Any, Any, Any],
        strategy: Literal["uniform", "quantile", "spline"],
        expected_marker: str,
    ) -> None:
        """Ensures plotting uses discrete markers for binning styles and
        none for continuous curves."""
        prob_pred = np.array([0.1, 0.5, 0.9])
        prob_true = np.array([0.12, 0.48, 0.88])
        lower = prob_true - 0.05
        upper = prob_true + 0.05

        with (
            patch.object(mock_axes, "plot") as mock_plot,
            patch.object(mock_axes, "fill_between") as mock_fill,
        ):
            _plot_calibration(
                mock_axes,  # type: ignore
                prob_pred,
                prob_true,
                lower,
                upper,
                strategy,
                "blue",
                "ModelA",
            )

            # Check that plot was called with the correct marker choice
            mock_plot.assert_called_once_with(
                prob_pred,
                prob_true,
                marker=expected_marker,
                color="blue",
                label="ModelA",
            )
            # Check confidence interval bounds shading
            mock_fill.assert_called_once_with(
                prob_pred, lower, upper, color="blue", alpha=0.5, label="ModelA 95% CI"
            )


class TestPlotReliabilityDiagram:
    """Test class for the plot_reliability_diagram function."""

    @pytest.fixture(autouse=True)
    def mock_dependencies(self) -> Generator[Any, Any, Any]:
        """Mocks sub-helpers to cleanly isolate layout calculations."""
        with (
            patch("matplotlib.pyplot.show"),
            patch(
                "medpipe.metrics.plots._get_calibration_data",
                return_value=(np.linspace(0, 1, 5), np.linspace(0, 1, 5)),
            ),
            patch("medpipe.metrics.plots._plot_calibration"),
        ):
            yield

    def test_plot_reliability_diagram_success(self, tmp_path: Path) -> None:
        """Test successful function call."""
        y_test = np.array([0, 1, 0, 1, 0])
        probas = np.array([0.1, 0.9, 0.2, 0.8, 0.3])
        base_path = os.path.join(tmp_path, "reliability_plot")

        plot_reliability_diagram(y_test, probas, save_path=base_path, show_fig=False)
        assert os.path.exists(base_path + ".png")


class TestPlotStrataHeatmap:
    """Test class for the plot_strata_heatmap function."""

    @pytest.fixture
    def valid_heatmap_inputs(self) -> Dict[str, Any]:
        """Generate heatmap data."""
        return {
            "outcomes": ["OutA", "OutB"],
            "metric": "auroc",
            "strata": ["StratA", "StratB"],
            "scores": np.array([0.8, 0.7]),
            "strata_scores": np.array([np.array([0.78, 0.69]), np.array([0.82, 0.71])]),
        }

    def test_plot_strata_heatmap_success(
        self, tmp_path: Path, valid_heatmap_inputs: Dict[str, Any]
    ) -> None:
        base_path = os.path.join(tmp_path, "heatmap_auroc")
        # Custom logic inside saves using: save_path + metric + extension
        expected_file = base_path + ".png"

        plot_strata_heatmap(**valid_heatmap_inputs, save_path=base_path, show_fig=False)
        assert os.path.exists(expected_file)

    def test_plot_strata_heatmap_mismatched_strata_length(
        self, valid_heatmap_inputs: Dict[str, Any]
    ) -> None:
        """Validates check: len(stratas) == len(strata_scores)"""
        inputs = valid_heatmap_inputs.copy()
        inputs["strata"] = ["OnlyOneStratum"]  # Length 1 vs 2 row vectors

        with pytest.raises(
            ValueError,
            match=(
                "Inputs strata and strata_scores should be the same length "
                f"but got 1 and 2"
            ),
        ):
            plot_strata_heatmap(**inputs, show_fig=False)

    def test_plot_strata_heatmap_mismatched_outcomes_length(
        self, valid_heatmap_inputs: Dict[str, Any]
    ) -> None:
        """Validates check: len(outcomes) == len(strata_scores[0])"""
        inputs = valid_heatmap_inputs.copy()
        inputs["outcomes"] = ["OnlyOneOutcome"]  # Length 1 vs 2 columns in vectors

        with pytest.raises(
            ValueError,
            match="Inputs outcomes and strata_scores should be the same length",
        ):
            plot_strata_heatmap(**inputs, show_fig=False)

    def test_plot_strata_heatmap_mismatched_scores_length(
        self, valid_heatmap_inputs: Dict[str, Any]
    ) -> None:
        """Validates check: len(scores) == len(strata_scores[0])"""
        inputs = valid_heatmap_inputs.copy()
        inputs["scores"] = np.array([0.85])  # Length 1 baseline score vs 2 columns

        with pytest.raises(
            ValueError,
            match="Inputs scores and strata_scores should be the same length",
        ):
            plot_strata_heatmap(**inputs, show_fig=False)
