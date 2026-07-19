#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test functions for the metrics.plots module
"""

import os
from unittest.mock import patch

import numpy as np
import pytest

from medpipe.metrics.plots import plot_probability_distribution


class TestPlotProbabilityDistribution:
    """Test class for the plot_probability_distribution function."""

    @pytest.fixture
    def mock_probas(self):
        """Generates mock 1D and 2D probability data matrices."""
        np.random.seed(42)
        probas_1d = np.random.uniform(0, 1, size=(100,))
        probas_2d = np.vstack([1 - probas_1d, probas_1d]).T
        return probas_1d, probas_2d

    @pytest.fixture(autouse=True)
    def mock_matplotlib_show(self):
        """Automatically mocks plt.show across all tests to prevent visual popups."""
        with patch("matplotlib.pyplot.show") as mock_show:
            yield mock_show

    @pytest.mark.parametrize("dim", [1, 2])
    def test_successful_plot_and_save(self, tmp_path, mock_probas, dim):
        """Validates that both 1D and 2D array inputs successfully save files to disk using tmp_path."""
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

    def test_invalid_dimensions_raises_index_error(self):
        """Validates error case when parsing higher dimensional arrays down to 2D column subsets."""
        invalid_probas = np.random.uniform(0, 1, size=(10, 2, 2))

        # Higher dimensions fail pandas/numpy array operations within standard plotting contexts
        with pytest.raises(ValueError):
            plot_probability_distribution(probas=invalid_probas, show_fig=False)
