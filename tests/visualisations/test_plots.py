"""Tests for the stateless drawing primitives in plots.py."""

from unittest.mock import MagicMock

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure

matplotlib.use("Agg")  # Non-interactive backend for headless testing

from medpipe.visualisation.plots import draw_probability_distribution, draw_roc_curve


@pytest.fixture(autouse=True)
def _close_figures():
    """Automatically close all Matplotlib figures after each test."""
    yield
    plt.close("all")


@pytest.fixture
def dummy_roc_data():
    """Provides standard false positive and true positive rate arrays."""
    fpr = np.array([0.0, 0.2, 0.5, 0.8, 1.0])
    tpr = np.array([0.0, 0.4, 0.7, 0.9, 1.0])
    return fpr, tpr


class TestDrawProbabilityDistribution:
    """Tests for the stateless `draw_probability_distribution` rendering function."""

    def test_draw_probability_distribution_default_axes(self) -> None:
        """Test drawing histogram with 1D probas and default axes creation."""
        probas = np.array([0.1, 0.25, 0.4, 0.75, 0.9])

        fig, ax = draw_probability_distribution(
            probas=probas, n_bins=10, label="Predictions"
        )

        assert isinstance(fig, (Figure, SubFigure))
        assert isinstance(ax, Axes)
        assert ax.get_xlabel() == "Predicted Probabilities"
        assert ax.get_ylabel() == "Count"
        # 10 bins should produce 10 rectangle patches
        assert len(ax.patches) == 10

    def test_draw_probability_distribution_2d_probas(self) -> None:
        """Test probability distribution plotting with 2D array input (n_samples, 2)."""
        probas_1d = np.array([0.1, 0.3, 0.6, 0.8])
        probas_2d = np.column_stack((1 - probas_1d, probas_1d))

        fig, ax = draw_probability_distribution(probas=probas_2d, n_bins=5)

        assert isinstance(fig, (Figure, SubFigure))
        assert isinstance(ax, Axes)

    def test_draw_probability_distribution_existing_axes(self) -> None:
        """Test drawing onto a pre-existing Matplotlib axes instance."""
        existing_fig, existing_ax = plt.subplots(figsize=(8, 8))
        probas = np.array([0.2, 0.5, 0.8])

        fig, ax = draw_probability_distribution(probas=probas, ax=existing_ax)

        assert fig is existing_fig
        assert ax is existing_ax

    def test_draw_probability_distribution_custom_styling(self) -> None:
        """Test custom color, title, and spine visibility options."""
        probas = np.array([0.15, 0.45, 0.85])

        fig, ax = draw_probability_distribution(
            probas=probas,
            color="#FF0000",
            title="Custom Distribution Title",
            show_spines=True,
        )

        assert ax.get_title() == "Custom Distribution Title"
        assert ax.spines["top"].get_visible() is True
        assert ax.spines["right"].get_visible() is True

    def test_draw_probability_distribution_detached_axes_raises(self) -> None:
        """Edge case: raise ValueError when provided Axes is detached from a Figure."""
        mock_ax = MagicMock(spec=Axes)
        mock_ax.get_figure.return_value = None

        with pytest.raises(
            ValueError,
            match="The provided Axes instance is not attached to a Figure",
        ):
            draw_probability_distribution(probas=np.array([0.1, 0.2]), ax=mock_ax)


class TestDrawRocCurve:
    """Tests for the stateless `draw_roc_curve` rendering function."""

    def test_draw_roc_curve_default_axes_creation(self, dummy_roc_data) -> None:
        """Test drawing ROC curve when ax=None (creates new figure and axes)."""
        fpr, tpr = dummy_roc_data

        fig, ax = draw_roc_curve(fpr=fpr, tpr=tpr, label="Test Model")

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert ax.get_xlabel() == "False Positive Rate (1 - Specificity)"
        assert ax.get_ylabel() == "True Positive Rate (Sensitivity)"

        # Verify chance level and model curve lines exist
        lines = ax.get_lines()
        assert len(lines) == 2  # Chance line + Model line
        assert lines[0].get_label() == "Chance level"
        assert lines[1].get_label() == "Test Model"

    def test_draw_roc_curve_existing_axes(self, dummy_roc_data) -> None:
        """Test drawing onto a pre-existing Matplotlib axes instance."""
        fpr, tpr = dummy_roc_data
        existing_fig, existing_ax = plt.subplots(figsize=(8, 8))

        fig, ax = draw_roc_curve(fpr=fpr, tpr=tpr, ax=existing_ax)

        assert fig is existing_fig
        assert ax is existing_ax

    def test_draw_roc_curve_with_confidence_intervals(self, dummy_roc_data) -> None:
        """Test rendering with pre-computed lower and upper confidence interval arrays."""
        fpr, tpr = dummy_roc_data
        lower_ci = tpr - 0.05
        upper_ci = tpr + 0.05

        _, ax = draw_roc_curve(
            fpr=fpr,
            tpr=tpr,
            lower_ci=lower_ci,
            upper_ci=upper_ci,
            label="Model with CI",
            ci_color="#FF0000",
            ci_alpha=0.2,
        )

        # Check that PolyCollection (fill_between) artist was added
        assert len(ax.collections) == 1
        labels = [
            text.get_text() for text in ax.get_legend().get_texts()  # type: ignore
        ]
        assert "Model with CI 95% CI" in labels

    def test_draw_roc_curve_custom_styling(self, dummy_roc_data) -> None:
        """Test custom style parameters including titles and spine visibility."""
        fpr, tpr = dummy_roc_data

        _, ax = draw_roc_curve(
            fpr=fpr,
            tpr=tpr,
            color="#333333",
            linestyle="--",
            linewidth=3.0,
            title="Custom ROC Title",
            show_spines=True,
        )

        assert ax.get_title() == "Custom ROC Title"
        assert ax.spines["top"].get_visible() is True
        assert ax.spines["right"].get_visible() is True

    def test_draw_roc_curve_pop_line_kwargs(self, dummy_roc_data) -> None:
        """Test that line_kwargs safely removes potential parameter collisions."""
        fpr, tpr = dummy_roc_data

        # Pass kwargs that clash with explicit arguments
        _, ax = draw_roc_curve(
            fpr=fpr,
            tpr=tpr,
            alpha=0.7,
            **{"color": "red", "linewidth": 5.0, "label": "Colliding Label"},
        )

        lines = ax.get_lines()
        # Verify kwargs didn't crash execution and main line was plotted
        assert len(lines) == 2

    def test_draw_roc_curve_detached_axes_raises_value_error(
        self, dummy_roc_data
    ) -> None:
        """Edge case: raise ValueError when provided Axes is detached from a Figure."""
        fpr, tpr = dummy_roc_data
        mock_ax = MagicMock(spec=Axes)
        mock_ax.get_figure.return_value = None

        with pytest.raises(
            ValueError,
            match="The provided Axes instance is not attached to a Figure",
        ):
            draw_roc_curve(fpr=fpr, tpr=tpr, ax=mock_ax)
