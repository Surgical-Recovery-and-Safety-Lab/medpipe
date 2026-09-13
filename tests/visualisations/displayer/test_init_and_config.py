"""
Tests for MedpipeClassifierDisplayer initialization and its configuration-resolution
helpers: _normalize_plot_type, _resolve_plot_config, _format_stratum_label.
"""

from unittest.mock import MagicMock

import pytest

from medpipe.utils.config import DisplayConfig, DisplayDefaultsConfig
from medpipe.visualisation.displayer import MedpipeClassifierDisplayer
from medpipe.visualisation.themes import MedpipeTheme


class TestMedpipeDisplayerInit:
    """Tests for MedpipeClassifierDisplayer initialization."""

    def test_init_default_theme(self, mock_orchestrator) -> None:
        """Test initialization with default MedpipeTheme."""
        displayer = MedpipeClassifierDisplayer(orchestrator=mock_orchestrator)

        assert displayer.orchestrator == mock_orchestrator
        assert displayer.run_dir == mock_orchestrator.run_dir
        assert isinstance(displayer.theme, MedpipeTheme)
        assert displayer.logger is not None

    def test_init_custom_theme(self, mock_orchestrator) -> None:
        """Test initialization with a custom MedpipeTheme."""
        custom_theme = MedpipeTheme(primary_color="#FF0000", dpi=150)
        displayer = MedpipeClassifierDisplayer(
            orchestrator=mock_orchestrator, theme=custom_theme
        )

        assert displayer.theme.primary_color == "#FF0000"
        assert displayer.theme.dpi == 150


class TestNormalizePlotType:
    """Tests for MedpipeClassifierDisplayer._normalize_plot_type static helper."""

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
        assert MedpipeClassifierDisplayer._normalize_plot_type(input_type) == expected

    def test_normalize_plot_type_case_insensitive(self) -> None:
        """Test that normalization handles uppercase inputs."""
        assert (
            MedpipeClassifierDisplayer._normalize_plot_type("CALIBRATION")
            == "reliability"
        )
        assert (
            MedpipeClassifierDisplayer._normalize_plot_type("PR_CURVE")
            == "precision_recall"
        )


class TestResolvePlotConfig:
    """Tests for MedpipeClassifierDisplayer._resolve_plot_config hierarchical
    resolution."""

    @pytest.fixture
    def mock_displayer(self) -> MedpipeClassifierDisplayer:
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
        return MedpipeClassifierDisplayer(orchestrator=mock_orchestrator)

    def test_resolve_default_values(
        self, mock_displayer: MedpipeClassifierDisplayer
    ) -> None:
        """Test fallback to global defaults when no overrides exist for plot type."""
        resolved = mock_displayer._resolve_plot_config(plot_type="roc")

        assert resolved["n_bootstraps"] == 1000
        assert resolved["save"] is True
        assert resolved["show"] is False

    def test_resolve_global_plot_override(
        self, mock_displayer: MedpipeClassifierDisplayer
    ) -> None:
        """Test that plot-level overrides take precedence over global defaults."""
        resolved = mock_displayer._resolve_plot_config(plot_type="reliability")

        assert resolved["n_bootstraps"] == 200
        assert resolved["strategy"] == "spline"
        assert resolved["save"] is True

    def test_resolve_alias_plot_override(
        self, mock_displayer: MedpipeClassifierDisplayer
    ) -> None:
        """Test that canonical alias normalization resolves plot-level overrides."""
        resolved = mock_displayer._resolve_plot_config(plot_type="calibration")

        assert resolved["n_bootstraps"] == 200
        assert resolved["strategy"] == "spline"

    def test_resolve_outcome_override(
        self, mock_displayer: MedpipeClassifierDisplayer
    ) -> None:
        """Test that outcome-specific overrides take precedence over global
        plot overrides."""
        resolved = mock_displayer._resolve_plot_config(
            plot_type="calibration", outcome="MORTALITY_30D"
        )

        assert resolved["n_bootstraps"] == 50
        assert resolved["strategy"] == "uniform"

    def test_resolve_runtime_kwargs_precedence(
        self, mock_displayer: MedpipeClassifierDisplayer
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
        self, mock_displayer: MedpipeClassifierDisplayer
    ) -> None:
        """Test that None values passed as runtime kwargs do not overwrite
        configured values."""
        resolved = mock_displayer._resolve_plot_config(
            plot_type="calibration",
            n_bootstraps=None,
        )

        assert resolved["n_bootstraps"] == 200

    def test_resolve_when_display_config_is_none(self, mock_orchestrator) -> None:
        """Test that _resolve_plot_config falls back to system defaults when
        display config is None."""
        displayer = MedpipeClassifierDisplayer(orchestrator=mock_orchestrator)
        resolved = displayer._resolve_plot_config(plot_type="roc")

        assert resolved["n_bootstraps"] == 1000
        assert resolved["save"] is True
        assert resolved["show"] is False
        assert resolved["n_bins"] == 10
        assert resolved["strategy"] == "uniform"


class TestFormatStratumLabel:
    """Tests for MedpipeClassifierDisplayer._format_stratum_label static helper."""

    @pytest.mark.parametrize(
        "stratum_var, cat_key, expected",
        [
            ("AGE", "[18, 50]", "AGE: 18-50"),
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
        """Test formatting of interval strings, open-ended bounds, and
        categorical keys."""
        result = MedpipeClassifierDisplayer._format_stratum_label(stratum_var, cat_key)
        assert result == expected

    def test_format_stratum_label_bracketed_non_pair_falls_back_to_raw(self) -> None:
        """Test that a bracketed value which doesn't parse into a 2-element
        list/tuple (e.g. 3 values, or a single value) falls back to the raw
        bracketed string rather than being (mis)treated as a range."""
        assert (
            MedpipeClassifierDisplayer._format_stratum_label("AGE", "[18, 30, 50]")
            == "AGE: [18, 30, 50]"
        )
        assert (
            MedpipeClassifierDisplayer._format_stratum_label("AGE", "[18]")
            == "AGE: [18]"
        )
