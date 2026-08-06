"""
Tests for the medpipe.visualisation.theme module.
"""

import pytest

from medpipe.visualisation.themes import MedpipeTheme


class TestMedpipeTheme:
    """Test suite for MedpipeTheme configuration and utility methods."""

    def test_default_initialisation(self) -> None:
        """Test default theme attribute values."""
        theme = MedpipeTheme()

        assert theme.primary_color == "#2D90D8"
        assert len(theme.palette) == 5
        assert theme.ci_alpha == 0.3
        assert theme.linewidth == 2.0
        assert theme.style_sheet == "seaborn-v0_8-whitegrid"
        assert theme.dpi == 300
        assert theme.font_family == "sans-serif"
        assert theme.title_fontsize == 12
        assert theme.label_fontsize == 10
        assert theme.show_spines is False

    def test_custom_initialisation(self) -> None:
        """Test theme instantiation with custom overrides."""
        theme = MedpipeTheme(
            primary_color="#FF0000",
            dpi=600,
            show_spines=True,
            palette=["#FF0000", "#00FF00"],
        )

        assert theme.primary_color == "#FF0000"
        assert theme.dpi == 600
        assert theme.show_spines is True
        assert theme.palette == ["#FF0000", "#00FF00"]

    def test_to_rc_params(self) -> None:
        """Test conversion of theme attributes into Matplotlib rcParams dictionary."""
        theme = MedpipeTheme(
            font_family="serif",
            title_fontsize=16,
            label_fontsize=12,
            dpi=150,
            show_spines=True,
        )
        rc_params = theme.to_rc_params()

        assert rc_params["font.family"] == "serif"
        assert rc_params["axes.titlesize"] == 16
        assert rc_params["axes.labelsize"] == 12
        assert rc_params["axes.titleweight"] == "bold"
        assert rc_params["axes.labelweight"] == "bold"
        assert rc_params["figure.dpi"] == 150
        assert rc_params["savefig.dpi"] == 150
        assert rc_params["axes.spines.top"] is True
        assert rc_params["axes.spines.right"] is True

    def test_get_color_indexing(self) -> None:
        """Test palette color retrieval with exact indexing and cyclic wrapping."""
        theme = MedpipeTheme(palette=["#111111", "#222222", "#333333"])

        # Direct indexing
        assert theme.get_color(0) == "#111111"
        assert theme.get_color(1) == "#222222"
        assert theme.get_color(2) == "#333333"

        # Cyclic wrap-around indexing
        assert theme.get_color(3) == "#111111"
        assert theme.get_color(5) == "#333333"

    def test_get_color_empty_palette(self) -> None:
        """Test that get_color falls back to primary_color when palette is empty."""
        theme = MedpipeTheme(primary_color="#999999", palette=[])

        assert theme.get_color(0) == "#999999"
        assert theme.get_color(10) == "#999999"

    def test_from_dict_valid_keys(self) -> None:
        """Test instantiating theme from a valid dictionary configuration."""
        config_dict = {
            "primary_color": "#000000",
            "dpi": 400,
            "ci_alpha": 0.5,
        }
        theme = MedpipeTheme.from_dict(config_dict)

        assert theme.primary_color == "#000000"
        assert theme.dpi == 400
        assert theme.ci_alpha == 0.5

    def test_from_dict_filters_unrecognized_keys(self) -> None:
        """Test that extra or unknown keys in dictionary input are filtered out without error."""
        config_dict = {
            "primary_color": "#123456",
            "unrecognized_key": "ignored_value",
            "invalid_setting": 999,
        }
        theme = MedpipeTheme.from_dict(config_dict)

        assert theme.primary_color == "#123456"
        assert not hasattr(theme, "unrecognized_key")
        assert not hasattr(theme, "invalid_setting")
