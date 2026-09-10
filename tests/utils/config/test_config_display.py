"""
Test functions for the DisplayDefaultsConfig and DisplayConfig schemas
of the config module.
"""

import pytest
from pydantic import ValidationError

from medpipe.utils.config import DisplayConfig, DisplayDefaultsConfig


class TestDisplayDefaultsConfig:
    """Test class for the DisplayDefaultsConfig schema."""

    def test_default_values(self) -> None:
        """Test default values when no parameters are provided."""
        config = DisplayDefaultsConfig()

        assert config.n_bootstraps == 1000
        assert config.save is True
        assert config.show is False
        assert config.n_bins == 10
        assert config.strategy == "uniform"

    def test_custom_values(self) -> None:
        """Test instantiating with custom values."""
        config = DisplayDefaultsConfig(
            n_bootstraps=200,
            save=False,
            show=True,
            n_bins=20,
            strategy="spline",
        )

        assert config.n_bootstraps == 200
        assert config.save is False
        assert config.show is True
        assert config.n_bins == 20
        assert config.strategy == "spline"

    def test_extra_fields_allowed(self) -> None:
        """Test that extra fields are allowed for flexibility in plot parameters."""
        config = DisplayDefaultsConfig.model_validate(
            {"n_bootstraps": 500, "dpi": 300, "custom_color": "blue"}
        )

        assert config.n_bootstraps == 500
        assert getattr(config, "dpi", None) == 300
        assert getattr(config, "custom_color", None) == "blue"

    def test_n_bootstraps_limit(self) -> None:
        """Test that n_bootstraps must be non-negative."""
        with pytest.raises(
            ValidationError, match="Input should be greater than or equal to 0"
        ):
            DisplayDefaultsConfig(n_bootstraps=-1)

    def test_n_bins_limit(self) -> None:
        """Test that n_bins must be at least 1."""
        with pytest.raises(
            ValidationError, match="Input should be greater than or equal to 1"
        ):
            DisplayDefaultsConfig(n_bins=0)

    def test_invalid_strategy(self) -> None:
        """Test that an unsupported strategy raises a ValidationError."""
        with pytest.raises(
            ValidationError,
            match="Input should be 'uniform', 'quantile' or 'spline'",
        ):
            DisplayDefaultsConfig.model_validate({"strategy": "unsupported"})


class TestDisplayConfig:
    """Test class for the DisplayConfig schema."""

    def test_valid_config(self) -> None:
        """Pass a fully defined valid configuration to DisplayConfig."""
        raw_config = {
            "defaults": {
                "n_bootstraps": 1000,
                "save": True,
                "show": False,
                "n_bins": 10,
                "strategy": "uniform",
            },
            "overrides": {
                "calibration": {
                    "n_bootstraps": 200,
                    "strategy": "spline",
                },
                "distribution": {"n_bins": 25},
            },
            "outcome_overrides": {
                "MORTALITY_30D": {
                    "calibration": {
                        "n_bootstraps": 50,
                        "strategy": "uniform",
                    }
                }
            },
            "theme": {"primary_color": "#2D90D8", "show_grid": False},
        }

        config = DisplayConfig.model_validate(raw_config)

        assert config.defaults.n_bootstraps == 1000
        assert config.overrides["calibration"]["n_bootstraps"] == 200
        assert (
            config.outcome_overrides["MORTALITY_30D"]["calibration"]["strategy"]
            == "uniform"
        )
        assert config.theme["primary_color"] == "#2D90D8"

    def test_default_values(self) -> None:
        """Test that defaults, overrides, outcome_overrides, and theme have
        sane defaults when the config is empty."""
        config = DisplayConfig.model_validate({})

        assert config.defaults == DisplayDefaultsConfig()
        assert config.overrides == {}
        assert config.outcome_overrides == {}
        assert config.theme is None

    def test_flat_and_legacy_keys_transformed_to_defaults(self) -> None:
        """Test that legacy flat keys get mapped into defaults during pre-validation."""
        raw_config = {
            "calibration_strategy": "spline",
            "n_bootstraps": 500,
            "save": False,
        }

        config = DisplayConfig.model_validate(raw_config)

        assert config.defaults.strategy == "spline"
        assert config.defaults.n_bootstraps == 500
        assert config.defaults.save is False

    def test_non_dict_defaults_reset_before_legacy_key_merge(self) -> None:
        """Test that a non-dict 'defaults' value is discarded rather than
        merged with legacy flat keys."""
        raw_config = {
            "defaults": None,
            "calibration_strategy": "spline",
        }

        config = DisplayConfig.model_validate(raw_config)

        assert config.defaults.strategy == "spline"

    def test_legacy_keys_do_not_override_explicit_defaults(self) -> None:
        """Test that an explicit 'defaults' dict wins over legacy flat keys."""
        raw_config = {
            "defaults": {"strategy": "quantile"},
            "calibration_strategy": "spline",
        }

        config = DisplayConfig.model_validate(raw_config)

        assert config.defaults.strategy == "quantile"

    @pytest.mark.parametrize(
        "invalid_plot_key",
        ["invalid_plot_name", "unknown_curve", "scatter_plot"],
    )
    def test_invalid_override_plot_key(self, invalid_plot_key: str) -> None:
        """Test that unknown plot override types raise a ValidationError."""
        raw_config = {
            "overrides": {
                invalid_plot_key: {"n_bootstraps": 100},
            }
        }

        with pytest.raises(
            ValidationError, match=f"Unknown plot override type '{invalid_plot_key}'"
        ):
            DisplayConfig.model_validate(raw_config)

    def test_invalid_outcome_override_plot_key(self) -> None:
        """Test that unknown plot types inside outcome_overrides raise a ValidationError."""
        raw_config = {
            "outcome_overrides": {
                "MORTALITY_30D": {
                    "unsupported_diagram": {"n_bins": 5},
                }
            }
        }

        with pytest.raises(
            ValidationError, match="Unknown plot override type 'unsupported_diagram'"
        ):
            DisplayConfig.model_validate(raw_config)

    def test_from_dict(self) -> None:
        """Test that from_dict instantiates a DisplayConfig equivalent to
        model_validate."""
        raw_config = {"defaults": {"n_bootstraps": 300}}

        config = DisplayConfig.from_dict(raw_config)

        assert config == DisplayConfig.model_validate(raw_config)

    def test_extra_fields_forbidden(self) -> None:
        """Test that unknown top-level fields raise a ValidationError."""
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            DisplayConfig.model_validate({"unexpected_flag": True})

    def test_non_dict_input_skips_legacy_key_handling(self) -> None:
        """Test that a non-dict input bypasses legacy key handling and
        fails standard pydantic model validation instead."""
        with pytest.raises(
            ValidationError, match="Input should be a valid dictionary"
        ):
            DisplayConfig.model_validate("not_a_dict")  # type: ignore
