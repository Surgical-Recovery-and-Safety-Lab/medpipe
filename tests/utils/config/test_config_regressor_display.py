"""
Test functions for the RegressorDisplayConfig schema of the config module.
"""

import pytest
from pydantic import ValidationError

from medpipe.utils.config import DisplayDefaultsConfig, RegressorDisplayConfig


class TestRegressorDisplayConfig:
    """Test class for the RegressorDisplayConfig schema."""

    def test_valid_config(self) -> None:
        """Pass a fully defined valid configuration to RegressorDisplayConfig."""
        raw_config = {
            "defaults": {
                "n_bootstraps": 1000,
                "save": True,
                "show": False,
            },
            "overrides": {
                "residuals": {"n_bootstraps": 200},
                "predicted_vs_actual": {"save": False},
            },
            "outcome_overrides": {
                "LOS_DAYS": {
                    "residuals": {"n_bootstraps": 50},
                }
            },
            "theme": {"primary_color": "#2D90D8"},
        }

        config = RegressorDisplayConfig.model_validate(raw_config)

        assert config.defaults.n_bootstraps == 1000
        assert config.overrides["residuals"]["n_bootstraps"] == 200
        assert (
            config.outcome_overrides["LOS_DAYS"]["residuals"]["n_bootstraps"] == 50
        )
        assert config.theme["primary_color"] == "#2D90D8"

    def test_default_values(self) -> None:
        """Test that defaults, overrides, outcome_overrides, and theme have
        sane defaults when the config is empty."""
        config = RegressorDisplayConfig.model_validate({})

        assert config.defaults == DisplayDefaultsConfig()
        assert config.overrides == {}
        assert config.outcome_overrides == {}
        assert config.theme is None

    @pytest.mark.parametrize(
        "invalid_plot_key",
        ["roc", "calibration", "unknown_curve"],
    )
    def test_invalid_override_plot_key(self, invalid_plot_key: str) -> None:
        """Test that classifier-only or unknown plot override types raise a
        ValidationError on the regressor schema."""
        raw_config = {
            "overrides": {
                invalid_plot_key: {"n_bootstraps": 100},
            }
        }

        with pytest.raises(
            ValidationError, match=f"Unknown plot override type '{invalid_plot_key}'"
        ):
            RegressorDisplayConfig.model_validate(raw_config)

    def test_valid_shared_heatmap_plot_key(self) -> None:
        """Test that the shared strata_heatmap plot type is accepted."""
        raw_config = {"overrides": {"strata_heatmap": {"save": True}}}

        config = RegressorDisplayConfig.model_validate(raw_config)

        assert config.overrides["strata_heatmap"]["save"] is True

    def test_from_dict(self) -> None:
        """Test that from_dict instantiates a RegressorDisplayConfig
        equivalent to model_validate."""
        raw_config = {"defaults": {"n_bootstraps": 300}}

        config = RegressorDisplayConfig.from_dict(raw_config)

        assert config == RegressorDisplayConfig.model_validate(raw_config)

    def test_extra_fields_forbidden(self) -> None:
        """Test that unknown top-level fields raise a ValidationError."""
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            RegressorDisplayConfig.model_validate({"unexpected_flag": True})
