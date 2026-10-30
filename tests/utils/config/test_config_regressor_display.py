"""
Test functions for the RegressorDisplayConfig schema of the config module.
"""

import pytest
from pydantic import ValidationError

from medpipe.utils.config import (
    RegressorDisplayConfig,
    RegressorDisplayDefaultsConfig,
)


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
                "coverage": {"n_bootstraps": 200},
                "pit_histogram": {"save": False},
            },
            "outcome_overrides": {
                "LOS_DAYS": {
                    "coverage": {"n_bootstraps": 50},
                }
            },
            "theme": {"primary_color": "#2D90D8"},
        }

        config = RegressorDisplayConfig.model_validate(raw_config)

        assert config.defaults.n_bootstraps == 1000
        assert config.overrides["coverage"]["n_bootstraps"] == 200
        assert config.outcome_overrides["LOS_DAYS"]["coverage"]["n_bootstraps"] == 50
        assert config.theme["primary_color"] == "#2D90D8"

    def test_default_values(self) -> None:
        """Test that defaults, overrides, outcome_overrides, and theme have
        sane defaults when the config is empty."""
        config = RegressorDisplayConfig.model_validate({})

        assert config.defaults == RegressorDisplayDefaultsConfig()
        assert config.overrides == {}
        assert config.outcome_overrides == {}
        assert config.theme is None

    @pytest.mark.parametrize(
        "invalid_plot_key",
        ["roc", "calibration", "residuals", "predicted_vs_actual", "unknown_curve"],
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


class TestRegressorDisplayDefaultsConfig:
    """Test class for the RegressorDisplayDefaultsConfig schema."""

    def test_default_values(self) -> None:
        """Test the default values."""
        config = RegressorDisplayDefaultsConfig()

        assert config.n_bootstraps == 1000
        assert config.save is True
        assert config.show is False
        assert config.n_jobs == 1
        assert config.dist_n_bins == 20
        assert config.dist_yscale == "linear"
        assert config.coverage_levels == [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]
        assert config.pit_n_bins == 20

    def test_default_coverage_levels_are_not_shared(self) -> None:
        """Test that the default list is a fresh copy for each instance."""
        first = RegressorDisplayDefaultsConfig()
        second = RegressorDisplayDefaultsConfig()

        first.coverage_levels.append(99)

        assert 99 not in second.coverage_levels

    def test_custom_values(self) -> None:
        """Test instantiating with custom values."""
        config = RegressorDisplayDefaultsConfig(
            n_bootstraps=50,
            dist_n_bins=30,
            dist_yscale="log",
            coverage_levels=[50, 80.5, 95],
            pit_n_bins=10,
        )

        assert config.coverage_levels == [50, 80.5, 95]
        assert config.dist_n_bins == 30
        assert config.dist_yscale == "log"
        assert config.pit_n_bins == 10

    @pytest.mark.parametrize("key", ["n_bins", "strategy", "cv_splines", "unknown"])
    def test_classifier_only_and_unknown_keys_forbidden(self, key: str) -> None:
        """Test that keys that have no effect on regression plots are rejected."""
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            RegressorDisplayDefaultsConfig.model_validate({key: 1})

    @pytest.mark.parametrize("levels", [[0], [100], [-5, 50], [50, 120]])
    def test_coverage_levels_must_be_percentages(self, levels: list) -> None:
        """Test that levels outside the open interval (0, 100) are rejected."""
        with pytest.raises(ValidationError, match="strictly between 0 and 100"):
            RegressorDisplayDefaultsConfig(coverage_levels=levels)

    def test_coverage_levels_cannot_be_empty(self) -> None:
        """Test that an empty list of levels is rejected."""
        with pytest.raises(ValidationError, match="should not be empty"):
            RegressorDisplayDefaultsConfig(coverage_levels=[])

    @pytest.mark.parametrize("key", ["dist_n_bins", "pit_n_bins"])
    def test_bin_counts_limit(self, key: str) -> None:
        """Test that the bin counts must be at least 1."""
        with pytest.raises(
            ValidationError, match="Input should be greater than or equal to 1"
        ):
            RegressorDisplayDefaultsConfig.model_validate({key: 0})

    def test_invalid_dist_yscale(self) -> None:
        """Test that an unsupported dist_yscale is rejected."""
        with pytest.raises(ValidationError, match="Input should be 'linear', 'log'"):
            RegressorDisplayDefaultsConfig(dist_yscale="unsupported")  # type: ignore

    def test_regressor_display_config_rejects_classifier_default(self) -> None:
        """Test that RegressorDisplayConfig does not accept the classifier-only
        default keys."""
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            RegressorDisplayConfig.model_validate({"defaults": {"n_bins": 10}})
