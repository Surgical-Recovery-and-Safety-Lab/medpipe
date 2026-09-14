"""
Tests for BaseDisplayer: the shared config-resolution/heatmap engine
extracted from MedpipeClassifierDisplayer, and its extension points
(`_PLOT_TYPE_ALIASES`, `_FALLBACK_DISPLAY_DEFAULTS`) for outcome-type-specific
subclasses.
"""

from typing import ClassVar

from medpipe.visualisation.displayer import BaseDisplayer, MedpipeClassifierDisplayer


class TestBaseDisplayerInheritance:
    """Tests confirming MedpipeClassifierDisplayer is a BaseDisplayer."""

    def test_medpipe_classifier_displayer_is_a_base_displayer(self) -> None:
        """Test that MedpipeClassifierDisplayer subclasses BaseDisplayer."""
        assert issubclass(MedpipeClassifierDisplayer, BaseDisplayer)


class TestBaseDisplayerDefaults:
    """Tests for BaseDisplayer's own (unspecialized) defaults, used directly
    by any subclass that does not override them."""

    def test_normalize_plot_type_is_identity_without_aliases(self) -> None:
        """Test that with no configured aliases, normalization only
        lowercases the input."""
        assert BaseDisplayer._normalize_plot_type("ROC") == "roc"
        assert BaseDisplayer._normalize_plot_type("custom_plot") == "custom_plot"

    def test_resolve_plot_config_fallback_defaults(self, mock_orchestrator) -> None:
        """Test that _resolve_plot_config falls back to the base class's
        minimal defaults (n_bootstraps, save, show, n_jobs) when no display
        config is set."""
        displayer = BaseDisplayer(orchestrator=mock_orchestrator)

        resolved = displayer._resolve_plot_config(plot_type="anything")

        assert resolved == {
            "n_bootstraps": 1000,
            "save": True,
            "show": False,
            "n_jobs": 1,
        }

    def test_fallback_defaults_not_mutated_across_calls(
        self, mock_orchestrator
    ) -> None:
        """Test that resolving config repeatedly does not mutate the shared
        class-level _FALLBACK_DISPLAY_DEFAULTS dict."""
        displayer = BaseDisplayer(orchestrator=mock_orchestrator)

        displayer._resolve_plot_config(plot_type="anything", extra_param=123)

        assert "extra_param" not in BaseDisplayer._FALLBACK_DISPLAY_DEFAULTS


class TestBaseDisplayerSubclassExtensionPoints:
    """Tests proving a new outcome-type-specific displayer subclass can
    override `_PLOT_TYPE_ALIASES` and `_FALLBACK_DISPLAY_DEFAULTS`
    independently of MedpipeClassifierDisplayer, in preparation for a
    MedpipeRegressorDisplayer."""

    def test_subclass_alias_override_does_not_leak_to_base_or_siblings(self) -> None:
        """Test that a subclass's alias table is isolated from the base
        class's and from unrelated sibling subclasses."""

        class _FakeRegressorDisplayer(BaseDisplayer):
            _PLOT_TYPE_ALIASES: ClassVar[dict[str, str]] = {
                "marginal_calib": "marginal_calibration"
            }

        assert (
            _FakeRegressorDisplayer._normalize_plot_type("marginal_calib")
            == "marginal_calibration"
        )
        assert BaseDisplayer._normalize_plot_type("marginal_calib") == "marginal_calib"
        assert (
            MedpipeClassifierDisplayer._normalize_plot_type("marginal_calib")
            == "marginal_calib"
        )

    def test_subclass_fallback_defaults_override_is_isolated(
        self, mock_orchestrator
    ) -> None:
        """Test that a subclass's own _FALLBACK_DISPLAY_DEFAULTS is used
        instead of the base class's, without mutating the base."""

        class _FakeRegressorDisplayer(BaseDisplayer):
            _FALLBACK_DISPLAY_DEFAULTS: ClassVar[dict[str, object]] = {
                **BaseDisplayer._FALLBACK_DISPLAY_DEFAULTS,
                "n_coverage_levels": 10,
            }

        displayer = _FakeRegressorDisplayer(orchestrator=mock_orchestrator)
        resolved = displayer._resolve_plot_config(plot_type="coverage")

        assert resolved["n_coverage_levels"] == 10
        assert "n_coverage_levels" not in BaseDisplayer._FALLBACK_DISPLAY_DEFAULTS


class TestBaseDisplayerHeatmaps:
    """Tests confirming plot_strata_heatmap/plot_all_heatmaps work directly
    on BaseDisplayer, independent of any classifier-specific plotting."""

    def test_plot_all_heatmaps_directly_on_base_displayer(
        self, mock_orchestrator, sample_evaluations
    ) -> None:
        """Test that BaseDisplayer itself (not just MedpipeClassifierDisplayer)
        can render subgroup heatmaps from an evaluations dict."""
        displayer = BaseDisplayer(orchestrator=mock_orchestrator)

        heatmaps = displayer.plot_all_heatmaps(
            evaluations=sample_evaluations, save=False, show=False
        )

        assert set(heatmaps.keys()) == {"roc_auc", "log_loss"}
        for fig, ax in heatmaps.values():
            assert fig is not None
            assert ax is not None
