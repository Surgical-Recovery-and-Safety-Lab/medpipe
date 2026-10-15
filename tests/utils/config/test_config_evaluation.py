"""
Test functions for the MetricsConfig, FairnessConfig, and
EvaluationSubConfig schemas of the config module.
"""

from re import escape

import pytest
from pydantic import ValidationError

from medpipe.metrics.core import METRICS
from medpipe.utils.config import EvaluationSubConfig, FairnessConfig, MetricsConfig


class TestMetricsConfig:
    """Test class for the MetricsConfig class"""

    def _get_valid_config_dict(self, **overrides) -> dict:
        """Creates a fresh valid config dict to override."""
        config_dict = {
            "metrics": ["roc_auc", "log_loss", "ici"],
            "n_bootstraps": 1000,
            "ci_level": 0.95,
            "cv_splines": 3,
        }
        config_dict.update(overrides)

        return config_dict

    def test_valid_config(self) -> None:
        """Pass valid configuration to MetricsConfig."""
        raw_config = self._get_valid_config_dict()
        config = MetricsConfig.model_validate(raw_config)

        assert config.model_dump() == raw_config

    def test_default_values(self) -> None:
        """Test default metrics, n_bootstraps, ci_level, and cv_splines when
        omitted."""
        config = MetricsConfig.model_validate({})

        assert config.metrics == ["roc_auc", "ici"]
        assert config.n_bootstraps == 200
        assert config.ci_level == 0.95
        assert config.cv_splines == 3

    def test_invalid_metric(self) -> None:
        """Test case when invalid metric is provided."""
        match_expr = (
            "invalid was not found in available metric "
            f"list. Available metrics are {METRICS}"
        )

        with pytest.raises(ValidationError, match=escape(match_expr)):
            MetricsConfig.model_validate(
                self._get_valid_config_dict(metrics=["invalid"])
            )

    def test_n_bootstraps_limit(self) -> None:
        """Test that n_bootstraps must be non-negative."""
        with pytest.raises(
            ValidationError, match="Input should be greater than or equal to 0"
        ):
            MetricsConfig.model_validate(self._get_valid_config_dict(n_bootstraps=-1))

    @pytest.mark.parametrize(
        "ci_level, match_expr",
        [
            (-0.1, "Input should be greater than or equal to 0"),
            (1.1, "Input should be less than or equal to 1"),
        ],
    )
    def test_ci_level_limits(self, ci_level: float, match_expr: str) -> None:
        """Test that ci_level must be within [0, 1]."""
        with pytest.raises(ValidationError, match=match_expr):
            MetricsConfig.model_validate(self._get_valid_config_dict(ci_level=ci_level))

    def test_extra_fields_forbidden(self) -> None:
        """Test that extra fields at the metrics config level are forbidden."""
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            MetricsConfig.model_validate(
                self._get_valid_config_dict(unexpected_flag=True)
            )


class TestFairnessConfig:
    """Test class for the FairnessConfig class"""

    def _get_valid_config_dict(self, **overrides) -> dict:
        """Creates a fresh valid config dict to override."""
        config_dict = {
            "strata": ["AGE", "SEX"],
            "groups": {"AGE": [[18, 50], [51, 120]]},
        }
        config_dict.update(overrides)

        return config_dict

    def test_valid_config(self) -> None:
        """Pass valid configuration to TestFairnessConfig."""
        raw_config = self._get_valid_config_dict()
        config = FairnessConfig.model_validate(raw_config)

        assert config.model_dump() == raw_config

    def test_groups_optional(self) -> None:
        """Test that groups defaults to None when omitted."""
        config = FairnessConfig.model_validate({"strata": ["AGE", "SEX"]})

        assert config.groups is None

    def test_strata_required(self) -> None:
        """Test that strata is a required field."""
        with pytest.raises(ValidationError, match="Field required"):
            FairnessConfig.model_validate({})

    def test_group_not_in_strata(self) -> None:
        """Test case when groups are not in strata."""
        with pytest.raises(
            ValidationError, match="invalid should be in the strata list"
        ):
            FairnessConfig.model_validate(
                self._get_valid_config_dict(groups={"invalid": []})
            )

    def test_group_is_empty(self) -> None:
        """Test case when group has an empty list as value."""
        with pytest.raises(ValidationError, match="AGE should not have an empty list"):
            FairnessConfig.model_validate(
                self._get_valid_config_dict(groups={"AGE": []})
            )

    def test_extra_fields_forbidden(self) -> None:
        """Test that extra fields at the fairness config level are forbidden."""
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            FairnessConfig.model_validate(
                self._get_valid_config_dict(unexpected_flag=True)
            )


class TestEvaluationSubConfig:
    """Test class for the EvaluationSubConfig class"""

    def _get_valid_config_dict(self, **overrides) -> dict:
        """Creates a fresh valid config dict to override."""
        config_dict = {
            "metrics": {
                "metrics": ["roc_auc", "ici"],
                "n_bootstraps": 1000,
                "ci_level": 0.95,
                "cv_splines": 3,
            },
            "fairness": {
                "strata": ["AGE", "SEX"],
                "groups": {"AGE": [[18, 50], [51, 120]]},
            },
        }
        config_dict.update(overrides)

        return config_dict

    def test_valid_config(self) -> None:
        """Pass valid configuration to TestEvaluationSubConfig."""
        raw_config = self._get_valid_config_dict()
        config = EvaluationSubConfig.model_validate(raw_config)

        assert config.model_dump() == raw_config

    def test_fairness_optional(self) -> None:
        """Test that fairness defaults to None when omitted."""
        raw_config = self._get_valid_config_dict()
        del raw_config["fairness"]

        config = EvaluationSubConfig.model_validate(raw_config)

        assert config.fairness is None

    def test_metrics_required(self) -> None:
        """Test that metrics is a required field."""
        with pytest.raises(ValidationError, match="Field required"):
            EvaluationSubConfig.model_validate({})

    def test_extra_fields_forbidden(self) -> None:
        """Test that extra fields at the evaluation sub config level are
        forbidden."""
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            EvaluationSubConfig.model_validate(
                self._get_valid_config_dict(unexpected_flag=True)
            )
