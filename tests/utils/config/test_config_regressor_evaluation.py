"""
Test functions for the RegressorMetricsConfig and RegressorEvaluationSubConfig
schemas of the config module.
"""

from re import escape

import pytest
from pydantic import ValidationError

from medpipe.metrics.core import METRICS
from medpipe.utils.config import RegressorEvaluationSubConfig, RegressorMetricsConfig


class TestRegressorMetricsConfig:
    """Test class for the RegressorMetricsConfig class"""

    def _get_valid_config_dict(self, **overrides) -> dict:
        """Creates a fresh valid config dict to override."""
        config_dict = {
            "metrics": ["rmse", "mae"],
            "n_bootstraps": 1000,
            "ci_level": 0.95,
        }
        config_dict.update(overrides)

        return config_dict

    def test_valid_config(self) -> None:
        """Pass valid configuration to RegressorMetricsConfig."""
        raw_config = self._get_valid_config_dict()
        config = RegressorMetricsConfig.model_validate(raw_config)

        assert config.model_dump() == raw_config

    def test_default_values(self) -> None:
        """Test default metrics, n_bootstraps, and ci_level when omitted."""
        config = RegressorMetricsConfig.model_validate({})

        assert config.metrics == ["rmse", "mae"]
        assert config.n_bootstraps == 200
        assert config.ci_level == 0.95

    def test_invalid_metric(self) -> None:
        """Test case when invalid metric is provided."""
        match_expr = (
            "invalid was not found in available metric "
            f"list. Available metrics are {METRICS}"
        )

        with pytest.raises(ValidationError, match=escape(match_expr)):
            RegressorMetricsConfig.model_validate(
                self._get_valid_config_dict(metrics=["invalid"])
            )

    def test_n_bootstraps_limit(self) -> None:
        """Test that n_bootstraps must be non-negative."""
        with pytest.raises(
            ValidationError, match="Input should be greater than or equal to 0"
        ):
            RegressorMetricsConfig.model_validate(
                self._get_valid_config_dict(n_bootstraps=-1)
            )

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
            RegressorMetricsConfig.model_validate(
                self._get_valid_config_dict(ci_level=ci_level)
            )

    def test_extra_fields_forbidden(self) -> None:
        """Test that extra fields at the metrics config level are forbidden."""
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            RegressorMetricsConfig.model_validate(
                self._get_valid_config_dict(unexpected_flag=True)
            )


class TestRegressorEvaluationSubConfig:
    """Test class for the RegressorEvaluationSubConfig class"""

    def _get_valid_config_dict(self, **overrides) -> dict:
        """Creates a fresh valid config dict to override."""
        config_dict = {
            "metrics": {
                "metrics": ["rmse", "mae"],
                "n_bootstraps": 1000,
                "ci_level": 0.95,
            },
            "fairness": {
                "strata": ["AGE", "SEX"],
                "groups": {"AGE": [[18, 50], [51, 120]]},
            },
        }
        config_dict.update(overrides)

        return config_dict

    def test_valid_config(self) -> None:
        """Pass valid configuration to RegressorEvaluationSubConfig."""
        raw_config = self._get_valid_config_dict()
        config = RegressorEvaluationSubConfig.model_validate(raw_config)

        assert config.model_dump() == raw_config

    def test_fairness_optional(self) -> None:
        """Test that fairness defaults to None when omitted."""
        raw_config = self._get_valid_config_dict()
        del raw_config["fairness"]

        config = RegressorEvaluationSubConfig.model_validate(raw_config)

        assert config.fairness is None

    def test_metrics_required(self) -> None:
        """Test that metrics is a required field."""
        with pytest.raises(ValidationError, match="Field required"):
            RegressorEvaluationSubConfig.model_validate({})

    def test_extra_fields_forbidden(self) -> None:
        """Test that extra fields at the evaluation sub config level are
        forbidden."""
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            RegressorEvaluationSubConfig.model_validate(
                self._get_valid_config_dict(unexpected_flag=True)
            )
