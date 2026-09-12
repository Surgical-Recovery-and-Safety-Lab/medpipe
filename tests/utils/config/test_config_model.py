"""
Test functions for the RecalibrationConfig and ModelSetup schemas
of the config module.
"""

import pytest
from pydantic import ValidationError

from medpipe.utils.config import ModelSetup, RecalibrationConfig


class TestRecalibrationConfig:
    """Test class for the RecalibrationConfig class"""

    def _get_valid_config_dict(self, **overrides) -> dict:
        """Creates a fresh valid config dict to override."""
        config_dict = {
            "recalibrate": True,
            "method": "isotonic",
            "hyperparameters": {},
        }
        config_dict.update(overrides)

        return config_dict

    def test_valid_config(self) -> None:
        """Pass valid configuration to RecalibrationConfig."""
        raw_config = self._get_valid_config_dict()
        config = RecalibrationConfig.model_validate(raw_config)

        assert config.model_dump() == raw_config

    @pytest.mark.parametrize("recalibrate_val", [True, False])
    def test_recalibration_flag_boolean_values(self, recalibrate_val: bool) -> None:
        """Pass True and False recalibrate flags to RecalibrationConfig."""
        config = RecalibrationConfig(recalibrate=recalibrate_val, method="isotonic")
        assert config.recalibrate is recalibrate_val

    def test_recalibrate_required(self) -> None:
        """Test that recalibrate is a required field."""
        with pytest.raises(ValidationError, match="Field required"):
            RecalibrationConfig.model_validate({})

    def test_default_method(self) -> None:
        """Test that method defaults to 'isotonic' when omitted."""
        config = RecalibrationConfig.model_validate({"recalibrate": True})

        assert config.method == "isotonic"

    def test_default_hyperparameters(self) -> None:
        """Test that hyperparameters defaults to an empty dict when omitted."""
        config = RecalibrationConfig.model_validate({"recalibrate": True})

        assert config.hyperparameters == {}

    @pytest.mark.parametrize("method", ["sigmoid", "temperature"])
    def test_valid_methods(self, method: str) -> None:
        """Test all supported recalibration methods."""
        config = RecalibrationConfig.model_validate(
            self._get_valid_config_dict(method=method)
        )

        assert config.method == method

    def test_invalid_method(self) -> None:
        """Test that an unsupported method raises a ValidationError."""
        with pytest.raises(
            ValidationError,
            match="Input should be 'isotonic', 'sigmoid' or 'temperature'",
        ):
            RecalibrationConfig.model_validate(
                self._get_valid_config_dict(method="unsupported")
            )

    def test_extra_fields_forbidden(self) -> None:
        """Test that extra fields at the recalibration config level are
        forbidden."""
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            RecalibrationConfig.model_validate(
                self._get_valid_config_dict(unexpected_flag=True)
            )


class TestModelSetupConfig:
    """Test class for the ModelSetup class"""

    def test_model_setup_valid(self) -> None:
        """Test valid model setup with defaults."""
        model = ModelSetup(algorithm="LogisticRegression")
        assert model.algorithm == "LogisticRegression"
        assert model.hyperparameters == {}
        assert model.recalibration is None

    def test_model_setup_with_recalibration(self) -> None:
        """Test valid model setup including recalibration parameters."""
        model = ModelSetup(
            algorithm="XGBClassifier",
            hyperparameters={"learning_rate": 0.1},
            recalibration={  # type: ignore
                "recalibrate": True,
                "method": "isotonic",
                "hyperparameters": {"out_of_bounds": "clip"},
            },
        )
        assert model.recalibration
        assert model.recalibration.recalibrate
        assert model.recalibration.method == "isotonic"
        assert model.recalibration.hyperparameters["out_of_bounds"] == "clip"

    def test_algorithm_required(self) -> None:
        """Test that algorithm is a required field."""
        with pytest.raises(ValidationError, match="Field required"):
            ModelSetup.model_validate({})

    def test_model_setup_extra_forbidden(self) -> None:
        """Test that extra fields at the model setup level are strictly forbidden."""
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            ModelSetup(algorithm="RF", unexpected_flag=True)  # type: ignore
