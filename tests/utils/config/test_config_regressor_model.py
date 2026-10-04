"""
Test functions for the RegressorModelSetup schema of the config module.
"""

import pytest
from pydantic import ValidationError

from medpipe.utils.config import RegressorModelSetup


class TestRegressorModelSetupConfig:
    """Test class for the RegressorModelSetup class"""

    def test_model_setup_valid(self) -> None:
        """Test valid model setup with defaults."""
        model = RegressorModelSetup(algorithm="NGBRegressor")
        assert model.algorithm == "NGBRegressor"
        assert model.hyperparameters == {}

    def test_model_setup_with_hyperparameters(self) -> None:
        """Test valid model setup including hyperparameters."""
        model = RegressorModelSetup(
            algorithm="NGBRegressor",
            hyperparameters={"n_estimators": 300},
        )
        assert model.hyperparameters["n_estimators"] == 300

    def test_algorithm_required(self) -> None:
        """Test that algorithm is a required field."""
        with pytest.raises(ValidationError, match="Field required"):
            RegressorModelSetup.model_validate({})

    def test_no_recalibration_field(self) -> None:
        """Test that recalibration is not a recognized field, since
        post-hoc recalibration is a classification-only concept."""
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            RegressorModelSetup.model_validate(
                {
                    "algorithm": "NGBRegressor",
                    "recalibration": {"recalibrate": True, "method": "isotonic"},
                }
            )

    def test_model_setup_extra_forbidden(self) -> None:
        """Test that extra fields at the model setup level are strictly forbidden."""
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            RegressorModelSetup(algorithm="NGBRegressor", unexpected_flag=True)  # type: ignore
