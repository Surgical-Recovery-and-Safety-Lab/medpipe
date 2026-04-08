import pytest

# Assuming your file is named calibrators.py
from medpipe.models.calibrators import (
    IsotonicCalibrator,
    LogisticCalibrator,
    create_calibrator,
)

# --- Test Functions ---


def test_cannot_instantiate_base_calibrator():
    """Ensures the Abstract Base Class cannot be created directly."""
    from medpipe.models.calibrators import BaseCalibrator

    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        BaseCalibrator(model_type="logistic")


@pytest.mark.parametrize(
    "model_type, output_model_type",
    [("isotonic", IsotonicCalibrator), ("logistic", LogisticCalibrator)],
)
def test_create_calibrator_success(model_type, output_model_type):
    """Test that the create_calibrator function creates calibrators"""
    calibrator = create_calibrator(model_type)

    assert type(calibrator) is output_model_type


def test_create_calibrator_value_error():
    """Test that the create_calibrator function throws a ValueError when wrong type"""
    with pytest.raises(
        ValueError,
        match="model type should be isotonic or logistic, but got wrong_type",
    ):
        create_calibrator("wrong_type")
