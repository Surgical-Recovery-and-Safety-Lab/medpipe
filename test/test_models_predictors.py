import pytest

# Assuming your file is named predictors.py
from medpipe.models.predictors import BasePredictor, HGBClassifier, create_predictor

# --- Test Functions ---


def test_cannot_instantiate_base_predictor():
    """Ensures the Abstract Base Class cannot be created directly."""
    from medpipe.models.predictors import BasePredictor

    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        BasePredictor(model_type="hgb-c")


@pytest.mark.parametrize(
    "model_type, output_model_type",
    [("hgb-c", HGBClassifier)],
)
def test_create_predictor_success(model_type, output_model_type):
    """Test that the create_predictor function creates predictors"""
    predictor = create_predictor(model_type)

    assert type(predictor) is output_model_type


def test_create_predictor_value_error():
    """Test that the create_predictor function throws a ValueError when wrong type"""
    with pytest.raises(
        ValueError,
        match="model type should be hgb-c, but got wrong_type",
    ):
        create_predictor("wrong_type")
