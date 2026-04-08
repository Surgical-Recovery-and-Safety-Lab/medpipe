from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Assuming your file is named calibrators.py
from medpipe.models.calibrators import IsotonicCalibrator, LogisticCalibrator

# --- Fixtures ---


@pytest.fixture
def sample_data():
    """Provides consistent dummy data for tests."""
    X = np.array([0.1, 0.4, 0.7, 0.9])
    y = np.array([0, 0, 1, 1])
    return X, y


# --- Test Functions ---


@patch("medpipe.models.calibrators.create_model")
def test_logistic_calibrator_fit_reshapes_input(mock_create, sample_data):
    """Verifies LogisticCalibrator reshapes 1D input to 2D during fit."""
    X, y = sample_data
    mock_model = MagicMock()
    mock_create.return_value = mock_model

    calibrator = LogisticCalibrator()
    calibrator.fit(X, y)

    # Check if the internal sklearn model received (4, 1) instead of (4,)
    passed_X = mock_model.fit.call_args[0][0]
    assert passed_X.shape == (4, 1)


@patch("medpipe.models.calibrators.create_model")
def test_isotonic_calibrator_predict_proba_format(mock_create, sample_data):
    """Verifies IsotonicCalibrator returns a (N, 2) array."""
    X, _ = sample_data
    mock_model = MagicMock()
    # Isotonic.predict returns 1D probas
    mock_model.predict.return_value = np.array([0.1, 0.2, 0.8, 0.9])
    mock_create.return_value = mock_model

    calibrator = IsotonicCalibrator()
    probas = calibrator.predict_proba(X)

    assert probas.shape == (4, 2)
    # Ensure probabilities sum to 1 across the columns
    np.testing.assert_allclose(probas.sum(axis=1), 1.0)


@patch("medpipe.models.calibrators.create_model")
def test_base_predict_rounding_logic(mock_create, sample_data):
    """Tests that the base class properly rounds float predictions to int labels."""
    X, _ = sample_data
    mock_model = MagicMock()
    # Mocking regressor-style output
    mock_model.predict.return_value = np.array([0.1, 0.49, 0.51, 0.9])
    mock_create.return_value = mock_model

    calibrator = IsotonicCalibrator()
    labels = calibrator.predict(X)

    expected = np.array([0, 0, 1, 1])
    np.testing.assert_array_equal(labels, expected)
    assert np.issubdtype(labels.dtype, np.integer)


def test_cannot_instantiate_base_calibrator():
    """Ensures the Abstract Base Class cannot be created directly."""
    from medpipe.models.calibrators import BaseCalibrator

    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        BaseCalibrator(model_type="logistic")
