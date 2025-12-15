"""
Calibrator class.

This class creates a Calibrator to train and make predictions.

"""

from numpy import round

from .core import create_model

SCRIPT_NAME = "data/Calibrator"


class Calibrator:
    """
    Class that creates a Calibrator.

    Attributes
    ----------
    model : LogisticRegression, IsotonicRegression, or MultiOutputRegressor
        Calibrator model.
    model_type : {"logistic", "isotonic"}
        Model type.
    n_classes : int
        Number of classes to predict.
    logger : logging.Logger or None, default: None
        Logger object to log prints. If None print to terminal.

    Methods
    -------
    __init__(predictor_config, logger)
        Init method.
    _set_model()
        Set the model to default parameters.
    fit(X, y)
        Fits the predictor based on input data.
    predict_proba(X)
        Predicts probabilities from input data.
    predict(X)
        Predicts labels from input data.
    """

    def __init__(
        self,
        model_type,
        n_classes=1,
        logger=None,
    ):
        """
        Initialise a Calibrator class instance.

        Parameters
        ----------
        model_type : {"logistic", "isotonic"}
            Model type.
        n_classes : int, default: 1
            Number of classes to predict.
        logger : logging.Logger or None, default: None
            Logger object to log prints. If None print to terminal.

        Returns
        -------
        None
            Nothing is returned.

        """
        self.model_type = model_type
        self.n_classes = n_classes
        self.logger = logger

        # Create model based on attributes
        self._set_model()

    def _set_model(self):
        """
        Set the model to default parameters.

        Parameters
        ----------
        None
            No parameters

        Returns
        -------
        None
            Nothing is returned.

        """
        self.model = create_model(
            self.model_type, n_classes=self.n_classes, logger=self.logger
        )

    def fit(self, X, y):
        """
        Fits the predictor to the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples, n_classes)
            Prediction data.

        Returns
        -------
        None
            Nothing is returned.

        """
        self.model.fit(X, y.squeeze())

    def predict_proba(self, X):
        """
        Predicts probabilities from input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        predictions : array-like of shape (n_samples, n_classes)
            Predicted labels.

        """
        return self.model.predict(X)

    def predict(self, X):
        """
        Predicts labels from input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        predictions : array-like of shape (n_samples, n_classes)
            Predicted labels.

        """
        return round(self.model.predict(X))
