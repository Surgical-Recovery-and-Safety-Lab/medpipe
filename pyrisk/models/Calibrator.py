"""
Calibrator class.

This class creates a Calibrator to calibrate predictions.

"""

from numpy import array, expand_dims, round

from .core import create_model, get_full_proba

SCRIPT_NAME = "models/Calibrator"


class Calibrator:
    """
    Class that creates a Calibrator.

    Attributes
    ----------
    model : LogisticRegression or IsotonicRegression
        Calibrator model.
    model_type : {"logistic", "isotonic"}
        Model type.
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
        hyperparameters={},
        logger=None,
    ):
        """
        Initialise a Calibrator class instance.

        Parameters
        ----------
        model_type : {"logistic", "isotonic"}
            Model type.
        hyperparameters : dict[str, value]
            Model hyperparameter dictionary.
        logger : logging.Logger or None, default: None
            Logger object to log prints. If None print to terminal.

        Returns
        -------
        None
            Nothing is returned.

        """
        self.model_type = model_type
        self.hyperparameters = hyperparameters
        self.logger = logger
        self.model = []  # Empty list of models (one per class)

        # Create model based on attributes
        self._set_model()

    def _set_model(self, quiet: bool = False):
        """
        Set the model to default parameters.

        Parameters
        ----------
        quiet : bool, default: False
            Flag to create a model without printing.

        Returns
        -------
        None
            Nothing is returned.

        """
        self.model = create_model(
            self.model_type,
            logger=self.logger,
            quiet=quiet,
            **self.hyperparameters,
        )

    def fit(self, X, y):
        """
        Fits the predictor to the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples,)
            Training data.
        y : array-like of shape (n_samples,)
            Prediction data.

        Returns
        -------
        None
            Nothing is returned.

        """
        self.model.fit(X, y)

    def predict_proba(self, X):
        """
        Predicts probabilities from input data.

        Parameters
        ----------
        X : array-like of shape (n_samples,)
            Training data.

        Returns
        -------
        probabilities : np.array of shape (n_samples, 2)
            Predicted probabilities.

        """
        if self.model_type == "isotonic":
            predictions = self.model.predict(X)
            return get_full_proba(expand_dims(predictions, 1))
        else:
            return self.model.predict_proba(X)

    def predict(self, X):
        """
        Predicts labels from input data.

        Parameters
        ----------
        X : array-like of shape (n_samples,)
            Training data.

        Returns
        -------
        predictions : array-like of shape (n_samples,)
            Predicted labels.

        """
        labels = round(self.model.predict(X))
        return array(labels).T
