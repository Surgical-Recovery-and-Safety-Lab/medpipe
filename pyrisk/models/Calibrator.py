"""
Calibrator class.

This class creates a Calibrator to calibrate predictions.

"""

from numpy import array, round

from .core import create_model, get_full_proba

SCRIPT_NAME = "data/Calibrator"


class Calibrator:
    """
    Class that creates a Calibrator.

    Attributes
    ----------
    model : list[LogisticRegression or IsotonicRegression]
        List of calibrator model (one per class).
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
        hyperparameters={},
        n_classes=1,
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
        self.hyperparameters = hyperparameters
        self.n_classes = n_classes
        self.logger = logger
        self.model = []  # Empty list of models (one per class)

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
        for i in range(self.n_classes):
            self.model.append(
                create_model(
                    self.model_type,
                    n_classes=1,
                    logger=self.logger,
                    **self.hyperparameters,
                )
            )

    def fit(self, X, y):
        """
        Fits the predictor to the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_classes)
            Training data.
        y : array-like of shape (n_samples, n_classes)
            Prediction data.

        Returns
        -------
        None
            Nothing is returned.

        """
        for i in range(self.n_classes):
            self.model[i].fit(X[:, i].reshape(-1, 1), y[:, i])

    def predict_proba(self, X):
        """
        Predicts probabilities from input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_classes)
            Training data.

        Returns
        -------
        probabilities : np.array (n_classes,) of arrays (n_samples, 2)
            Predicted probabilities.

        """
        predictions = []
        for i in range(self.n_classes):
            if self.model_type == "isotonic":
                predictions.append(self.model[i].predict(X[:, i].reshape(-1, 1)))
            else:
                predictions.append(self.model[i].predict_proba(X[:, i].reshape(-1, 1)))
        if self.model_type == "isotonic":
            return get_full_proba(array(predictions).T)
        else:
            return predictions

    def predict(self, X):
        """
        Predicts labels from input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_classes)
            Training data.

        Returns
        -------
        predictions : array-like of shape (n_samples, n_classes)
            Predicted labels.

        """
        labels = []
        for i in range(self.n_classes):
            labels.append(round(self.model[i].predict(X[:, i].reshape(-1, 1))))
        return array(labels).T
