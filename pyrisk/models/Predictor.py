"""
Predictor class.

This class creates a Predictor to train and make predictions.

"""

from .core import create_model

SCRIPT_NAME = "models/Predictor"


class Predictor:
    """
    Class that creates a Predictor.

    Attributes
    ----------
    model : dict[str, model]
        Predictor model. The key is the predicted label, the model
        is a HistGradientBoostingClassifier.
    model_type : {"hgb-c"}
        Model type.
    hyperparameters : dict[str, value]
        Model hyperparameter dictionary.
    logger : logging.Logger or None, default: None
        Logger object to log prints. If None print to terminal.

    Methods
    -------
    __init__(model_type, hyperparameters, logger=None)
        Initialise a Predictor class instance.
    _set_model()
        Set the model to default parameters.
    fit(X_train, y_train, weights=None):
        Fits the predictor based on input data.
    predict_proba(X)
        Predicts probabilities from input data.
    predict(X)
        Predicts labels from input data.
    """

    def __init__(self, model_type, hyperparameters, logger=None):
        """
        Initialise a Predictor class instance.

        Parameters
        ----------
        model_type : {"hgb-c"}
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
            self.logger,
            quiet=quiet,
            **self.hyperparameters,
        )

    def fit(self, X_train, y_train, weights=None):
        """
        Fits the predictor to the training data.

        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            Training data.
        y_train : array-like of shape (n_samples,)
            Prediction labels.
        X_test : array-like, default: []
            Test data.
        y_test : array-like, default: []
            Test labels.
        weights : array-like of shape (n_samples,) or None, default: None
            Weights to address class imbalance.

        Returns
        -------
        None
            Nothing is returned.

        """
        self.model.fit(X_train, y_train.squeeze(), sample_weight=weights)

    def predict_proba(self, X):
        """
        Predicts probabilities from input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        probabilities : np.array of shape (n_samples, 2)
            Predicted probabilities.

        """
        return self.model.predict_proba(X)

    def predict(self, X):
        """
        Predicts labels from input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        labels : array-like of shape (n_samples,)
            Predicted labels.

        """
        return self.model.predict(X)
