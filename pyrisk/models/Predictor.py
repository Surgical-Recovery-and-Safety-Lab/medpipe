"""
Predictor class.

This class creates a Predictor to train and make predictions.

"""

from copy import deepcopy

from torch.accelerator import current_accelerator, is_available

from .core import create_model

SCRIPT_NAME = "data/Predictor"


class Predictor:
    """
    Class that creates a Predictor.

    Attributes
    ----------
    model : HistGradBoostingClassifier, SVC or AIRiskNN.
        Predictor model.
    model_type : {"hgb", "svm", "nn"}
        Model type.
    n_features : int
        Number of input features for the predictor.
    n_classes : int
        Number of classes to predict.
    hyperparameters : dict[str, value]
        Model hyperparameter dictionary.
    architecture : dict[str, value] or None
        Architecture dictionary for neural networks only.
    logger : logging.Logger or None, default: None
        Logger object to log prints. If None print to terminal.

    Methods
    -------
    __init__(predictor_config, logger)
        Init method.
    _set_model()
        Set the model to default parameters.
    fit(X_train, y_train, X_test=[], y_test=[], weights=None):
        Fits the predictor based on input data.
    predict_proba(X)
        Predicts probabilities from input data.
    predict(X)
        Predicts labels from input data.
    """

    def __init__(
        self,
        model_type,
        hyperparameters,
        n_features=-1,
        n_classes=1,
        architecture={},
        logger=None,
    ):
        """
        Initialise a Predictor class instance.

        Parameters
        ----------
        model_type : {"hgb", "svm", "nn"}
            Model type.
        hyperparameters : dict[str, value]
            Model hyperparameter dictionary.
        n_features : int, default: -1
            Number of input features for the predictor.
        n_classes : int, default: 1
            Number of classes to predict.
        architecture : dict[str, value], default: {}
            Architecture dictionary for neural networks only.
        logger : logging.Logger or None, default: None
            Logger object to log prints. If None print to terminal.

        Returns
        -------
        None
            Nothing is returned.

        """
        self.model_type = model_type
        self.n_features = n_features
        self.n_classes = n_classes
        self.hyperparameters = hyperparameters
        self.architecture = architecture
        self.logger = logger

        # Create model based on attributes
        self._set_model()

        # Create device to load the model on GPU if available
        self.device = current_accelerator().type if is_available() else "cpu"

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
        if self.model_type == "nn":
            self.model = create_model(
                self.model_type,
                self.n_features,
                self.n_classes,
                self.logger,
                **deepcopy(self.architecture),
            )

        else:
            self.model = create_model(
                self.model_type,
                self.n_features,
                self.n_classes,
                self.logger,
                **self.hyperparameters,
            )

    def fit(self, X_train, y_train, X_test=[], y_test=[], weights=None):
        """
        Fits the predictor to the training data.

        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            Training data.
        y_train : array-like of shape (n_samples, n_classes)
            Prediction labels.
        X_test : array-like, default: []
            Test data.
        y_test : array-like, default: []
            Test labels.
        weights : array-like or None, default: None
            Weights to address class imbalance.
            Class weights for nn of shape (n_classes,).
            Sample weights for hgb and svm of shape (n_samples,).

        Returns
        -------
        None
            Nothing is returned.

        """
        if self.model_type == "nn":
            if weights is None:
                weights = []
            self.model.fit(
                X_train,
                y_train,
                X_test=X_test,
                y_test=y_test,
                class_weights=weights,
                **self.hyperparameters,
            )

        else:
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
        probabilities : np.array (n_classes,) of arrays (n_samples, 2)
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
        labels : array-like of shape (n_samples, n_classes)
            Predicted labels.

        """
        return self.model.predict(X)
