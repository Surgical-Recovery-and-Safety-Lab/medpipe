"""
Predictor class.

This class creates a Predictor to train and make predictions.

"""

from copy import deepcopy

from numpy import array, expand_dims, ones
from torch.accelerator import current_accelerator, is_available

import pyrisk.data.weighting as weight
from pyrisk.data.sampler import data_sampler

from .core import create_model

SCRIPT_NAME = "data/Predictor"


class Predictor:
    """
    Class that creates a Predictor.

    Attributes
    ----------
    model : list[HistGradBoostingClassifier or SVC] or AIRiskNN.
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
    _sample_data(X, y, groups)
        Samples the data based on configuration.
    _weight_data(y)
        Gets the weights for the data based on configuration.
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
        if self.model_type == "nn":
            self.model = create_model(
                self.model_type,
                self.n_features,
                self.n_classes,
                self.logger,
                quiet=quiet,
                **deepcopy(self.architecture),
            )

        else:
            self.model = []
            for i in range(self.n_classes):
                self.model.append(
                    create_model(
                        self.model_type,
                        self.n_features,
                        1,
                        self.logger,
                        quiet=quiet,
                        **self.hyperparameters,
                    )
                )

    def fit(
        self,
        X_train,
        y_train,
        X_test=[],
        y_test=[],
        groups=[],
        sampler_config={},
        weighting_config={},
    ):
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
        groups : array-like
            List of groups in which labels belong of shape (n_samples,).
        sampler_config : dict[str, str]
            Configuration parameters for the sampler function.
        weighting_config : dict[str, str]
            Configuration parameters for the weighting function.

        Returns
        -------
        None
            Nothing is returned.

        """
        if self.model_type == "nn":
            # Sample and weight if needed
            X_train, y_train, _ = self._sample_data(
                X_train, y_train, groups, sampler_config
            )
            weights = self._weight_data(y_train, weighting_config)

            self.model.fit(
                X_train,
                y_train,
                X_test=X_test,
                y_test=y_test,
                class_weights=weights,
                **self.hyperparameters,
            )

        else:
            for i in range(self.n_classes):
                # Sample and weight if needed
                X, y, _ = self._sample_data(
                    X_train, expand_dims(y_train[:, i], 1), groups, sampler_config
                )
                weights = self._weight_data(y, weighting_config)

                self.model[i].fit(X, y.squeeze(), sample_weight=weights)

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
        if self.model_type == "nn":
            return self.model.predict_proba(X)
        else:
            predictions = []
            for i in range(self.n_classes):
                predictions.append(self.model[i].predict_proba(X[:, i].reshape(-1, 1)))
            return predictions

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
        if self.model_type == "nn":
            return self.model.predict(X)
        else:
            labels = []
            for i in range(self.n_classes):
                labels.append(round(self.model[i].predict(X[:, i].reshape(-1, 1))))
            return array(labels).T

    def _sample_data(self, X, y, groups, sampler_config):
        """
        Samples the data based on configuration.

        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features)
            Data to sample.
        y : np.array of shape (n_samples,)
            Labels to sample.
        groups : pd.Series of shape (n_samples,) or None
            Groups of the examples, None if not specified.
        sampler_config : dict[str, str]
            Configuration parameters for the sampler function.

        Returns
        -------
        X_sampled : pd.DataFrame of shape (n_sampled_samples, n_features)
            Sampled data.
        y_sampled : np.array of shape (n_sampled_samples,)
            Sampled labels.
        groups_sampled : pd.Series of shape(n_sampled_samples,) or None
            Groups of the examples, None if not specified.

        """
        sampler_fn = sampler_config["sampler_fn"]

        if sampler_fn:
            return data_sampler(X, y, groups=groups, **sampler_config)

        return X, y, groups

    def _weight_data(self, y, weighting_config):
        """
        Gets the weights for the data based on configuration.

        Parameters
        ----------
        y : np.array of shape (n_samples,)
            Labels needed for creation of weights.
        weighting_config : dict[str, str]
            Configuration parameters for the weighting function.

        Returns
        -------
        weights : np.array of shape (n_samples,) or (n_classes,) or None
            Sample or class weights based on labels.
            Sample weights are of shape (n_samples,).
            Class weights are of shape (n_classes,).
            None if no weighting function is provided.

        """
        weighting_fn = weighting_config["weighting_fn"]
        if weighting_fn:
            return getattr(weight, weighting_fn)(y)

        if self.model_type == "nn":
            if len(y) > 1:
                return ones(y.shape[1])
            return array([1])

        return ones(y.shape[0])
