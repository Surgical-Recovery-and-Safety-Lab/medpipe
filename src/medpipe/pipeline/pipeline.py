"""
Pipeline class.

This class creates a Pipeline to prepare data, fit a predictor and
a calibrator.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd
from numpy import arange, array, expand_dims, ones

import medpipe.data.weighting as weight
from medpipe._types import Data, FProbas, Labels, PData, PProbas
from medpipe.data.preprocessing import extract_labels, get_validation_idx, train_test_it
from medpipe.data.preprocessor import Preprocessor
from medpipe.data.sampler import data_sampler
from medpipe.metrics.core import print_metrics
from medpipe.models.calibrators import create_calibrator
from medpipe.models.core import get_positive_proba, test_model
from medpipe.models.predictors import create_predictor
from medpipe.utils.config import get_configuration, split_version_number
from medpipe.utils.logger import print_message

SCRIPT_NAME = "pipeline/pipeline"
if TYPE_CHECKING:
    import logging

    import numpy.typing as npt


class Pipeline:
    """
    Class that creates a Pipeline.

    Attributes
    ----------
    version : str
        Version number.
    label_list : list[str]
        List of labels to predict.
    n_labels : int
        Number of labels to predict.
    predictor_type : str
        Model type of the predictor.
    calibrator_type : str
        Model type of the calibrator.
    preprocessor_config : dict[str, Any]
        Configuration dictionary for the preprocessor.
    predictor_config : dict[str, Any]
        Configuration dictionary for the predictor.
        Model type of the predictor.
    calibrator_config : dict[str, Any]
        Configuration dictionary for the calibrator.
    preprocessor : Preprocessor
        Data preprocessor object.
    predictor : dict[label, Predictor]
        Dictionary of Predictor instances for each label.
    calibrator : dict[label, Calibrator]
        Dictionary of Calibrator instances for each label.
    predictor_probabilities : dict[label, dict[int, array]]
        Dictionary of predicted probabilities for each predictor
        The dictionary keys are the labels and the values are
        the predicted probabilities of the predictor for that fold.
    calibrator_probabilities : dict[label, dict[int, array]]
        Dictionary of predicted probabilities for each calibrator
        The dictionary keys are the labels and the values are
        the predicted probabilities of the calibrator for that fold.
    logger : logging.Logger | None, default: None
        Logger object to log prints. If None print to terminal.

    Methods
    -------
    __init__(pipeline_config={}, logger=None)
        Init method.
    fit_preprocessor(X)
        Fits the preprocessor operations based on input data.
    transform(X)
        Transforms input data based on preprocessor fitted operations.
    fit_transform(X)
        Fits the preprocessor operations and transforms the input data.
    fit_model(X, y, model, **kwargs)
        Fits the predictor or calibrator model on the provided dataset.
    test_model(X, y, model, label_list, key=None)
        Tests the predictor or calibrator model on the provided dataset.
    run(X)
        Run pipeline with input data.
    _train_models(X_train, y_train, X_cal, y_cal, label, **kwargs)
        Trains the predictor and calibrator models.
    predict_proba(X)
        Predicts probabilities from predictor or calibrator based on input data.
    predict(X)
        Predicts labels from predictor or calibrator based on input data.
    _predictor_pred_wrapper(X, label, prediction_type)
        Wrapper function to create predictions with the predictor.
    _calibrator_pred_wrapper(X, label, prediction_type)
        Wrapper function to create predictions with the calibrator.
    _sample_data(X, y, groups)
        Samples the data based on configuration.
    _weight_data(y)
        Gets the weights for the data based on configuration.
    _get_calibrator_data(X, label)
        Get the calibrator data based on the calibrator type.
    """

    def __init__(
        self, pipeline_config: dict[str, Any] = {}, logger: logging.Logger | None = None
    ) -> None:
        """
        Initialise a Pipeline class instance.

        Parameters
        ----------
        pipeline_config : dict[str, Any]
            Configuration parameters for the pipeline object.
        logger : logging.Logger | None, default: None
            Logger object to log prints. If None print to terminal.

        Returns
        -------
        None
            Nothing is returned.

        """
        self.version = pipeline_config["version"]
        self.predictor_type = pipeline_config["predictor_type"]
        self.logger = logger
        self.predictor_probabilities = (
            {}
        )  # Empty dict for predictor predicted probabilities
        self.calibrator_probabilities = (
            {}
        )  # Empty dict for calibrator predicted probabilities

        print_message("Setting up Pipeline", self.logger, SCRIPT_NAME)

        # Get the different configuration dictionaries
        data_version, model_version = split_version_number(pipeline_config["version"])

        # Get predictor configuration parameters
        self.predictor_config = get_configuration(
            pipeline_config["model_parameters"],
            model_version,
        )

        # Get data configuration parameters
        self.preprocessor_config = get_configuration(
            pipeline_config["data_parameters"],
            data_version,
        )

        # Get the calibrator configuration parameters from the predictor config
        self.calibrator_type = self.predictor_config["calibrator"]["calibrator_type"]
        self.calibrator_config = self.predictor_config["calibrator"]

        # Define variables needed to initialise other objects
        self.label_list = self.predictor_config["labels"]["label_list"]
        self.n_labels = len(self.label_list)

        self.predictor = {}
        self.calibrator = {}

        for label in self.label_list:
            self.predictor[label] = create_predictor(
                self.predictor_type,
                hyperparameters=self.predictor_config["hyperparameters"],
                logger=self.logger,
            )

            self.predictor_probabilities[label] = {}
            if self.calibrator_type != "":
                # Only if a calibrator type is provided
                self.calibrator[label] = create_calibrator(
                    self.calibrator_type,
                    hyperparameters=self.calibrator_config["hyperparameters"],
                    logger=self.logger,
                )
                self.calibrator_probabilities[label] = {}
        self.preprocessor = Preprocessor(
            self.preprocessor_config["preprocessing"], logger=self.logger
        )

    def fit_preprocessor(self, X: pd.DataFrame) -> None:
        """
        Fits the preprocessor operations based on input data.

        Parameters
        ----------
        X : pd.Dataframe
            Data of shape (n_samples, n_features) to clean.

        Returns
        -------
        None
            Nothings is returned.

        """
        self.preprocessor.fit(X)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms input data based on preprocessor fitted operations.

        Parameters
        ----------
        X : pd.Dataframe
            Data to clean of shape (n_samples, n_features).

        Returns
        -------
        data : pd.Dataframe
             Transformed data of shape (n_samples, n_features).

        """
        return self.preprocessor.transform(X)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fits the preprocessor operations and transforms the input data.

        Parameters
        ----------
        X : pd.Dataframe
            Data to clean of shape (n_samples, n_features).

        Returns
        -------
        data : pd.Dataframe
             Transformed data of shape (n_samples, n_features).

        """
        return self.preprocessor.fit_transform(X)

    def get_test_data(
        self, X: pd.DataFrame, test_group_vals: list[Any] | None = None
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Returns train and test data based on input data.

        Parameters
        ----------
        X : pd.DataFrame
            Data of shape (n_samples, n_features) to split.
        test_group_vals : list[Any] | None, default: None
            Group values that should be in the test set.

        Returns
        -------
        X_train : pd.DataFrame
            Train set of shape (n_samples, n_features).
        X_test : pd.DataFrame
            Test set of shape (n_samples, n_features).

        """
        split_vars = self.preprocessor_config["split_variables"]

        if split_vars["group_name"] and not test_group_vals is None:
            train_idx, test_idx = get_validation_idx(
                arange(len(X), dtype=int),
                X[split_vars["group_name"]].to_numpy(),
                test_group_vals,
            )
            X_test = X.iloc[test_idx]
            X_test = X_test.drop(split_vars["group_name"], axis=1)

        else:
            # No groups just get specified percent of the data
            train_idx, test_idx = get_validation_idx(
                arange(len(X), dtype=int), val_size=split_vars["test_size"]
            )
            X_test = X.iloc[test_idx]

        X_train = X.iloc[train_idx]

        return X_train, X_test

    def fit_model(
        self,
        X: Data,
        y: Labels,
        model: str,
        label: str,
        **kwargs: Any,
    ) -> None:
        """
        Fits the predictor or calibrator model on the provided dataset.

        Parameters
        ----------
        X : Data
            Training data of shape (n_samples, n_features) or (n_samples,).
        y : Labels
            Prediction labels of shape (n_samples, self.n_labels).
        model : {"predictor", "calibrator"}
            Model to fit.
        label : str
            Label associated with the model to use.
        **kwargs : Any
            Extra arguments for fitting the models.

        Returns
        -------
        None
            Nothing is returned.

        Raises
        ------
        ValueError
            If model is not "predictor" or "calibrator".

        """
        match model:
            case "predictor":
                self.predictor[label].fit(X, y, **kwargs)
            case "calibrator":
                self.calibrator[label].fit(X, y, **kwargs)
            case _:
                raise ValueError(
                    f"Model should be predictor or calibrator, but got {model}"
                )

    def test_model(self, X: Data, y: Labels, model: str, label: str) -> None:
        """
        Tests the predictor or calibrator model on the provided dataset.

        Parameters
        ----------
        X : Data
            Training data of shape (n_samples, n_features).
        y : Labels
            Prediction labels of shape (n_samples,).
        model : {"predictor", "calibrator"}
            Model to test.
        label : str
            Label associated with the model to use.

        Returns
        -------
        None
            Nothing is returned.

        Raises
        ------
        ValueError
            If model is not "predictor" or "calibrator".

        """
        match model:
            case "predictor":
                message = "Uncalibrated metrics"

            case "calibrator":
                message = "Calibrated metrics"

            case _:
                raise ValueError(
                    f"Model should be predictor or calibrator, but got {model}"
                )

        metric_dict = test_model(
            y,
            self.predict(X, label_list=label, model_type=model),
            array(self.predict_proba(X, label_list=label, model_type=model)),
        )
        print_message(message, self.logger, SCRIPT_NAME)
        print_metrics(metric_dict, [label], self.logger)

    def run(self, X: pd.DataFrame) -> None:
        """
        Run pipeline with input data.

        Parameters
        ----------
        X : pd.DataFrame
            Training data of shape (n_samples, n_features).

        Returns
        -------
        None
            Nothing is returned.

        """
        if self.preprocessor.operations:
            # If operations are already set then simply transform the data
            data = self.transform(X)
        else:
            # Fit and transform
            data = self.fit_transform(X)

        # Get test and cv groups and drop flags
        cv_group_name = self.preprocessor_config["cv_variables"]["group_name"]
        cv_drop = self.preprocessor_config["cv_variables"].pop("drop")
        split_group_name = self.preprocessor_config["split_variables"]["group_name"]
        split_drop = self.preprocessor_config["split_variables"].pop("drop")

        weights = None
        X, y = extract_labels(data, self.label_list)  # Get prediction labels from data

        # Get the groups for splitting
        cv_groups = pd.Series(data[cv_group_name]) if cv_group_name else None
        split_groups = (
            pd.Series(data[split_group_name]) if split_group_name else pd.Series([])
        )

        # Create independent calibration set if calibrator is specified
        if self.calibrator_type != "":
            train_idx, val_idx = get_validation_idx(arange(len(y)), groups=split_groups)
            X_cal = X.iloc[val_idx]
            y_cal = y[val_idx]
            X = X.iloc[train_idx]
            y = y[train_idx]

            if cv_groups is not None:
                cv_groups = cv_groups.iloc[train_idx]
                if split_drop and split_group_name:
                    X_cal = X_cal.drop(split_group_name, axis=1)

        if split_drop and split_group_name and split_group_name != cv_group_name:
            # Drop test/train split group from data
            X = X.drop(split_group_name, axis=1)

        kfold_it = train_test_it(**self.preprocessor_config["cv_variables"])
        n_folds = kfold_it.get_n_splits(X, y[:, 0], groups=cv_groups)

        for i, (train_idx, test_idx) in enumerate(
            kfold_it.split(X, y[:, 0], groups=cv_groups)
        ):
            if cv_groups is not None:
                X_fold = X.drop(cv_groups.name, axis=1) if cv_drop else X
                fold = cv_groups.iloc[test_idx[0]]  # Use group as fold key
                fold_groups = cv_groups.iloc[train_idx]
                fold_message = f"  Fold number {fold} ({i+1}/{n_folds})"
            else:
                X_fold = X
                fold = i
                fold_groups = pd.Series([])
                fold_message = f"  Fold number {fold+1}/{n_folds}"

            # Create the different data sets
            X_train = X_fold.iloc[train_idx]
            y_train = y[train_idx]
            X_test = X_fold.iloc[test_idx]
            y_test = y[test_idx]

            for j, label in enumerate(self.label_list):
                # Sample and weight data if needed
                X_train_i, y_train_i, _ = self._sample_data(
                    X_train, expand_dims(y_train[:, j], 1), fold_groups
                )
                weights = self._weight_data(y_train_i)

                print_message(
                    f"Current metric: {self.label_list[j]}", self.logger, SCRIPT_NAME
                )
                print_message(fold_message, self.logger, SCRIPT_NAME)
                print_message(
                    f"  Train set size: {len(X_train_i)} examples",
                    self.logger,
                    SCRIPT_NAME,
                )
                print_message(
                    f"  Calibration set size: {len(X_cal)} examples",
                    self.logger,
                    SCRIPT_NAME,
                )
                print_message(
                    f"  Test set size: {len(X_test)} examples", self.logger, SCRIPT_NAME
                )

                if self.calibrator_type != "":
                    self._train_models(
                        X_train_i,
                        y_train_i,
                        label,
                        X_cal,
                        y_cal[:, j],
                        **{"weights": weights},
                    )

                    # Test, save probabilities, and reset calibrator
                    self.test_model(X_test, y_test[:, j].squeeze(), "calibrator", label)
                    self.calibrator_probabilities[label][fold] = get_positive_proba(
                        self.predict_proba(X_test, label, model_type="calibrator")
                    )
                    self.calibrator[label]._set_model(quiet=True)

                else:
                    # Train only predictor if no calibrator specified
                    self._train_models(
                        X_train_i, y_train_i, label, **{"weights": weights}
                    )

                # Test predictor on test set
                self.test_model(X_test, y_test[:, j].squeeze(), "predictor", label)

                # Save positive class predicted probabilities
                self.predictor_probabilities[label][fold] = get_positive_proba(
                    self.predict_proba(X_test, label, model_type="predictor")
                )

                # Rest predictor without printing
                self.predictor[label]._set_model(quiet=True)

        # Train final model on complete training set
        print_message("  Final training on all examples", self.logger, SCRIPT_NAME)
        if cv_groups is not None:
            if cv_drop:
                # Drop group names for final dataset if needed
                X = X.drop(cv_groups.name, axis=1)
        else:
            # Convert to a pd.Series for final sampling
            cv_groups = pd.Series([])

        for k, label in enumerate(self.label_list):
            X_train, y_train, _ = self._sample_data(
                X, expand_dims(y[:, k], 1), cv_groups
            )
            weights = self._weight_data(y_train)

            print_message(
                f"Current metric: {self.label_list[k]}", self.logger, SCRIPT_NAME
            )
            print_message(
                f"  Train set size: {len(X_train)} examples",
                self.logger,
                SCRIPT_NAME,
            )

            if self.calibrator_type != "":
                self._train_models(
                    X_train, y_train, label, X_cal, y_cal[:, k], **{"weights": weights}
                )
            else:
                self._train_models(X_train, y_train, label, **{"weights": weights})

    def _train_models(
        self,
        X_train: PData,
        y_train: Labels,
        label: str,
        X_cal: PProbas = array([]),
        y_cal: Labels = array([]),
        **kwargs: Any,
    ) -> None:
        """
        Trains the predictor and calibrator models.

        The calibrator is trained only if X_cal and y_cal are specified.

        Parameters
        ----------
        X_train : Data
            Train data of shape (n_samples, n_features) for the predictor.
        y_train : Labels
            Train labels of shape (n_samples,) for the predictor.
        label: str
            Label associated with the model to train.
        X_cal : PProbas, default: np.array([])
            Calibration data of shape (n_samples,) for the calibrator.
        y_cal : Labels, default: np.array([])
            Calibration labels of shape (n_samples,) for the calibrator.
        **kwargs : Any
            Extra arguments for fitting the predictor.

        Returns
        -------
        None
            Nothing is returned.

        """
        # Fit predictor on train set
        self.fit_model(X_train, y_train, "predictor", label, **kwargs)

        # Fit calibrator on validation set
        if self.calibrator_type != "":
            self.fit_model(
                self._get_calibrator_data(X_cal, label),
                y_cal,
                "calibrator",
                label,
            )

    def predict_proba(
        self,
        X: Data,
        label_list: str | list[str] = "all",
        model_type: str = "predictor",
    ) -> FProbas:
        """
        Predicts probabilities from predictor or calibrator based on input data.

        Parameters
        ----------
        X : Data
            Data to use of shape (n_samples, n_features) or (n_samples,).
        label_list : str | list[str], default: "all"
            Label or list of labels associated with the model to use.
            If all, all models are used.
        model_type : {"predictor", "calibrator"}, default: "predictor"
            Model to use.

        Returns
        -------
        probabilities : FProbas
            Full predicted probabilities of shape (n_samples, 2).

        Raises
        ------
        ValueError
            If model is not "predictor" or "calibrator".
        TypeError
            If label_list is not str or list.

        """
        match model_type:
            case "predictor":
                pred_fn = self._predictor_pred_wrapper
            case "calibrator":
                pred_fn = self._calibrator_pred_wrapper
            case _:
                raise ValueError(
                    f"Model should be predictor or calibrator, but got {model_type}"
                )

        if type(label_list) is str:
            if label_list == "all":
                # Convert to list of all labels
                label_list = self.label_list
            else:
                # Single label
                return pred_fn(X, label_list, "predict_proba")

        if type(label_list) is not type([]):
            raise TypeError(
                f"Label list should be str or list, but got {type(label_list)}"
            )

        probabilities = []
        for label in label_list:
            # Loop over all labels to get probabilities for each model
            pred_probas = pred_fn(X, label, "predict_proba")
            if type(pred_probas) is type([]):
                # Account for potential multilabel
                probabilities += pred_probas
            else:
                probabilities.append(pred_probas)
        return array(probabilities)

    def predict(
        self,
        X: Data,
        label_list: str | list[str] = "all",
        model_type: str = "predictor",
    ) -> Labels:
        """
        Predicts labels from predictor or calibrator based on input data.

        Parameters
        ----------
        X : Data
            Data to use of shape (n_samples, n_features) or (n_samples,).
        label_list : str | list[str], default: "all"
            Label or list of labels associated with the model to use.
            If all, all models are used.
        model_type : {"predictor", "calibrator"}, default: "predictor"
            Model to use.

        Returns
        -------
        labels : Labels
            Predicted labels of shape (n_samples,).

        Raises
        ------
        ValueError
            If model_type is not "predictor" or "calibrator".
        TypeError
            If label_list is not str or list.

        """
        match model_type:
            case "predictor":
                pred_fn = self._predictor_pred_wrapper
            case "calibrator":
                pred_fn = self._calibrator_pred_wrapper
            case _:
                raise ValueError(
                    f"Model should be predictor or calibrator, but got {model_type}"
                )

        if type(label_list) is str:
            if label_list == "all":
                # Convert to list of all labels
                label_list = self.label_list
            else:
                # Single label
                return pred_fn(X, label_list, "predict").astype(int)

        if type(label_list) is not type([]):
            raise TypeError(
                f"Label list should be str or list, but got {type(label_list)}"
            )

        labels = []
        for _label in label_list:
            # Loop over all labels to get labels for each model
            pred_labels = pred_fn(X, _label, "predict")
            if type(pred_labels) is type([]):
                # Account for potential multilabel
                labels += pred_labels
            else:
                labels.append(pred_labels)
        return array(labels)

    def _predictor_pred_wrapper(
        self, X: PData, label: str, prediction_type: str
    ) -> Labels | FProbas:
        """
        Wrapper function to create predictions with the predictor.

        Parameters
        ----------
        X : PData
            Data for predictor classes on of shape (n_samples, n_features).
        label : str
            Label associated with the model to use.
        prediction_type : {"predict", "predict_proba"}
            Prediction function to use.

        Returns
        -------
        estimates : Labels | FProbas
            Labels or probabilities estimated from model based on prediction_type.

        Raises
        ------
        ValueError
            If prediction_type is not "predict" or "predict_proba".


        """
        match prediction_type:
            case "predict":
                return self.predictor[label].predict(X)
            case "predict_proba":
                return self.predictor[label].predict_proba(X)
            case _:
                raise ValueError(
                    "Prediction type should be predict or predict_proba, "
                    f"but got {prediction_type}"
                )

    def _calibrator_pred_wrapper(
        self, X: PProbas, label: str, prediction_type: str
    ) -> Labels | FProbas:
        """
        Wrapper function to create predictions with the calibrator.

        Parameters
        ----------
        X : PProbas
            Data for calibrator class of shape (n_samples,).
        label : str
            Label associated with the model to use.
        prediction_type : {"predict", "predict_proba"}
            Prediction function to use.

        Returns
        -------
        estimates : Labels | FProbas
            Labels or probabilities estimated from model based on prediction_type.

        Raises
        ------
        ValueError
            If prediction_type is not "predict" or "predict_proba".

        """
        match prediction_type:
            case "predict":
                return self.calibrator[label].predict(
                    self._get_calibrator_data(X, label)
                )
            case "predict_proba":
                return self.calibrator[label].predict_proba(
                    self._get_calibrator_data(X, label)
                )
            case _:
                raise ValueError(
                    "Prediction type should be predict or predict_proba, "
                    f"but got {prediction_type}"
                )

    def _sample_data(
        self, X: PData, y: Labels, groups: pd.Series
    ) -> tuple[PData, Labels, pd.Series]:
        """
        Samples the data based on configuration.

        Parameters
        ----------
        X : PData
            Data to sample of shape (n_samples, n_features).
        y : Labels
            Labels to sample of shape (n_samples,).
        groups : pd.Series
            Groups of the examples of shape (n_samples,) or empty.

        Returns
        -------
        X_sampled : pd.DataFrame
            Sampled data of shape (n_sampled_samples, n_features).
        y_sampled : Labels
            Sampled labels of shape (n_sampled_samples,).
        groups_sampled : pd.Series
            Groups of the examples of shape (n_sampled_samples,) or empty

        """
        sampler_fn = self.predictor_config["sampler"]["sampler_fn"]

        if sampler_fn:
            return data_sampler(X, y, groups=groups, **self.predictor_config["sampler"])

        return X, y, groups

    def _weight_data(self, y: Labels) -> npt.NDArray | None:
        """
        Gets the weights for the data based on configuration.

        Parameters
        ----------
        y : Labels
            Labels of shape (n_samples, self.n_labels) needed for creation of weights.

        Returns
        -------
        weights : npt.NDArray or None
            Sample or class weights based on labels.
            Sample weights are of shape (n_samples,).
            Class weights are of shape (1).
            None if no weighting function is provided.

        """
        weighting_fn = self.predictor_config["weighting"]["weighting_fn"]

        if weighting_fn:
            return getattr(weight, weighting_fn)(y)

        return ones(y.shape[0])

    def _get_calibrator_data(self, X: PData, label: str) -> PProbas:
        """
        Get the calibrator data based on the calibrator type.

        Parameters
        ----------
        X : PData
            Data of shape (n_samples, n_features) to transform for calibrator.
        label : str
            Label associated with the model to use.

        Returns
        -------
        calibrator_data : npt.NDArray
            Data of shape (n_samples,) or (n_samples, 2) for calibrator model
            based on its type.

        """
        calibrator_data = get_positive_proba(
            self.predict_proba(X, label, model_type="predictor")
        )

        if self.calibrator_type == "isotonic" and calibrator_data.shape[1] == 2:
            # Only provide positive probabilities
            calibrator_data = calibrator_data[:, 1]

        return calibrator_data
