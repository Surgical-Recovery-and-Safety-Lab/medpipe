"""
Pipeline class.

This class creates a Pipeline to prepare data, fit a predictor and
a calibrator.

"""

from numpy import arange, array, expand_dims, ones

import pyrisk.data.weighting as weight
from pyrisk.data.preprocessing import extract_labels, get_validation_idx, test_train_it
from pyrisk.data.Preprocessor import Preprocessor
from pyrisk.data.sampler import data_sampler
from pyrisk.metrics.core import print_metrics
from pyrisk.models.Calibrator import Calibrator
from pyrisk.models.core import get_positive_proba, test_model
from pyrisk.models.Predictor import Predictor
from pyrisk.utils.config import get_configuration, split_version_number
from pyrisk.utils.logger import print_message

SCRIPT_NAME = "pipeline/Pipeline"


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
    preprocessor_config : dict[str, attr]
        Configuration dictionary for the preprocessor.
    predictor_config : dict[str, attr]
        Configuration dictionary for the predictor.
        Model type of the predictor.
    calibrator_config : dict[str, attr]
        Configuration dictionary for the calibrator.
    preprocessor : Preprocessor
        Data preprocessor object.
    predictor : Predictor
        Prediction model object.
    calibrator : Calibrator
        Calibration model object.
    predictor_metrics : dict[str, dict[str, list[float or tuple(array-like)]]
        Dictionary of the predictor performance.
        Keys are the metric name and values are the metric value.
        The test metrics used are:
         - accuracy
         - f1
         - precision
         - recall
         - log_loss
         - roc (Receiver Operator Characteristic)
         - auroc (Area Under Receiver Operator Characteristic)
         - prc (Precision-Recall Curve)
         - ap (Average Precision)
    calibrator_metrics : dict[str, dict[str, list[float or tuple(array-like)]]
        Dictionary of the calibrator performance.
        Keys are the metric name and values are the metric value.
        The test metrics used are:
         - accuracy
         - f1
         - precision
         - recall
         - log_loss
         - roc (Receiver Operator Characteristic)
         - auroc (Area Under Receiver Operator Characteristic)
         - prc (Precision-Recall Curve)
         - ap (Average Precision)
    logger : logging.Logger or None, default: None
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
    _train_models(X_train, y_train, X_cal, y_cal, idx=0, **kwargs)
        Trains the predictor and calibrator models.
    predict_proba(X)
        Predicts probabilities from predictor or calibrator based on input data.
    predict(X)
        Predicts labels from predictor or calibrator based on input data.
    _predictor_pred_wrapper(X, idx, prediction_type)
        Wrapper function to create predictions with the predictor.
    _calibrator_pred_wrapper(X, idx, prediction_type)
        Wrapper function to create predictions with the calibrator.
    _sample_data(X, y, groups)
        Samples the data based on configuration.
    _weight_data(y)
        Gets the weights for the data based on configuration.
    _get_calibrator_data(X, idx)
        Get the calibrator data based on the calibrator type.
    """

    def __init__(self, pipeline_config={}, logger=None):
        """
        Initialise a Pipeline class instance.

        Parameters
        ----------
        pipeline_config : dict[str, parameters]
            Configuration parameters for the pipeline object.
        logger : logging.Logger or None, default: None
            Logger object to log prints. If None print to terminal.

        Returns
        -------
        None
            Nothing is returned.

        """
        self.version = pipeline_config["version"]
        self.predictor_type = pipeline_config["predictor_type"]
        self.logger = logger
        self.predictor_metrics = {}  # Empty dict for predictor metrics
        self.calibrator_metrics = {}  # Empty dict for calibrator metrics

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
        n_features = len(self.preprocessor_config["features"]["feature_list"]) - len(
            self.label_list
        )

        if self.preprocessor_config["split_variables"]["group_name"]:
            # Remove group name if using GroupKFold
            n_features -= 1
        self.n_labels = len(self.label_list)

        if self.predictor_type == "nn":
            architecture = self.predictor_config["architecture"]
        else:
            architecture = {}

        self.predictor = []
        self.calibrator = []

        for i in range(self.n_labels):
            self.predictor.append(
                Predictor(
                    self.predictor_type,
                    hyperparameters=self.predictor_config["hyperparameters"],
                    architecture=architecture,
                    n_features=n_features,
                    logger=self.logger,
                )
            )
            if self.calibrator_type:
                # Only if a calibrator type is provided
                self.calibrator.append(
                    Calibrator(
                        self.calibrator_type,
                        hyperparameters=self.calibrator_config["hyperparameters"],
                        logger=self.logger,
                    )
                )
        self.preprocessor = Preprocessor(
            self.preprocessor_config["preprocessing"], logger=self.logger
        )

    def fit_preprocessor(self, X):
        """
        Fits the preprocessor operations based on input data.

        Parameters
        ----------
        X : pd.Dataframe of shape (n_samples, n_features)
            Data to clean.

        Returns
        -------
        None
            Nothings is returned.

        """
        self.preprocessor.fit(X)

    def transform(self, X):
        """
        Transforms input data based on preprocessor fitted operations.

        Parameters
        ----------
        X : pd.Dataframe of shape (n_samples, n_features)
            Data to clean.

        Returns
        -------
        data : pd.Dataframe of shape (n_samples, n_features)
             Transformed data.

        """
        return self.preprocessor.transform(X)

    def fit_transform(self, X):
        """
        Fits the preprocessor operations and transforms the input data.

        Parameters
        ----------
        X : pd.Dataframe of shape (n_samples, n_features)
            Data to clean.

        Returns
        -------
        data : pd.Dataframe of shape (n_samples, n_features)
             Transformed data.

        """
        return self.preprocessor.fit_transform(X)

    def get_test_data(self, X):
        """
        Returns train and test data based on input data.

        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features)
            Data to split.

        Returns
        -------
        X_train : pd.DataFrame of shape (n_samples, n_features)
            Train set.
        X_test : pd.DataFrame of shape (n_samples, n_features)
            Test set.

        """
        split_vars = self.preprocessor_config["split_variables"]

        if split_vars["group_name"]:
            train_idx, test_idx = get_validation_idx(
                arange(len(X), dtype=int), X[split_vars["group_name"]]
            )
            X_test = X.iloc[test_idx]
            X_test = X_test.drop(split_vars["group_name"], axis=1)

        else:
            # No groups just get 10 percent of the data
            train_idx, test_idx = get_validation_idx(arange(len(X), dtype=int))
            X_test = X.iloc[test_idx]

        X_train = X.iloc[train_idx]

        return X_train, X_test

    def fit_model(self, X, y, model, idx=0, **kwargs):
        """
        Fits the predictor or calibrator model on the provided dataset.

        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples, self.n_labels)
            Prediction labels.
        model : {"predictor", "calibrator"}
            Model to fit.
        idx : int, default: 0
            Index of model to fit.
        **kwargs
            Extra arguments for fitting the models.

        Returns
        -------
        None
            Nothing is returned.

        Raises
        ------
        ValueError
            If model is not "predictor" or "calibrator"

        """
        match model:
            case "predictor":
                self.predictor[idx].fit(X, y, **kwargs)
            case "calibrator":
                self.calibrator[idx].fit(X, y, **kwargs)
            case _:
                raise ValueError(
                    f"Model should be predictor or calibrator, but got {model}"
                )

    def test_model(self, X, y, model, idx=0, key=None):
        """
        Tests the predictor or calibrator model on the provided dataset.

        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Prediction labels.
        model : {"predictor", "calibrator"}
            Model to test.
        idx : int, default: 0
            Index of model to test.
        key : str or None, default: None
            Key for updating the metric dictionaries.
            If None, the dictionaries are not updated.

        Returns
        -------
        None
            Nothing is returned.

        Raises
        ------
        ValueError
            If model is not "predictor" or "calibrator"

        """
        match model:
            case "predictor":
                message = "Uncalibrated metrics"
                metric_dict = test_model(
                    y,
                    self.predict(X, idx=idx, model_type="predictor"),
                    array(self.predict_proba(X, idx=idx, model_type="predictor")),
                )

                if key:
                    self.predictor_metrics.update({key: metric_dict})

            case "calibrator":
                message = "Calibrated metrics"
                metric_dict = test_model(
                    y,
                    self.predict(X, idx=idx, model_type="calibrator"),
                    array(self.predict_proba(X, idx=idx, model_type="calibrator")),
                )

                if key:
                    self.calibrator_metrics.update({key: metric_dict})

            case _:
                raise ValueError(
                    f"Model should be predictor or calibrator, but got {model}"
                )

        print_message(message, self.logger, SCRIPT_NAME)
        print_metrics(metric_dict, [self.label_list[idx]], self.logger)

    def run(self, X):
        """
        Run pipeline with input data.

        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features)
            Training data.

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

        group_name = self.preprocessor_config["split_variables"]["group_name"]
        weights = None
        X, y = extract_labels(data, self.label_list)  # Get prediction labels from data

        if group_name:
            groups = data[group_name]  # Get the groups for splitting
        else:
            groups = None

        # Create independent calibration set
        train_idx, val_idx = get_validation_idx(arange(len(y)), groups)
        X_cal = X.iloc[val_idx]
        y_cal = y[val_idx]
        X = X.iloc[train_idx]
        y = y[train_idx]

        if group_name:
            groups = groups.iloc[train_idx]
            X_cal = X_cal.drop(groups.name, axis=1)  # Remove groups in calibration

        kfold_it = test_train_it(**self.preprocessor_config["split_variables"])
        n_folds = kfold_it.get_n_splits(X, y[:, 0], groups=groups)

        for i, (train_idx, test_idx) in enumerate(
            kfold_it.split(X, y[:, 0], groups=groups)
        ):
            if group_name:
                X_fold = X.drop(groups.name, axis=1)
                fold = int(
                    groups.iloc[test_idx[0]]
                )  # Use the test year as the fold number
                fold_groups = groups.iloc[train_idx]
                fold_message = f"  Fold number {fold} ({i+1}/{n_folds})"
            else:
                X_fold = X
                fold = i
                fold_groups = None
                fold_message = f"  Fold number {fold+1}/{n_folds}"

            # Create the different data sets
            X_train = X_fold.iloc[train_idx]
            y_train = y[train_idx]
            X_test = X_fold.iloc[test_idx]
            y_test = y[test_idx]

            for j in range(self.n_labels):
                # Sample and weight data if needed
                X_train_i, y_train_i, fold_groups_i = self._sample_data(
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

                self._train_models(
                    X_train_i,
                    y_train_i,
                    X_cal,
                    y_cal[:, j],
                    j,
                    **{
                        "X_test": X_test,
                        "y_test": y_test[:, j],
                        "weights": weights,
                    },
                )

                # Test predictor and calibrator on test set
                self.test_model(X_test, y_test[:, j].squeeze(), "predictor", j, fold)
                self.test_model(X_test, y_test[:, j].squeeze(), "calibrator", j, fold)

                # Rest predictor and calibrator without printing
                self.predictor[j]._set_model(quiet=True)
                self.calibrator[j]._set_model(quiet=True)

        # Train final model on complete training set
        print_message("  Final training on all examples", self.logger, SCRIPT_NAME)
        if group_name:
            # Drop group names for final dataset
            X = X.drop(groups.name, axis=1)

        for k in range(self.n_labels):
            X_train, y_train, _ = self._sample_data(X, expand_dims(y[:, k], 1), groups)
            weights = self._weight_data(y_train)

            print_message(
                f"Current metric: {self.label_list[k]}", self.logger, SCRIPT_NAME
            )
            print_message(
                f"  Train set size: {len(X_train)} examples",
                self.logger,
                SCRIPT_NAME,
            )
            self._train_models(
                X_train, y_train, X_cal, y_cal[:, k], k, **{"weigths": weights}
            )

    def _train_models(self, X_train, y_train, X_cal, y_cal, idx=0, **kwargs):
        """
        Trains the predictor and calibrator models.

        Parameters
        ----------
        X_train : pd.DataFrame of shape (n_samples, n_features)
            Train data for the predictor.
        y_train : np.array of shape (n_samples,)
            Train labels for the predictor.
        X_cal : pd.DataFrame of shape (n_samples, n_features)
            Calibration data for the calibrator.
        y_cal : np.array of shape (n_samples,)
            Calibration labels for the calibrator.
        idx : int, default: 0
            Index of model to train.
        **kwargs
            Extra arguments for fitting the predictor.

        Returns
        -------
        None
            Nothing is returned.

        """
        # Fit predictor on train set
        self.fit_model(X_train, y_train, "predictor", idx, **kwargs)

        # Fit calibrator on validation set
        self.fit_model(
            self._get_calibrator_data(X_cal, idx),
            y_cal,
            "calibrator",
            idx,
        )

    def predict_proba(self, X, idx=-1, model_type="calibrator"):
        """
        Predicts probabilities from predictor or calibrator based on input data.

        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features)
            Data to make predictions on.
        idx : int, default: -1
            Index of model to use.
        model_type : {"predictor", "calibrator"}
            Model to use.

        Returns
        -------
        probabilities : np.array of shape (n_samples, 2)
            Predicted probabilities.

        Raises
        ------
        ValueError
            If model is not "predictor" or "calibrator"

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

        if idx == -1:
            probabilities = []
            for i in range(self.n_labels):
                # Loop over all labels to get probabilities for each model
                probabilities.append(pred_fn(X, i, "predict_proba"))
            return probabilities
        else:
            return pred_fn(X, idx, "predict_proba")

    def predict(self, X, idx=-1, model_type="calibrator"):
        """
        Predicts labels from predictor or calibrator based on input data.

        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features)
            Data to make predictions on.
        idx : int, default: 0
            Index of model to use.
        model_type : {"predictor", "calibrator"}
            Model to use.

        Returns
        -------
        labels : array-like of shape (n_samples,)
            Predicted labels.

        Raises
        ------
        ValueError
            If model_type is not "predictor" or "calibrator"

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

        if idx == -1:
            labels = []
            for i in range(self.n_labels):
                # Loop over all labels to get labels for each model
                labels.append(pred_fn(X, i, "predict"))
            return labels
        else:
            return pred_fn(X, idx, "predict")

    def _predictor_pred_wrapper(self, X, idx, prediction_type):
        """
        Wrapper function to create predictions with the predictor.

        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features)
            Data to make predictions on.
        idx : int
            Index of model to use.
        prediction_type : {"predict", "predict_proba"}
            Prediction function to use.

        Returns
        -------
        estimates : np.array(n_samples,) or np.array(n_samples, 2)
            Labels or probabilities estimated from model based on prediction_type

        Raises
        ------
        ValueError
            If prediction_type is not "predict" or "predict_proba"


        """
        match prediction_type:
            case "predict":
                return self.predictor[idx].predict(X)
            case "predict_proba":
                return self.predictor[idx].predict_proba(X)
            case _:
                raise ValueError(
                    "Prediction type should be predict or predict_proba, "
                    f"but got {prediction_type}"
                )

    def _calibrator_pred_wrapper(self, X, idx, prediction_type):
        """
        Wrapper function to create predictions with the calibrator.

        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features)
            Data to make predictions on.
        idx : int
            Index of model to use.
        prediction_type : {"predict", "predict_proba"}
            Prediction function to use.

        Returns
        -------
        estimates : np.array(n_samples,) or np.array(n_samples, 2)
            Labels or probabilities estimated from model based on prediction_type

        Raises
        ------
        ValueError
            If prediction_type is not "predict" or "predict_proba"

        """
        match prediction_type:
            case "predict":
                return self.calibrator[idx].predict(self._get_calibrator_data(X, idx))
            case "predict_proba":
                return self.calibrator[idx].predict_proba(
                    self._get_calibrator_data(X, idx)
                )
            case _:
                raise ValueError(
                    "Prediction type should be predict or predict_proba, "
                    f"but got {prediction_type}"
                )

    def _sample_data(self, X, y, groups):
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

        Returns
        -------
        X_sampled : pd.DataFrame of shape (n_sampled_samples, n_features)
            Sampled data.
        y_sampled : np.array of shape (n_sampled_samples,)
            Sampled labels.
        groups_sampled : pd.Series of shape (n_sampled_samples,) or None
            Groups of the examples, None if not specified.

        """
        sampler_fn = self.predictor_config["sampler"]["sampler_fn"]

        if sampler_fn:
            return data_sampler(X, y, groups=groups, **self.predictor_config["sampler"])

        return X, y, groups

    def _weight_data(self, y):
        """
        Gets the weights for the data based on configuration.

        Parameters
        ----------
        y : np.array of shape (n_samples, self.n_labels)
            Labels needed for creation of weights.

        Returns
        -------
        weights : np.array of shape (n_samples,), or None
            Sample or class weights based on labels.
            Sample weights are of shape (n_samples,).
            Class weights are of shape (1).
            None if no weighting function is provided.

        """
        weighting_fn = self.predictor_config["weighting"]["weighting_fn"]

        if weighting_fn:
            return getattr(weight, weighting_fn)(y)

        if self.predictor_type == "nn":
            return array([1])

        return ones(y.shape[0])

    def _get_calibrator_data(self, X, idx):
        """
        Get the calibrator data based on the calibrator type.

        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features)
            Data to transform for calibrator.
        idx : int, default: 0
            Index of model to train.

        Returns
        -------
        calibrator_data : np.array of shape (n_samples,) or (n_samples, 2)
            Data for calibrator model based on its type.

        """
        calibrator_data = get_positive_proba(
            self.predict_proba(X, idx=idx, model_type="predictor")
        )

        if self.calibrator_type == "isotonic" and calibrator_data.shape[1] == 2:
            # Only provide positive probabilities
            calibrator_data = calibrator_data[:, 1]

        return calibrator_data
