"""
Pipeline class.

This class creates a Pipeline to prepare data, fit a predictor and
a calibrator.

"""

from pyrisk.data.Preprocessor import Preprocessor
from pyrisk.models.Calibrator import Calibrator
from pyrisk.models.Predictor import Predictor
from pyrisk.utils.config import get_configuration, split_version_number

SCRIPT_NAME = "pipeline/Pipeline"


class Pipeline:
    """
    Class that creates a Pipeline.

    Attributes
    ----------
    version : str
        Version number.
    preprocessor : Preprocessor
        Data preprocessor object.
    predictor : Predictor
        Prediction model object.
    calibrator : Calibrator
        Calibration model object.
    logger : logging.Logger or None, default: None
        Logger object to log prints. If None print to terminal.

    Methods
    -------
    __init__(pipeline_config={}, logger=None)
        Init method.
    fit(X, y, **kwargs)
        Fits the predictor and calibrator.
    predict_proba(X)
        Predicts probabilities from predictor based on input data.
    predict_calibrated_proba(X)
        Predicts probabilities from calibrator based on input data.
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
        self.calibrator_config = self.predictor_config["calibrator"]["hyperparameters"]

        # Define variables needed to initialise other objects
        label_list = self.predictor_config["labels"]["label_list"]
        n_features = len(self.preprocessor_config["features"]["feature_list"]) - len(
            label_list
        )
        if self.preprocessor_config["split_variables"]["group_name"]:
            # Remove group name if using GroupKFold
            n_features -= 1
        n_classes = len(label_list)

        self.predictor = Predictor(
            self.predictor_type,
            hyperparameters=self.predictor_config["hyperparameters"],
            n_features=n_features,
            n_classes=n_classes,
            logger=self.logger,
        )
        if self.calibrator_type:
            # Only if a calibrator type is provided
            self.calibrator = Calibrator(
                self.calibrator_type,
                n_classes=n_classes,
                logger=self.logger,
                **self.calibrator_config,
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

    def fit(self, X, y):
        """
        Train the model on the provided dataset.

        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples, n_classes)
            Prediction labels.

        Returns
        -------
        None
            Nothing is returned.

        """

    def predict_proba(self, X):
        """
        Predicts probabilities from predictor based on input data.

        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        probabilities : np.array (n_classes,) of arrays (n_samples, 2)
            Predicted probabilities.

        """
        probabilities = []

        if outputs.shape[1] == 1:
            return probabilities[0]
        else:
            return probabilities
