"""
Pipeline class.

This class creates a Pipeline to prepare data, fit a predictor and
a calibrator.

"""

SCRIPT_NAME = "pipeline/Pipeline"


class Pipeline:
    """
    Class that creates a Pipeline.

    Attributes
    ----------
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
    __init__(preprocessor_config, predictor_config, calibrator_config, logger)
    fit(X, y, **kwargs)
        Fits the predictor and calibrator.
    predict_proba(X)
        Predicts probabilities from predictor based on input data.
    predict_calibrated_proba(X)
        Predicts probabilities from calibrator based on input data.
    """

    def __init__(
        self, preprocessor_config, predictor_config, calibrator_config, logger=None
    ):
        """
        Initialise a Pipeline class instance.

        Parameters
        ----------
        preprocessor_config : dict[]
            Configuration parameters for the preprocessor object.
        predictor_config : dict[]
            Configuration parameters for the predictor object.
        calibrator_config : dict[]
            Configuration parameters for the calibrator object.
        logger : logging.Logger or None, default: None
            Logger object to log prints. If None print to terminal.

        Returns
        -------
        None
            Nothing is returned.

        """
        self.logger = logger

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
