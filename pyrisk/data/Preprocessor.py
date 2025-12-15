"""
Preprocessor class.

This class creates a Preprocessor to prepare data.

"""

from pyrisk.utils.logger import print_message

from .preprocessing import convert_object_to_categorical, preprocess_data

SCRIPT_NAME = "data/Preprocessor"


class Preprocessor:
    """
    Class that creates a Preprocessor.

    Attributes
    ----------
    preprocess : bool
        Flag to preprocess data or not.
    transform_seq : dict[str, dict[str,list[str]]]
        Transformation sequence for the data.
    logger : logging.Logger or None, default: None
        Logger object to log prints. If None print to terminal.

    Methods
    -------
    __init__(preprocessor_config, logger)
        Init method.
    _clean_data(X)
        Cleans data in preparation for transformation.
    transform(X)
        Transforms input data based on transformation sequence.
    """

    def __init__(self, preprocessor_config, logger=None):
        """
        Initialise a Preprocessor class instance.

        Parameters
        ----------
        preprocessor_config : dict[str, dict[str, list[str]]]
            Configuration parameters for the preprocessor object.
        logger : logging.Logger or None, default: None
            Logger object to log prints. If None print to terminal.

        Returns
        -------
        None
            Nothing is returned.

        """
        self.preprocess = preprocessor_config.pop("preprocess")
        self.transform_seq = preprocessor_config
        self.logger = logger

    def _clean_data(self, X):
        """
        Cleans data before transformation.

        Removes rows with Nan values and converts objects to
        categoricals.

        Parameters
        ----------
        X : pd.Dataframe of shape (n_samples, n_features)
            Data to clean.

        Returns
        -------
        data : pd.Dataframe of shape (n_samples, n_features)
             Cleaned data.

        """
        # Convert objects to categorical (not saved so needs to be here)
        data = convert_object_to_categorical(X)

        # Remove NaN values
        nb_nan_rows = data.isna().any(axis=1).sum()
        data = data.dropna()

        print_message(
            f"Dropped {nb_nan_rows} rows with NaN values", self.logger, SCRIPT_NAME
        )

        return data

    def transform(self, X):
        """
        Transforms input data based on transformation sequence.

        Parameters
        ----------
        X : pd.Dataframe of shape (n_samples, n_features)
            Data to clean.

        Returns
        -------
        data : pd.Dataframe of shape (n_samples, n_features)
             Cleaned data.

        """
        data = self._clean_data(X)  # Clean data before transformation

        if self.preprocess:
            # If the preprocess flag is true
            print_message("Preprocessing data", self.logger, SCRIPT_NAME)
            for key in self.transform_seq.keys():
                data = preprocess_data(
                    data, self.transform_seq[key]["feature_list"], key
                )

        return data
