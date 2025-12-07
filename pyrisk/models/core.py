"""
Models functions module.

This module provides functions to create, train, and test models.

Functions:
- create_model: Creates a new model.
- train_model: Trains a given model on some train data.
- test_model: Tests a model on some test data.
- save_model: Pickles a model.
- load_model: Loads a pickled model.
"""

import pickle
from copy import deepcopy

import pandas as pd
import sklearn as skl
from sklearn.multioutput import MultiOutputClassifier
from torch.accelerator import current_accelerator, is_available

import pyrisk.data.weighting as weight
from pyrisk.data.preprocessing import extract_labels
from pyrisk.data.sampler import data_sampler
from pyrisk.metrics.core import (
    compute_pred_metrics,
    compute_score_metrics,
    print_metrics,
)
from pyrisk.models.AIRiskNN import AIRiskNN
from pyrisk.utils.exceptions import array_check, array_dim_check, file_checks
from pyrisk.utils.logger import print_message

SCRIPT_NAME = "models/core"


def create_model(
    model_type: str, n_features: int = -1, logger=None, n_classes=1, **config_params
):
    """
    Creates a AI model.

    Parameters
    ----------
    model_type : {"hgb", "svm", "nn"}
        Type of model to create.
            hgb: histogram gradient boosting.
            svm: support vector machine.
            nn: AIRiskNN neural network.
            tabp: TabP foundational model.
    n_features : int, default: -1
        Number of features in the data, only needed for NN models.
    logger : logging.Logger, default: None
        Logger object to log prints. If None print to terminal.
    n_classes : int, default: 1
        Number of classes. Used to call MultiOutputClassifier.
    **config_params
        Configuration parameters for the model.

    Returns
    -------
    model : HistGradBoostingClassifier, SVC or AIRiskNN.
        Created model.

    Raises
    ------
    TypeError
        If model_type is not a str.
        If an unexpected keyword argument is present.
    ValueError
        If model_type is not "hgb", "svm", "nn", or "tabp".

    """
    if type(model_type) is not str:
        raise TypeError(f"{model_type} shoud be a string")

    match model_type:
        case "hgb":
            print_message(
                "Creating a Histogram Gradient Boosting model", logger, SCRIPT_NAME
            )
            model = skl.ensemble.HistGradientBoostingClassifier(**config_params)

            if n_classes > 1:
                model = MultiOutputClassifier(model)

        case "svm":
            print_message(
                "Creating a Support Vector Machine model", logger, SCRIPT_NAME
            )
            model = skl.svm.SVC(**config_params)

            if n_classes > 1:
                model = MultiOutputClassifier(model)

        case "nn":
            print_message("Creating a Neural Network model", logger, SCRIPT_NAME)

            if n_features == -1:
                raise ValueError("For nn models, please specify feature number")

            device = current_accelerator().type if is_available() else "cpu"
            print_message(f"Using {device} device", logger, SCRIPT_NAME)
            model = AIRiskNN(n_features, logger, **config_params).to(device)

        case _:
            raise ValueError(f"{model_type} invalid model type. See function docstring")

    return model


def train_model(
    model,
    data,
    kfold_it,
    label_list,
    group_name="",
    logger=None,
    **model_config,
):
    """
    Trains an AI model.

    The model with the best precision is selected.

    Parameters
    ----------
    model : HistGradBoostingClassifier, SVC or AIRiskNN.
        Model to train.
    data : pd.DataFrame
        Data to train on.
    kfold_it : StratifiedKFold or GroupKFold
        KFold iterator to create train and test sets.
    label_list : list[str]
        Labels to predict.
    group_name : str, default: ""
        Group name used to extract the group data for splitting.
    logger : logging.Logger, default: None
        Logger object to log prints. If None print to terminal.
    weighting_fn : str, default: ""
        Name of the weighting function to use for the samples.
    **model_config
        Additional argument dictionary for fitting.

    Returns
    -------
    model_metrics : dict[int, dict[str, float or tuple(array-like)]
        Model metrics for each fold.
        The test metrics used are:
         - accuracy
         - f1
         - precision
         - recall
         - roc (Receiver Operator Characteristic)
         - auroc (Area Under Receiver Operator Characteristic)
         - prc (Precision-Recall Curve)
         - ap (Average Precision)

    Raises
    ------
    TypeError
        If data is not a pd.DataFrame.

    """
    if type(data) is not type(pd.DataFrame()):
        raise TypeError(f"data should be a pd.DataFrame, but got {type(data)}")

    # Initialise variables
    group_flag = False
    precision = 0.0  # Precision used to select the best model
    best_fold = 0  # Fold with best precision
    untrained_model = deepcopy(model)  # Untouched model to reset at each fold
    model_tmp = model  # Temporary model variable to keep best model
    model_metrics = {}  # Dict to store model metrics for each fold
    weights = []  # Weights for class imbalance

    X, y = extract_labels(data, label_list)  # Get prediction labels from data

    if group_name != "":
        group_flag = True
        groups = data[group_name]  # Get the groups for splitting
    else:
        groups = None

    sampler_fn = model_config["sampler"]["sampler_fn"]
    weighting_fn = model_config["weighting"]["weighting_fn"]

    if sampler_fn:
        # Sample data first
        X, y, groups = data_sampler(X, y, groups=groups, **model_config["sampler"])

    if weighting_fn:
        # Get sample weights if needed
        weights = getattr(weight, weighting_fn)(y)

    n_folds = kfold_it.get_n_splits(X, y[:, 0], groups=groups)

    for i, (train_idx, test_idx) in enumerate(
        kfold_it.split(X, y[:, 0], groups=groups)
    ):
        X_train = X.iloc[train_idx]
        y_train = y[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y[test_idx]

        if group_flag:
            fold = int(groups.iloc[test_idx[0]])  # Use the test year as the fold number
            # Drop the group (OP_YEAR) from data
            X_train = X_train.drop(groups.name, axis=1)
            X_test = X_test.drop(groups.name, axis=1)
            print_message(
                f"  Fold number {fold} ({i+1}/{n_folds})", logger, SCRIPT_NAME
            )
        else:
            fold = i
            print_message(f"  Fold number {fold+1}/{n_folds}", logger, SCRIPT_NAME)

        print_message(f"  Train set size: {len(X_train)} examples", logger, SCRIPT_NAME)

        if type(model) is AIRiskNN:
            # Pass test data for epoch print
            device = current_accelerator().type if is_available() else "cpu"
            model.to(device)  # Load untrained model to device
            model.fit(
                X_train,
                y_train,
                X_test,
                y_test,
                weights,
                **model_config["hyperparameters"],
            )  # Train model
        else:
            if len(weights) == 0:
                sample_weights = None
            else:
                sample_weights = weights[train_idx]
            model.fit(
                X_train, y_train.squeeze(), sample_weight=sample_weights
            )  # Train model
        metric_dict = test_model(model, X_test, y_test.squeeze())
        model_metrics.update({fold: metric_dict})

        # Print model metrics
        print_metrics(metric_dict, label_list, logger)

        if metric_dict["precision"][-1] > precision:
            # Temporarily save model with best precision
            print_message("  New best model", logger, SCRIPT_NAME)
            precision = metric_dict["precision"][-1]
            best_fold = fold
            model_tmp = deepcopy(model)

        model = deepcopy(untrained_model)  # Reset the model to the original

    model = model_tmp  # Set model with best precision
    print_message(f"  Best fold number {best_fold}", logger, SCRIPT_NAME)
    return model_metrics


def test_model(model, X_test, y_test):
    """
    Computes different metrics to test the model.

    Parameters
    ----------
    model : HistGradBoostingClassier, SVC, or AIRiskNN
        Model to test.
    X_test : array-like of shape (n_samples, n_features)
        Test data to make predictions on.
    y_test : array-like of shape (n_samples, n_classes)
        Ground truth test labels.

    Returns
    -------
    metric_dict : dict[str, dict[str, list[float or tuple(array-like)]]
        Dictionary of the model performance for one fold.
        Keys are the metric name and values are the metric value.
        The test metrics used are:
         - accuracy
         - f1
         - precision
         - recall
         - roc (Receiver Operator Characteristic)
         - auroc (Area Under Receiver Operator Characteristic)
         - prc (Precision-Recall Curve)
         - ap (Average Precision)

    Raises
    ------
    TypeError
        If X_test or y_test are not an array-like.
    ValueError
        If X_test and y_test do not have the same dimensions.

    """
    # Check that inputs are correct
    array_check(X_test)
    array_check(y_test)
    array_dim_check(X_test, y_test, 0)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    metric_dict = compute_pred_metrics(
        ["accuracy", "f1", "recall", "precision"], y_test, y_pred
    )
    metric_dict.update(
        compute_score_metrics(["roc", "auroc", "prc", "ap"], y_test, y_pred_proba)
    )
    return metric_dict


def save_model(model, save_file, model_metrics=None, extension=".pkl") -> None:
    """
    Saves an AI model to file.

    Parameters
    ----------
    model : HistGradBoostingClassifier, SVC, or AIRiskNN
        Model to save.
    save_file : str
        Path to the file to save the model.
    extension : str, default: ".pkl"
        Extension of the save file.
    model_metrics : dict[int, dict[str, float or tuple(array-like)]], default None
        Model metrics for different folds.

    Returns
    -------
    None
        Nothing is returned.

    Raises
    ------
    TypeError
        If save_file is not a str.
    FileNotFoundError
        If save_file does not exist.
    IsADirectoryError
        If save_file is a directory.
    ValueError
        If save_file extension is not extension.

    """
    file_checks(save_file, extension, exists=False)

    if extension != ".pkl":
        # Saving an AIRiskNN model
        model.save_model(save_file)

    else:
        with open(save_file, "wb") as f:
            pickle.dump([model, model_metrics], f)


def load_model(load_file: str):
    """
    Loads an AI model and its metric data from a .pkl file.

    Parameters
    ----------
    load_file : str
        Path to the file to load the model from.

    Returns
    -------
    model : HistGradBoostingClassifier, SVC, or AIRiskNN
        Loaded model or state_dict in the case of an AIRiskNN.
    model_metrics : dict[int, dict[str, float or tuple(array-like)]]
        Model metrics for different folds.

    Raises
    ------
    TypeError
        If load_file is not a str.
    FileNotFoundError
        If load_file does not exist.
    IsADirectoryError
        If load_file is a directory.
    ValueError
        If load_file extension is not .pkl file.

    """
    file_checks(load_file, ".pkl")

    with open(load_file, "rb") as f:
        loaded_data = pickle.load(f)

    return loaded_data[0], loaded_data[1]
