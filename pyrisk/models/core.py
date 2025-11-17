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

import pandas as pd
import sklearn as skl
from tabpfn import TabPFNClassifier
from torch.accelerator import current_accelerator, is_available

from pyrisk.data.preprocessing import extract_labels
from pyrisk.models.AIRiskNN import AIRiskNN
from pyrisk.utils.exceptions import array_check, array_dim_check, file_checks


def create_model(model_type: str, n_features: int = -1, **config_params):
    """
    Creates a AI model.

    Parameters
    ----------
    model_type : {"hgb", "svm", "nn", "tabp"}
        Type of model to create.
            hgb: histogram gradient boosting.
            svm: support vector machine.
            nn: AIRiskNN neural network.
            tabp: TabP foundational model.
    n_features : int, default: -1
        Number of features in the data, only needed for NN models.
    **config_params
        Configuration parameters for the model.

    Returns
    -------
    model : HistGradBoostingClassifier, SVC, AIRiskNN or TabPFNClassifier.
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
            print("[INFO] Creating a Histogram Gradient Boosting model")
            model = skl.ensemble.HistGradientBoostingClassifier(**config_params)

        case "svm":
            print("[INFO] Creating a Support Vector Machine model")
            model = skl.svm.LinearSVC(**config_params)

        case "nn":
            print("[INFO] Creating a Neural Network model")

            if n_features == -1:
                raise ValueError("For nn models, please specify feature number")

            device = current_accelerator().type if is_available() else "cpu"
            print(f"[INFO] Using {device} device")
            model = AIRiskNN(n_features, **config_params).to(device)

        case "tabp":
            print("[INFO] Creating a TabP Foundational Model")
            model = TabPFNClassifier()

        case _:
            raise ValueError(f"{model_type} invalid model type. See function docstring")

    return model


def train_model(model, data, kfold_it, labels, group_name="", **kwargs):
    """
    Trains an AI model.

    Parameters
    ----------
    model : HistGradBoostingClassifier or SVC.
        Model to train.
    data : pd.DataFrame
        Data to train on.
    kfold_it : StratifiedKFold or GroupKFold
        KFold iterator to create train and test sets.
    labels : str or list[str]
        Labels to predict.
    group_name : str, default: ""
        Group name used to extract the group data for splitting.
    **kwargs
        Additional argument dictionary for fitting.

    Returns
    -------
    model_metrics : dict[int, dict[str, float or tuple(array-like)]
        Model metrics for each fold.
        The test metrics used are:
         - accuracy
         - f1
         - precision
         - roc (Receiver Operator Characteristic)
         - auroc (Area Under Receiver Operator Characteristic)
         - prc (Precision-Recall Curve)
         - ap (Average Precision)

    Raises
    ------
    TypeError
        If sample_weight are not an array-like.
        If data is not a pd.DataFrame.
    ValueError
        If the train_labels and sample_weight do not have the same dimensions.

    """
    if type(data) is not type(pd.DataFrame()):
        raise TypeError(f"data should be a pd.DataFrame, but got {type(data)}")

    group_flag = False
    model_metrics = {}  # Dict to store model metrics for each fold
    X, y = extract_labels(data, labels)  # Get prediction labels from data

    if group_name != "":
        group_flag = True
        groups = data[group_name]  # Get the groups for splitting
    else:
        groups = None

    for i, (train_idx, test_idx) in enumerate(kfold_it.split(X, y, groups=groups)):
        if group_flag:
            fold = int(groups[test_idx].iloc[0])  # Use the test year as the fold number
            print(f"  Fold number {fold}")
        else:
            fold = i
            print(f" Fold number {fold}")

        X_train, y_train = extract_labels(data.iloc[train_idx], labels)
        X_test, y_test = extract_labels(data.iloc[test_idx], labels)

        if "sample_weight" in kwargs.keys():
            array_check(kwargs["sample_weight"])
            array_dim_check(y_train, kwargs["sample_weight"])

        model.fit(X_train, y_train.ravel(), **kwargs)  # Train model

        model_metrics.update({fold: test_model(model, X_test, y_test.ravel())})

def test_model(model, X, y, sample_weight=None) -> float:
    """
    Trains an AI model.

    Parameters
    ----------
    model : skl.ensemble.HistGradBoostingClassifier or skl.svm.SVC
        Model to test.
    X : array-like of shape (n_samples, n_features)
        Testing data.
    y : array-like of shape (n_samples, n_classes)
        Prediction labels.
    sample_weight : array-like of shape (n_samples, n_classes), default: None
        Weight of each sample to help address class imbalance.

    Returns
    -------
    score : float
        Accuracy of the model.

    Raises
    ------
    TypeError
        If X, y, or sample_weight are not an array-like.
    ValueError
        If the X and y do not have the same dimensions.
        If the y and sample_weight do not have the same dimensions.

    """
    # Check that inputs are correct
    array_check(X)
    array_check(y)
    array_dim_check(X, y, 0)

    if sample_weight:
        array_check(sample_weight)
        return model.score(X, y, sample_weight=sample_weight)

    else:
        return model.score(X, y)


def save_model(model, save_file, extension=".pkl") -> None:
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
            pickle.dump(model, f)


def load_model(load_file: str):
    """
    Loads an AI model from a .pkl file.

    Parameters
    ----------
    load_file : str
        Path to the file to load the model from.

    Returns
    -------
    model : skl.ensemble.HistGradBoostingClassifier or skl.svm.SVC
        Loaded model.

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
        model = pickle.load(f)

    return model
