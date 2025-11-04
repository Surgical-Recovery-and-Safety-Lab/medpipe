"""
Models functions module.

This module provides functions to create, train, and test models.

Functions:
- create_model: Creates a new model.
"""

import sklearn as skl


def create_model(model_type: str, **config_params: dict):
    """
    Creates a AI model.

    Parameters
    ----------
    model_type : {"hgb", "svm"}
        Type of model to create.
            hgb: histogram gradient boosting.
            svm: support vector machine.
    **config_params
        Configuration parameters for the model.

    Returns
    -------
    model : skl.ensemble.HistGradBoostingClassifier or skl.svm.SVC
        Created model.

    Raises
    ------
    TypeError
        If model_type is not a str.
        If an unexpected keyword argument is present.
    ValueError
        If model_type is not "hgb" or "svm".

    """
    if type(model_type) is not str:
        raise TypeError(f"{model_type} shoud be a string")

    match model_type:
        case "hgb":
            print("Creating a Histogram Gradient Boosting model")
            model = skl.ensemble.HistGradientBoostingClassifier(**config_params)

        case "svm":
            print("Creating a Support Vector Machine model")
            model = skl.svm.SVC(**config_params)

        case _:
            raise ValueError(f"{model_type} invalid model type. See function docstring")

    return model
