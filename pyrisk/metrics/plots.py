"""
Plot functions module.

This module provides functions to plot results.

Functions:
- plot_from_display: Plot results from a Display class.
"""

import matplotlib.pyplot as plt
import sklearn as skl

from pyrisk.utils.exceptions import array_check, array_dim_check


def plot_from_display(y_true, y_pred, display, **kwargs) -> None:
    """
    Plot results from a Display class.

    See sklearn.metrics for the Display classes and the from_predictions method.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth labels.
    y_pred : array-like of shape (n_samples,)
        Predicted labels.
    display : str {"roc", "confusion", "precision-recall"}
        Type of display class to use.
    **kwargs :
        Extra arguments for the display classes.

    Returns
    -------
    ValueError
        If display is not a valid option.

    """
    # Check that inputs are correct
    array_check(y_pred)
    array_check(y_true)

    if display not in ["roc", "confusion", "precision-recall"]:
        raise ValueError(f"Invalid display, got {display}")

    match display:
        case "roc":
            print("[INFO] Plotting ROC curve")
            display_fct = skl.metrics.RocCurveDisplay.from_predictions
        case "confusion":
            print("[INFO] Plotting confusion matrix")
            display_fct = skl.metrics.ConfusionMatrixDisplay.from_predictions
        case "precision-recall":
            print("[INFO] Plotting precision-recall curve")
            display_fct = skl.metrics.PrecisionRecallDisplay.from_predictions
        case _:
            # Default to ROC curve just in case
            display_fct = skl.metrics.RocCurveDisplay.from_predictions

    display_fct(y_true, y_pred, **kwargs)
    plt.show()
