"""
Weighting functions module.

This module provides functions to create sample weigths to address
class imbalance.

Functions:
- inverse_class_sum_weighting: Create sample weights using inverse number of
    classes summed over the labels.
"""

import numpy as np

from pyrisk.utils.exceptions import array_check


def inverse_class_sum_weighting(labels):
    """
    Create sample weights using inverse number of classes summed over the labels.

    Parameters
    ----------
    labels : array-like
        Binary prediction labels of shape (n_samples, n_classes)

    Returns
    -------
    sample_weights : np.array(n_samples,)
        Weight for each sample.

    Raises
    ------
    TypeError
        If labels is not array-like.

    Notes
    -----
    The number of class instances is counted and the weight is calculated as:
        np.sum(len(labels) / (pos_weight + neg_weight), axis=1)
    where pos_weight is the number of positive labels for the classes and
    neg_weight is the number of negative labels for the classes.

    """
    array_check(labels)  # Check that labels is array-like

    pos_counts = np.sum(labels, axis=0)
    neg_counts = len(labels) - pos_counts

    pos_weight = pos_counts * labels
    neg_weight = neg_counts * ~np.array(labels, dtype=bool)  # Invert for negatives

    return np.sum(len(labels) / (pos_weight + neg_weight), axis=1)
