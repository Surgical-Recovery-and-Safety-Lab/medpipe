"""
Weighting functions module.

This module provides functions to create sample weigths to address
class imbalance.

Functions:
- inverse_frequency_sum_sample_weights: Create sample weights using inverse number of
    classes summed over the labels.
- inverse_frequency_class_weights: Create class weights using inverse frequency
    of classes.
- negative_positive_ratio_sample_weights: Create sample weights using the ratio betwee
    negative and positive classes.
- negative_positive_ratio_class_weights: Create class weights using the ratio between
    negative and positive classes.
"""

import numpy as np

from pyrisk.utils.exceptions import array_check


def inverse_frequency_sum_sample_weights(labels):
    """
    Create sample weights using the total number of samples over the number of
    positive and negative samples.

    Parameters
    ----------
    labels : array-like
        Binary prediction labels of shape (n_samples, n_classes)

    Returns
    -------
    sample_weights : np.array(n_samples, n_classes)
        Weight for each sample.

    Raises
    ------
    TypeError
        If labels is not array-like.
    ValueError
        If labels is empty.
    ZeroDivisionError
        If there are no positive labels.

    Notes
    -----
    The number of class instances is counted and the weight is calculated as:
        len(labels) / (pos_weight + neg_weight)
    where pos_weight is an array with 1 for the positive labels for the classes and
    neg_weight is an array with 1 for negative labels for the classes.

    """
    array_check(labels)  # Check that labels is array-like

    if len(labels) == 0:
        raise ValueError("The input labels are empty")

    pos_counts = np.sum(labels, axis=0)
    neg_counts = len(labels) - pos_counts

    if pos_counts.any() == 0:
        raise ZeroDivisionError("No positive labels found")

    pos_weight = pos_counts * labels
    neg_weight = neg_counts * ~np.array(labels, dtype=bool)  # Invert for negatives

    return len(labels) / (pos_weight + neg_weight)


def inverse_frequency_class_weights(labels):
    """
    Create class weights of the positive class using inverse frequency of
    the positive class.

    Parameters
    ----------
    labels : array-like
        Binary prediction labels of shape (n_samples, n_classes)

    Returns
    -------
    class_weights : np.array(n_classes,)
        Weight for each class.

    Raises
    ------
    TypeError
        If labels is not array-like.
    ValueError
        If labels is empty.
    ZeroDivisionError
        If there are no positive labels.

    """
    array_check(labels)

    if len(labels) == 0:
        raise ValueError("The input labels are empty")

    pos_counts = np.sum(labels, axis=0)

    if pos_counts.any() == 0:
        raise ZeroDivisionError("No positive labels found")

    return len(labels) / pos_counts


def negative_positive_ratio_sample_weights(labels):
    """
    Create sample weights using the ratio between negative and
    positive samples.

    Parameters
    ----------
    labels : array-like
        Binary prediction labels of shape (n_samples, n_classes)

    Returns
    -------
    sample_weights : np.array(n_samples, n_classes)
        Weight for each sample.

    Raises
    ------
    TypeError
        If labels is not array-like.
    ValueError
        If labels is empty.
    ZeroDivisionError
        If there are no positive labels.

    """
    array_check(labels)  # Check that labels is array-like

    if len(labels) == 0:
        raise ValueError("The input labels are empty")

    pos_counts = np.sum(labels, axis=0)
    neg_counts = len(labels) - pos_counts

    if pos_counts.any() == 0:
        raise ZeroDivisionError("No positive labels found")

    pos_weight = (neg_counts / pos_counts) * labels
    neg_weight = 1 * ~np.array(labels, dtype=bool)  # Invert for negatives

    return pos_weight + neg_weight


def negative_positive_ratio_class_weights(labels):
    """
    Create class weights of the positive class using the ratio
    between the number of samples in the negative and positive classes.

    Parameters
    ----------
    labels : array-like
        Binary prediction labels of shape (n_samples, n_classes)

    Returns
    -------
    class_weights : np.array(n_classes,)
        Weight for each class.

    Raises
    ------
    TypeError
        If labels is not array-like.
    ValueError
        If labels is empty.
    ZeroDivisionError
        If there are no positive labels.

    """
    array_check(labels)

    if len(labels) == 0:
        raise ValueError("The input labels are empty")

    pos_counts = np.sum(labels, axis=0)
    neg_counts = len(labels) - pos_counts

    if pos_counts.any() == 0:
        raise ZeroDivisionError("No positive labels found")

    return neg_counts / pos_counts
