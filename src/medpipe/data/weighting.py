"""
Weighting functions module.

This module provides functions to create sample weigths to address
class imbalance.

Functions:
- inverse_frequency_multiclass_sample_weights: Create sample weights using the total
    number of samples over the number of positive and negative samples.
- inverse_frequency_single_sample_weights: Create sample weights using the inverse
    frequency of positive and negative samples.
- inverse_frequency_class_weights: Create class weights using inverse frequency
    of classes.
- negative_positive_ratio_sample_weights: Create sample weights using the ratio betwee
    negative and positive classes.
- negative_positive_ratio_class_weights: Create class weights using the ratio between
    negative and positive classes.
"""

import numpy as np

from medpipe.utils.exceptions import array_check


def inverse_frequency_multiclass_sample_weights(labels):
    """
    Create sample weights using the total number of samples over the number of
    positive and negative samples.

    Each class has its own set of weights for positive and negative examples
    based on the number of positive and negative examples in that class.

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
    For each class, the weights are calculated as:
        len(labels) / (pos_weight + neg_weight),
    where pos_weight is an array of shape (n_samples, n_classes) for the positive examples
    with the total number of positive samples in each class, and neg_weight is similar
    but for the negative examples.

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


def inverse_frequency_single_sample_weights(labels):
    """
    Create sample weights using the inverse frequency of positive
    and negative samples.

    One set of weights is created and used for each class based on the
    total number of positive and negative examples. Weights are normalised
    so that negative weights are 1.

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
    ValueError
        If labels is empty.
    ZeroDivisionError
        If there are no positive labels.

    """
    array_check(labels)  # Check that labels is array-like

    if len(labels) == 0:
        raise ValueError("The input labels are empty")

    pos_examples = np.sum(labels, axis=1) > 0
    pos_count = np.sum(pos_examples)

    if pos_count == 0:
        raise ZeroDivisionError("No positive labels found")

    # Create weights
    weights = np.ones(len(labels))  # Negative weight is 1
    pos_weight = (len(labels) - pos_count) / pos_count
    weights[pos_examples] *= pos_weight

    return weights


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
