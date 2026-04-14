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

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from medpipe._types import Labels
from medpipe.utils.exceptions import array_check

if TYPE_CHECKING:
    import numpy.typing as npt


def inverse_frequency_multiclass_sample_weights(labels: Labels) -> npt.NDArray:
    """
    Create sample weights using the total number of samples over the number of
    positive and negative samples.

    Each class has its own set of weights for positive and negative examples
    based on the number of positive and negative examples in that class.

    Parameters
    ----------
    labels : Labels
        Binary prediction labels of shape (n_samples, n_classes)

    Returns
    -------
    sample_weights : npt.NDArray
        Weight for each sample of shape (n_samples, n_classes).

    Raises
    ------
    TypeError
        If labels is not npt.NDArray.
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
    array_check(labels)
    n_samples = len(labels)
    if n_samples == 0:
        raise ValueError("The input labels are empty")

    pos_counts = np.sum(labels, axis=0)

    # Check for zero positive labels in ANY class column
    if np.any(pos_counts == 0):
        missing_classes = np.where(pos_counts == 0)[0]
        raise ZeroDivisionError(f"Classes {missing_classes} have no positive labels.")

    neg_counts = n_samples - pos_counts

    # Broadcasting the counts across the label mask
    pos_part = labels * pos_counts
    neg_part = (labels == 0) * neg_counts

    return n_samples / (pos_part + neg_part)


def inverse_frequency_single_sample_weights(labels: Labels) -> npt.NDArray:
    """
    Create sample weights using the inverse frequency of positive
    and negative samples.

    One set of weights is created and used for each class based on the
    total number of positive and negative examples. Weights are normalised
    so that negative weights are 1.

    Parameters
    ----------
    labels : Labels
        Binary prediction labels of shape (n_samples, n_classes)

    Returns
    -------
    sample_weights : npt.NDArray
        Weight for each sample of shape (n_samples,).

    Raises
    ------
    TypeError
        If labels is not npt.NDArray.
    ValueError
        If labels is empty.
    ZeroDivisionError
        If there are no positive labels.

    """
    array_check(labels)
    n_samples = len(labels)
    if n_samples == 0:
        raise ValueError("The input labels are empty")

    # Use any() to find samples with at least one positive label
    is_pos_sample = np.any(labels > 0, axis=1)
    pos_count = np.sum(is_pos_sample)

    if pos_count == 0:
        raise ZeroDivisionError("No samples with positive labels found")

    weights = np.ones(n_samples, dtype=float)
    # Ratio: (Total - Pos) / Pos
    pos_weight_val = (n_samples - pos_count) / pos_count
    weights[is_pos_sample] = pos_weight_val

    return weights


def inverse_frequency_class_weights(labels: Labels) -> npt.NDArray:
    """
    Create class weights of the positive class using inverse frequency of
    the positive class.

    Parameters
    ----------
    labels : Labels
        Binary prediction labels of shape (n_samples, n_classes)

    Returns
    -------
    class_weights : npt.NDArray
        Weight for each class of shape (n_classes,).

    Raises
    ------
    TypeError
        If labels is not npt.NDArray.
    ValueError
        If labels is empty.
    ZeroDivisionError
        If there are no positive labels.

    """
    array_check(labels)
    n_samples = len(labels)
    if n_samples == 0:
        raise ValueError("The input labels are empty")

    pos_counts = np.sum(labels, axis=0)
    if np.any(pos_counts == 0):
        raise ZeroDivisionError("One or more classes have no positive labels")

    return n_samples / pos_counts


def negative_positive_ratio_sample_weights(labels: Labels) -> npt.NDArray:
    """
    Create sample weights using the ratio between negative and
    positive samples.

    Parameters
    ----------
    labels : Labels
        Binary prediction labels of shape (n_samples, n_classes)

    Returns
    -------
    sample_weights : npt.NDArray
        Weight for each sample of shape (n_samples, n_classes).

    Raises
    ------
    TypeError
        If labels is not npt.NDArray.
    ValueError
        If labels is empty.
    ZeroDivisionError
        If there are no positive labels.

    """
    array_check(labels)
    n_samples = len(labels)
    if n_samples == 0:
        raise ValueError("The input labels are empty")

    pos_counts = np.sum(labels, axis=0)
    if np.any(pos_counts == 0):
        raise ZeroDivisionError("One or more classes have no positive labels")

    neg_counts = n_samples - pos_counts
    ratios = neg_counts / pos_counts

    # Result is 1 for negatives, and ratio for positives
    return np.where(labels > 0, ratios, 1.0)


def negative_positive_ratio_class_weights(labels: Labels) -> npt.NDArray:
    """
    Create class weights of the positive class using the ratio
    between the number of samples in the negative and positive classes.

    Parameters
    ----------
    labels : Labels
        Binary prediction labels of shape (n_samples, n_classes)

    Returns
    -------
    class_weights : npt.NDArray
        Weight for each class of shape (n_classes,).

    Raises
    ------
    TypeError
        If labels is not npt.NDArray.
    ValueError
        If labels is empty.
    ZeroDivisionError
        If there are no positive labels.

    """
    array_check(labels)
    n_samples = len(labels)
    if n_samples == 0:
        raise ValueError("The input labels are empty")

    pos_counts = np.sum(labels, axis=0)
    if np.any(pos_counts == 0):
        raise ZeroDivisionError("One or more classes have no positive labels")

    return (n_samples - pos_counts) / pos_counts
