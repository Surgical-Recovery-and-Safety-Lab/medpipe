#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_data_weighting.py

Test functions for the weighting function that create weights for
samples or classes to adjust class imbalance.
"""
import numpy as np
import pytest

from pyrisk.data.weighting import (
    inverse_frequency_class_weights,
    inverse_frequency_multiclass_sample_weights,
    inverse_frequency_single_sample_weights,
    negative_positive_ratio_class_weights,
    negative_positive_ratio_sample_weights,
)

# Single label test data
single_labels = np.array(
    [[0], [0], [0], [0], [1], [1]]
)  # 2 minority (1s) and 4 majority (0s)
bi_labels = np.array(
    [
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1],
    ]
)  # 3 minority and 8 majority
tri_labels = np.array(
    [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 1, 1],
        [1, 1, 1],
        [1, 0, 1],
        [1, 0, 0],
        [1, 1, 0],
    ]  # 5 minority and 10 majority
)
labels_empty = np.array([])  # Empty array to test edge case
labels_no_minority = np.zeros((5, 1))


# Basic functionality tests
@pytest.mark.parametrize(
    "labels, true_weights",
    [
        (single_labels, [3.0]),
        (bi_labels, [6.0, 6.0]),
        (tri_labels, [15 / 4, 5.0, 5.0]),
    ],
)
def test_inverse_frequency_class_weights_success(labels, true_weights):
    weights = inverse_frequency_class_weights(labels)
    assert (weights == np.array(true_weights)).all()


def test_inverse_frequency_class_weights_bad_input():
    with pytest.raises(ValueError):
        _ = inverse_frequency_class_weights(labels_empty)


@pytest.mark.parametrize(
    "labels",
    [
        42,
        3.14,
        None,
        ["a", "b", "c"],
    ],
)
def test_inverse_frequency_class_weights_type_error(labels):
    with pytest.raises(TypeError):
        _ = inverse_frequency_class_weights(labels)


def test_inverse_frequency_class_weights_no_positive():
    with pytest.raises(ZeroDivisionError):
        _ = inverse_frequency_class_weights(labels_no_minority)


# Basic functionality tests
@pytest.mark.parametrize(
    "labels, true_weights",
    [(single_labels, [2.0]), (bi_labels, [5.0, 5.0]), (tri_labels, [2.75, 4.0, 4.0])],
)
def test_negative_positive_ratio_class_weights_success(labels, true_weights):
    weights = negative_positive_ratio_class_weights(labels)
    assert (weights == np.array(true_weights)).all()


def test_negative_positive_ratio_class_weights_bad_input():
    with pytest.raises(ValueError):
        _ = negative_positive_ratio_class_weights(labels_empty)


@pytest.mark.parametrize(
    "labels",
    [
        42,
        3.14,
        None,
        ["a", "b", "c"],
    ],
)
def test_negative_positive_ratio_class_weights_type_error(labels):
    with pytest.raises(TypeError):
        _ = negative_positive_ratio_class_weights(labels)


def test_negative_positive_ratio_class_weights_no_positive():
    with pytest.raises(ZeroDivisionError):
        _ = negative_positive_ratio_class_weights(labels_no_minority)


# Basic functionality tests
@pytest.mark.parametrize(
    "labels, true_weights",
    [
        (single_labels, [[1.5], [3.0]]),
        (bi_labels, [[1.2, 1.2], [6.0, 6.0]]),
        (tri_labels, [[15 / 11, 1.25, 1.25], [15 / 4, 5.0, 5.0]]),
    ],
)
def test_inverse_frequency_multiclass_sample_weights_success(labels, true_weights):
    weights = inverse_frequency_multiclass_sample_weights(labels)
    neg_weights = ~np.array(labels, dtype=bool) * true_weights[0]
    pos_weights = labels * true_weights[1]
    assert (weights == np.array(neg_weights + pos_weights)).all()


def test_inverse_frequency_multiclass_sample_weights_bad_input():
    with pytest.raises(ValueError):
        _ = inverse_frequency_multiclass_sample_weights(labels_empty)


@pytest.mark.parametrize(
    "labels",
    [
        42,
        3.14,
        None,
        ["a", "b", "c"],
    ],
)
def test_inverse_frequency_multiclass_sample_weights_type_error(labels):
    with pytest.raises(TypeError):
        _ = inverse_frequency_multiclass_sample_weights(labels)


def test_inverse_frequency_multiclass_sample_weights_no_positive():
    with pytest.raises(ZeroDivisionError):
        _ = inverse_frequency_multiclass_sample_weights(labels_no_minority)


# Basic functionality tests
@pytest.mark.parametrize(
    "labels, pos_weight",
    [
        (single_labels, 2.0),
        (bi_labels, 3.0),
        (tri_labels, 2.0),
    ],
)
def test_inverse_frequency_single_sample_weights_success(labels, pos_weight):
    weights = inverse_frequency_single_sample_weights(labels)
    pos_examples = np.sum(labels, axis=1) > 0
    true_weights = np.ones(len(labels))
    true_weights[pos_examples] *= pos_weight
    assert (weights == true_weights).all()


def test_inverse_frequency_single_sample_weights_bad_input():
    with pytest.raises(ValueError):
        _ = inverse_frequency_single_sample_weights(labels_empty)


@pytest.mark.parametrize(
    "labels",
    [
        42,
        3.14,
        None,
    ],
)
def test_inverse_frequency_single_sample_weights_type_error(labels):
    with pytest.raises(TypeError):
        _ = inverse_frequency_single_sample_weights(labels)


def test_inverse_frequency_single_sample_weights_no_positive():
    with pytest.raises(ZeroDivisionError):
        _ = inverse_frequency_single_sample_weights(labels_no_minority)


# Basic functionality tests
@pytest.mark.parametrize(
    "labels, true_weights",
    [
        (single_labels, [[1.0], [2.0]]),
        (bi_labels, [[1.0, 1.0], [5.0, 5.0]]),
        (tri_labels, [[1.0, 1.0, 1.0], [2.75, 4.0, 4.0]]),
    ],
)
def test_negative_positive_ratio_sample_weights_success(labels, true_weights):
    weights = negative_positive_ratio_sample_weights(labels)
    neg_weights = ~np.array(labels, dtype=bool) * true_weights[0]
    pos_weights = labels * true_weights[1]
    assert (weights == np.array(neg_weights + pos_weights)).all()


def test_negative_positive_ratio_sample_weights_bad_input():
    with pytest.raises(ValueError):
        _ = negative_positive_ratio_sample_weights(labels_empty)


@pytest.mark.parametrize(
    "labels",
    [
        42,
        3.14,
        None,
        ["a", "b", "c"],
    ],
)
def test_negative_positive_ratio_sample_weights_type_error(labels):
    with pytest.raises(TypeError):
        _ = negative_positive_ratio_sample_weights(labels)


def test_negative_positive_ratio_sample_weights_no_positive():
    with pytest.raises(ZeroDivisionError):
        _ = negative_positive_ratio_sample_weights(labels_no_minority)
