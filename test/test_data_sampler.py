#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_data_sampler.py

Test functions for the sampler functions, which balances the class distribution
by selecting a target ratio between the minority and majority classes.
"""
import numpy as np
import pytest

from pyrisk.data.sampler import (
    group_mean_dist_sampler,
    group_random_undersampler,
    mean_dist_sampler,
    random_undersampler,
)

# Single label test data
single_labels = np.array(
    [
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [1],
        [1],
        [1],
        [1],
    ]
)  # 4 minority (1s) and 16 majority (0s)
single_groups = np.array(
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1]
)  # Group 0: 1 min, 4 maj Group 1: 3 min, 12 maj
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
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1],
        [1, 0],
        [0, 1],
        [1, 1],
    ]
)  # 6 minority and 14 majority
bi_groups = np.array(
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 0, 1, 1, 1, 2, 2]
)  # Group 0: 1 min, 3 maj Group 1: 3 min, 7 maj, Group 2: 2 min, 4 maj
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
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 1, 1],
        [1, 1, 1],
        [1, 0, 1],
        [1, 0, 0],
        [0, 1, 0],
    ]  # 5 minority and 15 majority
)
tri_groups = np.array(
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1]
)  # Group 0: 3 min, 10 maj Group 1: 2 min, 5 maj
labels_empty = np.array([])  # Empty array to test edge case
labels_no_minority = np.zeros((20, 1))
labels_groups_no_maj = np.array(
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
)


# Basic functionality tests
@pytest.mark.parametrize(
    "target_ratio, labels, n_min_class",
    [
        (0.5, single_labels, 4),
        (0.5, bi_labels, 6),
        (0.5, tri_labels, 5),
    ],
)
def test_random_undersampler_success(target_ratio, labels, n_min_class):
    sample_idx = random_undersampler(labels, target_ratio)
    sampled_labels = labels[sample_idx]

    if sampled_labels.shape[1] == 1:
        minority_count = np.sum(sampled_labels > 0)
    else:
        minority_count = np.sum(np.sum(sampled_labels, axis=1) > 0)

    assert minority_count == n_min_class
    assert (
        len(sampled_labels) == n_min_class + (1 / target_ratio) * n_min_class
    )  # Total samples length based on target_ratio and class imbalance


@pytest.mark.parametrize(
    "target_ratio, labels, n_maj_class",
    [
        (0.5, single_labels, 8),
        (1.0, single_labels, 4),
        (2.0, single_labels, 2),
        (0.5, bi_labels, 12),
        (1.0, bi_labels, 6),
        (3.0, bi_labels, 2),
        (0.5, tri_labels, 10),
        (1.0, tri_labels, 5),
        (5.0, tri_labels, 1),
    ],
)
def test_random_undersampler_target_ratio(target_ratio, labels, n_maj_class):
    sample_idx = random_undersampler(labels, target_ratio)
    sampled_labels = labels[sample_idx]

    if sampled_labels.shape[1] == 1:
        minority_count = np.sum(sampled_labels > 0)
        majority_count = np.sum(sampled_labels == 0)
    else:
        minority_count = np.sum(np.sum(sampled_labels, axis=1) > 0)
        majority_count = np.sum(np.sum(sampled_labels, axis=1) == 0)

    # Check if the ratio is close to target ratio
    if minority_count > 0:
        ratio = majority_count / minority_count
        assert np.isclose(
            ratio, 1 / target_ratio, atol=0.1
        )  # Allow for slight imprecision

    # Assert that the target ratio is close to what we selected
    assert majority_count == n_maj_class  # Should match expected number of majority


# Edge case: No minority class samples
def test_random_undersampler_no_minority():
    sample_idx = random_undersampler(labels_no_minority, target_ratio=0.5)
    sampled_labels = labels_no_minority[sample_idx]

    # If no minority class, only majority class samples should be selected
    assert np.all(sampled_labels == 0)


# Raises a ValueError
@pytest.mark.parametrize(
    "target_ratio, labels",
    [
        (0.5, labels_empty),
        (0.1, single_labels),
        (0.0, single_labels),
        (0.1, bi_labels),
        (0.0, bi_labels),
        (0.2, tri_labels),
        (-0.1, tri_labels),
    ],
)
def test_random_undersampler_value_error(target_ratio, labels):
    with pytest.raises(ValueError):
        random_undersampler(labels, target_ratio)


# Raises a TypeError
@pytest.mark.parametrize(
    "labels",
    [0.5, "string"],
)
def test_random_undersampler_type_error(labels):
    with pytest.raises(TypeError):
        random_undersampler(labels, target_ratio=0.5)


# Basic functionality test
@pytest.mark.parametrize(
    "target_ratio, labels, groups, n_min_class_per_group",
    [
        (1.0, single_labels, single_groups, [1, 3]),
        (1.0, bi_labels, bi_groups, [1, 3, 2]),
        (1.0, tri_labels, tri_groups, [3, 2]),
    ],
)
def test_group_random_undersampler_success(
    target_ratio, labels, groups, n_min_class_per_group
):
    sample_idx = group_random_undersampler(labels, target_ratio, groups)
    sampled_labels = labels[sample_idx]

    for group, n_min_class in zip(np.unique(groups), n_min_class_per_group):
        group_idx = np.where(groups == group)[0]
        group_samples = sampled_labels[np.isin(sample_idx, group_idx)]

        if group_samples.shape[1] == 1:  # Single class case (binary)
            minority_count = np.sum(group_samples > 0)
        else:  # Multiclass case
            minority_count = np.sum(np.sum(group_samples, axis=1) > 0)

        assert (
            minority_count == n_min_class
        )  # Ensure the expected minority count for the group
        assert (
            len(group_samples) == n_min_class + (1 / target_ratio) * n_min_class
        )  # Total samples length based on target_ratio and class imbalance


# Basic functionality test
@pytest.mark.parametrize(
    "target_ratio, labels, groups, n_maj_class_per_group",
    [
        (1.0, single_labels, single_groups, [1, 3]),
        (1.0, bi_labels, bi_groups, [1, 3, 2]),
        (1.0, tri_labels, tri_groups, [3, 2]),
        (0.5, single_labels, single_groups, [2, 6]),
        (0.5, bi_labels, bi_groups, [2, 6, 4]),
        (0.5, tri_labels, tri_groups, [6, 4]),
    ],
)
def test_group_random_undersampler_target_ratio(
    target_ratio, labels, groups, n_maj_class_per_group
):
    sample_idx = group_random_undersampler(labels, target_ratio, groups)
    sampled_labels = labels[sample_idx]

    for group, n_maj_class in zip(np.unique(groups), n_maj_class_per_group):
        group_idx = np.where(groups == group)[0]
        group_samples = sampled_labels[np.isin(sample_idx, group_idx)]

        if group_samples.shape[1] == 1:  # Single class case (binary)
            minority_count = np.sum(group_samples > 0)
            majority_count = np.sum(group_samples == 0)
        else:  # Multiclass case
            minority_count = np.sum(np.sum(group_samples, axis=1) > 0)
            majority_count = np.sum(np.sum(group_samples, axis=1) == 0)

        # Check if the ratio is close to target ratio
        if minority_count > 0:
            ratio = majority_count / minority_count
            assert np.isclose(
                ratio, 1 / target_ratio, atol=0.1
            )  # Allow for slight imprecision

        assert majority_count == n_maj_class  # Should match expected number of minority


# Raises a TypeError if labels is not array-like
@pytest.mark.parametrize(
    "labels",
    [0.5, "string", None],  # Invalid inputs (should raise TypeError)
)
def test_group_random_undersampler_type_error(labels):
    with pytest.raises(TypeError):
        group_random_undersampler(labels, target_ratio=0.5, groups=single_groups)


# Labels and groups should have the same number of samples
def test_group_random_undersampler_mismatched_dimensions():
    with pytest.raises(ValueError):
        groups = np.array([0, 1])
        group_random_undersampler(single_labels, target_ratio=0.5, groups=groups)


# Edge case: No minority class samples in any group
def test_group_random_undersampler_no_minority():
    sample_idx = group_random_undersampler(
        labels_no_minority, target_ratio=0.5, groups=labels_groups_no_maj
    )
    sampled_labels = single_labels[sample_idx]

    # If no minority class, only majority class samples should be selected
    assert np.all(sampled_labels == 0)
