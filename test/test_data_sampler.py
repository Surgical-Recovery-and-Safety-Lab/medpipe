#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_data_sampler.py

Test functions for the sampler functions, which balances the class distribution
by selecting a target ratio between the minority and majority classes.
"""

import numpy as np
import pandas as pd
import pytest

from medpipe.data.sampler import (
    data_sampler,
    group_mean_dist_sampler,
    group_random_oversampler,
    group_random_undersampler,
    group_smote,
    mean_dist_sampler,
    random_oversampler,
    random_undersampler,
    smote,
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


@pytest.mark.parametrize(
    "labels, data_len",
    [
        (single_labels, 12),
        (bi_labels, 13),
        (tri_labels, 13),
    ],
)
def test_data_sampler_success(labels, data_len):
    data = pd.DataFrame(np.random.rand(20, 2))
    X, y, grps = data_sampler(data, labels, 0.5)

    assert len(X) == data_len
    assert len(y) == data_len
    assert len(grps) == 0


@pytest.mark.parametrize(
    "labels, reduction_factor, data_len",
    [
        (single_labels, 1.0, 8),
        (single_labels, 0.25, 16),
        (bi_labels, 1.0, 12),
        (bi_labels, 0.25, 16),
        (tri_labels, 1.0, 10),
        (tri_labels, 0.25, 16),
    ],
)
def test_data_sampler_reduction_factor(labels, reduction_factor, data_len):
    data = pd.DataFrame(np.random.rand(20, 2))
    X, y, grps = data_sampler(data, labels, reduction_factor)

    assert len(X) == data_len
    assert len(y) == data_len
    assert len(grps) == 0


@pytest.mark.parametrize(
    "sampler_fn, data_len",
    [
        ("random_undersampler", [8, 12, 10]),
        ("random_oversampler", [32, 28, 30]),
        ("mean_dist_sampler", [8, 12, 10]),
        ("smote", [32, 28, 20]),
    ],
)
def test_data_sampler_sampler_fn(sampler_fn, data_len):
    data = pd.DataFrame(np.random.rand(20, 2))
    labels = (single_labels, bi_labels, tri_labels)
    for i in range(3):
        if sampler_fn == "smote":
            X, y, grps = data_sampler(
                data,
                labels[i],
                reduction_factor=1.0,
                sampler_fn=sampler_fn,
                k_neighbors=1,
            )
        else:
            X, y, grps = data_sampler(
                data, labels[i], reduction_factor=1.0, sampler_fn=sampler_fn
            )

        assert len(X) == data_len[i]
        assert len(y) == data_len[i]
        assert len(grps) == 0


@pytest.mark.parametrize(
    "reduction_factor",
    [1.5, -1.0],
)
def test_data_sampler_value_error(reduction_factor):
    data = pd.DataFrame(np.random.rand(20, 2))
    with pytest.raises(ValueError):
        data_sampler(data, single_labels, reduction_factor)


@pytest.mark.parametrize(
    "labels",
    ["str", (1, 2, 3), {"a": 1}],
)
def test_data_sampler_type_error(labels):
    data = pd.DataFrame(np.random.rand(20, 2))
    with pytest.raises(TypeError):
        data_sampler(data, labels)


# Basic functionality tests
@pytest.mark.parametrize(
    "target_ratio, labels, n_min_class",
    [
        (2, single_labels, 4),
        (1, bi_labels, 6),
        (3, tri_labels, 5),
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
        len(sampled_labels) == n_min_class + target_ratio * n_min_class
    )  # Total samples length based on target_ratio and class imbalance


@pytest.mark.parametrize(
    "target_ratio, labels, n_maj_class",
    [
        (2.0, single_labels, 8),
        (1.0, single_labels, 4),
        (4.0, single_labels, 16),
        (2.0, bi_labels, 12),
        (1.0, bi_labels, 6),
        (1.5, bi_labels, 9),
        (2.0, tri_labels, 10),
        (1.0, tri_labels, 5),
        (3.0, tri_labels, 15),
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
        assert np.isclose(ratio, target_ratio, atol=0.1)  # Allow for slight imprecision

    # Assert that the target ratio is close to what we selected
    assert majority_count == n_maj_class  # Should match expected number of majority


# Edge case: No minority class samples
def test_random_undersampler_no_minority():
    sample_idx = random_undersampler(labels_no_minority, target_ratio=1)
    sampled_labels = labels_no_minority[sample_idx]

    # If no minority class, only majority class samples should be selected
    assert np.all(sampled_labels == 0)


# Raises a ValueError
@pytest.mark.parametrize(
    "target_ratio, labels",
    [
        (0.5, labels_empty),
        (0.0, single_labels),
        (0.9, bi_labels),
        (-0.1, tri_labels),
    ],
)
def test_random_undersampler_value_error(target_ratio, labels):
    with pytest.raises(ValueError):
        random_undersampler(labels, target_ratio)


# Raises a TypeError
@pytest.mark.parametrize(
    "labels",
    [1.0, "string"],
)
def test_random_undersampler_type_error(labels):
    with pytest.raises(TypeError):
        random_undersampler(labels, target_ratio=1)


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
        (2, single_labels, single_groups, [2, 6]),
        (2, bi_labels, bi_groups, [2, 6, 4]),
        (2, tri_labels, tri_groups, [6, 4]),
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
                ratio, target_ratio, atol=0.1
            )  # Allow for slight imprecision

        assert majority_count == n_maj_class  # Should match expected number of minority


# Raises a TypeError if labels is not array-like
@pytest.mark.parametrize(
    "labels",
    [0.5, "string", None],  # Invalid inputs (should raise TypeError)
)
def test_group_random_undersampler_type_error(labels):
    with pytest.raises(TypeError):
        group_random_undersampler(labels, target_ratio=1.0, groups=single_groups)


# Labels and groups should have the same number of samples
def test_group_random_undersampler_mismatched_dimensions():
    with pytest.raises(ValueError):
        groups = np.array([0, 1])
        group_random_undersampler(single_labels, target_ratio=1.0, groups=groups)


# Edge case: No minority class samples in any group
def test_group_random_undersampler_no_minority():
    sample_idx = group_random_undersampler(
        labels_no_minority, target_ratio=1.0, groups=labels_groups_no_maj
    )
    sampled_labels = labels_no_minority[sample_idx]

    # If no minority class, only majority class samples should be selected
    assert np.all(sampled_labels == 0)


@pytest.mark.parametrize(
    "target_ratio, labels, n_maj_class",
    [
        (1.0, single_labels, 16),
        (1.0, bi_labels, 14),
        (1.0, tri_labels, 15),
    ],
)
def test_random_oversampler_success(target_ratio, labels, n_maj_class):
    sample_idx = random_oversampler(labels, target_ratio)
    sampled_labels = labels[sample_idx]

    # Find majority samples (where all class columns are 0)
    is_maj = np.all(sampled_labels == 0, axis=1)
    majority_count = np.sum(is_maj)

    assert majority_count == n_maj_class
    # Total length should be n_maj + (n_maj * target_ratio)
    expected_len = n_maj_class + int(np.round(n_maj_class * target_ratio))
    assert len(sampled_labels) == expected_len


@pytest.mark.parametrize(
    "target_ratio, labels, expected_min_count",
    [
        (2.0, tri_labels, 8),  # 15 maj / 2.0  = 7.5 -> 8
        (1.0, tri_labels, 15),  # 15 maj / 1.0 = 15
    ],
)
def test_random_oversampler_target_ratio(target_ratio, labels, expected_min_count):
    sample_idx = random_oversampler(labels, target_ratio)
    sampled_labels = labels[sample_idx]

    is_min = np.any(sampled_labels > 0, axis=1)
    is_maj = ~is_min

    minority_count = np.sum(is_min)
    majority_count = np.sum(is_maj)

    # Check ratio: maj / min should be target_ratio
    calculated_ratio = majority_count / minority_count
    assert np.isclose(calculated_ratio, target_ratio, atol=0.4)
    assert minority_count == expected_min_count


# Raises a TypeError
@pytest.mark.parametrize(
    "labels",
    [0.5, "string"],
)
def test_random_oversampler_type_error(labels):
    with pytest.raises(TypeError):
        random_oversampler(labels, target_ratio=1.0)


@pytest.mark.parametrize(
    "target_ratio, labels, groups, expected_counts",
    [
        # Grp 0: 4 maj * 1.0 = 4 min | Grp 1: 12 maj * 1.0 = 12 min
        (1.0, single_labels, single_groups, {"min": [4, 12], "maj": [4, 12]}),
        # Grp 0: 3 maj * 1.0 = 3 min | Grp 1: 7 maj * 1.0 = 7 min | Grp 2: 4 maj * 1.0 = 4 min
        (1.0, bi_labels, bi_groups, {"min": [3, 7, 4], "maj": [3, 7, 4]}),
    ],
)
def test_group_random_oversampler_success(
    target_ratio, labels, groups, expected_counts
):
    sample_idx = group_random_oversampler(labels, target_ratio, groups)

    # To test groups, we need the group IDs for the sampled indices
    sampled_groups = groups[sample_idx]
    sampled_labels = labels[sample_idx]

    unique_grps = np.unique(groups)
    for i, grp_id in enumerate(unique_grps):
        grp_mask = sampled_groups == grp_id
        grp_labels = sampled_labels[grp_mask]

        is_min = np.any(grp_labels > 0, axis=1)

        assert np.sum(is_min) == expected_counts["min"][i]
        assert np.sum(~is_min) == expected_counts["maj"][i]


def test_group_random_oversampler_no_minority():
    # if no minority, just return majority.
    sample_idx = group_random_oversampler(labels_no_minority, 1.5, labels_groups_no_maj)
    assert len(sample_idx) == len(labels_no_minority)


@pytest.mark.parametrize(
    "target_ratio, labels",
    [
        (0.5, labels_empty),
        (-0.1, tri_labels),
    ],
)
def test_random_oversampler_value_error(target_ratio, labels):
    with pytest.raises(ValueError):
        random_oversampler(labels, target_ratio)


@pytest.mark.parametrize(
    "labels",
    [0.5, "string", None],  # Invalid inputs (should raise TypeError)
)
def test_group_random_oversampler_type_error(labels):
    with pytest.raises(TypeError):
        group_random_oversampler(labels, target_ratio=1.0, groups=single_groups)


# Labels and groups should have the same number of samples
def test_group_random_oversampler_mismatched_dimensions():
    with pytest.raises(ValueError):
        groups = np.array([0, 1])
        group_random_oversampler(single_labels, target_ratio=1.0, groups=groups)


def test_mean_dist_sampler_basic():
    # Setup: 4 features, 10 samples.
    # Majority class (label sum 0) at index 0-7, Minority at 8-9.
    data = pd.DataFrame(np.random.rand(10, 4))
    labels = np.zeros((10, 2))
    labels[8:] = 1

    # If target_ratio is 3, and we have 2 minority samples,
    # we need 2*3 = 6 majority samples.
    res_idx = mean_dist_sampler(data, labels, target_ratio=3.0, hard_percent=0.5)
    assert len(res_idx) == 8  # 2 minority + 4 majority
    assert isinstance(res_idx, np.ndarray)


def test_mean_dist_sampler_exceptions():
    data = pd.DataFrame(np.random.rand(5, 2))
    labels = np.zeros((5, 1))

    with pytest.raises(ValueError):
        mean_dist_sampler(data, labels, target_ratio=1.0, hard_percent=1.5)

    with pytest.raises(ValueError):
        mean_dist_sampler(data, labels, target_ratio=-0.1)


def test_group_mean_dist_sampler():
    data = pd.DataFrame(np.random.rand(20, 2))
    labels = np.zeros((20, 1))
    labels[18:] = 1  # 2 minority samples
    groups = pd.Series([0] * 10 + [1] * 10, name="group_col")

    # target_ratio 1.0 means it will try to match minority count per group
    # This function iterates through unique groups and calls mean_dist_sampler
    res_idx = group_mean_dist_sampler(data, labels, 1.0, groups)

    assert len(res_idx) > 0
    # Ensure it handled the group name removal from data if present
    data_with_group = data.copy()
    data_with_group["group_col"] = groups
    res_idx_with_name = group_mean_dist_sampler(data_with_group, labels, 1.0, groups)
    assert len(res_idx_with_name) == len(res_idx)


def test_smote_output_shapes():
    # Create a simple dataset where SMOTE can find neighbors
    data = pd.DataFrame(
        {
            "f1": [1, 1.1, 1.2, 5, 5.1, 5.2],
            "f2": [1, 1.1, 1.2, 5, 5.1, 5.2],
            "SEX": [0, 1, 0, 1, 0, 1],
        }
    )
    # Multilabel: 2 classes
    labels = np.array([[0, 0], [0, 0], [0, 0], [1, 1], [1, 1], [1, 1]])

    # Requesting a higher ratio to force generation
    X_gen, y_gen = smote(data, labels, target_ratio=1.0, k_neighbors=2)

    assert isinstance(X_gen, pd.DataFrame)
    assert y_gen.shape[1] == 2  # Check label dimensionality
    # Check if 'SEX' was rounded as per function logic
    if "SEX" in X_gen.columns:
        assert all(X_gen["SEX"].isin([0, 1]))


def test_smote_invalid_ratio():
    data = pd.DataFrame(np.random.rand(10, 2))
    labels = np.zeros((10, 2))
    with pytest.raises(ValueError):
        smote(data, labels, target_ratio=0, k_neighbors=1)


# --- Mock Data Setup ---
def create_smote_data():
    # Group 0: 20 majority, 5 minority (Needs oversampling)
    # Group 1: 20 majority, 20 minority (Already balanced)
    # Group 2: 5 total samples (Too small for k_neighbors=5, should be skipped)

    labels = np.array(
        [[0]] * 20
        + [[1]] * 5  # Group 0
        + [[0]] * 20
        + [[1]] * 20  # Group 1
        + [[0]] * 4
        + [[1]] * 1  # Group 2
    )

    # 70 total samples, 10 features
    data = np.random.rand(70, 10)
    groups = np.array([0] * 25 + [1] * 40 + [2] * 5)

    return data, labels, groups


def test_group_smote_logic():
    data, labels, groups = create_smote_data()
    target_ratio = 1.0  # Want 1:1 ratio
    k_neighbors = 2  # Small k to ensure it works on smaller groups

    _, y_res, g_res = group_smote(data, labels, target_ratio, groups, k_neighbors)

    # Check Group 0 (Was 20 maj, 5 min -> Should now be 20 maj, 20 min)
    g0_mask = g_res == 0
    g0_y = y_res[g0_mask]
    assert np.sum(g0_y == 0) == 20
    assert np.sum(g0_y == 1) == 20

    # Check Group 1 (Was already 20:20 -> Should remain 40 total)
    g1_mask = g_res == 1
    assert len(y_res[g1_mask]) == 40

    # Check Group 2 (Too small or already processed)
    # Note: Our optimized code skips or returns original if SMOTE fails
    g2_mask = g_res == 2
    assert len(y_res[g2_mask]) == 5


def test_group_smote_pandas_compatibility():
    data, labels, groups = create_smote_data()
    df_data = pd.DataFrame(data, columns=[f"feat_{i}" for i in range(10)])

    X_res, y_res, _ = group_smote(df_data, labels, 1.0, groups, k_neighbors=3)

    # Verify X_res is still a DataFrame
    assert isinstance(X_res, pd.DataFrame)
    assert X_res.shape[1] == 10
    # Verify index is reset or handled (concatenation of DFs)
    assert len(X_res) == len(y_res)


def test_group_smote_k_neighbors_error_handling():
    # Create a group where minority class has only 1 sample
    labels = np.array([[0]] * 10 + [[1]] * 1)
    data = np.random.rand(11, 2)
    groups = np.array([0] * 11)

    # SMOTE with k=5 on 1 minority sample will raise a ValueError internally.
    # Our optimized function catches this and should return the original data.
    X_res, y_res, g_res = group_smote(data, labels, 1.0, groups, k_neighbors=5)

    assert len(X_res) == 11
    assert np.sum(y_res == 1) == 1  # No new samples generated due to error handling


@pytest.mark.parametrize("bad_ratio", [0, -1.0])
def test_group_smote_value_errors(bad_ratio):
    data, labels, groups = create_smote_data()
    with pytest.raises(ValueError):
        group_smote(data, labels, bad_ratio, groups, k_neighbors=3)


def test_group_smote_mismatched_dimensions():
    data, labels, _ = create_smote_data()
    wrong_groups = np.array([0, 1])
    with pytest.raises(ValueError):
        group_smote(data, labels, 0.5, wrong_groups, k_neighbors=3)
