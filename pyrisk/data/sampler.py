"""
Sampler functions module.

This module provides functions to sample the data to address
class imbalance.

Functions:
- data_sampler: Samples the data and labels to adjust the class imbalance.
- random_sampler: Randomly select labels to achieve the target ratio
    between minority and majority classes.
- group_random_sampler: Randomly select labels to achieve the target ratio
    between minority and majority classes in each group.
- mean_dist_sampler: Computes the mean data sample of the majority class
    and uses the distance to it to select examples.
- group_mean_dist_sampler: Computes the mean data sample of the majority
    class in each group and uses the distance to it to select examples.
"""

from copy import deepcopy

import numpy as np

from pyrisk.utils.exceptions import array_check, array_dim_check


def data_sampler(
    data, labels, target_ratio=0.25, sampler_fn="random_sampler", groups=[], **kwargs
):
    """
    Samples the data and labels to adjust the class imbalance.

    The majority class is assumed to have a False or 0 label.

    Parameters
    ----------
    data : pd.DataFrame
        Data to sample of shape (n_samples, n_features).
    labels : array-like
        Binary prediction labels of shape (n_samples, n_classes).
    target_ratio : float, default: 0.25
        Target ratio between the minority and majority classes.
    sampler_fn : str, default: "random_sampler"
        Sampler function to use to sample the data.
    groups : pd.Series, default: []
        List containing groups for the group_sampler function.
    **kwargs
        Extra arguments for the sampler functions.

    Returns
    -------
    X : pd.DataFrame
        Sampled data.
    y : np.array
        Sampled labels.
    groups : pd.Series or None
        Groups of the examples, None if not specified.

    Raises
    ------
    TypeError
        If labels is not array-like.

    """
    sample_idx = np.array([])  # Empty sample index

    match sampler_fn:
        case "random_sampler":
            sample_idx = random_sampler(labels, target_ratio)
        case "group_random_sampler":
            sample_idx = group_random_sampler(labels, target_ratio, groups)
        case "mean_dist_sampler":
            sample_idx = mean_dist_sampler(
                data, labels, target_ratio, kwargs["hard_percent"]
            )
        case "group_mean_dist_sampler":
            sample_idx = group_mean_dist_sampler(
                data, labels, target_ratio, groups, kwargs["hard_percent"]
            )
        case _:
            raise ValueError(f"{sampler_fn} invalid sampler function")

    X = data.iloc[sample_idx]
    y = labels[sample_idx]

    if len(groups) != 0:
        return X, y, groups.iloc[sample_idx]

    return X, y, None


def random_sampler(labels, target_ratio):
    """
    Randomly select labels to achieve the target ratio between minority and
    majority classes.

    Parameters
    ----------
    labels : array-like
        Binary prediction labels of shape (n_samples, n_classes).
    target_ratio : float
        Ratio of minority over majority classes to achieve.

    Returns
    -------
    sample_idx : np.array(n_samples,)
        Index list of examples to achieve target ratio.

    Raises
    ------
    TypeError
        If labels is not array-like.

    """
    array_check(labels)  # Check that labels is array-like

    label_sums = np.sum(labels, axis=1)  # Sum to find example with at least one 1
    n_min_class = np.sum(label_sums != 0)  # Minority class examples
    n_maj_class = np.round(n_min_class / target_ratio)  # Majority class examples

    min_idx = np.where(label_sums > 1)[0]
    maj_idx = np.random.choice(  # Select examples so that target ratio is achieved
        np.where(label_sums == 0)[0], size=int(n_maj_class), replace=False
    )

    return np.concatenate((min_idx, maj_idx))


def group_random_sampler(labels, target_ratio, groups):
    """
    Randomly select labels to achieve the target ratio between minority and
    majority classes in each group.

    Parameters
    ----------
    labels : array-like
        Binary prediction labels of shape (n_samples, n_classes).
    target_ratio : float
        Ratio of minority over majority classes to achieve.
    groups : array-like
        List of groups in which labels belong of shape (n_samples,).

    Returns
    -------
    sample_idx : np.array(n_samples,)
        Index list of examples to achieve target ratio.

    Raises
    ------
    TypeError
        If labels is not array-like.
    ValueError
        If labels and group do not have the same dimension.

    """
    array_check(labels)
    array_dim_check(labels, groups, dim=0)

    sample_idx = np.array([], dtype=int)  # Empty array for the majority class index
    n_groups = np.unique(groups)

    for group in n_groups:
        group_idx = np.where(groups == group)[0]
        group_data = labels[group_idx]
        sample_idx = np.concatenate(
            (sample_idx, group_idx[random_sampler(group_data, target_ratio)])
        )

    return sample_idx


def mean_dist_sampler(data, labels, target_ratio, hard_percent=0.5):
    """
    Computes the mean data sample of the majority class and uses the
    distance to it to select examples.

    The examples are sorted based on their distance to the mean.
    The hardest examples are the ones that have the greatest distance to
    the mean and the easiest are the ones closest to the mean.

    Parameters
    ----------
    data : pd.DataFrame
        Data to sample of shape (n_samples, n_features).
    labels : array-like
        Binary prediction labels of shape (n_samples, n_classes).
    target_ratio : float
        Ratio of minority over majority classes to achieve.
    hard_percent : float, default: 0.5
        Percentage of examples that are considered hard, between 0 and 1.
        If hard_percent is 0.5, half of the examples are chosen from
        the end of the sorted list and the other half from the beginning.

    Returns
    -------
    sample_idx : np.array(n_samples,)
        Index list of examples to achieve target ratio.

    Raises
    ------
    TypeError
        If labels is not array-like.
    ValueError
        If hard_percent is not between 0 and 1.

    """
    array_check(labels)
    if hard_percent > 1 or hard_percent < 0:
        raise ValueError(
            f"hard_percent should be between 0 and 1, but got {hard_percent}"
        )

    label_sums = np.sum(labels, axis=1)  # Sum to find example with at least one 1
    n_min_class = np.sum(label_sums != 0)  # Minority class examples
    n_maj_class = np.round(n_min_class / target_ratio)  # Majority class examples

    maj_class_data = data.iloc[label_sums == 0]
    mean_maj_class = np.mean(maj_class_data, axis=0)

    # Get the distance to the mean
    dist = np.linalg.norm(mean_maj_class - maj_class_data, axis=1)
    sorted_dist_idx = np.argsort(dist)

    hard_samples_idx = sorted_dist_idx[-round(n_maj_class * hard_percent) :]
    easy_samples_idx = sorted_dist_idx[: round(n_maj_class * (1 - hard_percent))]

    return np.concatenate((easy_samples_idx, hard_samples_idx))


def group_mean_dist_sampler(data, labels, target_ratio, groups, hard_percent=0.5):
    """
    Computes the mean data sample of the majority class in each group and
    uses the distance to it to select examples.

    The examples are sorted based on their distance to the mean.
    The hardest examples are the ones that have the greatest distance to
    the mean and the easiest are the ones closest to the mean.

    Parameters
    ----------
    data : pd.DataFrame
        Data to sample of shape (n_samples, n_features).
    labels : array-like
        Binary prediction labels of shape (n_samples, n_classes).
    target_ratio : float
        Ratio of minority over majority classes to achieve.
    groups : array-like
        List of groups in which labels belong of shape (n_samples,).
    hard_percent : float, default: 0.5
        Percentage of examples that are considered hard, between 0 and 1.
        If hard_percent is 0.5, half of the examples are chosen from
        the end of the sorted list and the other half from the beginning.

    Returns
    -------
    sample_idx : np.array(n_samples,)
        Index list of examples to achieve target ratio.

    Raises
    ------
    TypeError
        If labels is not array-like.
    ValueError
        If labels and group do not have the same dimension.

    """
    array_check(labels)
    array_dim_check(labels, groups, dim=0)
    X = deepcopy(data)  # Create copy of data to not mess with actual data

    if groups.name in X.columns:
        # Remove group name to avoid calculation in the mean
        X = X.drop(groups.name, axis=1)

    sample_idx = np.array([], dtype=int)  # Empty array for the majority class index
    n_groups = np.unique(groups)

    for group in n_groups:
        group_idx = np.where(groups == group)[0]
        group_data = X.iloc[group_idx]
        group_labels = labels[group_idx]
        sample_idx = np.concatenate(
            (
                sample_idx,
                group_idx[
                    mean_dist_sampler(
                        group_data, group_labels, target_ratio, hard_percent
                    )
                ],
            )
        )

    return sample_idx
