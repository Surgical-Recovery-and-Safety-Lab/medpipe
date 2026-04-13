"""
Sampler functions module.

This module provides functions to sample the data to address
class imbalance.

Functions:
- data_sampler: Samples the data and labels to adjust the class imbalance.
- random_undersampler: Randomly select labels to achieve the target ratio
    between minority and majority classes by undersampling majority class.
- group_random_undersampler: Randomly select labels to achieve the target ratio
    between minority and majority classes in each group.
- random_oversampler: Randomly select labels to achieve the target ratio
    between minority and majority classes by oversampling minority class.
- group_random_oversampler: Randomly select labels to achieve the target ratio
    between minority and majority classes in each group.
- mean_dist_sampler: Computes the mean data sample of the majority class
    and uses the distance to it to select examples.
- group_mean_dist_sampler: Computes the mean data sample of the majority
    class in each group and uses the distance to it to select examples.
- smote: Oversample minority class using Synthetic Minority Over-Sampling
    Technique (SMOTE).
- group_smote: Oversample minority class using Synthetic Minority
    Over-Sampling Technique (SMOTE) in each group.
"""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from imblearn.over_sampling import SMOTE
from pandas import DataFrame, Index, Series, concat

from medpipe._types import Labels, PData
from medpipe.utils.exceptions import array_check, array_dim_check

from .utils import get_data_from_idx

if TYPE_CHECKING:
    import numpy.typing as npt


def data_sampler(
    data: PData,
    labels: Labels,
    target_ratio: float = 0.25,
    sampler_fn: str = "random_undersampler",
    groups: Series = Series([]),
    **kwargs: Any,
) -> tuple[PData, Labels, Series]:
    """
    Samples the data and labels to adjust the class imbalance.

    The majority class is assumed to have a False or 0 label.
    The new set will have an imbalance equal to:
        IR * target_ratio, where IR is the current imbalance ratio.

    If the target ratio is too small, the algorithm defaults to
    obtain a balanced dataset.

    Parameters
    ----------
    data : PData
        Data to sample of shape (n_samples, n_features).
    labels : Labels
        Binary prediction labels of shape (n_samples, n_classes).
    target_ratio : float, default: 0.25
        Target ratio between the minority and majority classes.
    sampler_fn : str, default: "random_undersampler"
        Sampler function to use to sample the data.
    groups : Series, default: Series([])
        List containing groups for the group_sampler function.
    **kwargs : Any
        Extra arguments for the sampler functions.

    Returns
    -------
    X : PData
        Sampled data.
    y : Labels
        Sampled labels.
    groups : Series
        Groups of the examples. Empty series if not needed.

    Raises
    ------
    TypeError
        If labels is not npt.NDArray.
    ValueError
        If target_ratio is less than 0.0.

    """
    sample_idx = np.array([])  # Empty sample index

    if target_ratio > 0:
        imbalance_ratio = (len(labels) - np.sum(labels)) / np.sum(labels)
        new_ratio = 1 / (imbalance_ratio * target_ratio)

        if (imbalance_ratio * new_ratio) < 1:
            # Set to 1 to get balanced dataset
            new_ratio = 1

    elif target_ratio == 0:
        new_ratio = 1  # Set to 1 to get balanced dataset
    else:
        raise ValueError(f"Target ratio should be positive, but got {target_ratio}")

    match sampler_fn:
        case "random_undersampler":
            sample_idx = random_undersampler(labels, new_ratio)
        case "group_random_undersampler":
            sample_idx = group_random_undersampler(labels, new_ratio, groups)
        case "random_oversampler":
            sample_idx = random_oversampler(labels, new_ratio)
        case "group_random_oversampler":
            sample_idx = group_random_oversampler(labels, new_ratio, groups)
        case "mean_dist_sampler":
            sample_idx = mean_dist_sampler(
                data, labels, new_ratio, kwargs["hard_percent"]
            )
        case "group_mean_dist_sampler":
            sample_idx = group_mean_dist_sampler(
                data, labels, new_ratio, groups, kwargs["hard_percent"]
            )
        case "smote":
            X_gen, y_gen = smote(data, labels, new_ratio, kwargs["k_neighbors"])
            if isinstance(data, DataFrame):
                X = concat((data, cast(DataFrame, X_gen)))
            else:
                X = np.concatenate((data, X_gen))
            return X, np.concatenate((labels, y_gen)), Series([])
        case "group_smote":
            return group_smote(data, labels, new_ratio, groups, kwargs["k_neighbors"])
        case _:
            raise ValueError(f"{sampler_fn} invalid sampler function")

    X = get_data_from_idx(data, sample_idx)
    y = labels[sample_idx]

    if groups.empty:
        return X, y, groups
    return X, y, groups.iloc[sample_idx]


def random_undersampler(labels: Labels, target_ratio: float) -> npt.NDArray:
    """
    Randomly select labels to achieve the target ratio between minority and
    majority classes by undersampling majority class.

    Parameters
    ----------
    labels : Labels
        Binary prediction labels of shape (n_samples, n_classes).
    target_ratio : float
        Ratio of minority over majority classes to achieve.

    Returns
    -------
    sample_idx : npt.NDArray
        Index list of examples to achieve target ratio of shape (n_samples,).

    Raises
    ------
    TypeError
        If labels is not npt.NDArray.
    ValueError
        If target_ratio is less than 0.0.

    """
    array_check(labels)  # Check that labels is array-like

    if target_ratio <= 0:
        raise ValueError(f"Target ratio should be positive, but got {target_ratio}")

    label_sums = np.sum(labels, axis=1)  # Sum to find example with at least one 1
    n_min_class = np.sum(label_sums != 0)  # Minority class examples
    n_maj_class = np.round(n_min_class / target_ratio)  # Majority class examples

    min_idx = np.where(label_sums > 0)[0]
    maj_idx = np.random.choice(  # Select examples so that target ratio is achieved
        np.where(label_sums == 0)[0], size=int(n_maj_class), replace=False
    )

    return np.concatenate((min_idx, maj_idx))


def group_random_undersampler(
    labels: Labels, target_ratio: float, groups: Series
) -> npt.NDArray:
    """
    Randomly select labels to achieve the target ratio between minority and
    majority classes in each group.

    Parameters
    ----------
    labels : Labels
        Binary prediction labels of shape (n_samples, n_classes).
    target_ratio : float
        Ratio of minority over majority classes to achieve.
    groups : Series
        List of groups in which labels belong of shape (n_samples,).

    Returns
    -------
    sample_idx : npt.NDArray
        Index list of examples to achieve target ratio of shape (n_samples,).

    Raises
    ------
    TypeError
        If labels is not npt.NDArray.
    ValueError
        If labels and group do not have the same dimension.
        If target_ratio is less than 0.0.

    """
    array_check(labels)
    array_dim_check(labels, groups, dim=0)

    if target_ratio <= 0:
        raise ValueError(f"Target ratio should be positive, but got {target_ratio}")

    sample_idx = np.array([], dtype=int)  # Empty array for the majority class index
    n_groups = np.unique(groups)

    for group in n_groups:
        group_idx = np.where(groups == group)[0]
        group_data = labels[group_idx]
        sample_idx = np.concatenate(
            (sample_idx, group_idx[random_undersampler(group_data, target_ratio)])
        )

    return sample_idx


def random_oversampler(labels: Labels, target_ratio: float) -> npt.NDArray:
    """
    Randomly select labels to achieve the target ratio between minority and
    majority classes by oversampling minority class.

    Parameters
    ----------
    labels : Labels
        Binary prediction labels of shape (n_samples, n_classes).
    target_ratio : float
        Ratio of minority over majority classes to achieve.

    Returns
    -------
    sample_idx : npt.NDArray
        Index list of examples to achieve target ratio of shape (n_samples,).

    Raises
    ------
    TypeError
        If labels is not npt.NDArray.
    ValueError
        If target_ratio is less than 0.0.

    """
    array_check(labels)  # Check that labels is array-like
    if target_ratio <= 0:
        raise ValueError(f"Target ratio should be positive, but got {target_ratio}")

    label_sums = np.sum(labels, axis=1)  # Sum to find example with at least one 1
    n_min_class = np.sum(label_sums != 0)  # Minority class examples
    n_maj_class = len(labels) - n_min_class  # Majority class examples

    if n_min_class == 0:
        raise ValueError("No minority examples found")

    # Indices of minority and majority examples
    maj_idx = np.where(label_sums == 0)[0]
    min_idx = np.random.choice(  # Select examples so that target ratio is achieved
        np.where(label_sums > 0)[0], size=int(n_maj_class * target_ratio), replace=True
    )

    return np.concatenate((min_idx, maj_idx))


def group_random_oversampler(
    labels: Labels, target_ratio: float, groups: Series
) -> npt.NDArray:
    """
    Randomly select labels to achieve the target ratio between minority and
    majority classes in each group.

    Parameters
    ----------
    labels : Labels
        Binary prediction labels of shape (n_samples, n_classes).
    target_ratio : float
        Ratio of minority over majority classes to achieve.
    groups : Series
        List of groups in which labels belong of shape (n_samples,).

    Returns
    -------
    sample_idx : npt.NDArray
        Index list of examples to achieve target ratio of shape (n_samples,).

    Raises
    ------
    TypeError
        If labels is not npt.NDArray.
    ValueError
        If labels and group do not have the same dimension.
        If target_ratio is less than 0.0.

    """
    array_check(labels)
    array_dim_check(labels, groups, dim=0)

    if target_ratio <= 0:
        raise ValueError(f"Target ratio should be positive, but got {target_ratio}")

    sample_idx = np.array([], dtype=int)  # Empty array for the majority class index
    n_groups = np.unique(groups)

    for group in n_groups:
        group_idx = np.where(groups == group)[0]
        group_data = labels[group_idx]
        sample_idx = np.concatenate(
            (sample_idx, group_idx[random_oversampler(group_data, target_ratio)])
        )

    return sample_idx


def mean_dist_sampler(
    data: PData, labels: Labels, target_ratio: float, hard_percent=0.5
) -> npt.NDArray:
    """
    Computes the mean data sample of the majority class and uses the
    distance to it to select examples.

    The examples are sorted based on their distance to the mean.
    The hardest examples are the ones that have the greatest distance to
    the mean and the easiest are the ones closest to the mean.

    Parameters
    ----------
    data : PData
        Data to sample of shape (n_samples, n_features).
    labels : Labels
        Binary prediction labels of shape (n_samples, n_classes).
    target_ratio : float
        Ratio of minority over majority classes to achieve.
    hard_percent : float, default: 0.5
        Percentage of examples that are considered hard, between 0 and 1.
        If hard_percent is 0.5, half of the examples are chosen from
        the end of the sorted list and the other half from the beginning.

    Returns
    -------
    sample_idx : npt.NDArray
        Index list of examples to achieve target ratio of shape (n_samples,).

    Raises
    ------
    TypeError
        If labels is not npt.NDArray.
    ValueError
        If hard_percent is not between 0 and 1.
        If target_ratio is less than 0.0.

    """
    array_check(labels)
    if hard_percent > 1 or hard_percent < 0:
        raise ValueError(
            f"hard_percent should be between 0 and 1, but got {hard_percent}"
        )
    if target_ratio <= 0:
        raise ValueError(f"Target ratio should be positive, but got {target_ratio}")

    label_sums = np.sum(labels, axis=1)  # Sum to find example with at least one 1
    n_min_class = np.sum(label_sums != 0)  # Minority class examples
    n_maj_class = np.round(n_min_class / target_ratio)  # Majority class examples

    if isinstance(data, DataFrame):
        maj_class_data = data.iloc[label_sums == 0]
    else:
        maj_class_data = data[np.where(label_sums == 0)[0]]
    mean_maj_class = np.mean(maj_class_data, axis=0)

    # Get the distance to the mean
    dist = np.linalg.norm(mean_maj_class - maj_class_data, axis=1)
    sorted_dist_idx = np.argsort(dist)

    hard_samples_idx = sorted_dist_idx[-round(n_maj_class * hard_percent) :]
    easy_samples_idx = sorted_dist_idx[: round(n_maj_class * (1 - hard_percent))]

    return np.concatenate((easy_samples_idx, hard_samples_idx))


def group_mean_dist_sampler(
    data: PData,
    labels: Labels,
    target_ratio: float,
    groups: Series,
    hard_percent: float = 0.5,
) -> npt.NDArray:
    """
    Computes the mean data sample of the majority class in each group and
    uses the distance to it to select examples.

    The examples are sorted based on their distance to the mean.
    The hardest examples are the ones that have the greatest distance to
    the mean and the easiest are the ones closest to the mean.

    Parameters
    ----------
    data : PData
        Data to sample of shape (n_samples, n_features).
    labels : Labels
        Binary prediction labels of shape (n_samples, n_classes).
    target_ratio : float
        Ratio of minority over majority classes to achieve.
    groups : Series
        List of groups in which labels belong of shape (n_samples,).
    hard_percent : float, default: 0.5
        Percentage of examples that are considered hard, between 0 and 1.
        If hard_percent is 0.5, half of the examples are chosen from
        the end of the sorted list and the other half from the beginning.

    Returns
    -------
    sample_idx : npt.NDArray
        Index list of examples to achieve target ratio of shape (n_samples,).

    Raises
    ------
    TypeError
        If labels is not npt.NDArray.
    ValueError
        If labels and group do not have the same dimension.

    """
    array_check(labels)
    array_dim_check(labels, groups, dim=0)
    X = deepcopy(data)  # Create copy of data to not mess with actual data

    sample_idx = np.array([], dtype=int)  # Empty array for the majority class index
    n_groups = np.unique(groups)

    for group in n_groups:
        group_idx = np.where(groups == group)[0]
        group_data = get_data_from_idx(X, group_idx)
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


def smote(
    data: PData, labels: Labels, target_ratio: float, k_neighbors: int
) -> tuple[PData, Labels]:
    """
    Oversample minority class using Synthetic Minority Over-Sampling Technique
    (SMOTE).

    Parameters
    ----------
    data : PData
        Data to sample of shape (n_samples, n_features).
    labels : Labels
        Binary prediction labels of shape (n_samples, n_classes).
    target_ratio : float
        Ratio of minority over majority classes to achieve.
    k_neighbors : int
        Number of neighbors to use for SMOTE knn.

    Returns
    -------
    X_gen : PData
        Generated data.
    multilabels_gen : Labels
        Generated labels.

    Raises
    ------
    TypeError
        If labels is not npt.NDArray.
    ValueError
        If target_ratio is less than 0.0.

    """
    array_check(labels)
    X = deepcopy(data)

    if target_ratio <= 0:
        raise ValueError(f"Target ratio should be positive, but got {target_ratio}")

    label_sums = np.sum(labels, axis=1)  # Sum to find example with at least one 1
    n_maj_class = np.sum(label_sums == 0)  # Majority class examples
    n_min_class = np.round(n_maj_class * target_ratio) - np.sum(
        label_sums > 0
    )  # Minority class examples

    # Convert labels into unique classes
    unique_multilabels, class_labels = np.unique(labels, axis=0, return_inverse=True)

    sm = SMOTE(k_neighbors=k_neighbors)
    X_gen, y_gen, *_ = sm.fit_resample(X, class_labels)

    min_idx = np.random.choice(  # Select examples so that target ratio is achieved
        np.arange(len(labels), len(y_gen)), size=int(n_min_class), replace=True
    )

    return get_data_from_idx(X_gen, min_idx), unique_multilabels[y_gen[min_idx]]


def group_smote(
    data: PData,
    labels: Labels,
    target_ratio: float,
    groups: Series,
    k_neighbors: int,
) -> tuple[PData, Labels, Series]:
    """
    Oversample minority class using Synthetic Minority Over-Sampling Technique
    (SMOTE) in each group.

    Parameters
    ----------
    data : PData
        Data to sample of shape (n_samples, n_features).
    labels : Labels
        Binary prediction labels of shape (n_samples, n_classes).
    target_ratio : float
        Ratio of minority over majority classes to achieve.
    groups : Series
        List of groups in which labels belong of shape (n_samples,).
    k_neighbors : int
        Number of neighbors to use for SMOTE knn.

    Returns
    -------
    X_gen : PData
        Generated data.
    multilabels_gen : Labels
        Generated labels.
    groups_gen : Series
        Generated groups.

    Raises
    ------
    TypeError
        If labels is not npt.NDArray
    ValueError
        If labels and group do not have the same dimension.

    """
    array_check(labels)
    array_dim_check(labels, groups, dim=0)
    X = deepcopy(data)  # Create copy of data to not mess with actual data
    y = deepcopy(labels)
    grps = deepcopy(groups)

    n_groups = np.unique(groups)

    for group in n_groups:
        group_idx = np.where(groups == group)[0]
        group_data = get_data_from_idx(data, group_idx)
        group_labels = labels[group_idx]

        # Generate new data for groups
        X_gen, y_gen = smote(group_data, group_labels, target_ratio, k_neighbors)
        group_gen = group * np.ones(y_gen.shape[0])
        if isinstance(X, DataFrame):
            X = concat((X, cast(DataFrame, X_gen)))
        else:
            X = np.concatenate((X, X_gen))
        y = np.concatenate((y, y_gen))
        grps = concat((grps, Series(group_gen.squeeze(), name=grps.name)))

    return X, y, grps
