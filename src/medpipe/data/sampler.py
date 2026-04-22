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

from typing import TYPE_CHECKING, Any, cast

import numpy as np
from imblearn.over_sampling import SMOTE
from pandas import DataFrame, concat

from medpipe._types import Labels, PredData
from medpipe.utils.exceptions import array_check, array_dim_check

from .utils import get_data_from_idx

if TYPE_CHECKING:
    import numpy.typing as npt


def data_sampler(
    data: PredData,
    labels: Labels,
    reduction_factor: float = 0.25,
    sampler_fn: str = "random_undersampler",
    groups: npt.NDArray = np.array([]),
    **kwargs: Any,
) -> tuple[PredData, Labels, npt.NDArray]:
    """
    Samples the data and labels to adjust the class imbalance.

    The majority class is assumed to have a False or 0 label.
    The new set will have an imbalance equal to:
        IR - IR * reduction_factor, where IR is the current imbalance ratio.
        or a IR = 1, if reduction factor is 1.

    If the reduction factor is too small, the algorithm defaults to
    obtain a balanced dataset.

    Parameters
    ----------
    data : PredData
        Data to sample of shape (n_samples, n_features).
    labels : Labels
        Binary prediction labels of shape (n_samples, n_classes).
    reduction_factor : float, default: 0.25
        Reduction factor to apply to the current imbalance ratio
    sampler_fn : str, default: "random_undersampler"
        Sampler function to use to sample the data.
    groups : npt.NDArray, default: np.array([])
        List containing groups for the group_sampler function.
    **kwargs : Any
        Extra arguments for the sampler functions.

    Returns
    -------
    X : PredData
        Sampled data.
    y : Labels
        Sampled labels.
    groups : npt.NDArray
        Groups of the examples. Empty array if not needed.

    Raises
    ------
    TypeError
        If labels is not npt.NDArray.
    ValueError
        If reduction_factor is less than 0.0 or greater than 1.0

    """
    if reduction_factor < 0:
        raise ValueError(
            f"Reduction factor should be positive, but got {reduction_factor}"
        )
    if reduction_factor > 1.0:
        raise ValueError(
            f"Reduction factor should be less than 1, but got {reduction_factor}"
        )
    array_check(labels)

    # Calculate target_ratio
    n_samples = len(labels)
    n_pos = np.sum(labels)

    if reduction_factor == 0 or n_pos == 0:
        return data, labels, groups  # Exit early
    elif reduction_factor == 1:
        target_ratio = 1
    else:
        imbalance_ratio = (n_samples - n_pos) / n_pos
        target_ratio = imbalance_ratio - imbalance_ratio * reduction_factor

    # Dispatch logic
    if sampler_fn == "smote":
        X_gen, y_gen = smote(data, labels, target_ratio, kwargs.get("k_neighbors", 5))
        if isinstance(data, DataFrame):
            return (
                concat((data, cast(DataFrame, X_gen))),
                np.concatenate((labels, y_gen)),
                np.array([]),
            )
        return (
            np.concatenate((data, X_gen)),
            np.concatenate((labels, y_gen)),
            np.array([]),
        )

    if sampler_fn == "group_smote":
        return group_smote(
            data, labels, target_ratio, groups, kwargs.get("k_neighbors", 5)
        )

    # Index-based samplers
    sampler_map = {
        "random_undersampler": lambda: random_undersampler(labels, target_ratio),
        "group_random_undersampler": lambda: group_random_undersampler(
            labels, target_ratio, groups
        ),
        "random_oversampler": lambda: random_oversampler(labels, target_ratio),
        "group_random_oversampler": lambda: group_random_oversampler(
            labels, target_ratio, groups
        ),
        "mean_dist_sampler": lambda: mean_dist_sampler(
            data, labels, target_ratio, kwargs.get("hard_percent", 0.5)
        ),
        "group_mean_dist_sampler": lambda: group_mean_dist_sampler(
            data, labels, target_ratio, groups, kwargs.get("hard_percent", 0.5)
        ),
    }

    if sampler_fn not in sampler_map:
        raise ValueError(f"{sampler_fn} invalid sampler function")

    sample_idx = sampler_map[sampler_fn]()

    X = get_data_from_idx(data, sample_idx)
    y = labels[sample_idx]

    res_groups = np.array([])
    if groups.size > 0:
        res_groups = cast(np.ndarray, get_data_from_idx(groups, sample_idx))

    return X, y, res_groups


def random_undersampler(labels: Labels, target_ratio: float) -> npt.NDArray:
    """
    Randomly select labels to achieve the target ratio between minority and
    majority classes by undersampling majority class.

    Parameters
    ----------
    labels : Labels
        Binary prediction labels of shape (n_samples, n_classes).
    target_ratio : float
        Ratio of majority over minority classes to achieve.

    Returns
    -------
    sample_idx : npt.NDArray
        Index list of examples to achieve target ratio of shape (n_samples,).

    Raises
    ------
    TypeError
        If labels is not npt.NDArray.
    ValueError
        If target_ratio is less than 1.

    """
    array_check(labels)

    if target_ratio < 1:
        raise ValueError(
            f"Target ratio should be greater than 1, but got {target_ratio}"
        )

    # Any row with a 1 is minority; rows with all 0s are majority
    is_min = np.any(labels != 0, axis=1)
    is_maj = ~is_min

    min_idx = np.where(is_min)[0]
    maj_potential_idx = np.where(is_maj)[0]

    n_min_class = len(min_idx)
    n_maj_available = len(maj_potential_idx)

    if n_min_class == 0:
        # Returning an empty array or original indices is safer than crashing
        return maj_potential_idx.astype(int)

    # Calculate target majority count
    n_maj_target = int(np.round(n_min_class * target_ratio))
    n_to_sample = min(n_maj_target, n_maj_available)

    # Random selection
    maj_idx = np.random.choice(maj_potential_idx, size=n_to_sample, replace=False)

    # Return as integer array for consistent downstream indexing
    return np.concatenate((min_idx, maj_idx)).astype(int)


def group_random_undersampler(
    labels: Labels, target_ratio: float, groups: npt.NDArray
) -> npt.NDArray:
    """
    Randomly select labels to achieve the target ratio between minority and
    majority classes in each group.

    Parameters
    ----------
    labels : Labels
        Binary prediction labels of shape (n_samples, n_classes).
    target_ratio : float
        Ratio of majority over minority classes to achieve.
    groups : npt.NDArray
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
        If target_ratio is less than 1.

    """
    array_check(labels)
    array_dim_check(labels, groups, dim=0)

    if target_ratio < 1:
        raise ValueError(
            f"Target ratio should be greater than 1, but got {target_ratio}"
        )

    all_sampled_indices = []  # List for sampled indices

    # Get unique groups
    n_groups = np.unique(groups)

    for group in n_groups:
        # Get absolute indices for this specific group
        group_indices = np.where(groups == group)[0]
        relative_idx = random_undersampler(labels[group_indices], target_ratio)

        # Update list
        all_sampled_indices.append(group_indices[relative_idx])

    # Concatenate all indices
    if not all_sampled_indices:
        return np.array([], dtype=int)

    return np.concatenate(all_sampled_indices).astype(int)


def random_oversampler(labels: Labels, target_ratio: float) -> npt.NDArray:
    """
    Randomly select labels to achieve the target ratio between minority and
    majority classes by oversampling minority class.

    Parameters
    ----------
    labels : Labels
        Binary prediction labels of shape (n_samples, n_classes).
    target_ratio : float
        Ratio of majority over minority classes to achieve.

    Returns
    -------
    sample_idx : npt.NDArray
        Index list of examples to achieve target ratio of shape (n_samples,).

    Raises
    ------
    TypeError
        If labels is not npt.NDArray.
    ValueError
        If target_ratio is less than 1.
        If there is no minority class.

    """
    array_check(labels)

    if target_ratio < 1:
        raise ValueError(
            f"Target ratio should be greater than 1, but got {target_ratio}"
        )

    # Generate class masks and indices once
    is_min = np.any(labels != 0, axis=1)
    is_maj = ~is_min

    min_indices = np.where(is_min)[0]
    maj_indices = np.where(is_maj)[0]

    n_min_class = len(min_indices)
    n_maj_class = len(maj_indices)

    if n_min_class == 0:
        raise ValueError("No minority examples found; cannot oversample.")

    # Calculate target size
    target_n_min = int(np.round(n_maj_class / target_ratio))

    # Perform oversampling
    oversampled_min_idx = np.random.choice(min_indices, size=target_n_min, replace=True)

    # Concatenate and ensure integer type
    return np.concatenate((oversampled_min_idx, maj_indices)).astype(int)


def group_random_oversampler(
    labels: Labels, target_ratio: float, groups: npt.NDArray
) -> npt.NDArray:
    """
    Randomly select labels to achieve the target ratio between minority and
    majority classes in each group.

    Parameters
    ----------
    labels : Labels
        Binary prediction labels of shape (n_samples, n_classes).
    target_ratio : float
        Ratio of majority over minority classes to achieve.
    groups : npt.NDArray
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
        If target_ratio is less than 1.

    """
    array_check(labels)
    array_dim_check(labels, groups, dim=0)

    if target_ratio < 1:
        raise ValueError(
            f"Target ratio should be greater than 1, but got {target_ratio}"
        )

    all_sampled_indices = []  # List to contain sampled indices
    unique_groups = np.unique(groups)

    for group in unique_groups:
        # Get absolute indices for this group
        group_indices = np.where(groups == group)[0]
        group_labels = labels[group_indices]

        # If not minority class keep the majority samples.
        has_minority = np.any(group_labels != 0)

        if has_minority:
            # Get relative indices from our optimized random_oversampler
            relative_idx = random_oversampler(group_labels, target_ratio)
            # Map to absolute indices and store
            all_sampled_indices.append(group_indices[relative_idx])
        else:
            all_sampled_indices.append(group_indices)

    # Concatenate sampled indices
    if not all_sampled_indices:
        return np.array([], dtype=int)

    return np.concatenate(all_sampled_indices).astype(int)


def mean_dist_sampler(
    data: PredData, labels: Labels, target_ratio: float, hard_percent=0.5
) -> npt.NDArray:
    """
    Computes the mean data sample of the majority class and uses the
    distance to it to select examples.

    The examples are sorted based on their distance to the mean.
    The hardest examples are the ones that have the greatest distance to
    the mean and the easiest are the ones closest to the mean.

    Parameters
    ----------
    data : PredData
        Data to sample of shape (n_samples, n_features).
    labels : Labels
        Binary prediction labels of shape (n_samples, n_classes).
    target_ratio : float
        Ratio of majority over minority classes to achieve.
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
        If target_ratio is less than 1.

    """
    array_check(labels)

    if not (0 <= hard_percent <= 1):
        raise ValueError(f"hard_percent must be between 0 and 1, got {hard_percent}")
    if target_ratio < 1:
        raise ValueError(
            f"Target ratio should be greater than 1, but got {target_ratio}"
        )

    # Identify minority and majority masks
    is_min = np.any(labels != 0, axis=1)
    is_maj = ~is_min

    min_indices = np.where(is_min)[0]
    maj_indices = np.where(is_maj)[0]

    n_min = len(min_indices)
    n_maj_available = len(maj_indices)

    if n_min == 0:
        return np.arange(
            len(labels)
        )  # Return everything if no minority to balance against

    # Extract majority data
    maj_data = get_data_from_idx(data, is_maj)

    # Distance calculation
    mean_maj = np.mean(maj_data, axis=0)
    # Vectorized subtraction and norm
    dist = np.linalg.norm(maj_data - mean_maj, axis=1)
    sorted_rel_idx = np.argsort(dist)

    # Determine target counts
    n_maj_target = int(np.round(n_min * target_ratio))
    n_to_sample = min(n_maj_target, n_maj_available)

    n_hard = int(np.round(n_to_sample * hard_percent))
    n_easy = n_to_sample - n_hard

    # Map relative indices back to absolute indices
    selected_maj_indices = np.concatenate(
        (
            maj_indices[sorted_rel_idx[:n_easy]],
            maj_indices[sorted_rel_idx[-n_hard:]] if n_hard > 0 else [],
        )
    )

    # Return minority + selected majority
    return np.concatenate((min_indices, selected_maj_indices)).astype(int)


def group_mean_dist_sampler(
    data: PredData,
    labels: Labels,
    target_ratio: float,
    groups: npt.NDArray,
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
    data : PredData
        Data to sample of shape (n_samples, n_features).
    labels : Labels
        Binary prediction labels of shape (n_samples, n_classes).
    target_ratio : float
        Ratio of majority over minority classes to achieve.
    groups : npt.NDArray
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
        If target ratio is less than 1.

    """
    array_check(labels)
    array_dim_check(labels, groups, dim=0)

    if target_ratio < 1:
        raise ValueError(
            f"Target ratio should be greater than 1, but got {target_ratio}"
        )
    all_sampled_indices = []  # List for sampled indices
    n_groups = np.unique(groups)

    for group in n_groups:
        # Get absolute indices for the group
        group_indices = np.where(groups == group)[0]
        group_labels = labels[group_indices]

        # If no majority samples exist, we can't calculate a mean_dist.
        # If no minority samples exist, we don't need to balance.
        has_maj = np.any(np.all(group_labels == 0, axis=1))
        has_min = np.any(np.any(group_labels != 0, axis=1))

        if has_maj and has_min:
            group_data = get_data_from_idx(data, group_indices)

            # The inner sampler returns relative indices [0, 1, 2...]
            relative_idx = mean_dist_sampler(
                group_data, group_labels, target_ratio, hard_percent
            )
            # Map back to absolute and store
            all_sampled_indices.append(group_indices[relative_idx])
        else:
            # If we can't balance (missing a class), keep the group as-is
            all_sampled_indices.append(group_indices)

    # Final concatenation
    if not all_sampled_indices:
        return np.array([], dtype=int)

    return np.concatenate(all_sampled_indices).astype(int)


def smote(
    data: PredData, labels: Labels, target_ratio: float, k_neighbors: int
) -> tuple[PredData, Labels]:
    """
    Oversample minority class using Synthetic Minority Over-Sampling Technique
    (SMOTE).

    Parameters
    ----------
    data : PredData
        Data to sample of shape (n_samples, n_features).
    labels : Labels
        Binary prediction labels of shape (n_samples, n_classes).
    target_ratio : float
        Ratio of majority over minority classes to achieve.
    k_neighbors : int
        Number of neighbors to use for SMOTE knn.

    Returns
    -------
    X_gen : PredData
        Generated data.
    multilabels_gen : Labels
        Generated labels.

    Raises
    ------
    TypeError
        If labels is not npt.NDArray.
    ValueError
        It target_ratio is less than 1.

    """
    array_check(labels)

    if target_ratio < 1:
        raise ValueError(
            f"Target ratio should be greater than 1, but got {target_ratio}"
        )

    # Identify counts
    is_min = np.any(labels != 0, axis=1)
    n_maj = np.sum(~is_min)
    n_min_existing = np.sum(is_min)

    # Calculate how many NEW samples we need to generate
    n_target_total_min = int(np.round(n_maj / target_ratio))
    n_to_generate = n_target_total_min - n_min_existing

    if n_to_generate <= 0:
        # Return empty structures of the correct type if no generation needed
        return (data.iloc[:0] if isinstance(data, DataFrame) else data[:0]), labels[:0]

    # Map multi-label rows to unique IDs.
    unique_combinations, inverse_indices = np.unique(
        labels, axis=0, return_inverse=True
    )

    # We find the most frequent minority combination to use for generation.
    sm = SMOTE(k_neighbors=k_neighbors)
    try:
        X_resampled, y_resampled, *_ = sm.fit_resample(data, inverse_indices)
    except ValueError:
        # Fallback: if k_neighbors is too high for the group size, we skip generation
        return (data.iloc[:0] if isinstance(data, DataFrame) else data[:0]), labels[:0]

    # Generated samples are appended to the end.
    n_original = len(data)
    X_generated = get_data_from_idx(
        cast(PredData, X_resampled), np.arange(n_original, len(X_resampled))
    )
    y_gen_indices = y_resampled[n_original:]

    # Sub-sample the generated data to match the exact target_ratio
    if len(X_generated) > n_to_generate:
        sel = np.random.choice(len(X_generated), size=n_to_generate, replace=False)
        X_generated = get_data_from_idx(X_generated, sel)
        y_gen_indices = y_gen_indices[sel]

    return X_generated, unique_combinations[y_gen_indices]


def group_smote(
    data: PredData,
    labels: Labels,
    target_ratio: float,
    groups: npt.NDArray,
    k_neighbors: int,
) -> tuple[PredData, Labels, npt.NDArray]:
    """
    Oversample minority class using Synthetic Minority Over-Sampling Technique
    (SMOTE) in each group.

    Parameters
    ----------
    data : PredData
        Data to sample of shape (n_samples, n_features).
    labels : Labels
        Binary prediction labels of shape (n_samples, n_classes).
    target_ratio : float
        Ratio of majority over minority classes to achieve.
    groups : npt.NDArray
        List of groups in which labels belong of shape (n_samples,).
    k_neighbors : int
        Number of neighbors to use for SMOTE knn.

    Returns
    -------
    X_gen : PredData
        Generated data.
    multilabels_gen : Labels
        Generated labels.
    groups_gen : npt.NDArray
        Generated groups.

    Raises
    ------
    TypeError
        If labels is not npt.NDArray
    ValueError
        If labels and group do not have the same dimension.
        It target_ratio is less than 1.

    """
    array_check(labels)
    array_dim_check(labels, groups, dim=0)

    if target_ratio < 1:
        raise ValueError(
            f"Target ratio should be greater than 1, but got {target_ratio}"
        )
    unique_groups = np.unique(groups)

    # Start with original data to avoid repeated copying
    all_X = [data]
    all_y = [labels]
    all_g = [groups]

    for group in unique_groups:
        grp_idx = np.where(groups == group)[0]
        # Skip SMOTE if the group is too small to find neighbors
        if len(grp_idx) <= k_neighbors:
            continue

        grp_data = get_data_from_idx(data, grp_idx)
        grp_labels = labels[grp_idx]

        X_gen, y_gen = smote(grp_data, grp_labels, target_ratio, k_neighbors)

        if len(y_gen) > 0:
            all_X.append(X_gen)
            all_y.append(y_gen)
            all_g.append(np.full(len(y_gen), group))

    # Final concatenation
    if isinstance(data, DataFrame):
        final_X = concat(cast(list[DataFrame], all_X), axis=0)
    else:
        final_X = np.concatenate(all_X, axis=0)

    return final_X, np.concatenate(all_y, axis=0), np.concatenate(all_g, axis=0)
