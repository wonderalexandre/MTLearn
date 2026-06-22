"""Small train/test split helper shared by mtlearn datasets.

The project avoids importing scikit-learn merely to split dataset indices.
This module provides the subset of ``train_test_split`` behavior needed by the
dataset helpers while keeping imports lightweight for notebook and package
startup paths.
"""

from __future__ import annotations

import numpy as np


def _split_indices(
    num_samples: int,
    test_size: float | int = 0.25,
    *,
    shuffle: bool = True,
    random_state: int | np.random.RandomState | None = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Split dataset indices without requiring scikit-learn at import time.

    Args:
        num_samples: Total number of samples in the dataset.
        test_size: Fraction or absolute number of samples reserved for testing.
        shuffle: Whether to shuffle indices before splitting.
        random_state: Seed or ``RandomState`` used when ``shuffle`` is true.

    Returns:
        ``(train_idx, test_idx)`` arrays suitable for ``torch.utils.data.Subset``.
    """

    if num_samples <= 0:
        raise ValueError("Cannot split an empty dataset")

    if isinstance(test_size, float):
        if not 0.0 < test_size < 1.0:
            raise ValueError("test_size as a float must be between 0 and 1")
        num_test = int(np.ceil(num_samples * test_size))
    elif isinstance(test_size, int):
        if not 0 < test_size < num_samples:
            raise ValueError("test_size as an int must be between 1 and len(dataset) - 1")
        num_test = test_size
    else:
        raise TypeError("test_size must be a float or int")

    num_train = num_samples - num_test
    if num_train <= 0:
        raise ValueError("test_size leaves no samples for training")

    if shuffle:
        rng = (
            random_state
            if isinstance(random_state, np.random.RandomState)
            else np.random.RandomState(random_state)
        )
        permutation = rng.permutation(num_samples)
        test_idx = permutation[:num_test]
        train_idx = permutation[num_test : num_test + num_train]
    else:
        indices = np.arange(num_samples)
        train_idx = indices[:num_train]
        test_idx = indices[num_train : num_train + num_test]

    return train_idx, test_idx
