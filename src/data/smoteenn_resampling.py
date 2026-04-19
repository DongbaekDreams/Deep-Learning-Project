'''
References:

1. G. Batista, R. C. Prati, M. C. Monard. “A study of the behavior of several methods for balancing machine learning training data,” ACM Sigkdd Explorations Newsletter 6 (1), 20-29, 2004.
'''

from __future__ import annotations


import numpy as np
from imblearn.combine import SMOTEENN


def apply_smoteenn_to_reduced_features(
    X_train_reduced: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
    sampling_strategy: str = "not majority",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTEENN to reduced training features only.

    This should be used after feature reduction (autoencoder or pooling)
    and before classifier training.
    """
    X_train_reduced = np.asarray(X_train_reduced, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int64).ravel()

    sampler = SMOTEENN(
        random_state=random_state,
        sampling_strategy=sampling_strategy,
    )

    X_resampled, y_resampled = sampler.fit_resample(X_train_reduced, y_train)
    return X_resampled.astype(np.float32), y_resampled.astype(np.int64)