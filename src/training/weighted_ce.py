from __future__ import annotations

import numpy as np
import torch 
import torch.nn as nn

def make_weighted_ce_loss(
    y_train: np.ndarray,
    num_classes: int,
    svt_class_index: int | None = None,
    svt_multiplier: float = 1.25,
    device: str | torch.device = "cpu",
) -> nn.Module:
    """
    Build a CrossEntropyLoss with inverse-frequency class weights and
    an optional mild extra boost for the SVT class.
    """
    y_train = np.asarray(y_train, dtype=np.int64).ravel()
    
    # Validation
    assert len(y_train) > 0, "y_train is empty"
    assert y_train.min() >= 0 and y_train.max() < num_classes, \
        f"Labels must be in [0, {num_classes-1}], got [{y_train.min()}, {y_train.max()}]"
    
    if svt_class_index is not None:
        assert 0 <= svt_class_index < num_classes, \
            f"svt_class_index {svt_class_index} out of bounds [0, {num_classes-1}]"
    
    # Compute weights
    counts = np.bincount(y_train, minlength=num_classes).astype(np.float64)
    
    # Warn about zero-count classes
    zero_classes = np.where(counts == 0)[0]
    if len(zero_classes) > 0:
        print(f"Warning: Classes {zero_classes} have zero samples")
    
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    
    if svt_class_index is not None:
        weights[svt_class_index] *= float(svt_multiplier)
    
    weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    
    return nn.CrossEntropyLoss(weight=weight_tensor)