'''
References:
1. Cui, Y., Jia, M., Lin, T.-Y., Song, Y., & Belongie, S. (2019). Class-balanced loss based on effective number of samples. 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 9268–9277. https://doi.org/10.1109/cvpr.2019.00949
2. Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2018). Focal loss for dense object detection. arXiv. https://arxiv.org/abs/1708.02002
'''

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassBalancedFocalLoss(nn.Module):
    def __init__(
        self,
        alpha: torch.Tensor,
        gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.register_buffer("alpha", alpha.float())
        self.gamma = float(gamma)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        alpha_t = self.alpha[targets]
        loss = alpha_t * (1.0 - pt).pow(self.gamma) * ce
        return loss.mean()


def make_cb_focal_loss(
    y_train: np.ndarray,
    num_classes: int,
    beta: float = 0.9999,
    gamma: float = 2.0,
    svt_class_index: int | None = None,
    svt_multiplier: float = 1.0,
    device: str | torch.device = "cpu",
) -> nn.Module:
    """
    Build a class-balanced focal loss using effective-number weights,
    with an optional extra boost for the SVT class.
    """
    y_train = np.asarray(y_train, dtype=np.int64).ravel()

    # Validation
    assert len(y_train) > 0, "y_train is empty"
    assert y_train.min() >= 0 and y_train.max() < num_classes, \
        f"Labels must be in [0, {num_classes-1}], got [{y_train.min()}, {y_train.max()}]"
    assert 0.0 < beta < 1.0, f"beta must be in (0, 1), got {beta}"
    assert gamma >= 0.0, f"gamma must be >= 0, got {gamma}"

    if svt_class_index is not None:
        assert 0 <= svt_class_index < num_classes, \
            f"svt_class_index {svt_class_index} out of bounds [0, {num_classes-1}]"

    # Compute class counts
    counts = np.bincount(y_train, minlength=num_classes).astype(np.float64)

    # Warn about zero-count classes
    zero_classes = np.where(counts == 0)[0]
    if len(zero_classes) > 0:
        print(f"Warning: Classes {zero_classes} have zero samples")

    # Effective-number weights
    effective_num = 1.0 - np.power(beta, counts)
    alpha = (1.0 - beta) / np.maximum(effective_num, 1e-12)
    alpha = alpha / alpha.mean()

    # Optional extra SVT boost
    if svt_class_index is not None:
        alpha[svt_class_index] *= float(svt_multiplier)

    alpha_tensor = torch.tensor(alpha, dtype=torch.float32, device=device)

    return ClassBalancedFocalLoss(alpha=alpha_tensor, gamma=gamma)