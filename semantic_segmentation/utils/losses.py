"""Loss functions for semantic segmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class CrossEntropyLoss(nn.Module):
    """Standard pixel-wise cross-entropy loss.

    Args:
        ignore_index (int): Class index to ignore (e.g. void label).
        class_weight (Tensor | None): Per-class loss weights.
        reduction (str): ``'mean'`` or ``'sum'``.
    """

    def __init__(self, ignore_index: int = 255,
                 class_weight: Optional[torch.Tensor] = None,
                 reduction: str = "mean") -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.class_weight = class_weight
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            pred, target.long(),
            weight=self.class_weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
        )


class OHEMCrossEntropyLoss(nn.Module):
    """Online Hard Example Mining cross-entropy loss.

    Keeps only the ``thresh`` fraction of pixels with highest loss
    (at least ``min_kept`` pixels).

    Args:
        thresh (float): Fraction of pixels to keep.
        min_kept (int): Minimum number of hard pixels to keep.
        ignore_index (int): Label to ignore.
    """

    def __init__(self, thresh: float = 0.7, min_kept: int = 10000,
                 ignore_index: int = 255) -> None:
        super().__init__()
        self.thresh = thresh
        self.min_kept = min_kept
        self.ignore_index = ignore_index

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        B, C, H, W = pred.shape
        prob = F.softmax(pred.detach(), dim=1)

        # Gather probability of the ground-truth class
        target_flat = target.view(-1).long()
        prob_flat    = prob.permute(0, 2, 3, 1).reshape(-1, C)

        valid_mask = target_flat != self.ignore_index
        valid_targets = target_flat[valid_mask]
        valid_probs   = prob_flat[valid_mask]

        gt_prob = valid_probs[torch.arange(valid_targets.numel()), valid_targets]

        # Sort by ascending probability → hardest examples first
        sorted_prob, _ = gt_prob.sort()
        num_valid = sorted_prob.numel()
        # Select the threshold index: prefer min_kept but cap at the top-thresh fraction
        top_thresh_idx = max(1, int(num_valid * (1 - self.thresh)))
        keep_idx = min(self.min_kept, top_thresh_idx) - 1
        threshold_prob = sorted_prob[keep_idx]

        hard_mask = (gt_prob <= max(threshold_prob, self.thresh))
        # Build pixel-level hard mask (only valid pixels are considered)
        full_hard_mask = torch.zeros_like(target_flat, dtype=torch.bool)
        valid_indices = valid_mask.nonzero(as_tuple=False).squeeze(1)
        hard_valid_indices = valid_indices[hard_mask]
        full_hard_mask[hard_valid_indices] = True

        full_hard_mask = full_hard_mask.view(B, H, W)
        target_ohem = target.clone()
        target_ohem[~full_hard_mask] = self.ignore_index

        return F.cross_entropy(pred, target_ohem.long(),
                               ignore_index=self.ignore_index)


class DiceLoss(nn.Module):
    """Soft Dice loss for semantic segmentation.

    Args:
        smooth (float): Smoothing factor to avoid division by zero.
        ignore_index (int): Class index to ignore.
    """

    def __init__(self, smooth: float = 1.0, ignore_index: int = 255) -> None:
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = pred.shape[1]
        prob = F.softmax(pred, dim=1)

        # One-hot encode target, ignoring void pixels
        valid = (target != self.ignore_index)
        target_valid = target.clone()
        target_valid[~valid] = 0

        target_oh = F.one_hot(target_valid.long(), num_classes)  # B,H,W,C
        target_oh = target_oh.permute(0, 3, 1, 2).float()        # B,C,H,W
        # Zero out void pixels in target
        target_oh = target_oh * valid.unsqueeze(1).float()
        prob      = prob      * valid.unsqueeze(1).float()

        intersection = (prob * target_oh).sum(dim=(0, 2, 3))
        cardinality  = (prob + target_oh).sum(dim=(0, 2, 3))
        dice_per_cls = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice_per_cls.mean()


class SegmentationLoss(nn.Module):
    """Combined loss = CE + aux_CE (optional) + Dice (optional).

    Args:
        num_classes (int): Number of classes.
        ignore_index (int): Void label.
        use_dice (bool): Add Dice loss component.
        use_ohem (bool): Use OHEM instead of standard CE.
        aux_weight (float): Weight for auxiliary head loss.
        dice_weight (float): Weight for Dice loss component.
    """

    def __init__(self, num_classes: int = 19, ignore_index: int = 255,
                 use_dice: bool = False, use_ohem: bool = False,
                 aux_weight: float = 0.4, dice_weight: float = 0.5) -> None:
        super().__init__()
        self.aux_weight  = aux_weight
        self.dice_weight = dice_weight
        self.use_dice    = use_dice

        if use_ohem:
            self.main_ce = OHEMCrossEntropyLoss(ignore_index=ignore_index)
        else:
            self.main_ce = CrossEntropyLoss(ignore_index=ignore_index)

        self.aux_ce = CrossEntropyLoss(ignore_index=ignore_index)
        self.dice   = DiceLoss(ignore_index=ignore_index) if use_dice else None

    def forward(self, outputs: dict, target: torch.Tensor) -> torch.Tensor:
        loss = self.main_ce(outputs["out"], target)
        if self.use_dice and self.dice is not None:
            loss = loss + self.dice_weight * self.dice(outputs["out"], target)
        if "aux" in outputs:
            loss = loss + self.aux_weight * self.aux_ce(outputs["aux"], target)
        return loss


__all__ = [
    "CrossEntropyLoss",
    "OHEMCrossEntropyLoss",
    "DiceLoss",
    "SegmentationLoss",
]
