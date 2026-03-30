"""
Loss functions for landslide semantic segmentation.

Includes:
  - CrossEntropyLoss (weighted / unweighted)
  - DiceLoss (soft multi-class)
  - FocalLoss (hard-example mining)
  - BoundaryLoss (for EGBR auxiliary output)
  - LandslideSegLoss (composite: CE + Dice + Focal + Boundary)
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────── individual losses ───────────────────────────────

class WeightedCELoss(nn.Module):
    """Pixel-wise cross-entropy with optional per-class weights."""

    def __init__(self, num_classes: int = 2,
                 weight: Optional[torch.Tensor] = None,
                 ignore_index: int = 255,
                 label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.register_buffer("weight", weight)
        self.ignore_index   = ignore_index
        self.label_smoothing = label_smoothing
        self.num_classes    = num_classes

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            pred, target.long(),
            weight=self.weight,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
        )


class DiceLoss(nn.Module):
    """Soft Dice loss averaged over valid classes."""

    def __init__(self, smooth: float = 1.0,
                 ignore_index: int = 255) -> None:
        super().__init__()
        self.smooth       = smooth
        self.ignore_index = ignore_index

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        num_classes = pred.shape[1]
        prob = pred.softmax(dim=1)

        valid = (target != self.ignore_index)
        t_clean = target.clone()
        t_clean[~valid] = 0

        t_oh = F.one_hot(t_clean.long(), num_classes).permute(0, 3, 1, 2).float()
        t_oh = t_oh * valid.unsqueeze(1).float()
        prob  = prob  * valid.unsqueeze(1).float()

        inter = (prob * t_oh).sum(dim=(0, 2, 3))
        denom = (prob + t_oh).sum(dim=(0, 2, 3))
        dice  = (2.0 * inter + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    """Focal loss for imbalanced landslide pixel distributions.

    Args:
        gamma: Focusing parameter (default 2.0).
        alpha: Scalar foreground weight (default 0.25).
    """

    def __init__(self, gamma: float = 2.0,
                 alpha: float = 0.25,
                 ignore_index: int = 255) -> None:
        super().__init__()
        self.gamma        = gamma
        self.alpha        = alpha
        self.ignore_index = ignore_index

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        log_p = F.log_softmax(pred, dim=1)
        p     = log_p.exp()

        # Gather probabilities of ground-truth classes
        B, C, H, W = pred.shape
        flat_target = target.clone().long()
        flat_target[flat_target == self.ignore_index] = 0

        valid  = (target != self.ignore_index).float()
        gt_log = log_p.gather(1, flat_target.unsqueeze(1)).squeeze(1)
        gt_p   = p.gather(1, flat_target.unsqueeze(1)).squeeze(1)

        focal_weight = (1 - gt_p) ** self.gamma
        loss = -focal_weight * gt_log * valid
        denom = valid.sum().clamp(min=1.0)
        return loss.sum() / denom


class BoundaryLoss(nn.Module):
    """Binary cross-entropy loss on the predicted boundary map.

    The target boundary map is derived from the segmentation mask by
    detecting edges with a 3×3 max-pool / min-pool approach.

    Args:
        ignore_index: Void label value in the mask.
    """

    def __init__(self, ignore_index: int = 255) -> None:
        super().__init__()
        self.ignore_index = ignore_index

    @staticmethod
    def _mask_to_boundary(mask: torch.Tensor,
                          dilation_ratio: float = 0.02) -> torch.Tensor:
        """Convert integer mask (B,H,W) → float boundary map (B,1,H,W)."""
        mask_f = mask.float().unsqueeze(1)
        H, W   = mask.shape[-2:]
        k = max(int(min(H, W) * dilation_ratio), 1)
        k = k if k % 2 == 1 else k + 1
        dilated  = F.max_pool2d(mask_f, k, stride=1, padding=k // 2)
        eroded   = -F.max_pool2d(-mask_f, k, stride=1, padding=k // 2)
        boundary = ((dilated != eroded) & (mask.unsqueeze(1) != 255)).float()
        return boundary

    def forward(self, boundary_logit: torch.Tensor,
                target_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            boundary_logit: (B, 1, H', W') raw logits from EGBR
            target_mask:    (B, H, W) integer segmentation labels
        """
        H, W = boundary_logit.shape[-2:]
        # Downscale mask to match boundary resolution
        tgt = F.interpolate(
            (target_mask != self.ignore_index).float().unsqueeze(1),
            (H, W), mode="nearest"
        ).squeeze(1).long()
        tgt_mask_down = F.interpolate(
            target_mask.float().unsqueeze(1),
            (H, W), mode="nearest"
        ).squeeze(1).long()

        boundary_gt = self._mask_to_boundary(tgt_mask_down)
        valid       = (tgt_mask_down.unsqueeze(1) != self.ignore_index).float()

        loss = F.binary_cross_entropy_with_logits(
            boundary_logit, boundary_gt,
            reduction="none"
        ) * valid
        return loss.sum() / valid.sum().clamp(min=1.0)


# ─────────────────────────── composite loss ──────────────────────────────────

class LandslideSegLoss(nn.Module):
    """Composite loss for LSFormer (and generic use).

    Total loss
    ----------
    L = w_ce   · CE
      + w_dice · Dice
      + w_focal· Focal
      + w_aux  · (CE_aux_deep + CE_aux_mid)   [LSFormer auxiliary heads]
      + w_bnd  · BoundaryLoss                 [LSFormer EGBR auxiliary]

    For non-LSFormer models only ``CE + Dice + Focal`` are applied.

    Args:
        num_classes:     Number of segmentation classes.
        class_weights:   Per-class weight tensor (helps with imbalance).
        ignore_index:    Void label.
        w_ce, w_dice, w_focal: Main loss weights.
        w_aux:           Weight for deep-supervision auxiliary heads.
        w_bnd:           Weight for boundary auxiliary loss.
        label_smoothing: CE label smoothing value.
        use_focal:       Whether to include focal component.
    """

    def __init__(
        self,
        num_classes:    int = 2,
        class_weights:  Optional[torch.Tensor] = None,
        ignore_index:   int = 255,
        w_ce:           float = 1.0,
        w_dice:         float = 0.5,
        w_focal:        float = 0.5,
        w_aux:          float = 0.4,
        w_bnd:          float = 0.3,
        label_smoothing: float = 0.05,
        use_focal:      bool = True,
    ) -> None:
        super().__init__()
        self.w_ce    = w_ce
        self.w_dice  = w_dice
        self.w_focal = w_focal
        self.w_aux   = w_aux
        self.w_bnd   = w_bnd
        self.use_focal = use_focal

        self.ce    = WeightedCELoss(num_classes, class_weights,
                                    ignore_index, label_smoothing)
        self.dice  = DiceLoss(ignore_index=ignore_index)
        self.focal = FocalLoss(ignore_index=ignore_index) if use_focal else None
        self.bnd   = BoundaryLoss(ignore_index=ignore_index)

    def forward(self, outputs: dict,
                target: torch.Tensor) -> torch.Tensor:
        pred = outputs["out"]

        loss  = self.w_ce   * self.ce(pred, target)
        loss += self.w_dice * self.dice(pred, target)
        if self.use_focal and self.focal is not None:
            loss += self.w_focal * self.focal(pred, target)

        # LSFormer deep-supervision auxiliary heads
        if "aux_deep" in outputs:
            loss += self.w_aux * self.ce(outputs["aux_deep"], target)
        if "aux_mid" in outputs:
            loss += self.w_aux * 0.5 * self.ce(outputs["aux_mid"], target)

        # LSFormer boundary auxiliary
        if "boundary" in outputs:
            loss += self.w_bnd * self.bnd(outputs["boundary"], target)

        return loss


__all__ = [
    "WeightedCELoss", "DiceLoss", "FocalLoss",
    "BoundaryLoss", "LandslideSegLoss",
]
