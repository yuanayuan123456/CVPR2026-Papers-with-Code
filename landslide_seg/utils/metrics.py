"""
Evaluation metrics for semantic segmentation.

Implements:
  - mIoU (mean Intersection-over-Union)
  - F1 / Dice score per class and mean
  - Overall Accuracy (OA / pixel accuracy)
  - Cohen's Kappa coefficient
  - Precision and Recall per class
"""

from typing import Dict, List, Optional

import numpy as np
import torch


class SegmentationMetrics:
    """Accumulates predictions/labels and computes segmentation metrics.

    Usage::

        m = SegmentationMetrics(num_classes=2, ignore_index=255)
        for pred_logit, label in loader:
            m.update(pred_logit.argmax(1), label)
        results = m.compute()

    Args:
        num_classes:  Number of segmentation classes.
        ignore_index: Label value to ignore (e.g. void / boundary).
    """

    def __init__(self, num_classes: int, ignore_index: int = 255) -> None:
        self.num_classes  = num_classes
        self.ignore_index = ignore_index
        self._cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    # ── accumulate ─────────────────────────────────────────────────────────
    def reset(self) -> None:
        self._cm[:] = 0

    @torch.no_grad()
    def update(self, pred: torch.Tensor, label: torch.Tensor) -> None:
        """Update confusion matrix.

        Args:
            pred:  Predicted class indices (B, H, W) or (H, W).
            label: Ground-truth labels     (B, H, W) or (H, W).
        """
        pred  = pred .cpu().numpy().astype(np.int64).ravel()
        label = label.cpu().numpy().astype(np.int64).ravel()
        mask  = label != self.ignore_index
        pred, label = pred[mask], label[mask]
        valid = ((label >= 0) & (label < self.num_classes) &
                 (pred  >= 0) & (pred  < self.num_classes))
        pred, label = pred[valid], label[valid]
        idx = self.num_classes * label + pred
        self._cm += np.bincount(idx, minlength=self.num_classes ** 2
                                ).reshape(self.num_classes, self.num_classes)

    # ── compute ─────────────────────────────────────────────────────────────
    def compute(self) -> Dict[str, float]:
        """Return dict of all metrics."""
        cm  = self._cm.astype(np.float64)
        tp  = np.diag(cm)
        fp  = cm.sum(0) - tp
        fn  = cm.sum(1) - tp
        total = cm.sum()

        # IoU per class
        iou_denom = tp + fp + fn
        iou = np.where(iou_denom > 0, tp / iou_denom, np.nan)

        # Dice / F1 per class
        dice_denom = 2 * tp + fp + fn
        dice = np.where(dice_denom > 0, 2 * tp / dice_denom, np.nan)

        # Precision / Recall per class
        prec_denom = tp + fp
        rec_denom  = tp + fn
        precision  = np.where(prec_denom > 0, tp / prec_denom, np.nan)
        recall     = np.where(rec_denom  > 0, tp / rec_denom,  np.nan)

        # Overall accuracy
        oa = float(tp.sum() / total) if total > 0 else 0.0

        # Kappa
        expected = (cm.sum(0) * cm.sum(1)) / (total * total)
        kappa_num = oa - expected.sum()
        kappa_den = 1.0 - expected.sum()
        kappa = float(kappa_num / kappa_den) if kappa_den > 0 else 0.0

        return {
            "miou":          float(np.nanmean(iou)),
            "iou_per_class": iou.tolist(),
            "mean_dice":     float(np.nanmean(dice)),
            "dice_per_class": dice.tolist(),
            "mean_precision": float(np.nanmean(precision)),
            "mean_recall":   float(np.nanmean(recall)),
            "prec_per_class": precision.tolist(),
            "rec_per_class":  recall.tolist(),
            "oa":            oa,
            "kappa":         kappa,
        }

    # ── pretty print ────────────────────────────────────────────────────────
    def __str__(self) -> str:
        r = self.compute()
        lines = [
            f"mIoU:    {r['miou']*100:.2f}%",
            f"mDice:   {r['mean_dice']*100:.2f}%",
            f"OA:      {r['oa']*100:.2f}%",
            f"Kappa:   {r['kappa']:.4f}",
            f"mPrec:   {r['mean_precision']*100:.2f}%",
            f"mRecall: {r['mean_recall']*100:.2f}%",
        ]
        for i, (iou, dice) in enumerate(
                zip(r["iou_per_class"], r["dice_per_class"])):
            if not np.isnan(iou):
                lines.append(
                    f"  Class {i:2d}:  IoU={iou*100:5.2f}%  Dice={dice*100:5.2f}%"
                )
        return "\n".join(lines)


# ─────────────────────────── batch helpers ───────────────────────────────────

@torch.no_grad()
def batch_iou(pred: torch.Tensor, label: torch.Tensor,
              num_classes: int, ignore_index: int = 255) -> float:
    """Compute mean IoU for a single batch (no state accumulation)."""
    m = SegmentationMetrics(num_classes, ignore_index)
    m.update(pred, label)
    return m.compute()["miou"]


__all__ = ["SegmentationMetrics", "batch_iou"]
