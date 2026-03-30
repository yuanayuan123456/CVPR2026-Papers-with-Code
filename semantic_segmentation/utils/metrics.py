"""Evaluation metrics for semantic segmentation."""

import numpy as np
import torch
from typing import Optional


class SegmentationMetric:
    """Accumulates pixel-level predictions and computes mIoU / pixel accuracy.

    Usage::

        metric = SegmentationMetric(num_classes=19, ignore_index=255)
        for pred, label in loader:
            metric.update(pred.argmax(1), label)
        print(metric.compute())

    Args:
        num_classes (int): Number of foreground classes.
        ignore_index (int | None): Label value to ignore.
    """

    def __init__(self, num_classes: int, ignore_index: Optional[int] = 255) -> None:
        self.num_classes  = num_classes
        self.ignore_index = ignore_index
        self.confusion    = np.zeros((num_classes, num_classes), dtype=np.int64)

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.confusion[:] = 0

    # ------------------------------------------------------------------
    @torch.no_grad()
    def update(self, pred: torch.Tensor, label: torch.Tensor) -> None:
        """Update the confusion matrix.

        Args:
            pred (Tensor): Predicted class indices, shape ``(B, H, W)``.
            label (Tensor): Ground-truth labels,    shape ``(B, H, W)``.
        """
        pred  = pred.cpu().numpy().astype(np.int64).ravel()
        label = label.cpu().numpy().astype(np.int64).ravel()

        mask = label != self.ignore_index
        pred, label = pred[mask], label[mask]

        # Clip to valid range
        valid = (label >= 0) & (label < self.num_classes) & \
                (pred  >= 0) & (pred  < self.num_classes)
        pred, label = pred[valid], label[valid]

        indices = self.num_classes * label + pred
        self.confusion += np.bincount(indices,
                                      minlength=self.num_classes ** 2
                                      ).reshape(self.num_classes, self.num_classes)

    # ------------------------------------------------------------------
    def compute(self) -> dict:
        """Return a dict with ``miou``, ``iou_per_class``, and ``pixel_acc``."""
        cm = self.confusion.astype(np.float64)
        tp  = np.diag(cm)
        fp  = cm.sum(axis=0) - tp
        fn  = cm.sum(axis=1) - tp

        denom = tp + fp + fn
        iou   = np.where(denom > 0, tp / denom, np.nan)

        miou      = float(np.nanmean(iou))
        pixel_acc = float(tp.sum() / cm.sum()) if cm.sum() > 0 else 0.0

        return {
            "miou":          miou,
            "iou_per_class": iou.tolist(),
            "pixel_acc":     pixel_acc,
        }

    # ------------------------------------------------------------------
    def __str__(self) -> str:
        r = self.compute()
        return (f"mIoU: {r['miou']*100:.2f}%  |  "
                f"Pixel Acc: {r['pixel_acc']*100:.2f}%")


__all__ = ["SegmentationMetric"]
