"""
Visualization utilities for training curves, predictions, and comparisons.

Functions
---------
plot_training_curves   – loss + metric curves from a training log CSV
plot_prediction        – side-by-side image / GT / prediction overlay
plot_confusion_matrix  – normalized confusion matrix heat-map
plot_model_comparison  – bar chart comparing mIoU across models
save_prediction_grid   – save a grid of model predictions for qualitative eval
"""

import os
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")   # non-interactive backend for server environments
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from .metrics import SegmentationMetrics


# ─────────────────────────── colour palettes ─────────────────────────────────

# Default 2-class palette: background=dark, landslide=red
DEFAULT_PALETTE = {
    0: (30,  30,  30),    # background
    1: (220, 50,  50),    # landslide
    2: (50,  200, 50),    # class 2 (e.g. debris)
    3: (50,  50,  220),   # class 3 (e.g. rockfall)
    4: (220, 200, 50),    # class 4
    255: (0,  0,  0),     # ignore
}


def _palette_to_rgb(mask: np.ndarray,
                    palette: Optional[Dict[int, tuple]] = None) -> np.ndarray:
    if palette is None:
        palette = DEFAULT_PALETTE
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cls_id, color in palette.items():
        rgb[mask == cls_id] = color
    return rgb


def _denormalize(img: np.ndarray,
                 mean=(0.485, 0.456, 0.406),
                 std =(0.229, 0.224, 0.225)) -> np.ndarray:
    """Undo ImageNet normalisation and return uint8 HxWx3."""
    img = img.transpose(1, 2, 0).astype(np.float32)
    img = (img * np.array(std) + np.array(mean)).clip(0, 1)
    return (img * 255).astype(np.uint8)


# ─────────────────────────── training curves ─────────────────────────────────

def plot_training_curves(
    log_path: str,
    save_path: Optional[str] = None,
    show: bool = False,
) -> None:
    """Plot loss and metric curves from a CSV training log.

    The CSV is expected to have at least columns:
    ``epoch, train_loss, val_loss, val_miou`` (others are optional).

    Args:
        log_path:  Path to the training log CSV file.
        save_path: Where to save the figure (``None`` = do not save).
        show:      Whether to call ``plt.show()``.
    """
    import csv
    rows = []
    with open(log_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) if v not in ("", "nan") else float("nan")
                         for k, v in row.items()})

    epochs = [r["epoch"] for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Training History", fontsize=14, fontweight="bold")

    # Loss curve
    ax = axes[0]
    ax.plot(epochs, [r["train_loss"] for r in rows], label="Train Loss",
            color="#2196F3", linewidth=2)
    if "val_loss" in rows[0]:
        ax.plot(epochs, [r.get("val_loss", float("nan")) for r in rows],
                label="Val Loss", color="#F44336", linewidth=2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Training / Validation Loss")
    ax.legend(); ax.grid(True, alpha=0.3)

    # mIoU curve
    ax = axes[1]
    if "val_miou" in rows[0]:
        ax.plot(epochs, [r.get("val_miou", float("nan")) for r in rows],
                label="Val mIoU", color="#4CAF50", linewidth=2)
    if "train_miou" in rows[0]:
        ax.plot(epochs, [r.get("train_miou", float("nan")) for r in rows],
                label="Train mIoU", color="#9C27B0", linewidth=2, linestyle="--")
    ax.set_xlabel("Epoch"); ax.set_ylabel("mIoU")
    ax.set_title("Mean IoU")
    ax.legend(); ax.grid(True, alpha=0.3)

    # Additional metrics (OA, Dice, Kappa)
    ax = axes[2]
    for col, color, label in [
        ("val_oa",    "#FF9800", "OA"),
        ("val_dice",  "#009688", "Dice"),
        ("val_kappa", "#795548", "Kappa"),
    ]:
        if col in rows[0]:
            ax.plot(epochs, [r.get(col, float("nan")) for r in rows],
                    label=label, color=color, linewidth=2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Score")
    ax.set_title("Additional Metrics")
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Training curves saved → {save_path}")
    if show:
        plt.show()
    plt.close()


def plot_loss_and_metric(
    train_losses: List[float],
    val_losses:   List[float],
    val_mious:    List[float],
    save_dir:     str,
    model_name:   str = "model",
) -> None:
    """Convenience function: plot from in-memory lists (called during training)."""
    epochs = list(range(1, len(train_losses) + 1))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{model_name} — Training Curves", fontsize=13, fontweight="bold")

    ax = axes[0]
    ax.plot(epochs, train_losses, label="Train Loss",
            color="#2196F3", linewidth=2)
    ax.plot(epochs[:len(val_losses)], val_losses, label="Val Loss",
            color="#F44336", linewidth=2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Loss Curves")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(epochs[:len(val_mious)], [v * 100 for v in val_mious],
            label="Val mIoU (%)", color="#4CAF50", linewidth=2)
    if len(val_mious) > 0:
        best_epoch = val_mious.index(max(val_mious)) + 1
        best_val   = max(val_mious) * 100
        ax.axvline(best_epoch, linestyle="--", color="gray", alpha=0.5)
        ax.annotate(f"Best: {best_val:.2f}%",
                    xy=(best_epoch, best_val),
                    xytext=(best_epoch + 1, best_val - 3),
                    fontsize=9, color="green")
    ax.set_xlabel("Epoch"); ax.set_ylabel("mIoU (%)")
    ax.set_title("Validation mIoU")
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{model_name}_training_curves.png")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training curves saved → {save_path}")


# ─────────────────────────── prediction visualisation ────────────────────────

def plot_prediction(
    image:      np.ndarray,
    gt_mask:    np.ndarray,
    pred_mask:  np.ndarray,
    save_path:  Optional[str] = None,
    palette:    Optional[Dict[int, tuple]] = None,
    title:      str = "",
    show:       bool = False,
) -> None:
    """Side-by-side: original | ground-truth | prediction (+ overlay).

    Args:
        image:     (C, H, W) normalised float tensor or (H, W, 3) uint8.
        gt_mask:   (H, W) integer label array.
        pred_mask: (H, W) integer prediction array.
        save_path: File path to save figure.
        palette:   Class colour dict.
        title:     Figure title.
        show:      Call plt.show().
    """
    if image.ndim == 3 and image.shape[0] in (1, 3, 4):
        image = _denormalize(image)
    gt_rgb   = _palette_to_rgb(gt_mask,   palette)
    pred_rgb = _palette_to_rgb(pred_mask, palette)

    # Overlay prediction on image
    overlay = image.copy().astype(np.float32)
    fg = (pred_mask > 0)
    color = np.array(DEFAULT_PALETTE.get(1, (220, 50, 50)), dtype=np.float32)
    overlay[fg] = 0.5 * overlay[fg] + 0.5 * color
    overlay = overlay.clip(0, 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, arr, ttl in zip(
        axes,
        [image, gt_rgb, pred_rgb, overlay],
        ["Image", "Ground Truth", "Prediction", "Overlay"],
    ):
        ax.imshow(arr); ax.set_title(ttl, fontsize=11)
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


# ─────────────────────────── confusion matrix ────────────────────────────────

def plot_confusion_matrix(
    cm:           np.ndarray,
    class_names:  Optional[List[str]] = None,
    save_path:    Optional[str] = None,
    normalize:    bool = True,
    show:         bool = False,
) -> None:
    """Plot a (normalised) confusion matrix heat-map."""
    if normalize:
        denom = cm.sum(axis=1, keepdims=True).clip(min=1)
        cm_plot = cm.astype(float) / denom
    else:
        cm_plot = cm.astype(float)

    n = cm.shape[0]
    if class_names is None:
        class_names = [f"Class {i}" for i in range(n)]

    fig, ax = plt.subplots(figsize=(max(5, n), max(4, n)))
    im = ax.imshow(cm_plot, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_xticks(range(n)); ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix" + (" (Normalised)" if normalize else ""))
    for i in range(n):
        for j in range(n):
            val  = cm_plot[i, j]
            text = f"{val:.2f}" if normalize else f"{int(val)}"
            ax.text(j, i, text, ha="center", va="center",
                    color="white" if val > 0.6 else "black", fontsize=9)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


# ─────────────────────────── model comparison bar chart ──────────────────────

def plot_model_comparison(
    results:    Dict[str, Dict[str, float]],
    save_path:  Optional[str] = None,
    metrics:    Optional[List[str]] = None,
    show:       bool = False,
) -> None:
    """Bar chart comparing multiple models on several metrics.

    Args:
        results: ``{model_name: {metric_name: value, ...}, ...}``
        save_path: Path to save figure.
        metrics: Which metrics to display (default: mIoU, mean_dice, oa, kappa).
        show: Call plt.show().

    Example::

        plot_model_comparison({
            "LSFormer":      {"miou": 0.82, "mean_dice": 0.85, "oa": 0.91},
            "UNet":          {"miou": 0.74, "mean_dice": 0.78, "oa": 0.88},
            "DeepLabV3+":    {"miou": 0.77, "mean_dice": 0.81, "oa": 0.89},
            "SegFormer":     {"miou": 0.79, "mean_dice": 0.83, "oa": 0.90},
            "HRNet":         {"miou": 0.76, "mean_dice": 0.80, "oa": 0.89},
        })
    """
    if metrics is None:
        metrics = ["miou", "mean_dice", "oa", "kappa"]

    model_names = list(results.keys())
    x = np.arange(len(model_names))
    n_metrics = len(metrics)
    width = 0.8 / n_metrics

    fig, ax = plt.subplots(figsize=(max(10, 2 * len(model_names)), 6))

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0",
              "#F44336", "#009688", "#795548"]
    metric_labels = {
        "miou":       "mIoU",
        "mean_dice":  "Mean Dice",
        "oa":         "Overall Accuracy",
        "kappa":      "Kappa",
        "mean_precision": "Precision",
        "mean_recall":   "Recall",
    }

    bars = []
    for k, metric in enumerate(metrics):
        vals   = [results[m].get(metric, 0) * 100 for m in model_names]
        offset = (k - n_metrics / 2 + 0.5) * width
        b = ax.bar(x + offset, vals, width,
                   label=metric_labels.get(metric, metric),
                   color=colors[k % len(colors)], alpha=0.85)
        bars.append(b)
        for bar, v in zip(b, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=11)
    ax.set_ylabel("Score (%)")
    ax.set_title("Model Comparison — Landslide Segmentation", fontsize=13)
    ax.legend(loc="lower right")
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Comparison chart saved → {save_path}")
    if show:
        plt.show()
    plt.close()


# ─────────────────────────── prediction grid ─────────────────────────────────

def save_prediction_grid(
    images:    List[np.ndarray],
    gt_masks:  List[np.ndarray],
    pred_masks: List[np.ndarray],
    save_path: str,
    palette:   Optional[Dict[int, tuple]] = None,
    max_cols:  int = 4,
) -> None:
    """Save a grid of (image, GT, pred) triplets for qualitative evaluation."""
    n = min(len(images), max_cols * 2)
    fig, axes = plt.subplots(3, n, figsize=(4 * n, 12))
    if n == 1:
        axes = axes[:, None]

    row_labels = ["Image", "Ground Truth", "Prediction"]
    for col in range(n):
        img = images[col]
        if img.ndim == 3 and img.shape[0] in (1, 3, 4):
            img = _denormalize(img)
        gt  = _palette_to_rgb(gt_masks  [col], palette)
        pr  = _palette_to_rgb(pred_masks[col], palette)
        for row, arr in enumerate([img, gt, pr]):
            ax = axes[row][col]
            ax.imshow(arr); ax.axis("off")
            if col == 0:
                ax.set_ylabel(row_labels[row], fontsize=11)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Prediction grid saved → {save_path}")


__all__ = [
    "plot_training_curves",
    "plot_loss_and_metric",
    "plot_prediction",
    "plot_confusion_matrix",
    "plot_model_comparison",
    "save_prediction_grid",
]
