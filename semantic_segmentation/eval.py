"""Evaluation / inference script for semantic segmentation.

Usage::

    # Evaluate a trained model on the validation split
    python eval.py --config configs/default.yaml --checkpoint work_dirs/best.pth

    # Run single-image inference and save coloured prediction
    python eval.py --config configs/default.yaml --checkpoint work_dirs/best.pth \\
                   --image /path/to/image.jpg --output pred.png
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from semantic_segmentation.models import Segmentor
from semantic_segmentation.datasets import ADE20K, Cityscapes, VOCSegmentation
from semantic_segmentation.utils import SegmentationMetric, build_val_transform

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Colour palette (Cityscapes-style, auto-generated for arbitrary class counts)
# ---------------------------------------------------------------------------

def _random_palette(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(n, 3), dtype=np.uint8)


CITYSCAPES_PALETTE = np.array([
    [128,  64, 128], [244,  35, 232], [ 70,  70,  70], [102, 102, 156],
    [190, 153, 153], [153, 153, 153], [250, 170,  30], [220, 220,   0],
    [107, 142,  35], [152, 251, 152], [ 70, 130, 180], [220,  20,  60],
    [255,   0,   0], [  0,   0, 142], [  0,   0,  70], [  0,  60, 100],
    [  0,  80, 100], [  0,   0, 230], [119,  11,  32],
], dtype=np.uint8)


def pred_to_color(pred: np.ndarray, palette: np.ndarray,
                  ignore_index: int = 255) -> np.ndarray:
    """Convert a 2-D label map to an RGB image using ``palette``."""
    h, w  = pred.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id in range(len(palette)):
        mask = pred == cls_id
        color[mask] = palette[cls_id]
    return color


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(cfg: dict) -> torch.nn.Module:
    mcfg = cfg["model"]
    return Segmentor.from_config({
        "backbone":        mcfg["backbone"],
        "head":            mcfg["head"],
        "num_classes":     cfg["dataset"]["num_classes"],
        "channels":        mcfg.get("channels", 256),
        "aux_head":        mcfg.get("aux_head", True),
        "aux_channels":    mcfg.get("aux_channels", 256),
        "backbone_kwargs": mcfg.get("backbone_kwargs", {}),
        "head_kwargs":     mcfg.get("head_kwargs", {}),
    })


def build_val_dataset(cfg: dict, split: str = "val"):
    name     = cfg["dataset"]["name"]
    root     = cfg["dataset"]["root"]
    ignore   = cfg["dataset"].get("ignore_index", 255)
    val_size = cfg["dataset"].get("val_size", 512)
    transform = build_val_transform(val_size)
    datasets = {
        "cityscapes": Cityscapes,
        "ade20k":     ADE20K,
        "voc":        VOCSegmentation,
    }
    return datasets[name](root=root, split=split, transform=transform,
                          ignore_index=ignore)


# ---------------------------------------------------------------------------
# Multi-scale test-time augmentation
# ---------------------------------------------------------------------------

@torch.no_grad()
def ms_inference(model, image_tensor: torch.Tensor, device,
                 scales=(0.75, 1.0, 1.25), flip: bool = True) -> torch.Tensor:
    """Multi-scale + optional horizontal-flip TTA."""
    model.eval()
    H, W = image_tensor.shape[-2:]
    all_logits = torch.zeros(1, model.decode_head.linear_pred.out_channels
                             if hasattr(model.decode_head, "linear_pred")
                             else model.decode_head.cls_seg.out_channels,
                             H, W, device=device)
    for scale in scales:
        nh, nw = int(H * scale), int(W * scale)
        img = F.interpolate(image_tensor, size=(nh, nw),
                            mode="bilinear", align_corners=False)
        logits = model(img.to(device))["out"]
        logits = F.interpolate(logits, size=(H, W),
                               mode="bilinear", align_corners=False)
        all_logits += logits
        if flip:
            img_f  = torch.flip(img, dims=[-1])
            logits_f = model(img_f.to(device))["out"]
            logits_f = torch.flip(logits_f, dims=[-1])
            logits_f = F.interpolate(logits_f, size=(H, W),
                                     mode="bilinear", align_corners=False)
            all_logits += logits_f

    return all_logits


# ---------------------------------------------------------------------------
# Evaluate on dataset
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, num_classes: int, ignore_index: int,
             device, multiscale: bool = False) -> dict:
    model.eval()
    metric = SegmentationMetric(num_classes, ignore_index)
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        if multiscale:
            logits = ms_inference(model, images, device)
        else:
            logits = model(images)["out"]
        preds = logits.argmax(dim=1).cpu()
        metric.update(preds, labels)
    return metric.compute()


# ---------------------------------------------------------------------------
# Single-image inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def infer_single(model, image_path: str, cfg: dict, device) -> np.ndarray:
    """Return a predicted label map (H, W) as a NumPy array."""
    from semantic_segmentation.utils.transforms import ToTensor, Normalize, Compose
    transform = Compose([ToTensor(), Normalize()])
    image = Image.open(image_path).convert("RGB")
    img_t, _ = transform(image, image)   # label is a dummy here
    img_t = img_t.unsqueeze(0).to(device)
    model.eval()
    logits = model(img_t)["out"]
    return logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Semantic Segmentation Evaluation")
    parser.add_argument("--config",     required=True,
                        help="Path to YAML config file")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--split",      default="val",
                        help="Dataset split to evaluate on")
    parser.add_argument("--multiscale", action="store_true",
                        help="Use multi-scale TTA")
    parser.add_argument("--device",     default="cuda")
    # Single-image mode
    parser.add_argument("--image",  default=None,
                        help="If provided, run inference on this single image")
    parser.add_argument("--output", default="prediction.png",
                        help="Output path for single-image prediction PNG")
    args = parser.parse_args()

    cfg    = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ── Load model ───────────────────────────────────────────────────────────
    model = build_model(cfg).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state)
    logger.info(f"Loaded checkpoint: {args.checkpoint}")

    num_classes  = cfg["dataset"]["num_classes"]
    ignore_index = cfg["dataset"].get("ignore_index", 255)

    # ── Single-image mode ────────────────────────────────────────────────────
    if args.image is not None:
        pred = infer_single(model, args.image, cfg, device)
        if num_classes == 19:
            palette = CITYSCAPES_PALETTE
        else:
            palette = _random_palette(num_classes)
        color = pred_to_color(pred, palette, ignore_index)
        Image.fromarray(color).save(args.output)
        logger.info(f"Prediction saved → {args.output}")
        return

    # ── Dataset evaluation ───────────────────────────────────────────────────
    val_ds  = build_val_dataset(cfg, split=args.split)
    val_ldr = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=4)

    results = evaluate(model, val_ldr, num_classes, ignore_index,
                       device, multiscale=args.multiscale)

    logger.info(
        f"mIoU:      {results['miou']*100:.2f}%\n"
        f"Pixel Acc: {results['pixel_acc']*100:.2f}%"
    )
    logger.info("Per-class IoU:")
    for i, iou in enumerate(results["iou_per_class"]):
        if not np.isnan(iou):
            logger.info(f"  Class {i:3d}: {iou*100:.2f}%")


if __name__ == "__main__":
    main()
