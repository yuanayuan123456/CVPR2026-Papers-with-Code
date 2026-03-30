"""
Evaluation script — evaluate a trained model on the test split.

Usage
-----
# Evaluate a single model
python eval.py --model lsformer \
               --checkpoint work_dirs/lsformer/best.pth \
               --config     configs/default.yaml

# Compare multiple models (generates a bar chart)
python eval.py --compare \
    lsformer:work_dirs/comparison/lsformer/best.pth \
    unet:work_dirs/comparison/unet/best.pth \
    deeplabv3plus:work_dirs/comparison/deeplabv3plus/best.pth \
    segformer:work_dirs/comparison/segformer/best.pth \
    hrnet:work_dirs/comparison/hrnet/best.pth \
    --config configs/comparison.yaml \
    --output_dir ./eval_results
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from landslide_seg.data    import build_datasets, build_dataloaders
from landslide_seg.models  import get_model
from landslide_seg.utils   import (
    LandslideSegLoss, SegmentationMetrics,
    plot_confusion_matrix, plot_model_comparison,
    save_prediction_grid, plot_prediction,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────── helpers ─────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_model(model_name: str, checkpoint: str, cfg: dict,
               device: torch.device) -> torch.nn.Module:
    mcfg  = cfg["model"]
    model = get_model(
        model_name,
        num_classes = mcfg.get("num_classes", 2),
        pretrained  = False,   # load from checkpoint only
        in_channels = mcfg.get("in_channels", 3),
    ).to(device)
    ckpt  = torch.load(checkpoint, map_location=device)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state)
    logger.info(f"Loaded {model_name} from {checkpoint}")
    return model


@torch.no_grad()
def evaluate_model(model, loader, num_classes: int, ignore_index: int,
                   device: torch.device, use_tta: bool = False,
                   use_amp: bool = False) -> dict:
    model.eval()
    metric = SegmentationMetrics(num_classes, ignore_index)
    all_imgs, all_gts, all_preds = [], [], []
    cm_total = np.zeros((num_classes, num_classes), dtype=np.int64)

    for images, labels in loader:
        images = images.to(device, non_blocking=True, dtype=torch.float32)
        labels = labels.to(device, non_blocking=True)

        with autocast(enabled=use_amp):
            if use_tta:
                logits = _tta_inference(model, images)
            else:
                logits = model(images)["out"]

        preds = logits.argmax(dim=1)
        metric.update(preds, labels)

        for b in range(images.size(0)):
            all_imgs.append(images[b].cpu().numpy())
            all_gts.append(  labels[b].cpu().numpy())
            all_preds.append(preds[b].cpu().numpy())

    results = metric.compute()
    results["cm"] = metric._cm.tolist()
    return results, all_imgs, all_gts, all_preds


def _tta_inference(model, images: torch.Tensor,
                   scales=(0.75, 1.0, 1.25),
                   flip: bool = True) -> torch.Tensor:
    """Multi-scale + horizontal-flip test-time augmentation."""
    H, W = images.shape[-2:]
    # Determine num_classes from a test forward pass
    with torch.no_grad():
        n_cls = model(images)["out"].shape[1]
    acc = torch.zeros(images.size(0), n_cls, H, W, device=images.device)
    for s in scales:
        nh, nw = int(H * s), int(W * s)
        img_s  = F.interpolate(images, (nh, nw), mode="bilinear",
                               align_corners=False)
        logits = F.interpolate(model(img_s)["out"], (H, W),
                               mode="bilinear", align_corners=False)
        acc += logits.softmax(dim=1)
        if flip:
            logits_f = F.interpolate(
                model(torch.flip(img_s, [-1]))["out"],
                (H, W), mode="bilinear", align_corners=False
            )
            acc += torch.flip(logits_f, [-1]).softmax(dim=1)
    return acc


# ─────────────────────────── single-model eval ───────────────────────────────

def eval_single(args, cfg, device):
    dcfg = cfg["dataset"]
    ds_dict = build_datasets(
        image_dir    = dcfg["image_dir"],
        mask_dir     = dcfg["mask_dir"],
        crop_size    = dcfg.get("crop_size", 512),
        val_size     = dcfg.get("val_size",  512),
        num_classes  = dcfg.get("num_classes", 2),
        ignore_index = dcfg.get("ignore_index", 255),
        train_ratio  = dcfg.get("train_ratio", 0.7),
        val_ratio    = dcfg.get("val_ratio",   0.15),
        split_file   = dcfg.get("split_file",  None),
        img_suffix   = dcfg.get("img_suffix",  ".jpg"),
        mask_suffix  = dcfg.get("mask_suffix", ".png"),
        seed         = dcfg.get("seed", 42),
    )
    loaders = build_dataloaders(
        ds_dict, batch_size=1, val_batch=1, num_workers=4
    )
    split   = args.split
    loader  = loaders.get(split, loaders["test"])

    model = load_model(args.model, args.checkpoint, cfg, device)
    num_classes  = dcfg.get("num_classes", 2)
    ignore_index = dcfg.get("ignore_index", 255)
    use_amp = bool(cfg["train"].get("amp", False))

    logger.info(f"Evaluating {args.model} on {split} split "
                f"({len(ds_dict[split])} samples)…")
    results, imgs, gts, preds = evaluate_model(
        model, loader, num_classes, ignore_index,
        device, use_tta=args.tta, use_amp=use_amp
    )

    # Print metrics
    class_names = dcfg.get("class_names", [f"Class {i}" for i in range(num_classes)])
    print("\n" + "=" * 60)
    print(f"Model: {args.model}  |  Split: {split}")
    print(f"  mIoU:        {results['miou']*100:.2f}%")
    print(f"  Mean Dice:   {results['mean_dice']*100:.2f}%")
    print(f"  OA:          {results['oa']*100:.2f}%")
    print(f"  Kappa:       {results['kappa']:.4f}")
    print(f"  Precision:   {results['mean_precision']*100:.2f}%")
    print(f"  Recall:      {results['mean_recall']*100:.2f}%")
    print("\nPer-class IoU:")
    for i, iou in enumerate(results["iou_per_class"]):
        if not (isinstance(iou, float) and np.isnan(iou)):
            name = class_names[i] if i < len(class_names) else f"Class {i}"
            print(f"  {name:20s}: {iou*100:.2f}%")
    print("=" * 60)

    # Save outputs
    out_dir = args.output_dir or os.path.join(os.path.dirname(args.checkpoint), "eval")
    os.makedirs(out_dir, exist_ok=True)

    # Confusion matrix
    cm = np.array(results["cm"])
    plot_confusion_matrix(cm, class_names,
                          save_path=os.path.join(out_dir, "confusion_matrix.png"))

    # Prediction grid
    n_vis = min(8, len(imgs))
    save_prediction_grid(imgs[:n_vis], gts[:n_vis], preds[:n_vis],
                         os.path.join(out_dir, "prediction_grid.png"))

    # JSON results
    res_clean = {k: v for k, v in results.items() if k != "cm"}
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(res_clean, f, indent=2)
    logger.info(f"Results saved → {out_dir}")


# ─────────────────────────── multi-model comparison ──────────────────────────

def eval_compare(args, cfg, device):
    """Evaluate multiple models and produce comparison figures."""
    dcfg = cfg["dataset"]
    ds_dict = build_datasets(
        image_dir    = dcfg["image_dir"],
        mask_dir     = dcfg["mask_dir"],
        crop_size    = dcfg.get("crop_size", 512),
        val_size     = dcfg.get("val_size",  512),
        num_classes  = dcfg.get("num_classes", 2),
        ignore_index = dcfg.get("ignore_index", 255),
        train_ratio  = dcfg.get("train_ratio", 0.7),
        val_ratio    = dcfg.get("val_ratio",   0.15),
        split_file   = dcfg.get("split_file",  None),
        img_suffix   = dcfg.get("img_suffix",  ".jpg"),
        mask_suffix  = dcfg.get("mask_suffix", ".png"),
        seed         = dcfg.get("seed", 42),
    )
    loader = build_dataloaders(
        ds_dict, batch_size=1, val_batch=1, num_workers=4
    )["test"]

    num_classes  = dcfg.get("num_classes", 2)
    ignore_index = dcfg.get("ignore_index", 255)
    use_amp      = bool(cfg["train"].get("amp", False))
    out_dir      = args.output_dir or "./eval_results"
    os.makedirs(out_dir, exist_ok=True)

    all_results = {}
    # Format: "model_name:checkpoint_path"
    for entry in args.compare:
        name, ckpt = entry.split(":", 1)
        logger.info(f"\nEvaluating {name}…")
        model = load_model(name, ckpt, cfg, device)
        results, imgs, gts, preds = evaluate_model(
            model, loader, num_classes, ignore_index,
            device, use_tta=args.tta, use_amp=use_amp
        )
        all_results[name] = results
        logger.info(
            f"  {name}: mIoU={results['miou']*100:.2f}%  "
            f"Dice={results['mean_dice']*100:.2f}%  OA={results['oa']*100:.2f}%"
        )
        # Save confusion matrix per model
        cm = np.array(results["cm"])
        class_names = dcfg.get("class_names",
                                [f"Class {i}" for i in range(num_classes)])
        plot_confusion_matrix(cm, class_names,
            save_path=os.path.join(out_dir, f"{name}_confusion_matrix.png"))

    # Comparison bar chart
    plot_model_comparison(
        {n: r for n, r in all_results.items()},
        save_path=os.path.join(out_dir, "model_comparison.png"),
        metrics=["miou", "mean_dice", "oa", "kappa"],
    )

    # JSON summary
    summary = {n: {k: v for k, v in r.items() if k not in ("cm",)}
               for n, r in all_results.items()}
    with open(os.path.join(out_dir, "comparison_results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nComparison results saved → {out_dir}")

    # Print table
    print("\n" + "=" * 70)
    print(f"{'Model':<20} {'mIoU':>8} {'Dice':>8} {'OA':>8} {'Kappa':>8}")
    print("-" * 70)
    for name, res in all_results.items():
        print(f"{name:<20} {res['miou']*100:>7.2f}% "
              f"{res['mean_dice']*100:>7.2f}% "
              f"{res['oa']*100:>7.2f}% "
              f"{res['kappa']:>8.4f}")
    print("=" * 70)


# ─────────────────────────── CLI ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate landslide segmentation models"
    )
    parser.add_argument("--config",     default="configs/default.yaml")
    parser.add_argument("--device",     default="cuda")
    parser.add_argument("--tta",        action="store_true",
                        help="Use multi-scale + flip TTA")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--split",      default="test",
                        choices=["train", "val", "test"])

    sub = parser.add_subparsers(dest="mode")

    # Single model evaluation
    single = sub.add_parser("single", help="Evaluate one model")
    single.add_argument("--model",      required=True,
                        choices=["mlformer", "lsformer", "unet",
                                 "deeplabv3plus", "segformer", "hrnet"])
    single.add_argument("--checkpoint", required=True)

    # Multi-model comparison
    comp = sub.add_parser("compare", help="Compare multiple models")
    comp.add_argument("--compare", nargs="+",
                      metavar="model:checkpoint",
                      help="e.g. lsformer:path/best.pth unet:path/best.pth")

    args = parser.parse_args()

    # Support flat CLI (no subcommand) for backward compatibility
    if args.mode is None:
        if hasattr(args, "compare") and args.compare:
            args.mode = "compare"
        else:
            parser.print_help(); return

    cfg    = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.mode == "single":
        eval_single(args, cfg, device)
    elif args.mode == "compare":
        eval_compare(args, cfg, device)


if __name__ == "__main__":
    main()
