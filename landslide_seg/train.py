"""
Training script for landslide semantic segmentation.

Usage examples
--------------
# Train proposed LSFormer model
python train.py --model lsformer --config configs/default.yaml

# Train a comparison model
python train.py --model unet         --config configs/comparison.yaml
python train.py --model deeplabv3plus --config configs/comparison.yaml
python train.py --model segformer    --config configs/comparison.yaml
python train.py --model hrnet        --config configs/comparison.yaml

# Resume from checkpoint
python train.py --model lsformer --resume work_dirs/lsformer/best.pth

# Override config values on CLI
python train.py --model lsformer --epochs 50 --batch_size 4 --lr 1e-4
"""

import argparse
import csv
import logging
import math
import os
import sys
import time
from pathlib import Path
from copy import deepcopy

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

import yaml

# ── project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))
from landslide_seg.data    import build_datasets, build_dataloaders
from landslide_seg.models  import get_model
from landslide_seg.utils   import (
    LandslideSegLoss, SegmentationMetrics,
    plot_loss_and_metric, plot_prediction,
    plot_confusion_matrix, save_prediction_grid,
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


def merge_config(cfg: dict, args) -> dict:
    """Override cfg fields from argparse namespace (non-None values win)."""
    overrides = {
        "model.name":        args.model,
        "train.epochs":      args.epochs,
        "train.batch_size":  args.batch_size,
        "optimizer.lr":      args.lr,
        "output.save_dir":   args.save_dir,
    }
    for dotpath, val in overrides.items():
        if val is None:
            continue
        keys = dotpath.split(".")
        d = cfg
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = val
    return cfg


def get_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    ocfg  = cfg["optimizer"]
    otype = ocfg.get("type", "adamw").lower()
    lr    = float(ocfg.get("lr", 6e-5))
    wd    = float(ocfg.get("weight_decay", 0.01))
    mult  = float(ocfg.get("backbone_lr_multiplier", 0.1))

    # Separate backbone and head parameters
    backbone_ids = set()
    for name in ("backbone", "cnn_stem", "cnn_layer1", "cnn_layer2",
                 "cnn_layer3", "cnn_layer4"):
        mod = getattr(model, name, None)
        if mod is not None:
            backbone_ids.update(id(p) for p in mod.parameters())

    head_params     = [p for p in model.parameters()
                       if p.requires_grad and id(p) not in backbone_ids]
    backbone_params = [p for p in model.parameters()
                       if p.requires_grad and id(p) in backbone_ids]

    param_groups = [
        {"params": backbone_params, "lr": lr * mult},
        {"params": head_params,     "lr": lr},
    ]

    if otype == "adamw":
        return torch.optim.AdamW(param_groups, weight_decay=wd,
                                 betas=tuple(ocfg.get("betas", [0.9, 0.999])))
    elif otype == "sgd":
        return torch.optim.SGD(param_groups, momentum=0.9,
                                weight_decay=wd, nesterov=True)
    else:
        raise ValueError(f"Unknown optimizer: {otype}")


def get_scheduler(optimizer, cfg: dict, total_iters: int, warmup_iters: int):
    scfg  = cfg.get("scheduler", {})
    stype = scfg.get("type", "poly").lower()

    if stype == "poly":
        power = float(scfg.get("power", 1.0))
        min_lr = float(scfg.get("min_lr", 1e-6))
        base_lrs = [pg["lr"] for pg in optimizer.param_groups]

        def lr_lambda(it):
            if it < warmup_iters:
                return (it + 1) / max(warmup_iters, 1)
            prog = (it - warmup_iters) / max(total_iters - warmup_iters, 1)
            scale = (1 - prog) ** power
            # enforce min_lr
            for i, base in enumerate(base_lrs):
                if base * scale < min_lr:
                    return min_lr / base
            return scale

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    elif stype == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_iters,
            eta_min=float(scfg.get("min_lr", 1e-6))
        )
    elif stype == "onecycle":
        max_lrs = [pg["lr"] for pg in optimizer.param_groups]
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=max_lrs, total_steps=total_iters
        )
    else:
        raise ValueError(f"Unknown scheduler: {stype}")


def save_checkpoint(state: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    logger.info(f"Checkpoint saved → {path}")


class EMA:
    """Exponential Moving Average of model weights."""
    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        self.model = deepcopy(model).eval()
        self.decay = decay

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for ema_p, p in zip(self.model.parameters(), model.parameters()):
            ema_p.mul_(self.decay).add_(p.data, alpha=1 - self.decay)


# ─────────────────────────── training loop ───────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, scheduler,
                    scaler, device, epoch: int, cfg: dict,
                    ema: "EMA | None" = None) -> float:
    model.train()
    use_amp      = bool(cfg["train"].get("amp", False))
    grad_clip    = float(cfg["train"].get("grad_clip", 0.0))
    log_interval = int(cfg["output"].get("log_interval", 20))
    total_loss   = 0.0
    t0 = time.time()

    for i, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True, dtype=torch.float32)
        labels = labels.to(device, non_blocking=True)

        with autocast(enabled=use_amp):
            outputs = model(images)
            loss    = criterion(outputs, labels)

        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        scheduler.step()
        if ema is not None:
            ema.update(model)

        total_loss += loss.item()
        if (i + 1) % log_interval == 0:
            lr_now = optimizer.param_groups[-1]["lr"]
            logger.info(
                f"Ep{epoch:3d} [{i+1:4d}/{len(loader)}]  "
                f"loss={total_loss/(i+1):.4f}  lr={lr_now:.2e}  "
                f"t={time.time()-t0:.1f}s"
            )
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device, cfg: dict,
             save_dir: str = None, epoch: int = 0,
             split: str = "val") -> dict:
    model.eval()
    num_classes  = cfg["dataset"]["num_classes"]
    ignore_index = cfg["dataset"].get("ignore_index", 255)
    use_amp      = bool(cfg["train"].get("amp", False))

    metric     = SegmentationMetrics(num_classes, ignore_index)
    total_loss = 0.0
    pred_images, pred_gts, pred_preds = [], [], []
    max_vis = 4

    for images, labels in loader:
        images = images.to(device, non_blocking=True, dtype=torch.float32)
        labels = labels.to(device, non_blocking=True)
        with autocast(enabled=use_amp):
            outputs = model(images)
            loss    = criterion(outputs, labels)
        total_loss += loss.item()
        preds = outputs["out"].argmax(dim=1)
        metric.update(preds, labels)

        # Collect samples for visualisation
        if len(pred_images) < max_vis:
            for b in range(min(images.size(0), max_vis - len(pred_images))):
                pred_images.append(images[b].cpu().numpy())
                pred_gts.append(   labels[b].cpu().numpy())
                pred_preds.append( preds[b].cpu().numpy())

    results = metric.compute()
    results["val_loss"] = total_loss / len(loader)

    # Save qualitative grid
    if (save_dir and pred_images and
            cfg["output"].get("save_pred_grid", False)):
        grid_path = os.path.join(save_dir, f"{split}_grid_ep{epoch:04d}.png")
        save_prediction_grid(pred_images, pred_gts, pred_preds, grid_path)

    return results


# ─────────────────────────── main ────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train a landslide segmentation model"
    )
    parser.add_argument("--model",  default=None,
        choices=["lsformer", "unet", "deeplabv3plus", "segformer", "hrnet"],
        help="Model architecture to train")
    parser.add_argument("--config", default="configs/default.yaml",
        help="Path to YAML configuration file")
    parser.add_argument("--resume", default=None,
        help="Path to checkpoint for resuming training")
    parser.add_argument("--device", default="cuda",
        help="Device: cuda | cpu")
    # Quick CLI overrides
    parser.add_argument("--epochs",     type=int,   default=None)
    parser.add_argument("--batch_size", type=int,   default=None)
    parser.add_argument("--lr",         type=float, default=None)
    parser.add_argument("--save_dir",   type=str,   default=None)
    args = parser.parse_args()

    # ── Config ───────────────────────────────────────────────────────────────
    cfg = load_config(args.config)
    cfg = merge_config(cfg, args)

    model_name = cfg["model"]["name"]
    save_dir   = os.path.join(cfg["output"]["save_dir"], model_name)
    os.makedirs(save_dir, exist_ok=True)
    cfg["output"]["save_dir"] = save_dir

    # Save resolved config
    with open(os.path.join(save_dir, "config.yaml"), "w") as f:
        yaml.dump(cfg, f)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Model: {model_name}  |  Device: {device}  |  Save: {save_dir}")

    # ── Data ─────────────────────────────────────────────────────────────────
    dcfg    = cfg["dataset"]
    ds_dict = build_datasets(
        image_dir    = dcfg["image_dir"],
        mask_dir     = dcfg["mask_dir"],
        crop_size    = dcfg.get("crop_size",  512),
        val_size     = dcfg.get("val_size",   512),
        num_classes  = dcfg.get("num_classes", 2),
        ignore_index = dcfg.get("ignore_index", 255),
        train_ratio  = dcfg.get("train_ratio", 0.7),
        val_ratio    = dcfg.get("val_ratio",   0.15),
        split_file   = dcfg.get("split_file",  None),
        img_suffix   = dcfg.get("img_suffix",  ".jpg"),
        mask_suffix  = dcfg.get("mask_suffix", ".png"),
        use_elastic  = dcfg.get("use_elastic", True),
        use_cutmix   = dcfg.get("use_cutmix",  False),
        seed         = dcfg.get("seed",        42),
    )
    tcfg = cfg["train"]
    loaders = build_dataloaders(
        ds_dict,
        batch_size  = tcfg.get("batch_size",  8),
        val_batch   = tcfg.get("val_batch",   4),
        num_workers = tcfg.get("num_workers", 4),
        pin_memory  = tcfg.get("pin_memory",  True),
    )
    logger.info(
        f"Dataset: train={len(ds_dict['train'])}  "
        f"val={len(ds_dict['val'])}  test={len(ds_dict['test'])}"
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    mcfg  = cfg["model"]
    model = get_model(
        model_name,
        num_classes = mcfg.get("num_classes", 2),
        pretrained  = mcfg.get("pretrained",  True),
        in_channels = mcfg.get("in_channels", 3),
    ).to(device)

    # ── Loss ──────────────────────────────────────────────────────────────────
    lcfg = cfg.get("loss", {})
    class_weights = None
    if lcfg.get("use_class_weights", True):
        logger.info("Computing class weights…")
        weights = ds_dict["train"].compute_class_weights()
        class_weights = torch.tensor(weights, device=device)
        logger.info(f"Class weights: {weights}")

    criterion = LandslideSegLoss(
        num_classes    = dcfg.get("num_classes", 2),
        class_weights  = class_weights,
        ignore_index   = dcfg.get("ignore_index", 255),
        w_ce           = float(lcfg.get("w_ce",    1.0)),
        w_dice         = float(lcfg.get("w_dice",  0.5)),
        w_focal        = float(lcfg.get("w_focal", 0.5)),
        w_aux          = float(lcfg.get("w_aux",   0.4)),
        w_bnd          = float(lcfg.get("w_bnd",   0.3)),
        label_smoothing = float(lcfg.get("label_smoothing", 0.05)),
        use_focal      = bool(lcfg.get("use_focal", True)),
    )

    # ── Optimiser / Scheduler ─────────────────────────────────────────────────
    epochs       = int(tcfg.get("epochs", 100))
    warmup_epochs = int(cfg.get("scheduler", {}).get("warmup_epochs", 5))
    iters_per_ep  = len(loaders["train"])
    total_iters   = epochs * iters_per_ep
    warmup_iters  = warmup_epochs * iters_per_ep

    optimizer = get_optimizer(model, cfg)
    scheduler = get_scheduler(optimizer, cfg, total_iters, warmup_iters)
    scaler    = GradScaler(enabled=bool(tcfg.get("amp", False)))
    ema       = EMA(model) if tcfg.get("ema", False) else None

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 0
    best_miou   = 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_miou   = ckpt.get("best_miou", 0.0)
        logger.info(
            f"Resumed from epoch {start_epoch-1}  (best_miou={best_miou:.4f})"
        )

    # ── CSV log ───────────────────────────────────────────────────────────────
    log_csv = os.path.join(save_dir, "train_log.csv")
    csv_fields = ["epoch", "train_loss", "val_loss", "val_miou",
                  "val_dice", "val_oa", "val_kappa"]
    if not os.path.exists(log_csv):
        with open(log_csv, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=csv_fields).writeheader()

    # ── Training history (for plots) ──────────────────────────────────────────
    train_losses, val_losses, val_mious = [], [], []

    eval_interval = int(tcfg.get("eval_interval", 1))
    save_interval = int(tcfg.get("save_interval", 10))

    # ── Main loop ─────────────────────────────────────────────────────────────
    for epoch in range(start_epoch, epochs):
        # Train
        avg_loss = train_one_epoch(
            model, loaders["train"], criterion, optimizer, scheduler,
            scaler, device, epoch, cfg, ema
        )
        train_losses.append(avg_loss)
        logger.info(f"Epoch {epoch:3d}  train_loss={avg_loss:.4f}")

        # Validate
        val_results = {}
        if (epoch + 1) % eval_interval == 0 or epoch == epochs - 1:
            eval_model = ema.model if ema else model
            val_results = evaluate(eval_model, loaders["val"], criterion,
                                   device, cfg, save_dir, epoch, "val")
            miou  = val_results["miou"]
            vdice = val_results.get("mean_dice", 0.0)
            voa   = val_results.get("oa", 0.0)
            vkap  = val_results.get("kappa", 0.0)
            vloss = val_results.get("val_loss", 0.0)
            val_losses.append(vloss)
            val_mious.append(miou)

            logger.info(
                f"  → val mIoU={miou*100:.2f}%  Dice={vdice*100:.2f}%  "
                f"OA={voa*100:.2f}%  κ={vkap:.4f}"
            )

            # CSV log
            with open(log_csv, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=csv_fields).writerow({
                    "epoch":      epoch,
                    "train_loss": round(avg_loss, 6),
                    "val_loss":   round(vloss, 6),
                    "val_miou":   round(miou, 6),
                    "val_dice":   round(vdice, 6),
                    "val_oa":     round(voa, 6),
                    "val_kappa":  round(vkap, 6),
                })

            # Best model checkpoint
            if miou > best_miou:
                best_miou = miou
                save_checkpoint(
                    {"epoch": epoch, "model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "scheduler": scheduler.state_dict(),
                     "best_miou": best_miou},
                    os.path.join(save_dir, "best.pth"),
                )

        # Periodic checkpoint
        if (epoch + 1) % save_interval == 0:
            save_checkpoint(
                {"epoch": epoch, "model": model.state_dict(),
                 "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(),
                 "best_miou": best_miou},
                os.path.join(save_dir, f"epoch_{epoch:04d}.pth"),
            )

        # Update training curves plot every eval cycle
        if len(val_mious) > 0:
            plot_loss_and_metric(
                train_losses, val_losses, val_mious,
                save_dir, model_name
            )

    logger.info(
        f"Training complete — best val mIoU: {best_miou*100:.2f}%"
    )


if __name__ == "__main__":
    main()
