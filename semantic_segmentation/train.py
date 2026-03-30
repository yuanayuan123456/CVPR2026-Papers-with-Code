"""Training script for semantic segmentation.

Usage::

    python train.py --config configs/default.yaml [--resume path/to/ckpt.pth]
"""

import argparse
import logging
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

import yaml

# Add project root to path so relative imports work when called directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from semantic_segmentation.models import Segmentor
from semantic_segmentation.datasets import ADE20K, Cityscapes, VOCSegmentation
from semantic_segmentation.utils import (
    SegmentationLoss, SegmentationMetric,
    build_train_transform, build_val_transform,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_dataset(cfg: dict, split: str):
    name      = cfg["dataset"]["name"]
    root      = cfg["dataset"]["root"]
    ignore    = cfg["dataset"].get("ignore_index", 255)
    crop_size = cfg["dataset"].get("crop_size", 512)
    val_size  = cfg["dataset"].get("val_size",  512)
    scales    = cfg["dataset"].get("scales", None)

    if split == "train":
        transform = build_train_transform(crop_size, scales, ignore)
    else:
        transform = build_val_transform(val_size)

    datasets = {
        "cityscapes": Cityscapes,
        "ade20k":     ADE20K,
        "voc":        VOCSegmentation,
    }
    assert name in datasets, f"Unknown dataset '{name}'"
    return datasets[name](root=root, split=split, transform=transform,
                          ignore_index=ignore)


def build_model(cfg: dict) -> nn.Module:
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


def build_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    ocfg = cfg["optimizer"]
    otype = ocfg.get("type", "sgd").lower()
    lr   = float(ocfg.get("lr", 0.01))
    wd   = float(ocfg.get("weight_decay", 1e-4))

    # Separate backbone (lower LR) from head
    backbone_params = list(model.backbone.parameters())
    backbone_ids    = set(id(p) for p in backbone_params)
    head_params     = [p for p in model.parameters() if id(p) not in backbone_ids]

    param_groups = [
        {"params": backbone_params, "lr": lr * 0.1},
        {"params": head_params,     "lr": lr},
    ]

    if otype == "sgd":
        return torch.optim.SGD(
            param_groups,
            momentum=float(ocfg.get("momentum", 0.9)),
            weight_decay=wd,
            nesterov=bool(ocfg.get("nesterov", True)),
        )
    elif otype == "adamw":
        return torch.optim.AdamW(param_groups, weight_decay=wd)
    else:
        raise ValueError(f"Unknown optimizer type: {otype}")


def build_scheduler(optimizer, cfg: dict, total_iters: int):
    scfg = cfg.get("scheduler", {})
    stype = scfg.get("type", "poly").lower()

    if stype == "poly":
        power = float(scfg.get("power", 0.9))
        def poly_fn(i):
            return (1 - i / total_iters) ** power
        return torch.optim.lr_scheduler.LambdaLR(optimizer, poly_fn)
    elif stype == "cosine":
        min_lr = float(scfg.get("min_lr", 1e-5))
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_iters, eta_min=min_lr)
    elif stype == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(scfg.get("step_size", 30)),
            gamma=float(scfg.get("gamma", 0.1)),
        )
    else:
        raise ValueError(f"Unknown scheduler type: {stype}")


def save_checkpoint(state: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    logger.info(f"Checkpoint saved → {path}")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, scheduler,
                    scaler, device, epoch, cfg):
    model.train()
    log_interval = cfg["output"].get("log_interval", 50)
    grad_clip    = float(cfg["train"].get("grad_clip", 0.0))
    use_amp      = bool(cfg["train"].get("amp", False))

    total_loss = 0.0
    t0 = time.time()
    for i, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
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
        total_loss += loss.item()

        if (i + 1) % log_interval == 0:
            avg_loss = total_loss / (i + 1)
            lr_now   = optimizer.param_groups[-1]["lr"]
            elapsed  = time.time() - t0
            logger.info(
                f"Epoch {epoch} [{i+1}/{len(loader)}]  "
                f"loss={avg_loss:.4f}  lr={lr_now:.6f}  "
                f"time={elapsed:.1f}s"
            )

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, num_classes, ignore_index, device):
    model.eval()
    metric = SegmentationMetric(num_classes, ignore_index)
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        preds   = outputs["out"].argmax(dim=1).cpu()
        metric.update(preds, labels)
    return metric.compute()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Semantic Segmentation Training")
    parser.add_argument("--config",  default="configs/default.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--resume",  default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--device",  default="cuda",
                        help="Device: 'cuda' or 'cpu'")
    args = parser.parse_args()

    cfg = load_config(args.config)
    save_dir = cfg["output"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # ── Datasets & loaders ──────────────────────────────────────────────────
    train_ds = build_dataset(cfg, split="train")
    val_ds   = build_dataset(cfg, split="val")

    tcfg = cfg["train"]
    train_loader = DataLoader(
        train_ds,
        batch_size=tcfg["batch_size"],
        shuffle=True,
        num_workers=tcfg.get("num_workers", 4),
        pin_memory=tcfg.get("pin_memory", True),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=tcfg.get("val_batch_size", tcfg["batch_size"]),
        shuffle=False,
        num_workers=tcfg.get("num_workers", 4),
        pin_memory=tcfg.get("pin_memory", True),
    )

    # ── Model ────────────────────────────────────────────────────────────────
    model = build_model(cfg).to(device)
    logger.info(f"Model: {cfg['model']['backbone']} + {cfg['model']['head']}")

    # ── Loss ─────────────────────────────────────────────────────────────────
    lcfg      = cfg.get("loss", {})
    criterion = SegmentationLoss(
        num_classes=cfg["dataset"]["num_classes"],
        ignore_index=cfg["dataset"].get("ignore_index", 255),
        use_dice=lcfg.get("use_dice", False),
        use_ohem=lcfg.get("use_ohem", False),
        aux_weight=float(lcfg.get("aux_weight", 0.4)),
        dice_weight=float(lcfg.get("dice_weight", 0.5)),
    )

    # ── Optimizer / Scheduler ────────────────────────────────────────────────
    epochs       = int(tcfg["epochs"])
    total_iters  = epochs * len(train_loader)
    optimizer    = build_optimizer(model, cfg)
    scheduler    = build_scheduler(optimizer, cfg, total_iters)
    scaler       = GradScaler(enabled=bool(tcfg.get("amp", False)))

    # ── Resume ───────────────────────────────────────────────────────────────
    start_epoch = 0
    best_miou   = 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_miou   = ckpt.get("best_miou", 0.0)
        logger.info(f"Resumed from epoch {start_epoch - 1} (best mIoU={best_miou:.4f})")

    # ── Training loop ────────────────────────────────────────────────────────
    eval_interval = int(tcfg.get("eval_interval", 5))
    save_interval = int(tcfg.get("save_interval", 10))
    num_classes   = cfg["dataset"]["num_classes"]
    ignore_index  = cfg["dataset"].get("ignore_index", 255)

    for epoch in range(start_epoch, epochs):
        avg_loss = train_one_epoch(model, train_loader, criterion,
                                   optimizer, scheduler, scaler,
                                   device, epoch, cfg)
        logger.info(f"Epoch {epoch} finished — avg loss: {avg_loss:.4f}")

        # Periodic evaluation
        if (epoch + 1) % eval_interval == 0 or epoch == epochs - 1:
            results = evaluate(model, val_loader, num_classes, ignore_index, device)
            miou    = results["miou"]
            pacc    = results["pixel_acc"]
            logger.info(
                f"[Eval] Epoch {epoch}  mIoU={miou*100:.2f}%  "
                f"PixAcc={pacc*100:.2f}%"
            )
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
                os.path.join(save_dir, f"epoch_{epoch}.pth"),
            )

    logger.info(f"Training complete. Best mIoU: {best_miou*100:.2f}%")


if __name__ == "__main__":
    main()
