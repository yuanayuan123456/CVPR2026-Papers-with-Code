"""
Single-image inference script.

Usage
-----
python predict.py --model lsformer \
                  --checkpoint work_dirs/lsformer/best.pth \
                  --config     configs/default.yaml \
                  --image      /path/to/image.jpg \
                  --output     prediction.png

# Batch inference on a directory
python predict.py --model lsformer \
                  --checkpoint work_dirs/lsformer/best.pth \
                  --config     configs/default.yaml \
                  --image_dir  /path/to/images \
                  --output_dir ./predictions
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from landslide_seg.models  import get_model
from landslide_seg.data.augmentation import normalize, IMAGENET_MEAN, IMAGENET_STD
from landslide_seg.utils.visualize   import (
    plot_prediction, _palette_to_rgb, DEFAULT_PALETTE
)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


# ─────────────────────────── helpers ─────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def preprocess(image_path: str, size: int = 512) -> torch.Tensor:
    """Load, resize, normalise image → float32 tensor (1, C, H, W)."""
    import cv2
    img = cv2.imread(image_path)
    if img is None:
        img = np.array(Image.open(image_path).convert("RGB"))
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    img = normalize(img, IMAGENET_MEAN, IMAGENET_STD)
    img = img.transpose(2, 0, 1)                  # C,H,W
    return torch.from_numpy(img).unsqueeze(0)      # 1,C,H,W


@torch.no_grad()
def predict_single(model, image_tensor: torch.Tensor,
                   device: torch.device,
                   tta: bool = False) -> np.ndarray:
    """Return predicted label mask (H, W)."""
    model.eval()
    image_tensor = image_tensor.to(device, dtype=torch.float32)
    if tta:
        logits = _tta_inference(model, image_tensor)
    else:
        logits = model(image_tensor)["out"]
    return logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int32)


def _tta_inference(model, images: torch.Tensor,
                   scales=(0.75, 1.0, 1.25)) -> torch.Tensor:
    H, W = images.shape[-2:]
    n_cls = model(images)["out"].shape[1]
    acc   = torch.zeros(images.size(0), n_cls, H, W, device=images.device)
    for s in scales:
        nh, nw = int(H * s), int(W * s)
        img_s  = F.interpolate(images, (nh, nw), mode="bilinear",
                               align_corners=False)
        logits = F.interpolate(model(img_s)["out"], (H, W),
                               mode="bilinear", align_corners=False)
        acc += logits.softmax(1)
        logits_f = F.interpolate(
            model(torch.flip(img_s, [-1]))["out"],
            (H, W), mode="bilinear", align_corners=False
        )
        acc += torch.flip(logits_f, [-1]).softmax(1)
    return acc


# ─────────────────────────── main ────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run single-image or batch inference"
    )
    parser.add_argument("--model",      required=True,
                        choices=["mlformer", "lsformer", "unet",
                                 "deeplabv3plus", "segformer", "hrnet"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config",     default="configs/default.yaml")
    parser.add_argument("--device",     default="cuda")
    parser.add_argument("--tta",        action="store_true")

    # Single image
    parser.add_argument("--image",  default=None)
    parser.add_argument("--output", default="prediction.png")

    # Batch
    parser.add_argument("--image_dir",  default=None)
    parser.add_argument("--output_dir", default="./predictions")

    args = parser.parse_args()

    cfg    = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    mcfg   = cfg["model"]
    size   = cfg["dataset"].get("val_size", 512)

    model = get_model(
        args.model,
        num_classes = mcfg.get("num_classes", 2),
        pretrained  = False,
        in_channels = mcfg.get("in_channels", 3),
    ).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt.get("model", ckpt))
    print(f"Loaded {args.model} from {args.checkpoint}")

    # ── Single image ─────────────────────────────────────────────────────
    if args.image:
        img_t = preprocess(args.image, size)
        pred  = predict_single(model, img_t, device, tta=args.tta)

        # Coloured prediction
        color = _palette_to_rgb(pred, DEFAULT_PALETTE)
        Image.fromarray(color).save(args.output)
        print(f"Prediction saved → {args.output}")

        # Side-by-side visualisation
        img_np = img_t.squeeze(0).numpy()
        base   = os.path.splitext(args.output)[0]
        plot_prediction(img_np, np.zeros_like(pred), pred,
                        save_path=base + "_vis.png",
                        title=f"{args.model} — {os.path.basename(args.image)}")
        print(f"Visualisation saved → {base}_vis.png")
        return

    # ── Batch inference ──────────────────────────────────────────────────
    if args.image_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        img_files = sorted(
            p for p in Path(args.image_dir).iterdir()
            if p.suffix.lower() in IMG_EXTS
        )
        print(f"Found {len(img_files)} images in {args.image_dir}")

        for img_path in img_files:
            img_t = preprocess(str(img_path), size)
            pred  = predict_single(model, img_t, device, tta=args.tta)
            color = _palette_to_rgb(pred, DEFAULT_PALETTE)
            out   = Path(args.output_dir) / (img_path.stem + "_pred.png")
            Image.fromarray(color).save(out)
            print(f"  ✓  {img_path.name}  →  {out}")
        print("Done.")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
