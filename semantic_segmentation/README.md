# Semantic Segmentation Framework

A clean, modular deep-learning semantic segmentation framework in PyTorch.  
It supports multiple backbones and decode heads, three benchmark datasets,
rich data augmentation, flexible losses, and multi-scale test-time augmentation.

---

## 📁 Project Structure

```
semantic_segmentation/
├── __init__.py
├── requirements.txt
├── train.py                    # Training entry point
├── eval.py                     # Evaluation / inference entry point
├── configs/
│   ├── default.yaml            # ResNet-50 + UPerHead on Cityscapes
│   └── segformer_b2_ade20k.yaml
├── models/
│   ├── segmentor.py            # Top-level encoder-decoder model
│   ├── backbone/
│   │   └── __init__.py         # ResNet & Mix Transformer (MiT-B0…B5)
│   └── head/
│       └── __init__.py         # UPerHead & SegFormerHead
├── datasets/
│   ├── base.py                 # Abstract base dataset
│   ├── cityscapes.py           # Cityscapes (19 classes)
│   ├── ade20k.py               # ADE20K (150 classes)
│   └── voc.py                  # Pascal VOC 2012 (21 classes)
└── utils/
    ├── losses.py               # CE, OHEM-CE, Dice, combined loss
    ├── metrics.py              # mIoU & pixel accuracy
    └── transforms.py           # Joint image-label augmentations
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r semantic_segmentation/requirements.txt
```

### 2. Prepare dataset

Download and extract your dataset (e.g. Cityscapes) and update `root` in the
config file:

```yaml
dataset:
  name: cityscapes
  root: /data/cityscapes
```

### 3. Train

```bash
python semantic_segmentation/train.py \
    --config semantic_segmentation/configs/default.yaml
```

Resume from a checkpoint:

```bash
python semantic_segmentation/train.py \
    --config semantic_segmentation/configs/default.yaml \
    --resume work_dirs/cityscapes_resnet50_uper/epoch_99.pth
```

### 4. Evaluate

```bash
python semantic_segmentation/eval.py \
    --config semantic_segmentation/configs/default.yaml \
    --checkpoint work_dirs/cityscapes_resnet50_uper/best.pth
```

Multi-scale TTA evaluation:

```bash
python semantic_segmentation/eval.py \
    --config semantic_segmentation/configs/default.yaml \
    --checkpoint work_dirs/cityscapes_resnet50_uper/best.pth \
    --multiscale
```

Single-image inference:

```bash
python semantic_segmentation/eval.py \
    --config semantic_segmentation/configs/default.yaml \
    --checkpoint work_dirs/cityscapes_resnet50_uper/best.pth \
    --image /path/to/image.jpg \
    --output prediction.png
```

---

## 🔧 Supported Configurations

### Backbones

| Name | Params | Notes |
|------|--------|-------|
| `resnet18` | 11 M | Basic block, lightweight |
| `resnet50` | 25 M | Bottleneck, good baseline |
| `resnet101` | 44 M | Higher capacity |
| `mit_b0` | 4 M | SegFormer-B0 |
| `mit_b2` | 25 M | SegFormer-B2 (recommended) |
| `mit_b5` | 82 M | SegFormer-B5 (best accuracy) |

### Decode Heads

| Name | Description |
|------|-------------|
| `uper` | UPerHead with PPM, FPN fusion — universal, pairs well with ResNet |
| `segformer` | Lightweight all-MLP head — efficient, pairs well with MiT |

### Datasets

| Name | Classes | Split key |
|------|---------|-----------|
| `cityscapes` | 19 | `train` / `val` |
| `ade20k` | 150 | `training` / `validation` |
| `voc` | 21 | `train` / `val` |

### Loss Functions

| Loss | Config key |
|------|-----------|
| Cross Entropy | default |
| OHEM Cross Entropy | `loss.use_ohem: true` |
| Dice | `loss.use_dice: true` |
| CE + Dice | `use_ohem: false, use_dice: true` |

---

## 🧩 Programmatic API

```python
from semantic_segmentation import Segmentor

# Build a model from config dict
model = Segmentor.from_config({
    "backbone": "resnet50",
    "head":     "uper",
    "num_classes": 19,
    "channels": 256,
    "aux_head": True,
})

# Forward pass (returns dict with 'out' and optionally 'aux' during training)
import torch
x = torch.randn(2, 3, 512, 512)
model.train()
outputs = model(x)
print(outputs["out"].shape)   # torch.Size([2, 19, 512, 512])
```

---

## 📊 Architecture Overview

```
Input Image (B, 3, H, W)
        │
   ┌────▼────┐
   │Backbone │  ResNet or MixTransformer
   │(4 stages│  Outputs 4 multi-scale feature maps
   └────┬────┘
        │  [C1, C2, C3, C4]  (stride 4/8/16/32)
   ┌────▼────┐
   │  Head   │  UPerHead (PPM + FPN) or SegFormerHead (MLP)
   └────┬────┘
        │  Logits (B, num_classes, H/4, W/4)
   ┌────▼─────┐
   │ Bilinear │  Upsample to original resolution
   │ Upsample │
   └────┬─────┘
        │
   Logits (B, num_classes, H, W)
```
