# Landslide Semantic Segmentation Framework

A complete deep-learning framework for **landslide detection via semantic segmentation** of remote-sensing imagery, featuring a novel architecture (LSFormer) and four comparison baselines, built on PyTorch.

---

## 📁 Project Structure

```
landslide_seg/
├── data/
│   ├── labelme2mask.py      # LabelMe JSON → mask PNG converter
│   ├── dataset.py           # PyTorch Dataset + auto train/val/test split
│   └── augmentation.py      # Geometric + radiometric augmentation pipeline
├── models/
│   ├── __init__.py          # get_model() factory (select by name)
│   ├── lsformer.py          # ★ Proposed: LandSlide Segmentation Transformer
│   └── baselines.py         # UNet / DeepLabV3+ / SegFormer / HRNet
├── utils/
│   ├── losses.py            # CE + Dice + Focal + Boundary composite loss
│   ├── metrics.py           # mIoU / Dice / OA / Kappa confusion-matrix metrics
│   └── visualize.py         # Training curves / confusion matrix / comparison chart
├── configs/
│   ├── default.yaml         # Full default configuration
│   └── comparison.yaml      # Same-setting config for fair comparison
├── train.py                 # Training script (--model flag selects architecture)
├── eval.py                  # Evaluation & multi-model comparison
├── predict.py               # Single-image / batch inference
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r landslide_seg/requirements.txt
```

### 2. Prepare data

**Step A — Convert LabelMe JSON annotations to mask images**

```bash
# Binary segmentation (landslide=1, background=0)
python -m landslide_seg.data.labelme2mask \
    --input_dir  ./raw_annotations \
    --mask_dir   ./data/masks \
    --image_dir  ./data/images \
    --class_map  landslide:1 \
    --suffix     .png

# Multi-class
python -m landslide_seg.data.labelme2mask \
    --input_dir  ./raw_annotations \
    --mask_dir   ./data/masks \
    --class_map  landslide:1 debris:2 rockfall:3
```

**Step B — Verify your directory layout**

```
data/
  images/   *.jpg   (or .png)
  masks/    *.png   (integer label, 0=background, 1=landslide, 255=ignore)
```

The dataset is automatically split into train/val/test (70/15/15 by default).

---

### 3. Train

```bash
cd landslide_seg  # or run from project root

# Proposed LSFormer
python train.py --model lsformer --config configs/default.yaml

# Baselines (for comparison study)
python train.py --model unet          --config configs/comparison.yaml
python train.py --model deeplabv3plus --config configs/comparison.yaml
python train.py --model segformer     --config configs/comparison.yaml
python train.py --model hrnet         --config configs/comparison.yaml

# Override config values on the command line
python train.py --model lsformer --epochs 80 --batch_size 4 --lr 3e-5

# Resume training
python train.py --model lsformer \
    --checkpoint work_dirs/comparison/lsformer/epoch_0049.pth
```

Checkpoints, training logs (`train_log.csv`), and curve plots are saved under `work_dirs/<model_name>/`.

---

### 4. Evaluate

```bash
# Single model evaluation on test split
python eval.py single \
    --model      lsformer \
    --checkpoint work_dirs/comparison/lsformer/best.pth \
    --config     configs/default.yaml \
    --tta           # optional: multi-scale + flip TTA

# Compare all models (produces bar chart + per-model confusion matrices)
python eval.py compare \
    --compare \
      lsformer:work_dirs/comparison/lsformer/best.pth \
      unet:work_dirs/comparison/unet/best.pth \
      deeplabv3plus:work_dirs/comparison/deeplabv3plus/best.pth \
      segformer:work_dirs/comparison/segformer/best.pth \
      hrnet:work_dirs/comparison/hrnet/best.pth \
    --config     configs/comparison.yaml \
    --output_dir ./eval_results
```

---

### 5. Predict

```bash
# Single image
python predict.py \
    --model      lsformer \
    --checkpoint work_dirs/comparison/lsformer/best.pth \
    --config     configs/default.yaml \
    --image      /path/to/aerial_image.jpg \
    --output     prediction.png

# Batch
python predict.py \
    --model      lsformer \
    --checkpoint work_dirs/comparison/lsformer/best.pth \
    --config     configs/default.yaml \
    --image_dir  /path/to/images \
    --output_dir ./predictions
```

---

## ★ LSFormer — Novel Architecture

**LandSlide Segmentation Transformer** is designed to address the key challenges of landslide detection from remote-sensing imagery:

```
Input Image (B, 3, H, W)
        │
   ┌────┴────┐
   │         │
CNN Branch  Transformer Branch
(ResNet-50) (Efficient Multi-Scale Transformer)
   │         │
   └────┬────┘
        │
   MSCAF Module
   Multi-Scale Cross-Attention Fusion
   (Bidirectional cross-attention at 4 scales
    + adaptive gating)
        │
   EGBR Module
   Edge-Guided Boundary Refinement
   (Learnable edge detector → boundary attention)
        │
   HAD — Hierarchical Adaptive Decoder
   (FPN + spatial attention gates)
        │
   Output + Auxiliary Heads (deep supervision)
```

### Key Innovations

| Component | Innovation | Challenge Addressed |
|-----------|-----------|-------------------|
| **Dual-Path Encoder** | CNN + Transformer in parallel | Scale variation, local+global feature extraction |
| **MSCAF** | Bidirectional cross-attention fusion at all 4 scales with adaptive gating | Complementary CNN texture and Transformer context |
| **EGBR** | Learnable boundary detector + boundary attention maps | Ambiguous landslide edges |
| **HAD** | Spatial attention gates on skip connections | Background clutter in decoder skip paths |
| **Multi-task loss** | CE + Dice + Focal + boundary auxiliary + deep supervision | Class imbalance, gradient flow |

---

## 📊 Comparison Models

| Model | Year | Key Feature |
|-------|------|-------------|
| **UNet** | 2015 | Encoder-decoder with skip connections |
| **DeepLabV3+** | 2018 | ASPP + low-level feature fusion |
| **SegFormer** | 2021 | Pure transformer encoder + lightweight MLP head |
| **HRNet** | 2020 | High-resolution feature maintenance |
| **LSFormer** ★ | Proposed | Dual CNN-Transformer + MSCAF + EGBR |

---

## ⚙️ Configuration

Edit `configs/default.yaml` or `configs/comparison.yaml`.  
All fields can be overridden via CLI (`--epochs`, `--batch_size`, `--lr`, `--save_dir`).

Key settings:

```yaml
dataset:
  image_dir:   ./data/images
  mask_dir:    ./data/masks
  num_classes: 2          # binary: background / landslide
  crop_size:   512
  use_elastic: true       # elastic deformation augmentation

model:
  name: lsformer          # lsformer | unet | deeplabv3plus | segformer | hrnet
  pretrained: true

loss:
  w_ce:    1.0
  w_dice:  0.5
  w_focal: 0.5
  w_aux:   0.4
  w_bnd:   0.3

optimizer:
  type: adamw
  lr:   6.0e-5

scheduler:
  type: poly
  warmup_epochs: 5
```

---

## 📈 Training Outputs

Each run saves in `work_dirs/<model_name>/`:

```
work_dirs/lsformer/
  config.yaml                  resolved configuration
  train_log.csv                epoch metrics (loss, mIoU, Dice, OA, Kappa)
  best.pth                     best validation checkpoint
  lsformer_training_curves.png loss + mIoU curves
  val_grid_ep0099.png          qualitative prediction grid
```

---

## 🔬 Loss Functions

| Loss | Weight | Purpose |
|------|--------|---------|
| Cross-Entropy (label smooth 0.05) | 1.0 | Pixel classification |
| Soft Dice | 0.5 | Overlap maximisation, robust to class imbalance |
| Focal Loss (γ=2) | 0.5 | Hard-example mining (small/subtle landslides) |
| Auxiliary CE (deep supervision) | 0.4 / 0.2 | Gradient flow in encoder |
| Boundary Loss | 0.3 | Precise landslide edge delineation |

---

## 📐 Evaluation Metrics

- **mIoU** — Mean Intersection over Union
- **Mean Dice / F1** — Harmonic mean of precision and recall
- **OA** — Overall Accuracy (pixel accuracy)
- **κ (Kappa)** — Cohen's Kappa (chance-corrected accuracy)
- **Precision / Recall** — Per class and mean

---

## 📄 Citation

If you use LSFormer in your research, please cite:

```bibtex
@article{lsformer2026,
  title   = {LSFormer: A Dual-Path Cross-Attention Transformer for
             Landslide Semantic Segmentation in Remote Sensing Imagery},
  author  = {Your Name},
  journal = {IEEE Transactions on Geoscience and Remote Sensing},
  year    = {2026},
}
```
