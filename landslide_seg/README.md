# Landslide Semantic Segmentation Framework

A complete deep-learning framework for **landslide detection via semantic segmentation** of remote-sensing imagery, featuring two novel architectures and four comparison baselines, built on PyTorch.

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
│   ├── mlformerv2.py        # ★★★ Best: CVPR'25 inspired (LocalVSS+PGCSA+DOABM+IBR)
│   ├── mlformer.py          # ★★  V1: VSS + FSSF + DBAM + CAPD
│   ├── lsformer.py          # ★   V0: Dual CNN-Transformer + MSCAF + EGBR
│   └── baselines.py         # UNet / DeepLabV3+ / SegFormer / HRNet
├── utils/
│   ├── losses.py            # CE + Dice + Focal + Boundary composite loss
│   ├── metrics.py           # mIoU / Dice / OA / Kappa confusion-matrix metrics
│   └── visualize.py         # Training curves / confusion matrix / comparison chart
├── configs/
│   ├── mlformerv2.yaml      # MLFormerV2 config (recommended)
│   ├── mlformer.yaml        # MLFormerV1 config
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
# ★★★ Proposed MLFormerV2 (best — CVPR 2025 based)
python train.py --model mlformerv2 --config configs/mlformerv2.yaml

# ★★ Proposed MLFormer (V1)
python train.py --model mlformer --config configs/mlformer.yaml

# ★ Proposed LSFormer (V0)
python train.py --model lsformer --config configs/default.yaml

# Baselines (for comparison study)
python train.py --model unet          --config configs/comparison.yaml
python train.py --model deeplabv3plus --config configs/comparison.yaml
python train.py --model segformer     --config configs/comparison.yaml
python train.py --model hrnet         --config configs/comparison.yaml

# Override config values on the command line
python train.py --model mlformerv2 --epochs 100 --batch_size 8 --lr 6e-5

# Resume training
python train.py --model mlformerv2 \
    --checkpoint work_dirs/mlformerv2/epoch_0049.pth
```

Checkpoints, training logs (`train_log.csv`), and curve plots are saved under `work_dirs/<model_name>/`.

---

### 4. Evaluate

```bash
# Single model evaluation on test split
python eval.py single \
    --model      mlformerv2 \
    --checkpoint work_dirs/mlformerv2/best.pth \
    --config     configs/mlformerv2.yaml \
    --tta           # optional: multi-scale + flip TTA

# Compare all models (produces bar chart + per-model confusion matrices)
python eval.py compare \
    --compare \
      mlformerv2:work_dirs/mlformerv2/best.pth \
      mlformer:work_dirs/mlformer/best.pth \
      lsformer:work_dirs/lsformer/best.pth \
      unet:work_dirs/unet/best.pth \
      deeplabv3plus:work_dirs/deeplabv3plus/best.pth \
      segformer:work_dirs/segformer/best.pth \
      hrnet:work_dirs/hrnet/best.pth \
    --config     configs/comparison.yaml \
    --output_dir ./eval_results
```

---

### 5. Predict

```bash
# Single image
python predict.py \
    --model      mlformerv2 \
    --checkpoint work_dirs/mlformerv2/best.pth \
    --config     configs/mlformerv2.yaml \
    --image      /path/to/aerial_image.jpg \
    --output     prediction.png

# Batch
python predict.py \
    --model      mlformerv2 \
    --checkpoint work_dirs/mlformerv2/best.pth \
    --config     configs/mlformerv2.yaml \
    --image_dir  /path/to/images \
    --output_dir ./predictions
```

---

## ★★★ MLFormerV2 — Flagship Architecture (CVPR 2025 Based)

**MLFormerV2** improves MLFormerV1 on four axes, each addressing a specific limitation with a technique grounded in a 2024-2025 CVPR/top-venue paper.

```
Input Image (B, 3, H, W)
        │
   ┌────┴────┐
   │         │
CNN Branch  LocalVSSEncoder         ← [A] Window-partitioned 4-dir scan
(ResNet-50) (ws=8, global-dw gate)       LocalMamba 2024 + MambaVision CVPR'25
   │         │
   └────┬────┘
        │
   PGCSAModule                      ← [B] Prototype-guided cross-scale attention
   (K=8 learned prototypes)              MaskDINO CVPR'23 + SAM2 CVPR'25
        │
   DOABModule                       ← [C] Orientation-aware boundary deformation
   (boundary + sinθ/cosθ + deform)       DSCNet ICCV'23 + OrientedRepPoints CVPR'22
        │
   CAPDecoder                            (unchanged from V1 — CARAFE+Mask2Former)
        │
   IBRHead                          ← [D] Iterative boundary refinement
   (top-N uncertain pixels + MLP)        PointRend CVPR'20 + SAM2 CVPR'25
        │
   Output + Aux Heads (training)
```

### [A] LocalVSSBlock — Window-Partitioned Selective Scan

| Aspect | MLFormerV1 | MLFormerV2 |
|--------|-----------|-----------|
| GRU sequence length | O(W) or O(H) per row/col | Fixed ws=8 (window-local) |
| Cross-window context | None | 7×7 global depthwise conv |
| Balance mechanism | Fixed | Learnable scalar gate α (tanh) |
| Memory at 128×128 feat | GRU states ∝ H or W | GRU states ∝ ws=8 (constant) |

**Key insight**: Long GRU sequences over full rows/columns lose locality for high-resolution features. Window partitioning bounds the sequence length to ws=8, while the parallel global depthwise conv ensures spatial information propagates across windows.

### [B] PGCSAModule — Prototype-Guided Fusion

| Aspect | FSSF (V1) | PGCSA (V2) |
|--------|-----------|-----------|
| Gate signal | FFT magnitude statistics | K=8 learned prototype vectors |
| Attention | Fixed frequency bottleneck | Dynamic cross-attention (K × 64 tokens) |
| Semantic structure | None | Prototypes capture landslide archetypes |

**Key insight**: FFT gates are computed from fixed spectral statistics. Prototype cross-attention allows the model to build a semantic dictionary of landslide appearance patterns, making the fusion gate context-sensitive.

### [C] DOABModule — Orientation-Aware Boundary Deformation

Extends DBAM with explicit boundary orientation (sin θ, cos θ). Offsets are decomposed into:
```
dx = s_along · cos θ + s_across · (−sin θ)   ← along boundary tangent
dy = s_along · sin θ + s_across ·    cos θ   ← across boundary normal
```
where s_along and s_across are predicted scalars. This forces the deformable sampling to be **boundary-aligned**, capturing both boundary texture (along) and landslide-vs-background transition (across).

### [D] IBRHead — Iterative Boundary Refinement

Corrects the coarse decoder output at the N=196 most uncertain boundary pixels:
1. Compute pixel-wise prediction entropy from softmax probabilities
2. Select top-N uncertain pixels (boundary pixels have highest entropy)
3. Sample stride-4 fine features at those locations via `grid_sample`
4. Concatenate: `[fine_features ∥ coarse_logits]` at N points
5. 3-layer MLP → refined class logits for those N points
6. Scatter refined predictions back into the output map

---

## ★★ MLFormer (V1) — Previous Best Architecture

**MLFormer (Mamba-Land Segmentation Transformer)** combining four innovations from CVPR 2024-2025:

| Component | Inspired by | Novel Contribution |
|-----------|-------------|-------------------|
| **VSS Encoder** | VMamba CVPR'24, MambaVision CVPR'25 | Full-image 4-dir GRU scan; O(N) vs O(N²) |
| **FSSF Fusion** | FDA CVPR'20, FocalNet NeurIPS'22 | FFT-guided CNN-SSM cross-branch fusion |
| **DBAM** | DSCNet ICCV'23, DCNv3 CVPR'23 | Boundary-conditioned deformable sampling |
| **CAPD** | CARAFE ICCV'19, Mask2Former CVPR'22 | Content-adaptive CARAFE upsampling |

---

## ★ LSFormer (V0) — Original Proposed Architecture

**LSFormer** is our first proposed model (dual CNN+Transformer with MSCAF + EGBR).

---

## 📊 Model Comparison Table

| Model | Venue | Key Innovation | Trainable |
|-------|-------|----------------|-----------|
| **MLFormerV2** ★★★ | Proposed (CVPR'25 based) | LocalVSS + PGCSA + DOABM + IBR | `mlformerv2` |
| **MLFormer** ★★ | Proposed (CVPR'24 based) | VSS + FSSF + DBAM + CAPD | `mlformer` |
| **LSFormer** ★ | Proposed | Dual CNN-Trans + MSCAF + EGBR | `lsformer` |
| **SegFormer** | NeurIPS 2021 | Pure transformer encoder + MLP head | `segformer` |
| **HRNet** | TPAMI 2020 | High-resolution parallel streams | `hrnet` |
| **DeepLabV3+** | ECCV 2018 | ASPP + low-level feature fusion | `deeplabv3plus` |
| **UNet** | MICCAI 2015 | Encoder-decoder with skip connections | `unet` |

---

## 📚 Paper References

All innovations in this framework are grounded in peer-reviewed publications:

### MLFormerV2 Core References (CVPR 2025 based)

| # | Reference |
|---|-----------|
| [1] | **LocalMamba**: Huang W et al. "LocalMamba: Visual State Space Model with Windowed Selective Scan." arXiv:2403.09338, 2024. |
| [2] | **MambaVision**: Hatamizadeh A, Kautz J. "MambaVision: A Hybrid Mamba-Transformer Vision Backbone." *CVPR*, 2025. |
| [3] | **SAM 2**: Ravi N et al. "SAM 2: Segment Anything in Images and Videos." *CVPR*, 2025. |
| [4] | **MaskDINO**: Li F et al. "MaskDINO: Towards a Unified Transformer-based Framework for Object and Instance Segmentation." *CVPR*, 2023. |
| [5] | **DSCNet**: Qi J et al. "Dynamic Snake Convolution based on Topological Geometric Constraints for Tubular Structure Segmentation." *ICCV*, 2023. |
| [6] | **Oriented RepPoints**: Li W et al. "Oriented RepPoints for Aerial Object Detection." *CVPR*, 2022. |
| [7] | **PointRend**: Kirillov A et al. "PointRend: Image Segmentation as Rendering." *CVPR*, 2020. |
| [8] | **Slot Attention**: Locatello F et al. "Object-Centric Learning with Slot Attention." *NeurIPS*, 2020. |
| [9] | **Swin Transformer**: Liu Z et al. "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows." *ICCV*, 2021. |

### MLFormerV1 Core References

| # | Reference |
|---|-----------|
| [10] | **VMamba**: Liu Y et al. "VMamba: Visual State Space Model." *CVPR*, 2024. |
| [11] | **FDA**: Yang Y et al. "FDA: Fourier Domain Adaptation for Semantic Segmentation." *CVPR*, 2020. |
| [12] | **FocalNet**: Yang J et al. "Focal Modulation Networks." *NeurIPS*, 2022. |
| [13] | **InternImage/DCNv3**: Wang W et al. "InternImage: Exploring Large-Scale Vision Foundation Models with Deformable Convolutions." *CVPR*, 2023. |
| [14] | **CARAFE**: Wang J et al. "CARAFE: Content-Aware ReAssembly of FEatures." *ICCV*, 2019. |
| [15] | **Mask2Former**: Cheng B et al. "Masked-attention Mask Transformer for Universal Image Segmentation." *CVPR*, 2022. |

### Baselines

| # | Reference |
|---|-----------|
| [16] | **UNet**: Ronneberger O et al. "U-Net: Convolutional Networks for Biomedical Image Segmentation." *MICCAI*, 2015. |
| [17] | **DeepLabV3+**: Chen L-C et al. "Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation." *ECCV*, 2018. |
| [18] | **SegFormer**: Xie E et al. "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers." *NeurIPS*, 2021. |
| [19] | **HRNet**: Wang J et al. "Deep High-Resolution Representation Learning for Visual Recognition." *TPAMI*, 2020. |

---

## ⚙️ Configuration

Edit `configs/mlformerv2.yaml` (recommended) or `configs/default.yaml`.  
All fields can be overridden via CLI (`--epochs`, `--batch_size`, `--lr`, `--save_dir`).

Key settings:

```yaml
model:
  name: mlformerv2
  window_size: 8          # LocalVSS window size
  num_prototypes: 8       # PGCSA prototype count K
  num_ibr_points: 196     # IBRHead refinement points N

loss:
  w_ce:    1.0
  w_dice:  0.5
  w_focal: 0.5
  w_aux:   0.4
  w_bnd:   0.3

optimizer:
  type: adamw
  lr:   6.0e-5
  weight_decay: 0.01
```

