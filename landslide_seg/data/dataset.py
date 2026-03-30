"""
PyTorch Dataset for landslide semantic segmentation.

Directory layout expected:
    root/
      images/   *.jpg or *.png   (RGB remote-sensing images)
      masks/    *.png             (integer label masks, same stem)

A train/val/test split can be provided as a plain text file listing stems
(one per line), or the dataset automatically splits the files.
"""

import os
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from .augmentation import TrainTransform, ValTransform


# ─────────────────────────── helpers ─────────────────────────────────────────

IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def _find_images(directory: str) -> List[Path]:
    directory = Path(directory)
    files = sorted(
        p for p in directory.iterdir()
        if p.suffix.lower() in IMG_EXTS
    )
    return files


def auto_split(stems: List[str],
               train_ratio: float = 0.7,
               val_ratio:   float = 0.15,
               seed: int = 42) -> Tuple[List[str], List[str], List[str]]:
    """Randomly split ``stems`` into (train, val, test) lists."""
    random.seed(seed)
    shuffled = stems.copy()
    random.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    train = shuffled[:n_train]
    val   = shuffled[n_train:n_train + n_val]
    test  = shuffled[n_train + n_val:]
    return train, val, test


# ─────────────────────────── dataset ─────────────────────────────────────────

class LandslideDataset(Dataset):
    """Landslide segmentation dataset.

    Args:
        image_dir: Directory of input RGB images.
        mask_dir:  Directory of corresponding integer-label PNG masks.
        stems:     List of file name stems (without extension) to include.
                   If ``None``, all images found in ``image_dir`` are used.
        transform: Joint image-mask transform.  If ``None``, images are
                   returned as raw NumPy arrays (uint8 / uint8).
        img_suffix: Image file extension (default ``'.jpg'``).
        mask_suffix: Mask file extension (default ``'.png'``).
        num_classes: Number of segmentation classes.
        ignore_index: Void label used in mask.
    """

    def __init__(
        self,
        image_dir: str,
        mask_dir:  str,
        stems: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
        img_suffix:  str = ".jpg",
        mask_suffix: str = ".png",
        num_classes: int = 2,
        ignore_index: int = 255,
    ) -> None:
        self.image_dir    = Path(image_dir)
        self.mask_dir     = Path(mask_dir)
        self.transform    = transform
        self.img_suffix   = img_suffix
        self.mask_suffix  = mask_suffix
        self.num_classes  = num_classes
        self.ignore_index = ignore_index

        if stems is not None:
            self.stems = stems
        else:
            # Discover from image_dir and verify mask exists
            self.stems = []
            for p in sorted(self.image_dir.iterdir()):
                if p.suffix.lower() in IMG_EXTS:
                    mpath = self.mask_dir / (p.stem + self.mask_suffix)
                    if mpath.exists():
                        self.stems.append(p.stem)

        if len(self.stems) == 0:
            raise RuntimeError(
                f"No valid image-mask pairs found in "
                f"{image_dir} / {mask_dir}"
            )

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int):
        stem = self.stems[idx]
        img_path  = self.image_dir / (stem + self.img_suffix)
        mask_path = self.mask_dir  / (stem + self.mask_suffix)

        # Fallback: try common extensions if default not found
        if not img_path.exists():
            for ext in IMG_EXTS:
                candidate = self.image_dir / (stem + ext)
                if candidate.exists():
                    img_path = candidate
                    break

        image = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        mask  = np.array(Image.open(mask_path),                dtype=np.uint8)

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        return image, mask

    # ------------------------------------------------------------------
    def compute_class_weights(self,
                              smooth: float = 1e-6) -> np.ndarray:
        """Compute inverse-frequency class weights for imbalanced segmentation."""
        counts = np.zeros(self.num_classes, dtype=np.float64)
        for stem in self.stems:
            mask_path = self.mask_dir / (stem + self.mask_suffix)
            mask = np.array(Image.open(mask_path), dtype=np.uint8)
            for c in range(self.num_classes):
                counts[c] += (mask == c).sum()
        total  = counts.sum()
        freqs  = (counts + smooth) / (total + smooth)
        weights = 1.0 / freqs
        weights /= weights.sum()  # normalise
        return weights.astype(np.float32)


# ─────────────────────────── factory ─────────────────────────────────────────

def build_datasets(
    image_dir:   str,
    mask_dir:    str,
    crop_size:   int = 512,
    val_size:    int = 512,
    num_classes: int = 2,
    ignore_index: int = 255,
    train_ratio: float = 0.7,
    val_ratio:   float = 0.15,
    split_file:  Optional[str] = None,
    img_suffix:  str = ".jpg",
    mask_suffix: str = ".png",
    use_elastic: bool = True,
    use_cutmix:  bool = False,
    seed: int = 42,
) -> Dict[str, LandslideDataset]:
    """Build train / val / test ``LandslideDataset`` instances.

    Args:
        image_dir:   Directory of images.
        mask_dir:    Directory of masks.
        crop_size:   Training random-crop size.
        val_size:    Validation resize target.
        num_classes: Number of segmentation classes.
        ignore_index: Void label value.
        train_ratio: Fraction of data used for training (auto-split mode).
        val_ratio:   Fraction used for validation.
        split_file:  Path to a JSON file ``{"train": [...], "val": [...], "test": [...]}``
                     listing stems per split.  Overrides ``train_ratio``.
        img_suffix:  Image extension.
        mask_suffix: Mask extension.
        use_elastic: Enable elastic deformation augmentation.
        use_cutmix:  Enable CutMix augmentation.
        seed:        Random seed for auto-split.

    Returns:
        Dict with keys ``'train'``, ``'val'``, ``'test'``.
    """
    # Discover all stems
    image_dir_path = Path(image_dir)
    all_stems = sorted(
        p.stem for p in image_dir_path.iterdir()
        if p.suffix.lower() in IMG_EXTS
           and (Path(mask_dir) / (p.stem + mask_suffix)).exists()
    )

    if split_file is not None and os.path.exists(split_file):
        import json
        with open(split_file) as f:
            splits = json.load(f)
        train_stems = splits.get("train", [])
        val_stems   = splits.get("val",   [])
        test_stems  = splits.get("test",  [])
    else:
        train_stems, val_stems, test_stems = auto_split(
            all_stems, train_ratio, val_ratio, seed
        )

    train_tf = TrainTransform(crop_size=crop_size, ignore_index=ignore_index,
                              use_elastic=use_elastic, use_cutmix=use_cutmix)
    val_tf   = ValTransform(size=val_size)

    datasets = {
        "train": LandslideDataset(image_dir, mask_dir, train_stems,
                                   train_tf, img_suffix, mask_suffix,
                                   num_classes, ignore_index),
        "val":   LandslideDataset(image_dir, mask_dir, val_stems,
                                   val_tf, img_suffix, mask_suffix,
                                   num_classes, ignore_index),
        "test":  LandslideDataset(image_dir, mask_dir, test_stems,
                                   val_tf, img_suffix, mask_suffix,
                                   num_classes, ignore_index),
    }
    return datasets


def build_dataloaders(
    datasets: Dict[str, LandslideDataset],
    batch_size:   int = 8,
    val_batch:    int = 4,
    num_workers:  int = 4,
    pin_memory:   bool = True,
) -> Dict[str, DataLoader]:
    """Wrap dataset dict into DataLoader dict."""
    loaders = {}
    for split, ds in datasets.items():
        is_train = (split == "train")
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size if is_train else val_batch,
            shuffle=is_train,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=is_train,
        )
    return loaders


__all__ = [
    "LandslideDataset",
    "auto_split",
    "build_datasets",
    "build_dataloaders",
]
