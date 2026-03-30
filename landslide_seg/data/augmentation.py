"""
Augmentation pipeline for landslide semantic segmentation.

Designed for remote sensing / aerial imagery; supports:
  - Strong geometric augmentations (flip, rotate, scale, elastic)
  - Radiometric augmentations (brightness, contrast, blur, noise)
  - MixUp-style CutMix for alleviating class imbalance
  - Both train and validation pipelines
"""

import random
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image


# ─────────────────────────── helpers ─────────────────────────────────────────

def _to_numpy(img) -> np.ndarray:
    if isinstance(img, Image.Image):
        return np.array(img)
    return img


def _to_pil(arr: np.ndarray, mode: Optional[str] = None) -> Image.Image:
    return Image.fromarray(arr, mode=mode)


# ─────────────────────────── geometric ───────────────────────────────────────

def random_hflip(image: np.ndarray, mask: np.ndarray,
                 p: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    if random.random() < p:
        return image[:, ::-1].copy(), mask[:, ::-1].copy()
    return image, mask


def random_vflip(image: np.ndarray, mask: np.ndarray,
                 p: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    if random.random() < p:
        return image[::-1, :].copy(), mask[::-1, :].copy()
    return image, mask


def random_rotate(image: np.ndarray, mask: np.ndarray,
                  max_angle: float = 30.0,
                  ignore_index: int = 255,
                  p: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    if random.random() >= p:
        return image, mask
    angle = random.uniform(-max_angle, max_angle)
    h, w  = image.shape[:2]
    M     = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated_img  = cv2.warpAffine(image, M, (w, h),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REFLECT_101)
    rotated_mask = cv2.warpAffine(mask, M, (w, h),
                                  flags=cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=ignore_index)
    return rotated_img, rotated_mask


def random_scale_crop(image: np.ndarray, mask: np.ndarray,
                      crop_size: int = 512,
                      scales: Tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0),
                      ignore_index: int = 255,
                      p: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Scale the image by a random factor and then crop to ``crop_size``."""
    if random.random() >= p:
        return image, mask
    scale = random.choice(scales)
    h, w  = image.shape[:2]
    nh, nw = int(h * scale), int(w * scale)

    image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    mask  = cv2.resize(mask,  (nw, nh), interpolation=cv2.INTER_NEAREST)

    # Pad if necessary
    pad_h = max(crop_size - nh, 0)
    pad_w = max(crop_size - nw, 0)
    if pad_h > 0 or pad_w > 0:
        image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w,
                                   cv2.BORDER_REFLECT_101)
        mask  = cv2.copyMakeBorder(mask,  0, pad_h, 0, pad_w,
                                   cv2.BORDER_CONSTANT, value=ignore_index)

    # Random crop
    h2, w2 = image.shape[:2]
    y0 = random.randint(0, h2 - crop_size)
    x0 = random.randint(0, w2 - crop_size)
    image = image[y0:y0 + crop_size, x0:x0 + crop_size]
    mask  = mask [y0:y0 + crop_size, x0:x0 + crop_size]
    return image, mask


def resize(image: np.ndarray, mask: np.ndarray,
           size: int = 512) -> Tuple[np.ndarray, np.ndarray]:
    """Centre-resize to a fixed square for validation."""
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    mask  = cv2.resize(mask,  (size, size), interpolation=cv2.INTER_NEAREST)
    return image, mask


def random_elastic(image: np.ndarray, mask: np.ndarray,
                   alpha: float = 80.0, sigma: float = 10.0,
                   ignore_index: int = 255,
                   p: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
    """Elastic deformation (simulates terrain deformation in remote sensing)."""
    if random.random() >= p:
        return image, mask
    h, w = image.shape[:2]
    dx = cv2.GaussianBlur((np.random.rand(h, w) * 2 - 1).astype(np.float32),
                          (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur((np.random.rand(h, w) * 2 - 1).astype(np.float32),
                          (0, 0), sigma) * alpha
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)
    image_out = cv2.remap(image, map_x, map_y,
                          interpolation=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT_101)
    mask_out  = cv2.remap(mask,  map_x, map_y,
                          interpolation=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=ignore_index)
    return image_out, mask_out


# ─────────────────────────── radiometric ─────────────────────────────────────

def random_brightness_contrast(image: np.ndarray,
                                brightness_limit: float = 0.3,
                                contrast_limit: float = 0.3,
                                p: float = 0.5) -> np.ndarray:
    if random.random() >= p:
        return image
    alpha = 1.0 + random.uniform(-contrast_limit, contrast_limit)
    beta  = random.uniform(-brightness_limit, brightness_limit) * 255
    image = (image.astype(np.float32) * alpha + beta).clip(0, 255).astype(np.uint8)
    return image


def random_gaussian_blur(image: np.ndarray,
                         max_kernel: int = 7,
                         p: float = 0.3) -> np.ndarray:
    if random.random() >= p:
        return image
    k = random.choice(range(3, max_kernel + 1, 2))
    return cv2.GaussianBlur(image, (k, k), 0)


def random_noise(image: np.ndarray, var: float = 10.0,
                 p: float = 0.3) -> np.ndarray:
    if random.random() >= p:
        return image
    noise = np.random.normal(0, var ** 0.5, image.shape).astype(np.float32)
    return (image.astype(np.float32) + noise).clip(0, 255).astype(np.uint8)


def random_channel_shuffle(image: np.ndarray, p: float = 0.1) -> np.ndarray:
    """Shuffle RGB channels to simulate spectral variation."""
    if random.random() < p and image.ndim == 3:
        perm = list(range(image.shape[-1]))
        random.shuffle(perm)
        return image[..., perm]
    return image


# ─────────────────────────── CutMix for segmentation ─────────────────────────

def cutmix(image1: np.ndarray, mask1: np.ndarray,
           image2: np.ndarray, mask2: np.ndarray,
           alpha: float = 1.0,
           p: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
    """CutMix augmentation adapted for segmentation (no label mixing, only cut)."""
    if random.random() >= p:
        return image1, mask1
    h, w = image1.shape[:2]
    lam  = np.random.beta(alpha, alpha)
    cut_ratio = (1 - lam) ** 0.5
    cut_h, cut_w = int(h * cut_ratio), int(w * cut_ratio)
    cx = random.randint(0, w)
    cy = random.randint(0, h)
    x1 = max(cx - cut_w // 2, 0)
    y1 = max(cy - cut_h // 2, 0)
    x2 = min(cx + cut_w // 2, w)
    y2 = min(cy + cut_h // 2, h)
    out_img  = image1.copy()
    out_mask = mask1.copy()
    out_img [y1:y2, x1:x2] = image2[y1:y2, x1:x2]
    out_mask[y1:y2, x1:x2] = mask2 [y1:y2, x1:x2]
    return out_img, out_mask


# ─────────────────────────── normalisation ───────────────────────────────────

# ImageNet statistics (widely used for remote-sensing transfer learning)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def normalize(image: np.ndarray,
              mean: np.ndarray = IMAGENET_MEAN,
              std:  np.ndarray = IMAGENET_STD) -> np.ndarray:
    """Normalize a uint8 HxWx3 image to float32 in ImageNet stats."""
    return (image.astype(np.float32) / 255.0 - mean) / std


# ─────────────────────────── pipeline classes ─────────────────────────────────

class TrainTransform:
    """Standard training augmentation pipeline for landslide segmentation."""

    def __init__(self,
                 crop_size: int = 512,
                 scales: Tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0),
                 ignore_index: int = 255,
                 use_elastic: bool = True,
                 use_cutmix: bool = False,
                 mean: np.ndarray = IMAGENET_MEAN,
                 std:  np.ndarray = IMAGENET_STD) -> None:
        self.crop_size    = crop_size
        self.scales       = scales
        self.ignore_index = ignore_index
        self.use_elastic  = use_elastic
        self.use_cutmix   = use_cutmix
        self.mean = mean
        self.std  = std

    def __call__(self, image, mask, image2=None, mask2=None):
        image = _to_numpy(image)
        mask  = _to_numpy(mask).astype(np.uint8)

        # Geometric
        image, mask = random_hflip(image, mask)
        image, mask = random_vflip(image, mask, p=0.3)
        image, mask = random_rotate(image, mask, max_angle=30,
                                    ignore_index=self.ignore_index, p=0.5)
        if self.use_elastic:
            image, mask = random_elastic(image, mask,
                                         ignore_index=self.ignore_index, p=0.3)
        image, mask = random_scale_crop(image, mask, self.crop_size,
                                        self.scales, self.ignore_index)

        # CutMix
        if self.use_cutmix and image2 is not None and mask2 is not None:
            image2 = _to_numpy(image2)
            mask2  = _to_numpy(mask2).astype(np.uint8)
            image2, mask2 = random_scale_crop(image2, mask2, self.crop_size,
                                              self.scales, self.ignore_index)
            image, mask = cutmix(image, mask, image2, mask2, p=0.5)

        # Radiometric (image only)
        image = random_brightness_contrast(image)
        image = random_gaussian_blur(image)
        image = random_noise(image)
        image = random_channel_shuffle(image)

        # Normalize
        image = normalize(image, self.mean, self.std)
        # HxWxC → CxHxW
        image = image.transpose(2, 0, 1).copy()
        return image, mask.astype(np.int64)


class ValTransform:
    """Validation/test pipeline (deterministic)."""

    def __init__(self, size: int = 512,
                 mean: np.ndarray = IMAGENET_MEAN,
                 std:  np.ndarray = IMAGENET_STD) -> None:
        self.size = size
        self.mean = mean
        self.std  = std

    def __call__(self, image, mask):
        image = _to_numpy(image)
        mask  = _to_numpy(mask).astype(np.uint8)
        image, mask = resize(image, mask, self.size)
        image = normalize(image, self.mean, self.std)
        image = image.transpose(2, 0, 1).copy()
        return image, mask.astype(np.int64)


__all__ = [
    "TrainTransform", "ValTransform",
    "random_hflip", "random_vflip", "random_rotate", "random_scale_crop",
    "random_elastic", "random_brightness_contrast", "random_gaussian_blur",
    "random_noise", "random_channel_shuffle", "cutmix",
    "normalize", "resize",
    "IMAGENET_MEAN", "IMAGENET_STD",
]
