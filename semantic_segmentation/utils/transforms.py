"""Data augmentation / preprocessing transforms for segmentation."""

import random
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from typing import Tuple, List, Optional, Union


class Compose:
    """Apply a sequence of transforms to ``(image, label)`` pairs."""

    def __init__(self, transforms: list) -> None:
        self.transforms = transforms

    def __call__(self, image: Image.Image,
                 label: Image.Image) -> Tuple[Image.Image, Image.Image]:
        for t in self.transforms:
            image, label = t(image, label)
        return image, label


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, image, label):
        if random.random() < self.p:
            return TF.hflip(image), TF.hflip(label)
        return image, label


class RandomVerticalFlip:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, image, label):
        if random.random() < self.p:
            return TF.vflip(image), TF.vflip(label)
        return image, label


class RandomScale:
    """Randomly scale the image and label by a factor sampled from ``scales``."""

    def __init__(self, scales: List[float] = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0),
                 interpolation: int = Image.BILINEAR) -> None:
        self.scales = scales
        self.interpolation = interpolation

    def __call__(self, image, label):
        scale = random.choice(self.scales)
        w, h = image.size
        nw, nh = int(w * scale), int(h * scale)
        image = image.resize((nw, nh), self.interpolation)
        label = label.resize((nw, nh), Image.NEAREST)
        return image, label


class RandomCrop:
    """Randomly crop ``(image, label)`` to ``crop_size``.

    Pads with ``fill`` / ``ignore_index`` if the image is smaller than
    the requested crop size.

    Args:
        crop_size (int | tuple): ``(height, width)`` or a single integer.
        fill (int | tuple): Padding value for the image (default 0).
        ignore_index (int): Padding value for the label (default 255).
    """

    def __init__(self, crop_size: Union[int, Tuple[int, int]],
                 fill: Union[int, Tuple] = 0,
                 ignore_index: int = 255) -> None:
        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)
        self.crop_h, self.crop_w = crop_size
        self.fill         = fill
        self.ignore_index = ignore_index

    def __call__(self, image, label):
        w, h = image.size
        pad_h = max(self.crop_h - h, 0)
        pad_w = max(self.crop_w - w, 0)

        if pad_h > 0 or pad_w > 0:
            image = TF.pad(image, [0, 0, pad_w, pad_h], fill=self.fill)
            label = TF.pad(label, [0, 0, pad_w, pad_h], fill=self.ignore_index)

        w, h = image.size
        x = random.randint(0, w - self.crop_w)
        y = random.randint(0, h - self.crop_h)
        image = TF.crop(image, y, x, self.crop_h, self.crop_w)
        label = TF.crop(label, y, x, self.crop_h, self.crop_w)
        return image, label


class RandomRotation:
    """Rotate image and label by a random angle within ``[-degrees, degrees]``."""

    def __init__(self, degrees: float = 10.0,
                 fill: Union[int, Tuple] = 0,
                 ignore_index: int = 255) -> None:
        self.degrees      = degrees
        self.fill         = fill
        self.ignore_index = ignore_index

    def __call__(self, image, label):
        angle = random.uniform(-self.degrees, self.degrees)
        image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR,
                          fill=self.fill)
        label = TF.rotate(label, angle, interpolation=TF.InterpolationMode.NEAREST,
                          fill=self.ignore_index)
        return image, label


class ColorJitter:
    """Randomly change brightness, contrast, saturation, and hue of the image."""

    def __init__(self, brightness: float = 0.4, contrast: float = 0.4,
                 saturation: float = 0.4, hue: float = 0.1) -> None:
        import torchvision.transforms as T
        self._jitter = T.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, image, label):
        return self._jitter(image), label


class Resize:
    """Resize image and label to ``size`` (H, W)."""

    def __init__(self, size: Tuple[int, int]) -> None:
        self.size = size  # (H, W)

    def __call__(self, image, label):
        h, w = self.size
        image = image.resize((w, h), Image.BILINEAR)
        label = label.resize((w, h), Image.NEAREST)
        return image, label


class ToTensor:
    """Convert PIL image to float tensor and label to long tensor."""

    def __call__(self, image, label):
        img_t   = TF.to_tensor(image)
        label_t = torch.from_numpy(np.array(label, dtype=np.int64))
        return img_t, label_t


class Normalize:
    """Normalize image tensor with mean and std."""

    def __init__(self, mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
                 std:  Tuple[float, ...] = (0.229, 0.224, 0.225)) -> None:
        self.mean = mean
        self.std  = std

    def __call__(self, image: torch.Tensor,
                 label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        image = TF.normalize(image, self.mean, self.std)
        return image, label


def build_train_transform(crop_size: int = 512,
                          scales: Optional[List[float]] = None,
                          ignore_index: int = 255) -> Compose:
    """Construct a standard training transform pipeline."""
    if scales is None:
        scales = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    return Compose([
        RandomScale(scales),
        RandomHorizontalFlip(),
        ColorJitter(),
        RandomCrop(crop_size, ignore_index=ignore_index),
        ToTensor(),
        Normalize(),
    ])


def build_val_transform(base_size: int = 512) -> Compose:
    """Construct a standard validation transform pipeline."""
    return Compose([
        Resize((base_size, base_size)),
        ToTensor(),
        Normalize(),
    ])


__all__ = [
    "Compose", "RandomHorizontalFlip", "RandomVerticalFlip",
    "RandomScale", "RandomCrop", "RandomRotation",
    "ColorJitter", "Resize", "ToTensor", "Normalize",
    "build_train_transform", "build_val_transform",
]
