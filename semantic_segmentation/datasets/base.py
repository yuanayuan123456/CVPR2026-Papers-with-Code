"""Base segmentation dataset."""

import os
from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class SegmentationDataset(Dataset, ABC):
    """Abstract base class for segmentation datasets.

    Sub-classes only need to implement :meth:`__len__` and
    :meth:`_get_pair` which should return a ``(image_path, label_path)``
    tuple for the given index.
    """

    def __init__(self, root: str, split: str = "train",
                 transform: Optional[Callable] = None,
                 ignore_index: int = 255) -> None:
        self.root         = root
        self.split        = split
        self.transform    = transform
        self.ignore_index = ignore_index

    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def _get_pair(self, index: int) -> Tuple[str, str]:
        """Return (image_path, label_path) for the given index."""
        ...

    def __getitem__(self, index: int) -> Tuple:
        img_path, lbl_path = self._get_pair(index)
        image = Image.open(img_path).convert("RGB")
        label = Image.open(lbl_path)
        if self.transform is not None:
            image, label = self.transform(image, label)
        return image, label
