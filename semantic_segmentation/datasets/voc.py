"""Pascal VOC 2012 / SBD segmentation dataset."""

import os
from typing import Callable, Optional, Tuple

from PIL import Image

from .base import SegmentationDataset


VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "potted plant", "sheep", "sofa", "train", "tv/monitor",
]


class VOCSegmentation(SegmentationDataset):
    """Pascal VOC 2012 semantic segmentation dataset.

    Expected directory layout::

        root/
          VOCdevkit/
            VOC2012/
              JPEGImages/       *.jpg
              SegmentationClass/ *.png
              ImageSets/
                Segmentation/
                  train.txt  val.txt  trainval.txt

    Args:
        root (str): Dataset root (the folder that contains ``VOCdevkit``).
        year (str): Dataset year, ``'2012'`` or ``'2007'``.
        split (str): ``'train'``, ``'val'``, or ``'trainval'``.
        transform: Joint image-label transform.
        ignore_index (int): Void label value.
    """

    NUM_CLASSES = 21   # 20 foreground + background

    def __init__(self, root: str, year: str = "2012",
                 split: str = "train",
                 transform: Optional[Callable] = None,
                 ignore_index: int = 255) -> None:
        super().__init__(root, split, transform, ignore_index)
        self.voc_root = os.path.join(root, "VOCdevkit", f"VOC{year}")
        self._samples = self._load_pairs()

    def _load_pairs(self):
        split_file = os.path.join(
            self.voc_root, "ImageSets", "Segmentation", f"{self.split}.txt"
        )
        img_dir = os.path.join(self.voc_root, "JPEGImages")
        lbl_dir = os.path.join(self.voc_root, "SegmentationClass")
        pairs = []
        with open(split_file) as f:
            for line in f:
                stem = line.strip()
                if not stem:
                    continue
                img_path = os.path.join(img_dir, stem + ".jpg")
                lbl_path = os.path.join(lbl_dir, stem + ".png")
                if os.path.exists(img_path) and os.path.exists(lbl_path):
                    pairs.append((img_path, lbl_path))
        return pairs

    def __len__(self) -> int:
        return len(self._samples)

    def _get_pair(self, index: int) -> Tuple[str, str]:
        return self._samples[index]
