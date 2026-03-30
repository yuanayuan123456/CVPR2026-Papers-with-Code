"""Cityscapes dataset (19-class fine-annotation split)."""

import os
from typing import Callable, Optional, Tuple

import numpy as np
from PIL import Image

from .base import SegmentationDataset


# Cityscapes label → training id mapping
# (labels with trainId=-1 or 255 are ignored)
_LABEL_MAP = np.full(256, 255, dtype=np.uint8)
for _line in [
    # trainId, label_id(s)
    (0,  [7]),        # road
    (1,  [8]),        # sidewalk
    (2,  [11]),       # building
    (3,  [12]),       # wall
    (4,  [13]),       # fence
    (5,  [17]),       # pole
    (6,  [19]),       # traffic light
    (7,  [20]),       # traffic sign
    (8,  [21]),       # vegetation
    (9,  [22]),       # terrain
    (10, [23]),       # sky
    (11, [24]),       # person
    (12, [25]),       # rider
    (13, [26]),       # car
    (14, [27]),       # truck
    (15, [28]),       # bus
    (16, [31]),       # train
    (17, [32]),       # motorcycle
    (18, [33]),       # bicycle
]:
    for _lid in _line[1]:
        _LABEL_MAP[_lid] = _line[0]

CITYSCAPES_CLASSES = [
    "road", "sidewalk", "building", "wall", "fence",
    "pole", "traffic light", "traffic sign", "vegetation", "terrain",
    "sky", "person", "rider", "car", "truck",
    "bus", "train", "motorcycle", "bicycle",
]


class Cityscapes(SegmentationDataset):
    """Cityscapes fine-annotation dataset.

    Expected directory layout::

        root/
          leftImg8bit/
            train/  val/  test/
              city/
                *_leftImg8bit.png
          gtFine/
            train/  val/  test/
              city/
                *_gtFine_labelIds.png

    Args:
        root (str): Dataset root directory.
        split (str): One of ``'train'``, ``'val'``, or ``'test'``.
        transform: Joint image-label transform.
        ignore_index (int): Ignored class label.
    """

    NUM_CLASSES = 19

    def __init__(self, root: str, split: str = "train",
                 transform: Optional[Callable] = None,
                 ignore_index: int = 255) -> None:
        super().__init__(root, split, transform, ignore_index)
        self._samples = self._load_pairs()

    def _load_pairs(self):
        img_dir = os.path.join(self.root, "leftImg8bit", self.split)
        lbl_dir = os.path.join(self.root, "gtFine",       self.split)
        pairs = []
        for city in sorted(os.listdir(img_dir)):
            img_city = os.path.join(img_dir, city)
            for fname in sorted(os.listdir(img_city)):
                if not fname.endswith("_leftImg8bit.png"):
                    continue
                img_path = os.path.join(img_city, fname)
                lbl_name = fname.replace("_leftImg8bit.png", "_gtFine_labelIds.png")
                lbl_path = os.path.join(lbl_dir, city, lbl_name)
                if os.path.exists(lbl_path):
                    pairs.append((img_path, lbl_path))
        return pairs

    def __len__(self) -> int:
        return len(self._samples)

    def _get_pair(self, index: int) -> Tuple[str, str]:
        return self._samples[index]

    def __getitem__(self, index: int):
        img_path, lbl_path = self._get_pair(index)
        image = Image.open(img_path).convert("RGB")
        label_raw = np.array(Image.open(lbl_path), dtype=np.uint8)
        label = Image.fromarray(_LABEL_MAP[label_raw])
        if self.transform is not None:
            image, label = self.transform(image, label)
        return image, label
