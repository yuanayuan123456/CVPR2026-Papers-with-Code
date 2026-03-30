"""ADE20K dataset (150 semantic categories)."""

import os
from typing import Callable, Optional, Tuple

from PIL import Image

from .base import SegmentationDataset


ADE20K_CLASSES = [
    "wall", "building", "sky", "floor", "tree", "ceiling", "road", "bed", "windowpane",
    "grass", "cabinet", "sidewalk", "person", "earth", "door", "table", "mountain",
    "plant", "curtain", "chair", "car", "water", "painting", "sofa", "shelf",
    "house", "sea", "mirror", "rug", "field", "armchair", "seat", "fence", "desk",
    "rock", "wardrobe", "lamp", "bathtub", "railing", "cushion", "base", "box",
    "column", "signboard", "chest of drawers", "counter", "sand", "sink",
    "skyscraper", "fireplace", "refrigerator", "grandstand", "path", "stairs",
    "runway", "case", "pool table", "pillow", "screen door", "stairway", "river",
    "bridge", "bookcase", "blind", "coffee table", "toilet", "flower", "book",
    "hill", "bench", "countertop", "stove", "palm", "kitchen island", "computer",
    "swivel chair", "boat", "bar", "arcade machine", "hovel", "bus", "towel",
    "light", "truck", "tower", "chandelier", "awning", "streetlight", "booth",
    "television receiver", "airplane", "dirt track", "apparel", "pole", "land",
    "bannister", "escalator", "ottoman", "bottle", "buffet", "poster", "stage",
    "van", "ship", "fountain", "conveyer belt", "canopy", "washer", "plaything",
    "swimming pool", "stool", "barrel", "basket", "waterfall", "tent", "bag",
    "minibike", "cradle", "oven", "ball", "food", "step", "tank", "trade name",
    "microwave", "pot", "animal", "bicycle", "lake", "dishwasher", "screen",
    "blanket", "sculpture", "hood", "sconce", "vase", "traffic light", "tray",
    "ashcan", "fan", "pier", "crt screen", "plate", "monitor", "bulletin board",
    "shower", "radiator", "glass", "clock", "flag",
]


class ADE20K(SegmentationDataset):
    """ADE20K segmentation dataset.

    Expected directory layout::

        root/
          images/
            training/   *.jpg
            validation/ *.jpg
          annotations/
            training/   *.png   (1-indexed class IDs; 0 = ignore)
            validation/ *.png

    Args:
        root (str): Dataset root.
        split (str): ``'training'`` or ``'validation'``.
        transform: Joint transform.
        ignore_index (int): Label to ignore (ADE20K uses 0 as background/ignore).
    """

    NUM_CLASSES = 150

    def __init__(self, root: str, split: str = "training",
                 transform: Optional[Callable] = None,
                 ignore_index: int = 255) -> None:
        super().__init__(root, split, transform, ignore_index)
        self._samples = self._load_pairs()

    def _load_pairs(self):
        img_dir = os.path.join(self.root, "images",      self.split)
        lbl_dir = os.path.join(self.root, "annotations", self.split)
        pairs = []
        for fname in sorted(os.listdir(img_dir)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            stem = os.path.splitext(fname)[0]
            lbl_path = os.path.join(lbl_dir, stem + ".png")
            if os.path.exists(lbl_path):
                pairs.append((os.path.join(img_dir, fname), lbl_path))
        return pairs

    def __len__(self) -> int:
        return len(self._samples)

    def _get_pair(self, index: int) -> Tuple[str, str]:
        return self._samples[index]

    def __getitem__(self, index: int):
        img_path, lbl_path = self._get_pair(index)
        image = Image.open(img_path).convert("RGB")
        # ADE20K labels are 1-indexed; subtract 1 so 0-based, 0-pad → ignore
        import numpy as np
        label_raw = np.array(Image.open(lbl_path), dtype=np.int32)
        label_raw = (label_raw - 1).clip(-1)       # 0-indexed; -1 → will map to ignore
        label_raw[label_raw < 0] = self.ignore_index
        label = Image.fromarray(label_raw.astype(np.uint8))
        if self.transform is not None:
            image, label = self.transform(image, label)
        return image, label
