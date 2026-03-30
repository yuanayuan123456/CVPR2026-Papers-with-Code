from .base import SegmentationDataset
from .cityscapes import Cityscapes, CITYSCAPES_CLASSES
from .ade20k import ADE20K, ADE20K_CLASSES
from .voc import VOCSegmentation, VOC_CLASSES

__all__ = [
    "SegmentationDataset",
    "Cityscapes", "CITYSCAPES_CLASSES",
    "ADE20K", "ADE20K_CLASSES",
    "VOCSegmentation", "VOC_CLASSES",
]
