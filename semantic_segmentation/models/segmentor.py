"""Top-level segmentor that glues backbone + decode-head together."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from .backbone import ResNet, MixTransformer
from .head import UPerHead, SegFormerHead


_BACKBONES = {
    "resnet18":  lambda kw: ResNet(depth=18,  **kw),
    "resnet34":  lambda kw: ResNet(depth=34,  **kw),
    "resnet50":  lambda kw: ResNet(depth=50,  **kw),
    "resnet101": lambda kw: ResNet(depth=101, **kw),
    "resnet152": lambda kw: ResNet(depth=152, **kw),
    "mit_b0":    lambda kw: MixTransformer(variant="b0", **kw),
    "mit_b1":    lambda kw: MixTransformer(variant="b1", **kw),
    "mit_b2":    lambda kw: MixTransformer(variant="b2", **kw),
    "mit_b3":    lambda kw: MixTransformer(variant="b3", **kw),
    "mit_b4":    lambda kw: MixTransformer(variant="b4", **kw),
    "mit_b5":    lambda kw: MixTransformer(variant="b5", **kw),
}

_HEADS = {
    "uper":      UPerHead,
    "segformer": SegFormerHead,
}

# Default output channel widths for each backbone
_BACKBONE_CHANNELS = {
    "resnet18":  [64,  128,  256,  512],
    "resnet34":  [64,  128,  256,  512],
    "resnet50":  [256, 512, 1024, 2048],
    "resnet101": [256, 512, 1024, 2048],
    "resnet152": [256, 512, 1024, 2048],
    "mit_b0":    [32,  64,  160,  256],
    "mit_b1":    [64,  128, 320,  512],
    "mit_b2":    [64,  128, 320,  512],
    "mit_b3":    [64,  128, 320,  512],
    "mit_b4":    [64,  128, 320,  512],
    "mit_b5":    [64,  128, 320,  512],
}


class Segmentor(nn.Module):
    """Encoder-decoder segmentation model.

    Combines a configurable backbone with a configurable decode head and
    resizes the output logits to the input resolution.

    Args:
        backbone (str): Backbone name, e.g. ``'resnet50'``, ``'mit_b2'``.
        head (str): Decode-head name, either ``'uper'`` or ``'segformer'``.
        num_classes (int): Number of segmentation classes.
        channels (int): Internal feature channels for the decode head.
        backbone_kwargs (dict): Extra kwargs forwarded to the backbone.
        head_kwargs (dict): Extra kwargs forwarded to the decode head.
        aux_head (bool): Whether to attach an auxiliary segmentation head on
            stage-3 features (helpful during training).
        aux_channels (int): Channels for the auxiliary head.
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        head: str = "uper",
        num_classes: int = 19,
        channels: int = 256,
        backbone_kwargs: Optional[Dict] = None,
        head_kwargs: Optional[Dict] = None,
        aux_head: bool = True,
        aux_channels: int = 256,
    ) -> None:
        super().__init__()
        backbone_kwargs = backbone_kwargs or {}
        head_kwargs = head_kwargs or {}

        assert backbone in _BACKBONES, \
            f"Unknown backbone '{backbone}'. Choose from {list(_BACKBONES)}"
        assert head in _HEADS, \
            f"Unknown head '{head}'. Choose from {list(_HEADS)}"

        self.backbone = _BACKBONES[backbone](backbone_kwargs)
        in_channels: List[int] = _BACKBONE_CHANNELS[backbone]

        HeadCls = _HEADS[head]
        if head == "uper":
            self.decode_head = HeadCls(
                in_channels=in_channels,
                channels=channels,
                num_classes=num_classes,
                **head_kwargs,
            )
        else:  # segformer
            self.decode_head = HeadCls(
                in_channels=in_channels,
                embed_dim=channels,
                num_classes=num_classes,
                **head_kwargs,
            )

        self.aux_head: Optional[nn.Module] = None
        if aux_head:
            self.aux_head = nn.Sequential(
                nn.Conv2d(in_channels[2], aux_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(aux_channels),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),
                nn.Conv2d(aux_channels, num_classes, 1),
            )

    def forward(
        self, x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Return dict with keys ``'out'`` and optionally ``'aux'``."""
        H, W = x.shape[-2:]
        features: List[torch.Tensor] = self.backbone(x)

        logits = self.decode_head(features)
        logits = F.interpolate(logits, size=(H, W),
                               mode="bilinear", align_corners=False)

        result: Dict[str, torch.Tensor] = {"out": logits}

        if self.aux_head is not None and self.training:
            aux = self.aux_head(features[2])
            aux = F.interpolate(aux, size=(H, W),
                                mode="bilinear", align_corners=False)
            result["aux"] = aux

        return result

    @classmethod
    def from_config(cls, cfg: Dict) -> "Segmentor":
        """Instantiate a Segmentor from a configuration dictionary."""
        return cls(
            backbone=cfg.get("backbone", "resnet50"),
            head=cfg.get("head", "uper"),
            num_classes=cfg.get("num_classes", 19),
            channels=cfg.get("channels", 256),
            backbone_kwargs=cfg.get("backbone_kwargs", {}),
            head_kwargs=cfg.get("head_kwargs", {}),
            aux_head=cfg.get("aux_head", True),
            aux_channels=cfg.get("aux_channels", 256),
        )
