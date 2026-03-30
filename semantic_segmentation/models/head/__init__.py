"""Segmentation heads: UPerHead and SegFormerHead."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1, dilation: int = 1) -> None:
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class PPM(nn.Module):
    """Pyramid Pooling Module used inside UPerHead."""

    def __init__(self, in_ch: int, out_ch: int,
                 pool_sizes: List[int] = (1, 2, 3, 6)) -> None:
        super().__init__()
        # Note: BatchNorm is omitted in the pooling branches because the
        # spatial size after AdaptiveAvgPool2d can be as small as 1×1,
        # which makes BatchNorm fail when batch_size == 1.
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(s),
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.ReLU(inplace=True),
            )
            for s in pool_sizes
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2:]
        parts = [F.interpolate(stage(x), size=(H, W),
                               mode="bilinear", align_corners=False)
                 for stage in self.stages]
        return torch.cat([x] + parts, dim=1)


# ---------------------------------------------------------------------------
# UPerHead
# ---------------------------------------------------------------------------

class UPerHead(nn.Module):
    """Unified Perceptual Parsing (UPer) decode head.

    Fuses multi-scale features from the backbone with an FPN-style neck and
    produces a single segmentation logit map.

    Args:
        in_channels (list[int]): Number of channels from each backbone stage.
        channels (int): Internal feature dimension.
        num_classes (int): Number of output segmentation classes.
        pool_scales (list[int]): Scales for the PPM module (applied to the
            deepest feature map).
        dropout (float): Dropout ratio before the final prediction layer.
    """

    def __init__(self, in_channels: List[int], channels: int,
                 num_classes: int, pool_scales: List[int] = (1, 2, 3, 6),
                 dropout: float = 0.1) -> None:
        super().__init__()

        # PPM on the deepest feature map
        self.ppm = PPM(in_channels[-1], channels // len(pool_scales), pool_scales)
        ppm_out_ch = in_channels[-1] + channels // len(pool_scales) * len(pool_scales)
        self.bottleneck = ConvBNReLU(ppm_out_ch, channels)

        # Lateral 1×1 convolutions for all but the last stage
        self.lateral_convs = nn.ModuleList([
            ConvBNReLU(in_ch, channels, 1, padding=0)
            for in_ch in in_channels[:-1]
        ])

        # FPN output convolutions
        self.fpn_convs = nn.ModuleList([
            ConvBNReLU(channels, channels)
            for _ in in_channels[:-1]
        ])

        self.fusion_conv = ConvBNReLU(channels * len(in_channels), channels)
        self.dropout = nn.Dropout2d(dropout)
        self.cls_seg = nn.Conv2d(channels, num_classes, 1)

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        # Apply PPM to the last (deepest) feature map
        x = self.bottleneck(self.ppm(inputs[-1]))
        fpn_outs = [x]

        # Top-down FPN path
        for i in range(len(self.lateral_convs) - 1, -1, -1):
            lat = self.lateral_convs[i](inputs[i])
            x = F.interpolate(x, size=lat.shape[-2:],
                              mode="bilinear", align_corners=False)
            x = self.fpn_convs[i](lat + x)
            fpn_outs.insert(0, x)

        # Upsample all to the size of the largest (first) feature map
        target_size = fpn_outs[0].shape[-2:]
        fpn_outs = [
            F.interpolate(f, size=target_size, mode="bilinear", align_corners=False)
            if f.shape[-2:] != target_size else f
            for f in fpn_outs
        ]

        out = self.fusion_conv(torch.cat(fpn_outs, dim=1))
        out = self.cls_seg(self.dropout(out))
        return out


# ---------------------------------------------------------------------------
# SegFormerHead
# ---------------------------------------------------------------------------

class SegFormerHead(nn.Module):
    """Lightweight all-MLP decoder from SegFormer.

    Projects each backbone-stage feature to a common ``embed_dim``, then
    fuses them with a single convolution.

    Args:
        in_channels (list[int]): Channel widths from backbone stages.
        embed_dim (int): Common projection dimension.
        num_classes (int): Number of output segmentation classes.
        dropout (float): Dropout ratio before the classification layer.
    """

    def __init__(self, in_channels: List[int], embed_dim: int,
                 num_classes: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear_projs = nn.ModuleList([
            nn.Linear(in_ch, embed_dim) for in_ch in in_channels
        ])
        self.linear_fuse = ConvBNReLU(embed_dim * len(in_channels), embed_dim, 1, padding=0)
        self.dropout = nn.Dropout2d(dropout)
        self.linear_pred = nn.Conv2d(embed_dim, num_classes, 1)

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        # Use the resolution of the highest-resolution (first) feature map
        target_size = inputs[0].shape[-2:]
        outs = []
        for proj, x in zip(self.linear_projs, inputs):
            B, C, H, W = x.shape
            # Apply linear projection channel-wise
            x = proj(x.flatten(2).transpose(1, 2))          # B, H*W, embed_dim
            x = x.transpose(1, 2).reshape(B, -1, H, W)
            x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
            outs.append(x)

        out = self.linear_fuse(torch.cat(outs, dim=1))
        out = self.linear_pred(self.dropout(out))
        return out


__all__ = ["UPerHead", "SegFormerHead"]
