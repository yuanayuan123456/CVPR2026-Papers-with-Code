"""ResNet backbone and Mix Transformer (MiT) backbone for semantic segmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------

def _make_divisible(v: float, divisor: int = 8, min_value: int = None) -> int:
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


# ---------------------------------------------------------------------------
# ResNet Backbone
# ---------------------------------------------------------------------------

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1,
                 downsample: nn.Module = None, dilation: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=stride,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=dilation,
                               dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1,
                 downsample: nn.Module = None, dilation: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class ResNet(nn.Module):
    """ResNet backbone with dilated convolutions for dense prediction.

    Args:
        depth (int): Network depth, one of {18, 34, 50, 101, 152}.
        dilations (tuple): Output strides for each stage; defaults to
            ``(1, 1, 2, 4)`` which keeps the feature-map at stride 8.
        out_indices (tuple): Indices of stages whose output is returned.
        pretrained (str | None): Path to pre-trained weights.
    """

    _arch_settings = {
        18:  (BasicBlock,    [2, 2, 2, 2]),
        34:  (BasicBlock,    [3, 4, 6, 3]),
        50:  (Bottleneck,    [3, 4, 6, 3]),
        101: (Bottleneck,    [3, 4, 23, 3]),
        152: (Bottleneck,    [3, 8, 36, 3]),
    }

    def __init__(self, depth: int = 50,
                 dilations: Tuple[int, ...] = (1, 1, 2, 4),
                 out_indices: Tuple[int, ...] = (0, 1, 2, 3),
                 pretrained: str = None) -> None:
        super().__init__()
        assert depth in self._arch_settings, f"Unsupported ResNet depth: {depth}"
        block, layers = self._arch_settings[depth]
        self.out_indices = out_indices
        self.inplanes = 64

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        strides = [1, 2, 1, 1]  # stage-level strides; dilation controls receptive field
        planes  = [64, 128, 256, 512]
        self.layers = nn.ModuleList()
        for i in range(4):
            self.layers.append(
                self._make_layer(block, planes[i], layers[i],
                                 stride=strides[i], dilation=dilations[i])
            )

        if pretrained is not None:
            self._load_pretrained(pretrained)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample, dilation=1)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def _load_pretrained(self, path: str) -> None:
        state = torch.load(path, map_location="cpu")
        if "state_dict" in state:
            state = state["state_dict"]
        missing, unexpected = self.load_state_dict(state, strict=False)
        if missing:
            print(f"[ResNet] Missing keys: {missing}")
        if unexpected:
            print(f"[ResNet] Unexpected keys: {unexpected}")

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.maxpool(self.stem(x))
        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)
        return outs


# ---------------------------------------------------------------------------
# Mix Transformer (MiT) Backbone  – simplified, self-contained implementation
# ---------------------------------------------------------------------------

class DWConv(nn.Module):
    """Depth-wise convolution used in the MLP feed-forward network."""

    def __init__(self, dim: int = 768) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        return x.flatten(2).transpose(1, 2)


class MixFFN(nn.Module):
    def __init__(self, in_features: int, hidden_features: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dw_conv = DWConv(hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = self.act(self.dw_conv(self.fc1(x), H, W))
        return self.fc2(x)


class EfficientSelfAttention(nn.Module):
    """Efficient self-attention with spatial reduction (SR).

    Args:
        dim (int): Feature dimension.
        num_heads (int): Number of attention heads.
        sr_ratio (int): Spatial reduction ratio.
    """

    def __init__(self, dim: int, num_heads: int = 8, sr_ratio: int = 1) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.norm(self.sr(x_).reshape(B, C, -1).permute(0, 2, 1))
        else:
            x_ = x
        kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 sr_ratio: int = 1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientSelfAttention(dim, num_heads, sr_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MixFFN(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.mlp(self.norm2(x), H, W)
        return x


class OverlapPatchEmbed(nn.Module):
    """Overlapping patch embedding with stride 4, 2, 2, 2 across stages."""

    def __init__(self, patch_size: int = 7, stride: int = 4,
                 in_chans: int = 3, embed_dim: int = 64) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=stride,
                              padding=patch_size // 2)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        return self.norm(x), H, W


class MixTransformer(nn.Module):
    """Mix Transformer backbone (MiT-B0 … MiT-B5).

    Reference: SegFormer (Xie et al., NeurIPS 2021).

    Args:
        variant (str): One of ``'b0'``, ``'b1'``, ``'b2'``, ``'b3'``,
            ``'b4'``, ``'b5'``.
        out_indices (tuple): Stage indices to include in the output.
        pretrained (str | None): Path to pre-trained weights.
    """

    _variants = {
        #              embed_dims,          depths,      num_heads, sr_ratios
        "b0": dict(embed_dims=[32,  64,  160,  256], depths=[2, 2, 2, 2],
                   num_heads=[1, 2, 5, 8], sr_ratios=[8, 4, 2, 1]),
        "b1": dict(embed_dims=[64,  128, 320,  512], depths=[2, 2, 2, 2],
                   num_heads=[1, 2, 5, 8], sr_ratios=[8, 4, 2, 1]),
        "b2": dict(embed_dims=[64,  128, 320,  512], depths=[3, 4, 6, 3],
                   num_heads=[1, 2, 5, 8], sr_ratios=[8, 4, 2, 1]),
        "b3": dict(embed_dims=[64,  128, 320,  512], depths=[3, 4, 18, 3],
                   num_heads=[1, 2, 5, 8], sr_ratios=[8, 4, 2, 1]),
        "b4": dict(embed_dims=[64,  128, 320,  512], depths=[3, 8, 27, 3],
                   num_heads=[1, 2, 5, 8], sr_ratios=[8, 4, 2, 1]),
        "b5": dict(embed_dims=[64,  128, 320,  512], depths=[3, 6, 40, 3],
                   num_heads=[1, 2, 5, 8], sr_ratios=[8, 4, 2, 1]),
    }

    def __init__(self, variant: str = "b2",
                 out_indices: Tuple[int, ...] = (0, 1, 2, 3),
                 pretrained: str = None) -> None:
        super().__init__()
        assert variant in self._variants, f"Unknown MiT variant: {variant}"
        cfg = self._variants[variant]
        embed_dims = cfg["embed_dims"]
        depths     = cfg["depths"]
        num_heads  = cfg["num_heads"]
        sr_ratios  = cfg["sr_ratios"]
        self.out_indices = out_indices

        patch_sizes = [7, 3, 3, 3]
        strides     = [4, 2, 2, 2]

        self.patch_embeds = nn.ModuleList()
        self.stages       = nn.ModuleList()
        self.norms        = nn.ModuleList()

        in_chans = 3
        for i in range(4):
            self.patch_embeds.append(
                OverlapPatchEmbed(patch_sizes[i], strides[i], in_chans, embed_dims[i])
            )
            self.stages.append(nn.ModuleList([
                TransformerBlock(embed_dims[i], num_heads[i],
                                 mlp_ratio=4.0, sr_ratio=sr_ratios[i])
                for _ in range(depths[i])
            ]))
            self.norms.append(nn.LayerNorm(embed_dims[i]))
            in_chans = embed_dims[i]

        if pretrained is not None:
            self._load_pretrained(pretrained)

    def _load_pretrained(self, path: str) -> None:
        state = torch.load(path, map_location="cpu")
        if "state_dict" in state:
            state = state["state_dict"]
        missing, unexpected = self.load_state_dict(state, strict=False)
        if missing:
            print(f"[MixTransformer] Missing keys: {missing}")
        if unexpected:
            print(f"[MixTransformer] Unexpected keys: {unexpected}")

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outs = []
        for i in range(4):
            x, H, W = self.patch_embeds[i](x)
            for blk in self.stages[i]:
                x = blk(x, H, W)
            x = self.norms[i](x)
            B, _, C = x.shape
            x = x.transpose(1, 2).reshape(B, C, H, W)
            if i in self.out_indices:
                outs.append(x)
        return outs


__all__ = ["ResNet", "MixTransformer"]
