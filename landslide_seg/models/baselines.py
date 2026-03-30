"""
Baseline models for landslide segmentation comparison:
  - UNet (Ronneberger et al., 2015)
  - DeepLabV3+ (Chen et al., 2018) with ResNet-50/101 backbone
  - SegFormer (Xie et al., 2021) simplified
  - HRNet (Wang et al., 2020) simplified high-resolution fusion
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, resnet101
from torchvision.models import ResNet50_Weights, ResNet101_Weights


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Shared building blocks                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3,
                 stride: int = 1, padding: int = 1,
                 dilation: int = 1, bias: bool = False) -> None:
        super().__init__(
            nn.Conv2d(in_ch, out_ch, k, stride=stride,
                      padding=padding * dilation, dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  1. U-Net                                                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class UNetEncoder(nn.Module):
    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        def _block(ic, oc):
            return nn.Sequential(
                ConvBNReLU(ic, oc), ConvBNReLU(oc, oc)
            )
        self.enc1 = _block(in_channels, 64)
        self.enc2 = _block(64,  128)
        self.enc3 = _block(128, 256)
        self.enc4 = _block(256, 512)
        self.pool  = nn.MaxPool2d(2)
        self.bottle = _block(512, 1024)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottle(self.pool(e4))
        return b, [e1, e2, e3, e4]


class UNetDecoder(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        def _up(ic, sc, oc):
            return nn.Sequential(ConvBNReLU(ic + sc, oc), ConvBNReLU(oc, oc))
        self.up4 = _up(1024, 512, 512)
        self.up3 = _up(512,  256, 256)
        self.up2 = _up(256,  128, 128)
        self.up1 = _up(128,   64,  64)
        self.head = nn.Conv2d(64, num_classes, 1)

    def _upsample(self, x, skip):
        x = F.interpolate(x, skip.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([x, skip], dim=1)

    def forward(self, bottleneck, skips):
        e1, e2, e3, e4 = skips
        x = self.up4(self._upsample(bottleneck, e4))
        x = self.up3(self._upsample(x,          e3))
        x = self.up2(self._upsample(x,          e2))
        x = self.up1(self._upsample(x,          e1))
        return {"out": self.head(x)}


class UNet(nn.Module):
    """Standard U-Net for semantic segmentation.

    Reference: Ronneberger et al., U-Net: Convolutional Networks for
    Biomedical Image Segmentation. MICCAI 2015.
    """

    def __init__(self, num_classes: int = 2, in_channels: int = 3, **kwargs) -> None:
        super().__init__()
        self.encoder = UNetEncoder(in_channels)
        self.decoder = UNetDecoder(num_classes)

    def forward(self, x: torch.Tensor) -> dict:
        b, skips = self.encoder(x)
        return self.decoder(b, skips)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  2. DeepLabV3+                                                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling."""

    def __init__(self, in_channels: int = 2048, out_channels: int = 256,
                 dilations: Tuple[int, ...] = (6, 12, 18)) -> None:
        super().__init__()
        self.branches = nn.ModuleList([
            ConvBNReLU(in_channels, out_channels, k=1, padding=0),
            *[ConvBNReLU(in_channels, out_channels, dilation=d)
              for d in dilations],
        ])
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        total = out_channels * (len(dilations) + 2)
        self.project = ConvBNReLU(total, out_channels, k=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W   = x.shape[-2:]
        parts  = [b(x) for b in self.branches]
        gp     = F.interpolate(self.global_pool(x), (H, W),
                               mode="bilinear", align_corners=False)
        parts.append(gp)
        return self.project(torch.cat(parts, dim=1))


class DeepLabV3Plus(nn.Module):
    """DeepLabV3+ with ResNet-50 or ResNet-101 backbone.

    Reference: Chen et al., Encoder-Decoder with Atrous Separable Convolution
    for Semantic Image Segmentation. ECCV 2018.
    """

    def __init__(self, num_classes: int = 2, backbone: str = "resnet50",
                 pretrained: bool = True) -> None:
        super().__init__()

        if backbone == "resnet50":
            weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            _r = resnet50(weights=weights)
        else:
            weights = ResNet101_Weights.IMAGENET1K_V1 if pretrained else None
            _r = resnet101(weights=weights)

        # Use dilated layer3/layer4 to get stride-8 output
        _r.layer3[0].conv2.stride = (1, 1)
        _r.layer3[0].downsample[0].stride = (1, 1)
        for m in _r.layer3[1:]:
            m.conv2.dilation = (2, 2)
            m.conv2.padding  = (2, 2)
        _r.layer4[0].conv2.stride = (1, 1)
        _r.layer4[0].downsample[0].stride = (1, 1)
        for m in _r.layer4[1:]:
            m.conv2.dilation = (4, 4)
            m.conv2.padding  = (4, 4)

        self.backbone = nn.Sequential(
            _r.conv1, _r.bn1, _r.relu, _r.maxpool,
            _r.layer1, _r.layer2, _r.layer3, _r.layer4
        )
        # Low-level features at 1/4 from layer1
        self.low_level_layers = nn.Sequential(
            _r.conv1, _r.bn1, _r.relu, _r.maxpool, _r.layer1
        )

        high_ch = 2048
        self.aspp = ASPP(high_ch, 256)
        self.low_proj = ConvBNReLU(256, 48, k=1, padding=0)

        self.head = nn.Sequential(
            ConvBNReLU(256 + 48, 256),
            ConvBNReLU(256, 256),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, 1),
        )

        # Keep references for forward pass
        self._r = _r

    def forward(self, x: torch.Tensor) -> dict:
        H, W = x.shape[-2:]
        # Low-level features (1/4)
        low = self._r.conv1(x)
        low = self._r.bn1(low)
        low = self._r.relu(low)
        low = self._r.maxpool(low)
        low = self._r.layer1(low)    # 256ch, H/4

        # High-level features (1/8 with dilation)
        high = self._r.layer2(low)
        high = self._r.layer3(high)
        high = self._r.layer4(high)   # 2048ch, H/8

        x_aspp = F.interpolate(self.aspp(high), low.shape[-2:],
                               mode="bilinear", align_corners=False)
        x_low  = self.low_proj(low)
        x_cat  = torch.cat([x_aspp, x_low], dim=1)
        out    = F.interpolate(self.head(x_cat), (H, W),
                               mode="bilinear", align_corners=False)
        return {"out": out}


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  3. SegFormer (simplified)                                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class _MixFFN(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.dw  = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x, H, W):
        x = self.fc1(x)
        B, N, C = x.shape
        x = self.act(self.dw(x.transpose(1,2).view(B, C, H, W))
                        .flatten(2).transpose(1,2))
        return self.fc2(x)


class _ESA(nn.Module):
    def __init__(self, dim, heads=8, sr=1):
        super().__init__()
        self.h = heads; hd = dim // heads
        self.scale = hd ** -0.5
        self.q = nn.Linear(dim, dim); self.kv = nn.Linear(dim, 2*dim)
        self.proj = nn.Linear(dim, dim)
        self.sr = sr
        if sr > 1:
            self.sr_conv = nn.Conv2d(dim, dim, sr, stride=sr, bias=False)
            self.norm    = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape; hd = C // self.h
        q = self.q(x).reshape(B, N, self.h, hd).permute(0,2,1,3)
        if self.sr > 1:
            x2 = self.norm(self.sr_conv(
                x.transpose(1,2).reshape(B,C,H,W)
            ).flatten(2).transpose(1,2))
        else: x2 = x
        kv = self.kv(x2).reshape(B,-1,2,self.h,hd).permute(2,0,3,1,4)
        k,v = kv[0], kv[1]
        a = (q @ k.transpose(-2,-1)) * self.scale
        a = a.softmax(-1)
        x = (a @ v).transpose(1,2).reshape(B, N, C)
        return self.proj(x)


class _MiTBlock(nn.Module):
    def __init__(self, dim, heads, sr):
        super().__init__()
        self.n1 = nn.LayerNorm(dim); self.attn = _ESA(dim, heads, sr)
        self.n2 = nn.LayerNorm(dim); self.ffn  = _MixFFN(dim, dim*4)

    def forward(self, x, H, W):
        x = x + self.attn(self.n1(x), H, W)
        x = x + self.ffn(self.n2(x),  H, W)
        return x


class _OverlapEmbed(nn.Module):
    def __init__(self, patch, stride, in_ch, embed):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed, patch, stride=stride,
                              padding=patch//2, bias=False)
        self.norm = nn.LayerNorm(embed)

    def forward(self, x):
        x = self.proj(x); B,C,H,W = x.shape
        return self.norm(x.flatten(2).transpose(1,2)), H, W


class MiTEncoder(nn.Module):
    _stages = [
        dict(embed=64,  heads=1, sr=8, depth=3),
        dict(embed=128, heads=2, sr=4, depth=4),
        dict(embed=320, heads=5, sr=2, depth=6),
        dict(embed=512, heads=8, sr=1, depth=3),
    ]

    def __init__(self, variant: str = "b2"):
        super().__init__()
        # Depth scaling by variant
        depth_scale = {"b0": 0.34, "b1": 0.5, "b2": 1.0,
                       "b3": 1.5,  "b4": 2.0, "b5": 3.0}
        scale = depth_scale.get(variant, 1.0)
        patches = [7, 3, 3, 3]; strides = [4, 2, 2, 2]; in_ch = 3
        self.embeds = nn.ModuleList(); self.stages = nn.ModuleList()
        self.norms  = nn.ModuleList()
        for i, cfg in enumerate(self._stages):
            d = max(1, int(cfg["depth"] * scale))
            self.embeds.append(_OverlapEmbed(patches[i], strides[i], in_ch, cfg["embed"]))
            self.stages.append(nn.ModuleList([
                _MiTBlock(cfg["embed"], cfg["heads"], cfg["sr"]) for _ in range(d)
            ]))
            self.norms.append(nn.LayerNorm(cfg["embed"]))
            in_ch = cfg["embed"]

    def forward(self, x):
        outs = []
        for i in range(4):
            x, H, W = self.embeds[i](x)
            for blk in self.stages[i]: x = blk(x, H, W)
            x = self.norms[i](x)
            B,_,C = x.shape
            x = x.transpose(1,2).reshape(B,C,H,W)
            outs.append(x)
        return outs


class SegFormerHead(nn.Module):
    def __init__(self, in_channels, embed_dim, num_classes):
        super().__init__()
        self.projs = nn.ModuleList([nn.Linear(c, embed_dim) for c in in_channels])
        self.fuse  = ConvBNReLU(embed_dim * 4, embed_dim, k=1, padding=0)
        self.drop  = nn.Dropout2d(0.1)
        self.pred  = nn.Conv2d(embed_dim, num_classes, 1)

    def forward(self, feats, target_size):
        H, W = feats[0].shape[-2:]
        parts = []
        for proj, f in zip(self.projs, feats):
            B,C,fH,fW = f.shape
            f = proj(f.flatten(2).transpose(1,2)).transpose(1,2).reshape(B,-1,fH,fW)
            f = F.interpolate(f, (H, W), mode="bilinear", align_corners=False)
            parts.append(f)
        out = self.pred(self.drop(self.fuse(torch.cat(parts,1))))
        return F.interpolate(out, target_size, mode="bilinear", align_corners=False)


class SegFormer(nn.Module):
    """SegFormer with MiT-B2 encoder.

    Reference: Xie et al., SegFormer: Simple and Efficient Design for Semantic
    Segmentation with Transformers. NeurIPS 2021.
    """

    def __init__(self, num_classes: int = 2, variant: str = "b2", **kwargs) -> None:
        super().__init__()
        self.encoder = MiTEncoder(variant)
        in_chs = [64, 128, 320, 512]
        self.decoder = SegFormerHead(in_chs, 256, num_classes)

    def forward(self, x: torch.Tensor) -> dict:
        feats = self.encoder(x)
        out   = self.decoder(feats, x.shape[-2:])
        return {"out": out}


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  4. HRNet (simplified)                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class HRModule(nn.Module):
    """One HRNet exchange block: maintain multiple resolution streams and fuse."""

    def __init__(self, channels: List[int], num_blocks: int = 4) -> None:
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(*[
                nn.Sequential(
                    ConvBNReLU(c, c), ConvBNReLU(c, c)
                ) for _ in range(num_blocks)
            ]) for c in channels
        ])
        n = len(channels)
        # fuse[i][j]: project branch-j features to branch-i channels.
        # Convention: lower index = higher resolution (fewer channels).
        #   j < i  →  j is higher-res; need to DOWNsample j → i (strided conv chain)
        #   j > i  →  j is lower-res;  need to change channels only (spatial
        #             upsampling is done with F.interpolate in forward)
        fuse_rows = []
        for i in range(n):
            row = []
            for j in range(n):
                if i == j:
                    row.append(nn.Identity())
                elif j > i:
                    # lower-res j → higher-res i: 1×1 conv to change channels
                    row.append(ConvBNReLU(channels[j], channels[i], k=1, padding=0))
                else:
                    # higher-res j → lower-res i: chain of (i-j) stride-2 convs
                    layers: list = []
                    in_c = channels[j]
                    for step in range(i - j):
                        out_c = channels[i] if step == (i - j - 1) else channels[j + step + 1]
                        layers += [
                            nn.Conv2d(in_c, out_c, 3, stride=2, padding=1, bias=False),
                            nn.BatchNorm2d(out_c),
                            nn.ReLU(inplace=True),
                        ]
                        in_c = out_c
                    row.append(nn.Sequential(*layers))
            fuse_rows.append(nn.ModuleList(row))
        self.fuse = nn.ModuleList(fuse_rows)

    def forward(self, x_list: List[torch.Tensor]) -> List[torch.Tensor]:
        br = [self.branches[i](x_list[i]) for i in range(len(x_list))]
        outs = []
        for i in range(len(x_list)):
            y = br[i]
            for j in range(len(x_list)):
                if j == i:
                    continue
                f = self.fuse[i][j](br[j])
                if j > i:
                    # lower-res → higher-res: upsample spatially
                    f = F.interpolate(f, br[i].shape[-2:],
                                      mode="bilinear", align_corners=False)
                # j < i case is already spatially downsampled by the conv chain
                y = y + f
            outs.append(F.relu(y, inplace=True))
        return outs


class HRNetSeg(nn.Module):
    """Simplified HRNet for semantic segmentation.

    Reference: Wang et al., Deep High-Resolution Representation Learning for
    Visual Recognition. TPAMI 2020.
    """

    def __init__(self, num_classes: int = 2, **kwargs) -> None:
        super().__init__()
        # Stem: stride-2 twice → 1/4
        self.stem = nn.Sequential(
            ConvBNReLU(3, 64, k=3, stride=2, padding=1),
            ConvBNReLU(64, 64, k=3, stride=2, padding=1),
        )
        # Transition: create 1/8 branch
        self.tr1 = nn.Sequential(
            ConvBNReLU(64, 48),
            nn.Sequential(ConvBNReLU(64, 96, k=3, stride=2, padding=1)),
        )
        # Stage 2 — two resolutions [48, 96]
        self.stage2 = HRModule([48, 96], num_blocks=2)
        # Transition: create 1/16 branch
        self.tr2 = nn.Sequential(ConvBNReLU(96, 192, k=3, stride=2, padding=1))
        # Stage 3 — three resolutions [48, 96, 192]
        self.stage3 = HRModule([48, 96, 192], num_blocks=4)
        # Transition: create 1/32 branch
        self.tr3 = nn.Sequential(ConvBNReLU(192, 384, k=3, stride=2, padding=1))
        # Stage 4 — four resolutions [48, 96, 192, 384]
        self.stage4 = HRModule([48, 96, 192, 384], num_blocks=3)

        # Final head
        total_ch = 48 + 96 + 192 + 384
        self.head = nn.Sequential(
            ConvBNReLU(total_ch, 128),
            nn.Conv2d(128, num_classes, 1),
        )

    def forward(self, x: torch.Tensor) -> dict:
        H, W = x.shape[-2:]
        s = self.stem(x)   # 1/4, 64ch

        # Stage 1 branches
        tr_mods = list(self.tr1.children())
        b1_high = tr_mods[0](s)   # 1/4, 48ch
        b1_low  = tr_mods[1](s)   # 1/8, 96ch
        x_list  = self.stage2([b1_high, b1_low])

        x3 = self.tr2(x_list[1])   # 1/16, 192ch
        x_list = self.stage3(x_list + [x3])

        x4 = self.tr3(x_list[2])   # 1/32, 384ch
        x_list = self.stage4(x_list + [x4])

        # Upsample all to 1/4 and concatenate
        target = x_list[0].shape[-2:]
        upsampled = [
            F.interpolate(f, target, mode="bilinear", align_corners=False)
            for f in x_list
        ]
        fused = torch.cat(upsampled, dim=1)
        out = self.head(fused)
        out = F.interpolate(out, (H, W), mode="bilinear", align_corners=False)
        return {"out": out}


__all__ = ["UNet", "DeepLabV3Plus", "SegFormer", "HRNetSeg"]
