"""
LSFormer — LandSlide Segmentation Transformer
==============================================

A novel dual-path encoder–decoder architecture designed for high-accuracy
landslide semantic segmentation from remote-sensing imagery.

Key innovations
---------------
1. **Dual-path encoder**: a ResNet CNN branch (local texture / edges) and an
   Efficient Multi-Scale Transformer (EMT) branch (global context / long-range
   dependencies) run in parallel and exchange information at every scale.

2. **Multi-Scale Cross-Attention Fusion (MSCAF)**: at each of four resolution
   stages, bidirectional cross-attention with spatial reduction lets the CNN
   query the Transformer for global context *and* lets the Transformer query
   the CNN for fine local cues.  An adaptive gating unit learns how much weight
   to give each branch per spatial position.

3. **Edge-Guided Boundary Refinement (EGBR)**: a lightweight learnable edge
   detector produces a boundary attention map that explicitly sharpens features
   near landslide edges — the most ambiguous region in high-resolution imagery.

4. **Hierarchical Adaptive Decoder (HAD)**: an FPN-style decoder with spatial
   attention gates that selectively pass only relevant skip-connection features
   to each decoder stage, suppressing background clutter.

5. **Deep supervision**: auxiliary segmentation heads at the two deepest
   encoder scales provide gradient signal to all decoder components during
   training.

Publication target
------------------
IEEE TGRS / ISPRS JPRS / Remote Sensing of Environment (all Q1)
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Efficient Multi-Scale Transformer (EMT) — the Transformer branch       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class DWConvFFN(nn.Module):
    """Depth-wise convolution inside the FFN (keeps local context)."""
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.dw  = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = self.fc1(x)
        B, N, C = x.shape
        x = self.act(self.dw(x.transpose(1, 2).view(B, C, H, W))
                        .flatten(2).transpose(1, 2))
        return self.fc2(x)


class EfficientSA(nn.Module):
    """Efficient Self-Attention with spatial reduction ratio ``sr``."""
    def __init__(self, dim: int, num_heads: int = 8, sr: int = 1) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.h     = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.q     = nn.Linear(dim, dim)
        self.kv    = nn.Linear(dim, 2 * dim)
        self.proj  = nn.Linear(dim, dim)
        self.sr    = sr
        if sr > 1:
            self.reduce = nn.Conv2d(dim, dim, sr, stride=sr, bias=False)
            self.norm   = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        head_dim = C // self.h
        q = self.q(x).reshape(B, N, self.h, head_dim).permute(0, 2, 1, 3)
        if self.sr > 1:
            x2 = self.norm(self.reduce(
                x.transpose(1,2).reshape(B, C, H, W)
            ).flatten(2).transpose(1, 2))
        else:
            x2 = x
        kv = self.kv(x2).reshape(B, -1, 2, self.h, head_dim).permute(2,0,3,1,4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1,2).reshape(B, N, C)
        return self.proj(x)


class EMTBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8,
                 mlp_ratio: float = 4.0, sr: int = 1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = EfficientSA(dim, num_heads, sr)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn   = DWConvFFN(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.ffn(self.norm2(x),  H, W)
        return x


class OverlapPatch(nn.Module):
    """Overlapping patch embedding for multi-scale feature extraction."""
    def __init__(self, patch: int, stride: int,
                 in_ch: int, embed: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed, patch, stride=stride,
                              padding=patch // 2, bias=False)
        self.norm = nn.LayerNorm(embed)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        x = self.proj(x)
        B, C, H, W = x.shape
        return self.norm(x.flatten(2).transpose(1,2)), H, W


class EfficientMultiScaleTransformer(nn.Module):
    """4-stage Efficient Multi-Scale Transformer encoder.

    Outputs: list of four spatial feature maps at strides 4, 8, 16, 32.
    Channel widths match those of ResNet-50 for MSCAF alignment:
        [256, 512, 1024, 2048] → we project to [64, 128, 320, 512]
    """

    # Stage configuration: embed_dim, num_heads, sr_ratio, depth
    _stages = [
        dict(embed=64,  heads=1, sr=8, depth=3),
        dict(embed=128, heads=2, sr=4, depth=4),
        dict(embed=320, heads=5, sr=2, depth=6),
        dict(embed=512, heads=8, sr=1, depth=3),
    ]

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        patch_sizes = [7, 3, 3, 3]
        strides     = [4, 2, 2, 2]
        in_ch       = in_channels

        self.patch_embeds = nn.ModuleList()
        self.stages       = nn.ModuleList()
        self.norms        = nn.ModuleList()

        for i, cfg in enumerate(self._stages):
            self.patch_embeds.append(
                OverlapPatch(patch_sizes[i], strides[i], in_ch, cfg["embed"])
            )
            self.stages.append(nn.ModuleList([
                EMTBlock(cfg["embed"], cfg["heads"], sr=cfg["sr"])
                for _ in range(cfg["depth"])
            ]))
            self.norms.append(nn.LayerNorm(cfg["embed"]))
            in_ch = cfg["embed"]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outs = []
        for i in range(4):
            x, H, W = self.patch_embeds[i](x)
            for blk in self.stages[i]:
                x = blk(x, H, W)
            x = self.norms[i](x)
            B, _, C = x.shape
            x = x.transpose(1,2).reshape(B, C, H, W)
            outs.append(x)
        return outs   # [1/4, 1/8, 1/16, 1/32]


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MSCAF — Multi-Scale Cross-Attention Fusion                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class CrossAttentionFusion(nn.Module):
    """One-directional cross-attention: ``query`` attends over ``context``.

    Spatial-reduction is applied to ``context`` when ``sr > 1`` to keep
    memory tractable at high-resolution stages.
    """

    def __init__(self, q_dim: int, kv_dim: int, out_dim: int,
                 num_heads: int = 8, sr: int = 1) -> None:
        super().__init__()
        assert out_dim % num_heads == 0
        self.h     = num_heads
        self.hd    = out_dim // num_heads
        self.scale = self.hd ** -0.5

        self.q_proj  = nn.Linear(q_dim,     out_dim)
        self.kv_proj = nn.Linear(kv_dim,    2 * out_dim)
        self.o_proj  = nn.Linear(out_dim,   out_dim)
        self.sr      = sr
        if sr > 1:
            self.kv_sr   = nn.Conv2d(kv_dim, kv_dim, sr, stride=sr, bias=False)
            self.kv_norm = nn.LayerNorm(kv_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, query: torch.Tensor, context: torch.Tensor,
                H_kv: int, W_kv: int) -> torch.Tensor:
        B, N_q, _ = query.shape
        if self.sr > 1:
            B2, N_kv, C_kv = context.shape
            ctx = self.kv_norm(
                self.kv_sr(context.transpose(1,2).reshape(B2, C_kv, H_kv, W_kv))
                .flatten(2).transpose(1,2)
            )
        else:
            ctx = context
        q  = self.q_proj(query).reshape(B, N_q, self.h, self.hd).permute(0,2,1,3)
        kv = self.kv_proj(ctx).reshape(B, -1, 2, self.h, self.hd).permute(2,0,3,1,4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn = attn.softmax(dim=-1)
        out  = (attn @ v).transpose(1,2).reshape(B, N_q, -1)
        return self.norm(self.o_proj(out))


class MSCAFBlock(nn.Module):
    """Multi-Scale Cross-Attention Fusion Block.

    Given CNN features ``f_c`` and Transformer features ``f_t`` at the same
    spatial scale:
      - CNN queries Transformer → enriches local features with global context
      - Transformer queries CNN → enriches global features with local texture
    An adaptive gate learns to combine the two augmented representations.
    """

    def __init__(self, cnn_dim: int, vit_dim: int, out_dim: int,
                 num_heads: int = 8, sr: int = 1) -> None:
        super().__init__()
        # Project both branches to out_dim
        self.cnn_norm = nn.LayerNorm(cnn_dim)
        self.vit_norm = nn.LayerNorm(vit_dim)

        # CNN queries ViT (global → local enrichment)
        self.c_attn = CrossAttentionFusion(cnn_dim, vit_dim, out_dim,
                                           num_heads, sr)
        # ViT queries CNN (local → global enrichment)
        self.v_attn = CrossAttentionFusion(vit_dim, cnn_dim, out_dim,
                                           num_heads, sr)

        # Adaptive gating: per-channel scalar, range [0,1] per branch
        self.gate = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, 2),
            nn.Softmax(dim=-1),
        )

        # Output projection
        self.ffn = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, out_dim * 4),
            nn.GELU(),
            nn.Linear(out_dim * 4, out_dim),
        )

    def forward(self, f_c: torch.Tensor, f_t: torch.Tensor,
                H: int, W: int) -> torch.Tensor:
        """
        Args:
            f_c: CNN features   (B, C_cnn, H, W)
            f_t: Transformer features (B, C_vit, H, W)
        Returns:
            fused: (B, out_dim, H, W)
        """
        B, C_c, _, _ = f_c.shape
        B, C_t, _, _ = f_t.shape

        # Flatten spatial dims → (B, H*W, C)
        fc_flat = f_c.flatten(2).transpose(1, 2)   # B, N, C_c
        ft_flat = f_t.flatten(2).transpose(1, 2)   # B, N, C_t

        # Normalise
        fc_flat = self.cnn_norm(fc_flat)
        ft_flat = self.vit_norm(ft_flat)

        # Cross-attention (both directions)
        fc_att = self.c_attn(fc_flat, ft_flat, H, W)   # B, N, out_dim
        ft_att = self.v_attn(ft_flat, fc_flat, H, W)   # B, N, out_dim

        # Adaptive gating
        gate_input = torch.cat([fc_att, ft_att], dim=-1)   # B, N, 2*out
        gates = self.gate(gate_input)                       # B, N, 2
        g_c = gates[..., 0:1]    # B, N, 1
        g_t = gates[..., 1:2]    # B, N, 1
        fused = g_c * fc_att + g_t * ft_att                # B, N, out

        fused = fused + self.ffn(fused)
        return fused.transpose(1, 2).reshape(B, -1, H, W)


class MSCAFModule(nn.Module):
    """Apply MSCAF at all 4 encoder scales."""

    # (cnn_dim, vit_dim, out_dim, num_heads, sr)
    _config = [
        (256,  64,  64,  2, 4),
        (512,  128, 128, 4, 2),
        (1024, 320, 256, 8, 1),
        (2048, 512, 512, 8, 1),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            MSCAFBlock(cnn, vit, out, heads, sr)
            for cnn, vit, out, heads, sr in self._config
        ])

    def forward(self, cnn_feats: List[torch.Tensor],
                vit_feats: List[torch.Tensor]) -> List[torch.Tensor]:
        fused = []
        for i, block in enumerate(self.blocks):
            H, W = cnn_feats[i].shape[-2:]
            fused.append(block(cnn_feats[i], vit_feats[i], H, W))
        return fused   # [64,128,256,512] at [1/4,1/8,1/16,1/32]


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  EGBR — Edge-Guided Boundary Refinement                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class EGBRModule(nn.Module):
    """Edge-Guided Boundary Refinement Module.

    Uses the highest-resolution fused features (1/4 scale) to predict a soft
    boundary map.  The boundary map is used as an attention bias to sharpen
    features at all scales near landslide edges.

    Outputs:
        - ``boundary_logit``: (B, 1, H/4, W/4) — auxiliary loss target
        - ``refined_feats``:  list of 4 tensors at same dims as input ``fused``
    """

    def __init__(self, fine_dim: int = 64,
                 coarse_dims: Tuple[int, ...] = (128, 256, 512)) -> None:
        super().__init__()
        # Learnable edge detector from the finest-scale fused feature
        self.edge_detector = nn.Sequential(
            nn.Conv2d(fine_dim, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
        )

        # Per-scale boundary attention modulator
        self.modulators = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(d + 1, d, 1, bias=False),
                nn.BatchNorm2d(d),
                nn.Sigmoid(),
            )
            for d in coarse_dims
        ])

    def forward(self, fused_feats: List[torch.Tensor]
                ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        f0 = fused_feats[0]   # (B, 64, H/4, W/4) — finest
        boundary_logit = self.edge_detector(f0)
        boundary_sig   = boundary_logit.sigmoid()

        refined = [f0]  # finest scale not modulated
        for i, (f, mod) in enumerate(zip(fused_feats[1:], self.modulators)):
            H, W = f.shape[-2:]
            b_down = F.interpolate(boundary_sig, (H, W),
                                   mode="bilinear", align_corners=False)
            gate   = mod(torch.cat([f, b_down], dim=1))
            refined.append(f * gate + f)   # residual connection
        return boundary_logit, refined


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  HAD — Hierarchical Adaptive Decoder                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class AttentionGate(nn.Module):
    """Spatial attention gate that filters skip-connection features."""

    def __init__(self, in_channels: int, gate_channels: int) -> None:
        super().__init__()
        self.theta = nn.Conv2d(in_channels,  1, 1, bias=False)
        self.phi   = nn.Conv2d(gate_channels, 1, 1, bias=False)
        self.psi   = nn.Sequential(nn.ReLU(inplace=True),
                                   nn.Conv2d(1, 1, 1, bias=False),
                                   nn.Sigmoid())

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """x = skip connection (fine); g = gating signal (coarse, upsampled)."""
        g_up = F.interpolate(self.phi(g), x.shape[-2:],
                             mode="bilinear", align_corners=False)
        return x * self.psi(self.theta(x) + g_up)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.attn_gate = AttentionGate(skip_ch, in_ch)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        skip = self.attn_gate(skip, x)
        x    = F.interpolate(x, skip.shape[-2:],
                             mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class HADecoder(nn.Module):
    """Hierarchical Adaptive Decoder.

    Input fused feature channels: [64, 128, 256, 512] at scales 1/4…1/32.
    Decoding: 1/32→1/16→1/8→1/4→1/1.
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        # bottleneck at 1/32
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
        )
        self.dec3 = DecoderBlock(512, 256, 256)   # 1/32 → 1/16
        self.dec2 = DecoderBlock(256, 128, 128)   # 1/16 → 1/8
        self.dec1 = DecoderBlock(128,  64,  64)   # 1/8  → 1/4
        # final upsample 1/4 → 1/1
        self.head = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, 1),
        )

        # Auxiliary segmentation heads for deep supervision
        self.aux3 = nn.Conv2d(256, num_classes, 1)   # at 1/16
        self.aux2 = nn.Conv2d(128, num_classes, 1)   # at 1/8

    def forward(self, feats: List[torch.Tensor], target_size: Tuple[int, int]):
        f1, f2, f3, f4 = feats  # 1/4, 1/8, 1/16, 1/32

        x = self.bottleneck(f4)
        x = self.dec3(x, f3)    # 1/16, 256ch
        aux3 = self.aux3(x)

        x = self.dec2(x, f2)    # 1/8, 128ch
        aux2 = self.aux2(x)

        x = self.dec1(x, f1)    # 1/4, 64ch
        out = F.interpolate(self.head(x), target_size,
                            mode="bilinear", align_corners=False)
        aux3_up = F.interpolate(aux3, target_size,
                                mode="bilinear", align_corners=False)
        aux2_up = F.interpolate(aux2, target_size,
                                mode="bilinear", align_corners=False)
        return out, aux3_up, aux2_up


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  LSFormer — full model                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class LSFormer(nn.Module):
    """LandSlide Segmentation Transformer.

    Args:
        num_classes (int): Number of output segmentation classes.
        pretrained_cnn (bool): Load ImageNet pre-trained ResNet-50 weights for
            the CNN branch.
        in_channels (int): Number of input image channels (default 3).

    Outputs (during training)::

        {
          "out":       (B, num_classes, H, W),   # main prediction
          "aux_deep":  (B, num_classes, H, W),   # 1/16 scale auxiliary
          "aux_mid":   (B, num_classes, H, W),   # 1/8 scale auxiliary
          "boundary":  (B, 1, H/4, W/4),          # edge auxiliary
        }

    During evaluation only ``"out"`` is returned.
    """

    def __init__(self, num_classes: int = 2,
                 pretrained_cnn: bool = True,
                 in_channels: int = 3) -> None:
        super().__init__()

        # ── CNN Branch (ResNet-50) ─────────────────────────────────────────
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained_cnn else None
        _resnet = resnet50(weights=weights)

        self.cnn_stem  = nn.Sequential(_resnet.conv1, _resnet.bn1,
                                       _resnet.relu, _resnet.maxpool)
        self.cnn_layer1 = _resnet.layer1   # 1/4  , 256ch
        self.cnn_layer2 = _resnet.layer2   # 1/8  , 512ch
        self.cnn_layer3 = _resnet.layer3   # 1/16 , 1024ch
        self.cnn_layer4 = _resnet.layer4   # 1/32 , 2048ch

        # Handle non-RGB inputs
        if in_channels != 3:
            self.cnn_stem[0] = nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False)

        # ── Transformer Branch (EMT) ───────────────────────────────────────
        self.emt = EfficientMultiScaleTransformer(in_channels)

        # ── MSCAF Module ──────────────────────────────────────────────────
        self.mscaf = MSCAFModule()

        # ── EGBR Module ───────────────────────────────────────────────────
        self.egbr = EGBRModule(fine_dim=64, coarse_dims=(128, 256, 512))

        # ── Hierarchical Adaptive Decoder ─────────────────────────────────
        self.decoder = HADecoder(num_classes)

    # ── Forward ───────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> dict:
        B, C, H, W = x.shape

        # CNN encoder
        c = self.cnn_stem(x)
        c1 = self.cnn_layer1(c)    # 1/4 , 256
        c2 = self.cnn_layer2(c1)   # 1/8 , 512
        c3 = self.cnn_layer3(c2)   # 1/16, 1024
        c4 = self.cnn_layer4(c3)   # 1/32, 2048

        # Transformer encoder
        t1, t2, t3, t4 = self.emt(x)  # 1/4:64, 1/8:128, 1/16:320, 1/32:512

        # MSCAF: cross-attention fusion at all scales
        fused = self.mscaf(
            [c1, c2, c3, c4],
            [t1, t2, t3, t4],
        )   # [64, 128, 256, 512] at [1/4, 1/8, 1/16, 1/32]

        # EGBR: boundary-guided refinement
        boundary_logit, refined = self.egbr(fused)

        # HAD decoder
        out, aux3, aux2 = self.decoder(refined, (H, W))

        result = {"out": out}
        if self.training:
            result["aux_deep"]  = aux3
            result["aux_mid"]   = aux2
            result["boundary"]  = boundary_logit
        return result

    @classmethod
    def build(cls, num_classes: int = 2,
              pretrained: bool = True, **kwargs) -> "LSFormer":
        return cls(num_classes=num_classes, pretrained_cnn=pretrained, **kwargs)


__all__ = ["LSFormer"]
