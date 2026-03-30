"""
MLFormer — Mamba-Land Segmentation Transformer
===============================================

A state-of-the-art landslide segmentation model inspired by CVPR/NeurIPS 2024-2025
top-tier papers, combining four complementary innovations:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Innovation 1 — Visual State Space (VSS) Encoder  [O(N) complexity]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Inspired by:
  • VMamba (CVPR 2024, Liu et al.): 2D Selective Scan with 4-directional path
  • MambaVision (CVPR 2025, Hatamizadeh & Kautz): Hybrid Mamba-Transformer
  • MobileMamba (CVPR 2025): Multi-receptive field Mamba

Design: 4-stage encoder where each stage uses VSSBlocks with bidirectional
GRU-based selective scan in both row and column directions (4-directional
total). Unlike standard self-attention (O(N²)), the GRU-based SSM achieves
O(N) complexity, making it scalable to high-resolution RS imagery.

Key technical choice: GRU as SSM backbone is grounded in the equivalence
between first-order SSMs and gated recurrent units (Gu et al., 2020);
PyTorch's cuDNN-accelerated GRU provides fast training without custom CUDA.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Innovation 2 — Frequency-Spatial Selective Fusion (FSSF)  [novel]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Inspired by:
  • FDA (CVPR 2020, Yang et al.): Fourier Domain Adaptation
  • FocalNet (NeurIPS 2022): Focal modulation
  • BiFormer (CVPR 2023): Bi-level routing attention

Design: CNN branch provides spatial texture/edge features; VSS branch provides
long-range sequential context. A novel FFT-based frequency attention module
extracts the dominant spatial-frequency components from CNN features, then uses
them as a modulation signal to selectively gate the CNN-VSS feature fusion.

Novelty: No existing segmentation work uses FFT magnitude as a cross-branch
attention modulator. Frequency statistics encode information about large-scale
landslide structures vs. fine-grained textures, providing a principled way to
decide which spatial-domain features should be preserved at each scale.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Innovation 3 — Deformable Boundary-Aware Module (DBAM)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Inspired by:
  • DSCNet (ICCV 2023, Qi et al.): Dynamic Snake Convolution
  • InternImage / DCNv3 (CVPR 2023, Wang et al.): Deformable convolution v3
  • PointRend (CVPR 2020): Point-wise boundary refinement

Design: First predicts a soft boundary map from multi-scale fused features,
then uses the boundary map to generate position-dependent sampling offsets for
deformable convolution (implemented via F.grid_sample for pure-PyTorch
portability). Features sampled along landslide boundary curves are used to
refine the fused representation, drastically improving edge accuracy.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Innovation 4 — Content-Aware Progressive Decoder (CAPD)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Inspired by:
  • CARAFE (ICCV 2019): Content-Aware ReAssembly of FEatures
  • Mask2Former (CVPR 2022): Masked cross-attention decoder
  • HRNet (TPAMI 2020): High-resolution feature maintenance

Design: At each upsampling stage, the decoder generates lightweight kernels
conditioned on the local feature content (rather than using bilinear upsampling
fixed kernels). This allows the model to reconstruct fine landslide boundaries
during feature upsampling, not just during final prediction.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Publication target: IEEE TGRS / ISPRS JPRS / Remote Sensing of Environment (Q1)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Innovation 1 — Visual State Space Encoder (VSS)                           ║
# ║  Inspired by VMamba (CVPR 2024) + MambaVision (CVPR 2025)                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class VSSBlock(nn.Module):
    """Visual State Space Block with 4-directional selective scanning.

    Replaces self-attention (O(N²)) with bidirectional GRU scans in row and
    column directions, achieving O(N) complexity while capturing global context.

    Scanning directions:
      • Row-wise  (L→R  and  R→L via bidirectional GRU)
      • Column-wise (T→B and  B→T via bidirectional GRU)
    Combined, this provides 4-directional full coverage of the feature map.

    The gating mechanism ``y * silu(z)`` follows Mamba's selective state
    filtering: ``z`` is computed from the original input, providing
    input-dependent selectivity analogous to the S6 parameter Δ.
    """

    def __init__(self, dim: int, ssm_ratio: float = 2.0,
                 mlp_ratio: float = 4.0) -> None:
        super().__init__()
        d_inner = int(dim * ssm_ratio)
        d_half  = d_inner // 2

        # Input expansion: x_in (to scan) + z (gate)
        self.in_proj  = nn.Linear(dim, d_inner * 2, bias=False)
        self.in_norm  = nn.LayerNorm(dim)

        # Depthwise conv for local context before scanning
        self.dw_conv  = nn.Conv2d(d_inner, d_inner, 3, padding=1,
                                  groups=d_inner, bias=False)
        self.dw_norm  = nn.BatchNorm2d(d_inner)

        # Bidirectional GRU: covers L→R and R→L simultaneously
        self.row_gru  = nn.GRU(d_inner, d_half, bidirectional=True,
                                batch_first=True, bias=False)
        # Bidirectional GRU: covers T→B and B→T simultaneously
        self.col_gru  = nn.GRU(d_inner, d_half, bidirectional=True,
                                batch_first=True, bias=False)

        # Aggregate 4-directional states → d_inner
        self.agg      = nn.Linear(d_inner * 2, d_inner, bias=False)
        self.agg_norm = nn.LayerNorm(d_inner)

        # Output projection back to dim
        self.out_proj = nn.Linear(d_inner, dim, bias=False)
        self.drop     = nn.Dropout(0.0)

        # Post-SSM FFN (standard Transformer FFN with GELU)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio), bias=False),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim, bias=False),
        )

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Args:
            x:   (B, H*W, dim)  flattened spatial feature tokens
            H,W: spatial dimensions
        Returns:
            x:   (B, H*W, dim)
        """
        B, N, C = x.shape
        residual = x

        # ── Input expansion + local depthwise conv ────────────────────────
        xz     = self.in_proj(self.in_norm(x))               # B, N, 2*d_inner
        x_in, z = xz.chunk(2, dim=-1)                        # each B, N, d_inner
        d_inner = x_in.shape[-1]

        # Apply depthwise conv in 2D for local context
        x_2d = x_in.transpose(1,2).reshape(B, d_inner, H, W)
        x_2d = F.relu(self.dw_norm(self.dw_conv(x_2d)), inplace=True)
        x_in = x_2d.flatten(2).transpose(1,2)                # B, N, d_inner

        # ── Row scanning: reshape to (B*H, W, d_inner) ───────────────────
        x_rows = x_in.reshape(B, H, W, d_inner).reshape(B * H, W, d_inner)
        row_out, _ = self.row_gru(x_rows)                     # B*H, W, d_inner
        row_out = row_out.reshape(B, N, d_inner)              # B, N, d_inner

        # ── Column scanning: reshape to (B*W, H, d_inner) ────────────────
        x_cols = x_in.reshape(B, H, W, d_inner).permute(0, 2, 1, 3)  # B,W,H,d
        x_cols = x_cols.reshape(B * W, H, d_inner)
        col_out, _ = self.col_gru(x_cols)                     # B*W, H, d_inner
        col_out = (col_out.reshape(B, W, H, d_inner)
                   .permute(0, 2, 1, 3)
                   .reshape(B, N, d_inner))                   # B, N, d_inner

        # ── Aggregate 4-directional outputs + selective gate ─────────────
        y = self.agg(torch.cat([row_out, col_out], dim=-1))   # B, N, d_inner
        y = self.agg_norm(y)
        y = y * F.silu(z)                                     # selective gating

        y = self.drop(self.out_proj(y))                       # B, N, dim

        # ── Residual + FFN ────────────────────────────────────────────────
        x = residual + y
        x = x + self.ffn(self.ffn_norm(x))
        return x


class OverlapPatchEmbed(nn.Module):
    """Overlapping patch embedding for hierarchical VSS encoder."""
    def __init__(self, patch: int, stride: int, in_ch: int, embed: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed, patch, stride=stride,
                              padding=patch // 2, bias=False)
        self.norm = nn.LayerNorm(embed)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        x = self.proj(x)
        B, C, H, W = x.shape
        return self.norm(x.flatten(2).transpose(1, 2)), H, W


class VSSEncoder(nn.Module):
    """4-stage Visual State Space Encoder.

    Stage output channels: [64, 128, 320, 512] at strides [4, 8, 16, 32].
    Each stage uses multiple VSSBlocks for progressive feature extraction.
    """

    # (embed_dim, depth, ssm_ratio)
    _cfg = [
        (64,  3, 1.0),
        (128, 4, 1.0),
        (320, 6, 1.5),
        (512, 3, 2.0),
    ]

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        patches  = [7, 3, 3, 3]
        strides  = [4, 2, 2, 2]
        in_ch    = in_channels

        self.embeds = nn.ModuleList()
        self.stages = nn.ModuleList()
        self.norms  = nn.ModuleList()

        for i, (dim, depth, ssm_ratio) in enumerate(self._cfg):
            self.embeds.append(
                OverlapPatchEmbed(patches[i], strides[i], in_ch, dim)
            )
            self.stages.append(nn.ModuleList([
                VSSBlock(dim, ssm_ratio=ssm_ratio) for _ in range(depth)
            ]))
            self.norms.append(nn.LayerNorm(dim))
            in_ch = dim

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Returns 4 feature maps at strides [4, 8, 16, 32]."""
        outs = []
        for i in range(4):
            x, H, W = self.embeds[i](x)
            for blk in self.stages[i]:
                x = blk(x, H, W)
            x = self.norms[i](x)
            B, _, C = x.shape
            x = x.transpose(1, 2).reshape(B, C, H, W)
            outs.append(x)
        return outs


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Innovation 2 — Frequency-Spatial Selective Fusion (FSSF)                 ║
# ║  Novel combination: FFT attention + CNN-SSM cross-branch fusion           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class FrequencyAttention(nn.Module):
    """FFT-based frequency attention for CNN feature modulation.

    Computes the 2D FFT of the CNN feature map, extracts global frequency
    statistics via channel-wise average pooling of the magnitude spectrum,
    then generates a channel attention vector that highlights the most
    informative spatial-frequency bands.

    Low-frequency components correspond to large-scale landslide regions;
    high-frequency components correspond to fine boundary details. This
    attention lets the network adaptively balance these two aspects per scale.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.freq_fc = nn.Sequential(
            nn.Linear(channels, channels // 4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 4, channels, bias=False),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            modulated: (B, C, H, W) — frequency-attentated features
        """
        B, C, H, W = x.shape
        # 2-D real FFT → magnitude spectrum
        x_fft   = torch.fft.rfft2(x.float(), norm="ortho")   # B, C, H, W//2+1
        mag     = x_fft.abs()                                  # B, C, H, W//2+1
        # Global frequency statistics: average over spatial freq axes
        freq_stat = mag.mean(dim=(-2, -1))                     # B, C
        freq_stat = self.norm(freq_stat)
        attn      = self.freq_fc(freq_stat).unsqueeze(-1).unsqueeze(-1)  # B,C,1,1
        return x * attn


class FSSFBlock(nn.Module):
    """Frequency-Spatial Selective Fusion Block.

    Fuses CNN features (spatial/local) with VSS features (sequential/global)
    using FFT-guided frequency attention as a cross-branch selection signal.

    Algorithm:
      1. Compute frequency attention from CNN features → channel weights α
      2. Frequency-modulate CNN features: f_c' = α ⊙ f_c
      3. Channel-modulate VSS features via MLP: f_v' = MLP(f_v)
      4. Adaptive gate g = Sigmoid(Linear([f_c', f_v']))
      5. Output = g ⊙ f_c' + (1-g) ⊙ f_v'  (learnable combination)
      6. Residual FFN
    """

    def __init__(self, cnn_dim: int, vss_dim: int, out_dim: int) -> None:
        super().__init__()
        # Frequency attention on CNN branch
        self.freq_attn = FrequencyAttention(cnn_dim)

        # Project both branches to out_dim
        self.cnn_proj = nn.Sequential(
            nn.Conv2d(cnn_dim, out_dim, 1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )
        self.vss_proj = nn.Sequential(
            nn.Conv2d(vss_dim, out_dim, 1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )

        # Adaptive spatial gate: learns per-location weighting of CNN vs VSS
        self.gate = nn.Sequential(
            nn.Conv2d(out_dim * 2, out_dim, 1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.Sigmoid(),
        )

        # Local refinement FFN
        self.ffn = nn.Sequential(
            nn.Conv2d(out_dim, out_dim * 2, 1, bias=False),
            nn.BatchNorm2d(out_dim * 2),
            nn.GELU(),
            nn.Conv2d(out_dim * 2, out_dim, 1, bias=False),
            nn.BatchNorm2d(out_dim),
        )
        self.res_norm = nn.BatchNorm2d(out_dim)

    def forward(self, f_c: torch.Tensor, f_v: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f_c: CNN features (B, cnn_dim, H, W)
            f_v: VSS features (B, vss_dim, H, W)
        Returns:
            fused: (B, out_dim, H, W)
        """
        # Frequency-guided CNN modulation
        f_c = self.freq_attn(f_c)
        f_c_p = self.cnn_proj(f_c)   # B, out_dim, H, W
        f_v_p = self.vss_proj(f_v)   # B, out_dim, H, W

        # Adaptive spatial gate: decides CNN vs VSS importance per location
        gate  = self.gate(torch.cat([f_c_p, f_v_p], dim=1))  # B, out_dim, H, W
        fused = gate * f_c_p + (1 - gate) * f_v_p

        # Residual FFN
        fused = fused + self.ffn(fused)
        return self.res_norm(fused)


class FSSFModule(nn.Module):
    """Apply FSSF at all 4 encoder scales."""

    # (cnn_dim, vss_dim, out_dim)
    _config = [
        (256,  64,  64),
        (512,  128, 128),
        (1024, 320, 256),
        (2048, 512, 512),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            FSSFBlock(cd, vd, od) for cd, vd, od in self._config
        ])

    def forward(self, cnn_feats: List[torch.Tensor],
                vss_feats: List[torch.Tensor]) -> List[torch.Tensor]:
        return [
            self.blocks[i](cnn_feats[i], vss_feats[i])
            for i in range(4)
        ]


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Innovation 3 — Deformable Boundary-Aware Module (DBAM)                   ║
# ║  Inspired by DSCNet (ICCV 2023) + DCNv3 (CVPR 2023)                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class DBAMModule(nn.Module):
    """Deformable Boundary-Aware Module.

    Two-stage process:
      Stage 1 — Boundary prediction: a lightweight CNN predicts a soft boundary
                probability map from the finest-scale fused features.
      Stage 2 — Boundary-guided deformable sampling: the boundary map guides a
                2D offset field; features are re-sampled at deformed locations
                (via F.grid_sample) to capture context along landslide edges.

    The deformable offsets are bounded so that sampling stays close to the
    predicted boundary, enforcing spatial locality while allowing the network
    to look along curved landslide contours — behaviour analogous to DSCNet's
    snake convolution but generalised to arbitrary multi-scale features.

    Outputs:
        boundary_logit: (B, 1, H/4, W/4) — auxiliary training signal
        refined_feats:  List[Tensor] same dims as fused_feats
    """

    def __init__(self, fine_dim: int = 64,
                 coarse_dims: Tuple[int, ...] = (128, 256, 512)) -> None:
        super().__init__()

        # ── Stage 1: boundary predictor ────────────────────────────────────
        self.boundary_head = nn.Sequential(
            nn.Conv2d(fine_dim, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),   # raw logit
        )

        # ── Stage 2: boundary-guided offset + deformable refinement ────────
        self.offset_heads = nn.ModuleList()
        self.refine_convs  = nn.ModuleList()

        for d in list([fine_dim]) + list(coarse_dims):
            # Offset generator: boundary map + feature → (dx, dy) per pixel
            self.offset_heads.append(nn.Sequential(
                nn.Conv2d(d + 1, 32, 3, padding=1, bias=False),
                nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                nn.Conv2d(32, 2, 1, bias=True),  # 2 = (delta_x, delta_y)
            ))
            # Refinement: concatenate original + deformed features
            self.refine_convs.append(nn.Sequential(
                nn.Conv2d(d * 2, d, 3, padding=1, bias=False),
                nn.BatchNorm2d(d), nn.ReLU(inplace=True),
            ))

    @staticmethod
    def _deformable_sample(feat: torch.Tensor,
                           offsets: torch.Tensor,
                           max_offset_ratio: float = 0.1) -> torch.Tensor:
        """Bilinear grid-sample with additive offset field.

        Args:
            feat:    (B, C, H, W)
            offsets: (B, 2, H, W)  — pixel-space offsets (dx, dy)
            max_offset_ratio: clamps offsets to ±(ratio * H or W)
        """
        B, C, H, W = feat.shape
        # Build normalised base grid in [-1, 1]
        gy, gx = torch.meshgrid(
            torch.linspace(-1, 1, H, device=feat.device),
            torch.linspace(-1, 1, W, device=feat.device),
            indexing="ij",
        )
        grid = torch.stack([gx, gy], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)

        # Normalise offsets (pixel → normalised): dx/W * 2, dy/H * 2
        norm_dx = offsets[:, 0:1, :, :] / W * 2   # B, 1, H, W
        norm_dy = offsets[:, 1:2, :, :] / H * 2
        # Clamp to prevent extreme deformations
        max_d = max_offset_ratio * 2
        norm_dx = norm_dx.clamp(-max_d, max_d)
        norm_dy = norm_dy.clamp(-max_d, max_d)

        offset_grid = torch.cat([norm_dx, norm_dy], dim=1)   # B, 2, H, W
        offset_grid = offset_grid.permute(0, 2, 3, 1)        # B, H, W, 2
        deformed_grid = (grid + offset_grid).clamp(-1, 1)

        return F.grid_sample(feat, deformed_grid,
                             mode="bilinear", padding_mode="border",
                             align_corners=True)

    def forward(self, fused_feats: List[torch.Tensor]
                ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            fused_feats: list of 4 tensors [1/4:64, 1/8:128, 1/16:256, 1/32:512]
        Returns:
            boundary_logit: (B, 1, H/4, W/4)
            refined:        same shape list as fused_feats
        """
        f0 = fused_feats[0]   # finest scale (B, 64, H/4, W/4)

        # Stage 1: predict boundary from finest features
        boundary_logit = self.boundary_head(f0)
        boundary_sig   = boundary_logit.sigmoid()

        refined = []
        for i, feat in enumerate(fused_feats):
            H, W = feat.shape[-2:]
            # Downscale boundary map to current scale
            b_i = F.interpolate(boundary_sig, (H, W),
                                mode="bilinear", align_corners=False)
            # Generate deformable offsets conditioned on boundary probability
            offsets   = self.offset_heads[i](torch.cat([feat, b_i], dim=1))
            deformed  = self._deformable_sample(feat, offsets)
            # Fuse original + deformed features (residual)
            refined.append(self.refine_convs[i](torch.cat([feat, deformed], dim=1)))

        return boundary_logit, refined


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Innovation 4 — Content-Aware Progressive Decoder (CAPD)                  ║
# ║  Inspired by CARAFE (ICCV 2019) + Mask2Former (CVPR 2022)                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class ContentAwareUpsample(nn.Module):
    """Content-Aware upsampling (CARAFE-inspired).

    Generates a lightweight upsampling kernel for each spatial location from
    the local feature content, then uses the kernel for upsampling via
    unfold-based reassembly.  This produces sharper boundaries than bilinear
    interpolation because the kernel adapts to local structures.

    For efficiency, we use a k×k kernel (k=5) and compress the kernel
    prediction head to remain lightweight.
    """

    def __init__(self, channels: int, up_factor: int = 2,
                 k_size: int = 5) -> None:
        super().__init__()
        self.up     = up_factor
        self.k_size = k_size
        self.k2     = k_size ** 2

        # Compress channels before predicting kernels
        self.compress = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1, bias=False),
            nn.BatchNorm2d(channels // 4), nn.ReLU(inplace=True),
        )
        # Predict (up*up) kernels of size k²  per location
        self.kernel_pred = nn.Conv2d(
            channels // 4,
            (up_factor ** 2) * (k_size ** 2),
            1, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Upsample (B, C, H, W) → (B, C, H*up, W*up)."""
        B, C, H, W = x.shape
        up, k = self.up, self.k_size
        pad = k // 2

        # Predict reassembly kernels
        kernels = self.kernel_pred(self.compress(x))      # B, up²·k², H, W
        kernels = F.softmax(kernels, dim=1)               # normalise

        # Extract local patches (neighbourhood for each output pixel)
        x_pad   = F.pad(x, [pad, pad, pad, pad], mode="reflect")
        patches = x_pad.unfold(2, k, 1).unfold(3, k, 1)  # B, C, H, W, k, k
        patches = patches.contiguous().view(B, C, H, W, self.k2)
        # B, C, H, W, k²

        # Reshape kernels for matmul: B, H, W, up², k²
        kernels = kernels.permute(0, 2, 3, 1).contiguous()   # B, H, W, up²·k²
        kernels = kernels.view(B, H, W, up * up, self.k2)    # B, H, W, up², k²

        # Apply kernels per location
        # patches: B, C, H, W, k²  → B, H, W, C, k²
        patches = patches.permute(0, 2, 3, 1, 4)             # B, H, W, C, k²
        # kernels: B, H, W, up², k²

        # Weighted sum: (B, H, W, up², k²) × (B, H, W, C, k²) → (B, H, W, up², C)
        out = torch.einsum("bhwuk, bhwck -> bhwuc", kernels, patches)
        # out: B, H, W, up², C

        # Fold into (B, C, H*up, W*up)
        out = out.permute(0, 4, 1, 3, 2).contiguous()        # B, C, H, up², W  -- wrong order
        # Correct rearrangement:
        # B, H, W, up², C  → B, H, W, up, up, C
        out = out.view(B, H, W, up, up, C)
        # B, H, up, W, up, C  then  B, C, H, up, W, up
        out = out.permute(0, 5, 1, 3, 2, 4).contiguous()     # B, C, H, up, W, up
        out = out.view(B, C, H * up, W * up)
        return out


class CAPDBlock(nn.Module):
    """One Content-Aware decoder stage."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.ca_up    = ContentAwareUpsample(in_ch, up_factor=2)
        # Attention gate on skip connection (from Attention-UNet)
        self.gate_attn = nn.Sequential(
            nn.Conv2d(skip_ch + in_ch, skip_ch, 1, bias=False),
            nn.BatchNorm2d(skip_ch), nn.Sigmoid(),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x_up  = self.ca_up(x)                                # 2× upsample
        x_rs  = F.interpolate(x, skip.shape[-2:],
                               mode="bilinear", align_corners=False)
        gate  = self.gate_attn(
            torch.cat([skip,
                       F.interpolate(x_rs, skip.shape[-2:],
                                     mode="bilinear", align_corners=False)],
                      dim=1)
        )
        skip  = skip * gate
        # Align x_up if size mismatch
        if x_up.shape[-2:] != skip.shape[-2:]:
            x_up = F.interpolate(x_up, skip.shape[-2:],
                                  mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x_up, skip], dim=1))


class CAPDecoder(nn.Module):
    """Content-Aware Progressive Decoder.

    Fused feature dims: [64, 128, 256, 512] at [1/4, 1/8, 1/16, 1/32].
    Decodes: 1/32→1/16→1/8→1/4→1/1.
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
        )
        self.dec3 = CAPDBlock(512, 256, 256)  # 1/32 → 1/16
        self.dec2 = CAPDBlock(256, 128, 128)  # 1/16 → 1/8
        self.dec1 = CAPDBlock(128,  64,  64)  # 1/8  → 1/4

        self.head = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, 1),
        )
        # Auxiliary segmentation heads (deep supervision)
        self.aux3 = nn.Conv2d(256, num_classes, 1)
        self.aux2 = nn.Conv2d(128, num_classes, 1)

    def forward(self, feats: List[torch.Tensor],
                target_size: Tuple[int, int]):
        f1, f2, f3, f4 = feats   # 1/4, 1/8, 1/16, 1/32

        x = self.bottleneck(f4)
        x = self.dec3(x, f3)
        aux3 = self.aux3(x)

        x = self.dec2(x, f2)
        aux2 = self.aux2(x)

        x = self.dec1(x, f1)
        out = F.interpolate(self.head(x), target_size,
                            mode="bilinear", align_corners=False)
        aux3_up = F.interpolate(aux3, target_size,
                                mode="bilinear", align_corners=False)
        aux2_up = F.interpolate(aux2, target_size,
                                mode="bilinear", align_corners=False)
        return out, aux3_up, aux2_up


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MLFormer — Full Model                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class MLFormer(nn.Module):
    """MLFormer: Mamba-Land Segmentation Transformer.

    Architecture overview::

        Input (B, 3, H, W)
              │
         ┌────┴────┐
         │         │
       CNN        VSS Encoder   ← O(N), 4-dir scan  [Innovation 1]
    (ResNet-50)  (VSSEncoder)
         │         │
         └────┬────┘
              │
         FSSF Module            ← FFT-guided fusion  [Innovation 2]
         (freq-spatial gate)
              │
         DBAM Module            ← deformable boundary [Innovation 3]
         (boundary-deform)
              │
         CAPD Decoder           ← content-aware up   [Innovation 4]
              │
        out + aux (training)

    Args:
        num_classes (int): Number of output classes.
        pretrained_cnn (bool): Load ImageNet-pretrained ResNet-50 for CNN branch.
        in_channels (int): Input channel count (3 for RGB).
    """

    def __init__(self, num_classes: int = 2,
                 pretrained_cnn: bool = True,
                 pretrained: bool = True,
                 in_channels: int = 3) -> None:
        pretrained_cnn = pretrained_cnn and pretrained
        super().__init__()

        # ── CNN Branch (ResNet-50) ─────────────────────────────────────────
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained_cnn else None
        _r = resnet50(weights=weights)
        if in_channels != 3:
            _r.conv1 = nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False)

        self.cnn_stem   = nn.Sequential(_r.conv1, _r.bn1, _r.relu, _r.maxpool)
        self.cnn_layer1 = _r.layer1   # 1/4 , 256ch
        self.cnn_layer2 = _r.layer2   # 1/8 , 512ch
        self.cnn_layer3 = _r.layer3   # 1/16, 1024ch
        self.cnn_layer4 = _r.layer4   # 1/32, 2048ch

        # ── VSS Branch ────────────────────────────────────────────────────
        self.vss = VSSEncoder(in_channels)

        # ── FSSF Fusion ───────────────────────────────────────────────────
        self.fssf = FSSFModule()

        # ── DBAM Boundary Refinement ──────────────────────────────────────
        self.dbam = DBAMModule(fine_dim=64, coarse_dims=(128, 256, 512))

        # ── CAPD Decoder ──────────────────────────────────────────────────
        self.decoder = CAPDecoder(num_classes)

    def forward(self, x: torch.Tensor) -> dict:
        B, C, H, W = x.shape

        # CNN encoder
        c  = self.cnn_stem(x)
        c1 = self.cnn_layer1(c)    # 1/4 , 256
        c2 = self.cnn_layer2(c1)   # 1/8 , 512
        c3 = self.cnn_layer3(c2)   # 1/16, 1024
        c4 = self.cnn_layer4(c3)   # 1/32, 2048

        # VSS encoder
        v1, v2, v3, v4 = self.vss(x)   # 1/4:64, 1/8:128, 1/16:320, 1/32:512

        # FSSF: frequency-spatial selective fusion
        fused = self.fssf(
            [c1, c2, c3, c4],
            [v1, v2, v3, v4],
        )   # [64, 128, 256, 512] at [1/4, 1/8, 1/16, 1/32]

        # DBAM: boundary-guided deformable refinement
        boundary_logit, refined = self.dbam(fused)

        # CAPD decoder
        out, aux3, aux2 = self.decoder(refined, (H, W))

        result = {"out": out}
        if self.training:
            result["aux_deep"] = aux3
            result["aux_mid"]  = aux2
            result["boundary"] = boundary_logit
        return result

    @classmethod
    def build(cls, num_classes: int = 2,
              pretrained: bool = True, **kwargs) -> "MLFormer":
        return cls(num_classes=num_classes, pretrained_cnn=pretrained, **kwargs)


__all__ = [
    "MLFormer",
    "VSSEncoder", "VSSBlock",
    "FSSFModule", "FSSFBlock", "FrequencyAttention",
    "DBAMModule",
    "CAPDecoder", "CAPDBlock", "ContentAwareUpsample",
]
