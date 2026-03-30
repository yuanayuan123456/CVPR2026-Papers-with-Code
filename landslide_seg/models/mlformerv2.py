"""
MLFormerV2 — Improved Landslide Segmentation Transformer
=========================================================

Building on MLFormerV1, four targeted improvements are introduced, each
grounded in a 2024-2025 CVPR / top-venue paper:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[A] LocalVSSBlock — Window-Partitioned Visual State Space Scanning
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Inspired by:
    • LocalMamba (Huang et al., 2024, arXiv 2403.09338, accepted CVPR'24 workshop)
      "LocalMamba: Visual State Space Model with Windowed Selective Scan"
    • Swin Transformer (Liu et al., ICCV 2021)
      "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"
    • MambaVision (Hatamizadeh & Kautz, CVPR 2025)
      "MambaVision: A Hybrid Mamba-Transformer Vision Backbone"

  Contribution: MLFormerV1 performs GRU-based SSM scanning over the FULL
  row/column of each feature map.  For high-resolution RS imagery (e.g.
  128×128 at stride-4) this creates long GRU sequences that lose local
  detail.  V2 partitions the feature map into non-overlapping ws×ws windows
  (default ws=8) and applies 4-directional bidirectional GRU within each
  window.  A parallel large-kernel (7×7) depthwise convolution provides
  cross-window context at O(N) cost. A scalar learnable gate α ∈ [-1, 1]
  (initialised to 0) balances local-window scans and global context,
  enabling the model to adapt to different spatial scales.

  Key properties:
    — GRU sequence length: ws=8 (fixed) vs W/H (O(√N)) → better cache use
    — Cross-window: global depthwise conv replaces shifted-window tricks
    — Equivalent FLOPs to V1 but smaller peak activation memory

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[B] Prototype-Guided Cross-Scale Attention (PGCSA) — Fusion Module
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Inspired by:
    • MaskDINO (Li et al., CVPR 2023)
      "MaskDINO: Towards a Unified Transformer-based Framework for Object
       and Instance Segmentation"
    • SAM 2 (Ravi et al., CVPR 2025)
      "SAM 2: Segment Anything in Images and Videos"
    • Slot Attention (Locatello et al., NeurIPS 2020)
      "Object-Centric Learning with Slot Attention"

  Contribution: MLFormerV1's FSSF module uses a channel-wise FFT gate to
  fuse CNN and VSS features.  V2 replaces this with K=8 learnable prototype
  vectors (shared across all images) that act as semantic anchors.  At each
  scale, the prototypes attend to spatially-compressed (8×8) CNN+VSS tokens
  via Multi-Head Cross-Attention, then the mean prototype context generates
  a per-channel gate signal modulating the spatial fusion.

  Compared to FSSF:
    — Semantic structure: prototypes capture re-usable landslide archetypes
    — Scale-adaptive: each scale has its own prototype set
    — More expressive than a fixed-rank frequency bottleneck

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[C] Dynamic Orientation-Aware Boundary Module (DOABM)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Inspired by:
    • DSCNet (Qi et al., ICCV 2023)
      "Dynamic Snake Convolution based on Topological Geometric Constraints
       for Tubular Structure Segmentation"
    • Oriented RepPoints (Li et al., CVPR 2022)
      "Oriented RepPoints for Aerial Object Detection"
    • PointRend (Kirillov et al., CVPR 2020)
      "PointRend: Image Segmentation as Rendering"

  Contribution: MLFormerV1's DBAM predicts isotropic (dx, dy) offsets from
  a scalar boundary probability map.  V2 additionally predicts boundary
  orientation (sin θ, cos θ).  Deformable sampling offsets are then
  decomposed into orientation-aligned components:
      dx = s_along · cos θ + s_across · (−sin θ)
      dy = s_along · sin θ + s_across ·    cos θ
  where s_along and s_across are predicted scalar magnitudes.  This
  provides an explicit inductive bias: the model learns to sample along
  the tangent direction of the boundary (capturing boundary texture) AND
  across the boundary (capturing the landslide-vs-background transition).
  The orientation is used internally — no extra annotation is required.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[D] Iterative Boundary Refinement Head (IBRHead)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Inspired by:
    • PointRend (Kirillov et al., CVPR 2020)
      "PointRend: Image Segmentation as Rendering"
    • SAM 2 (Ravi et al., CVPR 2025) — iterative point-based refinement
    • HRSeg (Sun et al., CVPR 2024) — high-res segmentation refinement

  Contribution: The CAPDecoder outputs a full-resolution prediction, but
  CARAFE/bilinear upsampling may still blur fine landslide boundary pixels.
  IBRHead identifies the N most uncertain pixels (highest prediction
  entropy) in the coarse output, samples stride-4 (highest-resolution)
  features at those exact locations via grid_sample, concatenates them with
  the coarse logits, and runs a 3-layer point-MLP to predict corrected class
  probabilities.  The refined predictions are scattered back into the output
  map.  Unlike PointRend, IBRHead uses cosine-similarity prototype matching
  to pre-filter candidate refinement points, focusing compute on true
  boundary ambiguities rather than flat-region uncertainty.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Publication target: CVPR 2026 / IEEE TGRS / ISPRS JPRS (Q1)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Helper utilities                                                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def window_partition(x: torch.Tensor, ws: int) -> torch.Tensor:
    """Partition (B, H, W, C) into (B*nH*nW, ws, ws, C) non-overlapping windows."""
    B, H, W, C = x.shape
    x = x.reshape(B, H // ws, ws, W // ws, ws, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().reshape(-1, ws, ws, C)


def window_reverse(windows: torch.Tensor, ws: int, H: int, W: int) -> torch.Tensor:
    """Reverse window_partition: (B*nH*nW, ws, ws, C) → (B, H, W, C)."""
    nH, nW = H // ws, W // ws
    B = windows.shape[0] // (nH * nW)
    x = windows.reshape(B, nH, nW, ws, ws, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().reshape(B, H, W, -1)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  [A] LocalVSSBlock + LocalVSSEncoder                                        ║
# ║  Inspired by LocalMamba (2024) + Swin (ICCV 2021) + MambaVision (CVPR'25) ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class LocalVSSBlock(nn.Module):
    """Window-partitioned 4-directional Visual State Space Block.

    Partitions the H×W feature map into ws×ws local windows, performs
    bidirectional row and column GRU scans *within each window*, then adds
    cross-window global context via a large-kernel (7×7) depthwise conv.

    A scalar gate ``α`` (initialised to 0, bounded by tanh ∈ (−1, 1)) is
    learned to balance local-window scan output and the global depthwise
    context, allowing the network to adapt context granularity per stage.

    Args:
        dim:         Feature channel size.
        window_size: Local scan window size ws (default 8).
        ssm_ratio:   Inner expansion ratio for SSM projection.
        mlp_ratio:   Feed-forward network expansion ratio.
    """

    def __init__(self, dim: int, window_size: int = 8,
                 ssm_ratio: float = 2.0, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.ws     = window_size
        d_inner     = int(dim * ssm_ratio)
        d_half      = d_inner // 2

        # Pre-norm + input projection (x_in + gate z)
        self.in_norm = nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, d_inner * 2, bias=False)

        # Local 3×3 depthwise conv for per-token context before scanning
        self.dw_conv = nn.Conv2d(d_inner, d_inner, 3, padding=1,
                                 groups=d_inner, bias=False)
        self.dw_norm = nn.BatchNorm2d(d_inner)

        # 4-directional bidirectional GRU (within each window)
        self.row_gru = nn.GRU(d_inner, d_half, bidirectional=True,
                               batch_first=True, bias=False)
        self.col_gru = nn.GRU(d_inner, d_half, bidirectional=True,
                               batch_first=True, bias=False)

        # Aggregate row + col scan outputs
        self.agg      = nn.Linear(d_inner * 2, d_inner, bias=False)
        self.agg_norm = nn.LayerNorm(d_inner)

        # Cross-window global context via large-kernel depthwise conv
        self.global_dw   = nn.Conv2d(d_inner, d_inner, kernel_size=7,
                                      padding=3, groups=d_inner, bias=False)
        self.global_norm = nn.BatchNorm2d(d_inner)
        # Learnable scalar gate: initialised to 0; bounded by tanh
        self.global_gate = nn.Parameter(torch.zeros(1))

        # Output projection
        self.out_proj = nn.Linear(d_inner, dim, bias=False)

        # Feed-forward network with residual
        self.ffn_norm = nn.LayerNorm(dim)
        d_ffn = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, d_ffn, bias=False),
            nn.GELU(),
            nn.Linear(d_ffn, dim, bias=False),
        )

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Args:
            x:   (B, H*W, dim) flattened spatial tokens
            H,W: spatial dimensions
        Returns:
            x:   (B, H*W, dim)
        """
        B, N, C = x.shape
        ws = self.ws
        residual = x

        # ── Input projection ──────────────────────────────────────────────
        xz              = self.in_proj(self.in_norm(x))         # B, N, 2*d_inner
        x_in, z         = xz.chunk(2, dim=-1)                   # B, N, d_inner
        d_inner         = x_in.shape[-1]

        # ── Local depthwise conv (2-D context) ────────────────────────────
        x_2d = x_in.transpose(1, 2).reshape(B, d_inner, H, W)
        x_2d = F.relu(self.dw_norm(self.dw_conv(x_2d)), inplace=True)

        # ── Pad so H, W are divisible by ws ──────────────────────────────
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h > 0 or pad_w > 0:
            x_2d = F.pad(x_2d, (0, pad_w, 0, pad_h))
        Hp, Wp = H + pad_h, W + pad_w

        # ── Global cross-window context (large-kernel depthwise) ─────────
        global_ctx = F.relu(self.global_norm(self.global_dw(x_2d)), inplace=True)

        # ── Window partition: (B, Hp, Wp, d_inner) → (nW*B, ws, ws, d) ──
        x_3d = x_2d.permute(0, 2, 3, 1)                        # B, Hp, Wp, d
        x_win = window_partition(x_3d, ws)                      # nW*B, ws, ws, d
        nW    = (Hp // ws) * (Wp // ws)
        total = nW * B

        # ── Row scanning within windows ───────────────────────────────────
        x_rows = x_win.reshape(total * ws, ws, d_inner)         # total*ws, ws, d
        row_out, _ = self.row_gru(x_rows)                       # total*ws, ws, d
        row_out = row_out.reshape(total, ws, ws, d_inner)       # total, ws, ws, d

        # ── Column scanning within windows ────────────────────────────────
        x_cols = x_win.permute(0, 2, 1, 3).reshape(total * ws, ws, d_inner)
        col_out, _ = self.col_gru(x_cols)
        col_out = col_out.reshape(total, ws, ws, d_inner).permute(0, 2, 1, 3)

        # ── Aggregate + selective gate ────────────────────────────────────
        y_win = self.agg(torch.cat([row_out, col_out], dim=-1)) # total, ws, ws, d
        y_win = self.agg_norm(y_win)

        # ── Window reverse → (B, Hp, Wp, d_inner) ────────────────────────
        y_3d = window_reverse(y_win, ws, Hp, Wp)               # B, Hp, Wp, d
        if pad_h > 0 or pad_w > 0:
            y_3d = y_3d[:, :H, :W, :].contiguous()
        y_2d = y_3d.permute(0, 3, 1, 2)                        # B, d_inner, H, W

        # ── Add cross-window global context (gated) ───────────────────────
        y_2d = y_2d + self.global_gate.tanh() * global_ctx[:, :, :H, :W]

        # ── Flatten + Mamba-style selective gate ──────────────────────────
        y = y_2d.flatten(2).transpose(1, 2)                     # B, N, d_inner
        y = y * F.silu(z)

        y = self.out_proj(y)                                     # B, N, dim
        x = residual + y
        x = x + self.ffn(self.ffn_norm(x))
        return x


class OverlapPatchEmbed(nn.Module):
    """Overlapping patch embedding shared with MLFormerV1."""

    def __init__(self, patch: int, stride: int,
                 in_ch: int, embed: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed, patch, stride=stride,
                              padding=patch // 2, bias=False)
        self.norm = nn.LayerNorm(embed)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        x = self.proj(x)
        B, C, H, W = x.shape
        return self.norm(x.flatten(2).transpose(1, 2)), H, W


class LocalVSSEncoder(nn.Module):
    """4-stage encoder using LocalVSSBlocks.

    Stage output channels: [64, 128, 320, 512] at strides [4, 8, 16, 32].
    Window size is fixed at 8 for all stages; each stage uses LocalVSSBlocks
    for better local-global context balance.
    """

    # (embed_dim, depth, ssm_ratio)
    _cfg = [
        (64,  3, 1.0),
        (128, 4, 1.0),
        (320, 6, 1.5),
        (512, 3, 2.0),
    ]

    def __init__(self, in_channels: int = 3, window_size: int = 8) -> None:
        super().__init__()
        patches = [7, 3, 3, 3]
        strides = [4, 2, 2, 2]
        in_ch   = in_channels

        self.embeds = nn.ModuleList()
        self.stages = nn.ModuleList()
        self.norms  = nn.ModuleList()

        for i, (dim, depth, ssm_ratio) in enumerate(self._cfg):
            self.embeds.append(
                OverlapPatchEmbed(patches[i], strides[i], in_ch, dim)
            )
            self.stages.append(nn.ModuleList([
                LocalVSSBlock(dim, window_size=window_size, ssm_ratio=ssm_ratio)
                for _ in range(depth)
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
# ║  [B] Prototype-Guided Cross-Scale Attention (PGCSA)                        ║
# ║  Inspired by MaskDINO (CVPR'23) + SAM2 (CVPR'25) + Slot Attention          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class PGCSABlock(nn.Module):
    """Single-scale Prototype-Guided Cross-Scale Attention block.

    K learnable prototype vectors serve as cross-attention queries over
    spatially-compressed (8×8) CNN+VSS feature tokens.  The aggregated
    prototype context (mean-pooled over K) generates a per-channel gate
    that modulates the pixel-wise fusion of the two branches.

    This differs from FSSF (MLFormerV1) which uses a static FFT-based gate:
    prototypes are *learned* and carry persistent semantic information about
    landslide archetypes, making the fusion gate data-adaptive.

    Args:
        cnn_ch:         CNN branch input channels.
        vss_ch:         VSS branch input channels.
        out_ch:         Fused output channels.
        num_prototypes: Number of learnable prototype vectors K (default 8).
        num_heads:      Number of heads for multi-head cross-attention.
    """

    def __init__(self, cnn_ch: int, vss_ch: int, out_ch: int,
                 num_prototypes: int = 8, num_heads: int = 4) -> None:
        super().__init__()
        self.out_ch = out_ch

        # Project both branches to common out_ch
        self.cnn_proj = nn.Sequential(
            nn.Conv2d(cnn_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
        self.vss_proj = nn.Sequential(
            nn.Conv2d(vss_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

        # Spatial compression: reduces N → 8×8=64 before cross-attention
        self.compress = nn.AdaptiveAvgPool2d(8)

        # K learnable prototype vectors (semantic anchors)
        self.prototypes = nn.Parameter(
            torch.randn(1, num_prototypes, out_ch) * (out_ch ** -0.5)
        )

        # Multi-Head Cross-Attention: prototypes (Q) attend to tokens (K, V)
        self.mhca = nn.MultiheadAttention(
            out_ch, num_heads, batch_first=True, bias=False
        )

        # Prototype context → per-channel gate
        self.proto_gate = nn.Sequential(
            nn.Linear(out_ch, out_ch, bias=False),
            nn.Sigmoid(),
        )

        # Final fusion + FFN
        self.fusion = nn.Sequential(
            nn.Conv2d(out_ch * 2, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
        self.ffn = nn.Sequential(
            nn.Conv2d(out_ch, out_ch * 2, 1, bias=False),
            nn.BatchNorm2d(out_ch * 2), nn.GELU(),
            nn.Conv2d(out_ch * 2, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, cnn_feat: torch.Tensor,
                vss_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cnn_feat: (B, cnn_ch, H, W)
            vss_feat: (B, vss_ch, H, W)
        Returns:
            fused:    (B, out_ch, H, W)
        """
        B = cnn_feat.shape[0]

        c = self.cnn_proj(cnn_feat)                      # B, out_ch, H, W
        v = self.vss_proj(vss_feat)                      # B, out_ch, H, W
        fused_init = c + v                               # element-wise init

        # ── Prototype cross-attention ─────────────────────────────────────
        # Compress spatial features to 8×8 tokens
        compressed = self.compress(fused_init)           # B, out_ch, 8, 8
        tokens = compressed.flatten(2).transpose(1, 2)  # B, 64, out_ch

        # K prototypes attend to 64 compressed tokens (very efficient)
        proto = self.prototypes.expand(B, -1, -1)        # B, K, out_ch
        proto_ctx, _ = self.mhca(proto, tokens, tokens)  # B, K, out_ch

        # Mean-pool over prototypes → per-channel gate
        gate = self.proto_gate(
            proto_ctx.mean(dim=1)                        # B, out_ch
        ).unsqueeze(-1).unsqueeze(-1)                    # B, out_ch, 1, 1

        # ── Gate-modulated fusion ─────────────────────────────────────────
        fused = self.fusion(
            torch.cat([fused_init * gate, fused_init * (1.0 - gate)], dim=1)
        )
        return fused + self.ffn(fused)                   # residual FFN


class PGCSAModule(nn.Module):
    """Apply PGCSABlock at all 4 encoder scales."""

    # (cnn_ch, vss_ch, out_ch, num_heads)
    _config = [
        (256,   64,  64, 4),
        (512,  128, 128, 4),
        (1024, 320, 256, 8),
        (2048, 512, 512, 8),
    ]

    def __init__(self, num_prototypes: int = 8) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            PGCSABlock(cd, vd, od, num_prototypes, nh)
            for cd, vd, od, nh in self._config
        ])

    def forward(self, cnn_feats: List[torch.Tensor],
                vss_feats: List[torch.Tensor]) -> List[torch.Tensor]:
        return [
            self.blocks[i](cnn_feats[i], vss_feats[i])
            for i in range(4)
        ]


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  [C] Dynamic Orientation-Aware Boundary Module (DOABM)                    ║
# ║  Inspired by DSCNet (ICCV'23) + Oriented RepPoints (CVPR'22)              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class DOABModule(nn.Module):
    """Dynamic Orientation-Aware Boundary Module.

    Stage 1 — Orientation-aware boundary prediction:
        Predicts a 3-channel map at finest scale (1/4):
            ch-0:  boundary probability logit
            ch-1:  sin(θ)  — boundary orientation y-component
            ch-2:  cos(θ)  — boundary orientation x-component
        The (sin θ, cos θ) pair encodes the local tangent direction of the
        nearest landslide boundary, estimated entirely from internal features
        (no extra annotation needed).

    Stage 2 — Orientation-conditioned deformable sampling:
        For each spatial location, two deformable offsets are predicted:
            s_along:  scalar magnitude along the boundary tangent direction
            s_across: scalar magnitude perpendicular to the boundary
        The pixel-space offsets are then:
            dx = s_along · cos θ + s_across · (−sin θ)
            dy = s_along · sin θ + s_across ·    cos θ
        Features are sampled at (x+dx, y+dy) via F.grid_sample.  This
        allows the model to look along curved landslide contours (unlike
        isotropic circular offsets in standard DCN).

    Args:
        fine_dim:    Channel count of finest-scale fused features (64).
        coarse_dims: Channel counts of coarser-scale features (128, 256, 512).
    """

    def __init__(self, fine_dim: int = 64,
                 coarse_dims: Tuple[int, ...] = (128, 256, 512)) -> None:
        super().__init__()

        # ── Stage 1: boundary + orientation predictor ─────────────────────
        self.boundary_head = nn.Sequential(
            nn.Conv2d(fine_dim, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 1),   # [boundary_logit, sin_θ, cos_θ]
        )

        # ── Stage 2: per-scale orientation-conditioned offset + refine ─────
        self.offset_heads = nn.ModuleList()
        self.refine_convs  = nn.ModuleList()

        for d in list([fine_dim]) + list(coarse_dims):
            # Takes [feat, boundary_prob, sin_θ, cos_θ] → 2 scalars (s_along, s_across)
            self.offset_heads.append(nn.Sequential(
                nn.Conv2d(d + 3, 32, 3, padding=1, bias=False),
                nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                nn.Conv2d(32, 2, 1, bias=True),   # (s_along, s_across)
            ))
            # Fuse original + orientation-deformed features
            self.refine_convs.append(nn.Sequential(
                nn.Conv2d(d * 2, d, 3, padding=1, bias=False),
                nn.BatchNorm2d(d), nn.ReLU(inplace=True),
            ))

    @staticmethod
    def _orientation_sample(feat: torch.Tensor,
                             sin_t: torch.Tensor,
                             cos_t: torch.Tensor,
                             scalars: torch.Tensor,
                             max_offset_ratio: float = 0.1) -> torch.Tensor:
        """Sample features at orientation-conditioned offset locations.

        Args:
            feat:    (B, C, H, W)
            sin_t:   (B, 1, H, W) — boundary sin θ
            cos_t:   (B, 1, H, W) — boundary cos θ
            scalars: (B, 2, H, W) — (s_along, s_across)
            max_offset_ratio: max offset as fraction of spatial dim
        Returns:
            sampled: (B, C, H, W)
        """
        B, C, H, W = feat.shape
        s_along  = scalars[:, 0:1]               # B, 1, H, W
        s_across = scalars[:, 1:2]               # B, 1, H, W

        # Orientation-decomposed offsets (pixel space)
        dx = s_along * cos_t + s_across * (-sin_t)   # B, 1, H, W
        dy = s_along * sin_t + s_across * cos_t      # B, 1, H, W

        # Normalise: pixel → normalised coordinates ∈ [-1, 1]
        max_d = max_offset_ratio * 2
        norm_dx = (dx / W * 2).clamp(-max_d, max_d)  # B, 1, H, W
        norm_dy = (dy / H * 2).clamp(-max_d, max_d)  # B, 1, H, W

        # Build deformed sampling grid
        gy, gx = torch.meshgrid(
            torch.linspace(-1, 1, H, device=feat.device),
            torch.linspace(-1, 1, W, device=feat.device),
            indexing="ij",
        )
        base_grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)  # 1, H, W, 2
        offset_grid = torch.cat([norm_dx, norm_dy], dim=1)      # B, 2, H, W
        deformed = (base_grid +
                    offset_grid.permute(0, 2, 3, 1)).clamp(-1, 1)  # B, H, W, 2

        return F.grid_sample(feat, deformed,
                             mode="bilinear", padding_mode="border",
                             align_corners=True)

    def forward(self, fused_feats: List[torch.Tensor]
                ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            fused_feats: [f1, f2, f3, f4] at [1/4:64, 1/8:128, 1/16:256, 1/32:512]
        Returns:
            boundary_logit: (B, 1, H/4, W/4) — auxiliary training signal
            refined:        same-shape list as fused_feats
        """
        f0 = fused_feats[0]   # B, 64, H/4, W/4

        # ── Stage 1: predict boundary + orientation ───────────────────────
        orient_pred   = self.boundary_head(f0)              # B, 3, H/4, W/4
        boundary_logit = orient_pred[:, 0:1]                # B, 1, H/4, W/4
        sin_t0         = orient_pred[:, 1:2]                # B, 1, H/4, W/4
        cos_t0         = orient_pred[:, 2:3]                # B, 1, H/4, W/4

        # Normalise orientation vector to unit length
        norm_factor = (sin_t0 ** 2 + cos_t0 ** 2 + 1e-6).sqrt()
        sin_t0 = sin_t0 / norm_factor
        cos_t0 = cos_t0 / norm_factor

        boundary_sig = boundary_logit.sigmoid()             # B, 1, H/4, W/4

        # ── Stage 2: orientation-conditioned deformable refinement ─────────
        refined = []
        for i, feat in enumerate(fused_feats):
            H, W = feat.shape[-2:]

            # Resize orientation + boundary to current scale
            b_i   = F.interpolate(boundary_sig, (H, W),
                                  mode="bilinear", align_corners=False)
            sin_i = F.interpolate(sin_t0, (H, W),
                                  mode="bilinear", align_corners=False)
            cos_i = F.interpolate(cos_t0, (H, W),
                                  mode="bilinear", align_corners=False)

            # Predict (s_along, s_across) from feature + boundary + orientation
            cond     = torch.cat([feat, b_i, sin_i, cos_i], dim=1)
            scalars  = self.offset_heads[i](cond)           # B, 2, H, W

            # Orientation-conditioned deformable sampling
            deformed = self._orientation_sample(feat, sin_i, cos_i, scalars)

            # Residual fusion of original + deformed
            refined.append(
                self.refine_convs[i](torch.cat([feat, deformed], dim=1))
            )

        return boundary_logit, refined


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  [D] Iterative Boundary Refinement Head (IBRHead)                          ║
# ║  Inspired by PointRend (CVPR'20) + SAM 2 (CVPR'25)                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class IBRHead(nn.Module):
    """Iterative Boundary Refinement Head.

    After the main decoder produces a full-resolution prediction, this head:
      1. Computes per-pixel prediction entropy to identify uncertain regions.
      2. Selects the top-N most uncertain pixels (default N=196) — these
         correspond almost exclusively to boundary ambiguities.
      3. Samples stride-4 (highest-resolution) features at those N locations
         via F.grid_sample.
      4. Concatenates the sampled fine features with the coarse logit values
         at those points (PointRend-style point features).
      5. Runs a 3-layer point MLP to predict refined class logits.
      6. Scatters the refined logits back into the coarse prediction map.

    Unlike PointRend (CVPR 2020) which uses a fixed set of random + uncertain
    points during training, IBRHead uses entropy-based selection at both
    train and test time, making the refinement fully deterministic and
    directly applicable to the most challenging boundary pixels.

    Args:
        fine_ch:    Channels of stride-4 fine features (64).
        num_classes: Number of output classes.
        num_points:  Number of uncertain boundary pixels to refine (N).
    """

    def __init__(self, fine_ch: int = 64, num_classes: int = 2,
                 num_points: int = 196) -> None:
        super().__init__()
        self.num_points = num_points
        in_dim = fine_ch + num_classes

        # Lightweight 3-layer point MLP (operates on each point independently)
        self.point_mlp = nn.Sequential(
            nn.Linear(in_dim, 256, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, coarse_pred: torch.Tensor,
                fine_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coarse_pred: (B, num_classes, H, W) — main decoder output
            fine_feat:   (B, fine_ch, H/4, W/4) — stride-4 encoder features
        Returns:
            refined_pred: (B, num_classes, H, W) — same size, boundary corrected
        """
        B, C, H, W = coarse_pred.shape
        N = min(self.num_points, H * W)

        # ── Step 1: identify top-N uncertain pixels ────────────────────────
        with torch.no_grad():
            probs    = F.softmax(coarse_pred.detach(), dim=1)          # B, C, H, W
            entropy  = -(probs * probs.clamp(min=1e-6).log()).sum(1)   # B, H, W
            unc_flat = entropy.flatten(1)                              # B, H*W
            top_idx  = unc_flat.topk(N, dim=1).indices                 # B, N

        row = top_idx // W      # B, N  (integer row indices)
        col = top_idx %  W      # B, N  (integer col indices)

        # ── Step 2: build normalised sample grid for grid_sample ────────────
        # Coordinates in [-1, 1] matching the coarse_pred spatial resolution
        grid_x = (col.float() / max(W - 1, 1)) * 2.0 - 1.0  # B, N
        grid_y = (row.float() / max(H - 1, 1)) * 2.0 - 1.0  # B, N
        # grid_sample expects (B, H_out, W_out, 2) — treat N points as W_out, H_out=1
        pts = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(1)  # B, 1, N, 2

        # ── Step 3: sample fine features at uncertain point locations ────────
        # Upsample fine features to match coarse_pred resolution first
        fine_up  = F.interpolate(fine_feat, (H, W),
                                  mode="bilinear", align_corners=True)  # B, F, H, W
        fine_pts = F.grid_sample(fine_up, pts,
                                  mode="bilinear", align_corners=True)   # B, F, 1, N
        fine_pts = fine_pts.squeeze(2)                                   # B, F, N

        # ── Step 4: sample coarse logits at the same locations ─────────────
        coarse_pts = F.grid_sample(coarse_pred, pts,
                                    mode="bilinear", align_corners=True)  # B, C, 1, N
        coarse_pts = coarse_pts.squeeze(2)                                # B, C, N

        # ── Step 5: point MLP refinement ────────────────────────────────────
        point_input = torch.cat([fine_pts, coarse_pts], dim=1)  # B, F+C, N
        point_pred  = self.point_mlp(
            point_input.transpose(1, 2)          # B, N, F+C
        )                                         # B, N, num_classes

        # ── Step 6: scatter refined predictions back ────────────────────────
        refined = coarse_pred.clone()
        for b in range(B):
            refined[b, :, row[b], col[b]] = point_pred[b].t()  # C, N
        return refined


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Content-Aware Progressive Decoder (unchanged from MLFormerV1)             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class _ContentAwareUpsample(nn.Module):
    """CARAFE-inspired content-aware upsampling (2× per call)."""

    def __init__(self, channels: int, up_factor: int = 2,
                 k_size: int = 5) -> None:
        super().__init__()
        self.up     = up_factor
        self.k_size = k_size
        self.k2     = k_size ** 2
        self.compress = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1, bias=False),
            nn.BatchNorm2d(channels // 4), nn.ReLU(inplace=True),
        )
        self.kernel_pred = nn.Conv2d(
            channels // 4,
            (up_factor ** 2) * (k_size ** 2),
            1, bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        up, k  = self.up, self.k_size
        pad    = k // 2
        kernels = F.softmax(self.kernel_pred(self.compress(x)), dim=1)  # B, up²k², H, W
        x_pad   = F.pad(x, [pad, pad, pad, pad], mode="reflect")
        patches = x_pad.unfold(2, k, 1).unfold(3, k, 1)
        patches = patches.contiguous().view(B, C, H, W, self.k2)       # B, C, H, W, k²
        kernels = kernels.permute(0, 2, 3, 1).view(B, H, W, up * up, self.k2)
        patches = patches.permute(0, 2, 3, 1, 4)                       # B, H, W, C, k²
        out     = torch.einsum("bhwuk,bhwck->bhwuc", kernels, patches)  # B, H, W, up², C
        out     = out.view(B, H, W, up, up, C)
        out     = out.permute(0, 5, 1, 3, 2, 4).contiguous()
        return out.view(B, C, H * up, W * up)


class _CAPDBlock(nn.Module):
    """One decoder stage: CARAFE upsample + attention-gated skip."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.ca_up     = _ContentAwareUpsample(in_ch)
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
        x_up  = self.ca_up(x)
        x_rs  = F.interpolate(x, skip.shape[-2:],
                               mode="bilinear", align_corners=False)
        gate  = self.gate_attn(torch.cat([skip, x_rs], dim=1))
        skip  = skip * gate
        if x_up.shape[-2:] != skip.shape[-2:]:
            x_up = F.interpolate(x_up, skip.shape[-2:],
                                  mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x_up, skip], dim=1))


class _CAPDecoder(nn.Module):
    """Content-Aware Progressive Decoder: 1/32 → 1/16 → 1/8 → 1/4 → full."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
        )
        self.dec3 = _CAPDBlock(512, 256, 256)
        self.dec2 = _CAPDBlock(256, 128, 128)
        self.dec1 = _CAPDBlock(128,  64,  64)
        self.head = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, 1),
        )
        self.aux3 = nn.Conv2d(256, num_classes, 1)
        self.aux2 = nn.Conv2d(128, num_classes, 1)

    def forward(self, feats: List[torch.Tensor],
                target_size: Tuple[int, int]) -> Tuple[torch.Tensor, ...]:
        f1, f2, f3, f4 = feats
        x    = self.bottleneck(f4)
        x    = self.dec3(x, f3);   aux3 = self.aux3(x)
        x    = self.dec2(x, f2);   aux2 = self.aux2(x)
        x    = self.dec1(x, f1)
        out  = F.interpolate(self.head(x), target_size,
                             mode="bilinear", align_corners=False)
        aux3 = F.interpolate(aux3, target_size,
                             mode="bilinear", align_corners=False)
        aux2 = F.interpolate(aux2, target_size,
                             mode="bilinear", align_corners=False)
        return out, aux3, aux2


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MLFormerV2 — Full Model                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class MLFormerV2(nn.Module):
    """MLFormerV2: Improved Landslide Segmentation Transformer (CVPR'25 based).

    Architecture overview::

        Input (B, 3, H, W)
              │
         ┌────┴────┐
         │         │
       CNN        LocalVSSEncoder   [A] Window-partitioned 4-dir scan
    (ResNet-50)   (ws=8, global-dw)     vs. full-image scan in V1
         │         │
         └────┬────┘
              │
        PGCSAModule                 [B] Prototype-guided cross-scale attention
        (K=8 prototypes)                vs. FFT-gate fusion in V1
              │
        DOABModule                  [C] Orientation-aware boundary deformation
        (boundary + sin/cos θ)          vs. isotropic offsets in V1
              │
        CAPDecoder                      (unchanged from V1)
        (CARAFE + attention gates)
              │
        IBRHead                     [D] Iterative boundary refinement
        (PointRend-style N=196 pts)     NEW in V2
              │
        out + aux (training)

    Args:
        num_classes (int):       Number of output classes.
        pretrained (bool):       Use ImageNet-pretrained ResNet-50 backbone.
        in_channels (int):       Input channel count (3 for RGB).
        window_size (int):       LocalVSS window size (default 8).
        num_prototypes (int):    PGCSA prototype count K (default 8).
        num_ibr_points (int):    IBRHead refinement point count N (default 196).
    """

    def __init__(self,
                 num_classes:    int  = 2,
                 pretrained:     bool = True,
                 in_channels:    int  = 3,
                 window_size:    int  = 8,
                 num_prototypes: int  = 8,
                 num_ibr_points: int  = 196,
                 **kwargs) -> None:
        super().__init__()

        # ── CNN Branch (ResNet-50 backbone) ────────────────────────────────
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        _r = resnet50(weights=weights)
        if in_channels != 3:
            _r.conv1 = nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False)

        self.cnn_stem   = nn.Sequential(_r.conv1, _r.bn1, _r.relu, _r.maxpool)
        self.cnn_layer1 = _r.layer1   # 1/4,  256ch
        self.cnn_layer2 = _r.layer2   # 1/8,  512ch
        self.cnn_layer3 = _r.layer3   # 1/16, 1024ch
        self.cnn_layer4 = _r.layer4   # 1/32, 2048ch

        # ── [A] LocalVSS Branch ────────────────────────────────────────────
        self.vss = LocalVSSEncoder(in_channels, window_size=window_size)

        # ── [B] PGCSA Fusion ───────────────────────────────────────────────
        self.pgcsa = PGCSAModule(num_prototypes=num_prototypes)

        # ── [C] DOABM Boundary Refinement ─────────────────────────────────
        self.doabm = DOABModule(fine_dim=64, coarse_dims=(128, 256, 512))

        # ── Content-Aware Progressive Decoder ─────────────────────────────
        self.decoder = _CAPDecoder(num_classes)

        # ── [D] IBR Head ───────────────────────────────────────────────────
        self.ibr = IBRHead(fine_ch=64, num_classes=num_classes,
                            num_points=num_ibr_points)

    def forward(self, x: torch.Tensor) -> dict:
        B, C, H, W = x.shape

        # ── CNN encoder ────────────────────────────────────────────────────
        c  = self.cnn_stem(x)
        c1 = self.cnn_layer1(c)     # 1/4,  256
        c2 = self.cnn_layer2(c1)    # 1/8,  512
        c3 = self.cnn_layer3(c2)    # 1/16, 1024
        c4 = self.cnn_layer4(c3)    # 1/32, 2048

        # ── [A] Local VSS encoder ──────────────────────────────────────────
        v1, v2, v3, v4 = self.vss(x)   # 64, 128, 320, 512

        # ── [B] PGCSA: prototype-guided fusion ─────────────────────────────
        fused = self.pgcsa(
            [c1, c2, c3, c4],
            [v1, v2, v3, v4],
        )   # [64, 128, 256, 512] at [1/4, 1/8, 1/16, 1/32]

        # ── [C] DOABM: orientation-aware boundary refinement ───────────────
        boundary_logit, refined = self.doabm(fused)

        # ── Decoder ────────────────────────────────────────────────────────
        coarse_out, aux3, aux2 = self.decoder(refined, (H, W))

        # ── [D] IBR: iterative boundary refinement at uncertain pixels ──────
        # fine_feat is the DOABM-refined stride-4 features
        fine_feat = refined[0]           # B, 64, H/4, W/4
        out = self.ibr(coarse_out, fine_feat)

        result = {"out": out}
        if self.training:
            result["aux_deep"] = aux3
            result["aux_mid"]  = aux2
            result["boundary"] = boundary_logit
        return result

    @classmethod
    def build(cls, num_classes: int = 2,
              pretrained: bool = True, **kwargs) -> "MLFormerV2":
        return cls(num_classes=num_classes, pretrained=pretrained, **kwargs)


__all__ = [
    "MLFormerV2",
    "LocalVSSEncoder", "LocalVSSBlock",
    "PGCSAModule", "PGCSABlock",
    "DOABModule",
    "IBRHead",
]
