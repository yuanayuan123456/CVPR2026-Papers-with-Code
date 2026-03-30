"""Model registry — call ``get_model(name, **kwargs)`` to build any model."""

from .mlformerv2 import MLFormerV2      # ★★ Best: CVPR 2025 innovations (V2)
from .mlformer import MLFormer          # ★ CVPR 2024-2025 inspired (V1)
from .lsformer import LSFormer          # Dual-path CNN+Transformer baseline
from .baselines import UNet, DeepLabV3Plus, SegFormer, HRNetSeg

_REGISTRY = {
    # ── Proposed (best) ───────────────────────────────────────────────────
    "mlformerv2":    MLFormerV2,        # LocalVSS+PGCSA+DOABM+IBR (recommended)

    # ── Proposed (v1) ─────────────────────────────────────────────────────
    "mlformer":      MLFormer,          # VSS + FSSF + DBAM + CAPD

    # ── Proposed (v0) ─────────────────────────────────────────────────────
    "lsformer":      LSFormer,          # Dual-path CNN-Transformer + MSCAF + EGBR

    # ── Comparison baselines ──────────────────────────────────────────────
    "unet":          UNet,
    "deeplabv3plus": DeepLabV3Plus,
    "segformer":     SegFormer,
    "hrnet":         HRNetSeg,
}


def get_model(name: str, **kwargs):
    """Instantiate a segmentation model by name.

    Args:
        name: One of ``'mlformerv2'``, ``'mlformer'``, ``'lsformer'``,
              ``'unet'``, ``'deeplabv3plus'``, ``'segformer'``, ``'hrnet'``.
        **kwargs: Constructor keyword arguments forwarded to the model class.
            Common keys: ``num_classes``, ``pretrained``, ``in_channels``.

    Returns:
        A ``nn.Module`` with a ``forward(x) -> dict`` interface where the
        returned dict always contains key ``"out"`` (the main logit tensor).

    Example::

        model = get_model("mlformerv2", num_classes=2, pretrained=True)
        model = get_model("mlformer",   num_classes=2, pretrained=True)
        model = get_model("unet",       num_classes=2)
    """
    name = name.lower().strip()
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Available: {sorted(_REGISTRY.keys())}"
        )
    return _REGISTRY[name](**kwargs)


def list_models():
    """Return a sorted list of registered model names."""
    return sorted(_REGISTRY.keys())


__all__ = [
    "get_model", "list_models",
    "MLFormerV2", "MLFormer", "LSFormer",
    "UNet", "DeepLabV3Plus", "SegFormer", "HRNetSeg",
]
