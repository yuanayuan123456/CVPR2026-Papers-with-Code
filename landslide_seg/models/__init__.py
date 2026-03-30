"""Model registry — call ``get_model(name, **kwargs)`` to build any model."""

from .lsformer import LSFormer
from .baselines import UNet, DeepLabV3Plus, SegFormer, HRNetSeg

_REGISTRY = {
    # ── Proposed ──────────────────────────────────────────────────────────
    "lsformer":      LSFormer,

    # ── Baselines ─────────────────────────────────────────────────────────
    "unet":          UNet,
    "deeplabv3plus": DeepLabV3Plus,
    "segformer":     SegFormer,
    "hrnet":         HRNetSeg,
}


def get_model(name: str, **kwargs):
    """Instantiate a segmentation model by name.

    Args:
        name: One of ``'lsformer'``, ``'unet'``, ``'deeplabv3plus'``,
              ``'segformer'``, ``'hrnet'``.
        **kwargs: Constructor keyword arguments forwarded to the model class.
            Common keys: ``num_classes``, ``pretrained``, ``in_channels``.

    Returns:
        A ``nn.Module`` with a ``forward(x) -> dict`` interface where the
        returned dict always contains key ``"out"`` (the main logit tensor).

    Example::

        model = get_model("lsformer", num_classes=2, pretrained=True)
        model = get_model("unet",     num_classes=2)
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
    "LSFormer", "UNet", "DeepLabV3Plus", "SegFormer", "HRNetSeg",
]
