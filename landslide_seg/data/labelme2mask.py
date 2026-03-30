"""
LabelMe JSON → segmentation mask converter.

Supports:
  - Binary segmentation (landslide vs background)
  - Multi-class segmentation (if multiple labels present)
  - Batch processing of a whole directory
  - Visualization overlay

Usage (CLI):
    python labelme2mask.py --input_dir ./annotations --output_dir ./masks \
                           --class_map landslide:1 debris:2 --suffix .png

Usage (API):
    from data.labelme2mask import convert_json_to_mask, batch_convert
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

# ─────────────────────────── helpers ────────────────────────────────────────

def _load_json(json_path: str) -> dict:
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def _decode_image_data(data: str) -> np.ndarray:
    """Decode the base64-embedded image inside a LabelMe JSON."""
    import base64
    import io
    img_bytes = base64.b64decode(data)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return np.array(img)


def _polygon_to_mask(polygon_pts: List[List[float]],
                     height: int, width: int) -> np.ndarray:
    """Rasterise a single polygon onto an HxW binary mask."""
    pts = np.array([[int(round(x)), int(round(y))] for x, y in polygon_pts],
                   dtype=np.int32)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], color=1)
    return mask


def _rectangle_to_mask(pts: List[List[float]],
                       height: int, width: int) -> np.ndarray:
    """Rasterise a rectangle (top-left, bottom-right) onto a binary mask."""
    x1, y1 = int(round(pts[0][0])), int(round(pts[0][1]))
    x2, y2 = int(round(pts[1][0])), int(round(pts[1][1]))
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(mask, (min(x1, x2), min(y1, y2)),
                  (max(x1, x2), max(y1, y2)), color=1, thickness=-1)
    return mask


def _circle_to_mask(pts: List[List[float]],
                    height: int, width: int) -> np.ndarray:
    """Rasterise a circle onto a binary mask."""
    cx, cy = pts[0]
    rx, ry = pts[1]
    radius = int(round(((rx - cx) ** 2 + (ry - cy) ** 2) ** 0.5))
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, (int(round(cx)), int(round(cy))), radius, color=1,
               thickness=-1)
    return mask


# ─────────────────────────── core API ───────────────────────────────────────

def convert_json_to_mask(
    json_path: str,
    output_path: Optional[str] = None,
    class_map: Optional[Dict[str, int]] = None,
    ignore_classes: Optional[List[str]] = None,
    default_class_id: int = 1,
    ignore_index: int = 255,
    save_image: bool = False,
    image_output_path: Optional[str] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Convert a single LabelMe JSON annotation to a segmentation mask PNG.

    Args:
        json_path: Path to the LabelMe ``.json`` file.
        output_path: Where to save the mask image (PNG).  If ``None`` the mask
            is only returned without being saved.
        class_map: Mapping from label string to integer class ID.
            If ``None``, all shapes are treated as class ``default_class_id``
            (binary segmentation mode).
        ignore_classes: Label names that should be mapped to ``ignore_index``.
        default_class_id: Class ID used when a label is not found in
            ``class_map``.
        ignore_index: Value written to the mask for ignored regions.
        save_image: If ``True`` also save the embedded RGB image.
        image_output_path: Path for the saved RGB image (required when
            ``save_image=True`` and no sibling image exists).

    Returns:
        Tuple of ``(mask_array, image_array_or_None)``.
        ``mask_array`` is ``uint8``, shape ``(H, W)``.
    """
    data = _load_json(json_path)
    ignore_classes = ignore_classes or []

    # ── Determine canvas size ─────────────────────────────────────────────
    height = data.get("imageHeight") or data.get("image_height")
    width  = data.get("imageWidth")  or data.get("image_width")

    if height is None or width is None:
        # Try to read from sibling image file
        img_file = data.get("imagePath", "")
        base_dir = os.path.dirname(json_path)
        # LabelMe stores just the filename, not the full path
        img_path = os.path.join(base_dir, os.path.basename(img_file))
        if os.path.exists(img_path):
            pil_img = Image.open(img_path)
            width, height = pil_img.size
        elif data.get("imageData"):
            img_array = _decode_image_data(data["imageData"])
            height, width = img_array.shape[:2]
        else:
            raise ValueError(
                f"Cannot determine image size from {json_path}. "
                "Make sure imageHeight/imageWidth fields exist."
            )

    # ── Build mask ────────────────────────────────────────────────────────
    mask = np.zeros((height, width), dtype=np.uint8)

    for shape in data.get("shapes", []):
        label     = shape.get("label", "")
        shape_type = shape.get("shape_type", "polygon")
        points    = shape.get("points", [])

        if label in ignore_classes:
            cls_id = ignore_index
        elif class_map is not None:
            cls_id = class_map.get(label, default_class_id)
        else:
            cls_id = default_class_id

        if shape_type == "polygon":
            region = _polygon_to_mask(points, height, width)
        elif shape_type == "rectangle":
            region = _rectangle_to_mask(points, height, width)
        elif shape_type == "circle":
            region = _circle_to_mask(points, height, width)
        elif shape_type == "linestrip":
            # Treat line-strips as thin filled polygons (unusual but handle it)
            region = _polygon_to_mask(points, height, width)
        else:
            print(f"  [warn] Unsupported shape_type '{shape_type}' in {json_path}, skipping.")
            continue

        mask[region == 1] = cls_id

    # ── Save mask ─────────────────────────────────────────────────────────
    if output_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        Image.fromarray(mask).save(output_path)

    # ── Optionally save the embedded image ────────────────────────────────
    img_array = None
    if save_image:
        if data.get("imageData"):
            img_array = _decode_image_data(data["imageData"])
        else:
            img_file = data.get("imagePath", "")
            base_dir = os.path.dirname(json_path)
            img_path = os.path.join(base_dir, os.path.basename(img_file))
            if os.path.exists(img_path):
                img_array = np.array(Image.open(img_path).convert("RGB"))
        if img_array is not None and image_output_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(image_output_path)), exist_ok=True)
            Image.fromarray(img_array).save(image_output_path)

    return mask, img_array


def batch_convert(
    input_dir: str,
    mask_output_dir: str,
    image_output_dir: Optional[str] = None,
    class_map: Optional[Dict[str, int]] = None,
    ignore_classes: Optional[List[str]] = None,
    default_class_id: int = 1,
    ignore_index: int = 255,
    suffix: str = ".png",
    recursive: bool = False,
) -> None:
    """Batch-convert all ``*.json`` files in ``input_dir``.

    Args:
        input_dir: Directory containing LabelMe JSON files.
        mask_output_dir: Directory where mask images are saved.
        image_output_dir: If provided, also save the corresponding RGB images
            extracted from the JSON files (or copied from sibling files).
        class_map: Label → integer class ID mapping.
        ignore_classes: Labels mapped to ``ignore_index``.
        default_class_id: Fallback class ID.
        ignore_index: Void label value.
        suffix: Output file extension (``'.png'`` or ``'.jpg'``).
        recursive: Search sub-directories recursively.
    """
    input_dir = Path(input_dir)
    pattern = "**/*.json" if recursive else "*.json"
    json_files = sorted(input_dir.glob(pattern))

    if not json_files:
        print(f"[warn] No JSON files found in {input_dir}")
        return

    os.makedirs(mask_output_dir, exist_ok=True)
    if image_output_dir:
        os.makedirs(image_output_dir, exist_ok=True)

    success, failure = 0, 0
    for jf in json_files:
        stem = jf.stem
        rel  = jf.relative_to(input_dir).parent

        mask_out  = os.path.join(mask_output_dir, str(rel), stem + suffix)
        image_out = (os.path.join(image_output_dir, str(rel), stem + suffix)
                     if image_output_dir else None)
        os.makedirs(os.path.dirname(mask_out), exist_ok=True)

        try:
            convert_json_to_mask(
                json_path=str(jf),
                output_path=mask_out,
                class_map=class_map,
                ignore_classes=ignore_classes,
                default_class_id=default_class_id,
                ignore_index=ignore_index,
                save_image=image_output_dir is not None,
                image_output_path=image_out,
            )
            print(f"  ✓  {jf.name}  →  {mask_out}")
            success += 1
        except Exception as exc:
            print(f"  ✗  {jf.name}: {exc}")
            failure += 1

    print(f"\nDone: {success} converted, {failure} failed.")


def create_color_overlay(image: np.ndarray,
                         mask: np.ndarray,
                         class_colors: Optional[Dict[int, Tuple[int, int, int]]] = None,
                         alpha: float = 0.5) -> np.ndarray:
    """Blend a segmentation mask over an image for visual inspection.

    Args:
        image: RGB image array (H, W, 3).
        mask: Integer label mask (H, W).
        class_colors: ``{class_id: (R,G,B)}`` color table.
            Defaults to a built-in table (class 0 = transparent background,
            class 1 = red landslide).
        alpha: Opacity of the mask overlay.

    Returns:
        Blended RGB uint8 array.
    """
    if class_colors is None:
        class_colors = {
            0: (0,   0,   0),    # background – transparent
            1: (255, 0,   0),    # landslide – red
            2: (0,   255, 0),    # debris    – green
            3: (0,   0,   255),  # rockfall  – blue
        }

    overlay = image.copy().astype(np.float32)
    for cls_id, color in class_colors.items():
        region = mask == cls_id
        if cls_id == 0:
            continue   # leave background as-is
        overlay[region] = (
            (1 - alpha) * overlay[region]
            + alpha * np.array(color, dtype=np.float32)
        )
    return overlay.clip(0, 255).astype(np.uint8)


# ─────────────────────────── CLI ─────────────────────────────────────────────

def _parse_class_map(items: List[str]) -> Dict[str, int]:
    """Parse ``['landslide:1', 'debris:2']`` into ``{'landslide': 1, 'debris': 2}``."""
    result = {}
    for item in items:
        parts = item.split(":")
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(
                f"class_map entry '{item}' must be in 'label:id' format"
            )
        result[parts[0]] = int(parts[1])
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Convert LabelMe JSON annotations to segmentation mask images."
    )
    parser.add_argument("--input_dir",  required=True,
                        help="Directory containing LabelMe .json files")
    parser.add_argument("--mask_dir",   required=True,
                        help="Output directory for mask PNG/JPG images")
    parser.add_argument("--image_dir",  default=None,
                        help="(Optional) Output directory for extracted RGB images")
    parser.add_argument("--class_map",  nargs="+", default=None,
                        metavar="label:id",
                        help="Class mapping, e.g.  landslide:1 debris:2")
    parser.add_argument("--ignore",     nargs="+", default=None,
                        metavar="label",
                        help="Label names to map to ignore_index (255)")
    parser.add_argument("--default_id", type=int, default=1,
                        help="Class ID for labels not in --class_map (default 1)")
    parser.add_argument("--ignore_index", type=int, default=255)
    parser.add_argument("--suffix",     default=".png",
                        choices=[".png", ".jpg"],
                        help="Output file extension (default .png)")
    parser.add_argument("--recursive",  action="store_true",
                        help="Search subdirectories recursively")
    args = parser.parse_args()

    class_map = _parse_class_map(args.class_map) if args.class_map else None

    print(f"Input  : {args.input_dir}")
    print(f"Masks  : {args.mask_dir}")
    if class_map:
        print(f"Classes: {class_map}")
    else:
        print(f"Classes: binary (all labels → {args.default_id})")

    batch_convert(
        input_dir=args.input_dir,
        mask_output_dir=args.mask_dir,
        image_output_dir=args.image_dir,
        class_map=class_map,
        ignore_classes=args.ignore,
        default_class_id=args.default_id,
        ignore_index=args.ignore_index,
        suffix=args.suffix,
        recursive=args.recursive,
    )


if __name__ == "__main__":
    main()
