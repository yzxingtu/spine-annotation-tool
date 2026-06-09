"""Export pedicle annotations from full images to crop dataset."""

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

from .converter import YOLOConverter
from .crop_generator import _expand_and_clamp, _obb_to_aabb
from .models import OBBAnnotation, vertebra_sort_key


def save_pedicle_labels_for_image(
    info: dict,
    pedicle_data: dict,
    cache: dict,
    converter: YOLOConverter,
    output_dir: str,
    padding_ratio: float = 0.15,
) -> int:
    """Write label files for one image's pedicle annotations.

    Same coordinate system and format as export_pedicle_crops, but only
    writes labels (no images / meta).

    Returns:
        Number of label files written.
    """
    img_path = info["image_path"]
    rel_path = info["rel_path"]
    label_path = info.get("label_path")
    img_w = info["width"]
    img_h = info["height"]
    split = info.get("split", "")
    source_stem = Path(img_path).stem

    if not pedicle_data:
        return 0

    # Load OBB annotations
    try:
        ann = converter.load_single(
            img_path, label_path, img_w, img_h,
            cache_entry=cache.get(rel_path, {}),
        )
    except Exception:
        return 0

    # Build vert_name -> OBB annotation mapping
    vert_obb_map: Dict[str, OBBAnnotation] = {}
    for vert_ann in ann.annotations:
        if vert_ann.shape_type == "obb" and vert_ann.class_name:
            vert_obb_map[vert_ann.class_name] = vert_ann

    # Determine label output directory
    out_root = Path(output_dir)
    if split:
        lbl_dir = out_root / split / "labels"
    else:
        lbl_dir = out_root / "labels"
    lbl_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    sorted_verts = sorted(pedicle_data.keys(), key=lambda v: vertebra_sort_key(v))

    for vert_name in sorted_verts:
        pdata = pedicle_data[vert_name]
        if vert_name not in vert_obb_map:
            continue

        # Skip vertebrae with no actual pedicle points on either side
        left_data = pdata.get("left", {})
        right_data = pdata.get("right", {})
        has_left = bool(
            left_data and left_data.get("center")
            and left_data.get("visibility", 0) > 0
        )
        has_right = bool(
            right_data and right_data.get("center")
            and right_data.get("visibility", 0) > 0
        )
        if not has_left and not has_right:
            continue

        vert_ann = vert_obb_map[vert_name]

        # Compute AABB + expand with padding
        aabb_x1, aabb_y1, aabb_x2, aabb_y2 = _obb_to_aabb(vert_ann)
        crop_x1, crop_y1, crop_x2, crop_y2 = _expand_and_clamp(
            aabb_x1, aabb_y1, aabb_x2, aabb_y2,
            padding_ratio, img_w, img_h,
        )
        if crop_x2 - crop_x1 < 4 or crop_y2 - crop_y1 < 4:
            continue

        # Convert pedicle points to crop coordinates
        crop_left = _to_crop_coords(left_data, crop_x1, crop_y1)
        crop_right = _to_crop_coords(right_data, crop_x1, crop_y1)

        # Write label file
        crop_stem = f"{source_stem}-{vert_name}"
        crop_lbl_path = lbl_dir / f"{crop_stem}.txt"
        _write_label_file(crop_lbl_path, crop_left, crop_right)
        written += 1

    return written


def export_pedicle_crops(
    image_infos: List[dict],
    pedicle_data: Dict[str, dict],
    cache: dict,
    converter: YOLOConverter,
    output_dir: str,
    padding_ratio: float = 0.15,
) -> Dict:
    """Export pedicle annotations as crop dataset.

    Args:
        image_infos: List of image info dicts from scan_dataset().
        pedicle_data: Dict of {rel_path: {vert_name: pedicle_dict}}.
        cache: Progress cache (for loading annotations).
        converter: YOLOConverter instance.
        output_dir: Output directory for crop dataset.
        padding_ratio: AABB expansion ratio (default 0.15).

    Returns:
        Dict with 'total_crops' and 'per_vert' counts.
    """
    out_root = Path(output_dir)
    total_crops = 0
    per_vert: Dict[str, int] = {}

    for info in image_infos:
        img_path = info["image_path"]
        rel_path = info["rel_path"]
        label_path = info.get("label_path")
        img_w = info["width"]
        img_h = info["height"]
        split = info.get("split", "")
        source_stem = Path(img_path).stem

        # Get pedicle data for this image
        img_pedicle = pedicle_data.get(rel_path, {})
        if not img_pedicle:
            continue

        # Load OBB annotations
        try:
            ann = converter.load_single(
                img_path, label_path, img_w, img_h,
                cache_entry=cache.get(rel_path, {}),
            )
        except Exception:
            continue

        # Build vert_name -> OBB annotation mapping
        vert_obb_map: Dict[str, OBBAnnotation] = {}
        for vert_ann in ann.annotations:
            if vert_ann.shape_type == "obb" and vert_ann.class_name:
                vert_obb_map[vert_ann.class_name] = vert_ann

        # Open source image
        try:
            with Image.open(img_path) as src_img:
                src_array = np.array(src_img)
        except Exception:
            continue

        # Setup split directories
        if split:
            img_dir = out_root / split / "images"
            meta_dir = out_root / split / "meta"
            lbl_dir = out_root / split / "labels"
        else:
            img_dir = out_root / "images"
            meta_dir = out_root / "meta"
            lbl_dir = out_root / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        meta_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        # Process each vertebra with pedicle data, sorted anatomically
        sorted_verts = sorted(
            img_pedicle.keys(),
            key=lambda v: vertebra_sort_key(v),
        )

        for vert_name in sorted_verts:
            pdata = img_pedicle[vert_name]
            if vert_name not in vert_obb_map:
                continue

            # Skip vertebrae with no actual pedicle points on either side
            left_data = pdata.get("left", {})
            right_data = pdata.get("right", {})
            has_left = bool(
                left_data and left_data.get("center")
                and left_data.get("visibility", 0) > 0
            )
            has_right = bool(
                right_data and right_data.get("center")
                and right_data.get("visibility", 0) > 0
            )
            if not has_left and not has_right:
                continue

            vert_ann = vert_obb_map[vert_name]

            # Compute AABB
            aabb_x1, aabb_y1, aabb_x2, aabb_y2 = _obb_to_aabb(vert_ann)

            # Expand with padding
            crop_x1, crop_y1, crop_x2, crop_y2 = _expand_and_clamp(
                aabb_x1, aabb_y1, aabb_x2, aabb_y2,
                padding_ratio, img_w, img_h,
            )

            # Skip degenerate
            if crop_x2 - crop_x1 < 4 or crop_y2 - crop_y1 < 4:
                continue

            # Crop filename
            crop_stem = f"{source_stem}-{vert_name}"
            crop_img_path = img_dir / f"{crop_stem}.jpg"
            crop_meta_path = meta_dir / f"{crop_stem}.json"
            crop_lbl_path = lbl_dir / f"{crop_stem}.txt"

            # Crop image
            crop_array = src_array[crop_y1:crop_y2, crop_x1:crop_x2]
            crop_h, crop_w = crop_array.shape[:2]
            Image.fromarray(crop_array).save(str(crop_img_path), quality=95)

            # Convert pedicle points to crop coordinates
            left_data = pdata.get("left", {})
            right_data = pdata.get("right", {})

            crop_left = _to_crop_coords(left_data, crop_x1, crop_y1)
            crop_right = _to_crop_coords(right_data, crop_x1, crop_y1)

            # Build meta
            meta = {
                "source_image": img_path,
                "source_rel_path": rel_path,
                "vertebra": vert_name,
                "internal_class_id": vert_ann.class_id,
                "export_class_id": 0,
                "source_image_size": [img_w, img_h],
                "crop_image_size": [crop_w, crop_h],
                "source_aabb_xyxy": [
                    round(aabb_x1, 3), round(aabb_y1, 3),
                    round(aabb_x2, 3), round(aabb_y2, 3),
                ],
                "crop_aabb_xyxy": [crop_x1, crop_y1, crop_x2, crop_y2],
                "source_obb_points": [
                    [round(p.x, 3), round(p.y, 3)] for p in vert_ann.points[:4]
                ],
                "padding_ratio": padding_ratio,
                "image_left": crop_left,
                "image_right": crop_right,
                "flagged": pdata.get("flagged", False),
            }
            with open(crop_meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)

            # Build label file
            # Line 1: 0 left_x left_y left_v
            # Line 2: 1 right_x right_y right_v
            _write_label_file(crop_lbl_path, crop_left, crop_right)

            total_crops += 1
            per_vert[vert_name] = per_vert.get(vert_name, 0) + 1

    return {
        "total_crops": total_crops,
        "per_vert": per_vert,
    }


def _to_crop_coords(data: dict, crop_x1: int, crop_y1: int) -> dict:
    """Convert full-image pedicle coordinates to crop coordinates."""
    if not data or not data.get("center"):
        return {}
    cx = data["center"]["x"] - crop_x1
    cy = data["center"]["y"] - crop_y1
    return {
        "center": {"x": round(cx, 3), "y": round(cy, 3)},
        "visibility": data.get("visibility", 0),
    }


def _write_label_file(path: Path, left: dict, right: dict):
    """Write label file with two lines per vertebra.

    Line 1 (left):  0  left_x  left_y  left_v
    Line 2 (right): 1  right_x  right_y  right_v

    Coordinates are in crop-image pixel space.
    Missing sides are omitted (no line written).
    """
    lines = []

    if left and left.get("center") and left.get("visibility", 0) > 0:
        lines.append(
            f"0\t{left['center']['x']:.3f}\t{left['center']['y']:.3f}\t{left['visibility']}"
        )
    if right and right.get("center") and right.get("visibility", 0) > 0:
        lines.append(
            f"1\t{right['center']['x']:.3f}\t{right['center']['y']:.3f}\t{right['visibility']}"
        )

    with open(path, "w") as f:
        f.write("\n".join(lines))
        if lines:
            f.write("\n")
