"""Generate single-vertebra crop dataset from full-spine OBB annotations."""

import json
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from .converter import YOLOConverter
from .models import VERTEBRA_CLASSES, OBBAnnotation, vertebra_sort_key


def _obb_to_aabb(ann: OBBAnnotation) -> Tuple[float, float, float, float]:
    """Compute AABB (x1, y1, x2, y2) from OBB 4 corner points."""
    xs = [p.x for p in ann.points[:4]]
    ys = [p.y for p in ann.points[:4]]
    return min(xs), min(ys), max(xs), max(ys)


def _expand_and_clamp(
    x1: float, y1: float, x2: float, y2: float,
    padding_ratio: float, img_w: int, img_h: int,
) -> Tuple[int, int, int, int]:
    """Expand AABB by padding_ratio and clamp to image bounds."""
    w = x2 - x1
    h = y2 - y1
    pad_x = w * padding_ratio
    pad_y = h * padding_ratio
    cx1 = max(0, int(round(x1 - pad_x)))
    cy1 = max(0, int(round(y1 - pad_y)))
    cx2 = min(img_w, int(round(x2 + pad_x)))
    cy2 = min(img_h, int(round(y2 + pad_y)))
    return cx1, cy1, cx2, cy2


def generate_crops(
    image_infos: List[dict],
    cache: dict,
    converter: YOLOConverter,
    output_dir: str,
    padding_ratio: float = 0.15,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Dict:
    """Generate single-vertebra crop dataset from full-spine annotations.

    Args:
        image_infos: List of image info dicts from scan_dataset().
        cache: Progress cache dict (for reading latest annotation states).
        converter: YOLOConverter instance for loading annotations.
        output_dir: Root directory for the crop dataset output.
        padding_ratio: Padding expansion ratio for AABB (default 0.15).
        progress_callback: Optional callback(current, total, message).

    Returns:
        Summary dict with 'total_crops', 'per_source', and diagnostic 'stats'.
    """
    out_root = Path(output_dir)
    total = len(image_infos)
    total_crops = 0
    per_source: Dict[str, int] = {}

    # Diagnostic counters
    stats = {
        "total_images": total,
        "images_no_annotations": 0,
        "images_read_error": 0,
        "total_annotations_found": 0,
        "skipped_line_type": 0,
        "skipped_invalid_name": 0,
        "skipped_degenerate": 0,
    }
    errors: List[str] = []

    for idx, info in enumerate(image_infos):
        img_path = info["image_path"]
        rel_path = info["rel_path"]
        label_path = info.get("label_path")
        img_w = info["width"]
        img_h = info["height"]
        split = info.get("split", "")
        source_stem = Path(img_path).stem

        # Load annotations (cache takes priority for latest geometry)
        cache_entry = cache.get(rel_path, {})
        try:
            ann = converter.load_single(
                img_path, label_path, img_w, img_h, cache_entry=cache_entry,
            )
        except Exception as exc:
            errors.append(f"{source_stem}: 加载标注失败 - {exc}")
            stats["images_no_annotations"] += 1
            continue

        # Skip images with no annotations
        if not ann.annotations:
            stats["images_no_annotations"] += 1
            if progress_callback:
                progress_callback(idx + 1, total, f"跳过（无标注）: {source_stem}")
            continue

        stats["total_annotations_found"] += len(ann.annotations)

        # Open source image (read-only, never modified)
        try:
            with Image.open(img_path) as src_img:
                src_array = np.array(src_img)
        except Exception as exc:
            stats["images_read_error"] += 1
            errors.append(f"{source_stem}: 无法读取图片 - {exc}")
            if progress_callback:
                progress_callback(idx + 1, total, f"无法读取: {source_stem}")
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

        crop_count = 0
        # Sort annotations by anatomical order before generating crops
        sorted_annotations = sorted(
            ann.annotations,
            key=lambda a: vertebra_sort_key(a.class_name or ""),
        )
        for vert_ann in sorted_annotations:
            # Only process OBB type (skip line annotations)
            if vert_ann.shape_type != "obb":
                stats["skipped_line_type"] += 1
                continue
            # Only process vertebrae with valid class names
            vert_name = vert_ann.class_name  # e.g. "C7", "T1", "S1"
            if not vert_name or vert_name not in VERTEBRA_CLASSES.values():
                stats["skipped_invalid_name"] += 1
                continue

            # Compute AABB from OBB 4 corner points
            aabb_x1, aabb_y1, aabb_x2, aabb_y2 = _obb_to_aabb(vert_ann)

            # Expand with padding and clamp
            crop_x1, crop_y1, crop_x2, crop_y2 = _expand_and_clamp(
                aabb_x1, aabb_y1, aabb_x2, aabb_y2,
                padding_ratio, img_w, img_h,
            )

            # Skip degenerate crops
            if crop_x2 - crop_x1 < 4 or crop_y2 - crop_y1 < 4:
                stats["skipped_degenerate"] += 1
                continue

            # Crop filename: {source_stem}-{vertebra_name}
            crop_stem = f"{source_stem}-{vert_name}"
            crop_img_path = img_dir / f"{crop_stem}.jpg"
            crop_meta_path = meta_dir / f"{crop_stem}.json"

            # Crop image (numpy array slicing)
            crop_array = src_array[crop_y1:crop_y2, crop_x1:crop_x2]
            crop_h, crop_w = crop_array.shape[:2]

            # Save crop image
            Image.fromarray(crop_array).save(str(crop_img_path), quality=95)

            # Build and save meta JSON
            meta = {
                "source_image": info["image_path"],
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
            }
            with open(crop_meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)

            crop_count += 1
            total_crops += 1

        per_source[source_stem] = crop_count
        if progress_callback:
            progress_callback(
                idx + 1, total,
                f"{source_stem}: 生成 {crop_count} 张 crop",
            )

    return {
        "total_crops": total_crops,
        "per_source": per_source,
        "stats": stats,
        "errors": errors,
    }
