"""Crop dataset scanner, loader, and label saver for pedicle annotation."""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .models import (
    CropPedicleAnnotation,
    PediclePoint,
    Point,
    vertebra_sort_key,
)


class CropConverter:
    """Scan, load, and save crop pedicle annotations."""

    # cache 中存放元信息（如 last_image_path）的特殊 key
    META_KEY = "__meta__"

    # ------------------------------------------------------------------
    # Dataset validation & scanning
    # ------------------------------------------------------------------

    def validate_crop_dataset(self, dataset_root: str) -> Tuple[bool, str]:
        """Validate if directory is a valid crop dataset.

        Expected structure (with or without splits):
          root/images/ + root/meta/
          root/{split}/images/ + root/{split}/meta/

        Returns: (is_valid, message)
        """
        root = Path(dataset_root)
        if not root.exists():
            return False, f"目录不存在: {dataset_root}"
        if not root.is_dir():
            return False, f"不是目录: {dataset_root}"

        splits = ["train", "valid", "test"]
        has_splits = any((root / s / "images").exists() for s in splits)

        if has_splits:
            found = []
            for s in splits:
                img_dir = root / s / "images"
                if img_dir.exists():
                    count = len(list(img_dir.glob("*.jpg")))
                    found.append(f"{s}: {count} 张")
            message = "、".join(found) if found else "无图片"
            return True, message
        else:
            img_dir = root / "images"
            if not img_dir.exists():
                return False, (
                    "不是有效的 crop 数据集目录。\n\n"
                    "期望结构：\n"
                    "  根目录/images/ + 根目录/meta/\n"
                    "或按 split 分组：\n"
                    "  根目录/train/images/ + 根目录/train/meta/\n"
                    "  根目录/valid/images/ + 根目录/valid/meta/"
                )
            count = len(list(img_dir.glob("*.jpg")))
            return True, f"{count} 张"

    def scan_crop_dataset(self, dataset_root: str) -> List[dict]:
        """Scan crop dataset and return list of crop info dicts.

        Each dict contains:
          - image_path: absolute path to crop image
          - rel_path: relative path (for cache key)
          - meta_path: absolute path to meta JSON
          - meta: parsed meta dict
          - split: split name (train/valid/test or '')
          - vertebra: vertebra name (e.g. "C7")
          - source_stem: source image stem
          - label_path: path to label file (may not exist yet)
          - width, height: crop image dimensions
        """
        from PIL import Image

        root = Path(dataset_root)
        result = []

        splits = ["train", "valid", "test"]
        has_splits = any((root / s / "images").exists() for s in splits)

        if has_splits:
            scan_dirs = [(root / s, s) for s in splits if (root / s / "images").exists()]
        else:
            scan_dirs = [(root, "")]

        for base_dir, split_name in scan_dirs:
            img_dir = base_dir / "images"
            meta_dir = base_dir / "meta"
            lbl_dir = base_dir / "labels"

            for img_path in sorted(img_dir.glob("*.jpg")):
                img_abs = str(img_path.resolve())
                crop_stem = img_path.stem

                # Try to find matching meta file
                meta_path = meta_dir / f"{crop_stem}.json"
                meta = {}
                if meta_path.exists():
                    try:
                        with open(meta_path, "r", encoding="utf-8") as f:
                            meta = json.load(f)
                    except Exception:
                        pass

                # Label path (may not exist yet)
                label_path = lbl_dir / f"{crop_stem}.txt"

                # Get image dimensions
                try:
                    with Image.open(img_abs) as img:
                        w, h = img.size
                except Exception:
                    continue

                # Compute rel_path for cache key
                rel_path = str(img_path.resolve().relative_to(root.resolve())).replace("\\", "/")

                # Parse filename to extract source_stem and vertebra
                # Expected format: {source_stem}-{vertebra_name}
                parts = crop_stem.rsplit("-", 1)
                source_stem = parts[0] if len(parts) > 1 else crop_stem
                vertebra = parts[1] if len(parts) > 1 else meta.get("vertebra", "?")

                result.append({
                    "image_path": img_abs,
                    "rel_path": rel_path,
                    "meta_path": str(meta_path) if meta_path.exists() else None,
                    "meta": meta,
                    "split": split_name,
                    "vertebra": vertebra,
                    "source_stem": source_stem,
                    "label_path": str(label_path) if label_path.exists() else None,
                    "width": w,
                    "height": h,
                })

        return sorted(result, key=lambda info: (
            info["source_stem"],
            vertebra_sort_key(info["vertebra"]),
        ))

    # ------------------------------------------------------------------
    # Annotation loading & saving
    # ------------------------------------------------------------------

    def load_crop_annotation(
        self, image_path: str, label_path: Optional[str],
        img_w: int, img_h: int,
        cache_entry: Optional[dict] = None,
    ) -> CropPedicleAnnotation:
        """Load pedicle annotation for a single crop image.

        Priority:
        1. If cache_entry has pedicle_states, use that (latest edit state)
        2. Otherwise read from label file

        Label format: class_id left_x left_y left_v right_x right_y right_v
        (coordinates normalized to crop image size)
        """
        annotation = CropPedicleAnnotation(
            image_path=image_path,
            image_width=img_w,
            image_height=img_h,
        )

        # Try loading from label file first
        if label_path and Path(label_path).exists():
            try:
                with open(label_path, "r") as f:
                    line = f.readline().strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 7:
                        # class_id = int(parts[0])  # always 0 for now
                        lx = float(parts[1]) * img_w
                        ly = float(parts[2]) * img_h
                        lv = int(parts[3])
                        rx = float(parts[4]) * img_w
                        ry = float(parts[5]) * img_h
                        rv = int(parts[6])

                        annotation.image_left = PediclePoint(
                            center=Point(lx, ly) if lv > 0 else None,
                            visibility=lv,
                        )
                        annotation.image_right = PediclePoint(
                            center=Point(rx, ry) if rv > 0 else None,
                            visibility=rv,
                        )
            except Exception:
                pass

        # Override with cache state if available (latest edit takes priority)
        if cache_entry and "pedicle_states" in cache_entry:
            states = cache_entry["pedicle_states"]
            if "left" in states:
                ls = states["left"]
                annotation.image_left = PediclePoint(
                    center=Point(ls["x"], ls["y"]) if ls.get("center") else None,
                    visibility=ls.get("visibility", 0),
                )
            if "right" in states:
                rs = states["right"]
                annotation.image_right = PediclePoint(
                    center=Point(rs["x"], rs["y"]) if rs.get("center") else None,
                    visibility=rs.get("visibility", 0),
                )

        return annotation

    def build_pedicle_states(self, annotation: CropPedicleAnnotation) -> dict:
        """Extract serializable pedicle states from annotation for cache."""
        states = {}
        for side, pt in [("left", annotation.image_left), ("right", annotation.image_right)]:
            states[side] = {
                "center": pt.center is not None,
                "x": round(pt.center.x, 3) if pt.center else 0,
                "y": round(pt.center.y, 3) if pt.center else 0,
                "visibility": pt.visibility,
            }
        return states

    def save_crop_label(
        self, annotation: CropPedicleAnnotation, output_path: str,
    ) -> bool:
        """Save crop pedicle label to a .txt file.

        Format: class_id left_x left_y left_v right_x right_y right_v
        Coordinates normalized to crop image dimensions.
        v=0 with no center point writes 0 0 0.
        """
        w = annotation.image_width
        h = annotation.image_height
        if w <= 0 or h <= 0:
            return False

        lp = annotation.image_left
        rp = annotation.image_right

        # Normalize coordinates
        if lp.center and lp.visibility > 0:
            lx = round(lp.center.x / w, 6)
            ly = round(lp.center.y / h, 6)
        else:
            lx, ly = 0.0, 0.0

        if rp.center and rp.visibility > 0:
            rx = round(rp.center.x / w, 6)
            ry = round(rp.center.y / h, 6)
        else:
            rx, ry = 0.0, 0.0

        line = f"0 {lx} {ly} {lp.visibility} {rx} {ry} {rp.visibility}\n"

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            f.write(line)
        return True

    # ------------------------------------------------------------------
    # Progress cache (same pattern as YOLOConverter)
    # ------------------------------------------------------------------

    def load_progress_cache(self, cache_path: str) -> dict:
        """Load progress cache from JSON file."""
        if not Path(cache_path).exists():
            return {}
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def save_progress_cache(self, cache_path: str, progress: dict):
        """Save progress cache to JSON file."""
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(progress, f, indent=2, ensure_ascii=False)

    def get_last_image_path(self, cache: dict) -> Optional[str]:
        """Read last viewed image path from cache."""
        meta = cache.get(self.META_KEY, {})
        return meta.get("last_image_path") if isinstance(meta, dict) else None

    def set_last_image_path(self, cache: dict, image_path: str) -> None:
        """Set last viewed image path in cache (caller must persist)."""
        meta = cache.setdefault(self.META_KEY, {})
        meta["last_image_path"] = image_path
