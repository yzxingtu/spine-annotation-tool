"""YOLO format reader/writer for AABB and OBB annotations."""

import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .models import ImageAnnotation, OBBAnnotation, Point


class YOLOConverter:
    """Convert between YOLO formats and internal annotation model."""

    def __init__(self, class_names: Optional[Dict[int, str]] = None):
        self.class_names = class_names or {
            0: "Vertebra",
            1: "scoliosis spine",
            2: "normal spine",
            3: "upper_end_vertebra",  # 上端椎
            4: "lower_end_vertebra",  # 下端椎
        }

    def scan_dataset(self, dataset_root: str) -> List[dict]:
        """Scan dataset and return list of image info (without loading pixels).
        
        Auto-detects: if selected dir contains train/valid/test, use as root.
        If selected dir IS a split (contains images/), scan directly.
        """
        root = Path(dataset_root)
        result = []

        # Detect if this is the dataset root (has train/, valid/, test/ subdirs)
        splits = ["train", "valid", "test"]
        is_root = any((root / s / "images").exists() for s in splits)

        if is_root:
            scan_dirs = [(root / s / "images", root / s / "labels") for s in splits]
        else:
            # Assume selected dir is already a split directory
            scan_dirs = [(root / "images", root / "labels")]

        for images_dir, labels_dir in scan_dirs:
            if not images_dir.exists():
                continue

            for img_path in sorted(images_dir.glob("*.jpg")):
                img_abs = str(img_path.resolve())
                label_path = labels_dir / (img_path.stem + ".txt")

                # Get image dimensions without loading full pixel data
                from PIL import Image
                try:
                    with Image.open(img_abs) as img:
                        w_img, h_img = img.size
                except Exception:
                    continue

                result.append({
                    "image_path": img_abs,
                    "label_path": str(label_path) if label_path.exists() else None,
                    "width": w_img,
                    "height": h_img,
                    "has_labels": label_path.exists(),
                })

        return result

    def load_single(self, image_path: str, label_path: Optional[str],
                    img_w: int, img_h: int, cache_entry: Optional[dict] = None) -> ImageAnnotation:
        """Load annotations for a single image on demand.
        
        Args:
            cache_entry: Optional cache entry to restore end vertebra markers
        """
        annotation = ImageAnnotation(
            image_path=image_path,
            image_width=img_w,
            image_height=img_h,
        )

        if label_path and Path(label_path).exists():
            annotation.annotations = self._load_labels(
                Path(label_path), img_w, img_h
            )
            
            # Restore end vertebra markers from cache
            if cache_entry and "annotation_states" in cache_entry:
                states = cache_entry["annotation_states"]
                for i, ann in enumerate(annotation.annotations):
                    if i < len(states):
                        ann.is_upper_end = states[i].get("is_upper_end", False)
                        ann.is_lower_end = states[i].get("is_lower_end", False)

        return annotation

    def save_progress_cache(self, cache_path: str, progress: dict):
        """Save progress cache to JSON file (includes end vertebra markers)."""
        import json
        with open(cache_path, "w") as f:
            json.dump(progress, f, indent=2)

    def load_progress_cache(self, cache_path: str) -> dict:
        """Load progress cache from JSON file."""
        import json
        if not Path(cache_path).exists():
            return {}
        with open(cache_path, "r") as f:
            return json.load(f)

    def save_annotation_state(self, annotation: ImageAnnotation, cache: dict):
        """Save annotation state including end vertebra markers to cache."""
        img_path = annotation.image_path
        ann_states = []
        for ann in annotation.annotations:
            ann_states.append({
                "is_upper_end": ann.is_upper_end,
                "is_lower_end": ann.is_lower_end,
            })
        cache[img_path] = {
            "saved": True,
            "modified": annotation.modified,
            "annotation_states": ann_states,
        }

    def _load_labels(self, label_path: Path,
                     img_w: int, img_h: int) -> List[OBBAnnotation]:
        """Load YOLOv5 format labels and convert to OBB annotations."""
        annotations = []

        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                class_id = int(parts[0])
                cx_norm = float(parts[1])
                cy_norm = float(parts[2])
                w_norm = float(parts[3])
                h_norm = float(parts[4])

                # Convert to pixel coordinates
                cx = cx_norm * img_w
                cy = cy_norm * img_h
                w = w_norm * img_w
                h = h_norm * img_h

                class_name = self.class_names.get(class_id, f"class_{class_id}")
                ann = OBBAnnotation.from_aabb(class_id, class_name, cx, cy, w, h)
                annotations.append(ann)

        return annotations

    def save_obb_yolov8(self, annotation: ImageAnnotation,
                        output_dir: str, overwrite: bool = False):
        """Save annotations in YOLOv8-OBB format.
        
        Format: class_id x1 y1 x2 y2 x3 y3 x4 y4 (normalized)
        """
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        img_path = Path(annotation.image_path)
        label_name = img_path.stem + ".txt"
        label_path = output / label_name

        if label_path.exists() and not overwrite:
            return False

        w_img = annotation.image_width
        h_img = annotation.image_height

        with open(label_path, "w") as f:
            for ann in annotation.annotations:
                # Determine effective class_id: end vertebra markers override class
                eff_class = ann.class_id
                if ann.class_id == 0:  # Only vertebrae can be end vertebrae
                    if ann.is_upper_end:
                        eff_class = 3  # upper_end_vertebra
                    elif ann.is_lower_end:
                        eff_class = 4  # lower_end_vertebra

                # 4 corner points, normalized
                coords = []
                for p in ann.points:
                    coords.append(f"{p.x / w_img:.6f}")
                    coords.append(f"{p.y / h_img:.6f}")

                line = f"{eff_class} {' '.join(coords)}\n"
                f.write(line)

        return True

    def save_obb_xywhr(self, annotation: ImageAnnotation,
                       output_dir: str, overwrite: bool = False):
        """Save annotations in YOLOv8-OBB xywhr format.
        
        Format: class_id cx cy w h angle (normalized, angle in radians [-pi/4, pi/4))
        """
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        img_path = Path(annotation.image_path)
        label_name = img_path.stem + ".txt"
        label_path = output / label_name

        if label_path.exists() and not overwrite:
            return False

        w_img = annotation.image_width
        h_img = annotation.image_height

        with open(label_path, "w") as f:
            for ann in annotation.annotations:
                # Determine effective class_id
                eff_class = ann.class_id
                if ann.class_id == 0:
                    if ann.is_upper_end:
                        eff_class = 3
                    elif ann.is_lower_end:
                        eff_class = 4

                cx, cy, w, h, angle = ann.to_xywhr()

                # Normalize angle to [-pi/4, pi/4)
                angle = self._normalize_angle(angle)

                line = (
                    f"{eff_class} "
                    f"{cx / w_img:.6f} {cy / h_img:.6f} "
                    f"{w / w_img:.6f} {h / h_img:.6f} "
                    f"{angle:.6f}\n"
                )
                f.write(line)

        return True

    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-pi/4, pi/4) range for YOLOv8-OBB."""
        # Reduce to [-pi/2, pi/2)
        angle = angle % math.pi
        if angle >= math.pi / 2:
            angle -= math.pi

        # If angle is outside [-pi/4, pi/4), swap width/height
        if angle >= math.pi / 4:
            angle -= math.pi / 2
        elif angle < -math.pi / 4:
            angle += math.pi / 2

        return angle

    @staticmethod
    def load_yaml_config(yaml_path: str) -> dict:
        """Load dataset YAML config."""
        import yaml
        with open(yaml_path, "r") as f:
            return yaml.safe_load(f)
