"""YOLO format reader/writer for AABB and OBB annotations."""

import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .models import ImageAnnotation, OBBAnnotation, Point


class YOLOConverter:
    """Convert between YOLO formats and internal annotation model."""

    def __init__(self, class_names: Optional[Dict[int, str]] = None):
        self.class_names = class_names or {
            0: "Vertebra",
            1: "scoliosis spine",
            2: "normal spine",
        }

    def load_dataset(self, dataset_root: str) -> Dict[str, ImageAnnotation]:
        """Load all images and their annotations from a YOLO dataset.
        
        Looks for train/images, valid/images, test/images directories.
        Returns dict mapping image_path -> ImageAnnotation.
        """
        root = Path(dataset_root)
        result = {}

        for split in ["train", "valid", "test"]:
            images_dir = root / split / "images"
            labels_dir = root / split / "labels"

            if not images_dir.exists():
                continue

            for img_path in sorted(images_dir.glob("*.jpg")):
                img_abs = str(img_path.resolve())
                label_path = labels_dir / (img_path.stem + ".txt")

                img = cv2.imread(img_abs)
                if img is None:
                    continue
                h_img, w_img = img.shape[:2]

                annotation = ImageAnnotation(
                    image_path=img_abs,
                    image_width=w_img,
                    image_height=h_img,
                )

                if label_path.exists():
                    annotation.annotations = self._load_labels(
                        label_path, w_img, h_img
                    )

                result[img_abs] = annotation

        return result

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
                # 4 corner points, normalized
                coords = []
                for p in ann.points:
                    coords.append(f"{p.x / w_img:.6f}")
                    coords.append(f"{p.y / h_img:.6f}")

                line = f"{ann.class_id} {' '.join(coords)}\n"
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
                cx, cy, w, h, angle = ann.to_xywhr()

                # Normalize angle to [-pi/4, pi/4)
                angle = self._normalize_angle(angle)

                line = (
                    f"{ann.class_id} "
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
