"""YOLO format reader/writer for AABB and OBB annotations."""

import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .models import (
    EXPORT_CLASS_ID_S1,
    EXPORT_CLASS_ID_VERTEBRA,
    INTERNAL_CLASS_ID_S1,
    VERTEBRA_CLASSES,
    ImageAnnotation,
    OBBAnnotation,
    Point,
    to_export_class_id,
)


class YOLOConverter:
    """Convert between YOLO formats and internal annotation model."""

    # 旧版 class_id（已废弃，仅用于向后兼容加载）
    _LEGACY_CLASS_NAMES = {
        0: "Vertebra",
        1: "scoliosis spine",
        2: "normal spine",
    }

    def __init__(self, class_names: Optional[Dict[int, str]] = None):
        self.class_names = class_names or VERTEBRA_CLASSES.copy()

    def validate_dataset(self, dataset_root: str) -> Tuple[bool, str]:
        """Validate if directory is a valid YOLO dataset.
        
        Returns: (is_valid, message)
        """
        root = Path(dataset_root)
        if not root.exists():
            return False, f"目录不存在: {dataset_root}"
        if not root.is_dir():
            return False, f"不是目录: {dataset_root}"

        splits = ["train", "valid", "test"]
        is_root = any((root / s / "images").exists() for s in splits)

        if is_root:
            found = []
            for s in splits:
                img_dir = root / s / "images"
                if img_dir.exists():
                    count = len(list(img_dir.glob("*.jpg"))) + len(list(img_dir.glob("*.png")))
                    found.append(f"{s}: {count} 张")
            if not found:
                return False, "在 train/valid/test 目录下未找到图片（支持 .jpg/.png）"
            return True, f"检测到数据集结构: {', '.join(found)}"
        else:
            # Check if it's a single split directory
            img_dir = root / "images"
            if img_dir.exists():
                count = len(list(img_dir.glob("*.jpg"))) + len(list(img_dir.glob("*.png")))
                if count > 0:
                    return True, f"检测到单分片目录，包含 {count} 张图片"
            # Check if images are directly in this dir
            jpg_count = len(list(root.glob("*.jpg")))
            png_count = len(list(root.glob("*.png")))
            if jpg_count + png_count > 0:
                return False, (
                    f"目录中有 {jpg_count + png_count} 张图片，但不符合 YOLO 数据集格式。\n\n"
                    "期望格式：\n"
                    "  根目录/\n"
                    "    train/images/ + train/labels/\n"
                    "    valid/images/ + valid/labels/\n"
                    "    test/images/ + test/labels/\n\n"
                    "或者单分片：\n"
                    "  目录/\n"
                    "    images/\n"
                    "    labels/"
                )
            return False, (
                "不是有效的 YOLO 数据集目录。\n\n"
                "期望格式：\n"
                "  根目录/\n"
                "    train/images/ + train/labels/\n"
                "    valid/images/ + valid/labels/\n"
                "    test/images/ + test/labels/\n\n"
                "或者单分片：\n"
                "  目录/images/ + 目录/labels/"
            )

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
            scan_dirs = [(root / s / "images", root / s / "labels", s) for s in splits]
        else:
            # Assume selected dir is already a split directory
            scan_dirs = [(root / "images", root / "labels", root.name)]

        for images_dir, labels_dir, split_name in scan_dirs:
            if not images_dir.exists():
                continue

            for img_path in sorted(images_dir.glob("*.jpg")):
                img_abs = str(img_path.resolve())
                img_rel = str(img_path.resolve().relative_to(root.resolve())).replace("\\", "/")
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
                    "rel_path": img_rel,
                    "label_path": str(label_path) if label_path.exists() else None,
                    "width": w_img,
                    "height": h_img,
                    "has_labels": label_path.exists(),
                    "split": split_name,
                })

        return result

    # cache 中存放元信息（如 last_image_path）的特殊 key，
    # 以双下划线包围避免与真实图片路径冲突
    META_KEY = "__meta__"

    @staticmethod
    def migrate_cache_to_rel_paths(cache: dict, image_infos: list) -> dict:
        """将旧版绝对路径 cache key 迁移为相对路径（rel_path）。

        兼容处理：
        - 旧 cache 使用绝对路径（如 D:/dataset/train/images/001.jpg）
        - 新 cache 使用相对于数据集根目录的路径（如 train/images/001.jpg）
        - 通过文件名匹配进行迁移
        - 同时处理 __meta__.last_image_path 中的绝对路径

        Returns:
            迁移后的 cache（若无变化则原样返回）
        """
        import os
        if not cache:
            return cache

        # 检测是否存在旧格式：顶层绝对路径 key 或 __meta__.last_image_path 为绝对路径
        has_old_keys = any(
            os.path.isabs(k) for k in cache if k != YOLOConverter.META_KEY
        )
        meta = cache.get(YOLOConverter.META_KEY, {})
        last_path = meta.get("last_image_path", "") if isinstance(meta, dict) else ""
        has_old_meta = bool(last_path) and os.path.isabs(last_path)

        if not has_old_keys and not has_old_meta:
            return cache

        # 建立文件名 → rel_path 的映射（处理同名冲突时取第一个）
        stem_to_rel: dict[str, str] = {}
        for info in image_infos:
            stem = Path(info["image_path"]).stem
            stem_to_rel.setdefault(stem, info["rel_path"])

        new_cache: dict = {}
        migrated_count = 0
        for key, value in cache.items():
            if key == YOLOConverter.META_KEY:
                # last_image_path 也迁移为相对路径
                if isinstance(value, dict) and "last_image_path" in value:
                    old_path = value["last_image_path"]
                    stem = Path(old_path).stem
                    if stem in stem_to_rel:
                        value = dict(value)
                        value["last_image_path"] = stem_to_rel[stem]
                        migrated_count += 1
                    elif os.path.isabs(old_path):
                        # 找不到匹配文件：清除绝对路径，避免断点续标跳转失败
                        value = dict(value)
                        value.pop("last_image_path", None)
                        migrated_count += 1
                new_cache[key] = value
            elif os.path.isabs(key):
                stem = Path(key).stem
                if stem in stem_to_rel:
                    new_cache[stem_to_rel[stem]] = value
                    migrated_count += 1
                # 找不到匹配的文件则丢弃旧 key
            else:
                new_cache[key] = value

        return new_cache

    def load_single(self, image_path: str, label_path: Optional[str],
                    img_w: int, img_h: int,
                    cache_entry: Optional[dict] = None) -> ImageAnnotation:
        """Load annotations for a single image on demand.

        加载优先级：
        1. 若 cache_entry 中含有完整 `points`（OBB 四角点像素坐标），
           使用 cache 的几何状态重建（保留之前的旋转/移动/可见性编辑）
        2. 否则从原始 YOLO label 文件读 AABB（水平矩形）

        Args:
            cache_entry: 可选，用于从缓存恢复每个标注的 OBB 几何与可见性
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

            # 从 cache 恢复编辑后的 OBB 几何 + keypoint_visibility
            # cache 是用户最新编辑状态的权威来源（以数量为准）：
            # - 文件标注数 > cache state 数：用户删除了标注，截断文件多余部分
            # - 文件标注数 < cache state 数：用户新增了标注，从 cache 追加
            # - 文件标注数 = cache state 数：一一对应恢复
            # 注：使用位置匹配（而非 class_id），因新 2 类格式下 class_id 可能全部为 0
            if cache_entry and "annotation_states" in cache_entry:
                states = cache_entry["annotation_states"]

                # 以 cache 数量为准，截断或保留文件标注
                file_count = len(annotation.annotations)
                cache_count = len(states)
                match_count = min(file_count, cache_count)

                # 对位置重叠的标注恢复 cache 状态
                for i in range(match_count):
                    self._restore_annotation_state(
                        annotation.annotations[i], states[i]
                    )

                # 截断文件多出的标注（用户已删除）
                if cache_count < file_count:
                    annotation.annotations = annotation.annotations[:cache_count]

                # 追加 cache 中多出的新增标注（用户新绘制尚未保存）
                for i in range(file_count, cache_count):
                    new_ann = self._create_annotation_from_state(
                        states[i], img_w, img_h
                    )
                    if new_ann is not None:
                        annotation.annotations.append(new_ann)

        elif cache_entry and "annotation_states" in cache_entry:
            # 文件不存在，但 cache 中有状态：完全从 cache 恢复
            states = cache_entry["annotation_states"]
            for state in states:
                new_ann = self._create_annotation_from_state(
                    state, img_w, img_h
                )
                if new_ann is not None:
                    annotation.annotations.append(new_ann)

        return annotation

    def _restore_annotation_state(self, ann: OBBAnnotation, state: dict):
        """从 cache state 恢复单个标注的类别、几何与可见性状态。

        必须同时恢复 class_id/class_name，否则用户修改编号后未保存就切图，
        再切回时会丢失编号修改（因为 .txt 未更新而 cache 已更新）。
        """
        # 恢复类别编号（优先以 cache 为准，因为用户可能已改编号但未导出 .txt）
        cached_class_id = state.get("class_id")
        cached_class_name = state.get("class_name")
        if cached_class_id is not None:
            ann.class_id = int(cached_class_id)
        if cached_class_name is not None:
            ann.class_name = str(cached_class_name)

        # 恢复几何
        pts = state.get("points")
        shape_type = state.get("shape_type", "obb")
        if shape_type == "line":
            if (
                isinstance(pts, list) and len(pts) == 2
                and all(isinstance(p, (list, tuple)) and len(p) == 2 for p in pts)
            ):
                ann.points = [Point(float(p[0]), float(p[1])) for p in pts]
                ann.shape_type = "line"
                ann._update_geometry()
        elif (
            isinstance(pts, list) and len(pts) == 4
            and all(isinstance(p, (list, tuple)) and len(p) == 2 for p in pts)
        ):
            ann.points = [Point(float(p[0]), float(p[1])) for p in pts]
            ann._update_geometry()
        ann.keypoint_visibility = int(state.get("keypoint_visibility", 2))

    def _create_annotation_from_state(self, state: dict,
                                       img_w: int, img_h: int) -> Optional[OBBAnnotation]:
        """从 cache state 创建新的 OBBAnnotation（用于恢复新增标注）。"""
        class_id = state.get("class_id")
        class_name = state.get("class_name")
        pts = state.get("points")
        shape_type = state.get("shape_type", "obb")

        if class_id is None or class_name is None:
            return None
        if not isinstance(pts, list) or len(pts) < 2:
            return None

        try:
            if shape_type == "line" and len(pts) == 2:
                p0x, p0y = float(pts[0][0]), float(pts[0][1])
                p1x, p1y = float(pts[1][0]), float(pts[1][1])
                ann = OBBAnnotation.from_line(class_id, class_name, p0x, p0y, p1x, p1y)
            elif len(pts) == 4:
                points = [Point(float(p[0]), float(p[1])) for p in pts]
                ann = OBBAnnotation(class_id, class_name, points)
                ann._update_geometry()
            else:
                return None
        except (ValueError, TypeError, IndexError):
            return None

        ann.keypoint_visibility = int(state.get("keypoint_visibility", 2))
        ann.shape_type = shape_type
        return ann

    def save_progress_cache(self, cache_path: str, progress: dict):
        """Save progress cache to JSON file."""
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

    # --- 清空标注数据 ---

    def collect_label_files(self, output_dir: str, split: str) -> List[Path]:
        """收集指定 split 下所有 *.txt 训练标注文件。根据保存逻辑两种布局：
          - split 非空：{output_dir}/{split}/labels/*.txt
          - split 为空：{output_dir}/*.txt
        仅返回真实存在的普通文件（排除软链接）。
        """
        if not output_dir:
            return []
        out = Path(output_dir).resolve()
        if not out.exists() or not out.is_dir():
            return []

        if split:
            d = out / split / "labels"
            if not (d.exists() and d.is_dir() and not d.is_symlink()):
                return []
            candidates = sorted(d.glob("*.txt"))
        else:
            candidates = sorted(out.glob("*.txt"))

        result: List[Path] = []
        for p in candidates:
            if p.is_file() and not p.is_symlink():
                result.append(p)
        return result

    def clear_outputs(self, output_dir: str, split: str) -> dict:
        """删除指定 split 下所有 *.txt 训练标注文件（不可恢复）。

        仅删除 .txt 文件本身，不删除目录，不递归。
        返回: {'deleted': N, 'failed': [(path, err), ...], 'paths': [str, ...]}
        """
        files = self.collect_label_files(output_dir, split)
        deleted: List[str] = []
        failed: List[Tuple[str, str]] = []
        for p in files:
            try:
                p.unlink()
                deleted.append(str(p))
            except OSError as e:
                failed.append((str(p), str(e)))
        return {"deleted": len(deleted), "failed": failed, "paths": deleted}

    def clear_progress_cache(self, cache_path: str) -> bool:
        """删除进度缓存文件。返回是否实际删除了文件。"""
        if not cache_path:
            return False
        p = Path(cache_path)
        if not p.exists() or not p.is_file() or p.is_symlink():
            return False
        try:
            p.unlink()
            return True
        except OSError:
            return False

    def clear_label_for(self, image_stem: str, output_dir: str, split: str) -> bool:
        """删除单张图片对应的标注文件。返回是否实际删除了文件。"""
        if not image_stem or not output_dir:
            return False
        out = Path(output_dir).resolve()
        if split:
            label_path = out / split / "labels" / f"{image_stem}.txt"
        else:
            label_path = out / f"{image_stem}.txt"
        if not label_path.exists() or not label_path.is_file() or label_path.is_symlink():
            return False
        try:
            label_path.unlink()
            return True
        except OSError:
            return False

    # --- cache 元信息 helpers ---

    def get_last_image_path(self, cache: dict) -> Optional[str]:
        """读取上次编辑的图片路径（用于启动时智能跳转）。"""
        meta = cache.get(self.META_KEY, {})
        return meta.get("last_image_path")

    def set_last_image_path(self, cache: dict, image_path: str) -> None:
        """记录当前正在编辑的图片路径到 cache（不写盘，调用方负责落盘）。"""
        meta = cache.setdefault(self.META_KEY, {})
        meta["last_image_path"] = image_path

    def build_annotation_states(self, annotation: ImageAnnotation) -> list:
        """从 ImageAnnotation 提取每个标注的可序列化状态，用于写入 cache。

        保存的状态：
          - class_id / class_name: 标注类别（新增标注恢复时需要）
          - points: 4 个角点或 2 个端点的像素坐标 [[x, y], ...]
          - keypoint_visibility: YOLOv8-pose v 字段
          - shape_type: 'obb' 或 'line'
        """
        states = []
        for ann in annotation.annotations:
            states.append({
                "class_id": ann.class_id,
                "class_name": ann.class_name,
                "points": [[round(p.x, 3), round(p.y, 3)] for p in ann.points],
                "keypoint_visibility": int(ann.keypoint_visibility),
                "shape_type": ann.shape_type,
            })
        return states

    def _load_labels(self, label_path: Path,
                     img_w: int, img_h: int) -> List[OBBAnnotation]:
        """Load labels from YOLO format file (auto-detect AABB / OBB / pose).

        支持的格式：
        - AABB (5 fields): class_id cx cy w h  →  旧 YOLOv5 格式
        - OBB  (9 fields): class_id x1 y1 x2 y2 x3 y3 x4 y4  →  YOLOv8-OBB
        - Pose (17 fields): class_id cx cy w h x1 y1 v1 x2 y2 v2 x3 y3 v3 x4 y4 v4

        class_id 兼容矩阵：
        - 旧 YOLOv5 AABB：class_id 1/2 = 脊柱整体外框 (scoliosis/normal spine)，跳过；
                          class_id 0 = 泛化 "Vertebra"，按 center.y 自动编号为 C7→L5
        - 旧解剖学 OBB/pose（class_id ∈ 0..18 对应 C7..S1）：直接按 VERTEBRA_CLASSES 加载
        - 新 2 类 OBB/pose（class_id ∈ {0=vertebra, 1=S1}）：按 y 排序，
                          class_id=1 → S1 (内部 18)，class_id=0 → 从上到下补 C7..L5
        """
        raw_aabb_entries: list[tuple[int, float, float, float, float]] = []
        raw_obb_entries: list[tuple[int, List[Point], str]] = []

        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                class_id = int(parts[0])

                if len(parts) == 5:
                    # AABB 格式: class_id cx cy w h  (旧 YOLOv5)
                    # 跳过脊柱整体外框（1=scoliosis spine, 2=normal spine）
                    if class_id in (1, 2):
                        continue
                    cx_norm = float(parts[1])
                    cy_norm = float(parts[2])
                    w_norm = float(parts[3])
                    h_norm = float(parts[4])
                    raw_aabb_entries.append((
                        class_id,
                        cx_norm * img_w, cy_norm * img_h,
                        w_norm * img_w, h_norm * img_h,
                    ))

                elif len(parts) == 9:
                    # OBB 格式: class_id x1 y1 x2 y2 x3 y3 x4 y4
                    coords = [float(p) for p in parts[1:9]]
                    # 检测是否为退化矩形（line 类型，底边=顶边）
                    is_line = self._is_degenerate_obb(coords)
                    points = [
                        Point(coords[0] * img_w, coords[1] * img_h),
                        Point(coords[2] * img_w, coords[3] * img_h),
                        Point(coords[4] * img_w, coords[5] * img_h),
                        Point(coords[6] * img_w, coords[7] * img_h),
                    ]
                    if is_line:
                        # 仅保留前两个端点，重建为 line 标注
                        raw_obb_entries.append((
                            class_id,
                            [points[0], points[1]],
                            "line",
                        ))
                    else:
                        raw_obb_entries.append((
                            class_id, points, "obb",
                        ))

                elif len(parts) == 17:
                    # Pose 格式: class_id cx cy w h x1 y1 v1 x2 y2 v2 x3 y3 v3 x4 y4 v4
                    # 字段索引：
                    #   0      1  2  3 4   5  6  7   8  9  10  11 12 13  14 15 16
                    #   class  cx cy w h   x1 y1 v1  x2 y2 v2  x3 y3 v3  x4 y4 v4
                    v3 = int(float(parts[13]))
                    v4 = int(float(parts[16]))
                    # 检测是否为 line（底部两点不可见）
                    is_line = (v3 == 0 and v4 == 0)
                    x1 = float(parts[5]) * img_w
                    y1 = float(parts[6]) * img_h
                    x2 = float(parts[8]) * img_w
                    y2 = float(parts[9]) * img_h
                    x3 = float(parts[11]) * img_w
                    y3 = float(parts[12]) * img_h
                    x4 = float(parts[14]) * img_w
                    y4 = float(parts[15]) * img_h
                    if is_line:
                        raw_obb_entries.append((
                            class_id,
                            [Point(x1, y1), Point(x2, y2)],
                            "line",
                        ))
                    else:
                        raw_obb_entries.append((
                            class_id,
                            [Point(x1, y1), Point(x2, y2), Point(x3, y3), Point(x4, y4)],
                            "obb",
                        ))

        # 合并入参：如果文件中同时有 AABB 和 OBB/pose 行，以 OBB/pose 为主
        annotations: List[OBBAnnotation] = []

        # 处理 AABB 条目
        if raw_aabb_entries:
            is_legacy = all(e[0] == 0 for e in raw_aabb_entries)
            if is_legacy:
                sorted_entries = sorted(raw_aabb_entries, key=lambda e: e[2])
                for i, (class_id, cx, cy, w, h) in enumerate(sorted_entries):
                    if i < len(VERTEBRA_CLASSES):
                        auto_class_id = i
                        auto_class_name = VERTEBRA_CLASSES[i]
                    else:
                        auto_class_id = i
                        auto_class_name = f"V{i}"
                    ann = OBBAnnotation.from_aabb(auto_class_id, auto_class_name, cx, cy, w, h)
                    annotations.append(ann)
            else:
                for class_id, cx, cy, w, h in raw_aabb_entries:
                    class_name = VERTEBRA_CLASSES.get(class_id, f"class_{class_id}")
                    ann = OBBAnnotation.from_aabb(class_id, class_name, cx, cy, w, h)
                    annotations.append(ann)

        # 处理 OBB/pose 条目：先识别是 "新 2 类导出" 还是 "旧解剖学编号"
        if raw_obb_entries:
            annotations.extend(self._resolve_obb_entries(raw_obb_entries))

        # 限制最多加载 VERTEBRA_CLASSES 数量的标注（19 个：C7~L5 + S1），
        # 超出的多余框（如旧数据中的额外标注）直接丢弃，避免画布显示灰色 V 开头框
        max_count = len(VERTEBRA_CLASSES)
        if len(annotations) > max_count:
            annotations = annotations[:max_count]

        return annotations

    def _resolve_obb_entries(
        self,
        raw_obb_entries: List[Tuple[int, List[Point], str]],
    ) -> List[OBBAnnotation]:
        """把 (export_class_id, points, shape_type) 列表转成 OBBAnnotation。

        - 新 2 类格式（class_id ⊆ {0, 1} 且含 S1=1）：按 y 排序，
          class_id=1 恢复为 S1（内部 id=18），class_id=0 从上到下顺序补 C7..L5
        - 旧解剖学格式（出现 class_id > 1）：按 VERTEBRA_CLASSES 直接映射
        """
        export_class_ids = {e[0] for e in raw_obb_entries}
        is_new_2class = (
            export_class_ids <= {EXPORT_CLASS_ID_VERTEBRA, EXPORT_CLASS_ID_S1}
            and EXPORT_CLASS_ID_S1 in export_class_ids
        )

        annotations: List[OBBAnnotation] = []

        if is_new_2class:
            def _mean_y(entry: Tuple[int, List[Point], str]) -> float:
                _, points, _ = entry
                return sum(p.y for p in points) / max(len(points), 1)

            sorted_entries = sorted(raw_obb_entries, key=_mean_y)
            next_internal_id = 0  # 从 C7 起补
            for export_cid, points, shape_type in sorted_entries:
                if export_cid == EXPORT_CLASS_ID_S1:
                    internal_id = INTERNAL_CLASS_ID_S1
                    class_name = "S1"
                else:
                    internal_id = next_internal_id
                    class_name = VERTEBRA_CLASSES.get(internal_id, f"V{internal_id}")
                    next_internal_id += 1
                    # 跳过 S1 槽位（18），避免和真正的 S1 冲突
                    if next_internal_id == INTERNAL_CLASS_ID_S1:
                        next_internal_id += 1
                annotations.append(self._build_obb_annotation(
                    internal_id, class_name, points, shape_type,
                ))
        else:
            # 旧解剖学编号：直接按 class_id 映射，越界的跳过
            for class_id, points, shape_type in raw_obb_entries:
                if class_id < 0 or class_id not in VERTEBRA_CLASSES:
                    continue
                class_name = VERTEBRA_CLASSES[class_id]
                annotations.append(self._build_obb_annotation(
                    class_id, class_name, points, shape_type,
                ))

        return annotations

    def _build_obb_annotation(
        self,
        class_id: int,
        class_name: str,
        points: List[Point],
        shape_type: str,
    ) -> OBBAnnotation:
        """根据 shape_type 构造 OBB 或 line 标注。"""
        if shape_type == "line" and len(points) == 2:
            return OBBAnnotation.from_line(
                class_id, class_name,
                points[0].x, points[0].y, points[1].x, points[1].y,
            )
        ann = OBBAnnotation(class_id, class_name, points)
        ann._update_geometry()
        return ann

    def _is_degenerate_obb(self, coords: List[float]) -> bool:
        """检测 OBB 坐标是否为退化矩形（底边与顶边重合，即 line 类型）。

        保存时 line 存为: p0, p1, p1, p0 → 检测 x1≈x4, y1≈y4, x2≈x3, y2≈y3
        """
        if len(coords) != 8:
            return False
        x1, y1, x2, y2, x3, y3, x4, y4 = coords
        eps = 1e-5
        return (
            abs(x1 - x4) < eps and abs(y1 - y4) < eps and
            abs(x2 - x3) < eps and abs(y2 - y3) < eps
        )

    def save_obb_yolov8(self, annotation: ImageAnnotation,
                        output_dir: str, overwrite: bool = False):
        """Save annotations in YOLOv8-OBB format.

        Format: class_id x1 y1 x2 y2 x3 y3 x4 y4 (normalized)

        class_id 折叠为 2 类：S1 → 1，其它椎骨 → 0
        （内部解剖学编号仅用于 UI / cache，落盘时通过 to_export_class_id 转换）

        Line 类型标注：将 2 点扩展为退化 OBB（底边与顶边重合），
        底部两个关键点在 pose 格式中以 v=0 输出，OBB 格式中仍为 4 点。
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
                if ann.shape_type == "line":
                    # Line 标注：2 点扩展为退化 OBB 4 点
                    p0, p1 = ann.points[0], ann.points[1]
                    coords = [
                        f"{p0.x / w_img:.6f}", f"{p0.y / h_img:.6f}",
                        f"{p1.x / w_img:.6f}", f"{p1.y / h_img:.6f}",
                        f"{p1.x / w_img:.6f}", f"{p1.y / h_img:.6f}",
                        f"{p0.x / w_img:.6f}", f"{p0.y / h_img:.6f}",
                    ]
                else:
                    # OBB 标注：4 角点
                    coords = []
                    for p in ann.points:
                        coords.append(f"{p.x / w_img:.6f}")
                        coords.append(f"{p.y / h_img:.6f}")

                export_cid = to_export_class_id(ann.class_id)
                line = f"{export_cid} {' '.join(coords)}\n"
                f.write(line)

        return True

    def save_obb_xywhr(self, annotation: ImageAnnotation,
                       output_dir: str, overwrite: bool = False):
        """Save annotations in YOLOv8-OBB xywhr format.

        Format: class_id cx cy w h angle (normalized, angle in radians [-pi/4, pi/4))

        class_id 折叠为 2 类：S1 → 1，其它椎骨 → 0
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

                export_cid = to_export_class_id(ann.class_id)
                line = (
                    f"{export_cid} "
                    f"{cx / w_img:.6f} {cy / h_img:.6f} "
                    f"{w / w_img:.6f} {h / h_img:.6f} "
                    f"{angle:.6f}\n"
                )
                f.write(line)

        return True

    def save_pose_yolov8(self, annotation: ImageAnnotation,
                        output_dir: str, overwrite: bool = False):
        """Save annotations in YOLOv8-pose format.

        Format per line (all normalized to [0, 1]):
            class_id  cx cy w h  x1 y1 v1  x2 y2 v2  x3 y3 v3  x4 y4 v4

        - class_id 折叠为 2 类（与 scoliosis-pose/scoliosis.yaml 对齐）：
            0 = vertebra (C7~L5 共 18 节统一为一类)
            1 = S1       (骶骨第一节，作为下端解剖锚点)
          内部解剖学编号（C7=0..S1=18）只在 UI / cache 中使用，落盘时通过
          to_export_class_id 折叠。下游算法（如 Cobb 角后处理）通过 y 排序 +
          S1 锚点自行恢复每节椎骨的解剖序。
        - bbox(cx, cy, w, h): 包围 OBB 四个角点的 AABB
        - keypoints: 椎骨矩形的 4 个角点，顺时针排列
            x1,y1 = 左上, x2,y2 = 右上, x3,y3 = 右下, x4,y4 = 左下
        - v: 可见性 (0=不可见, 1=遮挡, 2=可见)
          取自 OBBAnnotation.keypoint_visibility（对该标注 4 个点统一生效），默认 2

        Line 类型标注：输出 4 个关键点，底部 2 个为 v=0

        坐标越界处理：标注可能部分超出画面（如 C7/ S1 贴边），导出时
        会将所有 x/y 坐标 clamp 到 [0, 1]，同时 bbox 从 clamped points 重新计算，
        避免 ultralytics 训练时报"越界"。可见性保留用户原始标记不变。
        """
        def _clamp01(v: float) -> float:
            return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)

        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        img_path = Path(annotation.image_path)
        label_name = img_path.stem + ".txt"
        label_path = output / label_name

        if label_path.exists() and not overwrite:
            return False

        w_img = annotation.image_width
        h_img = annotation.image_height

        # 按内部 class_id 升序导出，保证 .txt 中按 C7→T1→...→L5→S1 的解剖序排列
        # （即使导出 class_id 被折叠为 0/1，物理顺序仍是从上到下，方便人工核对）
        sorted_annotations = sorted(annotation.annotations, key=lambda a: a.class_id)

        with open(label_path, "w") as f:
            for ann in sorted_annotations:
                export_cid = to_export_class_id(ann.class_id)
                if ann.shape_type == "line":
                    # Line 标注：2 点 → pose 格式 4 关键点，底部 2 点 v=0
                    p0, p1 = ann.points[0], ann.points[1]
                    p0_xn = _clamp01(p0.x / w_img)
                    p0_yn = _clamp01(p0.y / h_img)
                    p1_xn = _clamp01(p1.x / w_img)
                    p1_yn = _clamp01(p1.y / h_img)
                    x_min = min(p0_xn, p1_xn)
                    x_max = max(p0_xn, p1_xn)
                    y_min = min(p0_yn, p1_yn)
                    y_max = max(p0_yn, p1_yn)
                    bbox_w = max(x_max - x_min, 1.0 / w_img)
                    bbox_h = max(y_max - y_min, 1.0 / h_img)
                    bbox_cx = x_min + bbox_w / 2
                    bbox_cy = y_min + bbox_h / 2

                    v = int(ann.keypoint_visibility)
                    parts = [
                        str(export_cid),
                        f"{bbox_cx:.6f}", f"{bbox_cy:.6f}",
                        f"{bbox_w:.6f}", f"{bbox_h:.6f}",
                        # 左上 / 右上使用用户可见性
                        f"{p0_xn:.6f}", f"{p0_yn:.6f}", str(v),
                        f"{p1_xn:.6f}", f"{p1_yn:.6f}", str(v),
                        # 右下 / 左下：line 语义下底边不存在，v=0
                        f"{p1_xn:.6f}", f"{p1_yn:.6f}", "0",
                        f"{p0_xn:.6f}", f"{p0_yn:.6f}", "0",
                    ]
                    f.write(" ".join(parts) + "\n")
                else:
                    # OBB 标注：clamp 到 [0,1] 后重算 bbox
                    xs = [_clamp01(p.x / w_img) for p in ann.points]
                    ys = [_clamp01(p.y / h_img) for p in ann.points]
                    x_min, x_max = min(xs), max(xs)
                    y_min, y_max = min(ys), max(ys)

                    bbox_w = max(x_max - x_min, 1.0 / w_img)
                    bbox_h = max(y_max - y_min, 1.0 / h_img)
                    bbox_cx = x_min + bbox_w / 2
                    bbox_cy = y_min + bbox_h / 2

                    v = int(ann.keypoint_visibility)
                    parts = [
                        str(export_cid),
                        f"{bbox_cx:.6f}", f"{bbox_cy:.6f}",
                        f"{bbox_w:.6f}", f"{bbox_h:.6f}",
                    ]
                    # Keypoints：顺时针 左上、右上、右下、左下
                    for x, y in zip(xs, ys):
                        parts.append(f"{x:.6f}")
                        parts.append(f"{y:.6f}")
                        parts.append(str(v))

                    f.write(" ".join(parts) + "\n")

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
