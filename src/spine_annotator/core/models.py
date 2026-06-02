"""Core data models for spine annotation."""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# 椎骨类别定义 (class_id → class_name)
# ---------------------------------------------------------------------------
# 内部解剖学编号 (UI 与缓存使用): C7=0, T1~T12=1~12, L1~L5=13~17, S1=18
# 导出到 YOLO .txt 时会折叠为 2 类（见下方 to_export_class_id）
VERTEBRA_CLASSES: dict[int, str] = {
    0:  "C7",
    1:  "T1",  2:  "T2",  3:  "T3",  4:  "T4",  5:  "T5",
    6:  "T6",  7:  "T7",  8:  "T8",  9:  "T9",  10: "T10",
    11: "T11", 12: "T12",
    13: "L1",  14: "L2",  15: "L3",  16: "L4",  17: "L5",
    18: "S1",
}

# 内部 S1 编号（用于 export 映射 / 加载时的特殊判断）
INTERNAL_CLASS_ID_S1: int = 18

# YOLO 训练标签的 class_id（与 scoliosis-pose/scoliosis.yaml 的 names 对齐）
#   0 = vertebra  (C7 ~ L5 共 18 节统一为一类)
#   1 = S1        (骶骨第一节，作为骨盆 / 下端解剖锚点)
EXPORT_CLASS_ID_VERTEBRA: int = 0
EXPORT_CLASS_ID_S1: int = 1
EXPORT_CLASS_NAMES: dict[int, str] = {
    EXPORT_CLASS_ID_VERTEBRA: "vertebra",
    EXPORT_CLASS_ID_S1: "S1",
}


def to_export_class_id(internal_class_id: int) -> int:
    """将内部解剖学 class_id (C7=0 .. S1=18) 折叠为 YOLO 训练用 class_id。

    S1 (内部 id=18) → 1
    其它椎骨        → 0
    """
    return EXPORT_CLASS_ID_S1 if internal_class_id == INTERNAL_CLASS_ID_S1 else EXPORT_CLASS_ID_VERTEBRA


class VertebraCategory:
    """椎骨大类：颈椎(C) / 胸椎(T) / 腰椎(L) / 骶椎(S)"""
    CERVICAL  = "cervical"
    THORACIC  = "thoracic"
    LUMBAR    = "lumbar"
    SACRAL    = "sacral"

    CATEGORY_NAMES = {
        CERVICAL:  "颈椎 (C)",
        THORACIC:  "胸椎 (T)",
        LUMBAR:    "腰椎 (L)",
        SACRAL:    "骶椎 (S)",
    }


def get_vertebra_category(class_id: int) -> Optional[str]:
    """根据 class_id 返回椎骨大类 (cervical/thoracic/lumbar/sacral)，无效则 None。"""
    if class_id == 0:
        return VertebraCategory.CERVICAL
    elif 1 <= class_id <= 12:
        return VertebraCategory.THORACIC
    elif 13 <= class_id <= 17:
        return VertebraCategory.LUMBAR
    elif class_id == 18:
        return VertebraCategory.SACRAL
    return None


def get_vertebra_class_id(class_name: str) -> Optional[int]:
    """根据椎骨名称（如 'C7', 'T5', 'L3', 'S1'）返回 class_id，无效则 None。"""
    for cid, cname in VERTEBRA_CLASSES.items():
        if cname == class_name:
            return cid
    return None


@dataclass
class Point:
    """2D point in pixel coordinates."""
    x: float
    y: float

    def to_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)

    def distance_to(self, other: "Point") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)


@dataclass
class OBBAnnotation:
    """Oriented Bounding Box annotation for a single vertebra.
    
    Stores 4 corner points in clockwise order starting from top-left.
    shape_type='obb' 时为矩形（4点），shape_type='line' 时为直线（仅前2点有效）。
    """
    class_id: int
    class_name: str
    points: List[Point]  # 4 corners (obb) or 2 endpoints (line), clockwise
    center: Point = field(init=False)
    angle: float = field(init=False)  # radians, 0 = horizontal
    width: float = field(init=False)
    height: float = field(init=False)

    # Visibility (layer control: 是否在画布上显示)
    visible: bool = True

    # 关键点可见性等级（YOLOv8-pose v 字段，对该标注的 4 个角点统一生效）
    #   2 = 可见（默认）
    #   1 = 遮挡（点存在但被遮挡，坐标仍有效）
    #   0 = 不可见 / 推断（图像中肉眼无法看清，根据相邻椎骨推断）
    keypoint_visibility: int = 2

    # 标注形状类型: 'obb'=四点矩形, 'line'=两点直线
    shape_type: str = "obb"

    def __post_init__(self):
        self._update_geometry()

    def _update_geometry(self):
        """Recalculate center, angle, width, height from corner points."""
        if self.shape_type == "line":
            if len(self.points) < 2:
                return
            p0, p1 = self.points[0], self.points[1]
            self.center = Point((p0.x + p1.x) / 2, (p0.y + p1.y) / 2)
            self.width = p0.distance_to(p1)
            self.height = 0.0
            dx = p1.x - p0.x
            dy = p1.y - p0.y
            self.angle = math.atan2(dy, dx)
            return

        if len(self.points) != 4:
            return
        
        xs = [p.x for p in self.points]
        ys = [p.y for p in self.points]
        self.center = Point(sum(xs) / 4, sum(ys) / 4)
        
        # Width = distance from p0 to p1 (top edge)
        # Height = distance from p1 to p2 (right edge)
        self.width = self.points[0].distance_to(self.points[1])
        self.height = self.points[1].distance_to(self.points[2])
        
        # Angle from top edge relative to horizontal
        dx = self.points[1].x - self.points[0].x
        dy = self.points[1].y - self.points[0].y
        self.angle = math.atan2(dy, dx)

    def rotate(self, angle_delta: float):
        """Rotate the box by angle_delta (radians) around center."""
        cos_a = math.cos(angle_delta)
        sin_a = math.sin(angle_delta)
        
        new_points = []
        for p in self.points:
            dx = p.x - self.center.x
            dy = p.y - self.center.y
            nx = dx * cos_a - dy * sin_a + self.center.x
            ny = dx * sin_a + dy * cos_a + self.center.y
            new_points.append(Point(nx, ny))
        
        self.points = new_points
        self._update_geometry()

    def move_corner(self, index: int, new_pos: Point):
        """Move a specific corner/endpoint."""
        max_idx = 2 if self.shape_type == "line" else 4
        if 0 <= index < max_idx and index < len(self.points):
            self.points[index] = new_pos
            self._update_geometry()

    def move(self, dx: float, dy: float):
        """Translate the entire box."""
        for p in self.points:
            p.x += dx
            p.y += dy
        self._update_geometry()

    @classmethod
    def from_aabb(cls, class_id: int, class_name: str,
                  cx: float, cy: float, w: float, h: float) -> "OBBAnnotation":
        """Create from axis-aligned bounding box (angle=0)."""
        half_w, half_h = w / 2, h / 2
        points = [
            Point(cx - half_w, cy - half_h),  # top-left
            Point(cx + half_w, cy - half_h),  # top-right
            Point(cx + half_w, cy + half_h),  # bottom-right
            Point(cx - half_w, cy + half_h),  # bottom-left
        ]
        return cls(class_id=class_id, class_name=class_name, points=points)

    @classmethod
    def from_line(cls, class_id: int, class_name: str,
                  x1: float, y1: float, x2: float, y2: float) -> "OBBAnnotation":
        """Create a line annotation (2 endpoints)."""
        points = [Point(x1, y1), Point(x2, y2)]
        return cls(class_id=class_id, class_name=class_name,
                   points=points, shape_type="line")

    def to_xywhr(self) -> Tuple[float, float, float, float, float]:
        """Convert to (center_x, center_y, width, height, angle_radians)."""
        return (self.center.x, self.center.y, self.width, self.height, self.angle)


def auto_sort_annotations(annotations: List[OBBAnnotation]) -> List[OBBAnnotation]:
    """按标注框中心 Y 坐标（从上到下）自动排序并分配椎骨编号。

    逻辑：
    1. 将所有标注按 center.y 升序排列（脊柱从上到下）
    2. 按解剖顺序依次分配 class_id：C7(0) → T1(1) → T2(2) → ... → L5(17) → S1(18)
    3. 仅对 obb 和 line 类型标注分配编号

    Returns:
        排序后的标注列表（class_id 和 class_name 已更新）
    """
    # 按 center.y 升序排列（脊柱从上到下）
    sorted_anns = sorted(annotations, key=lambda a: a.center.y)

    # 按解剖顺序分配编号
    ordered_class_ids = list(VERTEBRA_CLASSES.keys())  # [0, 1, 2, ..., 18]

    for i, ann in enumerate(sorted_anns):
        if i < len(ordered_class_ids):
            new_class_id = ordered_class_ids[i]
            ann.class_id = new_class_id
            ann.class_name = VERTEBRA_CLASSES[new_class_id]

    return sorted_anns


@dataclass
class ImageAnnotation:
    """All annotations for a single image."""
    image_path: str
    image_width: int
    image_height: int
    annotations: List[OBBAnnotation] = field(default_factory=list)
    modified: bool = False
