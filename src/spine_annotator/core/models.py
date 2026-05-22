"""Core data models for spine annotation."""

import math
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np


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
    """
    class_id: int
    class_name: str
    points: List[Point]  # 4 corners, clockwise
    center: Point = field(init=False)
    angle: float = field(init=False)  # radians, 0 = horizontal
    width: float = field(init=False)
    height: float = field(init=False)
    
    # End vertebra markers for Cobb angle calculation
    is_upper_end: bool = False  # 上端椎
    is_lower_end: bool = False  # 下端椎
    
    # Visibility (layer control)
    visible: bool = True

    def __post_init__(self):
        self._update_geometry()

    def _update_geometry(self):
        """Recalculate center, angle, width, height from corner points."""
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
        """Move a specific corner point."""
        if 0 <= index < 4:
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

    def to_xywhr(self) -> Tuple[float, float, float, float, float]:
        """Convert to (center_x, center_y, width, height, angle_radians)."""
        return (self.center.x, self.center.y, self.width, self.height, self.angle)


@dataclass
class ImageAnnotation:
    """All annotations for a single image."""
    image_path: str
    image_width: int
    image_height: int
    annotations: List[OBBAnnotation] = field(default_factory=list)
    modified: bool = False
