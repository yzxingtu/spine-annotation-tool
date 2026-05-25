"""Interactive image canvas for OBB annotation editing."""

import math
from typing import Callable, List, Optional, Tuple

from PyQt5.QtCore import QEvent, QPointF, QRectF, Qt, pyqtSignal
from PyQt5.QtGui import (
    QBrush, QColor, QFont, QPen, QPixmap, QPolygonF, QWheelEvent,
)
from PyQt5.QtWidgets import (
    QGraphicsItem, QGraphicsLineItem, QGraphicsPixmapItem, QGraphicsPolygonItem,
    QGraphicsRectItem, QGraphicsScene, QGraphicsSimpleTextItem,
    QGraphicsView, QInputDialog, QMessageBox,
)

from ..core.models import (
    OBBAnnotation, Point, VertebraCategory, VERTEBRA_CLASSES,
    get_vertebra_category, get_vertebra_class_id,
)

# ---------------------------------------------------------------------------
# 椎骨大类颜色方案 (C/T/L/S 各一种颜色)
# ---------------------------------------------------------------------------
CATEGORY_COLORS = {
    VertebraCategory.CERVICAL: QColor(0, 220, 255, 200),   # 青色 - 颈椎
    VertebraCategory.THORACIC: QColor(0, 255, 0, 200),     # 绿色 - 胸椎
    VertebraCategory.LUMBAR:   QColor(255, 165, 0, 200),   # 橙色 - 腰椎
    VertebraCategory.SACRAL:   QColor(200, 0, 255, 200),   # 紫色 - 骶椎
}

# 向后兼容：旧 class_id → 颜色映射（未知类别回退色）
_DEFAULT_COLOR = QColor(180, 180, 180, 180)  # 灰色

SELECTED_COLOR = QColor(255, 255, 0, 220)  # yellow
HANDLE_SIZE = 6  # corner handle radius in pixels
ROTATE_HANDLE_OFFSET = 25  # distance of rotation handle from box


class OBBGraphicsItem(QGraphicsPolygonItem):
    """Graphics item representing an OBB annotation."""

    def __init__(self, annotation: OBBAnnotation, index: int,
                 parent: Optional[QGraphicsItem] = None):
        self.annotation = annotation
        self.index = index
        self._selected = False
        self._handles: List[QPointF] = []
        self._rotate_handle: Optional[QPointF] = None
        self._label_item: Optional[QGraphicsSimpleTextItem] = None

        super().__init__(parent)
        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self._update_polygon()

    def _update_polygon(self):
        """Update the polygon from annotation points."""
        poly = QPolygonF()
        for p in self.annotation.points:
            poly.append(QPointF(p.x, p.y))

        self.setPolygon(poly)

        # Visibility control
        self.setVisible(self.annotation.visible)
        if not self.annotation.visible:
            return

        color = self._get_color()
        pen = self._build_pen(color, width=2)
        self.setPen(pen)
        self.setBrush(QBrush(QColor(0, 0, 0, 0)))
        # Restore Z-value based on class (not selection)
        self._apply_z_value()

    def _build_pen(self, color: QColor, width: int) -> QPen:
        """根据 keypoint_visibility 决定线型：v=2 实线，v=1 短虚线，v=0 长虚线。

        边框笔使用 cosmetic 模式，线宽在屏幕像素上恒定，不随视图缩放改变。
        """
        pen = QPen(color, width)
        pen.setCosmetic(True)  # 屏幕像素级恒定线宽
        v = int(getattr(self.annotation, "keypoint_visibility", 2))
        if v == 1:
            pen.setStyle(Qt.DashLine)
        elif v == 0:
            pen.setStyle(Qt.DashDotLine)
        else:
            pen.setStyle(Qt.SolidLine)
        return pen

    def _get_color(self):
        """Get color based on vertebra category (C/T/L/S)."""
        category = get_vertebra_category(self.annotation.class_id)
        if category and category in CATEGORY_COLORS:
            return CATEGORY_COLORS[category]
        return _DEFAULT_COLOR

    def _apply_z_value(self):
        """Set Z-value: all vertebrae get high Z (on top), line annotations get slightly lower."""
        area = max(self.annotation.width * max(self.annotation.height, 1), 1)
        inv_area = 1.0 / area
        base_z = 1000 if self.annotation.shape_type == "obb" else 900
        self.setZValue(base_z + inv_area * 100)

    def set_selected(self, selected: bool):
        """Update visual state when selected."""
        self._selected = selected
        if selected:
            pen = self._build_pen(SELECTED_COLOR, width=3)
            self.setPen(pen)
            area = max(self.annotation.width * max(self.annotation.height, 1), 1)
            inv_area = 1.0 / area
            self.setZValue(1100 + inv_area * 100)
        else:
            color = self._get_color()
            pen = self._build_pen(color, width=2)
            self.setPen(pen)
            self._apply_z_value()
        self.update()

    def paint(self, painter, option, widget=None):
        super().paint(painter, option, widget)

        if self._selected:
            self._draw_handles(painter)
            self._draw_label(painter)
        else:
            # 未选中也显示椎骨编号，便于快速核对准确性
            self._draw_label(painter, brief=True)

    def _get_view_scale(self) -> float:
        """获取当前视图的缩放因子（水平方向）。

        用于绘制时反缩放手柄，使其在屏幕上保持恒定像素大小。
        """
        scene = self.scene()
        if not scene:
            return 1.0
        views = scene.views()
        if not views:
            return 1.0
        t = views[0].transform()
        # 欧几里得范数（处理缩放+旋转混合变换）
        return math.hypot(t.m11(), t.m12()) or 1.0

    def _draw_handles(self, painter):
        """Draw corner handles and rotation handle (view-scale independent).

        S1 (class_id=18) 类别锁定几何：不绘制任何手柄，仅允许整体拖动。
        """
        # S1 不允许角点/旋转手柄调整
        if self.annotation.class_id == 18:
            self._handles = []
            self._rotate_handle = None
            return

        s = self._get_view_scale()
        inv_s = 1.0 / s if s > 0 else 1.0

        handle_pen = QPen(SELECTED_COLOR, 2 * inv_s)
        handle_brush = QBrush(QColor(255, 255, 0, 120))
        painter.setPen(handle_pen)
        painter.setBrush(handle_brush)

        if self.annotation.shape_type == "line":
            # Line 标注：只绘制 2 个端点手柄
            self._handles = []
            for p in self.annotation.points[:2]:
                center = QPointF(p.x, p.y)
                self._handles.append(center)
                painter.drawEllipse(center, HANDLE_SIZE * inv_s, HANDLE_SIZE * inv_s)
            self._rotate_handle = None
            return

        # OBB 标注：4 个角点手柄 + 旋转手柄
        self._handles = []
        for p in self.annotation.points:
            center = QPointF(p.x, p.y)
            self._handles.append(center)
            painter.drawEllipse(center, HANDLE_SIZE * inv_s, HANDLE_SIZE * inv_s)

        # Rotation handle (above top edge midpoint)
        p0 = self.annotation.points[0]
        p1 = self.annotation.points[1]
        mid_x = (p0.x + p1.x) / 2
        mid_y = (p0.y + p1.y) / 2

        angle = self.annotation.angle
        nx = -math.sin(angle) * ROTATE_HANDLE_OFFSET * inv_s
        ny = math.cos(angle) * ROTATE_HANDLE_OFFSET * inv_s
        # Flip: rotation handle goes "up" (away from the bottom of the spine)
        rotate_pos = QPointF(mid_x + nx, mid_y + ny)
        self._rotate_handle = rotate_pos

        # Draw rotation handle line and circle
        painter.drawLine(QPointF(mid_x, mid_y), rotate_pos)
        painter.setBrush(QBrush(QColor(0, 200, 255, 180)))
        painter.drawEllipse(rotate_pos, (HANDLE_SIZE + 1) * inv_s, (HANDLE_SIZE + 1) * inv_s)

    def _draw_label(self, painter, brief: bool = False):
        """Draw class name label above the box (view-scale independent).

        brief=True 时仅显示椎骨编号（未选中状态），使用该椎骨分类颜色；
        brief=False 时显示完整信息（选中状态），使用 SELECTED_COLOR。
        """
        s = self._get_view_scale()
        inv_s = 1.0 / s if s > 0 else 1.0

        if brief:
            label_color = self._get_color()
            font_size = 9.0
        else:
            label_color = SELECTED_COLOR
            font_size = 10.0

        # 字体大小和偏移量都反缩放，使 worldTransform 的缩放抵消后
        # 在屏幕上保持恒定的像素大小和位置
        font = QFont("Arial")
        font.setPointSizeF(font_size * inv_s)
        font.setBold(True)
        offset = 8.0 * inv_s

        painter.save()
        painter.setPen(QPen(label_color))
        painter.setFont(font)

        if self.annotation.shape_type == "line":
            p0 = self.annotation.points[0]
            label = f"{self.annotation.class_name}"
            if not brief:
                v = int(getattr(self.annotation, "keypoint_visibility", 2))
                if v != 2:
                    v_text = {1: "遮挡", 0: "不可见"}.get(v, "")
                    label += f" [v={v} {v_text}]"
            painter.drawText(QPointF(p0.x, p0.y - offset), label)
            painter.restore()
            return

        p0 = self.annotation.points[0]
        if brief:
            label = f"{self.annotation.class_name}"
        else:
            label = f"{self.annotation.class_name} ({math.degrees(self.annotation.angle):.1f}°)"
            v = int(getattr(self.annotation, "keypoint_visibility", 2))
            if v != 2:
                v_text = {1: "遮挡", 0: "不可见"}.get(v, "")
                label += f" [v={v} {v_text}]"

        painter.drawText(QPointF(p0.x, p0.y - offset), label)
        painter.restore()

    def hit_test_handle(self, scene_pos: QPointF) -> Tuple[str, int]:
        """Test if scene position hits a handle.

        Returns: ('corner', index), ('rotate', -1), or ('none', -1)

        S1 (class_id=18) 不响应任何手柄 hit，防止拖动角点或旋转。
        """
        # S1 跳过手柄检测
        if self.annotation.class_id == 18:
            return ("none", -1)

        threshold = HANDLE_SIZE + 3

        # Check rotation handle first
        if self._rotate_handle:
            if (scene_pos - self._rotate_handle).manhattanLength() < threshold * 2:
                return ("rotate", -1)

        # Check corner handles
        for i, h in enumerate(self._handles):
            if (scene_pos - h).manhattanLength() < threshold * 2:
                return ("corner", i)

        return ("none", -1)


class AnnotationCanvas(QGraphicsView):
    """Main canvas for viewing and editing annotations."""

    selection_changed = pyqtSignal(int)  # emits annotation index, -1 if none
    annotation_modified = pyqtSignal()
    annotation_created = pyqtSignal(object)  # emits the newly created OBBAnnotation
    annotation_relabel_requested = pyqtSignal(int)  # emits scene index of double-clicked annotation

    # 绘制模式
    DRAW_NONE = "none"       # 选择/编辑模式
    DRAW_RECT = "rect"       # 绘制矩形
    DRAW_LINE = "line"       # 绘制直线

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        self._pixmap_item: Optional[QGraphicsPixmapItem] = None
        self._obb_items: List[OBBGraphicsItem] = []
        self._current_selection: int = -1

        # Drag state
        self._drag_mode: str = "none"  # 'none', 'corner', 'rotate', 'pan', 'draw_rect', 'draw_line'
        self._drag_corner_index: int = -1
        self._drag_start_pos: Optional[QPointF] = None
        self._drag_start_angle: float = 0.0

        # Drawing state
        self._draw_mode: str = self.DRAW_NONE
        self._draw_start: Optional[QPointF] = None
        self._draw_preview_item = None  # 预览图形

        # 当前绘制预设的椎骨 class_id (由 main_window 设置)
        self._pending_class_id: Optional[int] = None

        # Setup view
        from PyQt5.QtGui import QPainter
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing
            | QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setDragMode(QGraphicsView.NoDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

    def set_draw_mode(self, mode: str, class_id: Optional[int] = None):
        """设置绘制模式 ('none', 'rect', 'line') 及预设椎骨类别。"""
        self._draw_mode = mode
        self._pending_class_id = class_id
        # 切换到绘制模式时取消选中
        if mode != self.DRAW_NONE:
            self.select_annotation(-1)
            if mode == self.DRAW_RECT:
                self.setCursor(Qt.CrossCursor)
            elif mode == self.DRAW_LINE:
                self.setCursor(Qt.CrossCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

    def load_image(self, image_path: str, annotations: List[OBBAnnotation]):
        """Load an image and its annotations into the canvas."""
        self._scene.clear()
        self._obb_items.clear()
        self._index_map = []
        self._current_selection = -1

        # Load image
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            return

        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._scene.setSceneRect(QRectF(pixmap.rect()))

        # Sort annotations by area (large first) for consistent ordering
        sorted_anns = sorted(enumerate(annotations), key=lambda x: x[1].width * x[1].height, reverse=True)
        
        self._index_map = []  # _index_map[scene_idx] = original_idx
        for scene_idx, (old_idx, ann) in enumerate(sorted_anns):
            item = OBBGraphicsItem(ann, old_idx)
            self._scene.addItem(item)
            self._obb_items.append(item)
            self._index_map.append(old_idx)

        self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)

    def reload_annotations(self, annotations: List[OBBAnnotation]):
        """Refresh annotation items on canvas without resetting zoom/pan.

        Use this when annotations are modified (delete, relabel) but the
        image and viewport transform should stay the same.
        """
        # Remove old OBB items from scene
        for item in self._obb_items:
            self._scene.removeItem(item)
        self._obb_items.clear()
        self._index_map = []
        self._current_selection = -1

        # Re-add annotations (keep existing pixmap)
        sorted_anns = sorted(enumerate(annotations), key=lambda x: x[1].width * max(x[1].height, 1), reverse=True)

        for scene_idx, (old_idx, ann) in enumerate(sorted_anns):
            item = OBBGraphicsItem(ann, old_idx)
            self._scene.addItem(item)
            self._obb_items.append(item)
            self._index_map.append(old_idx)

        self.viewport().update()

    def select_annotation(self, index: int):
        """Select annotation by index."""
        # Deselect previous
        if 0 <= self._current_selection < len(self._obb_items):
            self._obb_items[self._current_selection].set_selected(False)

        self._current_selection = index

        if 0 <= index < len(self._obb_items):
            self._obb_items[index].set_selected(True)

        self.selection_changed.emit(index)
        self.viewport().update()

    def get_selected_annotation(self) -> Optional[OBBAnnotation]:
        """Get the currently selected annotation."""
        if 0 <= self._current_selection < len(self._obb_items):
            return self._obb_items[self._current_selection].annotation
        return None

    def rotate_selected(self, angle_deg: float):
        """Rotate the selected annotation by given degrees.

        S1 (骥椎, class_id=18) 为 line 类型，角度永远锁定为 0，不允许旋转。
        """
        ann = self.get_selected_annotation()
        if ann is None:
            return

        # S1 角度锁定
        if ann.class_id == 18:
            return

        ann.rotate(math.radians(angle_deg))
        self._obb_items[self._current_selection]._update_polygon()
        self.annotation_modified.emit()
        self.viewport().update()

    def move_selected(self, dx: float, dy: float):
        """Move the selected annotation by dx, dy pixels."""
        ann = self.get_selected_annotation()
        if ann is None:
            return

        ann.move(dx, dy)
        self._obb_items[self._current_selection]._update_polygon()
        self.annotation_modified.emit()
        self.viewport().update()

    def get_original_index(self) -> int:
        """Get the original annotation index for the currently selected item."""
        if 0 <= self._current_selection < len(self._index_map):
            return self._index_map[self._current_selection]
        return -1

    # --- Mouse Event Handlers ---

    def mousePressEvent(self, event):
        # --- 绘制模式处理 ---
        if self._draw_mode != self.DRAW_NONE and event.button() == Qt.LeftButton:
            scene_pos = self.mapToScene(event.pos())
            self._draw_start = scene_pos
            self._drag_mode = f"draw_{self._draw_mode}"
            # 创建预览图形
            self._remove_draw_preview()
            if self._draw_mode == self.DRAW_RECT:
                self._draw_preview_item = QGraphicsRectItem(
                    QRectF(scene_pos, scene_pos)
                )
                pen = QPen(SELECTED_COLOR, 2, Qt.DashLine)
                self._draw_preview_item.setPen(pen)
                self._scene.addItem(self._draw_preview_item)
            elif self._draw_mode == self.DRAW_LINE:
                self._draw_preview_item = QGraphicsLineItem(
                    scene_pos.x(), scene_pos.y(), scene_pos.x(), scene_pos.y()
                )
                pen = QPen(SELECTED_COLOR, 2, Qt.DashLine)
                self._draw_preview_item.setPen(pen)
                self._scene.addItem(self._draw_preview_item)
            return

        if event.button() == Qt.MiddleButton:
            # Pan
            self._drag_mode = "pan"
            self._drag_start_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            return

        if event.button() != Qt.LeftButton:
            super().mousePressEvent(event)
            return

        scene_pos = self.mapToScene(event.pos())

        # Check if clicking on selected item's handles
        if 0 <= self._current_selection < len(self._obb_items):
            item = self._obb_items[self._current_selection]
            mode, idx = item.hit_test_handle(scene_pos)

            if mode == "corner":
                self._drag_mode = "corner"
                self._drag_corner_index = idx
                return
            elif mode == "rotate":
                self._drag_mode = "rotate"
                ann = item.annotation
                self._drag_start_angle = math.atan2(
                    scene_pos.y() - ann.center.y,
                    scene_pos.x() - ann.center.x,
                )
                return

        # Check if clicking on any annotation
        # Iterate from top (highest Z) to bottom to find the topmost visible item
        clicked_item = None
        for item in reversed(sorted(self._obb_items, key=lambda it: it.zValue())):
            if not item.isVisible():
                continue
            # Check if scene_pos is inside the polygon shape
            if item.shape().contains(item.mapFromScene(scene_pos)):
                clicked_item = item
                break
        
        if clicked_item is not None:
            # Find the scene item's index in _obb_items
            scene_idx = -1
            for i, item in enumerate(self._obb_items):
                if item is clicked_item:
                    scene_idx = i
                    break
            if scene_idx >= 0:
                self.select_annotation(scene_idx)
                self._drag_mode = "move"
                self._drag_start_pos = scene_pos
            return

        # Clicked on empty space - deselect
        self.select_annotation(-1)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_mode == "pan":
            delta = event.pos() - self._drag_start_pos
            self._drag_start_pos = event.pos()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x()
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y()
            )
            return

        # --- 绘制模式拖拽更新预览 ---
        if self._drag_mode in ("draw_rect", "draw_line") and self._draw_start:
            scene_pos = self.mapToScene(event.pos())
            if self._draw_preview_item:
                if self._drag_mode == "draw_rect":
                    rect = QRectF(self._draw_start, scene_pos).normalized()
                    self._draw_preview_item.setRect(rect)
                elif self._drag_mode == "draw_line":
                    self._draw_preview_item.setLine(
                        self._draw_start.x(), self._draw_start.y(),
                        scene_pos.x(), scene_pos.y()
                    )
            self.viewport().update()
            return

        if self._drag_mode == "none":
            super().mouseMoveEvent(event)
            return

        scene_pos = self.mapToScene(event.pos())

        if self._drag_mode == "corner" and 0 <= self._current_selection < len(self._obb_items):
            item = self._obb_items[self._current_selection]
            new_point = Point(scene_pos.x(), scene_pos.y())
            item.annotation.move_corner(self._drag_corner_index, new_point)
            item._update_polygon()
            self.annotation_modified.emit()

        elif self._drag_mode == "rotate" and 0 <= self._current_selection < len(self._obb_items):
            item = self._obb_items[self._current_selection]
            ann = item.annotation
            current_angle = math.atan2(
                scene_pos.y() - ann.center.y,
                scene_pos.x() - ann.center.x,
            )
            delta = current_angle - self._drag_start_angle
            ann.rotate(delta)
            self._drag_start_angle = current_angle
            item._update_polygon()
            self.annotation_modified.emit()

        elif self._drag_mode == "move" and 0 <= self._current_selection < len(self._obb_items):
            item = self._obb_items[self._current_selection]
            if self._drag_start_pos:
                dx = scene_pos.x() - self._drag_start_pos.x()
                dy = scene_pos.y() - self._drag_start_pos.y()
                item.annotation.move(dx, dy)
                item._update_polygon()
                self._drag_start_pos = scene_pos
                self.annotation_modified.emit()

        self.viewport().update()

    def mouseReleaseEvent(self, event):
        # --- 绘制完成 ---
        if self._drag_mode in ("draw_rect", "draw_line") and self._draw_start:
            scene_pos = self.mapToScene(event.pos())
            self._remove_draw_preview()
            self._finalize_draw(scene_pos)
            self._drag_mode = "none"
            self._draw_start = None
            return

        self._drag_mode = "none"
        self._drag_corner_index = -1
        self._drag_start_pos = None
        self.setCursor(Qt.ArrowCursor)
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QWheelEvent):
        """鼠标滚轮缩放 / 触控板双指滑动平移."""
        pixel = event.pixelDelta()
        angle = event.angleDelta()

        # 触控板双指滑动：pixelDelta 非零且 angleDelta 较小（通常 |y| < 15）
        # 鼠标滚轮：angleDelta 较大（通常 |y| ≥ 120），pixelDelta 可能也有值
        is_trackpad_scroll = (
            (pixel.x() != 0 or pixel.y() != 0) and
            abs(angle.y()) < 15 and abs(angle.x()) < 15
        )
        if is_trackpad_scroll:
            self.translate(-pixel.x(), -pixel.y())
            return

        # 鼠标滚轮缩放（用 angleDelta）
        delta = angle.y() or angle.x()
        if delta == 0:
            return
        factor = 1.15 if delta > 0 else 1.0 / 1.15
        self.scale(factor, factor)

    def event(self, event):
        """处理 macOS 触控板原生手势（捏合缩放、智能缩放）。"""
        if event.type() == QEvent.NativeGesture:
            gesture_type = event.gestureType()
            if gesture_type == Qt.ZoomNativeGesture:
                # macOS 触控板捏合：value 为增量缩放值
                value = event.value()
                if value != 0.0:
                    # 限制单次缩放幅度，防止跳跃
                    delta = max(-0.3, min(0.3, value))
                    factor = 1.0 + delta
                    self.scale(factor, factor)
                return True
            elif gesture_type == Qt.SmartZoomNativeGesture:
                # macOS 触控板双指轻点两下：智能缩放 / 适配
                if event.value() == 1.0:
                    self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)
                return True
        return super().event(event)

    def mouseDoubleClickEvent(self, event):
        """双击标注时弹出椎骨编号修改请求。"""
        if event.button() != Qt.LeftButton:
            super().mouseDoubleClickEvent(event)
            return

        scene_pos = self.mapToScene(event.pos())

        # 查找被双击的标注
        clicked_item = None
        for item in reversed(sorted(self._obb_items, key=lambda it: it.zValue())):
            if not item.isVisible():
                continue
            if item.shape().contains(item.mapFromScene(scene_pos)):
                clicked_item = item
                break

        if clicked_item is not None:
            scene_idx = -1
            for i, item in enumerate(self._obb_items):
                if item is clicked_item:
                    scene_idx = i
                    break
            if scene_idx >= 0:
                self.select_annotation(scene_idx)
                self.annotation_relabel_requested.emit(scene_idx)
                return

        super().mouseDoubleClickEvent(event)

    # --- 绘制辅助方法 ---

    def _remove_draw_preview(self):
        """移除绘制预览图形。"""
        if self._draw_preview_item is not None:
            self._scene.removeItem(self._draw_preview_item)
            self._draw_preview_item = None

    def _finalize_draw(self, end_pos: QPointF):
        """绘制完成后创建标注。"""
        if self._draw_start is None:
            return

        start = self._draw_start

        # 检查最小尺寸（避免误点击产生极小标注）
        dist = math.hypot(end_pos.x() - start.x(), end_pos.y() - start.y())
        if dist < 5:
            return

        class_id = self._pending_class_id
        class_name = VERTEBRA_CLASSES.get(class_id, f"class_{class_id}") if class_id is not None else None

        if self._draw_mode == self.DRAW_RECT:
            # 矩形：从拖拽区域创建 OBB
            rect = QRectF(start, end_pos).normalized()
            cx, cy = rect.center().x(), rect.center().y()
            w, h = rect.width(), rect.height()
            if class_id is not None:
                ann = OBBAnnotation.from_aabb(class_id, class_name, cx, cy, w, h)
                self.annotation_created.emit(ann)
            else:
                # 未预设类别，通知 main_window 弹出选择器
                ann = OBBAnnotation.from_aabb(-1, "", cx, cy, w, h)
                self.annotation_created.emit(ann)

        elif self._draw_mode == self.DRAW_LINE:
            # 直线：从拖拽创建 line 标注
            if class_id is not None:
                ann = OBBAnnotation.from_line(
                    class_id, class_name,
                    start.x(), start.y(), end_pos.x(), end_pos.y(),
                )
                self.annotation_created.emit(ann)
            else:
                ann = OBBAnnotation.from_line(
                    -1, "", start.x(), start.y(), end_pos.x(), end_pos.y(),
                )
                self.annotation_created.emit(ann)

    def add_annotation(self, annotation: OBBAnnotation):
        """将新标注添加到画布并刷新显示。"""
        if not annotation:
            return
        idx = len(self._obb_items)
        item = OBBGraphicsItem(annotation, idx)
        self._scene.addItem(item)
        self._obb_items.append(item)
        self._index_map.append(idx)
        self.viewport().update()
