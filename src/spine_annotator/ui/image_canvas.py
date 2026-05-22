"""Interactive image canvas for OBB annotation editing."""

import math
from typing import List, Optional, Tuple

from PyQt5.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt5.QtGui import (
    QBrush, QColor, QFont, QPen, QPixmap, QPolygonF, QWheelEvent,
)
from PyQt5.QtWidgets import (
    QGraphicsItem, QGraphicsPixmapItem, QGraphicsPolygonItem,
    QGraphicsRectItem, QGraphicsScene, QGraphicsSimpleTextItem,
    QGraphicsView,
)

from ..core.models import OBBAnnotation, Point

# Color palette for different classes
CLASS_COLORS = [
    QColor(0, 255, 0, 180),    # green - vertebra
    QColor(255, 0, 0, 180),    # red - scoliosis spine
    QColor(0, 100, 255, 180),  # blue - normal spine
]

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

        color = CLASS_COLORS[self.annotation.class_id % len(CLASS_COLORS)]
        pen = QPen(color, 2)
        self.setPen(pen)
        self.setBrush(QBrush(QColor(0, 0, 0, 0)))

    def set_selected(self, selected: bool):
        """Update visual state when selected."""
        self._selected = selected
        if selected:
            pen = QPen(SELECTED_COLOR, 3)
            self.setPen(pen)
            self.setZValue(10)
        else:
            color = CLASS_COLORS[self.annotation.class_id % len(CLASS_COLORS)]
            pen = QPen(color, 2)
            self.setPen(pen)
            self.setZValue(1)
        self.update()

    def paint(self, painter, option, widget=None):
        super().paint(painter, option, widget)

        if self._selected:
            self._draw_handles(painter)
            self._draw_label(painter)

    def _draw_handles(self, painter):
        """Draw corner handles and rotation handle."""
        handle_pen = QPen(SELECTED_COLOR, 2)
        handle_brush = QBrush(QColor(255, 255, 0, 120))
        painter.setPen(handle_pen)
        painter.setBrush(handle_brush)

        # Corner handles
        self._handles = []
        for p in self.annotation.points:
            center = QPointF(p.x, p.y)
            self._handles.append(center)
            painter.drawEllipse(center, HANDLE_SIZE, HANDLE_SIZE)

        # Rotation handle (above top edge midpoint)
        p0 = self.annotation.points[0]
        p1 = self.annotation.points[1]
        mid_x = (p0.x + p1.x) / 2
        mid_y = (p0.y + p1.y) / 2

        angle = self.annotation.angle
        nx = -math.sin(angle) * ROTATE_HANDLE_OFFSET
        ny = math.cos(angle) * ROTATE_HANDLE_OFFSET
        # Flip: rotation handle goes "up" (away from the bottom of the spine)
        rotate_pos = QPointF(mid_x + nx, mid_y + ny)
        self._rotate_handle = rotate_pos

        # Draw rotation handle line and circle
        painter.drawLine(QPointF(mid_x, mid_y), rotate_pos)
        painter.setBrush(QBrush(QColor(0, 200, 255, 180)))
        painter.drawEllipse(rotate_pos, HANDLE_SIZE + 1, HANDLE_SIZE + 1)

    def _draw_label(self, painter):
        """Draw class name label above the box."""
        p0 = self.annotation.points[0]
        label = f"{self.annotation.class_name} ({math.degrees(self.annotation.angle):.1f}°)"
        
        painter.setPen(QPen(SELECTED_COLOR))
        painter.setFont(QFont("Arial", 10, QFont.Bold))
        painter.drawText(QPointF(p0.x, p0.y - 8), label)

    def hit_test_handle(self, scene_pos: QPointF) -> Tuple[str, int]:
        """Test if scene position hits a handle.
        
        Returns: ('corner', index), ('rotate', -1), or ('none', -1)
        """
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

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        self._pixmap_item: Optional[QGraphicsPixmapItem] = None
        self._obb_items: List[OBBGraphicsItem] = []
        self._current_selection: int = -1

        # Drag state
        self._drag_mode: str = "none"  # 'none', 'corner', 'rotate', 'pan'
        self._drag_corner_index: int = -1
        self._drag_start_pos: Optional[QPointF] = None
        self._drag_start_angle: float = 0.0

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

    def load_image(self, image_path: str, annotations: List[OBBAnnotation]):
        """Load an image and its annotations into the canvas."""
        self._scene.clear()
        self._obb_items.clear()
        self._current_selection = -1

        # Load image
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            return

        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._scene.setSceneRect(QRectF(pixmap.rect()))

        # Add annotation items
        for i, ann in enumerate(annotations):
            item = OBBGraphicsItem(ann, i)
            self._scene.addItem(item)
            self._obb_items.append(item)

        self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)

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
        """Rotate the selected annotation by given degrees."""
        ann = self.get_selected_annotation()
        if ann is None:
            return

        ann.rotate(math.radians(angle_deg))
        self._obb_items[self._current_selection]._update_polygon()
        self.annotation_modified.emit()
        self.viewport().update()

    # --- Mouse Event Handlers ---

    def mousePressEvent(self, event):
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
        clicked_item = self._scene.itemAt(scene_pos, self.transform())
        if isinstance(clicked_item, OBBGraphicsItem):
            self.select_annotation(clicked_item.index)
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
        self._drag_mode = "none"
        self._drag_corner_index = -1
        self._drag_start_pos = None
        self.setCursor(Qt.ArrowCursor)
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QWheelEvent):
        """Zoom with mouse wheel."""
        factor = 1.15
        if event.angleDelta().y() < 0:
            factor = 1.0 / factor
        self.scale(factor, factor)
