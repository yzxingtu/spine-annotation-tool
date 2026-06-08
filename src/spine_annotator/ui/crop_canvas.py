"""Interactive canvas for single-vertebra crop pedicle point annotation."""

from typing import Optional

from PyQt5.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QFont, QImage, QPen, QPixmap
from PyQt5.QtWidgets import (
    QGraphicsEllipseItem, QGraphicsPixmapItem, QGraphicsRectItem,
    QGraphicsScene, QGraphicsSimpleTextItem, QGraphicsView,
)

from ..core.models import CropPedicleAnnotation, PediclePoint, Point

# Pedicle point visual style
LEFT_COLOR = QColor(0, 150, 255, 220)      # blue
RIGHT_COLOR = QColor(255, 60, 60, 220)     # red
SELECTED_RING = QColor(255, 255, 0, 200)   # yellow ring for active side
DEFAULT_POINT_RADIUS = 0.5   # base radius in scene coords (smaller default)
LABEL_OFFSET = 10          # label offset from point center


class CropCanvas(QGraphicsView):
    """Canvas for annotating left/right pedicle center points on crop images."""

    point_changed = pyqtSignal()      # emitted when any pedicle point is modified
    right_clicked = pyqtSignal()      # emitted on right-click (for flag toggle)
    side_deselected = pyqtSignal()    # emitted when clicking empty area (hide ring)
    side_selected = pyqtSignal()      # emitted when clicking on a point (select side)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        self.setRenderHints(self.renderHints())
        self.setDragMode(QGraphicsView.NoDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # Configurable point size
        self._point_radius: int = DEFAULT_POINT_RADIUS

        # State
        self._annotation: Optional[CropPedicleAnnotation] = None
        self._active_side: str = "left"  # always "left" or "right", never None
        self._show_ring: bool = True     # whether to show the yellow selection ring
        self._bg_item: Optional[QGraphicsPixmapItem] = None
        self._left_point_item: Optional[QGraphicsEllipseItem] = None
        self._right_point_item: Optional[QGraphicsEllipseItem] = None
        self._left_label: Optional[QGraphicsSimpleTextItem] = None
        self._right_label: Optional[QGraphicsSimpleTextItem] = None
        self._active_ring: Optional[QGraphicsEllipseItem] = None
        self._dragging: bool = False
        self._img_w: int = 0
        self._img_h: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_image(self, image_path: str, annotation: CropPedicleAnnotation):
        """Load a crop image and its pedicle annotation."""
        self._annotation = annotation
        self._img_w = annotation.image_width
        self._img_h = annotation.image_height
        self._scene.clear()
        # Reset ALL scene-item references to avoid dangling pointers
        self._bg_item = None
        self._left_point_item = None
        self._right_point_item = None
        self._left_label = None
        self._right_label = None
        self._active_ring = None

        # Background image
        pixmap = QPixmap(image_path)
        if not pixmap.isNull():
            self._bg_item = self._scene.addPixmap(pixmap)
            self._scene.setSceneRect(QRectF(pixmap.rect()))
            self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)

        # Render pedicle points
        self._render_points()
        self._render_active_ring()

    def set_active_side(self, side: str):
        """Set which side (left/right) is active for editing."""
        self._active_side = side
        self._show_ring = True
        self._render_active_ring()

    def set_point_radius(self, radius: float):
        """Set point display radius and re-render."""
        self._point_radius = max(0.5, min(radius, 20))
        if self._annotation:
            self._render_points()
            self._render_active_ring()

    def get_annotation(self) -> Optional[CropPedicleAnnotation]:
        """Return current annotation."""
        return self._annotation

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_points(self):
        """Render left and right pedicle points based on current annotation."""
        # Remove old items (guard against dangling references after scene.clear())
        for attr in ('_left_point_item', '_right_point_item',
                     '_left_label', '_right_label'):
            item = getattr(self, attr, None)
            if item is not None:
                try:
                    self._scene.removeItem(item)
                except RuntimeError:
                    pass  # C++ object already deleted
            setattr(self, attr, None)

        if not self._annotation:
            return

        ann = self._annotation

        # Left pedicle point
        if ann.image_left.center and ann.image_left.visibility > 0:
            self._left_point_item = self._add_point_item(
                ann.image_left.center.x, ann.image_left.center.y,
                ann.image_left.visibility, LEFT_COLOR,
            )
            self._left_label = self._add_label(
                ann.image_left.center.x, ann.image_left.center.y, "L", LEFT_COLOR,
            )

        # Right pedicle point
        if ann.image_right.center and ann.image_right.visibility > 0:
            self._right_point_item = self._add_point_item(
                ann.image_right.center.x, ann.image_right.center.y,
                ann.image_right.visibility, RIGHT_COLOR,
            )
            self._right_label = self._add_label(
                ann.image_right.center.x, ann.image_right.center.y, "R", RIGHT_COLOR,
            )

    def _add_point_item(self, x: float, y: float, visibility: int, color: QColor):
        """Add a circle item for a pedicle point."""
        r = self._point_radius
        rect = QRectF(x - r, y - r, r * 2, r * 2)

        if visibility == 3:
            # Solid circle
            item = self._scene.addEllipse(rect, QPen(color, 2), QBrush(color))
        elif visibility == 2:
            # Semi-transparent fill
            semi = QColor(color.red(), color.green(), color.blue(), 100)
            item = self._scene.addEllipse(rect, QPen(color, 2), QBrush(semi))
        elif visibility == 1:
            # Hollow with dashed border
            pen = QPen(color, 2, Qt.DashLine)
            item = self._scene.addEllipse(rect, pen, QBrush(Qt.NoBrush))
        else:
            # v=0: small gray cross
            gray = QColor(180, 180, 180, 150)
            item = self._scene.addEllipse(rect, QPen(gray, 1), QBrush(Qt.NoBrush))

        item.setZValue(100)
        return item

    def _add_label(self, x: float, y: float, text: str, color: QColor):
        """Add a text label near a point."""
        label = self._scene.addSimpleText(text)
        offset = self._point_radius + 1
        label.setPos(x + offset, y - offset)
        label.setBrush(QBrush(color))
        label.setZValue(101)
        # Very small text — always smaller than the circle
        font = QFont("Arial", 1)
        font.setBold(True)
        label.setFont(font)
        return label

    def _render_active_ring(self):
        """Show a yellow ring around the active side's point position."""
        if self._active_ring is not None:
            try:
                self._scene.removeItem(self._active_ring)
            except RuntimeError:
                pass  # C++ object already deleted
            self._active_ring = None

        if not self._annotation or not self._show_ring:
            return

        pt = (self._annotation.image_left if self._active_side == "left"
              else self._annotation.image_right)
        if not pt.center:
            return

        r = self._point_radius + 3
        rect = QRectF(pt.center.x - r, pt.center.y - r, r * 2, r * 2)
        pen = QPen(SELECTED_RING, 2, Qt.DashLine)
        self._active_ring = self._scene.addEllipse(rect, pen, QBrush(Qt.NoBrush))
        self._active_ring.setZValue(102)

    # ------------------------------------------------------------------
    # Mouse interaction
    # ------------------------------------------------------------------

    def mousePressEvent(self, event):
        """Handle mouse press: drag existing point or deselect."""
        if event.button() == Qt.LeftButton and self._annotation:
            scene_pos = self.mapToScene(event.pos())
            px, py = scene_pos.x(), scene_pos.y()

            # Check if clicking on existing point to start drag
            hit_side = self._hit_test_point(px, py)
            if hit_side is not None:
                # Clicked on a point → switch to that side and start drag
                self._active_side = hit_side
                self._show_ring = True
                self._dragging = True
                self._render_active_ring()
                self.side_selected.emit()
                return

            # Clicked on empty area → hide yellow ring (but keep side for double-click)
            self._show_ring = False
            self._render_active_ring()
            self.side_deselected.emit()

        elif event.button() == Qt.RightButton:
            # Right-click → emit signal for flag toggle (handled by window)
            self.right_clicked.emit()
            return

        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        """Double-click on empty area → place new point for active side.
        If active side already has a point, auto-switch to the other side."""
        if event.button() == Qt.LeftButton and self._annotation:
            scene_pos = self.mapToScene(event.pos())
            px, py = scene_pos.x(), scene_pos.y()

            # Only place if not clicking on an existing point
            if self._hit_test_point(px, py) is None:
                # If active side already has a point, switch to other side
                active_pt = (self._annotation.image_left if self._active_side == "left"
                             else self._annotation.image_right)
                if active_pt.center:
                    # Active side already has a point → switch to other side
                    self._active_side = "right" if self._active_side == "left" else "left"
                    self._show_ring = True
                    self.side_selected.emit()

                self._set_point(self._active_side, px, py)

    def mouseMoveEvent(self, event):
        """Handle mouse move: drag pedicle point."""
        if self._dragging and self._annotation:
            scene_pos = self.mapToScene(event.pos())
            px = max(0, min(scene_pos.x(), self._img_w))
            py = max(0, min(scene_pos.y(), self._img_h))
            self._set_point(self._active_side, px, py)

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """Handle mouse release: end drag."""
        if event.button() == Qt.LeftButton:
            self._dragging = False
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        """Zoom with mouse wheel."""
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _hit_test_point(self, x: float, y: float) -> Optional[str]:
        """Check if (x, y) is near left or right point. Returns side or None."""
        if not self._annotation:
            return None
        hit_r = self._point_radius + 6  # slightly larger than visual radius
        for side in ("left", "right"):
            pt = (self._annotation.image_left if side == "left"
                  else self._annotation.image_right)
            if pt.center and pt.visibility > 0:
                dx = pt.center.x - x
                dy = pt.center.y - y
                if dx * dx + dy * dy <= hit_r * hit_r:
                    return side
        return None

    def _set_point(self, side: str, x: float, y: float):
        """Set or update a pedicle point on the given side."""
        if not self._annotation:
            return
        px = max(0, min(x, self._img_w))
        py = max(0, min(y, self._img_h))

        if side == "left":
            self._annotation.image_left = PediclePoint(
                center=Point(px, py),
                visibility=max(self._annotation.image_left.visibility, 3),
            )
        else:
            self._annotation.image_right = PediclePoint(
                center=Point(px, py),
                visibility=max(self._annotation.image_right.visibility, 3),
            )
        self._annotation.modified = True
        self._render_points()
        self._render_active_ring()
        # Force viewport update to clear any rendering artifacts (ghost trails)
        self.viewport().update()
        self.point_changed.emit()

    def _clear_active_point(self):
        """Clear the active side's pedicle point (only when side is selected)."""
        if not self._annotation or not self._show_ring:
            return
        pt = (self._annotation.image_left if self._active_side == "left"
              else self._annotation.image_right)
        if not pt.center:
            return  # no point to clear
        if self._active_side == "left":
            self._annotation.image_left = PediclePoint()
        else:
            self._annotation.image_right = PediclePoint()
        self._annotation.modified = True
        self._render_points()
        self._render_active_ring()
        # Force viewport update to clear any rendering artifacts (ghost trails)
        self.viewport().update()
        self.point_changed.emit()

    def set_visibility(self, visibility: int):
        """Set visibility for the active side."""
        if not self._annotation:
            return
        if self._active_side == "left":
            self._annotation.image_left.visibility = visibility
        else:
            self._annotation.image_right.visibility = visibility
        self._annotation.modified = True
        self._render_points()
        self._render_active_ring()
        # Force viewport update to clear any rendering artifacts (ghost trails)
        self.viewport().update()
        self.point_changed.emit()

    def clear_active_point(self):
        """Public method to clear active side point (for keyboard shortcut)."""
        self._clear_active_point()
