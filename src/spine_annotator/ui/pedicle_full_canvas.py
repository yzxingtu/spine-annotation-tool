"""Canvas for pedicle annotation on full X-ray images with AABB boxes."""

from typing import Dict, List, Optional, Tuple

from PyQt5.QtCore import QRectF, Qt, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QFont, QPen, QPixmap
from PyQt5.QtWidgets import (
    QGraphicsEllipseItem, QGraphicsPixmapItem, QGraphicsRectItem,
    QGraphicsScene, QGraphicsSimpleTextItem, QGraphicsView,
)

from ..core.models import OBBAnnotation, VertebraCategory, get_vertebra_category

# Pedicle point visual style
LEFT_COLOR = QColor(0, 150, 255, 220)
RIGHT_COLOR = QColor(255, 60, 60, 220)
SELECTED_RING = QColor(255, 255, 0, 200)

# AABB box style
AABB_COLOR = QColor(0, 200, 0, 180)
AABB_SELECTED_COLOR = QColor(255, 255, 0, 220)
DEFAULT_POINT_RADIUS = 2.0


class PedicleFullCanvas(QGraphicsView):
    """Canvas showing full X-ray with AABB boxes and pedicle points."""

    vertebra_clicked = pyqtSignal(str)
    point_changed = pyqtSignal()
    right_clicked = pyqtSignal()
    side_deselected = pyqtSignal()
    side_selected = pyqtSignal()

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
        self._point_radius: float = DEFAULT_POINT_RADIUS

        # State
        self._image_path: Optional[str] = None
        self._img_w: int = 0
        self._img_h: int = 0
        self._obb_annotations: List[OBBAnnotation] = []
        self._pedicle_data: Dict[str, dict] = {}  # vert_name -> pedicle dict
        self._active_vertebra: Optional[str] = None
        self._active_side: str = "left"
        self._show_ring: bool = True

        # Scene items
        self._bg_item: Optional[QGraphicsPixmapItem] = None
        self._aabb_items: Dict[str, QGraphicsRectItem] = {}
        self._label_items: Dict[str, QGraphicsSimpleTextItem] = {}
        self._point_items: list = []  # all point ellipses (for cleanup)
        self._point_labels: list = []  # all point text labels (for cleanup)
        self._active_ring: Optional[QGraphicsEllipseItem] = None
        self._dragging: bool = False

        # Pan (middle mouse button drag)
        self._panning: bool = False
        self._pan_start_pos = None

        # Layer visibility (category -> bool)
        self._layer_visibility: Dict[str, bool] = {
            VertebraCategory.CERVICAL: True,
            VertebraCategory.THORACIC: True,
            VertebraCategory.LUMBAR: True,
            VertebraCategory.SACRAL: True,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_image(self, image_path: str, obb_annotations: List[OBBAnnotation],
                   img_w: int, img_h: int):
        """Load full X-ray image and vertebra OBB annotations."""
        self._image_path = image_path
        self._obb_annotations = obb_annotations
        self._img_w = img_w
        self._img_h = img_h

        self._scene.clear()
        # Reset all scene-item references
        self._bg_item = None
        self._aabb_items = {}
        self._label_items = {}
        self._point_items = []
        self._point_labels = []
        self._active_ring = None

        # Background image
        pixmap = QPixmap(image_path)
        if not pixmap.isNull():
            self._bg_item = self._scene.addPixmap(pixmap)
            self._scene.setSceneRect(QRectF(pixmap.rect()))
            self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)

        # Render AABB boxes
        self._render_aabb_boxes()
        # Render pedicle points for active vertebra
        self._render_points()
        self._render_active_ring()

    def set_active_vertebra(self, vert_name: str):
        """Set the currently active vertebra for pedicle annotation."""
        self._active_vertebra = vert_name
        self._show_ring = True
        self._update_aabb_highlight()
        self._render_points()
        self._render_active_ring()

    def set_pedicle_data(self, data: Dict[str, dict]):
        """Set the pedicle annotation data for all vertebrae."""
        self._pedicle_data = data
        self._render_points()
        self._render_active_ring()

    def get_pedicle_data(self) -> Dict[str, dict]:
        """Return current pedicle data."""
        return self._pedicle_data

    def set_active_side(self, side: str):
        """Set which side (left/right) is active."""
        self._active_side = side
        self._show_ring = True
        self._render_active_ring()

    def set_point_radius(self, radius: float):
        """Set point display radius and re-render."""
        self._point_radius = max(0.5, min(radius, 50))
        self._render_points()
        self._render_active_ring()

    def set_layer_visibility(self, visibility: Dict[str, bool]):
        """Set visibility for vertebra categories (cervical/thoracic/lumbar/sacral)."""
        self._layer_visibility.update(visibility)
        self._apply_layer_visibility()

    def _apply_layer_visibility(self):
        """Apply layer visibility to AABB boxes based on vertebra category."""
        for ann in self._obb_annotations:
            if ann.shape_type != "obb" or not ann.class_name:
                continue
            name = ann.class_name
            category = get_vertebra_category(ann.class_id)
            visible = True
            if category:
                visible = self._layer_visibility.get(category, True)

            # Set visibility on AABB box
            if name in self._aabb_items:
                self._aabb_items[name].setVisible(visible)
            if name in self._label_items:
                self._label_items[name].setVisible(visible)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_aabb_boxes(self):
        """Draw AABB rectangles for all vertebrae."""
        for ann in self._obb_annotations:
            if ann.shape_type != "obb" or not ann.class_name:
                continue

            xs = [p.x for p in ann.points[:4]]
            ys = [p.y for p in ann.points[:4]]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)

            rect = QRectF(x1, y1, x2 - x1, y2 - y1)
            pen = QPen(AABB_COLOR, 2)
            item = self._scene.addRect(rect, pen, QBrush(Qt.NoBrush))
            item.setZValue(50)
            self._aabb_items[ann.class_name] = item

            # Add label
            label = self._scene.addSimpleText(ann.class_name)
            label.setPos(x1, y1 - 14)
            label.setBrush(QBrush(AABB_COLOR))
            label.setZValue(51)
            font = QFont("Arial", 8)
            label.setFont(font)
            self._label_items[ann.class_name] = label

        self._update_aabb_highlight()

    def _update_aabb_highlight(self):
        """Update AABB box highlighting for active vertebra."""
        for name, item in self._aabb_items.items():
            if name == self._active_vertebra:
                pen = QPen(AABB_SELECTED_COLOR, 3)
                item.setPen(pen)
                item.setZValue(55)
                if name in self._label_items:
                    self._label_items[name].setBrush(QBrush(AABB_SELECTED_COLOR))
                    self._label_items[name].setZValue(56)
            else:
                pen = QPen(AABB_COLOR, 2)
                item.setPen(pen)
                item.setZValue(50)
                if name in self._label_items:
                    self._label_items[name].setBrush(QBrush(AABB_COLOR))
                    self._label_items[name].setZValue(51)

    def _render_points(self):
        """Render L/R pedicle points for ALL annotated vertebrae."""
        # Remove old point items
        for item in self._point_items + self._point_labels:
            try:
                self._scene.removeItem(item)
            except RuntimeError:
                pass
        self._point_items = []
        self._point_labels = []

        if not self._pedicle_data:
            return

        # Render points for all vertebrae that have annotations
        for vert_name, pdata in self._pedicle_data.items():
            # Check layer visibility for this vertebra
            # Find the vertebra's class_id to check category
            vert_visible = True
            for ann in self._obb_annotations:
                if ann.class_name == vert_name:
                    category = get_vertebra_category(ann.class_id)
                    if category and not self._layer_visibility.get(category, True):
                        vert_visible = False
                    break

            if not vert_visible:
                continue

            # Left point
            left = pdata.get("left", {})
            if left and left.get("center") and left.get("visibility", 0) > 0:
                cx, cy = left["center"]["x"], left["center"]["y"]
                item = self._add_point_item(cx, cy, left["visibility"], LEFT_COLOR)
                self._point_items.append(item)
                lbl = self._add_label(cx, cy, f"L{vert_name}", LEFT_COLOR, side="left")
                self._point_labels.append(lbl)

            # Right point
            right = pdata.get("right", {})
            if right and right.get("center") and right.get("visibility", 0) > 0:
                cx, cy = right["center"]["x"], right["center"]["y"]
                item = self._add_point_item(cx, cy, right["visibility"], RIGHT_COLOR)
                self._point_items.append(item)
                lbl = self._add_label(cx, cy, f"R{vert_name}", RIGHT_COLOR, side="right")
                self._point_labels.append(lbl)

    def _add_point_item(self, x: float, y: float, visibility: int, color: QColor):
        """Add a circle item for a pedicle point."""
        r = self._point_radius
        rect = QRectF(x - r, y - r, r * 2, r * 2)

        if visibility == 3:
            item = self._scene.addEllipse(rect, QPen(color, 2), QBrush(color))
        elif visibility == 2:
            semi = QColor(color.red(), color.green(), color.blue(), 100)
            item = self._scene.addEllipse(rect, QPen(color, 2), QBrush(semi))
        elif visibility == 1:
            pen = QPen(color, 2, Qt.DashLine)
            item = self._scene.addEllipse(rect, pen, QBrush(Qt.NoBrush))
        else:
            gray = QColor(180, 180, 180, 150)
            item = self._scene.addEllipse(rect, QPen(gray, 1), QBrush(Qt.NoBrush))

        item.setZValue(100)
        return item

    def _add_label(self, x: float, y: float, text: str, color: QColor,
                   side: str = "right"):
        """Add a text label near a point.

        L-side labels are placed at the upper-left of the point (text extends
        leftward) so they don't overlap the central AABB box.
        R-side labels stay at the upper-right (current behaviour).
        Font size scales proportionally with the point radius.
        """
        label = self._scene.addSimpleText(text)
        font_size = max(4, int(self._point_radius * 2.5))
        font = QFont("Arial", font_size)
        font.setBold(True)
        label.setFont(font)
        label.setBrush(QBrush(color))
        label.setZValue(101)

        gap = self._point_radius + 2  # same gap as R-side
        if side == "left":
            # Upper-left: mirror of R-side, text extends leftward
            label.setPos(0, 0)  # temp position to measure bounds
            br = label.boundingRect()
            label.setPos(x - gap - br.width(), y - gap)
        else:
            # Upper-right (unchanged)
            label.setPos(x + gap, y - gap)

        return label

    def _render_active_ring(self):
        """Show yellow ring around active side's point."""
        if self._active_ring is not None:
            try:
                self._scene.removeItem(self._active_ring)
            except RuntimeError:
                pass
            self._active_ring = None

        if not self._active_vertebra or not self._show_ring:
            return
        if self._active_vertebra not in self._pedicle_data:
            return

        pdata = self._pedicle_data[self._active_vertebra]
        side_data = pdata.get(self._active_side)
        if not side_data or not side_data.get("center"):
            return

        cx, cy = side_data["center"]["x"], side_data["center"]["y"]
        r = self._point_radius + 4
        rect = QRectF(cx - r, cy - r, r * 2, r * 2)
        pen = QPen(SELECTED_RING, 2, Qt.DashLine)
        self._active_ring = self._scene.addEllipse(rect, pen, QBrush(Qt.NoBrush))
        self._active_ring.setZValue(102)

    # ------------------------------------------------------------------
    # Mouse interaction
    # ------------------------------------------------------------------

    def mousePressEvent(self, event):
        """Handle mouse press."""
        if event.button() == Qt.MiddleButton:
            # Start panning
            self._panning = True
            self._pan_start_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            return

        if event.button() == Qt.LeftButton:
            scene_pos = self.mapToScene(event.pos())
            px, py = scene_pos.x(), scene_pos.y()

            # Priority 1: Check if clicking on the CURRENTLY SELECTED point
            # This ensures dragging moves the selected point, not an overlapping one
            # BUT only if no OTHER point is closer to the click position
            active_hit = False
            if self._active_vertebra and self._show_ring:
                if self._active_vertebra in self._pedicle_data:
                    pdata = self._pedicle_data[self._active_vertebra]
                    side_data = pdata.get(self._active_side, {})
                    if side_data and side_data.get("center") and side_data.get("visibility", 0) > 0:
                        cx, cy = side_data["center"]["x"], side_data["center"]["y"]
                        hit_r = self._point_radius + 8
                        dx, dy = cx - px, cy - py
                        active_dist_sq = dx * dx + dy * dy
                        if active_dist_sq <= hit_r * hit_r:
                            active_hit = True
                            # Check if another point is closer
                            other_hit = self._hit_test_point_excluding(
                                px, py, self._active_vertebra, self._active_side
                            )
                            if other_hit is not None:
                                other_vert, other_side = other_hit
                                other_pdata = self._pedicle_data[other_vert]
                                other_center = other_pdata[other_side]["center"]
                                odx = other_center["x"] - px
                                ody = other_center["y"] - py
                                other_dist_sq = odx * odx + ody * ody
                                if other_dist_sq < active_dist_sq:
                                    # Other point is closer, switch to it
                                    active_hit = False
                                    self._active_vertebra = other_vert
                                    self._active_side = other_side
                                    self._show_ring = True
                                    self._dragging = True
                                    self._update_aabb_highlight()
                                    self._render_active_ring()
                                    self.side_selected.emit()
                                    self.vertebra_clicked.emit(other_vert)
                                    return

                            if active_hit:
                                # Active point is closest, start dragging it
                                self._dragging = True
                                return

            # Priority 2: Check if clicking on any other pedicle point
            hit = self._hit_test_point(px, py)
            if hit is not None:
                vert_name, side = hit
                # Switch to that vertebra and side
                self._active_vertebra = vert_name
                self._active_side = side
                self._show_ring = True
                self._dragging = True
                self._update_aabb_highlight()
                self._render_active_ring()
                self.side_selected.emit()
                self.vertebra_clicked.emit(vert_name)
                return

            # Priority 3: Check if clicking on an AABB box
            hit_vert = self._hit_test_aabb(px, py)
            if hit_vert is not None:
                self._active_vertebra = hit_vert
                self._show_ring = True
                self._update_aabb_highlight()
                self._render_active_ring()
                self.vertebra_clicked.emit(hit_vert)
                return

            # Clicked on empty area
            self._show_ring = False
            self._render_active_ring()
            self.side_deselected.emit()

        elif event.button() == Qt.RightButton:
            self.right_clicked.emit()
            return

        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        """Double-click to place new pedicle point."""
        if event.button() == Qt.LeftButton:
            scene_pos = self.mapToScene(event.pos())
            px, py = scene_pos.x(), scene_pos.y()

            # First check if we're clicking on an existing point
            hit = self._hit_test_point(px, py)
            if hit is not None:
                return  # Already handled in mousePressEvent

            # Determine which AABB we're clicking on
            hit_vert = self._hit_test_aabb(px, py)
            if hit_vert is None:
                return  # Not on any AABB, ignore

            # Set active vertebra to the one under cursor
            self._active_vertebra = hit_vert
            self._show_ring = True
            self._update_aabb_highlight()
            self.vertebra_clicked.emit(hit_vert)

            # Auto-switch side if current side already has a point
            pdata = self._pedicle_data.get(self._active_vertebra, {})
            side_data = pdata.get(self._active_side, {})
            if side_data.get("center"):
                self._active_side = "right" if self._active_side == "left" else "left"
                self.side_selected.emit()

            self._set_point(self._active_side, px, py)

    def mouseMoveEvent(self, event):
        """Handle drag of pedicle point or panning."""
        if self._panning:
            # Pan the view
            delta = event.pos() - self._pan_start_pos
            self._pan_start_pos = event.pos()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x()
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y()
            )
            return

        if self._dragging and self._active_vertebra:
            scene_pos = self.mapToScene(event.pos())
            px = max(0, min(scene_pos.x(), self._img_w))
            py = max(0, min(scene_pos.y(), self._img_h))
            self._set_point(self._active_side, px, py)

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """End drag or pan."""
        if event.button() == Qt.MiddleButton:
            self._panning = False
            self._pan_start_pos = None
            self.setCursor(Qt.ArrowCursor)
            return

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

    def _hit_test_point(self, x: float, y: float) -> Optional[Tuple[str, str]]:
        """Check if (x, y) hits any pedicle point.
        Returns (vert_name, side) or None.
        
        Iterates in reverse insertion order so later vertebrae's points
        get priority when points overlap.
        """
        if not self._pedicle_data:
            return None

        hit_r = self._point_radius + 8
        best = None
        best_dist = float('inf')

        for vert_name, pdata in reversed(list(self._pedicle_data.items())):
            for side in ("left", "right"):
                side_data = pdata.get(side, {})
                if side_data and side_data.get("center") and side_data.get("visibility", 0) > 0:
                    cx, cy = side_data["center"]["x"], side_data["center"]["y"]
                    dx, dy = cx - x, cy - y
                    dist_sq = dx * dx + dy * dy
                    if dist_sq <= hit_r * hit_r and dist_sq < best_dist:
                        best = (vert_name, side)
                        best_dist = dist_sq
        return best

    def _hit_test_point_excluding(
        self, x: float, y: float,
        exclude_vert: str, exclude_side: str
    ) -> Optional[Tuple[str, str]]:
        """Like _hit_test_point but excludes a specific vertebra+side.
        Returns the closest hit or None.
        """
        if not self._pedicle_data:
            return None

        hit_r = self._point_radius + 8
        best = None
        best_dist = float('inf')

        for vert_name, pdata in reversed(list(self._pedicle_data.items())):
            for side in ("left", "right"):
                if vert_name == exclude_vert and side == exclude_side:
                    continue
                side_data = pdata.get(side, {})
                if side_data and side_data.get("center") and side_data.get("visibility", 0) > 0:
                    cx, cy = side_data["center"]["x"], side_data["center"]["y"]
                    dx, dy = cx - x, cy - y
                    dist_sq = dx * dx + dy * dy
                    if dist_sq <= hit_r * hit_r and dist_sq < best_dist:
                        best = (vert_name, side)
                        best_dist = dist_sq
        return best

    def _hit_test_aabb(self, x: float, y: float) -> Optional[str]:
        """Check if (x, y) is inside any AABB box. Returns vert name or None.
        
        Iterates in reverse order so later vertebrae (drawn on top) get priority
        when boxes overlap.
        """
        for ann in reversed(self._obb_annotations):
            if ann.shape_type != "obb" or not ann.class_name:
                continue
            xs = [p.x for p in ann.points[:4]]
            ys = [p.y for p in ann.points[:4]]
            if min(xs) <= x <= max(xs) and min(ys) <= y <= max(ys):
                return ann.class_name
        return None

    def _set_point(self, side: str, x: float, y: float):
        """Set or update a pedicle point."""
        if not self._active_vertebra:
            return

        if self._active_vertebra not in self._pedicle_data:
            self._pedicle_data[self._active_vertebra] = {
                "left": {}, "right": {}, "flagged": False,
            }

        pdata = self._pedicle_data[self._active_vertebra]
        pdata[side] = {
            "center": {"x": x, "y": y},
            "visibility": max(pdata.get(side, {}).get("visibility", 3), 3),
        }

        self._render_points()
        self._render_active_ring()
        self.viewport().update()
        self.point_changed.emit()

    def clear_active_point(self):
        """Clear active side's point (for Delete key)."""
        if not self._active_vertebra or not self._show_ring:
            return
        if self._active_vertebra not in self._pedicle_data:
            return
        pdata = self._pedicle_data[self._active_vertebra]
        side_data = pdata.get(self._active_side, {})
        if not side_data.get("center"):
            return
        pdata[self._active_side] = {}
        self._render_points()
        self._render_active_ring()
        self.viewport().update()
        self.point_changed.emit()

    def set_visibility(self, visibility: int):
        """Set visibility for active side."""
        if not self._active_vertebra or not self._show_ring:
            return
        if self._active_vertebra not in self._pedicle_data:
            return
        pdata = self._pedicle_data[self._active_vertebra]
        side_data = pdata.get(self._active_side, {})
        if not side_data.get("center"):
            return
        side_data["visibility"] = visibility
        self._render_points()
        self._render_active_ring()
        self.viewport().update()
        self.point_changed.emit()
