"""Window for pedicle annotation on full X-ray images."""

import os
from pathlib import Path
from typing import Dict, List, Optional

from PyQt5.QtCore import Qt, QSettings
from PyQt5.QtGui import QBrush, QColor, QKeySequence, QPalette
from PyQt5.QtWidgets import (
    QAction, QButtonGroup, QCheckBox, QDoubleSpinBox, QFileDialog, QHBoxLayout,
    QLabel, QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
    QPushButton, QRadioButton, QShortcut, QSplitter, QVBoxLayout, QWidget,
)

from ..core.converter import YOLOConverter
from ..core.models import PEDICLE_VISIBILITY, VERTEBRA_CLASSES, VertebraCategory, get_vertebra_category, vertebra_sort_key
from .pedicle_full_canvas import PedicleFullCanvas

_MUTED_STYLE = "color: #888; font-size: 11px;"

# Cache key for pedicle annotations
PEDICLE_CACHE_KEY = "pedicle_annotations"


class PedicleFullWindow(QMainWindow):
    """Window for annotating pedicles on full X-ray images."""

    def __init__(self, image_infos: List[dict], converter: YOLOConverter,
                 cache: dict, cache_path: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("整图椎弓根标注")
        self.resize(1400, 900)

        self._converter = converter
        self._image_infos = image_infos
        self._cache = cache
        self._cache_path = cache_path
        self._settings = QSettings("SpineAnnotator", "PedicleFullWindow")

        # State
        self._current_image_index: int = -1
        self._current_annotations: List = []  # OBB annotations
        self._pedicle_data: Dict[str, dict] = {}  # vert_name -> pedicle dict
        self._current_vert: Optional[str] = None
        self._has_unsaved_changes: bool = False

        # Track which images have been saved (persisted to cache on disk)
        self._saved_images: set = set()

        # Get pedicle data from cache
        self._all_pedicle_data: Dict[str, dict] = cache.get(PEDICLE_CACHE_KEY, {})
        # Mark images that already have pedicle data as saved
        self._saved_images = set(self._all_pedicle_data.keys())

        self._build_ui()
        self._init_shortcuts()

        # Apply saved point size
        raw = self._settings.value("point_radius", 2)
        try:
            radius = float(raw)
        except (TypeError, ValueError):
            radius = 2.0
        self._canvas.set_point_radius(radius)
        # Sync spinbox to the actual radius used (block signal to avoid
        # redundant set_point_radius call during init)
        self._spin_point_size.blockSignals(True)
        self._spin_point_size.setValue(radius)
        self._spin_point_size.blockSignals(False)

        # Load first image
        if self._image_infos:
            self._go_to_image(0)

    # ------------------------------------------------------------------
    # UI Setup
    # ------------------------------------------------------------------

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(4, 4, 4, 4)

        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        # --- Left: Image list ---
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(QLabel("图片列表"))

        self._image_list = QListWidget()
        self._image_list.currentRowChanged.connect(self._on_image_selected)
        left_layout.addWidget(self._image_list)

        # Populate image list
        for idx, info in enumerate(self._image_infos):
            name = Path(info["image_path"]).stem
            split = info.get("split", "")
            display = f"[{split}] {name}" if split else name
            item = QListWidgetItem(display)
            self._image_list.addItem(item)

        splitter.addWidget(left_widget)

        # --- Center: Canvas ---
        self._canvas = PedicleFullCanvas()
        self._canvas.vertebra_clicked.connect(self._on_vert_clicked)
        self._canvas.point_changed.connect(self._on_point_changed)
        self._canvas.right_clicked.connect(self._toggle_flag)
        self._canvas.side_deselected.connect(self._on_side_deselected)
        self._canvas.side_selected.connect(self._on_side_selected)
        splitter.addWidget(self._canvas)

        # --- Right: Control panel ---
        right_widget = QWidget()
        right_widget.setFixedWidth(280)
        right_layout = QVBoxLayout(right_widget)

        # Source info
        right_layout.addWidget(QLabel("来源图片"))
        self._info_source = QLabel("—")
        self._info_source.setStyleSheet(_MUTED_STYLE)
        right_layout.addWidget(self._info_source)

        # Vertebra list
        right_layout.addWidget(QLabel("椎骨列表"))
        self._vert_list = QListWidget()
        self._vert_list.currentRowChanged.connect(self._on_vert_list_selected)
        right_layout.addWidget(self._vert_list)

        # Current vertebra info
        right_layout.addWidget(QLabel("当前椎骨"))
        self._info_vert = QLabel("—")
        self._info_vert.setStyleSheet("font-size: 14px; font-weight: bold;")
        right_layout.addWidget(self._info_vert)

        # Side selection
        side_row = QHBoxLayout()
        side_row.addWidget(QLabel("侧边:"))
        self._btn_left = QPushButton("图像左 (L)")
        self._btn_left.setCheckable(True)
        self._btn_left.clicked.connect(lambda: self._set_active_side("left"))
        side_row.addWidget(self._btn_left)
        self._btn_right = QPushButton("图像右 (R)")
        self._btn_right.setCheckable(True)
        self._btn_right.clicked.connect(lambda: self._set_active_side("right"))
        side_row.addWidget(self._btn_right)
        right_layout.addLayout(side_row)

        # Visibility
        right_layout.addWidget(QLabel("可见性"))
        self._vis_group = QButtonGroup(self)
        vis_names = {0: "不可见 (0)", 1: "遮挡 (1)", 2: "模糊可见 (2)", 3: "清晰可见 (3)"}
        for v in (3, 2, 1, 0):
            rb = QRadioButton(vis_names[v])
            self._vis_group.addButton(rb, v)
            right_layout.addWidget(rb)
        self._vis_group.idClicked.connect(self._on_visibility_changed)
        btn_default = self._vis_group.button(3)
        if btn_default:
            btn_default.setChecked(True)

        # Point size
        size_row = QHBoxLayout()
        size_row.addWidget(QLabel("点位大小:"))
        self._spin_point_size = QDoubleSpinBox()
        self._spin_point_size.setRange(0.5, 50)
        self._spin_point_size.setSingleStep(0.5)
        self._spin_point_size.setDecimals(1)
        self._spin_point_size.setValue(2.0)
        self._spin_point_size.setToolTip("调整 L/R 标记圆圈的显示大小")
        self._spin_point_size.valueChanged.connect(self._on_point_size_changed)
        size_row.addWidget(self._spin_point_size)
        right_layout.addLayout(size_row)

        # Flag button
        self._btn_flag = QPushButton("标记难点 (M)")
        self._btn_flag.setCheckable(True)
        self._btn_flag.clicked.connect(self._toggle_flag)
        right_layout.addWidget(self._btn_flag)

        # Delete point button
        self._btn_delete = QPushButton("移除当前侧点位 (Del)")
        self._btn_delete.clicked.connect(self._on_clear_point)
        right_layout.addWidget(self._btn_delete)

        # Save button
        self._btn_save = QPushButton("保存当前 (Ctrl+S)")
        self._btn_save.clicked.connect(self._save)
        right_layout.addWidget(self._btn_save)

        right_layout.addStretch()

        # Layer control
        right_layout.addWidget(QLabel("图层控制"))
        self._chk_cervical = QCheckBox("颈椎 C")
        self._chk_cervical.setChecked(True)
        self._chk_cervical.stateChanged.connect(self._on_layer_changed)
        right_layout.addWidget(self._chk_cervical)

        self._chk_thoracic = QCheckBox("胸椎 T")
        self._chk_thoracic.setChecked(True)
        self._chk_thoracic.stateChanged.connect(self._on_layer_changed)
        right_layout.addWidget(self._chk_thoracic)

        self._chk_lumbar = QCheckBox("腰椎 L")
        self._chk_lumbar.setChecked(True)
        self._chk_lumbar.stateChanged.connect(self._on_layer_changed)
        right_layout.addWidget(self._chk_lumbar)

        self._chk_sacral = QCheckBox("骶椎 S")
        self._chk_sacral.setChecked(True)
        self._chk_sacral.stateChanged.connect(self._on_layer_changed)
        right_layout.addWidget(self._chk_sacral)

        right_layout.addStretch()

        # Export button
        self._btn_export = QPushButton("导出 Crop 数据集…")
        self._btn_export.setStyleSheet(
            "QPushButton { background-color: #2196F3; color: white; "
            "font-weight: bold; padding: 8px; }"
        )
        self._btn_export.clicked.connect(self._export_crops)
        right_layout.addWidget(self._btn_export)

        splitter.addWidget(right_widget)
        splitter.setSizes([200, 800, 280])

    def _init_shortcuts(self):
        # Image navigation
        QShortcut(QKeySequence("Ctrl+Left"), self, self._prev_image)
        QShortcut(QKeySequence("Ctrl+Right"), self, self._next_image)

        # Vertebra navigation
        QShortcut(QKeySequence("Up"), self, self._prev_vert)
        QShortcut(QKeySequence("Down"), self, self._next_vert)

        # Side selection
        QShortcut(QKeySequence("L"), self, lambda: self._set_active_side("left"))
        QShortcut(QKeySequence("R"), self, lambda: self._set_active_side("right"))
        QShortcut(QKeySequence("Left"), self, lambda: self._set_active_side("left"))
        QShortcut(QKeySequence("Right"), self, lambda: self._set_active_side("right"))

        # Visibility
        for v in range(4):
            QShortcut(QKeySequence(str(v)), self, lambda val=v: self._set_visibility(val))

        # Flag
        QShortcut(QKeySequence("M"), self, self._toggle_flag)

        # Delete
        QShortcut(QKeySequence("Delete"), self, self._on_clear_point)

        # Save
        QShortcut(QKeySequence("Ctrl+S"), self, self._save)

    # ------------------------------------------------------------------
    # Image navigation
    # ------------------------------------------------------------------

    def _go_to_image(self, index: int):
        if not (0 <= index < len(self._image_infos)):
            return
        if index == self._current_image_index:
            return

        # Auto-save previous image
        if self._current_image_index >= 0 and self._has_unsaved_changes:
            self._save_to_cache()
            # Mark previous image as saved
            prev_rel = self._image_infos[self._current_image_index]["rel_path"]
            self._saved_images.add(prev_rel)

        self._current_image_index = index
        self._has_unsaved_changes = False
        info = self._image_infos[index]

        # Load annotations
        try:
            ann = self._converter.load_single(
                info["image_path"], info.get("label_path"),
                info["width"], info["height"],
                cache_entry=self._cache.get(info["rel_path"], {}),
            )
            self._current_annotations = ann.annotations
        except Exception:
            self._current_annotations = []

        # Get pedicle data for this image
        rel_path = info["rel_path"]
        self._pedicle_data = self._all_pedicle_data.get(rel_path, {})

        # Load into canvas
        self._canvas.load_image(
            info["image_path"], self._current_annotations,
            info["width"], info["height"],
        )
        self._canvas.set_pedicle_data(self._pedicle_data)

        # Update vertebra list
        self._update_vert_list()

        # Update source info
        source_stem = Path(info["image_path"]).stem
        split = info.get("split", "")
        self._info_source.setText(f"{source_stem}\nsplit: {split or '—'}")

        # Select first vertebra
        if self._vert_list.count() > 0:
            self._vert_list.setCurrentRow(0)

        # Update image list selection
        self._image_list.blockSignals(True)
        self._image_list.setCurrentRow(index)
        self._image_list.blockSignals(False)

        # Update all image list item colours
        self._refresh_image_list_colours()

    def _prev_image(self):
        if self._current_image_index > 0:
            self._go_to_image(self._current_image_index - 1)

    def _next_image(self):
        if self._current_image_index < len(self._image_infos) - 1:
            self._go_to_image(self._current_image_index + 1)

    def _on_image_selected(self, row: int):
        if row >= 0 and row != self._current_image_index:
            self._go_to_image(row)

    # ------------------------------------------------------------------
    # Vertebra navigation
    # ------------------------------------------------------------------

    def _update_vert_list(self):
        """Populate vertebra list from current annotations."""
        self._vert_list.clear()
        vert_names = []
        for ann in self._current_annotations:
            if ann.shape_type == "obb" and ann.class_name:
                vert_names.append(ann.class_name)

        # Sort by anatomical order
        vert_names.sort(key=vertebra_sort_key)

        for name in vert_names:
            pdata = self._pedicle_data.get(name, {})
            has_left = bool(pdata.get("left", {}).get("center"))
            has_right = bool(pdata.get("right", {}).get("center"))
            flagged = pdata.get("flagged", False)

            prefix = "⚑ " if flagged else ""
            suffix = ""
            if has_left and has_right:
                suffix = " ✓"
            elif has_left or has_right:
                suffix = " ◐"

            item = QListWidgetItem(f"{prefix}{name}{suffix}")
            self._vert_list.addItem(item)

        self._current_vert = vert_names[0] if vert_names else None
        if self._current_vert:
            self._canvas.set_active_vertebra(self._current_vert)
            self._sync_ui_to_vert()

    def _update_vert_list_preserve_active(self):
        """Rebuild vertebra list while preserving the current active vertebra.

        Called during drag operations (via point_changed) so that the active
        vertebra is NOT reset to the first item on every mouse move.
        """
        # Remember current active vertebra from canvas (it may have been
        # changed by user interaction before this method runs)
        active_vert = self._canvas._active_vertebra

        self._vert_list.clear()
        vert_names = []
        for ann in self._current_annotations:
            if ann.shape_type == "obb" and ann.class_name:
                vert_names.append(ann.class_name)

        vert_names.sort(key=vertebra_sort_key)

        for name in vert_names:
            pdata = self._pedicle_data.get(name, {})
            has_left = bool(pdata.get("left", {}).get("center"))
            has_right = bool(pdata.get("right", {}).get("center"))
            flagged = pdata.get("flagged", False)

            prefix = "⚑ " if flagged else ""
            suffix = ""
            if has_left and has_right:
                suffix = " ✓"
            elif has_left or has_right:
                suffix = " ◐"

            item = QListWidgetItem(f"{prefix}{name}{suffix}")
            self._vert_list.addItem(item)

        # Restore selection to the active vertebra (not always first item)
        target = active_vert if active_vert and active_vert in vert_names else (vert_names[0] if vert_names else None)
        if target:
            self._current_vert = target
            for i in range(self._vert_list.count()):
                item_text = self._vert_list.item(i).text().lstrip("⚑ ").rstrip(" ✓◐")
                if item_text == target:
                    self._vert_list.blockSignals(True)
                    self._vert_list.setCurrentRow(i)
                    self._vert_list.blockSignals(False)
                    break
            self._sync_ui_to_vert()

    def _on_vert_clicked(self, vert_name: str):
        """Canvas clicked on a vertebra AABB."""
        self._current_vert = vert_name
        self._sync_ui_to_vert()
        # Update vert list selection
        for i in range(self._vert_list.count()):
            item_text = self._vert_list.item(i).text().lstrip("⚑ ").rstrip(" ✓◐")
            if item_text == vert_name:
                self._vert_list.blockSignals(True)
                self._vert_list.setCurrentRow(i)
                self._vert_list.blockSignals(False)
                break

    def _on_vert_list_selected(self, row: int):
        if row < 0 or row >= self._vert_list.count():
            return
        item_text = self._vert_list.item(row).text().lstrip("⚑ ").rstrip(" ✓◐")
        # Find the vert name from annotations
        for ann in self._current_annotations:
            if ann.shape_type == "obb" and ann.class_name == item_text:
                self._current_vert = item_text
                self._canvas.set_active_vertebra(item_text)
                self._sync_ui_to_vert()
                return

    def _prev_vert(self):
        row = self._vert_list.currentRow()
        if row > 0:
            self._vert_list.setCurrentRow(row - 1)

    def _next_vert(self):
        row = self._vert_list.currentRow()
        if row < self._vert_list.count() - 1:
            self._vert_list.setCurrentRow(row + 1)

    # ------------------------------------------------------------------
    # Side / Visibility / Flag
    # ------------------------------------------------------------------

    def _set_active_side(self, side: str):
        self._btn_left.setChecked(side == "left")
        self._btn_right.setChecked(side == "right")
        self._canvas.set_active_side(side)
        self._set_visibility_enabled(True)
        self._sync_visibility_to_side()

    def _set_visibility(self, visibility: int):
        if not self._canvas._show_ring:
            return
        self._canvas.set_visibility(visibility)

    def _on_visibility_changed(self, vis_id: int):
        self._set_visibility(vis_id)

    def _on_side_deselected(self):
        self._btn_left.setChecked(False)
        self._btn_right.setChecked(False)
        self._set_visibility_enabled(False)

    def _on_side_selected(self):
        side = self._canvas._active_side
        self._btn_left.setChecked(side == "left")
        self._btn_right.setChecked(side == "right")
        self._set_visibility_enabled(True)
        self._sync_visibility_to_side()

    def _sync_visibility_to_side(self):
        """Sync visibility radio to current side's point data."""
        if not self._current_vert or self._current_vert not in self._pedicle_data:
            return
        pdata = self._pedicle_data[self._current_vert]
        side = self._canvas._active_side
        side_data = pdata.get(side, {})
        vis = side_data.get("visibility", 3)
        btn = self._vis_group.button(vis)
        if btn:
            btn.setChecked(True)

    def _set_visibility_enabled(self, enabled: bool):
        for btn in self._vis_group.buttons():
            btn.setEnabled(enabled)

    def _toggle_flag(self):
        if not self._current_vert:
            return
        if self._current_vert not in self._pedicle_data:
            self._pedicle_data[self._current_vert] = {
                "left": {}, "right": {}, "flagged": False,
            }
        pdata = self._pedicle_data[self._current_vert]
        pdata["flagged"] = not pdata.get("flagged", False)
        self._btn_flag.setChecked(pdata["flagged"])
        self._has_unsaved_changes = True
        self._update_vert_list_preserve_active()
        self._refresh_image_list_colours()

    def _on_clear_point(self):
        self._canvas.clear_active_point()

    def _on_point_changed(self):
        self._has_unsaved_changes = True
        self._update_vert_list_preserve_active()
        self._refresh_image_list_colours()

    def _on_point_size_changed(self, value: float):
        self._canvas.set_point_radius(value)
        self._settings.setValue("point_radius", value)

    def _on_layer_changed(self):
        """Handle layer visibility checkbox changes."""
        visibility = {
            VertebraCategory.CERVICAL: self._chk_cervical.isChecked(),
            VertebraCategory.THORACIC: self._chk_thoracic.isChecked(),
            VertebraCategory.LUMBAR: self._chk_lumbar.isChecked(),
            VertebraCategory.SACRAL: self._chk_sacral.isChecked(),
        }
        self._canvas.set_layer_visibility(visibility)
        self._canvas.viewport().update()

    def _sync_ui_to_vert(self):
        """Sync UI controls to current vertebra state."""
        if not self._current_vert:
            self._info_vert.setText("—")
            self._btn_flag.setChecked(False)
            self._set_visibility_enabled(False)
            return

        self._info_vert.setText(self._current_vert)

        pdata = self._pedicle_data.get(self._current_vert, {})
        self._btn_flag.setChecked(pdata.get("flagged", False))
        self._set_visibility_enabled(True)
        self._sync_visibility_to_side()

    # ------------------------------------------------------------------
    # Save / Export
    # ------------------------------------------------------------------

    def _save(self):
        """Save pedicle annotations to cache."""
        self._save_to_cache()
        # Persist cache to disk
        if self._cache_path:
            self._converter.save_progress_cache(self._cache_path, self._cache)
        self._has_unsaved_changes = False
        # Mark current image as saved
        if self._current_image_index >= 0:
            rel_path = self._image_infos[self._current_image_index]["rel_path"]
            self._saved_images.add(rel_path)
        self._refresh_image_list_colours()
        self.statusBar().showMessage("已保存", 3000)

    def _save_to_cache(self):
        """Save current pedicle data to in-memory cache."""
        if self._current_image_index < 0:
            return
        info = self._image_infos[self._current_image_index]
        rel_path = info["rel_path"]

        if PEDICLE_CACHE_KEY not in self._cache:
            self._cache[PEDICLE_CACHE_KEY] = {}

        self._cache[PEDICLE_CACHE_KEY][rel_path] = self._pedicle_data
        self._all_pedicle_data = self._cache[PEDICLE_CACHE_KEY]

    def _export_crops(self):
        """Export pedicle annotations as crop dataset."""
        from ..core.pedicle_exporter import export_pedicle_crops

        if not self._all_pedicle_data:
            QMessageBox.information(self, "提示", "没有椎弓根标注数据可导出。")
            return

        # Auto-save first
        if self._has_unsaved_changes:
            self._save_to_cache()

        # Determine default export directory
        last_export = self._settings.value("last_export_dir", "")
        if last_export and Path(last_export).exists():
            default_dir = last_export
        else:
            # Default: pedicle_crop_dataset under dataset root's parent
            if self._image_infos:
                dataset_root = Path(self._image_infos[0]["image_path"]).parent.parent
                default_dir = str(dataset_root.parent / "pedicle_crop_dataset")
            else:
                default_dir = "."

        output_dir = QFileDialog.getExistingDirectory(
            self, "选择导出目录", default_dir,
        )
        if not output_dir:
            return

        # Remember export directory
        self._settings.setValue("last_export_dir", output_dir)

        try:
            result = export_pedicle_crops(
                image_infos=self._image_infos,
                pedicle_data=self._all_pedicle_data,
                cache=self._cache,
                converter=self._converter,
                output_dir=output_dir,
            )
            total = result["total_crops"]
            QMessageBox.information(
                self, "导出完成",
                f"已导出 {total} 张 crop 图片到:\n{output_dir}",
            )
        except Exception as exc:
            QMessageBox.critical(
                self, "导出失败",
                f"导出过程中出错:\n{exc}",
            )

    # ------------------------------------------------------------------
    # Window lifecycle
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        if self._has_unsaved_changes:
            self._save_to_cache()
            # Mark current image as saved before closing
            if self._current_image_index >= 0:
                rel_path = self._image_infos[self._current_image_index]["rel_path"]
                self._saved_images.add(rel_path)
            if self._cache_path:
                self._converter.save_progress_cache(self._cache_path, self._cache)
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Image list visual feedback
    # ------------------------------------------------------------------

    def _refresh_image_list_colours(self):
        """Update colours of all image list items based on saved state."""
        palette = self._image_list.palette()
        for i in range(self._image_list.count()):
            item = self._image_list.item(i)
            if i >= len(self._image_infos):
                break
            rel_path = self._image_infos[i]["rel_path"]
            is_current = (i == self._current_image_index)
            is_saved = rel_path in self._saved_images

            if is_current and self._has_unsaved_changes:
                # Orange: current image with unsaved modifications
                item.setForeground(QBrush(QColor("#e8590c")))
            elif is_saved:
                # Gray: already saved to disk
                item.setForeground(palette.brush(QPalette.Disabled, QPalette.Text))
            else:
                # Default: not yet worked on
                item.setForeground(palette.brush(QPalette.Active, QPalette.Text))

            font = item.font()
            font.setBold(is_current)
            item.setFont(font)
