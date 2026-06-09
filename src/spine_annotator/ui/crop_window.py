"""Independent window for single-vertebra crop pedicle annotation."""

import os
from pathlib import Path
from typing import List, Optional

from PyQt5.QtCore import Qt, QSettings
from PyQt5.QtGui import QColor, QKeySequence
from PyQt5.QtWidgets import (
    QAction, QButtonGroup, QDoubleSpinBox, QFileDialog, QHBoxLayout, QLabel,
    QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
    QProgressBar, QPushButton, QRadioButton, QShortcut,
    QSplitter, QVBoxLayout, QWidget,
)

from ..core.crop_converter import CropConverter
from ..core.models import PEDICLE_VISIBILITY, CropPedicleAnnotation
from .crop_canvas import CropCanvas

_MUTED_STYLE = "color: #888; font-size: 11px;"


class CropWindow(QMainWindow):
    """Main window for crop pedicle annotation workflow."""

    def __init__(self, dataset_root: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("椎弓根 Crop 标注工具")
        self.resize(1200, 800)

        self._converter = CropConverter()
        self._dataset_root: Optional[str] = None
        self._crop_infos: List[dict] = []
        self._current_index: int = -1
        self._current_annotation: Optional[CropPedicleAnnotation] = None
        self._cache: dict = {}
        self._progress_cache_path: Optional[str] = None
        self._settings = QSettings("SpineAnnotator", "CropWindow")

        self._build_ui()
        self._init_shortcuts()

        # Apply saved point size to canvas (handle both int and float from QSettings)
        try:
            radius = float(self._settings.value("point_radius", 2))
        except (TypeError, ValueError):
            radius = 2.0
        self._canvas.set_point_radius(radius)

        # Load dataset
        self._load_dataset(dataset_root)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        splitter = QSplitter(Qt.Horizontal)

        # --- Left: image list ---
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(4, 4, 4, 4)

        self._image_list = QListWidget()
        self._image_list.currentRowChanged.connect(self._on_list_selection)
        left_layout.addWidget(self._image_list)

        self._progress_bar = QProgressBar()
        self._progress_bar.setMaximum(1)
        left_layout.addWidget(self._progress_bar)

        self._progress_label = QLabel("0 / 0")
        self._progress_label.setStyleSheet(_MUTED_STYLE)
        left_layout.addWidget(self._progress_label)

        splitter.addWidget(left_widget)

        # --- Center: canvas ---
        self._canvas = CropCanvas()
        self._canvas.point_changed.connect(self._on_point_changed)
        self._canvas.right_clicked.connect(self._toggle_flag_current)
        self._canvas.side_deselected.connect(self._on_side_deselected)
        self._canvas.side_selected.connect(self._on_side_selected_from_canvas)
        splitter.addWidget(self._canvas)

        # --- Right: control panel ---
        right_widget = QWidget()
        right_widget.setFixedWidth(220)
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(8, 8, 8, 8)

        # Source info
        right_layout.addWidget(self._section_label("来源信息"))
        self._info_source = QLabel("—")
        self._info_source.setWordWrap(True)
        self._info_source.setStyleSheet(_MUTED_STYLE)
        right_layout.addWidget(self._info_source)

        self._info_vert = QLabel("—")
        self._info_vert.setStyleSheet(_MUTED_STYLE)
        right_layout.addWidget(self._info_vert)

        right_layout.addSpacing(12)

        # Active side
        right_layout.addWidget(self._section_label("当前标注侧"))
        self._btn_left = QPushButton("图像左 (L)")
        self._btn_right = QPushButton("图像右 (R)")
        self._btn_left.setCheckable(True)
        self._btn_right.setCheckable(True)
        self._btn_left.setChecked(True)
        self._btn_left.clicked.connect(lambda: self._set_active_side("left"))
        self._btn_right.clicked.connect(lambda: self._set_active_side("right"))
        side_row = QHBoxLayout()
        side_row.addWidget(self._btn_left)
        side_row.addWidget(self._btn_right)
        right_layout.addLayout(side_row)

        right_layout.addSpacing(8)

        # Visibility
        right_layout.addWidget(self._section_label("可见性"))
        self._vis_group = QButtonGroup(self)
        vis_names = {1: "遮挡 (1)", 2: "模糊可见 (2)", 3: "清晰可见 (3)"}
        for v in (3, 2, 1):
            rb = QRadioButton(vis_names[v])
            rb.setProperty("vis_value", v)
            self._vis_group.addButton(rb, v)
            right_layout.addWidget(rb)
        self._vis_group.idClicked.connect(self._on_visibility_changed)
        # Default to v=3
        btn_default = self._vis_group.button(3)
        if btn_default:
            btn_default.setChecked(True)

        right_layout.addSpacing(8)

        # Point size
        size_row = QHBoxLayout()
        size_row.addWidget(QLabel("点位大小:"))
        self._spin_point_size = QDoubleSpinBox()
        self._spin_point_size.setRange(0.5, 20)
        self._spin_point_size.setSingleStep(0.5)
        self._spin_point_size.setDecimals(1)
        # Read saved point radius (handle both int and float from QSettings)
        try:
            saved_radius = float(self._settings.value("point_radius", 2))
        except (TypeError, ValueError):
            saved_radius = 2.0
        self._spin_point_size.setValue(saved_radius)
        self._spin_point_size.setToolTip("调整 L/R 标记圆圈的显示大小")
        self._spin_point_size.valueChanged.connect(self._on_point_size_changed)
        size_row.addWidget(self._spin_point_size)
        right_layout.addLayout(size_row)

        right_layout.addSpacing(8)

        # Flag
        self._btn_flag = QPushButton("⚑ 标记难点 (M)")
        self._btn_flag.setCheckable(True)
        self._btn_flag.setToolTip("标记当前图片为标注难点")
        self._btn_flag.toggled.connect(self._on_flag_toggled)
        right_layout.addWidget(self._btn_flag)

        right_layout.addSpacing(4)

        # Actions
        self._btn_clear_point = QPushButton("清除当前侧点位 (Del)")
        self._btn_clear_point.clicked.connect(self._on_clear_point)
        right_layout.addWidget(self._btn_clear_point)

        right_layout.addSpacing(4)

        self._btn_save = QPushButton("保存当前 (Ctrl+S)")
        self._btn_save.clicked.connect(self._save_current)
        right_layout.addWidget(self._btn_save)

        right_layout.addSpacing(4)

        self._btn_save_all = QPushButton("全部导出")
        self._btn_save_all.clicked.connect(self._save_all)
        right_layout.addWidget(self._btn_save_all)

        right_layout.addStretch()

        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        splitter.setStretchFactor(2, 0)

        self.setCentralWidget(splitter)
        self.statusBar().showMessage("就绪")

    def _section_label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet("font-weight: bold; font-size: 12px;")
        return lbl

    def _init_shortcuts(self):
        # Side switch
        QShortcut(QKeySequence("["), self, lambda: self._set_active_side("left"))
        QShortcut(QKeySequence("]"), self, lambda: self._set_active_side("right"))
        # L/R and arrow keys for side selection
        QShortcut(QKeySequence("L"), self, lambda: self._set_active_side("left"))
        QShortcut(QKeySequence("R"), self, lambda: self._set_active_side("right"))
        QShortcut(QKeySequence("Left"), self, lambda: self._set_active_side("left"))
        QShortcut(QKeySequence("Right"), self, lambda: self._set_active_side("right"))
        # Visibility (1=遮挡, 2=模糊可见, 3=清晰可见)
        for v in range(1, 4):
            QShortcut(QKeySequence(str(v)), self, lambda val=v: self._set_visibility(val))
        # Clear
        QShortcut(QKeySequence("Delete"), self, self._on_clear_point)
        # Save
        QShortcut(QKeySequence("Ctrl+S"), self, self._save_current)
        # Flag
        QShortcut(QKeySequence("M"), self, self._toggle_flag_current)
        # Navigation
        QShortcut(QKeySequence("Up"), self, self._go_prev)
        QShortcut(QKeySequence("Down"), self, self._go_next)
        QShortcut(QKeySequence("W"), self, self._go_prev)
        QShortcut(QKeySequence("S"), self, self._go_next)

    # ------------------------------------------------------------------
    # Dataset loading
    # ------------------------------------------------------------------

    def _load_dataset(self, dataset_root: str):
        is_valid, message = self._converter.validate_crop_dataset(dataset_root)
        if not is_valid:
            QMessageBox.warning(self, "数据集格式错误", message)
            return

        self._dataset_root = dataset_root
        self._progress_cache_path = os.path.join(dataset_root, ".annotate_progress.json")
        self._crop_infos = self._converter.scan_crop_dataset(dataset_root)
        self._cache = self._converter.load_progress_cache(self._progress_cache_path)

        # Populate list
        self._image_list.clear()
        for idx, info in enumerate(self._crop_infos):
            vert = info["vertebra"]
            split = info["split"]
            source = info["source_stem"]
            display = f"[{split}] {source}-{vert}" if split else f"{source}-{vert}"
            item = QListWidgetItem(display)
            self._image_list.addItem(item)
            self._apply_item_style(idx)

        self._progress_bar.setMaximum(len(self._crop_infos))
        self._update_progress()

        self.statusBar().showMessage(f"已加载 {len(self._crop_infos)} 张 crop 图片")

        # Auto jump: last image or first unsaved
        target = self._resolve_initial_index()
        if target >= 0:
            self._go_to_image(target)

    def _resolve_initial_index(self) -> int:
        if not self._crop_infos:
            return -1
        last_path = self._converter.get_last_image_path(self._cache)
        if last_path:
            for i, info in enumerate(self._crop_infos):
                if info["rel_path"] == last_path:
                    if not self._is_saved(info["rel_path"]):
                        return i
            # Last image is saved, find first unsaved
        for i, info in enumerate(self._crop_infos):
            if not self._is_saved(info["rel_path"]):
                return i
        # All saved
        if last_path:
            for i, info in enumerate(self._crop_infos):
                if info["rel_path"] == last_path:
                    return i
        return 0

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def _on_list_selection(self, row: int):
        if row >= 0:
            self._go_to_image(row)

    def _go_to_image(self, index: int):
        if not (0 <= index < len(self._crop_infos)):
            return

        # Avoid redundant processing if already on this image
        # (can happen via signal re-entry from setCurrentRow)
        if index == self._current_index and self._current_annotation is not None:
            return

        # Auto-save current annotation to disk before switching (if modified)
        if (self._current_index >= 0 and self._current_annotation
                and self._current_annotation.modified):
            self._save_current(silent=True)

        prev = self._current_index
        self._current_index = index
        info = self._crop_infos[index]

        # Load annotation
        cache_entry = self._cache.get(info["rel_path"])
        self._current_annotation = self._converter.load_crop_annotation(
            info["image_path"],
            info.get("label_path"),
            info["width"], info["height"],
            cache_entry=cache_entry,
        )

        # Load canvas
        self._canvas.load_image(info["image_path"], self._current_annotation)

        # Reset visibility to disabled (no side selected on new image)
        self._set_visibility_enabled(False)

        # Update source info
        source = info["source_stem"]
        vert = info["vertebra"]
        split = info["split"]
        self._info_source.setText(f"原图: {source}\nsplit: {split or '—'}")
        self._info_vert.setText(f"椎骨: {vert}")

        # Record last image path (in-memory only, persist on close/save)
        self._converter.set_last_image_path(self._cache, info["rel_path"])

        # Sync flag button state
        is_flagged = self._is_flagged(info["rel_path"])
        self._btn_flag.blockSignals(True)
        self._btn_flag.setChecked(is_flagged)
        self._btn_flag.blockSignals(False)

        # Update list styles
        if prev >= 0 and prev != index:
            self._apply_item_style(prev)
        self._apply_item_style(index)

        self._update_progress()

    def _go_prev(self):
        if self._current_index > 0:
            self._image_list.setCurrentRow(self._current_index - 1)

    def _go_next(self):
        if self._current_index < len(self._crop_infos) - 1:
            self._image_list.setCurrentRow(self._current_index + 1)

    # ------------------------------------------------------------------
    # Side & visibility controls
    # ------------------------------------------------------------------

    def _set_active_side(self, side: str):
        self._btn_left.setChecked(side == "left")
        self._btn_right.setChecked(side == "right")
        self._canvas.set_active_side(side)
        self._set_visibility_enabled(True)
        # Sync visibility radio to current side
        if self._current_annotation:
            pt = (self._current_annotation.image_left if side == "left"
                  else self._current_annotation.image_right)
            btn = self._vis_group.button(pt.visibility)
            if btn:
                btn.setChecked(True)

    def _set_visibility_enabled(self, enabled: bool):
        """Enable or disable all visibility radio buttons."""
        for btn in self._vis_group.buttons():
            btn.setEnabled(enabled)

    def _on_side_deselected(self):
        """Canvas clicked empty area → uncheck side buttons, disable visibility."""
        self._btn_left.setChecked(False)
        self._btn_right.setChecked(False)
        self._set_visibility_enabled(False)

    def _on_side_selected_from_canvas(self):
        """Canvas clicked on a point → sync side buttons, enable visibility."""
        side = self._canvas._active_side
        self._btn_left.setChecked(side == "left")
        self._btn_right.setChecked(side == "right")
        self._set_visibility_enabled(True)
        # Sync visibility radio to current side's point
        if self._current_annotation:
            pt = (self._current_annotation.image_left if side == "left"
                  else self._current_annotation.image_right)
            btn = self._vis_group.button(pt.visibility)
            if btn:
                btn.setChecked(True)

    def _on_visibility_changed(self, vis_id: int):
        self._set_visibility(vis_id)

    def _set_visibility(self, visibility: int):
        if not self._canvas._show_ring:
            return  # ignore if no side is selected
        self._canvas.set_visibility(visibility)

    def _on_clear_point(self):
        self._canvas.clear_active_point()

    def _on_point_size_changed(self, value: int):
        """Point size spinbox changed."""
        self._canvas.set_point_radius(value)
        self._settings.setValue("point_radius", value)

    def _on_point_changed(self):
        """Canvas signals that a point was modified."""
        if self._current_annotation:
            self._current_annotation.modified = True
            self._apply_item_style(self._current_index)
            self._update_progress()
        # Sync visibility radio
        if self._current_annotation:
            pt = (self._current_annotation.image_left if self._canvas._active_side == "left"
                  else self._current_annotation.image_right)
            btn = self._vis_group.button(pt.visibility)
            if btn:
                btn.setChecked(True)

    # ------------------------------------------------------------------
    # Save & export
    # ------------------------------------------------------------------

    def _save_current(self, silent: bool = False):
        if not self._current_annotation or not self._dataset_root:
            if not silent:
                QMessageBox.warning(self, "提示", "没有可保存的标注")
            return

        info = self._crop_infos[self._current_index]
        rel_path = info["rel_path"]

        # Build label output path
        split = info["split"]
        crop_stem = Path(info["image_path"]).stem
        if split:
            lbl_dir = Path(self._dataset_root) / split / "labels"
        else:
            lbl_dir = Path(self._dataset_root) / "labels"
        lbl_dir.mkdir(parents=True, exist_ok=True)
        lbl_path = lbl_dir / f"{crop_stem}.txt"

        self._converter.save_crop_label(self._current_annotation, str(lbl_path))

        # Update cache (preserve existing fields like flagged)
        entry = self._cache.setdefault(rel_path, {})
        entry.update({
            "modified": False,
            "saved": True,
            "pedicle_states": self._converter.build_pedicle_states(self._current_annotation),
        })
        self._current_annotation.modified = False
        if self._progress_cache_path:
            self._converter.save_progress_cache(self._progress_cache_path, self._cache)

        self._apply_item_style(self._current_index)
        self._update_progress()
        if not silent:
            self.statusBar().showMessage(f"已保存: {crop_stem}", 3000)

    def _save_all(self):
        """Export all annotated crop labels."""
        if not self._crop_infos or not self._dataset_root:
            QMessageBox.warning(self, "提示", "没有可导出的数据")
            return

        count = 0
        for i, info in enumerate(self._crop_infos):
            rel_path = info["rel_path"]
            cache_entry = self._cache.get(rel_path, {})

            # Load annotation
            ann = self._converter.load_crop_annotation(
                info["image_path"], info.get("label_path"),
                info["width"], info["height"],
                cache_entry=cache_entry,
            )

            # Skip if no pedicle data at all
            has_data = (
                (ann.image_left.center and ann.image_left.visibility > 0)
                or (ann.image_right.center and ann.image_right.visibility > 0)
            )
            if not has_data:
                continue

            # Save label
            split = info["split"]
            crop_stem = Path(info["image_path"]).stem
            if split:
                lbl_dir = Path(self._dataset_root) / split / "labels"
            else:
                lbl_dir = Path(self._dataset_root) / "labels"
            lbl_dir.mkdir(parents=True, exist_ok=True)
            lbl_path = lbl_dir / f"{crop_stem}.txt"
            self._converter.save_crop_label(ann, str(lbl_path))

            # Update cache (preserve existing fields like flagged)
            entry = self._cache.setdefault(rel_path, {})
            entry.update({
                "modified": False,
                "saved": True,
                "pedicle_states": self._converter.build_pedicle_states(ann),
            })
            count += 1

        if self._progress_cache_path:
            self._converter.save_progress_cache(self._progress_cache_path, self._cache)

        for i in range(len(self._crop_infos)):
            self._apply_item_style(i)
        self._update_progress()

        QMessageBox.information(self, "导出完成", f"已导出 {count} 个标注文件")

    def _checkpoint_to_cache(self, in_memory_only: bool = False):
        """Persist current annotation geometry to cache.

        Args:
            in_memory_only: If True, only update in-memory cache (fast).
                           If False, also write to disk (for explicit save/close).
        """
        if not self._current_annotation or not self._progress_cache_path:
            return
        info = self._crop_infos[self._current_index]
        rel_path = info["rel_path"]
        existing = self._cache.get(rel_path, {})
        existing.update({
            "pedicle_states": self._converter.build_pedicle_states(self._current_annotation),
            "modified": True,
        })
        existing.setdefault("saved", False)
        self._cache[rel_path] = existing
        if not in_memory_only:
            self._converter.save_progress_cache(self._progress_cache_path, self._cache)

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------

    def _is_saved(self, rel_path: str) -> bool:
        return bool(self._cache.get(rel_path, {}).get("saved"))

    def _is_flagged(self, rel_path: str) -> bool:
        return bool(self._cache.get(rel_path, {}).get("flagged"))

    def _toggle_flag_current(self):
        """Toggle flag on current image."""
        if self._current_index < 0 or not self._crop_infos:
            return
        rel_path = self._crop_infos[self._current_index]["rel_path"]
        new_flag = not self._is_flagged(rel_path)
        entry = self._cache.setdefault(rel_path, {})
        entry["flagged"] = new_flag
        self._apply_item_style(self._current_index)
        self._btn_flag.blockSignals(True)
        self._btn_flag.setChecked(new_flag)
        self._btn_flag.blockSignals(False)
        self.statusBar().showMessage(
            "已标记难点" if new_flag else "已取消标记", 2000,
        )

    def _on_flag_toggled(self, checked: bool):
        """Flag button toggled from UI."""
        if self._current_index < 0 or not self._crop_infos:
            return
        rel_path = self._crop_infos[self._current_index]["rel_path"]
        entry = self._cache.setdefault(rel_path, {})
        entry["flagged"] = checked
        self._apply_item_style(self._current_index)

    def _apply_item_style(self, index: int):
        if not (0 <= index < self._image_list.count()):
            return
        item = self._image_list.item(index)
        if not item:
            return

        info = self._crop_infos[index]
        rel_path = info["rel_path"]
        is_current = (index == self._current_index)
        is_modified = (
            is_current
            and self._current_annotation is not None
            and self._current_annotation.modified
        )
        is_saved = self._is_saved(rel_path)
        is_flagged = self._is_flagged(rel_path)

        # Update display text with flag prefix
        vert = info["vertebra"]
        split = info["split"]
        source = info["source_stem"]
        prefix = "⚑ " if is_flagged else ""
        base = f"[{split}] {source}-{vert}" if split else f"{source}-{vert}"
        item.setText(f"{prefix}{base}")

        palette = self._image_list.palette()
        if is_modified:
            item.setForeground(QColor("#e8590c"))  # orange
        elif is_flagged:
            item.setForeground(QColor("#e03131"))  # red for flagged
        elif is_saved:
            item.setForeground(palette.brush(palette.Disabled, palette.Text).color())
        else:
            item.setForeground(palette.brush(palette.Active, palette.Text).color())

        font = item.font()
        font.setBold(is_current)
        item.setFont(font)

    def _update_progress(self):
        total = len(self._crop_infos)
        saved = sum(1 for info in self._crop_infos if self._is_saved(info["rel_path"]))
        self._progress_bar.setValue(saved)
        self._progress_label.setText(f"{saved} / {total}")

    def closeEvent(self, event):
        """Auto-save and persist cache to disk on close."""
        if self._current_annotation and self._current_annotation.modified:
            self._save_current(silent=True)
        elif self._progress_cache_path:
            # Always persist cache on close (for last_image_path etc.)
            self._converter.save_progress_cache(self._progress_cache_path, self._cache)
        super().closeEvent(event)
