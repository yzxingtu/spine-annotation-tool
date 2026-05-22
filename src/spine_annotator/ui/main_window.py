"""Main application window for the spine annotation tool."""

import json
import math
import os
from pathlib import Path
from typing import List, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QHBoxLayout,
    QLabel, QListWidget, QListWidgetItem, QMainWindow,
    QMessageBox, QProgressBar, QPushButton, QShortcut,
    QStatusBar, QVBoxLayout, QWidget,
)

from ..core.converter import YOLOConverter
from ..core.models import ImageAnnotation
from .image_canvas import AnnotationCanvas


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("脊柱椎骨标注工具 - Spine Annotator")
        self.resize(1400, 900)

        # Data
        self._converter = YOLOConverter()
        self._image_infos: List[dict] = []  # scanned image metadata
        self._current_index: int = -1
        self._output_dir: Optional[str] = None
        self._export_format: str = "yolov8_obb"
        self._dataset_root: Optional[str] = None
        self._progress_cache_path: Optional[str] = None

        # Current loaded annotation (only one at a time)
        self._current_annotation: Optional[ImageAnnotation] = None
        self._cache: dict = {}  # progress cache
        
        # Layer visibility
        self._show_vertebrae = True
        self._show_spine = True

        self._init_ui()
        self._init_shortcuts()
        self._init_statusbar()

    def _init_ui(self):
        """Initialize the UI layout."""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(4, 4, 4, 4)

        # --- Left: image list panel ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        left_layout.addWidget(QLabel("图片列表:"))

        self._image_list_widget = QListWidget()
        self._image_list_widget.currentRowChanged.connect(self._on_image_selected)
        left_layout.addWidget(self._image_list_widget)

        # Progress
        self._progress_bar = QProgressBar()
        self._progress_bar.setFormat("已标注: %v / %m")
        left_layout.addWidget(self._progress_bar)

        left_panel.setFixedWidth(220)

        # --- Center: canvas ---
        self._canvas = AnnotationCanvas()
        self._canvas.selection_changed.connect(self._on_annotation_selected)
        self._canvas.annotation_modified.connect(self._on_annotation_modified)

        # --- Right: controls panel ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(4, 4, 4, 4)

        # Dataset controls
        right_layout.addWidget(self._create_section_label("数据集"))
        self._btn_open_dataset = QPushButton("打开 YOLO 数据集")
        self._btn_open_dataset.clicked.connect(self._open_dataset)
        right_layout.addWidget(self._btn_open_dataset)

        self._dataset_path_label = QLabel("未加载")
        self._dataset_path_label.setWordWrap(True)
        self._dataset_path_label.setStyleSheet("color: gray; font-size: 11px;")
        right_layout.addWidget(self._dataset_path_label)

        # Export controls
        right_layout.addWidget(self._create_section_label("导出"))
        self._btn_set_output = QPushButton("设置输出目录")
        self._btn_set_output.clicked.connect(self._set_output_dir)
        right_layout.addWidget(self._btn_set_output)

        self._output_path_label = QLabel("未设置")
        self._output_path_label.setWordWrap(True)
        self._output_path_label.setStyleSheet("color: gray; font-size: 11px;")
        right_layout.addWidget(self._output_path_label)

        self._format_combo = QComboBox()
        self._format_combo.addItems([
            "YOLOv8-OBB (四角点)",
            "YOLOv8-OBB (xywhr)",
        ])
        self._format_combo.currentIndexChanged.connect(self._on_format_changed)
        right_layout.addWidget(self._format_combo)

        self._btn_save = QPushButton("保存当前 (Ctrl+S)")
        self._btn_save.clicked.connect(self._save_current)
        self._btn_save.setEnabled(False)
        right_layout.addWidget(self._btn_save)

        self._btn_save_all = QPushButton("全部导出")
        self._btn_save_all.clicked.connect(self._save_all)
        self._btn_save_all.setEnabled(False)
        right_layout.addWidget(self._btn_save_all)

        right_layout.addSpacing(16)

        # Layer control
        right_layout.addWidget(self._create_section_label("图层控制"))
        
        self._chk_vertebrae = QCheckBox("显示椎骨框 (绿色)")
        self._chk_vertebrae.setChecked(True)
        self._chk_vertebrae.stateChanged.connect(self._on_layer_changed)
        right_layout.addWidget(self._chk_vertebrae)
        
        self._chk_spine = QCheckBox("显示脊柱框 (红色/蓝色)")
        self._chk_spine.setChecked(True)
        self._chk_spine.stateChanged.connect(self._on_layer_changed)
        right_layout.addWidget(self._chk_spine)

        right_layout.addSpacing(12)

        # Annotation info
        right_layout.addWidget(self._create_section_label("当前标注"))
        self._ann_info_label = QLabel("无选中")
        self._ann_info_label.setWordWrap(True)
        right_layout.addWidget(self._ann_info_label)

        # Rotation controls
        right_layout.addWidget(self._create_section_label("旋转微调"))
        rotate_row = QHBoxLayout()
        self._btn_rotate_ccw = QPushButton("← 逆时针")
        self._btn_rotate_ccw.clicked.connect(lambda: self._canvas.rotate_selected(-5))
        self._btn_rotate_cw = QPushButton("顺时针 →")
        self._btn_rotate_cw.clicked.connect(lambda: self._canvas.rotate_selected(5))
        rotate_row.addWidget(self._btn_rotate_ccw)
        rotate_row.addWidget(self._btn_rotate_cw)
        right_layout.addLayout(rotate_row)

        # Fine rotation
        fine_row = QHBoxLayout()
        self._btn_rotate_ccw_fine = QPushButton("-1°")
        self._btn_rotate_ccw_fine.clicked.connect(lambda: self._canvas.rotate_selected(-1))
        self._btn_rotate_cw_fine = QPushButton("+1°")
        self._btn_rotate_cw_fine.clicked.connect(lambda: self._canvas.rotate_selected(1))
        fine_row.addWidget(self._btn_rotate_ccw_fine)
        fine_row.addWidget(self._btn_rotate_cw_fine)
        right_layout.addLayout(fine_row)

        # Custom angle
        angle_row = QHBoxLayout()
        self._angle_spin = QDoubleSpinBox()
        self._angle_spin.setRange(-180, 180)
        self._angle_spin.setDecimals(1)
        self._angle_spin.setSingleStep(1.0)
        self._angle_spin.setSuffix("°")
        self._angle_spin.valueChanged.connect(self._on_angle_spin_changed)
        angle_row.addWidget(self._angle_spin)

        self._btn_set_angle = QPushButton("应用")
        self._btn_set_angle.clicked.connect(self._apply_angle)
        angle_row.addWidget(self._btn_set_angle)
        right_layout.addLayout(angle_row)

        right_layout.addSpacing(12)

        # End vertebra markers
        right_layout.addWidget(self._create_section_label("端椎标记"))
        
        # 上端椎
        upper_layout = QHBoxLayout()
        self._chk_upper_end = QCheckBox("上端椎")
        self._chk_upper_end.stateChanged.connect(self._on_end_vertebra_changed)
        upper_layout.addWidget(self._chk_upper_end)
        
        # Tooltip button
        btn_tip_upper = QPushButton("?")
        btn_tip_upper.setFixedSize(20, 20)
        btn_tip_upper.setStyleSheet("font-weight: bold; color: #666;")
        btn_tip_upper.setToolTip("上端椎：脊柱侧弯弯曲开始处的第一个椎骨，其上终板用于测量Cobb角")
        upper_layout.addWidget(btn_tip_upper)
        right_layout.addLayout(upper_layout)
        
        # 下端椎
        lower_layout = QHBoxLayout()
        self._chk_lower_end = QCheckBox("下端椎")
        self._chk_lower_end.stateChanged.connect(self._on_end_vertebra_changed)
        lower_layout.addWidget(self._chk_lower_end)
        
        btn_tip_lower = QPushButton("?")
        btn_tip_lower.setFixedSize(20, 20)
        btn_tip_lower.setStyleSheet("font-weight: bold; color: #666;")
        btn_tip_lower.setToolTip("下端椎：脊柱侧弯弯曲结束处的最后一个椎骨，其下终板用于测量Cobb角")
        lower_layout.addWidget(btn_tip_lower)
        right_layout.addLayout(lower_layout)

        # Navigation
        right_layout.addSpacing(16)
        right_layout.addWidget(self._create_section_label("导航"))
        nav_row = QHBoxLayout()
        self._btn_prev = QPushButton("← 上一张")
        self._btn_prev.clicked.connect(self._prev_image)
        self._btn_next = QPushButton("下一张 →")
        self._btn_next.clicked.connect(self._next_image)
        nav_row.addWidget(self._btn_prev)
        nav_row.addWidget(self._btn_next)
        right_layout.addLayout(nav_row)

        # Shortcut reference
        right_layout.addSpacing(12)
        right_layout.addWidget(self._create_section_label("快捷键"))
        shortcut_help = QLabel(
            "旋转: R/E ±5° | T/Y ±1°\n"
            "精旋: Shift+R/E ±0.5°\n"
            "移动: W/A/S/D 5px\n"
            "精移: Shift+W/A/S/D 1px\n"
            "导航: ←/→ 图片 | ↑/↓ 标注\n"
            "其他: F 适配 | Esc 取消\n"
            "      Ctrl+S 保存"
        )
        shortcut_help.setStyleSheet("color: #666; font-size: 11px; line-height: 1.4;")
        shortcut_help.setWordWrap(True)
        right_layout.addWidget(shortcut_help)

        # Color legend
        right_layout.addSpacing(8)
        color_legend = QLabel(
            "■ 绿色: 椎骨 | ■ 青色: 上端椎\n"
            "■ 紫色: 下端椎 | ■ 红/蓝: 脊柱"
        )
        color_legend.setStyleSheet("color: #666; font-size: 11px;")
        color_legend.setWordWrap(True)
        right_layout.addWidget(color_legend)

        right_layout.addStretch()
        right_panel.setFixedWidth(240)

        # --- Assemble ---
        main_layout.addWidget(left_panel)
        main_layout.addWidget(self._canvas, stretch=1)
        main_layout.addWidget(right_panel)

    def _create_section_label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setStyleSheet(
            "font-weight: bold; font-size: 13px; "
            "padding: 4px 0; color: #333;"
        )
        return label

    def _init_shortcuts(self):
        """Initialize keyboard shortcuts."""
        shortcuts = {
            # Navigation
            QKeySequence("Left"): self._prev_image,
            QKeySequence("Right"): self._next_image,
            QKeySequence("Up"): self._select_prev_annotation,
            QKeySequence("Down"): self._select_next_annotation,
            # Rotation (coarse ±5°)
            QKeySequence("R"): lambda: self._canvas.rotate_selected(-5),
            QKeySequence("E"): lambda: self._canvas.rotate_selected(5),
            # Rotation (fine ±1°)
            QKeySequence("T"): lambda: self._canvas.rotate_selected(-1),
            QKeySequence("Y"): lambda: self._canvas.rotate_selected(1),
            # Rotation (super fine ±0.5°)
            QKeySequence("Shift+R"): lambda: self._canvas.rotate_selected(-0.5),
            QKeySequence("Shift+E"): lambda: self._canvas.rotate_selected(0.5),
            # Move (coarse 5px)
            QKeySequence("W"): lambda: self._canvas.move_selected(0, -5),
            QKeySequence("S"): lambda: self._canvas.move_selected(0, 5),
            QKeySequence("A"): lambda: self._canvas.move_selected(-5, 0),
            QKeySequence("D"): lambda: self._canvas.move_selected(5, 0),
            # Move (fine 1px)
            QKeySequence("Shift+W"): lambda: self._canvas.move_selected(0, -1),
            QKeySequence("Shift+S"): lambda: self._canvas.move_selected(0, 1),
            QKeySequence("Shift+A"): lambda: self._canvas.move_selected(-1, 0),
            QKeySequence("Shift+D"): lambda: self._canvas.move_selected(1, 0),
            # Save / undo / fit
            QKeySequence("Ctrl+S"): self._save_current,
            QKeySequence("Ctrl+Z"): self._undo,
            QKeySequence("Escape"): lambda: self._canvas.select_annotation(-1),
            QKeySequence("F"): self._fit_view,
        }
        for key, callback in shortcuts.items():
            shortcut = QShortcut(key, self)
            shortcut.activated.connect(callback)

    def _init_statusbar(self):
        self._status_label = QLabel("")
        self.statusBar().addPermanentWidget(self._status_label)

    # --- Dataset Operations ---

    def _open_dataset(self):
        """Open a YOLOv5/v8 dataset directory (scan only, no pixel loading)."""
        dir_path = QFileDialog.getExistingDirectory(
            self, "选择 YOLO 数据集目录",
            str(Path.home()),
        )
        if not dir_path:
            return

        self._dataset_root = dir_path
        self._progress_cache_path = os.path.join(dir_path, ".annotate_progress.json")
        self._dataset_path_label.setText(dir_path)
        self.statusBar().showMessage("正在扫描数据集...")

        # Scan images (fast, no pixel loading)
        self._image_infos = self._converter.scan_dataset(dir_path)

        # Load progress cache
        self._cache = self._converter.load_progress_cache(self._progress_cache_path)

        # Populate list
        self._image_list_widget.clear()
        for info in self._image_infos:
            name = Path(info["image_path"]).stem
            item = QListWidgetItem(name)
            # Mark if already processed
            if self._cache.get(info["image_path"], {}).get("saved"):
                item.setForeground(Qt.gray)
            self._image_list_widget.addItem(item)

        self._progress_bar.setMaximum(len(self._image_infos))
        self._update_progress()

        self.statusBar().showMessage(f"已扫描 {len(self._image_infos)} 张图片")
        self._btn_save_all.setEnabled(True)

        if self._image_infos:
            self._go_to_image(0)

    def _set_output_dir(self):
        """Set the output directory for exported annotations."""
        dir_path = QFileDialog.getExistingDirectory(
            self, "选择输出目录", str(Path.home()),
        )
        if dir_path:
            self._output_dir = dir_path
            self._output_path_label.setText(dir_path)
            self._btn_save.setEnabled(True)

    def _on_format_changed(self, index: int):
        self._export_format = "yolov8_obb" if index == 0 else "yolov8_xywhr"

    # --- Navigation ---

    def _go_to_image(self, index: int):
        """Navigate to a specific image (lazy load on demand)."""
        if not (0 <= index < len(self._image_infos)):
            return

        # Auto-save previous if modified
        if self._current_index >= 0 and self._current_annotation and self._current_annotation.modified:
            self._save_current(silent=True)

        self._current_index = index
        info = self._image_infos[index]

        # Lazy load annotations for this image
        cache_entry = self._cache.get(info["image_path"])
        self._current_annotation = self._converter.load_single(
            info["image_path"], info["label_path"], info["width"], info["height"],
            cache_entry=cache_entry,
        )

        # Apply layer visibility
        self._apply_layer_visibility()

        self._image_list_widget.setCurrentRow(index)
        self._canvas.load_image(info["image_path"], self._current_annotation.annotations)
        self._update_status()

    def _prev_image(self):
        self._go_to_image(self._current_index - 1)

    def _next_image(self):
        self._go_to_image(self._current_index + 1)

    def _on_image_selected(self, row: int):
        if row >= 0 and row != self._current_index:
            self._go_to_image(row)

    def _select_prev_annotation(self):
        if not self._image_infos:
            return
        if self._canvas._obb_items:
            idx = self._canvas._current_selection - 1
            if idx < 0:
                idx = len(self._canvas._obb_items) - 1
            self._canvas.select_annotation(idx)

    def _select_next_annotation(self):
        if not self._image_infos:
            return
        if self._canvas._obb_items:
            idx = self._canvas._current_selection + 1
            if idx >= len(self._canvas._obb_items):
                idx = 0
            self._canvas.select_annotation(idx)

    def _fit_view(self):
        """Fit the image to the canvas view."""
        if self._canvas._scene.sceneRect():
            self._canvas.fitInView(
                self._canvas._scene.sceneRect(), Qt.KeepAspectRatio
            )

    # --- Annotation Operations ---

    def _on_annotation_selected(self, index: int):
        """Update info panel when selection changes."""
        if index < 0 or not self._current_annotation:
            self._ann_info_label.setText("无选中")
            # Reset end vertebra checkboxes
            self._chk_upper_end.blockSignals(True)
            self._chk_upper_end.setChecked(False)
            self._chk_upper_end.blockSignals(False)
            self._chk_lower_end.blockSignals(True)
            self._chk_lower_end.setChecked(False)
            self._chk_lower_end.blockSignals(False)
            return

        # Map scene index to original annotation index
        if 0 <= index < len(self._canvas._index_map):
            orig_idx = self._canvas._index_map[index]
        else:
            orig_idx = index

        if 0 <= orig_idx < len(self._current_annotation.annotations):
            ann = self._current_annotation.annotations[orig_idx]
            angle_deg = math.degrees(ann.angle)
            self._ann_info_label.setText(
                f"类别: {ann.class_name} (ID={ann.class_id})\n"
                f"角度: {angle_deg:.1f}°\n"
                f"尺寸: {ann.width:.0f} x {ann.height:.0f}\n"
                f"中心: ({ann.center.x:.0f}, {ann.center.y:.0f})"
            )
            # Update angle spinbox without triggering signal
            self._angle_spin.blockSignals(True)
            self._angle_spin.setValue(angle_deg)
            self._angle_spin.blockSignals(False)

            # Update end vertebra checkboxes
            self._chk_upper_end.blockSignals(True)
            self._chk_upper_end.setChecked(ann.is_upper_end)
            self._chk_upper_end.blockSignals(False)
            self._chk_lower_end.blockSignals(True)
            self._chk_lower_end.setChecked(ann.is_lower_end)
            self._chk_lower_end.blockSignals(False)

    def _on_annotation_modified(self):
        """Mark current image as modified."""
        if self._current_annotation:
            self._current_annotation.modified = True
            self._on_annotation_selected(self._canvas._current_selection)
            self._update_status()

    def _on_layer_changed(self):
        """Handle layer visibility checkbox changes."""
        self._show_vertebrae = self._chk_vertebrae.isChecked()
        self._show_spine = self._chk_spine.isChecked()
        self._apply_layer_visibility()
        self._canvas.viewport().update()

    def _apply_layer_visibility(self):
        """Apply layer visibility to all annotations."""
        if not self._current_annotation:
            return
        for ann in self._current_annotation.annotations:
            if ann.class_id == 0:  # Vertebra
                ann.visible = self._show_vertebrae
            else:  # Spine boxes
                ann.visible = self._show_spine
        # Update canvas items
        for item in self._canvas._obb_items:
            item.setVisible(item.annotation.visible)

    def _on_end_vertebra_changed(self):
        """Handle end vertebra checkbox changes."""
        ann = self._canvas.get_selected_annotation()
        if ann is None:
            return
        ann.is_upper_end = self._chk_upper_end.isChecked()
        ann.is_lower_end = self._chk_lower_end.isChecked()
        if self._current_annotation:
            self._current_annotation.modified = True
        # Refresh canvas to show/hide markers
        if 0 <= self._canvas._current_selection < len(self._canvas._obb_items):
            self._canvas._obb_items[self._canvas._current_selection].update()
        self._canvas.viewport().update()

    def _on_angle_spin_changed(self, value: float):
        """Called when angle spinbox value changes (but not applied yet)."""
        pass

    def _apply_angle(self):
        """Apply the angle from spinbox to selected annotation."""
        ann = self._canvas.get_selected_annotation()
        if ann is None:
            return

        target_angle = math.radians(self._angle_spin.value())
        delta = target_angle - ann.angle
        self._canvas.rotate_selected(math.degrees(delta))

    def _undo(self):
        """Simple undo: reload original annotations for current image."""
        # For now, just reload from disk
        # TODO: implement proper undo stack
        pass

    # --- Save Operations ---

    def _save_current(self, silent: bool = False):
        """Save current image annotations."""
        if not self._image_infos or not self._output_dir:
            if not silent:
                QMessageBox.warning(self, "提示", "请先设置输出目录")
            return

        if not self._current_annotation:
            return

        info = self._image_infos[self._current_index]

        if self._export_format == "yolov8_obb":
            self._converter.save_obb_yolov8(self._current_annotation, self._output_dir, overwrite=True)
        else:
            self._converter.save_obb_xywhr(self._current_annotation, self._output_dir, overwrite=True)

        # Update cache with end vertebra state
        img_path = info["image_path"]
        ann_states = []
        for ann in self._current_annotation.annotations:
            ann_states.append({
                "is_upper_end": ann.is_upper_end,
                "is_lower_end": ann.is_lower_end,
            })
        self._cache[img_path] = {
            "modified": False,
            "saved": True,
            "annotation_states": ann_states,
        }
        self._current_annotation.modified = False

        # Save cache to disk
        self._converter.save_progress_cache(self._progress_cache_path, self._cache)

        # Update UI
        self._image_list_widget.item(self._current_index).setForeground(Qt.gray)
        self._update_progress()

        if not silent:
            self.statusBar().showMessage(f"已保存: {Path(img_path).name}")

    def _save_all(self):
        """Export all images that have been processed."""
        if not self._output_dir:
            QMessageBox.warning(self, "提示", "请先设置输出目录")
            return

        count = 0
        for i, info in enumerate(self._image_infos):
            # Only save if in cache or currently loaded
            img_path = info["image_path"]
            if img_path in self._cache or i == self._current_index:
                # Load if not current
                if i != self._current_index:
                    ann = self._converter.load_single(
                        info["image_path"], info["label_path"],
                        info["width"], info["height"]
                    )
                else:
                    ann = self._current_annotation

                if self._export_format == "yolov8_obb":
                    self._converter.save_obb_yolov8(ann, self._output_dir, overwrite=True)
                else:
                    self._converter.save_obb_xywhr(ann, self._output_dir, overwrite=True)

                self._cache[img_path] = {"modified": False, "saved": True}
                count += 1

        # Save cache
        self._converter.save_progress_cache(self._progress_cache_path, self._cache)
        self._update_progress()

        self.statusBar().showMessage(f"已导出 {count} 个标注文件")

    def _update_progress(self):
        """Update progress bar based on saved count."""
        saved = sum(1 for v in self._cache.values() if v.get("saved"))
        total = len(self._image_infos)
        self._progress_bar.setMaximum(total)
        self._progress_bar.setValue(saved)

    def _update_status(self):
        """Update status bar info."""
        if not self._image_infos or not self._current_annotation:
            return
        info = self._image_infos[self._current_index]
        modified_str = " [已修改]" if self._current_annotation.modified else ""
        self._status_label.setText(
            f"{self._current_index + 1}/{len(self._image_infos)} | "
            f"{Path(info['image_path']).name} | "
            f"{len(self._current_annotation.annotations)} 个标注{modified_str}"
        )
