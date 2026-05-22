"""Main application window for the spine annotation tool."""

import math
import os
from pathlib import Path
from typing import Dict, List, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
    QAction, QComboBox, QDoubleSpinBox, QFileDialog, QHBoxLayout,
    QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow,
    QMessageBox, QProgressBar, QPushButton, QShortcut, QSplitter,
    QStatusBar, QToolBar, QVBoxLayout, QWidget,
)

from ..core.converter import YOLOConverter
from ..core.models import ImageAnnotation, OBBAnnotation
from .image_canvas import AnnotationCanvas


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("脊柱椎骨标注工具 - Spine Annotator")
        self.resize(1400, 900)

        # Data
        self._converter = YOLOConverter()
        self._dataset: Dict[str, ImageAnnotation] = {}
        self._image_list: List[str] = []
        self._current_index: int = -1
        self._output_dir: Optional[str] = None
        self._export_format: str = "yolov8_obb"  # or 'yolov8_xywhr'

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
            QKeySequence("Left"): self._prev_image,
            QKeySequence("Right"): self._next_image,
            QKeySequence("Up"): self._select_prev_annotation,
            QKeySequence("Down"): self._select_next_annotation,
            QKeySequence("R"): lambda: self._canvas.rotate_selected(-5),
            QKeySequence("E"): lambda: self._canvas.rotate_selected(5),
            QKeySequence("T"): lambda: self._canvas.rotate_selected(-1),
            QKeySequence("Y"): lambda: self._canvas.rotate_selected(1),
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
        """Open a YOLOv5/v8 dataset directory."""
        dir_path = QFileDialog.getExistingDirectory(
            self, "选择 YOLO 数据集目录",
            str(Path.home()),
        )
        if not dir_path:
            return

        self._dataset_path_label.setText(dir_path)
        self.statusBar().showMessage("正在加载数据集...")

        self._dataset = self._converter.load_dataset(dir_path)
        self._image_list = sorted(self._dataset.keys())

        self._image_list_widget.clear()
        for path in self._image_list:
            name = Path(path).stem
            item = QListWidgetItem(name)
            self._image_list_widget.addItem(item)

        self._progress_bar.setMaximum(len(self._image_list))
        self._progress_bar.setValue(0)

        self.statusBar().showMessage(f"已加载 {len(self._image_list)} 张图片")
        self._btn_save_all.setEnabled(True)

        if self._image_list:
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
        """Navigate to a specific image."""
        if not (0 <= index < len(self._image_list)):
            return

        self._current_index = index
        img_path = self._image_list[index]
        annotation = self._dataset[img_path]

        self._image_list_widget.setCurrentRow(index)
        self._canvas.load_image(img_path, annotation.annotations)
        self._update_status()

    def _prev_image(self):
        self._go_to_image(self._current_index - 1)

    def _next_image(self):
        self._go_to_image(self._current_index + 1)

    def _on_image_selected(self, row: int):
        if row >= 0 and row != self._current_index:
            self._go_to_image(row)

    def _select_prev_annotation(self):
        if not self._image_list:
            return
        ann = self._dataset.get(self._image_list[self._current_index])
        if ann and ann.annotations:
            idx = self._canvas._current_selection - 1
            if idx < 0:
                idx = len(ann.annotations) - 1
            self._canvas.select_annotation(idx)

    def _select_next_annotation(self):
        if not self._image_list:
            return
        ann = self._dataset.get(self._image_list[self._current_index])
        if ann and ann.annotations:
            idx = self._canvas._current_selection + 1
            if idx >= len(ann.annotations):
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
        if index < 0:
            self._ann_info_label.setText("无选中")
            return

        if not self._image_list:
            return

        img_path = self._image_list[self._current_index]
        annotation = self._dataset[img_path]
        if 0 <= index < len(annotation.annotations):
            ann = annotation.annotations[index]
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

    def _on_annotation_modified(self):
        """Mark current image as modified."""
        if not self._image_list:
            return
        img_path = self._image_list[self._current_index]
        self._dataset[img_path].modified = True
        self._on_annotation_selected(self._canvas._current_selection)
        self._update_status()

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

    def _save_current(self):
        """Save current image annotations."""
        if not self._image_list or not self._output_dir:
            QMessageBox.warning(self, "提示", "请先设置输出目录")
            return

        img_path = self._image_list[self._current_index]
        annotation = self._dataset[img_path]

        if self._export_format == "yolov8_obb":
            self._converter.save_obb_yolov8(annotation, self._output_dir, overwrite=True)
        else:
            self._converter.save_obb_xywhr(annotation, self._output_dir, overwrite=True)

        annotation.modified = False
        self.statusBar().showMessage(f"已保存: {Path(img_path).name}")
        self._update_progress()

    def _save_all(self):
        """Export all annotations."""
        if not self._output_dir:
            QMessageBox.warning(self, "提示", "请先设置输出目录")
            return

        count = 0
        for img_path, annotation in self._dataset.items():
            if self._export_format == "yolov8_obb":
                saved = self._converter.save_obb_yolov8(
                    annotation, self._output_dir, overwrite=True
                )
            else:
                saved = self._converter.save_obb_xywhr(
                    annotation, self._output_dir, overwrite=True
                )
            if saved:
                count += 1

        self.statusBar().showMessage(f"已导出 {count} 个标注文件")
        self._update_progress()

    def _update_progress(self):
        """Update progress bar based on modified count."""
        modified = sum(1 for a in self._dataset.values() if a.modified)
        total = len(self._dataset)
        self._progress_bar.setMaximum(total)
        self._progress_bar.setValue(total - modified)

    def _update_status(self):
        """Update status bar info."""
        if not self._image_list:
            return
        img_path = self._image_list[self._current_index]
        ann = self._dataset[img_path]
        modified_str = " [已修改]" if ann.modified else ""
        self._status_label.setText(
            f"{self._current_index + 1}/{len(self._image_list)} | "
            f"{Path(img_path).name} | "
            f"{len(ann.annotations)} 个标注{modified_str}"
        )
