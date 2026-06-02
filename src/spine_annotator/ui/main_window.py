"""Main application window for the spine annotation tool."""

import logging
import math
import os
from pathlib import Path
from typing import List, Optional

from PyQt5.QtCore import Qt, QSettings, QTimer
from PyQt5.QtGui import QBrush, QColor, QKeySequence, QPalette
from PyQt5.QtWidgets import (
    QAction, QButtonGroup, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
    QDoubleSpinBox, QFileDialog, QHBoxLayout, QInputDialog, QLabel, QListWidget,
    QListWidgetItem, QMainWindow, QMenu, QMessageBox, QProgressBar, QPushButton,
    QRadioButton, QScrollArea, QShortcut, QSpinBox, QTextBrowser, QVBoxLayout,
    QWidget,
)

from .. import __version__
from ..core.converter import YOLOConverter
from ..core.image_enhancer import EnhanceParams
from ..core.models import (
    ImageAnnotation, OBBAnnotation, VERTEBRA_CLASSES,
    VertebraCategory, auto_sort_annotations,
    get_vertebra_category, get_vertebra_class_id,
)
from .image_canvas import AnnotationCanvas, CATEGORY_COLORS
from .enhancement_panel import EnhancementDialog, EnhancementToolbar
from .inference_worker import InferenceWorker
from ..core.inference import ModelManager

LOGGER = logging.getLogger("spine_annotator.inference")


class MainWindow(QMainWindow):
    """Main application window."""

    # QSettings 的 key 常量
    SETTINGS_LAST_DATASET = "last_dataset_dir"
    SETTINGS_LAST_OUTPUT = "last_output_dir"
    SETTINGS_LAST_FORMAT = "last_export_format"  # 字符串：yolov8_obb / yolov8_xywhr / yolov8_pose
    SETTINGS_SAVE_MIN_COUNT_ENABLED = "save_min_count_enabled"
    SETTINGS_SAVE_MIN_COUNT_VALUE = "save_min_count_value"
    SETTINGS_SAVE_MAX_COUNT_ENABLED = "save_max_count_enabled"
    SETTINGS_SAVE_MAX_COUNT_VALUE = "save_max_count_value"

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"脊柱椎骨标注工具 - Spine Annotator v{__version__}")
        self.resize(1400, 900)

        # 持久化设置（macOS 写到 ~/Library/Preferences、Linux 写 INI、Windows 写注册表）
        self._settings = QSettings("spine-annotator", "spine-annotator")

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

        # 标注数量全局检测结果（存放检测不通过的图片 index）
        self._count_check_failed: set = set()
        
        # Layer visibility
        self._show_cervical = True
        self._show_thoracic = True
        self._show_lumbar = True
        self._show_sacral = True

        # 绘制模式状态
        self._current_draw_shape: str = "none"  # "none", "rect", "line"
        self._current_draw_class_id: Optional[int] = None

        # AI 推理
        self._model_manager = ModelManager()
        self._inference_worker: Optional[InferenceWorker] = None

        # 全局通用设置
        self._save_min_count_enabled = bool(
            self._settings.value(self.SETTINGS_SAVE_MIN_COUNT_ENABLED, True, type=bool)
        )
        self._save_min_count_value = int(
            self._settings.value(self.SETTINGS_SAVE_MIN_COUNT_VALUE, 19, type=int)
        )
        self._save_max_count_enabled = bool(
            self._settings.value(self.SETTINGS_SAVE_MAX_COUNT_ENABLED, True, type=bool)
        )
        self._save_max_count_value = int(
            self._settings.value(self.SETTINGS_SAVE_MAX_COUNT_VALUE, 19, type=int)
        )

        self._init_ui()
        self._init_menubar()
        self._init_shortcuts()
        self._init_statusbar()

        # 启动后异步恢复上次会话（让窗口先显示出来，避免大数据集扫描时白屏）
        QTimer.singleShot(0, self._restore_last_session)

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
        self._image_list_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self._image_list_widget.customContextMenuRequested.connect(self._on_image_list_context_menu)
        left_layout.addWidget(self._image_list_widget)

        # 标记难点按钮
        flag_row = QHBoxLayout()
        self._btn_flag = QPushButton("⚑ 标记难点")
        self._btn_flag.setCheckable(True)
        self._btn_flag.setToolTip("标记当前图片为标注难点，稍后继续")
        self._btn_flag.toggled.connect(self._on_flag_toggled)
        flag_row.addWidget(self._btn_flag)
        left_layout.addLayout(flag_row)

        # Progress
        self._progress_bar = QProgressBar()
        self._progress_bar.setFormat("已标注: %v / %m")
        left_layout.addWidget(self._progress_bar)

        left_panel.setFixedWidth(220)

        # --- Center: canvas ---
        self._canvas = AnnotationCanvas()
        self._canvas.selection_changed.connect(self._on_annotation_selected)
        self._canvas.annotation_modified.connect(self._on_annotation_modified)
        self._canvas.annotation_created.connect(self._on_annotation_created)
        self._canvas.annotation_relabel_requested.connect(self._on_relabel_requested)

        # 画布顶部图像增强工具条（不影响原图与标注坐标）
        self._enhance_toolbar = EnhancementToolbar()
        self._enhance_toolbar.open_dialog_requested.connect(self._open_enhance_dialog)
        self._enhance_toolbar.invert_toggled.connect(self._on_enhance_invert_toggled)
        self._enhance_toolbar.reset_requested.connect(self._reset_enhance_params)
        self._enhance_dialog: Optional[EnhancementDialog] = None

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
        self._dataset_path_label.setStyleSheet(self._muted_text_style(11))
        right_layout.addWidget(self._dataset_path_label)

        # Export controls
        right_layout.addWidget(self._create_section_label("导出"))
        self._btn_set_output = QPushButton("设置输出目录")
        self._btn_set_output.clicked.connect(self._set_output_dir)
        right_layout.addWidget(self._btn_set_output)

        self._output_path_label = QLabel("未设置")
        self._output_path_label.setWordWrap(True)
        self._output_path_label.setStyleSheet(self._muted_text_style(11))
        right_layout.addWidget(self._output_path_label)

        self._format_combo = QComboBox()
        self._format_combo.addItems([
            "YOLOv8-OBB (四角点)",
            "YOLOv8-OBB (xywhr)",
            "YOLOv8-pose (bbox + 4 关键点)",
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

        # Drawing tools
        right_layout.addWidget(self._create_section_label("绘制工具"))

        draw_row = QHBoxLayout()
        self._btn_draw_rect = QPushButton("矩形 (B)")
        self._btn_draw_rect.setCheckable(True)
        self._btn_draw_rect.setToolTip("拖拽绘制矩形椎骨标注；绘制完成后弹窗选择椎骨编号\n快捷键：B（再次按 B 或 Esc 退出绘制）")
        self._btn_draw_rect.clicked.connect(lambda: self._toggle_draw_mode("rect"))
        draw_row.addWidget(self._btn_draw_rect)

        right_layout.addLayout(draw_row)

        draw_hint = QLabel("提示: 按 Esc 退出绘制；绘制完成后会弹出椎骨编号选择框")
        draw_hint.setStyleSheet("color: #888; font-size: 11px;")
        draw_hint.setWordWrap(True)
        right_layout.addWidget(draw_hint)

        # Delete annotation button
        self._btn_delete_ann = QPushButton("删除选中标注 (Del)")
        self._btn_delete_ann.clicked.connect(self._delete_selected_annotation)
        self._btn_delete_ann.setEnabled(False)
        right_layout.addWidget(self._btn_delete_ann)

        # Relabel annotation button
        self._btn_relabel_ann = QPushButton("更改编号 (双击)")
        self._btn_relabel_ann.setToolTip("双击标注或点此按钮修改椎骨编号")
        self._btn_relabel_ann.clicked.connect(self._relabel_selected_annotation)
        self._btn_relabel_ann.setEnabled(False)
        right_layout.addWidget(self._btn_relabel_ann)

        # Auto-sort / auto-renumber button
        self._btn_auto_sort = QPushButton("自动排序")
        self._btn_auto_sort.setToolTip(
            "按 Y 坐标从上到下自动排序并重编号 (C7→T1→...→L5→S1)\n"
            "快捷键：Ctrl+Shift+S"
        )
        self._btn_auto_sort.clicked.connect(self._auto_sort_annotations)
        right_layout.addWidget(self._btn_auto_sort)

        # Clear current image annotations (高危操作，红色醒目 + 二次确认)
        self._btn_clear_current = QPushButton("⚠️ 清空当前图片")
        self._btn_clear_current.setToolTip(
            "删除当前图片的所有标注、对应缓存条目与已导出的 .txt 文件"
            "\n快捷键：Cmd+Shift+Backspace"
        )
        self._btn_clear_current.setStyleSheet(
            "QPushButton { background-color: #d9534f; color: white; padding: 5px; }"
            "QPushButton:disabled { background-color: #f0a8a5; color: #f3f3f3; }"
        )
        self._btn_clear_current.clicked.connect(self._clear_current_image)
        self._btn_clear_current.setEnabled(False)
        right_layout.addWidget(self._btn_clear_current)

        right_layout.addSpacing(12)

        # Layer control
        right_layout.addWidget(self._create_section_label("图层控制"))

        self._chk_cervical = QCheckBox("颈椎 C (青色)")
        self._chk_cervical.setChecked(True)
        self._chk_cervical.stateChanged.connect(self._on_layer_changed)
        right_layout.addWidget(self._chk_cervical)

        self._chk_thoracic = QCheckBox("胸椎 T (绿色)")
        self._chk_thoracic.setChecked(True)
        self._chk_thoracic.stateChanged.connect(self._on_layer_changed)
        right_layout.addWidget(self._chk_thoracic)

        self._chk_lumbar = QCheckBox("腰椎 L (橙色)")
        self._chk_lumbar.setChecked(True)
        self._chk_lumbar.stateChanged.connect(self._on_layer_changed)
        right_layout.addWidget(self._chk_lumbar)

        self._chk_sacral = QCheckBox("骶椎 S (紫色)")
        self._chk_sacral.setChecked(True)
        self._chk_sacral.stateChanged.connect(self._on_layer_changed)
        right_layout.addWidget(self._chk_sacral)

        right_layout.addSpacing(12)

        # Annotation info
        right_layout.addWidget(self._create_section_label("当前标注"))
        self._ann_info_label = QLabel("无选中")
        self._ann_info_label.setWordWrap(True)
        right_layout.addWidget(self._ann_info_label)

        # Keypoint visibility (YOLOv8-pose v 字段，对该标注 4 个角点统一生效)
        right_layout.addWidget(self._create_section_label("关键点可见性"))
        vis_row = QHBoxLayout()
        self._vis_group = QButtonGroup(self)
        self._rb_vis_2 = QRadioButton("可见 (2)")
        self._rb_vis_1 = QRadioButton("遮挡 (1)")
        self._rb_vis_0 = QRadioButton("不可见 (0)")
        self._rb_vis_2.setToolTip("肉眼清晰可见，是默认值")
        self._rb_vis_1.setToolTip("被金属植入物 / 伪影遮挡，坐标可信")
        self._rb_vis_0.setToolTip("肉眼无法看清（脊柱骨过于透明），坐标基于相邻椎骨推断")
        self._vis_group.addButton(self._rb_vis_2, 2)
        self._vis_group.addButton(self._rb_vis_1, 1)
        self._vis_group.addButton(self._rb_vis_0, 0)
        self._rb_vis_2.setChecked(True)
        vis_row.addWidget(self._rb_vis_2)
        vis_row.addWidget(self._rb_vis_1)
        vis_row.addWidget(self._rb_vis_0)
        right_layout.addLayout(vis_row)
        self._vis_group.buttonClicked.connect(self._on_visibility_changed)

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

        # 快捷键帮助已移至 “帮助 → 快捷键” 菜单（快捷键: F1）

        # Color legend
        right_layout.addSpacing(8)
        color_legend = QLabel(
            "■ 青色: 颈椎 C | ■ 绿色: 胸椎 T\n"
            "■ 橙色: 腰椎 L | ■ 紫色: 骶椎 S"
        )
        color_legend.setStyleSheet(self._muted_text_style(11))
        color_legend.setWordWrap(True)
        right_layout.addWidget(color_legend)

        right_layout.addStretch()

        # 包装到滚动区，避免低分辨率 / 窗口高度不足时内容被裁切
        right_scroll = QScrollArea()
        right_scroll.setWidget(right_panel)
        right_scroll.setWidgetResizable(True)
        right_scroll.setFrameShape(QScrollArea.NoFrame)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        right_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        # 选 248px：240 内容区 + 约 8px 滚动条，避免滚动条出现时横向振荡
        right_panel.setMinimumWidth(240)
        right_scroll.setFixedWidth(248)

        # --- Assemble ---
        # 中央区域 = 增强工具条 + 画布（垂直堆叠）
        center_widget = QWidget()
        center_layout = QVBoxLayout(center_widget)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(0)
        center_layout.addWidget(self._enhance_toolbar)
        center_layout.addWidget(self._canvas, stretch=1)

        main_layout.addWidget(left_panel)
        main_layout.addWidget(center_widget, stretch=1)
        main_layout.addWidget(right_scroll)

    def _create_section_label(self, text: str) -> QLabel:
        label = QLabel(text)
        # 不设 color，让 Qt 自动用主题默认文字色（深 / 浅模式都能看清）
        # 仅用加粗 + 字号体现层次
        label.setStyleSheet("font-weight: bold; font-size: 13px; padding: 4px 0;")
        return label

    def _muted_text_style(self, font_size: int = 11) -> str:
        """返回适合当前主题（深 / 浅）的"次要信息"文本样式。

        深色模式：浅灰文字（#aaaaaa），在深色背景上清晰可读
        浅色模式：深灰文字（#666666），传统次要信息观感
        """
        is_dark = self.palette().color(QPalette.Window).lightness() < 128
        color = "#aaaaaa" if is_dark else "#666666"
        return f"color: {color}; font-size: {font_size}px;"

    def _init_menubar(self):
        """初始化菜单栏（工具 / 帮助）。"""
        menubar = self.menuBar()

        # 工具菜单
        tools_menu = menubar.addMenu("工具(&T)")

        act_inference = QAction("AI 推理预标注", self)
        act_inference.setShortcut(QKeySequence("Ctrl+I"))
        act_inference.setStatusTip("对当前图片执行 AI 推理，清空画布并填充预标注结果")
        act_inference.triggered.connect(self._run_inference)
        tools_menu.addAction(act_inference)

        tools_menu.addSeparator()

        act_general_settings = QAction("通用设置…", self)
        act_general_settings.setStatusTip("打开全局通用设置")
        act_general_settings.triggered.connect(self._open_general_settings)
        tools_menu.addAction(act_general_settings)

        tools_menu.addSeparator()

        act_check_count = QAction("检测标注数量…", self)
        act_check_count.setStatusTip("全局扫描所有图片，检测标注数量是否符合要求")
        act_check_count.triggered.connect(self._check_annotation_counts)
        tools_menu.addAction(act_check_count)

        act_clear_all = QAction("清空所有标注…", self)
        act_clear_all.setStatusTip("删除进度缓存与当前 split 的训练标注文件（不可恢复）")
        act_clear_all.triggered.connect(self._clear_all_data)
        tools_menu.addAction(act_clear_all)

        help_menu = menubar.addMenu("帮助(&H)")

        act_shortcuts = QAction("快捷键(&K)", self)
        act_shortcuts.setShortcut(QKeySequence("F1"))
        act_shortcuts.setStatusTip("查看所有快捷键")
        act_shortcuts.triggered.connect(self._show_shortcut_help)
        help_menu.addAction(act_shortcuts)

        help_menu.addSeparator()

        act_about = QAction("关于(&A)", self)
        act_about.setStatusTip("关于本软件")
        act_about.triggered.connect(self._show_about)
        help_menu.addAction(act_about)

    def _show_shortcut_help(self):
        """弹出快捷键帮助对话框（独立窗口）。"""
        dlg = QDialog(self)
        dlg.setWindowTitle("快捷键帮助")
        dlg.resize(560, 560)

        layout = QVBoxLayout(dlg)

        title = QLabel("<h2 style='margin:0'>快捷键一览</h2>")
        layout.addWidget(title)

        browser = QTextBrowser()
        browser.setOpenExternalLinks(False)
        browser.setHtml(self._build_shortcut_html())
        layout.addWidget(browser, 1)

        btn_box = QDialogButtonBox(QDialogButtonBox.Close)
        btn_box.button(QDialogButtonBox.Close).clicked.connect(dlg.accept)
        layout.addWidget(btn_box)

        dlg.exec_()

    def _open_general_settings(self):
        """打开全局通用设置窗口。"""
        dlg = QDialog(self)
        dlg.setWindowTitle("通用设置")
        dlg.setModal(True)
        dlg.resize(460, 240)

        layout = QVBoxLayout(dlg)

        # --- 最少标注数量 ---
        chk_save_min_count = QCheckBox("保存时检查标注数量不少于")
        chk_save_min_count.setChecked(self._save_min_count_enabled)

        row_min = QHBoxLayout()
        row_min.addWidget(chk_save_min_count)
        spn_save_min_count = QSpinBox()
        spn_save_min_count.setRange(1, 999)
        spn_save_min_count.setValue(max(1, int(self._save_min_count_value)))
        row_min.addWidget(spn_save_min_count)
        row_min.addWidget(QLabel("个"))
        row_min.addStretch(1)
        layout.addLayout(row_min)

        # --- 最多标注数量 ---
        chk_save_max_count = QCheckBox("保存时检查标注数量不多于")
        chk_save_max_count.setChecked(self._save_max_count_enabled)

        row_max = QHBoxLayout()
        row_max.addWidget(chk_save_max_count)
        spn_save_max_count = QSpinBox()
        spn_save_max_count.setRange(1, 999)
        spn_save_max_count.setValue(max(1, int(self._save_max_count_value)))
        row_max.addWidget(spn_save_max_count)
        row_max.addWidget(QLabel("个"))
        row_max.addStretch(1)
        layout.addLayout(row_max)

        hint = QLabel(
            "默认均启用，默认值 19（C7~S1 全部椎骨数量）。\n"
            "关闭后，保存/导出时不再校验对应方向的数量限制。"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(self._muted_text_style(11))
        layout.addWidget(hint)

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(dlg.accept)
        btn_box.rejected.connect(dlg.reject)
        layout.addWidget(btn_box)

        def _sync_enabled():
            spn_save_min_count.setEnabled(chk_save_min_count.isChecked())
            spn_save_max_count.setEnabled(chk_save_max_count.isChecked())

        chk_save_min_count.toggled.connect(_sync_enabled)
        chk_save_max_count.toggled.connect(_sync_enabled)
        _sync_enabled()

        if dlg.exec_() != QDialog.Accepted:
            return

        self._save_min_count_enabled = bool(chk_save_min_count.isChecked())
        self._save_min_count_value = int(spn_save_min_count.value())
        self._save_max_count_enabled = bool(chk_save_max_count.isChecked())
        self._save_max_count_value = int(spn_save_max_count.value())
        self._settings.setValue(self.SETTINGS_SAVE_MIN_COUNT_ENABLED, self._save_min_count_enabled)
        self._settings.setValue(self.SETTINGS_SAVE_MIN_COUNT_VALUE, self._save_min_count_value)
        self._settings.setValue(self.SETTINGS_SAVE_MAX_COUNT_ENABLED, self._save_max_count_enabled)
        self._settings.setValue(self.SETTINGS_SAVE_MAX_COUNT_VALUE, self._save_max_count_value)
        self.statusBar().showMessage("通用设置已保存", 2500)

    def _build_shortcut_html(self) -> str:
        """构建快捷键帮助 HTML 内容。"""
        rows = [
            ("图片导航", [
                ("← / →",            "上一张 / 下一张图片"),
                ("↑ / ↓",            "选上 / 选下一个标注"),
                ("Ctrl+N",            "跳到下一张未标注图片【断点续标】"),
                ("Ctrl+B",            "跳到上一张未标注图片"),
            ]),
            ("标注旋转（需选中标注）", [
                ("R / E",             "逆/顺时针 ± 5°（粗调）"),
                ("T / Y",             "逆/顺时针 ± 1°（细调）"),
                ("Shift+R / Shift+E", "逆/顺时针 ± 0.5°（精调）"),
            ]),
            ("标注移动（需选中标注）", [
                ("W / A / S / D",       "上/左/下/右 移动 5px（粗调）"),
                ("Shift+W / A / S / D", "上/左/下/右 移动 1px（精调）"),
            ]),
            ("标注编辑", [
                ("B",                 "进入/退出矩形绘制模式"),
                ("双击标注",            "修改椎骨编号"),
                ("Ctrl+Shift+S",     "自动排序 / 重编号 (C7→T1→...→S1)"),
                ("Del",               "删除选中标注"),
                ("Esc",               "取消选中 / 退出绘制模式"),
                ("F",                 "适配画布 (Fit)"),
            ]),
            ("AI 推理", [
                ("Ctrl+I",            "对当前图片执行 AI 推理预标注"),
            ]),
            ("文件 / 状态", [
                ("Ctrl+S",            "保存当前图片标注"),
                ("Ctrl+Z",            "撤销"),
                ("M",                 "标记/取消标记难点"),
                ("F1",                "打开本帮助窗口"),
            ]),
        ]

        html_parts = [
            "<style>",
            "  body { font-family: -apple-system, 'Helvetica Neue', Arial, sans-serif; font-size: 13px; }",
            "  h3 { color: #2563eb; margin: 14px 0 6px 0; border-bottom: 1px solid #e5e7eb; padding-bottom: 4px; }",
            "  table { width: 100%; border-collapse: collapse; }",
            "  td { padding: 4px 8px; vertical-align: top; }",
            "  td.key { font-family: 'SF Mono', Menlo, Consolas, monospace; color: #b91c1c; white-space: nowrap; width: 35%; }",
            "  td.desc { color: #1f2937; }",
            "</style>",
        ]

        for section, items in rows:
            html_parts.append(f"<h3>{section}</h3>")
            html_parts.append("<table>")
            for key, desc in items:
                html_parts.append(
                    f"<tr><td class='key'>{key}</td><td class='desc'>{desc}</td></tr>"
                )
            html_parts.append("</table>")

        html_parts.append(
            "<p style='color:#6b7280; font-size:12px; margin-top:16px;'>"
            "提示：选中标注后才能使用旋转/移动/删除类快捷键。"
            "</p>"
        )

        return "".join(html_parts)

    def _show_about(self):
        """关于对话框。"""
        QMessageBox.about(
            self,
            "关于 Spine Annotator",
            (
                f"<h3>脊柱椎骨标注工具</h3>"
                f"<p><b>版本</b>：v{__version__}</p>"
                f"<p>面向脊柱 X 光片的 OBB 标注工具，支持 C7/T1–T12/L1–L5/S1 分类与多种 YOLO 输出格式。</p>"
                f"<p style='color:#6b7280;'>按 F1 查看快捷键</p>"
            ),
        )

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
            QKeySequence("Escape"): lambda: (self._canvas.select_annotation(-1), self._toggle_draw_mode("none")),
            QKeySequence("F"): self._fit_view,
            QKeySequence("Delete"): self._delete_selected_annotation,
            # 绘制模式切换：矩形（再按一次退出绘制）
            QKeySequence("B"): self._toggle_rect_shortcut,
            QKeySequence("Ctrl+Shift+S"): self._auto_sort_annotations,
            # 清空当前图片（高危，需二次确认）
            QKeySequence("Ctrl+Shift+Backspace"): self._clear_current_image,
            # 标记难点
            QKeySequence("M"): self._toggle_flag_current,
            # AI 推理预标注
            QKeySequence("Ctrl+I"): self._run_inference,
            # 跳到下一张 / 上一张未标注图片（断点续标核心快捷键）
            QKeySequence("Ctrl+N"): self._jump_to_next_unannotated,
            QKeySequence("Ctrl+B"): self._jump_to_prev_unannotated,
            # 帮助快捷键
            QKeySequence("F1"): self._show_shortcut_help,
        }
        for key, callback in shortcuts.items():
            shortcut = QShortcut(key, self)
            shortcut.activated.connect(callback)

    def _init_statusbar(self):
        # 永久显示在右侧的"进度统计"标签（已标注 N/Total · X%）
        self._progress_label = QLabel("")
        self._progress_label.setStyleSheet("color: #2c7be5; font-weight: bold;")
        self.statusBar().addPermanentWidget(self._progress_label)
        # 永久显示在最右侧的"当前图片"信息
        self._status_label = QLabel("")
        self.statusBar().addPermanentWidget(self._status_label)

    # --- Dataset Operations ---

    def _open_dataset(self):
        """用户主动选择 YOLOv5/v8 数据集目录。"""
        dir_path = QFileDialog.getExistingDirectory(
            self, "选择 YOLO 数据集目录",
            self._settings.value(self.SETTINGS_LAST_DATASET, str(Path.home())),
        )
        if not dir_path:
            return
        self._load_dataset(dir_path)

    def _load_dataset(self, dir_path: str, *, silent_invalid: bool = False) -> bool:
        """加载指定路径的 YOLO 数据集（共用：用户主动打开 & 启动自动恢复）。

        Args:
            silent_invalid: True 时，校验失败不弹窗（用于启动恢复，避免打扰）
        Returns:
            是否加载成功
        """
        is_valid, message = self._converter.validate_dataset(dir_path)
        if not is_valid:
            if not silent_invalid:
                QMessageBox.warning(self, "数据集格式错误", message)
            return False

        self._dataset_root = dir_path
        self._progress_cache_path = os.path.join(dir_path, ".annotate_progress.json")
        self._dataset_path_label.setText(f"{dir_path}\n{message}")
        self.statusBar().showMessage("正在扫描数据集...")

        # Scan images (fast, no pixel loading)
        self._image_infos = self._converter.scan_dataset(dir_path)

        # Load progress cache
        self._cache = self._converter.load_progress_cache(self._progress_cache_path)

        # 重置检测结果（新数据集加载时清空旧的警告标记）
        self._count_check_failed.clear()

        # Populate list (颜色按缓存状态决定)
        self._image_list_widget.clear()
        for idx, info in enumerate(self._image_infos):
            name = Path(info["image_path"]).stem
            split = info.get("split", "")
            display = f"[{split}] {name}" if split else name
            item = QListWidgetItem(display)
            self._image_list_widget.addItem(item)
            self._apply_item_style(idx)

        self._progress_bar.setMaximum(len(self._image_infos))
        self._update_progress()

        self.statusBar().showMessage(f"已扫描 {len(self._image_infos)} 张图片")
        self._btn_save_all.setEnabled(True)
        self._btn_clear_current.setEnabled(True)

        # 持久化数据集目录
        self._settings.setValue(self.SETTINGS_LAST_DATASET, dir_path)

        # 智能跳转：1) 优先恢复 last_image_path  2) 否则跳到第一张未标注
        target_index = self._resolve_initial_index()
        if target_index >= 0:
            self._go_to_image(target_index)
        return True

    def _restore_last_session(self):
        """启动后自动恢复上次的数据集 + 输出目录 + 导出格式。"""
        # 1) 导出格式（无 IO，最先恢复）
        last_fmt = self._settings.value(self.SETTINGS_LAST_FORMAT, "")
        if last_fmt in ("yolov8_obb", "yolov8_xywhr", "yolov8_pose"):
            self._export_format = last_fmt
            fmt_index = {"yolov8_obb": 0, "yolov8_xywhr": 1, "yolov8_pose": 2}[last_fmt]
            self._format_combo.blockSignals(True)
            self._format_combo.setCurrentIndex(fmt_index)
            self._format_combo.blockSignals(False)

        # 2) 输出目录（先于数据集恢复，确保保存功能可用）
        last_output = self._settings.value(self.SETTINGS_LAST_OUTPUT, "")
        if last_output and Path(last_output).is_dir():
            self._output_dir = last_output
            self._output_path_label.setText(last_output)
            self._btn_save.setEnabled(True)

        # 3) 数据集目录（异步加载）
        last_dataset = self._settings.value(self.SETTINGS_LAST_DATASET, "")
        if last_dataset and Path(last_dataset).is_dir():
            ok = self._load_dataset(last_dataset, silent_invalid=True)
            if ok:
                self.statusBar().showMessage(
                    f"已自动恢复上次会话：{Path(last_dataset).name}", 4000
                )

    def _resolve_initial_index(self) -> int:
        """决定打开数据集后应该跳转到哪张图片。

        策略：
        1. 如果 cache 中记录了 last_image_path 且该图片仍存在于扫描结果中：
           - 若该图片**未标注**，直接跳过去（用户继续上次的工作）
           - 若已标注，转入策略 2
        2. 跳到第一张未标注图片
        3. 全部已标注：跳回 last_image_path 或第一张
        """
        if not self._image_infos:
            return -1

        last_path = self._converter.get_last_image_path(self._cache)
        last_index = -1
        if last_path:
            for i, info in enumerate(self._image_infos):
                if info["image_path"] == last_path:
                    last_index = i
                    break

        # 优先：上次位置且未标注
        if last_index >= 0 and not self._is_saved(self._image_infos[last_index]["image_path"]):
            return last_index

        # 其次：第一张未标注
        for i, info in enumerate(self._image_infos):
            if not self._is_saved(info["image_path"]):
                return i

        # 兜底：全部已标注 → 上次位置或第一张
        return last_index if last_index >= 0 else 0

    def _is_saved(self, image_path: str) -> bool:
        return bool(self._cache.get(image_path, {}).get("saved"))

    def _is_flagged(self, image_path: str) -> bool:
        """检查图片是否被标记为标注难点。"""
        return bool(self._cache.get(image_path, {}).get("flagged"))

    def _set_flagged(self, image_path: str, flagged: bool):
        """设置/取消标记难点。"""
        entry = self._cache.setdefault(image_path, {})
        entry["flagged"] = flagged
        # 持久化到 cache 文件
        if self._progress_cache_path:
            self._converter.save_progress_cache(self._progress_cache_path, self._cache)

    def _on_flag_toggled(self, checked: bool):
        """标记难点按钮切换回调。"""
        if self._current_index < 0 or not self._image_infos:
            return
        img_path = self._image_infos[self._current_index]["image_path"]
        self._set_flagged(img_path, checked)
        self._apply_item_style(self._current_index)
        self._update_status()

    def _on_image_list_context_menu(self, pos):
        """图片列表右键菜单。"""
        item = self._image_list_widget.itemAt(pos)
        if item is None:
            return

        row = self._image_list_widget.row(item)
        img_path = self._image_infos[row]["image_path"]
        is_flagged = self._is_flagged(img_path)

        menu = QMenu(self)
        flag_action = QAction(
            "取消标记" if is_flagged else "标记难点",
            self
        )
        flag_action.triggered.connect(lambda: self._toggle_flag_by_row(row))
        menu.addAction(flag_action)

        go_action = QAction("跳转到此图片", self)
        go_action.triggered.connect(lambda: self._go_to_image(row))
        menu.addAction(go_action)

        menu.exec_(self._image_list_widget.mapToGlobal(pos))

    def _toggle_flag_by_row(self, row: int):
        """根据行号切换标记难点状态。"""
        img_path = self._image_infos[row]["image_path"]
        new_flag = not self._is_flagged(img_path)
        self._set_flagged(img_path, new_flag)
        self._apply_item_style(row)
        # 如果是当前图片，同步按钮状态
        if row == self._current_index:
            self._btn_flag.blockSignals(True)
            self._btn_flag.setChecked(new_flag)
            self._btn_flag.blockSignals(False)
        self._update_status()

    def _toggle_flag_current(self):
        """快捷键M：切换当前图片的标记难点状态。"""
        if self._current_index < 0 or not self._image_infos:
            return
        self._toggle_flag_by_row(self._current_index)

    def _set_output_dir(self):
        """Set the output directory for exported annotations."""
        start_dir = self._settings.value(self.SETTINGS_LAST_OUTPUT, str(Path.home()))
        dir_path = QFileDialog.getExistingDirectory(
            self, "选择输出目录", start_dir,
        )
        if dir_path:
            self._output_dir = dir_path
            self._output_path_label.setText(dir_path)
            self._btn_save.setEnabled(True)
            self._settings.setValue(self.SETTINGS_LAST_OUTPUT, dir_path)

    def _on_format_changed(self, index: int):
        if index == 0:
            self._export_format = "yolov8_obb"
        elif index == 1:
            self._export_format = "yolov8_xywhr"
        else:
            self._export_format = "yolov8_pose"
        self._settings.setValue(self.SETTINGS_LAST_FORMAT, self._export_format)

    # --- Navigation ---

    def _go_to_image(self, index: int):
        """Navigate to a specific image (lazy load on demand)."""
        if not (0 <= index < len(self._image_infos)):
            return

        # 切换前持久化当前张 OBB 几何到 cache（仅防编辑丢失，不算正式保存）
        if self._current_index >= 0 and self._current_annotation and self._current_annotation.modified:
            self._checkpoint_geometry_to_cache()

        # 切换前：把当前图的增强参数写入缓存（每图独立持久化）
        self._persist_enhance_params_for_current()

        prev_index = self._current_index
        self._current_index = index
        info = self._image_infos[index]

        # Lazy load annotations for this image
        cache_entry = self._cache.get(info["image_path"])
        self._current_annotation = self._converter.load_single(
            info["image_path"], info["label_path"], info["width"], info["height"],
            cache_entry=cache_entry,
        )

        # 加载到画布前，先从 cache 读出该图的增强参数并同步到 canvas / toolbar
        enhance_params = self._load_enhance_params_for(info["image_path"])
        self._canvas.set_enhance_params(enhance_params)
        self._enhance_toolbar.sync_from_params(enhance_params)
        if self._enhance_dialog is not None and self._enhance_dialog.isVisible():
            # 如果弹窗仍开着，重建以同步控件初值
            self._enhance_dialog.close()
            self._enhance_dialog = None

        # Apply layer visibility
        self._apply_layer_visibility()

        self._image_list_widget.setCurrentRow(index)
        self._canvas.load_image(info["image_path"], self._current_annotation.annotations)

        # 记录最后访问位置 + 落盘（便于断点续标）
        if self._progress_cache_path:
            self._converter.set_last_image_path(self._cache, info["image_path"])
            self._converter.save_progress_cache(self._progress_cache_path, self._cache)

        # 切换前后两行的视觉样式（更新加粗状态）
        if prev_index >= 0 and prev_index != index:
            self._apply_item_style(prev_index)
        self._apply_item_style(index)

        # 同步标记难点按钮状态
        is_flagged = self._is_flagged(info["image_path"])
        self._btn_flag.blockSignals(True)
        self._btn_flag.setChecked(is_flagged)
        self._btn_flag.blockSignals(False)

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
            self._sync_visibility_radios(default_v=2, enabled=False)
            self._btn_delete_ann.setEnabled(False)
            self._btn_relabel_ann.setEnabled(False)
            return

        # Map scene index to original annotation index
        if 0 <= index < len(self._canvas._index_map):
            orig_idx = self._canvas._index_map[index]
        else:
            orig_idx = index

        if 0 <= orig_idx < len(self._current_annotation.annotations):
            ann = self._current_annotation.annotations[orig_idx]
            angle_deg = math.degrees(ann.angle)
            v_label = {2: "可见", 1: "遮挡", 0: "不可见"}.get(
                int(ann.keypoint_visibility), "可见"
            )
            category = get_vertebra_category(ann.class_id)
            category_name = VertebraCategory.CATEGORY_NAMES.get(category, "未知") if category else "未知"
            shape_label = "直线" if ann.shape_type == "line" else "矩形"
            self._ann_info_label.setText(
                f"椎骨: {ann.class_name} (ID={ann.class_id})\n"
                f"类别: {category_name}\n"
                f"形状: {shape_label}\n"
                f"角度: {angle_deg:.1f}°\n"
                f"尺寸: {ann.width:.0f} x {ann.height:.0f}\n"
                f"中心: ({ann.center.x:.0f}, {ann.center.y:.0f})\n"
                f"可见性: v={int(ann.keypoint_visibility)} ({v_label})"
            )
            # Update angle spinbox without triggering signal
            self._angle_spin.blockSignals(True)
            self._angle_spin.setValue(angle_deg)
            self._angle_spin.blockSignals(False)
            # Sync visibility radio buttons
            self._sync_visibility_radios(int(ann.keypoint_visibility), enabled=True)
            self._btn_delete_ann.setEnabled(True)
            self._btn_relabel_ann.setEnabled(True)

    def _sync_visibility_radios(self, default_v: int, enabled: bool):
        """同步右侧"关键点可见性" radio button 至给定 v 值（不触发回调）。"""
        self._vis_group.blockSignals(True)
        btn = self._vis_group.button(default_v)
        if btn is not None:
            btn.setChecked(True)
        for v in (0, 1, 2):
            b = self._vis_group.button(v)
            if b is not None:
                b.setEnabled(enabled)
        self._vis_group.blockSignals(False)

    def _on_visibility_changed(self, button):
        """用户切换关键点可见性 radio button 时回调。"""
        ann = self._canvas.get_selected_annotation()
        if ann is None:
            return
        new_v = self._vis_group.id(button)
        if int(ann.keypoint_visibility) == new_v:
            return
        ann.keypoint_visibility = new_v
        if self._current_annotation:
            self._current_annotation.modified = True
        # 重绘画布（边框样式可能变化）+ 刷新信息面板
        if 0 <= self._canvas._current_selection < len(self._canvas._obb_items):
            self._canvas._obb_items[self._canvas._current_selection].update()
        self._canvas.viewport().update()
        self._on_annotation_selected(self._canvas._current_selection)
        self._update_status()
        # 列表项状态颜色更新（modified 状态变化）
        if self._current_index >= 0:
            self._apply_item_style(self._current_index)

    def _on_annotation_modified(self):
        """Mark current image as modified."""
        if self._current_annotation:
            self._current_annotation.modified = True
            self._on_annotation_selected(self._canvas._current_selection)
            self._update_status()
            if self._current_index >= 0:
                self._apply_item_style(self._current_index)

    def _on_layer_changed(self):
        """Handle layer visibility checkbox changes."""
        self._show_cervical = self._chk_cervical.isChecked()
        self._show_thoracic = self._chk_thoracic.isChecked()
        self._show_lumbar = self._chk_lumbar.isChecked()
        self._show_sacral = self._chk_sacral.isChecked()
        self._apply_layer_visibility()
        self._canvas.viewport().update()

    def _apply_layer_visibility(self):
        """Apply layer visibility to all annotations based on vertebra category."""
        if not self._current_annotation:
            return
        for ann in self._current_annotation.annotations:
            category = get_vertebra_category(ann.class_id)
            if category == VertebraCategory.CERVICAL:
                ann.visible = self._show_cervical
            elif category == VertebraCategory.THORACIC:
                ann.visible = self._show_thoracic
            elif category == VertebraCategory.LUMBAR:
                ann.visible = self._show_lumbar
            elif category == VertebraCategory.SACRAL:
                ann.visible = self._show_sacral
            else:
                ann.visible = True  # 未知类别默认可见
        # Update canvas items
        for item in self._canvas._obb_items:
            item.setVisible(item.annotation.visible)

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

    # --- 绘制工具方法 ---
    
    def _toggle_draw_mode(self, shape: str):
        """切换绘制模式；再次点击同一按钮则退出绘制。

        所有标注（含 S1）统一使用矩形绘制；S1 训练时直接以 OBB 形态导出。
        """
        # 重复点击当前模式按钮→退出绘制
        if self._current_draw_shape == shape and shape != "none":
            shape = "none"

        self._current_draw_shape = shape
        self._current_draw_class_id = None  # 不预设

        # 同步按钮选中状态
        self._btn_draw_rect.setChecked(shape == "rect")

        # 设置画布绘制模式（class_id=None，绘制完成后由弹窗选择）
        if shape == "none":
            self._canvas.set_draw_mode(AnnotationCanvas.DRAW_NONE)
        elif shape == "rect":
            self._canvas.set_draw_mode(AnnotationCanvas.DRAW_RECT, class_id=None)

    def _toggle_rect_shortcut(self):
        """快捷键 B 入口：与点击“矩形”按钮一致（再按一次退出绘制）。

        仅在加载了图片后生效；在其他输入控件聊焦点时依靠 QShortcut 默认上下文不干扰。
        """
        if not self._current_annotation:
            return
        self._toggle_draw_mode("rect")

    def _on_annotation_created(self, annotation: OBBAnnotation):
        """画布上绘制完成后回调，将标注添加到当前图片。"""
        if annotation is None:
            return

        if not self._current_annotation:
            return

        # 如果未预设椎骨类别，弹出选择对话框
        if annotation.class_id < 0 or not annotation.class_name:
            used = {
                a.class_id
                for a in self._current_annotation.annotations
                if a.class_id is not None and a.class_id >= 0
            }
            suggested = self._suggest_next_class_id(used)
            class_id = self._prompt_vertebra_selection(
                used_class_ids=used,
                suggested_class_id=suggested,
            )
            if class_id is None:
                return  # 用户取消
            annotation.class_id = class_id
            annotation.class_name = VERTEBRA_CLASSES[class_id]

        # 添加到数据模型
        self._current_annotation.annotations.append(annotation)
        self._current_annotation.modified = True

        # 添加到画布
        self._canvas.add_annotation(annotation)

        # 应用图层可见性
        self._apply_layer_visibility()

        # 切回选择模式
        self._toggle_draw_mode("none")

        # 选中新创建的标注
        if self._canvas._obb_items:
            self._canvas.select_annotation(len(self._canvas._obb_items) - 1)

        self._update_status()
        if self._current_index >= 0:
            self._apply_item_style(self._current_index)

    @staticmethod
    def _suggest_next_class_id(used_class_ids: set) -> int:
        """根据当前图已使用的编号推导下一个候选 class_id。
    
        规则：按脊柱从上到下（C7→T1...→L5→S1）递增，返回 max(used)+1；
        空集返回 C7(0)；均已到达 S1(18) 仍返回 18（允许用户仍可反复选，
        重复检查交由 _prompt_vertebra_selection 提示）。
        """
        if not used_class_ids:
            return 0  # C7
        max_used = max(used_class_ids)
        if max_used >= 18:
            return 18  # S1 到顶
        return max_used + 1
    
    def _prompt_vertebra_selection(
        self,
        current_class_id: Optional[int] = None,
        used_class_ids: Optional[set] = None,
        suggested_class_id: Optional[int] = None,
    ) -> Optional[int]:
        """弹出椎骨类型选择对话框，返回 class_id 或 None(取消)。
    
        参数：
          - current_class_id: 改编号场景下当前标注自身的编号（不视为冲突）
          - used_class_ids:    本图已占用的编号集合，冲突项会带 "(已用)"标记
          - suggested_class_id: 智能预选编号；未传时回退到 current_class_id。
        选中已占用编号时会弹二次确认；选中分组标题则要求重选。
        """
        used = set(used_class_ids) if used_class_ids else set()
        # 改编号场景：自身编号不计作冲突
        conflict_ids = {cid for cid in used if cid != current_class_id}
    
        items: List[str] = []
        item_class_ids: List[Optional[int]] = []
    
        def _add(label: str, cid: Optional[int]) -> None:
            if cid is not None and cid in conflict_ids:
                label = f"{label} (已用)"
            items.append(label)
            item_class_ids.append(cid)
    
        _add("颈椎 (C)", None)
        _add("  C7", 0)
        _add("胸椎 (T)", None)
        for i in range(1, 13):
            _add(f"  T{i}", i)
        _add("腰椎 (L)", None)
        for i in range(1, 6):
            _add(f"  L{i}", 12 + i)
        _add("骶椎 (S)", None)
        _add("  S1", 18)
    
        # 预选索引：优先 suggested，其次 current
        target_cid = suggested_class_id if suggested_class_id is not None else current_class_id
        default_idx = 0
        if target_cid is not None:
            for i, cid in enumerate(item_class_ids):
                if cid == target_cid:
                    default_idx = i
                    break
    
        while True:
            name, ok = QInputDialog.getItem(
                self, "选择椎骨类型", "请选择标注的椎骨:",
                items, default_idx, False,
            )
            if not ok:
                return None
            try:
                sel_idx = items.index(name)
            except ValueError:
                return None
            class_id = item_class_ids[sel_idx]
            # 选中分组标题→要求重选
            if class_id is None:
                QMessageBox.information(
                    self, "请选择具体编号",
                    "请选择具体的椎骨编号（如 C7、T3、L5 等）。",
                )
                default_idx = sel_idx
                continue
            # 重复性检查
            if class_id in conflict_ids:
                reply = QMessageBox.warning(
                    self, "编号已被使用",
                    f"当前图中已存在编号为 {VERTEBRA_CLASSES[class_id]} 的标注。\n\n"
                    "椎骨编号应唯一，重复使用会导致训练数据异常。\n"
                    "是否仍要使用该编号？（建议重选）",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                if reply != QMessageBox.Yes:
                    default_idx = sel_idx
                    continue
            return class_id

    def _delete_selected_annotation(self):
        """删除当前选中的标注。"""
        if not self._current_annotation:
            return

        sel_idx = self._canvas._current_selection
        if sel_idx < 0 or sel_idx >= len(self._canvas._obb_items):
            return

        # 获取原始索引
        orig_idx = self._canvas._index_map[sel_idx]

        # 确认删除
        ann = self._canvas._obb_items[sel_idx].annotation
        reply = QMessageBox.question(
            self, "确认删除",
            f"确定要删除标注 \"{ann.class_name}\" 吗？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        # 从数据模型中删除
        if 0 <= orig_idx < len(self._current_annotation.annotations):
            self._current_annotation.annotations.pop(orig_idx)
            self._current_annotation.modified = True

        # 重新加载画布（保持当前缩放）
        self._canvas.reload_annotations(self._current_annotation.annotations)
        self._apply_layer_visibility()

        self._update_status()
        if self._current_index >= 0:
            self._apply_item_style(self._current_index)

    def _on_relabel_requested(self, scene_idx: int):
        """双击标注后弹出椎骨编号修改对话框。"""
        if not self._current_annotation:
            return
        if scene_idx < 0 or scene_idx >= len(self._canvas._obb_items):
            return

        orig_idx = self._canvas._index_map[scene_idx]
        if orig_idx < 0 or orig_idx >= len(self._current_annotation.annotations):
            return

        ann = self._current_annotation.annotations[orig_idx]
        self._relabel_annotation(ann)

    def _relabel_selected_annotation(self):
        """按钮点击：对当前选中的标注执行更改编号。"""
        ann = self._canvas.get_selected_annotation()
        if ann is None:
            return
        self._relabel_annotation(ann)

    def _relabel_annotation(self, ann: OBBAnnotation):
        """弹出椎骨编号选择对话框并更改标注编号。"""
        used = set()
        if self._current_annotation:
            used = {
                a.class_id
                for a in self._current_annotation.annotations
                if a.class_id is not None and a.class_id >= 0
            }
        new_class_id = self._prompt_vertebra_selection(
            current_class_id=ann.class_id,
            used_class_ids=used,
        )
        if new_class_id is None or new_class_id == ann.class_id:
            return  # 取消或未变

        ann.class_id = new_class_id
        ann.class_name = VERTEBRA_CLASSES[new_class_id]
        if self._current_annotation:
            self._current_annotation.modified = True

        # 刷新画布（保持当前缩放）
        self._canvas.reload_annotations(self._current_annotation.annotations)
        self._apply_layer_visibility()

        self._on_annotation_selected(self._canvas._current_selection)
        self._update_status()
        if self._current_index >= 0:
            self._apply_item_style(self._current_index)

        self.statusBar().showMessage(f"编号已更改为 {ann.class_name}", 2000)

    def _auto_sort_annotations(self):
        """按 Y 坐标从上到下自动排序并重编号所有标注 (C7→T1→...→L5→S1)。"""
        if not self._current_annotation:
            return

        annotations = self._current_annotation.annotations
        if not annotations:
            return

        # 二次确认（重编号会覆盖现有编号，需用户明确同意）
        reply = QMessageBox.question(
            self, "确认自动排序",
            f"当前图片共有 {len(annotations)} 个标注。\n\n"
            "自动排序将按 Y 坐标从上到下重新分配编号：\n"
            "C7(0) → T1(1) → T2(2) → ... → L5(17) → S1(18)\n\n"
            "现有编号将被覆盖，是否继续？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        # 执行排序并重编号
        sorted_anns = auto_sort_annotations(annotations)
        self._current_annotation.annotations = sorted_anns
        self._current_annotation.modified = True

        # 刷新画布
        self._canvas.reload_annotations(sorted_anns)
        self._apply_layer_visibility()

        self._update_status()
        if self._current_index >= 0:
            self._apply_item_style(self._current_index)

        self.statusBar().showMessage(
            f"自动排序完成：{len(sorted_anns)} 个标注已按解剖顺序重编号",
            3000,
        )

    # --- Save Operations ---

    def _save_current(self, silent: bool = False):
        """Save current image annotations."""
        if not self._image_infos or not self._output_dir:
            if not silent:
                QMessageBox.warning(self, "提示", "请先设置输出目录")
            return

        if not self._current_annotation:
            return

        if self._save_min_count_enabled:
            ann_count = len(self._current_annotation.annotations)
            min_count = max(1, int(self._save_min_count_value))
            if ann_count < min_count:
                if not silent:
                    QMessageBox.warning(
                        self, "保存失败",
                        f"当前图片标注数量为 {ann_count}，少于设置要求的 {min_count}。\n"
                        '请补充标注，或在“工具 -> 通用设置”中调整该规则。'
                    )
                return
        
        if self._save_max_count_enabled:
            ann_count = len(self._current_annotation.annotations)
            max_count = max(1, int(self._save_max_count_value))
            if ann_count > max_count:
                if not silent:
                    QMessageBox.warning(
                        self, "保存失败",
                        f"当前图片标注数量为 {ann_count}，超过设置限制的 {max_count}。\n"
                        '请删除多余标注，或在“工具 -> 通用设置”中调整该规则。'
                    )
                return

        info = self._image_infos[self._current_index]

        # 校验 S1 数量：每张图片只能有 0-1 个 S1
        s1_count = sum(
            1 for ann in self._current_annotation.annotations
            if ann.class_id == 18
        )
        if s1_count > 1:
            if not silent:
                QMessageBox.warning(
                    self, "保存失败",
                    f"当前图片存在 {s1_count} 个 S1 标注。\n"
                    "每张图片最多只能有 1 个 S1（髅椎）标注，请删除多余的 S1。"
                )
            return

        # Compute split-aware output directory
        split = info.get("split", "")
        if split:
            out_dir = os.path.join(self._output_dir, split, "labels")
        else:
            out_dir = self._output_dir

        if self._export_format == "yolov8_obb":
            self._converter.save_obb_yolov8(self._current_annotation, out_dir, overwrite=True)
        elif self._export_format == "yolov8_xywhr":
            self._converter.save_obb_xywhr(self._current_annotation, out_dir, overwrite=True)
        else:  # yolov8_pose
            self._converter.save_pose_yolov8(self._current_annotation, out_dir, overwrite=True)

        # Update cache (含每个标注的 keypoint_visibility 状态)
        img_path = info["image_path"]
        self._cache[img_path] = {
            "modified": False,
            "saved": True,
            "annotation_states": self._converter.build_annotation_states(
                self._current_annotation
            ),
        }
        self._current_annotation.modified = False

        # Save cache to disk
        self._converter.save_progress_cache(self._progress_cache_path, self._cache)

        # Update UI
        self._apply_item_style(self._current_index)
        self._update_progress()
        self._update_status()

        if not silent:
            self.statusBar().showMessage(f"已保存: {Path(img_path).name}")

    def _checkpoint_geometry_to_cache(self):
        """把当前张 OBB 几何写入 cache（不算正式保存，仅防止编辑丢失）。

        触发场景：用户尚未设置输出目录但已经在编辑，切图时调用本方法
        把几何状态持久化到 .annotate_progress.json，下次回到该图能恢复编辑。
        """
        if not (self._current_annotation and self._progress_cache_path):
            return
        info = self._image_infos[self._current_index]
        img_path = info["image_path"]
        existing = self._cache.get(img_path, {})
        existing.update({
            # 注意：不改 saved 字段（仍保持 false / 之前的值），仅记录几何
            "annotation_states": self._converter.build_annotation_states(
                self._current_annotation
            ),
        })
        # 如果之前没有 saved 字段，确保至少有个 false
        existing.setdefault("saved", False)
        existing["modified"] = True
        self._cache[img_path] = existing
        self._converter.save_progress_cache(self._progress_cache_path, self._cache)

    # ---------------- 图像增强（不修改原图，仅辅助显示） ----------------

    def _load_enhance_params_for(self, img_path: str) -> EnhanceParams:
        """从 cache 读出该图的增强参数，不存在返回默认。"""
        entry = self._cache.get(img_path, {})
        return EnhanceParams.from_dict(entry.get("enhance_params", {}))

    def _persist_enhance_params_for_current(self):
        """把当前画布上的增强参数写入当前图的 cache（只在变化时写盘）。"""
        if not self._image_infos or self._current_index < 0 or not self._progress_cache_path:
            return
        info = self._image_infos[self._current_index]
        img_path = info["image_path"]
        cur_params = self._canvas.get_enhance_params()
        cur_dict = cur_params.to_dict()
        existing = self._cache.get(img_path, {})
        old_dict = existing.get("enhance_params")
        if old_dict == cur_dict:
            return  # 无变化，跳过写盘
        # 默认参数不必写入（保持 cache 精简）
        if cur_params.is_identity():
            if "enhance_params" in existing:
                existing.pop("enhance_params", None)
            else:
                return
        else:
            existing["enhance_params"] = cur_dict
        self._cache[img_path] = existing
        self._converter.save_progress_cache(self._progress_cache_path, self._cache)

    def _open_enhance_dialog(self):
        """打开/复用增强调参弹窗（非模态，边调边看）。"""
        if not self._image_infos or self._current_index < 0:
            QMessageBox.information(self, "提示", "请先打开数据集并选择一张图片")
            return
        if self._enhance_dialog is not None and self._enhance_dialog.isVisible():
            self._enhance_dialog.raise_()
            self._enhance_dialog.activateWindow()
            return
        cur = self._canvas.get_enhance_params()
        self._enhance_dialog = EnhancementDialog(cur, parent=self)
        self._enhance_dialog.params_changed.connect(self._on_enhance_params_changed)
        self._enhance_dialog.show()

    def _on_enhance_params_changed(self, params: EnhanceParams):
        """调参弹窗参数变化 → 同步到 canvas + toolbar，去抖后写缓存。"""
        self._canvas.set_enhance_params(params)
        self._enhance_toolbar.sync_from_params(params)
        # 使用状态栏微提示参数已应用，写盘在切图时统一处理
        self._persist_enhance_params_for_current()

    def _on_enhance_invert_toggled(self, checked: bool):
        """工具条反相按钮 → 单独切换。"""
        cur = self._canvas.get_enhance_params()
        new = EnhanceParams(
            brightness=cur.brightness,
            contrast=cur.contrast,
            gamma=cur.gamma,
            clahe=cur.clahe,
            invert=bool(checked),
        )
        self._canvas.set_enhance_params(new)
        self._enhance_toolbar.sync_from_params(new)
        self._persist_enhance_params_for_current()

    def _reset_enhance_params(self):
        """一键将当前图的增强参数恢复为默认。"""
        default = EnhanceParams()
        self._canvas.set_enhance_params(default)
        self._enhance_toolbar.sync_from_params(default)
        if self._enhance_dialog is not None and self._enhance_dialog.isVisible():
            self._enhance_dialog.close()
            self._enhance_dialog = None
        self._persist_enhance_params_for_current()

    def _save_all(self):
        """Export all images that have been processed."""
        if not self._output_dir:
            QMessageBox.warning(self, "提示", "请先设置输出目录")
            return

        opts = self._ask_save_all_options()
        if not opts:
            return

        export_mode = opts["mode"]
        enforce_min_count = bool(opts["enforce_min_count"])
        min_count = int(opts["min_count"])
        enforce_max_count = bool(opts["enforce_max_count"])
        max_count = int(opts["max_count"])

        count = 0
        skipped_incomplete = 0
        skipped_unsaved_new = 0
        skipped_min_count = 0
        skipped_max_count = 0
        for i, info in enumerate(self._image_infos):
            img_path = info["image_path"]
            label_path = info.get("label_path")
            cache_entry = self._cache.get(img_path, {})

            is_current = (i == self._current_index)
            has_label = bool(label_path) and Path(label_path).exists()

            # 导出条件：
            # - 导出已标注：仅导出在本工具中已正式保存过的图片
            # - 导出全部：导出已保存 + 磁盘已有 label（老数据迁移）
            cache_saved = bool(cache_entry.get("saved"))
            if export_mode == "annotated":
                should_export = cache_saved
            else:
                should_export = cache_saved or has_label

            # 防止误把“仅缓存中的未完成新标注”（无历史 label 且未正式保存）导出为已完成
            if not should_export:
                if cache_entry.get("modified"):
                    skipped_unsaved_new += 1
                elif (
                    is_current
                    and self._current_annotation is not None
                    and self._current_annotation.modified
                ):
                    skipped_unsaved_new += 1
                continue

            if is_current:
                ann = self._current_annotation
            else:
                ann = self._converter.load_single(
                    info["image_path"], label_path,
                    info["width"], info["height"],
                    cache_entry=self._cache.get(img_path),
                )

            if ann is None:
                continue

            if enforce_min_count and len(ann.annotations) < min_count:
                skipped_min_count += 1
                continue

            if enforce_max_count and len(ann.annotations) > max_count:
                skipped_max_count += 1
                continue

            # 与单张保存一致：每张图片最多只能有 1 个 S1
            s1_count = sum(1 for a in ann.annotations if a.class_id == 18)
            if s1_count > 1:
                skipped_incomplete += 1
                continue

            # Compute split-aware output directory
            split = info.get("split", "")
            if split:
                out_dir = os.path.join(self._output_dir, split, "labels")
            else:
                out_dir = self._output_dir

            if self._export_format == "yolov8_obb":
                self._converter.save_obb_yolov8(ann, out_dir, overwrite=True)
            elif self._export_format == "yolov8_xywhr":
                self._converter.save_obb_xywhr(ann, out_dir, overwrite=True)
            else:  # yolov8_pose
                self._converter.save_pose_yolov8(ann, out_dir, overwrite=True)

            # 合并更新 cache，保留 flagged/enhance_params 等字段
            updated = dict(cache_entry)
            updated.update({
                "modified": False,
                "saved": True,
                "annotation_states": self._converter.build_annotation_states(ann),
            })
            self._cache[img_path] = updated
            count += 1

        # Save cache
        self._converter.save_progress_cache(self._progress_cache_path, self._cache)
        self._update_progress()
        # 刷新所有列表项样式
        for i in range(len(self._image_infos)):
            self._apply_item_style(i)

        mode_text = "导出已标注" if export_mode == "annotated" else "导出全部"
        msg = f"{mode_text}：已导出 {count} 个标注文件"
        if skipped_min_count > 0:
            msg += f"（跳过 {skipped_min_count} 张：标注数 < {min_count}）"
        if skipped_max_count > 0:
            msg += f"（跳过 {skipped_max_count} 张：标注数 > {max_count}）"
        if skipped_incomplete > 0:
            msg += f"（跳过 {skipped_incomplete} 张：S1 数量异常）"
        if skipped_unsaved_new > 0:
            msg += f"（未导出 {skipped_unsaved_new} 张未正式保存的新标注）"
        self.statusBar().showMessage(msg)

    def _ask_save_all_options(self) -> Optional[dict]:
        """弹出“全部导出”选项对话框。"""
        dlg = QDialog(self)
        dlg.setWindowTitle("全部导出选项")
        dlg.setModal(True)
        dlg.resize(420, 220)

        layout = QVBoxLayout(dlg)
        layout.addWidget(QLabel("请选择导出范围："))

        rb_annotated = QRadioButton("导出已标注")
        rb_all = QRadioButton("导出全部")
        rb_annotated.setChecked(True)
        layout.addWidget(rb_annotated)
        layout.addWidget(rb_all)

        min_count_row = QHBoxLayout()
        chk_min_count = QCheckBox("检查标注数量不少于")
        chk_min_count.setChecked(True)
        spn_min_count = QSpinBox()
        spn_min_count.setRange(1, 999)
        spn_min_count.setValue(19)
        lbl_min_suffix = QLabel("个")
        min_count_row.addWidget(chk_min_count)
        min_count_row.addWidget(spn_min_count)
        min_count_row.addWidget(lbl_min_suffix)
        min_count_row.addStretch(1)
        layout.addLayout(min_count_row)

        max_count_row = QHBoxLayout()
        chk_max_count = QCheckBox("检查标注数量不多于")
        chk_max_count.setChecked(True)
        spn_max_count = QSpinBox()
        spn_max_count.setRange(1, 999)
        spn_max_count.setValue(19)
        lbl_max_suffix = QLabel("个")
        max_count_row.addWidget(chk_max_count)
        max_count_row.addWidget(spn_max_count)
        max_count_row.addWidget(lbl_max_suffix)
        max_count_row.addStretch(1)
        layout.addLayout(max_count_row)

        hint = QLabel(
            "提示：\n"
            "- 导出已标注：仅导出本工具里已正式保存过的图片\n"
            "- 导出全部：额外包含磁盘原有 label，适合老数据迁移\n"
            "- 标注数量检查对两种导出范围都生效"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(self._muted_text_style(11))
        layout.addWidget(hint)

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(dlg.accept)
        btn_box.rejected.connect(dlg.reject)
        layout.addWidget(btn_box)

        def _sync_count_enabled():
            min_on = chk_min_count.isChecked()
            spn_min_count.setEnabled(min_on)
            lbl_min_suffix.setEnabled(min_on)
            max_on = chk_max_count.isChecked()
            spn_max_count.setEnabled(max_on)
            lbl_max_suffix.setEnabled(max_on)

        rb_annotated.toggled.connect(_sync_count_enabled)
        chk_min_count.toggled.connect(_sync_count_enabled)
        chk_max_count.toggled.connect(_sync_count_enabled)
        _sync_count_enabled()

        if dlg.exec_() != QDialog.Accepted:
            return None

        use_annotated = rb_annotated.isChecked()
        return {
            "mode": "annotated" if use_annotated else "all",
            "enforce_min_count": bool(chk_min_count.isChecked()),
            "min_count": int(spn_min_count.value()),
            "enforce_max_count": bool(chk_max_count.isChecked()),
            "max_count": int(spn_max_count.value()),
        }

    # --- 清空标注数据 ---

    def _current_split(self) -> str:
        """获取当前选中图片所在 split（没有选中则返回空串）。"""
        if not self._image_infos or self._current_index < 0:
            return ""
        return self._image_infos[self._current_index].get("split", "")

    def _clear_all_data(self):
        """菜单入口：弹出 ClearDataDialog，清空进度缓存 + 当前 split 的训练标注。"""
        from .clear_dialog import ClearDataDialog

        if not self._image_infos:
            QMessageBox.information(self, "提示", "请先加载图片数据集。")
            return

        split = self._current_split()
        label_files = []
        if self._output_dir:
            label_files = self._converter.collect_label_files(self._output_dir, split)

        dlg = ClearDataDialog(
            self,
            cache_path=self._progress_cache_path,
            output_dir=self._output_dir,
            split=split,
            label_files=label_files,
        )
        if dlg.exec_() != QDialog.Accepted:
            return

        opts = dlg.get_options()
        deleted_labels = 0
        cache_cleared = False
        failed_msgs = []

        if opts.get("clear_labels") and self._output_dir:
            result = self._converter.clear_outputs(self._output_dir, split)
            deleted_labels = result["deleted"]
            for path, err in result.get("failed", []):
                failed_msgs.append(f"{path}: {err}")

        if opts.get("clear_cache"):
            cache_cleared = self._converter.clear_progress_cache(self._progress_cache_path)
            if cache_cleared:
                # 同时重置内存中的缓存对象
                self._cache = {}

        # 重载当前图片从磁盘（label 文件已删），刷新画布
        if self._current_index >= 0 and self._current_index < len(self._image_infos):
            info = self._image_infos[self._current_index]
            self._current_annotation = self._converter.load_single(
                info["image_path"], info["label_path"],
                info["width"], info["height"],
                cache_entry=self._cache.get(info["image_path"]),
            )
            self._canvas.load_image(
                info["image_path"], self._current_annotation.annotations
            )

        # 刷新列表项样式与进度
        for i in range(len(self._image_infos)):
            self._apply_item_style(i)
        self._update_progress()
        self._update_status()

        # 反馈
        msg_parts = []
        if cache_cleared:
            msg_parts.append("已删除进度缓存")
        if deleted_labels > 0:
            msg_parts.append(f"已删除 {deleted_labels} 个标注文件")
        if not msg_parts:
            msg_parts.append("未删除任何文件")
        summary = "；".join(msg_parts)
        if failed_msgs:
            QMessageBox.warning(
                self, "清空完成（部分失败）",
                summary + "\n\n以下文件删除失败：\n" + "\n".join(failed_msgs[:10]),
            )
        else:
            self.statusBar().showMessage(summary, 5000)

    def _clear_current_image(self):
        """清空当前图片的标注、缓存条目与已导出 .txt（需二次确认）。"""
        if not self._image_infos or self._current_index < 0:
            return
        if self._current_annotation is None:
            return

        info = self._image_infos[self._current_index]
        img_name = Path(info["image_path"]).name
        n_ann = len(self._current_annotation.annotations)

        reply = QMessageBox.question(
            self, "清空当前图片",
            (
                f"确定清空当前图片的所有标注吗？\n\n"
                f"图片：{img_name}\n"
                f"当前标注数：{n_ann}\n\n"
                f"将执行：\n"
                f"  • 清空画布上的所有标注\n"
                f"  • 删除 .annotate_progress.json 中该图片的缓存条目\n"
                f"  • 删除对应的已导出 .txt 文件（如果存在）\n\n"
                f"此操作不可恢复。"
            ),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        # 1. 清空内存中的标注
        self._current_annotation.annotations.clear()
        self._current_annotation.modified = True
        self._canvas.load_image(
            info["image_path"], self._current_annotation.annotations
        )

        # 2. 删除缓存条目
        img_path = info["image_path"]
        if img_path in self._cache:
            del self._cache[img_path]
        if self._progress_cache_path:
            self._converter.save_progress_cache(self._progress_cache_path, self._cache)

        # 3. 删除对应 .txt 文件
        deleted_label = False
        if self._output_dir:
            split = info.get("split", "")
            stem = Path(img_path).stem
            deleted_label = self._converter.clear_label_for(stem, self._output_dir, split)

        # 4. 刷新 UI
        self._apply_item_style(self._current_index)
        self._update_progress()
        self._update_status()

        msg = f"已清空当前图片标注"
        if deleted_label:
            msg += "（同时删除了 .txt 文件）"
        self.statusBar().showMessage(msg, 4000)

    # ------------------------------------------------------------------
    # AI 推理预标注
    # ------------------------------------------------------------------

    def _run_inference(self):
        """对当前图片执行 AI 推理，清空画布并填充预标注结果。"""
        # 前置检查
        if not self._image_infos or self._current_index < 0:
            LOGGER.warning("AI inference aborted: no dataset/image loaded")
            QMessageBox.warning(self, "AI 推理", "请先加载数据集。")
            return
        if self._current_annotation is None:
            LOGGER.warning("AI inference aborted: current annotation is None")
            return

        # 防止重复点击
        if self._inference_worker is not None and self._inference_worker.isRunning():
            LOGGER.info("AI inference ignored: worker is already running")
            return

        info = self._image_infos[self._current_index]
        image_path = info["image_path"]
        LOGGER.info(
            "AI inference requested: index=%d, image=%s, split=%s",
            self._current_index,
            image_path,
            info.get("split", ""),
        )
        LOGGER.info(
            "Inference context: dataset_root=%s, output_dir=%s, export_format=%s, existing_annotations=%d",
            self._dataset_root,
            self._output_dir,
            self._export_format,
            len(self._current_annotation.annotations),
        )

        # 若有未保存的标注，提示确认
        n_existing = len(self._current_annotation.annotations)
        if n_existing > 0:
            reply = QMessageBox.question(
                self, "AI 推理预标注",
                f"当前图片已有 {n_existing} 个标注。\n\n"
                f"执行推理将清空当前画布的所有标注，并用 AI 结果替换。\n"
                f"确定继续吗？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if reply != QMessageBox.Yes:
                LOGGER.info("AI inference cancelled by user confirmation dialog")
                return

        # 禁用推理按钮，显示进度
        self.statusBar().showMessage("AI 推理准备中…")

        # 创建并启动 worker
        self._inference_worker = InferenceWorker(
            image_path=image_path,
            model_manager=self._model_manager,
            parent=self,
        )
        self._inference_worker.finished.connect(self._on_inference_finished)
        self._inference_worker.error.connect(self._on_inference_error)
        self._inference_worker.progress.connect(self._on_inference_progress)
        self._inference_worker.start()
        LOGGER.info("Inference worker started for image: %s", image_path)

    def _on_inference_progress(self, message: str):
        """推理过程中的进度更新。"""
        LOGGER.info("Inference progress: %s", message)
        self.statusBar().showMessage(message)

    def _on_inference_finished(self, annotations: list):
        """推理成功，将结果填充到画布。"""
        self._inference_worker = None

        if self._current_annotation is None:
            return

        # 清空现有标注
        self._current_annotation.annotations.clear()

        # 写入推理结果
        self._current_annotation.annotations.extend(annotations)
        self._current_annotation.modified = True

        # 刷新画布（保持当前缩放/平移）
        info = self._image_infos[self._current_index]
        self._canvas.load_image(
            info["image_path"], self._current_annotation.annotations
        )

        # 应用图层可见性
        self._apply_layer_visibility()

        # 更新 cache
        img_path = info["image_path"]
        self._cache[img_path] = {
            "annotation_states": self._converter.build_annotation_states(
                self._current_annotation
            ),
        }
        if self._progress_cache_path:
            self._converter.save_progress_cache(
                self._progress_cache_path, self._cache
            )

        # 刷新 UI
        self._apply_item_style(self._current_index)
        self._update_progress()
        self._update_status()

        # 统计结果
        from ..core.models import get_vertebra_category
        cats = {"cervical": 0, "thoracic": 0, "lumbar": 0, "sacral": 0}
        for ann in annotations:
            cat = get_vertebra_category(ann.class_id)
            if cat:
                cats[cat] += 1

        self.statusBar().showMessage(
            f"AI 推理完成: {len(annotations)} 个椎骨 "
            f"(C={cats['cervical']}, T={cats['thoracic']}, "
            f"L={cats['lumbar']}, S={cats['sacral']})",
            5000,
        )
        LOGGER.info(
            "Inference applied to canvas: total=%d, C=%d, T=%d, L=%d, S=%d",
            len(annotations),
            cats["cervical"],
            cats["thoracic"],
            cats["lumbar"],
            cats["sacral"],
        )

    def _on_inference_error(self, error_msg: str):
        """推理失败，显示错误信息。"""
        self._inference_worker = None
        self.statusBar().showMessage("AI 推理失败", 5000)
        LOGGER.error("Inference failed and returned error:\n%s", error_msg)
        QMessageBox.critical(
            self, "AI 推理失败",
            f"推理过程中发生错误：\n\n{error_msg}\n\n"
            f"请确保：\n"
            f"  1. 已安装 spine-infer 和 onnxruntime\n"
            f"  2. 模型下载地址正确且可访问\n"
            f"  3. 网络连接正常\n\n"
            f"详细堆栈已输出到控制台；若为 exe 启动，请查看\n"
            f"%USERPROFILE%\\.cache\\spine-annotator\\logs\\app.log",
        )

    def _update_progress(self):
        """Update progress bar + 永久进度标签。"""
        saved = self._count_saved()
        total = len(self._image_infos)
        self._progress_bar.setMaximum(max(total, 1))
        self._progress_bar.setValue(saved)
        if total > 0:
            pct = saved / total * 100
            self._progress_label.setText(
                f"已标注 {saved} / {total}  ·  {pct:.1f}%"
            )
        else:
            self._progress_label.setText("")

    def _count_saved(self) -> int:
        return sum(
            1 for k, v in self._cache.items()
            if k != YOLOConverter.META_KEY and isinstance(v, dict) and v.get("saved")
        )

    def _update_status(self):
        """Update status bar info."""
        if not self._image_infos or not self._current_annotation:
            self._status_label.setText("")
            return
        info = self._image_infos[self._current_index]
        parts = []
        if self._current_annotation.modified:
            parts.append("⚠ 未保存")
        if self._is_flagged(info["image_path"]):
            parts.append("⚑ 难点")
        if self._is_saved(info["image_path"]):
            parts.append("✓")
        flags_str = " | ".join(parts)
        if flags_str:
            flags_str = f" [{flags_str}]"
        self._status_label.setText(
            f"{self._current_index + 1}/{len(self._image_infos)} | "
            f"{Path(info['image_path']).name}{flags_str} | "
            f"{len(self._current_annotation.annotations)} 个标注"
        )

    # --- 列表项三态视觉样式 ---

    def _check_annotation_counts(self):
        """全局检测标注数量，弹对话框配置检测参数，结果以警告标记显示在图片列表。"""
        if not self._image_infos:
            QMessageBox.information(self, "提示", "请先加载图片数据集。")
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("检测标注数量")
        dlg.setModal(True)
        dlg.resize(420, 220)

        layout = QVBoxLayout(dlg)
        layout.addWidget(QLabel("请选择检测范围和标准数量："))

        chk_annotated = QCheckBox("已标注的图片")
        chk_annotated.setChecked(True)
        chk_unannotated = QCheckBox("未标注的图片")
        chk_unannotated.setChecked(False)

        scope_row = QHBoxLayout()
        scope_row.addWidget(chk_annotated)
        scope_row.addWidget(chk_unannotated)
        scope_row.addStretch(1)
        layout.addLayout(scope_row)

        count_row = QHBoxLayout()
        count_row.addWidget(QLabel("标准标注数量："))
        spn_count = QSpinBox()
        spn_count.setRange(1, 999)
        spn_count.setValue(19)
        count_row.addWidget(spn_count)
        count_row.addWidget(QLabel("个"))
        count_row.addStretch(1)
        layout.addLayout(count_row)

        hint = QLabel(
            "提示：检测不通过的图片将在左侧列表中以红色 ⚠ 标记。\n"
            "已标注：在本工具中已正式保存的图片。\n"
            "未标注：尚未保存过的图片（含磁盘原有 label）。"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(self._muted_text_style(11))
        layout.addWidget(hint)

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(dlg.accept)
        btn_box.rejected.connect(dlg.reject)
        layout.addWidget(btn_box)

        if dlg.exec_() != QDialog.Accepted:
            return

        check_annotated = chk_annotated.isChecked()
        check_unannotated = chk_unannotated.isChecked()
        if not check_annotated and not check_unannotated:
            QMessageBox.information(self, "提示", "请至少勾选一种检测范围。")
            return

        target_count = int(spn_count.value())

        # 执行检测
        self._count_check_failed.clear()
        checked = 0
        failed = 0

        for i, info in enumerate(self._image_infos):
            img_path = info["image_path"]
            is_saved = self._is_saved(img_path)

            # 根据筛选条件决定是否检测
            if is_saved and not check_annotated:
                continue
            if not is_saved and not check_unannotated:
                continue

            # 计算标注数量
            # 优先读 cache annotation_states；其次从磁盘加载 label
            cache_entry = self._cache.get(img_path, {})
            states = cache_entry.get("annotation_states")
            if states is not None:
                ann_count = len(states)
            elif info.get("label_path") and Path(info["label_path"]).exists():
                # 从磁盘 label 文件读取（受 _load_labels 19 个上限保护）
                try:
                    ann = self._converter.load_single(
                        img_path, info["label_path"],
                        info["width"], info["height"],
                        cache_entry=None,
                    )
                    ann_count = len(ann.annotations)
                except Exception:
                    ann_count = 0
            else:
                ann_count = 0

            checked += 1
            if ann_count != target_count:
                self._count_check_failed.add(i)
                failed += 1

        # 刷新所有列表项样式
        for i in range(len(self._image_infos)):
            self._apply_item_style(i)

        passed = checked - failed
        self.statusBar().showMessage(
            f"检测完成：扫描 {checked} 张，"
            f"符合标准（{target_count} 个） {passed} 张，"
            f"不符合 {failed} 张"
            f"（已用红色 ⚠ 标记）",
            6000,
        )

    def _apply_item_style(self, index: int):
        """更新单个列表项的颜色 + 加粗状态。
    
        状态颜色（自动适配深 / 浅色主题）：
          - 已修改未保存：橙色（高饱和度，醒目提示）
          - 已保存：使用系统 Disabled 文本色（灰色，表示已完成）
          - 有标注(未保存)：系统 Active 文本色
          - 未标注：系统 Active 文本色
          - 已标记难点：前面加 ⚑ 前缀
          - 数量检测不通过：前面加红色 ⚠ 前缀，红色文字
        当前选中项额外加粗。
        """
        if not (0 <= index < self._image_list_widget.count()):
            return
        item = self._image_list_widget.item(index)
        if item is None:
            return
    
        info = self._image_infos[index]
        img_path = info["image_path"]
        is_current = (index == self._current_index)
        is_modified = (
            is_current
            and self._current_annotation is not None
            and self._current_annotation.modified
        )
        is_saved = self._is_saved(img_path)
        is_flagged = self._is_flagged(img_path)
        is_count_failed = index in self._count_check_failed
    
        # 更新显示文本（加前缀表示特殊状态）
        name = Path(info["image_path"]).stem
        split = info.get("split", "")
        # 基础前缀：难度标记 + 数量警告
        prefix_parts = []
        if is_flagged:
            prefix_parts.append("⚑")
        if is_count_failed:
            prefix_parts.append("⚠")
        prefix = " ".join(prefix_parts) + " " if prefix_parts else ""
        split_part = f"[{split}] " if split else ""
        display = f"{prefix}{split_part}{name}"
        item.setText(display)
    
        palette = self._image_list_widget.palette()
        if is_count_failed:
            # 红色：数量检测不通过，最醒目
            item.setForeground(QBrush(QColor("#e03131")))
        elif is_modified:
            # 橙色：醒目提示有未保存修改
            item.setForeground(QBrush(QColor("#e8590c")))
        elif is_saved:
            # 系统Disabled 文本色（自动深浅模式适配）
            item.setForeground(palette.brush(QPalette.Disabled, QPalette.Text))
        else:
            # 系统默认文本色
            item.setForeground(palette.brush(QPalette.Active, QPalette.Text))
    
        font = item.font()
        font.setBold(is_current)
        item.setFont(font)

    # --- 断点续标：跳转未标注图片 ---

    def _jump_to_next_unannotated(self):
        """从当前位置往后找下一张未标注图片。"""
        if not self._image_infos:
            return
        start = max(self._current_index + 1, 0)
        n = len(self._image_infos)
        # 先从 start 找到末尾，再从开头找到 start（环绕）
        for offset in range(n):
            i = (start + offset) % n
            if not self._is_saved(self._image_infos[i]["image_path"]):
                self._go_to_image(i)
                self.statusBar().showMessage(
                    f"跳转到第 {i + 1} 张未标注图片", 2000
                )
                return
        self.statusBar().showMessage("已全部标注完成 🎉", 3000)

    def _jump_to_prev_unannotated(self):
        """从当前位置往前找上一张未标注图片。"""
        if not self._image_infos:
            return
        start = self._current_index - 1
        n = len(self._image_infos)
        for offset in range(n):
            i = (start - offset) % n
            if not self._is_saved(self._image_infos[i]["image_path"]):
                self._go_to_image(i)
                self.statusBar().showMessage(
                    f"跳转到第 {i + 1} 张未标注图片", 2000
                )
                return
        self.statusBar().showMessage("已全部标注完成 🎉", 3000)

    # --- 关闭前未保存确认 ---

    def closeEvent(self, event):
        """关闭前如果当前图片未保存，弹窗询问。"""
        if self._current_annotation is not None and self._current_annotation.modified:
            name = Path(self._current_annotation.image_path).name
            choice = QMessageBox.question(
                self,
                "未保存的修改",
                f"图片 {name} 有未保存的修改，是否保存？",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                QMessageBox.Save,
            )
            if choice == QMessageBox.Cancel:
                event.ignore()
                return
            if choice == QMessageBox.Save:
                if not self._output_dir:
                    QMessageBox.warning(
                        self, "无法保存",
                        "尚未设置输出目录，请先设置后再关闭。"
                    )
                    event.ignore()
                    return
                self._save_current(silent=True)
        # 最后再 flush 一次 cache（确保 last_image_path 落盘）
        if self._progress_cache_path:
            self._converter.save_progress_cache(self._progress_cache_path, self._cache)
        super().closeEvent(event)
