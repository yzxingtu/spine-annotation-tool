# Spine Annotator

脊柱 X 光片椎骨旋转边界框（OBB）标注工具，用于将现有 YOLOv5/v8 AABB 标注转换为带倾斜角的四边形标注，支持导出为 YOLOv8-OBB 格式。

## 功能特性

- **自动加载** YOLOv5/v8 数据集（AABB 格式），自动转换为可编辑的四边形框
- **交互式编辑**：拖拽角点微调、旋转手柄旋转、整体拖拽移动
- **精确旋转控制**：支持 ±1° / ±5° 快捷键微调和角度输入框精确定位
- **多格式导出**：YOLOv8-OBB 四角点格式 / xywhr 格式
- **缩放与平移**：鼠标滚轮缩放、中键拖拽平移
- **进度追踪**：实时显示标注进度

## 安装

### 依赖

- Python >= 3.9
- PyQt5
- NumPy
- OpenCV
- Pillow

### 安装步骤

```bash
# 克隆仓库
git clone https://github.com/jiulingyun/spine-annotation-tool.git
cd spine-annotation-tool

# 安装依赖（推荐使用虚拟环境）
pip install -e .
```

或使用现有虚拟环境：

```bash
/path/to/venv/bin/pip install PyQt5 numpy opencv-python-headless Pillow
```

## 使用方法

### 启动

```bash
python main.py
```

### 操作流程

1. **打开数据集** — 点击右侧面板 "打开 YOLO 数据集" 按钮，选择 YOLOv5 数据集根目录（需包含 `train/images`、`train/labels` 等标准结构）
2. **选择图片** — 在左侧列表中点击任意图片，或在画布中使用 `←`/`→` 键切换
3. **选中椎骨** — 点击画布上的标注框，选中后高亮显示
4. **调整角度** — 使用以下任一方式：
   - 拖拽蓝色旋转手柄
   - 按 `R` / `E` 键旋转 ±5°
   - 按 `T` / `Y` 键旋转 ±1°
   - 在右侧角度输入框中输入精确值并点击 "应用"
5. **拖拽角点** — 选中后拖拽黄色角点微调四边形形状
6. **设置输出目录** — 点击 "设置输出目录" 选择导出路径
7. **保存** — 按 `Ctrl+S` 保存当前图片标注，或点击 "全部导出" 批量导出

### 快捷键

| 快捷键 | 功能 |
|--------|------|
| `←` / `→` | 上一张 / 下一张图片 |
| `↑` / `↓` | 上一个 / 下一个标注 |
| `R` / `E` | 逆时针 / 顺时针旋转 5° |
| `T` / `Y` | 逆时针 / 顺时针旋转 1° |
| `Ctrl+S` | 保存当前图片标注 |
| `F` | 缩放适配画布 |
| `Esc` | 取消选中 |
| 鼠标滚轮 | 缩放画布 |
| 鼠标中键拖拽 | 平移画布 |

## 导出格式

### YOLOv8-OBB 四角点格式（默认）

```
class_id x1 y1 x2 y2 x3 y3 x4 y4
```

8 个归一化坐标值，4 个角点按顺时针排列。

### YOLOv8-OBB xywhr 格式

```
class_id cx cy w h angle
```

中心点坐标、宽高（归一化）和旋转角度（弧度，范围 `[-π/4, π/4)`）。

两种格式均可直接用于 YOLOv8-OBB 模型训练。

## 数据集目录结构要求

```
dataset_root/
├── train/
│   ├── images/    # 训练图片
│   └── labels/    # YOLOv5 格式标注 (.txt)
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

标注文件格式（YOLOv5 AABB）：

```
class_id x_center y_center width height
```

所有坐标值为归一化值 `[0, 1]`。

## 项目结构

```
spine-annotation-tool/
├── main.py                          # 程序入口
├── pyproject.toml                   # 项目配置与依赖
└── src/spine_annotator/
    ├── core/
    │   ├── models.py                # OBB 数据模型（Point, OBBAnnotation, ImageAnnotation）
    │   └── converter.py             # YOLO 格式读写与转换
    └── ui/
        ├── image_canvas.py          # 交互式图像画布（基于 QGraphicsView）
        └── main_window.py           # 主窗口与控件面板
```

## 许可证

MIT
