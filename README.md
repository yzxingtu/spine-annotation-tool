# Spine Annotator

脊柱 X 光片椎骨旋转边界框（OBB）标注工具，用于将现有 YOLOv5/v8 AABB 标注转换为带倾斜角的四边形标注，支持导出为 YOLOv8-OBB / YOLOv8-pose 格式。

## 功能特性

- **自动加载** YOLOv5/v8 数据集（AABB 格式），自动转换为可编辑的四边形框
- **交互式编辑**：拖拽角点微调、旋转手柄旋转、整体拖拽移动
- **精确旋转控制**：支持 ±1° / ±5° 快捷键微调和角度输入框精确定位
- **多格式导出**：YOLOv8-OBB 四角点格式 / xywhr 格式 / YOLOv8-pose 关键点格式
- **关键点可见性**：每个标注可设置 v=2/1/0（可见/遮挡/不可见），适用于脊柱骨过透明肉眼难辨的场景
- **缩放与平移**：鼠标滚轮缩放、中键拖拽平移
- **断点续标友好**：
  - **启动自动恢复**：自动重新打开上次的数据集 + 输出目录 + 导出格式
  - 打开数据集后**自动跳转**到上次离开的位置或第一张未标注图片
  - **编辑状态持久化**：切走再回来，已编辑的 OBB 旋转角度、4 角点坐标、可见性全部完整还原（cache 中存了几何状态，不用反复从原始 AABB 重新加载）
  - 状态栏永久显示进度数字与百分比
  - 列表三态颜色：未标注（黑） / 已保存（灰） / 已修改未保存（橙）
  - `Ctrl+N` / `Ctrl+B` 一键跳转下/上一张未标注图片
  - 关闭软件前自动检查未保存修改并询问

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
| `Ctrl+N` / `Ctrl+B` | 跳到下一张 / 上一张**未标注**图片 |
| `R` / `E` | 逆时针 / 顺时针旋转 5° |
| `T` / `Y` | 逆时针 / 顺时针旋转 1° |
| `Shift+R` / `Shift+E` | 逆时针 / 顺时针旋转 0.5° |
| `W` / `A` / `S` / `D` | 上下左右移动 5px |
| `Shift+W/A/S/D` | 上下左右移动 1px |
| `Ctrl+S` | 保存当前图片标注 |
| `F` | 缩放适配画布 |
| `Esc` | 取消选中 |
| 鼠标滚轮 | 缩放画布 |
| 鼠标中键拖拽 | 平移画布 |

## 导出格式

> **关于 `class_id` 的折叠**：标注工具 UI 中按 19 节解剖编号操作（C7=0, T1=1, ..., L5=17, S1=18），方便人工识别与 QA；
> 但所有导出格式落盘时会自动折叠为训练用的 2 类（与 `scoliosis-pose/scoliosis.yaml` 对齐）：
> - 内部 S1 (18) → 导出 `class_id = 1`
> - 内部 C7~L5 (0~17) → 导出 `class_id = 0`
>
> 椎骨的解剖序（C7→S1）通过下游算法按 y 排序 + S1 锚点自行恢复。

### YOLOv8-OBB 四角点格式（默认）

```
class_id x1 y1 x2 y2 x3 y3 x4 y4
```

8 个归一化坐标值，4 个角点按顺时针排列（左上 → 右上 → 右下 → 左下）。

### YOLOv8-OBB xywhr 格式

```
class_id cx cy w h angle
```

中心点坐标、宽高（归一化）和旋转角度（弧度，范围 `[-π/4, π/4)`）。

两种 OBB 格式均可直接用于 YOLOv8-OBB 模型训练。

### YOLOv8-pose 格式（bbox + 关键点）

```
class_id cx cy w h x1 y1 v1 x2 y2 v2 x3 y3 v3 x4 y4 v4
```

- `class_id`：类别 ID（0 = vertebra, 1 = S1）
- `cx, cy, w, h`：包围 OBB 四个角点的 **AABB**（归一化）
- `x1..x4, y1..y4`：椎骨矩形的 4 个角点，**顺时针** 排列
  - `(x1, y1)` 左上 → `(x2, y2)` 右上 → `(x3, y3)` 右下 → `(x4, y4)` 左下
- `v`：可见性（取自标注的 `keypoint_visibility` 字段，4 个角点统一）
  - `2 = 可见`（默认）
  - `1 = 遮挡`（被金属植入物 / 伪影遮挡，坐标可信）
  - `0 = 不可见`（脊柱骨过透明肉眼难辨，坐标根据相邻椎骨推断）

画布上视觉差异：v=2 实线，v=1 短虚线，v=0 长虚线，标签会显示 `[v=1 遮挡]` 等标记。

可直接用于 YOLOv8-pose 模型训练（`kpt_shape: [4, 3]`，`nc: 2`）。

### 加载兼容性

打开数据集时，`_load_labels` 会自动识别下列三种 .txt 格式并恢复为内部解剖编号：

1. **新 2 类 OBB/pose**（`class_id ∈ {0, 1}` 且含 S1=1）：按 y 排序，C7→L5 顺序补齐，S1 锚定为内部 18
2. **旧解剖学 OBB/pose**（`class_id ∈ 0..18`）：按 `VERTEBRA_CLASSES` 直接映射
3. **旧 YOLOv5 AABB**（5 字段）：`class_id 1/2` 为脊柱整体外框，跳过；`class_id 0` 按 y 自动编号为 C7→L5

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
