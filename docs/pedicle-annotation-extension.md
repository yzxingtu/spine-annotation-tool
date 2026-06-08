# 单椎骨椎弓根中心点与可见性标注扩展设计

## 背景

当前 `spine-annotation-tool` 已支持整张脊柱 X 光片的椎骨 OBB / 4 角点 pose 标注，并已接入椎骨关键点模型做 AI 预标注。现有模型已经能够稳定输出每节椎骨的 AABB、OBB、编号，并支撑 Cobb 角和弯型推导。

因此椎弓根专用模型不应重复识别整张 X 光片中的所有椎骨。更合适的拆分是：

```text
整图脊柱模型：识别每节椎骨 AABB / OBB / 编号，并负责 Cobb 角、弯型等上层推导
椎弓根模型：只识别单节椎骨 crop 图中的左右椎弓根中心点和可见性
```

该方案可以降低训练难度，减少与现有脊柱模型的功能重复，并让椎弓根模型专注于局部结构。

## 当前软件调查结论

当前项目结构里与本需求相关的关键点：

- `src/spine_annotator/core/models.py`：核心数据结构为 `ImageAnnotation` 和 `OBBAnnotation`，每个 `OBBAnnotation` 表示整图上的一节椎骨。
- `src/spine_annotator/core/converter.py`：负责读取整图 YOLO 数据集、恢复缓存、导出 OBB / xywhr / 4 角点 pose。
- `src/spine_annotator/ui/image_canvas.py`：画布以整张 X 光片为背景，叠加多个椎骨 OBB 标注。
- `src/spine_annotator/ui/main_window.py`：管理整图数据集、AI 预标注、保存、全部导出和进度缓存。
- `src/spine_annotator/core/inference.py`：AI 预标注会把整图推理结果转换成 `OBBAnnotation`。

当前软件还没有：

- 基于椎骨 AABB 批量裁剪单椎骨图片的工具链。
- 单椎骨 crop 图片的数据集视图。
- 单椎骨 crop 图片上的左右椎弓根点标注模式。
- 单椎骨椎弓根标签导出格式。

因此本次扩展应新增一条独立工作流，不破坏现有整图 OBB 标注和导出。

## 设计原则

1. 不修改原始 X 光片，不覆盖原始整图 label。
2. 不改变现有 OBB / xywhr / 4 角点 pose 导出语义。
3. 用现有模型或现有标注中的椎骨 AABB 自动生成单椎骨 crop 图片。
4. crop 使用 AABB，不做旋转矫正，保持原图方向，避免左右语义混乱。
5. 椎弓根标注只在单椎骨 crop 图上完成。
6. 椎弓根模型只输出 crop 图像坐标系下的 `image_left` / `image_right` 点位和可见性。
7. AP/PA、患者左右、凸侧/凹侧、Nash-Moe 分级仍由上层软件处理。

## 新工作流

### 1. 生成单椎骨 crop 数据集

在整图数据集加载完成后，新增菜单或工具入口：

```text
工具 -> 生成椎弓根 crop 数据集
```

输入来源可以有两种：

- 使用当前整图标注中的椎骨框。
- 对未标注整图先运行现有 AI 椎骨模型，得到每节椎骨 AABB / OBB / 编号后再裁剪。

裁剪规则：

```text
crop_box = AABB 按比例扩大 padding 后得到的矩形
```

建议默认参数：

- `padding_ratio = 0.15`，可在 UI 中调整为 0-0.30。
- crop 超出原图边界时 clamp 到图像范围内。
- 不对 crop 做旋转矫正。
- 原图只读，裁剪结果另存到新目录。

### 2. crop 数据集目录结构

建议生成独立数据集目录：

```text
pedicle_crop_dataset/
├── images/
│   ├── sourceA_T1.jpg
│   ├── sourceA_T2.jpg
│   └── ...
├── labels/
│   ├── sourceA_T1.txt
│   ├── sourceA_T2.txt
│   └── ...
└── meta/
    ├── sourceA_T1.json
    ├── sourceA_T2.json
    └── ...
```

也可以按原始 split 保留结构：

```text
pedicle_crop_dataset/
├── train/
│   ├── images/
│   ├── labels/
│   └── meta/
├── valid/
│   ├── images/
│   ├── labels/
│   └── meta/
└── test/
    ├── images/
    ├── labels/
    └── meta/
```

### 3. crop 图片命名

推荐文件名包含来源和椎骨编号：

```text
{source_stem}_{vertebra_name}.jpg
```

如果同一来源可能重复生成，增加序号或 hash：

```text
{source_stem}_{vertebra_name}_{index:02d}.jpg
```

### 4. meta 文件

每张 crop 图保留回溯信息，方便将预测结果映射回原图，也方便质控：

```json
{
  "source_image": "train/images/sourceA.jpg",
  "source_label": "train/labels/sourceA.txt",
  "vertebra": "T8",
  "internal_class_id": 8,
  "export_class_id": 0,
  "source_image_size": [2048, 2048],
  "crop_image_size": [192, 160],
  "source_aabb_xyxy": [820.0, 710.0, 980.0, 850.0],
  "crop_aabb_xyxy": [796.0, 689.0, 1004.0, 871.0],
  "source_obb_points": [[...], [...], [...], [...]],
  "padding_ratio": 0.15
}
```

其中：

- `source_aabb_xyxy` 是椎骨原始 AABB。
- `crop_aabb_xyxy` 是加 padding 后真实裁剪区域。
- `source_obb_points` 仍保留，但仅用于上层几何推导和回溯，不作为椎弓根模型输入的必要标签。

## 单椎骨标注数据模型

crop 图上的标注不再需要保存整图 OBB。每张 crop 图只需要一组左右椎弓根标注：

```python
@dataclass
class PediclePoint:
    center: Optional[Point] = None
    visibility: int = 0

@dataclass
class CropPedicleAnnotation:
    image_path: str
    image_width: int
    image_height: int
    image_left: PediclePoint = field(default_factory=PediclePoint)
    image_right: PediclePoint = field(default_factory=PediclePoint)
    modified: bool = False
```

可见性建议：

| 值 | 含义 | 标注要求 |
| --- | --- | --- |
| `2` | 清晰可见 | 标中心点 |
| `1` | 模糊/重叠但可定位 | 标中心点，训练时可作为低置信样本 |
| `0` | 不可见/不可定位 | 不强迫标中心点 |

## 单椎骨标注 UI

建议新增“椎弓根 crop 标注模式”，打开 `pedicle_crop_dataset` 后进入单椎骨图片列表，而不是整图多椎骨画布。

右侧面板只保留与当前 crop 相关的控件：

- 当前来源：原图名、椎骨编号、split。
- 标注侧别：图像左椎弓根 / 图像右椎弓根。
- 可见性：可见(2) / 模糊可定位(1) / 不可见(0)。
- 清除当前侧点位。
- 保存当前、上一张、下一张。

画布行为：

1. 点击 crop 图，写入当前侧椎弓根中心点。
2. 拖拽已有点位可微调。
3. `v=2` 使用实心点，`v=1` 使用空心或虚线点，`v=0` 不强制显示点。
4. 点位坐标保存为 crop 图坐标，导出时归一化到 crop 图尺寸。

推荐快捷键：

- `[`：切到图像左椎弓根。
- `]`：切到图像右椎弓根。
- `0` / `1` / `2`：设置当前侧可见性。
- `Delete`：清除当前侧点位。

## 标签导出格式

每张 crop 图对应一个标签文件。因为每张图只有一个椎骨实例，推荐格式为：

```text
class_id left_x left_y left_v right_x right_y right_v
```

字段说明：

- `class_id`：建议保留，默认 `0 = vertebra_crop`。如果后续希望区分 C/T/L/S 或 S1，可扩展。
- `left_x left_y left_v`：crop 图像左侧椎弓根中心点和可见性。
- `right_x right_y right_v`：crop 图像右侧椎弓根中心点和可见性。
- 坐标归一化到 crop 图宽高。
- 当 `v=0` 且没有中心点时，坐标可写 `0 0 0`。

示例：

```text
0 0.428125 0.391667 2 0.584375 0.400000 1
```

如果训练脚本不需要类别，也可在训练前忽略第一列。

## 推理时坐标回原图

椎弓根模型输出 crop 坐标后，上层软件使用 meta 里的 `crop_aabb_xyxy` 映射回原图：

```text
source_x = crop_left + pred_x * crop_width
source_y = crop_top  + pred_y * crop_height
```

不做旋转矫正时，映射只需要平移和缩放，工程实现简单且稳定。

## 与上层算法的关系

椎弓根模型只输出：

```text
crop_image_left_pedicle
crop_image_right_pedicle
crop_image_left_visibility
crop_image_right_visibility
```

上层软件负责：

```text
整图椎骨识别与编号
AABB crop 生成
crop 坐标映射回原图
AP/PA/翻转判断
image_left/image_right -> patient_left/patient_right
右胸弯/左腰弯/顶椎/端椎判断
凸侧/凹侧推导
凸侧椎弓根偏移比例计算
Nash-Moe 0-4 级生成
```

## 推荐实现拆分

### 阶段 1：crop 数据集生成

- 新增基于现有 `OBBAnnotation` 的 AABB 计算工具。
- 新增 `export_pedicle_crops()`，把原图裁剪为单椎骨图片。
- 输出 `images/`、空 `labels/`、`meta/`。
- 支持 padding ratio 配置。
- 保证原始 X 光片和原始 label 只读，不被覆盖。

### 阶段 2：单椎骨标注模式

- 新增 crop 数据集扫描逻辑。
- 新增 `CropPedicleAnnotation` 数据结构。
- 新增单椎骨画布点位标注 UI。
- 新增 crop 标注缓存，避免切图丢失。

### 阶段 3：导出与训练准备

- 新增 `save_crop_pedicle_label()`。
- 单张保存和批量导出写入 `labels/{crop_stem}.txt`。
- README 补充 crop 数据集生成、标注和导出说明。
- 提供训练数据检查：每张 crop 至少有一个可定位椎弓根，字段数固定为 7。

### 阶段 4：AI 辅助

- 保留现有整图 AI 预标注作为 crop 数据来源。
- 椎弓根模型训练完成后，新增“椎弓根 crop AI 预标注”入口。
- 推理时逐节裁剪，调用椎弓根模型，再通过 meta / AABB 映射回原图。

## 验收标准

1. 生成 crop 数据集时不修改原图和原始 label。
2. 每节椎骨输出一张 crop 图，正常全脊柱每张原图约生成 18-19 张 crop。
3. crop 文件、label 文件、meta 文件能通过文件名一一对应。
4. crop 使用 AABB + padding，超出原图边界时正确 clamp。
5. 单椎骨 crop 图上可独立标注图像左/右椎弓根中心点。
6. 左右可见性可独立设置并持久化。
7. `v=0` 的椎弓根允许无中心点，不阻塞保存。
8. 导出标签每行固定 7 个字段。
9. 使用 meta 能把 crop 预测坐标准确映射回原图。
10. 现有整图 OBB、xywhr、4 角点 pose 导出不受影响。

## 风险与注意事项

- AABB crop 需要合理 padding，贴边裁剪可能丢失椎弓根上下文。
- 训练增强若做水平翻转，必须同步交换 image_left 和 image_right 标签。
- 不要在椎弓根模型里预测凸侧/凹侧，否则会把局部结构识别和上层临床推理混在一起。
- 对不可见椎弓根不要强迫标注员猜点，否则会引入系统性噪声。
- crop 数据集应记录来源 meta，否则后续难以做错误回溯和原图坐标映射。
