# 左右椎弓根中心点与可见性标注扩展设计

## 背景

当前 `spine-annotation-tool` 已支持脊柱 X 光片椎骨 OBB / 4 角点 pose 标注，并已接入椎骨关键点模型进行 AI 预标注。现有椎骨关键点模型已经能够稳定识别每一节椎骨外框、支撑 Cobb 角计算与弯型推导，因此后续椎弓根模型不需要承担整片方向、弯型、凸侧或 Nash-Moe 分级判断。

新需求的训练目标应收敛为局部结构识别：

- 输入：单节椎骨 ROI 或带椎骨框上下文的整片图像。
- 输出：图像左侧椎弓根中心点、图像右侧椎弓根中心点、左右独立可见性。

AP/PA、图像左右到患者左右的映射、右胸弯/左腰弯、凸侧椎弓根、椎弓根偏移比例、Nash-Moe 0-4 级均由上层软件基于已有椎骨框和业务规则推导。

## 设计原则

1. 不改动既有椎骨 OBB / 4 角点 pose 标注语义，避免影响已完成的椎骨关键点训练数据。
2. 椎弓根标注作为每个椎骨标注的附属字段保存，不新增独立椎骨对象。
3. 标注侧别使用 `image_left` / `image_right`，不在该模型内处理 `patient_left` / `patient_right`。
4. 左右椎弓根可见性相互独立，不复用现有 `keypoint_visibility` 字段。
5. 不强制不可见椎弓根标中心点；不可定位时坐标可为空，导出训练格式时按 YOLO pose 规则写占位坐标和 `v=0`。

## 数据模型扩展

建议在 `src/spine_annotator/core/models.py` 中新增椎弓根数据结构：

```python
@dataclass
class PediclePoint:
    center: Optional[Point] = None
    visibility: int = 0
    confidence: Optional[float] = None

@dataclass
class PedicleAnnotation:
    image_left: PediclePoint = field(default_factory=PediclePoint)
    image_right: PediclePoint = field(default_factory=PediclePoint)
```

并在 `OBBAnnotation` 上增加：

```python
pedicles: PedicleAnnotation = field(default_factory=PedicleAnnotation)
```

可见性建议沿用 YOLO pose 的 0/1/2 基础语义，但命名要明确为椎弓根可见性：

| 值 | 含义 | 标注要求 |
| --- | --- | --- |
| `2` | 清晰可见 | 标中心点 |
| `1` | 模糊/重叠但可定位 | 标中心点，训练时可作为低置信样本 |
| `0` | 不可见/不可定位 | 不要求标中心点 |

如果需要更细的质控，可在 UI 中展示 4 档，但导出训练时折叠为 0/1/2：

- 清晰可见 -> 2
- 模糊但可定位 -> 1
- 与椎体边缘/结构重叠但可定位 -> 1
- 不可见/不可定位 -> 0

## 缓存格式

当前进度缓存通过 `YOLOConverter.build_annotation_states()` 保存每节椎骨的 `points`、`keypoint_visibility`、`shape_type` 等状态。建议扩展 `annotation_states`：

```json
{
  "class_id": 8,
  "class_name": "T8",
  "points": [[...], [...], [...], [...]],
  "keypoint_visibility": 2,
  "shape_type": "obb",
  "pedicles": {
    "image_left": {
      "center": [123.4, 456.7],
      "visibility": 2
    },
    "image_right": {
      "center": null,
      "visibility": 0
    }
  }
}
```

加载逻辑需要在 `_restore_annotation_state()` 和 `_create_annotation_from_state()` 中恢复 `pedicles`。旧缓存没有该字段时按默认空椎弓根标注处理，保持兼容。

## UI 交互设计

### 标注模式

在右侧面板新增“椎弓根标注”区域：

- 模式切换：关闭 / 标图像左椎弓根 / 标图像右椎弓根。
- 可见性选择：清晰可见(2) / 模糊可定位(1) / 不可见(0)。
- 清除按钮：清除当前选中椎骨的左/右椎弓根点。

推荐快捷键：

- `P`：进入/退出椎弓根标注模式。
- `[`：切到图像左椎弓根。
- `]`：切到图像右椎弓根。
- `1` / `2` / `0`：设置当前侧椎弓根可见性。

### 画布行为

1. 选中某一节椎骨后，进入椎弓根标注模式。
2. 点击椎骨框内部，给当前侧写入中心点坐标。
3. 点位以固定屏幕大小圆点显示，不随缩放变大。
4. 图像左/右使用不同颜色，例如：
   - image_left: cyan
   - image_right: magenta
5. 可见性为 `1` 时使用空心或虚线圆；`0` 时不显示点或显示小型不可见标记。
6. 拖拽已有点位可微调坐标。

为了减少误标，建议只允许在选中椎骨的 OBB 多边形内落点；若点击框外，弹出轻提示或忽略。

## 导出格式

不建议把椎弓根塞进现有 `YOLOv8-pose (bbox + 4 关键点)`，因为该格式已经服务于椎骨四角点训练。

也不建议只导出 `cx cy w h` 水平外接框。`cx cy w h` 是 AABB，不包含椎骨旋转角度；如果椎弓根模型训练时看不到椎体 OBB 的 4 个角点，会丢失椎骨倾斜和旋转框几何，影响后续按椎体参考系学习左右椎弓根位置。

建议新增专用导出项：

```text
OBB + pedicle keypoints
```

每行对应一节椎骨：

```text
class_id x1 y1 x2 y2 x3 y3 x4 y4 left_x left_y left_v right_x right_y right_v
```

字段说明：

- `class_id`：继续沿用导出折叠规则，C7-L5 为 `0`，S1 为 `1`。如果 S1 不参与椎弓根训练，可在导出时跳过 S1。
- `x1 y1 x2 y2 x3 y3 x4 y4`：椎骨 OBB 四角点，顺时针排列，归一化到 `[0, 1]`，语义与现有 `YOLOv8-OBB (四角点)` 一致。
- `left_x left_y left_v`：图像左侧椎弓根中心点与可见性。
- `right_x right_y right_v`：图像右侧椎弓根中心点与可见性。
- 当 `v=0` 且没有中心点时，坐标可写 `0 0 0`，训练前也可以过滤不可见点。

该格式每行固定 `1 + 8 + 6 = 15` 个字段。它是本项目的训练中间格式，不是原生 YOLOv8-pose 格式；训练脚本可以在读取时用 OBB 四角点生成单节椎骨 ROI、椎体局部坐标系或外接 AABB，再监督左右椎弓根点位。

如果训练时不需要 S1，建议导出函数提供选项：`include_s1=False`。

## 与上层算法的关系

椎弓根模型只输出图像坐标系下的左右点位和可见性：

```text
image_left_pedicle
image_right_pedicle
image_left_visibility
image_right_visibility
```

上层软件负责：

```text
AP/PA/翻转判断
image_left/image_right -> patient_left/patient_right
右胸弯/左腰弯/顶椎/端椎判断
凸侧/凹侧推导
凸侧椎弓根偏移比例计算
Nash-Moe 0-4 级生成
```

因此该扩展不需要标注凸侧标签，也不需要人工标 Nash-Moe 等级。可在后续抽样验证集中由医师复核算法分级阈值。

## 推荐实现拆分

### 阶段 1：标注与缓存

- 新增 `PediclePoint` / `PedicleAnnotation` 数据结构。
- 扩展 cache 保存和恢复逻辑。
- 在画布上渲染左右椎弓根点。
- 支持点击新增、拖拽微调、清除点位。
- 右侧面板支持左右侧别和可见性编辑。

### 阶段 2：导出

- 新增 `save_pedicle_obb_keypoints()`。
- 导出格式新增到下拉框和 `_on_format_changed()`。
- 单张保存和全部导出都支持 OBB + pedicle keypoints。
- README 补充新格式说明。

### 阶段 3：AI 辅助

- 现有椎骨 AI 预标注继续只生成 OBB / 四角点。
- 当椎弓根模型训练完成后，新增“椎弓根 AI 预标注”入口。
- 椎弓根推理应基于当前椎骨框逐节 crop 或传入整片图像+椎骨框，返回每节的左右点和可见性。

## 验收标准

1. 打开旧数据集和旧缓存不报错，旧标注显示和导出结果不变。
2. 每节椎骨可独立标注图像左/右椎弓根中心点。
3. 左右椎弓根可见性可独立设置并可持久化。
4. 切图、关闭重开后，椎弓根点和可见性完整恢复。
5. 新增 OBB + pedicle keypoints 导出格式，输出字段数量固定为 15。
6. `v=0` 的点允许无中心点，不阻塞保存。
7. 导出的坐标均 clamp 到 `[0, 1]`。
8. 现有 OBB、xywhr、4 角点 pose 导出不受影响。

## 风险与注意事项

- 不要复用 `keypoint_visibility` 表示椎弓根可见性；该字段当前语义是椎骨四角点统一可见性。
- 不要在椎弓根模型里预测凸侧/凹侧，否则会把局部结构识别和上层临床推理混在一起。
- 训练增强若做左右翻转，必须同步交换 image_left/image_right 标签。
- 椎骨 ROI 旋转矫正后训练时，需要保留坐标映射关系，确保导出坐标仍能回到原图。
- 对不可见椎弓根不要强迫标注员猜点，否则会引入系统性噪声。
