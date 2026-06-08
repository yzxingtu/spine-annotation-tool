# 新增单椎骨 crop 椎弓根中心点 + 可见性标注与导出

## 背景

现有工具已经支持整张脊柱 X 光片的椎骨 OBB / 4 角点 pose 标注，并集成椎骨关键点模型进行 AI 预标注。当前模型已经能输出每节椎骨的 AABB、OBB 和编号，并支撑 Cobb 角与弯型推导。

椎弓根专用模型不应重复识别整张 X 光片中的椎骨组。新的共识是：利用现有脊柱模型先定位每节椎骨，再用每节椎骨的 AABB 裁剪出单椎骨图片，在 crop 图上标注左右椎弓根中心点和可见性，训练一个只处理单椎骨 crop 的专用模型。

## 目标

新增一条非破坏性的单椎骨 crop 标注工作流：

1. 基于现有整图椎骨标注或 AI 推理结果，按每节椎骨 AABB 自动裁剪单椎骨图片。
2. 保存 crop 图片、对应 label、对应 meta，不修改原图和原始整图 label。
3. 在 crop 图上标注：
   - 图像左侧椎弓根中心点
   - 图像右侧椎弓根中心点
   - 图像左侧椎弓根可见性
   - 图像右侧椎弓根可见性
4. 导出单椎骨 crop 标签，用于训练椎弓根专用模型。

## 非目标

- 不修改现有 OBB / xywhr / 4 角点 pose 的导出语义。
- 不在椎弓根模型中识别整张 X 光片上的椎骨组。
- 不要求标注员标注凸侧标签。
- 不要求标注员标注 Nash-Moe 0-4 级。
- 不在椎弓根模型中判断 AP/PA、患者左右或凸侧/凹侧。

## 建议方案

详细设计见 `docs/pedicle-annotation-extension.md`。

核心方案：

1. 新增“生成椎弓根 crop 数据集”工具。
2. 使用现有椎骨 AABB + padding 裁剪，不做旋转矫正。
3. crop 数据集独立保存到新目录，原图和原始 label 只读。
4. 新增单椎骨 crop 标注模式。
5. 每张 crop 图只导出一组左右椎弓根点和可见性。

推荐 crop 标签格式：

```text
class_id left_x left_y left_v right_x right_y right_v
```

说明：

- 每个 label 文件对应一张单椎骨 crop 图片。
- 坐标归一化到 crop 图宽高。
- `class_id` 默认 `0 = vertebra_crop`。
- 当 `v=0` 且没有中心点时，对应坐标可写 `0 0 0`。

## crop 数据集结构

```text
pedicle_crop_dataset/
├── images/
├── labels/
└── meta/
```

或保留原 split：

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

`meta` 需要保存来源原图、椎骨编号、source AABB、加 padding 后的 crop AABB、source OBB、crop 尺寸等信息，用于回溯和将 crop 坐标映射回原图。

## 实现任务

- [ ] 新增基于现有椎骨标注/AI 推理结果的 AABB crop 生成工具。
- [ ] 支持 padding ratio 配置，默认建议 `0.15`。
- [ ] 裁剪结果保存到独立 crop 数据集目录，不覆盖原图和原始 label。
- [ ] 为每张 crop 图生成 `meta/{crop_stem}.json`。
- [ ] 新增 crop 数据集扫描和加载逻辑。
- [ ] 新增单椎骨椎弓根标注数据结构。
- [ ] 新增 crop 图画布点位渲染和编辑交互。
- [ ] 支持标注图像左/右椎弓根中心点。
- [ ] 支持左右可见性独立设置。
- [ ] 支持清除当前侧点位。
- [ ] 新增 crop 椎弓根标签保存与批量导出。
- [ ] README 增补 crop 数据集生成、标注和导出说明。
- [ ] 增加基础单元测试或手工 QA checklist。

## 验收标准

- [ ] 生成 crop 数据集时不修改原图和原始 label。
- [ ] 正常全脊柱每张原图约生成 18-19 张单椎骨 crop。
- [ ] crop 图片、label、meta 能通过文件名一一对应。
- [ ] crop 使用 AABB + padding，超出原图边界时正确 clamp。
- [ ] 每张 crop 图可以独立保存左右椎弓根点和可见性。
- [ ] 切图、关闭重开后，椎弓根标注完整恢复。
- [ ] `v=0` 的椎弓根允许无中心点，不阻塞保存。
- [ ] 新导出格式每行固定 7 个字段。
- [ ] 可通过 meta 将 crop 预测坐标准确映射回原图。
- [ ] 旧的整图 OBB、xywhr、4 角点 pose 导出结果不变。

## 注意事项

- 不要复用现有 `keypoint_visibility` 表示椎弓根可见性；该字段当前属于椎骨四角点。
- 如果训练增强使用水平翻转，必须同步交换 `image_left` 和 `image_right`。
- AP/PA、患者左右、凸侧推导和 Nash-Moe 分级应由上层软件处理，而不是由椎弓根点位模型处理。
- AABB crop 不做旋转矫正，保持原图方向，坐标回原图只需缩放和平移。
