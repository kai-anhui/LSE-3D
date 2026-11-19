# LSE-3D：可运行的核心参考实现

本项目以原始《LSE-3D 模型整理与开发设计文档.md》为依据实现，不修改该设计文档。已实现从数据预处理、两级表面单元、文本角色汇聚、双状态选择性 SSM、递推内范围门，到监督、训练、阈值校准、评价和单查询推理的完整链路。

**当前 datasets_raw 三场景与 5090 使用方法见 [三场景操作说明](docs/THREE_SCENES_5090.md)**：新入口 `tools/prepare_dataset.py` 同时支持 TXT/LAS，默认每场景 65,536 点，配置 `configs/lse3d_5090.yaml`。下面原长青 2048 点示例保留作历史小规模调试，三场景训练请使用新说明。

这是 **PyTorch 核心参考实现**。目前没有把随机初始化的参考编码器称为 PTv3 / DINOv2，也没有把小样本调试称为论文实验复现。具体选型、偏差和未完成项见 [实现说明](docs/IMPLEMENTATION.md)，真实数据发现见 [数据检查记录](docs/DATA_AUDIT.md)。

## 项目结构

```text
LSE-3D/
├── configs/
│   ├── base.yaml                 # 通用配置
│   ├── lse3d_exp01.yaml          # 继承 base.yaml 的主实验
│   ├── smoke.yaml               # 小场景过拟合
│   ├── changqing_debug.yaml      # 现有真实数据调试
│   └── ablations/               # YAML 消融配置
├── data/                        # 只放数据集代码
│   ├── __init__.py
│   ├── dataset.py               # PyTorch Dataset
│   ├── transforms.py            # 共享增强
│   ├── build.py                 # build_dataloader
│   ├── geometry.py              # 表面单元与扫描预处理
│   └── projection.py            # 影像投影
├── models/
│   ├── __init__.py
│   ├── backbone/                # 场景、文本编码器及上下文/范围分支
│   ├── heads/                   # 双头点级解码
│   ├── layers.py                # SSM、扫描、池化和角色汇聚
│   ├── losses.py
│   ├── config.py
│   ├── lse3d.py                 # 模型组装
│   └── build.py                 # build_model
├── utils/
│   ├── logger.py
│   ├── checkpoint.py
│   ├── metrics.py
│   ├── seed.py
│   ├── visualize.py             # 导出预测着色 PLY
│   ├── config.py                # YAML 继承/覆盖
│   └── device.py
├── engine/
│   ├── trainer.py
│   └── evaluator.py
├── tools/
│   ├── train.py
│   ├── test.py                  # 测试及验证集阈值校准
│   ├── inference.py             # 无标注推理
│   └── ...                      # 数据导入、投影、检查工具
├── datasets_raw/                # 原始数据位置，Git 忽略
├── outputs/                     # 日志、权重、预处理产物、可视化，Git 忽略
├── tests/
├── docs/
├── requirements.txt
├── pyproject.toml
├── README.md
└── .gitignore
```

现有运行产物已从 `artifacts/` 迁移到 `outputs/`，例如 `outputs/demo/`、`outputs/changqing_sample/` 和 `outputs/overfit/`。`data/` 中只有代码。`datasets_raw/` 保留为空目录；原始点云继续读取原路径，避免复制数 GB 数据。

旧代码结构、JSON 实验配置和旧推理入口归档于 `outputs/legacy_structure/`，供恢复和对照；日常开发使用上述新目录和 YAML 配置。

## 环境

```powershell
python -m pip install -e ".[test]"
python -m pytest -q
```

也可使用 `python -m pip install -r requirements.txt`。主要入口同时支持 `python tools/train.py` 和 `python -m tools.train` 两种调用方式。

已在 Python 3.12.7、PyTorch 2.6.0+cu118、NumPy 2.1.1、SciPy 1.16.1 上验证，支持 CPU；真实数据调试使用 RTX 3060 Laptop 6 GB。项目代码不依赖 Windows 专有路径；只有下面的本机数据导入命令指定了实际位置。

## 快速复现训练、验证、测试

在项目根目录运行：

```powershell
python -m tools.make_demo --output outputs/demo
python -m tools.check_annotations --data outputs/demo
python tools/train.py --data outputs/demo --config configs/smoke.yaml --output outputs/overfit --device cpu --steps 500
python tools/test.py --calibrate --data outputs/demo --checkpoint outputs/overfit/last.pt --output outputs/overfit/calibrated.pt --device cpu
python tools/test.py --data outputs/demo --checkpoint outputs/overfit/calibrated.pt --output outputs/overfit/test_metrics.json --device cpu
```

合成数据包含一个完整域、3 条查询，以及 SUB 和 EQ 对。三个 split 是相同受控几何的独立命名副本，**仅验证工程流程与过拟合，不代表跨地点泛化**。实际运行的 500 步结果：mIoU=1.0、DirIoU=1.0、EQJoint@0.5=1.0。训练末次损失约 0.000143。

测试集入口要求 checkpoint 已包含验证集冻结阈值；不能在测试集调用 calibrate。重复运行建议使用新输出目录，日志追加而 checkpoint 更新。

## 无标注单查询推理

```powershell
python tools/inference.py --checkpoint outputs/overfit/calibrated.pt --scene outputs/demo/scenes/synthetic_test/synthetic_test/full --text "the narrow right region" --output outputs/overfit/prediction.npz
```

模型输入仅需场景目录的 `coord.npy`，可选 `color.npy` 与 `views.pt`；不需要 masks、pairs、父区域或第二条查询。输出 `prob`、`id_logit`、`scope_delta` 和 `mask`。阈值优先使用 `--threshold`，其次使用 checkpoint 验证阈值；均未提供时使用 0.5，并在 JSON 中注明为未校准默认值。原数据导入保留的 `source_index` 会一并保存。JSON 附带查询、阈值和点数。`--intervention ones|shuffle` 可干预范围门；另一查询的范围表示交换由模型 `intervention='swap_scope'` 提供，仅用于诊断。

### 在完整原始点云上显示红色/蓝色结果

对于通过 `import_mine` 导入的长青场景，`--visualize` 默认读取 `source.json` 指向的完整原始 TXT，保留全部点及世界坐标。概率达到阈值的点为纯红色 `(255,0,0)`，其余点为纯蓝色 `(0,0,255)`。输出为二进制 PLY，坐标使用 double，另含 probability 属性。配套 `.ply.json` 记录完整点数、匹配点数、阈值及映射方法。

```powershell
python tools/inference.py --checkpoint outputs/changqing_run01/last.pt --scene outputs/changqing_train/scenes/changqing/changqing/whole_scene_sample --text "采坑底部" --device cuda --output outputs/changqing_run01/prediction.npz --visualize outputs/changqing_run01/prediction_full.ply --threshold 0.5
```

已有预测无需重新训练或推理，可直接重新导出：

```powershell
python tools/visualize.py --scene outputs/changqing_train/scenes/changqing/changqing/whole_scene_sample --prediction outputs/changqing_run01/prediction.npz --output outputs/changqing_run01/prediction_full.ply --threshold 0.5
```

**全点云着色采用最近抽样点的概率回映射，不等同于全分辨率网络推理。** 抽样点的预测按原始索引保留，其余点在三维空间中寻找最近抽样点。2048 点抽样的空间分辨率有限，细边界可能粗糙；导出不使用标注掩码。原始 TXT 路径变化时使用 `--source` 指定新位置。若仅需要抽样点 PLY，推理时添加 `--visualize-scope sampled`；`full` 强制完整导出，原始文件缺失会报错，不静默退回抽样结果。没有 source.json 的其他数据集默认仍导出其输入点。

## 本机长青数据

已生成只读源数据的导入结果：`outputs/changqing_sample/`。其中 2048 点从整个原场景均匀抽样，保留原始点索引，没有按真值目标选点。原始点云和学生文件未修改。

重新导入时使用一个新输出目录：

```powershell
python -m tools.import_mine --source "G:\py\3dvg\标注\data\长青点云彩色.txt" --annotations "G:\py\3dvg\标注\学生标注\长青点云彩色" --core "G:\py\3dvg\标注\code\pointcloud_core.py" --output outputs/changqing_new --max-points 2048
python -m tools.check_annotations --data outputs/changqing_new
python tools/train.py --data outputs/changqing_new --config configs/changqing_debug.yaml --output outputs/changqing_run --allow-unverified --device cuda --steps 3
```

导入复用原标注程序的 `selection_mask`，按顺序执行 replace/add/subtract/intersect；使用原始 float64 世界坐标重放投影后，才减去场景原点存储 float32 坐标。预览 bbox 不用于生成真值。

默认 `--negative-policy unknown`：已选点为已知正例，其余点未知，禁止擅自把未标注区域当作确定背景。`--allow-unverified` 只允许这些数据进入调试，不会生成核验关系。此时损失下降不能证明分割效果，因为没有可靠负例。只有确认每条查询的掩码完整覆盖目标且域外确实为背景后，才适合使用 `--negative-policy assume-complete` 重新导入；该假设写入每条记录。正式数据应人工设置逐查询 valid / ignore。

`--max-points 0` 支持保留全部源点，但导入当前会在内存读取整个 TXT，且参考网络没有面向百万点的融合内核，不建议直接用 6 GB 显卡训练全量点云。抽样漏掉的小目标记录在 import_report.json，不会错误变成空目标查询。只有一个已标注地点，当前只生成 train，val/test 留空。

## 标准数据格式

```text
root/
  scenes/site_id/scene_id/block_id/
    coord.npy          float32 [M,3]，场景局部坐标，单位米
    color.npy          可选 float32 [M,3]，RGB 0..1
    normal.npy         可选 float32 [M,3]
    area_w.npy         float32 [M]，严格为正
    views.pt           可选：images、uv、view_weight
    cache/             按几何内容与配置 hash 缓存
  masks/scene_id/block_id/query_id.npz
  annotations/queries.jsonl
  annotations/pairs.jsonl
  annotations/splits.json
```

mask 文件：`target [M] bool`、`valid [M] bool`、可选 `ignore [M] bool`。实际有效域为 valid & ~ignore。

query 至少包含 query_id/site_id/scene_id/block_id/text/verified。可选 carrier_id、scope_type、parent_mask；parent_mask 指向相对 root 的同格式 NPZ，含独立有效域。分析用 spans 不进入模型。

pair 包含 q_a/q_b/relation/shared_carrier/shared_anchor/verified_by。SUB 固定为 b 是 a 的子集；SUP 为 b 包含 a。至少两个不同核验者才启用成对项。关系从标注记录读取，代码绝不从语义相似度推断关系。`check_annotations` 检查核验关系与共同有效域中的掩码一致性。

splits 示例：`{"train":["site_a"],"val":["site_b"],"test":["site_c"]}`。site_id 必须代表物理地点及其重复航线，不能通过重命名相邻区域绕过防泄漏断言。

## 图像和标定

`python -m tools.build_projection --input cameras.npz --output views.pt` 从显式内参和 world_to_camera 构造投影、深度可见性和 top-4 视角权重。输入格式见模块帮助字符串。图像需同尺寸；深度单位与 xyz 一致。没有观测深度时以源点 z-buffer 近似遮挡，可见性精度取决于点云密度。

views.pt：images `[V,3,H,W]`、uv `[V,M,2]`（grid_sample 归一化坐标）、view_weight `[V,M]`。没有图像的点融合项精确为零，仍使用几何路径。彩色点云 RGB 是点属性，不冒充带位姿影像。

## 代码入口

| 文件 | 功能 |
| --- | --- |
| data/geometry.py | PCA 几何、嵌套单元、Hilbert/Z-order、反向有向边、插值 |
| data/projection.py | 标定投影、遮挡和视角置信度 |
| data/dataset.py、data/transforms.py、data/build.py | 数据加载、地点划分、1:1 查询/查询对采样、共享增强 |
| models/lse3d.py、models/backbone/、models/heads/ | 编码、三角色汇聚、双分支、点级双头 |
| models/layers.py | fp32 精确门控递推与三种执行方式 |
| models/losses.py | 单查询、成对、身份与可靠门边监督，适用计数 |
| utils/metrics.py | 点级/权重 IoU、DirIoU、EQ、FalseChange、BF、地点宏平均 |
| engine/trainer.py、engine/evaluator.py | AdamW、阶段升权、累积、梯度裁剪、checkpoint、校准和测试 |
| tools/ | 导入、检查、演示数据、图像投影和无标签预测 |

## 配置与扩展

`configs/base.yaml` 是 D=256 的通用配置，`configs/lse3d_exp01.yaml` 使用 `_base_: base.yaml` 继承并覆盖主实验参数。`configs/smoke.yaml` 是快速过拟合配置；`configs/changqing_debug.yaml` 用于当前未复核真实数据的数值调试。基配置路径相对于当前 YAML 解析，嵌套字段深度合并，循环继承会报错。

默认训练入口：`python tools/train.py --data <预处理数据目录>`，使用 `configs/lse3d_exp01.yaml`，输出到 `outputs/lse3d_exp01/`。

数据集实现标准 `__len__`/`__getitem__`；`build_dataloader` 保留训练时单查询/成对采样、同场景共享编码和评价时每场景一次的行为。当前明确使用 `num_workers=0`，以保留可恢复的采样随机状态。`build_model` 负责统一从配置构建模型。

推理可添加 `--visualize outputs/prediction.ply` 导出预测着色点云。重构保留了模型 state_dict 键和张量形状，原有 checkpoint 可直接通过新入口加载。

`configs/ablations/` 包含单状态、无递推门、仅范围状态、几何门、无成对损失、无门监督、单方向和双方向配置。它们是机制消融，**未声称参数量已经匹配**。生成脚本为 `python -m tools.make_ablation_configs`。配置键采用 ModelConfig/LossConfig 的平坦 dataclass 字段，非法值会直接报错，不静默退回其他模型。

文本预训练支持 `model.text_model` 指向本地 Hugging Face 编码器目录（可用中文/多语模型），`freeze_text` 控制冻结，需要安装 `.[pretrained]`。默认 UTF-8 字节 Transformer 从零学习，不需要下载，适合接口验证，不具备预训练语义能力。高性能几何、影像骨干可替换 SceneEncoder 子模块，保持输出 `[M,D]` 与可见性公式不变。

checkpoint 保存模型、优化器、配置、随机状态、step、git hash（无 Git 仓库时为 null）。`--resume` 恢复状态；应保持配置、设备和计划总步数不变。训练产生 train.jsonl，记录损失、有效样本数、梯度与门统计。

尚未完成的论文实验部分，包括 Triton/CUDA 融合内核、官方 Mamba 对照数值验证、B1/B2 外部基线适配、定义未定的 B5/B6/B7、公平预算比较、真实跨地点评价，均明确列在实现说明中。
