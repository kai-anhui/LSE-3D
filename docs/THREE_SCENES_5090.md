# 三场景数据处理与 5090 训练

在项目根目录执行以下命令。新入口为 `tools/prepare_dataset.py`，不要继续用只针对长青 TXT 的旧 `tools/import_mine.py` 处理这三个场景。

## 数据映射

`configs/data/three_sites.yaml` 已对应本机文件名（目录中的不同写法按实际路径保留）：

| ID | 点云 | 标注目录 | 原始点数 | 默认用途 |
| --- | --- | --- | ---: | --- |
| changqing | point_data/长青点云彩色.txt | json/常青 | 3,778,599 | 验证 |
| baijialiang | point_data/白家梁.las | json/白家粱 | 93,181,170 | 训练 |
| hongjingta | point_data/弘景塔点云总.las | json/宏景塔 | 334,900,171 | 测试 |

默认仅一个矿区训练是三地点留出实验起点，后续可交换训练/验证/测试地点，分别报告结果。不能把同一矿区的改写句随机分到不同集合来冒充跨地点泛化。

## 环境

5090 应使用支持 Blackwell 的 PyTorch CUDA 构建。PyTorch 官方 2.7 发布说明记录了 Blackwell 与 CUDA 12.8 支持：https://pytorch.org/blog/pytorch-2-7/ 。不要直接照搬此前 3060 使用的 torch 2.6.0+cu118 环境。

```powershell
conda create -n lse3d5090 python=3.12 -y
conda activate lse3d5090
python -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name(0)); x=torch.randn(1024,1024,device='cuda'); print((x@x).mean().item())"
```

这是目标机器上的安装指引，本机没有 5090，未实际验证其驱动与峰值显存。已有兼容环境时直接激活该环境，无需重复创建。

## 处理数据

每场景取 **65,536 点**，是之前 2048 的 32 倍；查询共享同一组场景点，保留全域空间范围、原始索引及坐标原点。LAS 通过内存映射读取所选记录，不把几亿个 XYZ 一次展开为内存数组。TXT 当前整体读取后抽样。

以下命令的 `assume-complete` 明确假设：每条描述的选区完整覆盖其目标，选区外已观测点可作为背景。只有确认这一点后才使用。否则选择 `unknown`，此时只有正例，适合流程检查，不能作为完整分割监督。

```powershell
python tools/prepare_dataset.py --manifest configs/data/three_sites.yaml --raw-root datasets_raw --output outputs/three_sites_65536 --max-points 65536 --negative-policy assume-complete
python -m tools.check_annotations --data outputs/three_sites_65536
```

输出目录必须为空或不存在；需重新处理时使用新目录。会生成 scenes、masks、annotations，以及 `prepare_report.json`。报告记录每个版本、缺失描述、抽样漏掉的目标和同场景重复描述。没有自动生成范围查询对，也没有把记录自动标记为人工核验通过。默认保留不同 JSON 的来源；如果它们是修订版本而非独立目标，需要先选择采用的版本，不要同时训练互相冲突的标注。

若目标是把三个矿区全部用于拟合/应用，不做独立地点测试：在上面的转换命令末尾添加 `--split-mode all-train`，并换输出目录。随后 train 包含三个地点，val/test 留空，不能执行阈值校准或独立测试命令。

65,536 仍是全场随机采样，不是全点云训练，也不是已实现的滑窗分块训练。尤其弘景塔原始点数很大，仍可能漏掉很小的目标，需查看报告。不要直接改成 0 将数亿点送进模型。以后增加到 131,072 时需重新转换，并观察遗漏率和显存；不能保证仅增加点数就改善语义歧义或监督质量。

## 训练

配置 `configs/lse3d_5090.yaml` 使用 D=64、hidden=64、state=8、四方向双尺度、BF16、parallel 扫描、10 米体素。范围递推公式未改；这是资源起始配置，未宣称与 D=256 的参考容量相同。没有核验关系对，成对/身份一致性权重关闭，门辅助权重为 0.02。

先做 10 步显存和数值检查：

```powershell
python tools/train.py --config configs/lse3d_5090.yaml --data outputs/three_sites_65536 --output outputs/5090_memory_check --device cuda --allow-unverified --steps 10
```

日志的 `peak_cuda_gb` 为 PyTorch 峰值已分配显存，并非驱动总占用；可结合 nvidia-smi。观察损失/梯度是否有限。若 OOM，先将配置 hidden 改为 32、state 改为 4；必要时增大 voxel_size 降低单元数。两种调整都要记录，不能把 32 GB 当作任意规模的运行保证。

检查完成后，从头正式训练（不要把 10 步短调度 checkpoint 当作 10000 步训练恢复）：

```powershell
python tools/train.py --config configs/lse3d_5090.yaml --data outputs/three_sites_65536 --output outputs/lse3d_5090_run01 --device cuda --allow-unverified
```

配置默认 10,000 次优化更新，不是 epoch。长训练中断时加 `--resume outputs/lse3d_5090_run01/last.pt`，保持配置和总步数一致。`--allow-unverified` 是调试/探索性训练的显式开关，不表示标注已审定。

## 校准和测试（仅 holdout 划分）

```powershell
python tools/test.py --calibrate --data outputs/three_sites_65536 --checkpoint outputs/lse3d_5090_run01/last.pt --output outputs/lse3d_5090_run01/calibrated.pt --device cuda --allow-unverified
python tools/test.py --data outputs/three_sites_65536 --checkpoint outputs/lse3d_5090_run01/calibrated.pt --output outputs/lse3d_5090_run01/test_metrics.json --device cuda --allow-unverified
```

验证只使用长青来确定阈值，测试用弘景塔。未核验标注上的分数仅作探索性检查，非正式论文结论。评价按单查询依次执行，共享场景编码，控制大点数的显存。当前 bootstrap 置信区间不能弥补每个 split 只有一个物理地点的限制。

## 推理和完整点云可视化

```powershell
python tools/inference.py --checkpoint outputs/lse3d_5090_run01/calibrated.pt --scene outputs/three_sites_65536/scenes/hongjingta/hongjingta/whole_scene_sample --text "替换为该场景中可唯一定位目标的描述" --device cuda --output outputs/lse3d_5090_run01/hongjingta_prediction.npz --visualize outputs/lse3d_5090_run01/hongjingta_full.ply
```

all-train 模式改用 last.pt，并以 `--threshold 0.5` 指定未校准的展示阈值。完整 PLY 是原始全部点的红/蓝着色，未采样点采用最近采样点的概率，不等于全分辨率网络推理。

LAS 全量导出按块计算和写盘，仍需读取整场景；弘景塔 PLY 约 10.4 GB，白家梁约 2.9 GB，查看软件和磁盘需要能够承受。只想先检查预测可加 `--visualize-scope sampled`，无需导出全场。预处理产物迁移机器后，source.json 中原始文件路径可能失效，可用推理的 `--source` 指向该机器的原始 LAS/TXT。

新版导出会在旧路径失效时，自动尝试场景目录上级及当前工作目录的 `datasets_raw/point_data/原文件名`，支持识别 Windows 路径中的文件名。找到多个候选时要求显式指定 `--source`，不会随意选取；导出仍核对原始点数和采样坐标。原始点云没有复制到目标机器时，自动查找也无法恢复文件，需复制原始 TXT/LAS 或只导出 sampled 模式。

如果推理已经完成，但全场导出因旧路径报错，`prediction.npz` 已在导出之前保存，可直接重导出，无需重新训练或推理：

```bash
python tools/visualize.py --scene outputs/three_sites_65536/scenes/changqing/changqing/whole_scene_sample --prediction outputs/lse3d_5090_run01/prediction.npz --source datasets_raw/point_data/长青点云彩色.txt --output outputs/lse3d_5090_run01/prediction_full.ply --threshold 0.5
```

0.5 为未校准的展示阈值；如原预测使用其他阈值，请传入相同值。文件后缀使用 `.ply`，不是 `.pl`。

## 本次验证范围

已核对三个真实场景的文件映射、LAS header 点数及标注 schema；在 4096 点检查规模跑通三个场景的导入。新增 LAS 缩放/偏移、索引保留、分块 PLY 导出测试。正式 65,536 点数据和 5090 训练由用户按上述命令运行，未在当前 6 GB 设备上声称通过该规模训练。

## BF16 门损失修复

若旧版本在 `models/losses.py` 报 `binary_cross_entropy ... unsafe to autocast`，同步最新的 `models/losses.py`。范围分支原本已经在 FP32 下计算门概率，但其 BCE 曾在训练器外层 autocast 中执行，CUDA 会拒绝此操作。现已在门辅助 BCE 局部关闭 autocast，并显式使用 float32，保留 BF16 训练和原门损失定义。

无需重新处理数据、关闭门监督或更改模型参数。`gate_raw` 已是 sigmoid 概率，不能将它原样传给 BCEWithLogitsLoss。新增 `tests/test_amp_losses.py` 在 CPU/CUDA autocast 下验证损失和梯度，并在 CUDA BF16 下执行带可靠门边监督的模型优化器更新；本机已通过这三项测试。目标 5090 的全规模训练仍需重新执行显存检查命令。
