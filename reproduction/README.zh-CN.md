# 本机 README 复现

2026-10-28 新训练的 Trainroom 实验 ID：`66e819b3cb61`。
看板：<http://127.0.0.1:8765>。实时状态以看板、CSV 和 `pipeline.json` 为准。

两个模型现均已完成训练、Whitening 和 MSLS 最终评估，结果见 [与论文对照](results-comparison.zh-CN.md)。
新放入的数据集已完成适配及两模型 × 九数据集复评，见 [数据集协议说明](dataset-protocols.zh-CN.md)和[最终结果表](../logs/benchmark_author_release_v2/results.zh-CN.md)。
用户补充的 MSLS test 公开真值另行接入，见 [MSLS test 说明及结果](msls-test.zh-CN.md)。
训练 CSV 的验证 Recall 与最终评估的 Recall 使用不同分辨率、Whitening 设置，请分别查看。

旧实验 `d75657ba1939` 在 17:55:41 验证阶段因系统 OOM 被杀死，没有产生检查点。
原因是三个验证集的 49,152 维描述子同时积累在 RAM，拼接又产生副本。
旧目录保留供审计；新实验在修复后从头训练。

## 48GiB 显存 / 32GB 系统内存保护

- 每次实验由独立 systemd cgroup 托管：`MemoryMax=10G`、`MemoryHigh=9G`、`MemorySwapMax=0`，包含训练、worker、验证及 Whitening 子进程。该上限是本实验的预算，其他桌面进程不计入其中。
- 启动前至少需要 14GiB 可用 RAM；运行中可用 RAM 低于 4GiB 时保护停止本实验。
- PyTorch 分配器上限 38GiB；整卡用量超过 44GiB 时保护停止。FAISS 属于分配器之外的显存，计入整卡监测。
- 训练为 4 个 worker、每 worker 预取 1 批。验证描述子落盘，分块 GPU 检索，完成后清理临时文件。
- Whitening 使用 float64 分块矩阵运算、磁盘组件和对称特征值求解器，目标仍为同一 PCA；小矩阵的特征值、子空间、白化投影已对比通过，不承诺逐位一致。
- 完整三个验证集的形状（17,608 / 16,816 / 19,611 张，49,152 维）已在硬限制内通过磁盘存储、GPU 检索和清理测试，cgroup 峰值 9.0005GiB。使用合成特征验证容量，不是 benchmark Recall。
- 保护配置及实测数据见 `resource-validation.json`。新实验实时资源记录在 Trainroom 实验目录的 `resource-budget.json`。
- 每 500 个训练 step 保存完整恢复检查点；资源保护可能中断训练，仍以实际检查点文件为准。

## 实验协议

- GSV-Cities：原代码的 23 个城市，62,514 个符合条件的地点、524,701 张图片。
- 完整核对所有城市元数据对应的图片文件，缺失数量为 0，见 `data-audit.json`。这不是逐张图像解码或完整性校验。
- 训练：4 epochs，每批 60 个地点 × 4 图片，224×224，seed=1，FP16 mixed，DINOv2 ViT-B/14 最后 4 个 block，原损失、挖掘、优化器和调度器。
- 共 4,168 个训练 batch；AMP 跳过更新时，不能将 batch 数等同于实际有效更新数。
- 沿用原训练验证集 `pitts30k_val`、`pitts30k_test`、`msls_val`；checkpoint 选择指标仍为 `pitts30k_val/R1`。
- 训练结束后：10,000 个 GSV 地点拟合 Whitening，8192 维；MSLS 按原评估脚本的 322×322、FP16 推理，18,871 张参考图、740 个查询。
- 完整流程成功后，通过 Trainroom 新建独立的 192PrePool 实验：同样训练 4 epochs，Whitening 输出 4096 维，再评估 MSLS。任何阶段失败均停止后续流程。

主模型输出目录：`../logs/reproduction_20261028_guarded_full/`。
第二个模型输出目录：`../logs/reproduction_20261028_guarded_full_192/`。
后续完整 benchmark 使用同一批权重，已评估论文九集；Tokyo、Baidu 的本地图像来源条件另作标注，分数保留真实差距。

## 本机环境与数据

解释器为 `/home/admin123/miniconda3/envs/vladbuff/bin/python`。
已验证 Python 3.10、PyTorch 2.1.0、CUDA 12.1、Torchvision 0.16.0、Xformers 0.0.22.post7、Lightning 2.1.2、FAISS GPU 1.7.2。
`pip check` 通过。完整包清单见 `environment-after.txt` 和 `pip-freeze.txt`。

`env.sh` 配置共享数据及已有 DINOv2 缓存路径，可用同名环境变量覆盖。
原始数据和 DINOv2 缓存代码没有修改。后续 benchmark 在 logs 下生成 JPEG 派生视图，另存官方 Nordland 原包与图片，以及 Baidu 官方 camera 元数据；详见数据协议说明。

环境原先不能导入 PyTorch（`iJIT_NotifyEvent`），Lightning 也缺少 `pkg_resources`。
使用已有离线缓存替换了 MKL 2023.1、Intel OpenMP 2023.0、TBB 2021.8、Setuptools 80.10.2。
原环境清单保存在 `environment-before.txt`；`environment.yml` 补充兼容版本约束，避免重建时再次选择不兼容依赖。

## 接入与运行

本轮实验已完成；下方命令查看历史状态。重新训练请使用新的输出目录：

```bash
python3 /home/admin123/.codex/skills/local-training-terminal/scripts/trainroom.py status --id 66e819b3cb61
```

训练 CSV：
`../logs/reproduction_20261028_guarded_full/lightning_logs/version_0/metrics.csv`。
阶段状态及完整命令：`../logs/reproduction_20261028_guarded_full/pipeline.json`。
标准输出由 Trainroom 保存在：
`/home/admin123/一些设置/终端/train-terminal/.terminal/runs/66e819b3cb61/output.log`。
第二个实验启动后的 ID 保存在主模型输出目录的 `next-run.json`。

另开一次实验时使用新的输出目录：

```bash
python3 /home/admin123/.codex/skills/local-training-terminal/scripts/trainroom.py launch \
  --cwd /home/admin123/github/VLAD-BUFF \
  --name 'VLAD-BuFF reproduction' --total-steps 4168 -- \
  /usr/bin/bash /home/admin123/github/VLAD-BUFF/reproduction/run.sh \
  --variant full --run-dir /home/admin123/github/VLAD-BUFF/logs/NEW_EXPERIMENT \
  --chain-192
```

## 为运行修复的项目问题

- 训练数据、Pittsburgh、MSLS 支持显式环境变量路径；数据检查移入实际使用阶段。
- 评估按所选数据集导入 loader，避免缺少无关数据集阻止 MSLS 评估。
- DINOv2 可读取已存在的本地 Torch Hub 仓库，避免重复联网获取代码。
- Lightning 使用原生 CSV，记录 loss、学习率和按数据集区分的 Recall。完整 checkpoint 包含优化器和调度器状态；`train.py --resume_train PATH` 支持恢复。
- 修复单个验证 loader 时的重复列表嵌套。已用确定性的小型检索数据验证。
- PCA 特征可存入 float64 memmap，保持原算法；小矩阵对比的特征值、均值和投影子空间一致。完成 PCA 后释放特征矩阵引用。
- 评估保存 JSON 指标、查询和参考图数量、分辨率及 checkpoint 路径。

原 PCA 脚本实际输出文件名为 `wpca8192_last.ckpt` / `wpca4096_last.ckpt`，不包含 README 示例中的实验名前缀，串行脚本使用实际路径。
修改后的 Python、Shell 语法检查和 `git diff --check` 已通过。

FAISS 官方说明首次 GPU 调用可能因 CUDA 内核编译变慢：
<https://github.com/facebookresearch/faiss/wiki/Troubleshooting#slow-gpu-intialization>。
首次编译完成后，已实际验证 768 和 49,152 维的 GPU FP16 检索，已知匹配样本的 top-1 正确，复查合计约 1.45 秒。
该检查不能替代整轮验证；最终 Recall 仅以实际运行生成的结果为准。
