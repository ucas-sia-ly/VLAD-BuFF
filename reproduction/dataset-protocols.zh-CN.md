# 本机数据集适配与结果口径

本轮按“优先匹配论文协议”恢复论文/作者代码的评估流程，复用已完成的两份 Whitening 权重。
原始图片、原始 NPY 名单/GT、原先的 MSLS 评估 JSON 均保留。
两模型 × 九数据集共 18 项评估现已完成，没有阻塞项。最新主结果在 `logs/benchmark_author_release_v2/`。
其中七集复用经 checkpoint SHA256、完整 manifest SHA256 和协议指纹核验一致的 v1 结果；Nordland 官方图像及 Baidu 恢复真值后分别重新评估两模型。

- [最终九集结果表](../logs/benchmark_author_release_v2/results.zh-CN.md)
- [逐数据集审计](../logs/benchmark_author_release_v2/dataset-audit.zh-CN.md)
- [机器可读结果](../logs/benchmark_author_release_v2/results.json)
- [已完成的任务状态](../logs/benchmark_author_release_v2/pipeline.json)
- [保留的 v1 本地数据结果](../logs/benchmark_native_paper_v1/results.zh-CN.md)

| 数据集 | 参考图 / 查询图 | 采用的调整 |
|---|---:|---|
| MSLS val | 18,871 / 740 | 恢复新目录路径，沿用作者 qIdx，排除不在协议中的其余 7 张查询 |
| Nordland | 27,592 / 2,760 | 改用原始官方 VPR-Bench 发布包，冬季参考、夏季查询；全部 30,352 张图 SHA256/ZIP CRC 校验，全部 2,760 条 GT 与作者 NPY 一致，不重编码 |
| Pitts250k-test | 83,952 / 8,280 | 使用原始完整 Pittsburgh 目录及作者 GT，不以 Pitts30k-test 替代 |
| SPED | 607 / 607 | 路径适配；原生和作者 GT 一致 |
| SFSM | 27,191 / 1,000 | queries_v1、25m，保留 34 条空正样本查询，Recall 分母仍为 1000 |
| Tokyo247 | 75,984 / 315 | 稳定 panorama/tile ID 映射原始 PNG；只选作者的 315 查询；按官方格式化脚本生成 JPEG，查询短边缩放 480；随后评估统一 322×322，不额外裁剪或融合 |
| StLucia | 1,549 / 1,464 | 路径适配，作者顺序与 25m 规则 |
| AmsterTime | 1,231 / 1,231 | 按 CSV 和作者 pair ID 关联；现代 PNG 按官方流程转 RGB/default JPEG，旧照片保留；GT 为同编号配对，虚拟坐标不是物理位置 |
| Baidu | 689 / 2,292 | 恢复全部 2,981 个官方 camera，按作者原名顺序唯一对应本地图片；用原始未舍入 XYZ 的 10m GT，保留 44 条空正例；排除额外 4,518 / 916 张的另一组；图像来源仍标 conditional |

Tokyo 已按稳定 ID 与本机原始 MAT 核对全部身份集合及 315 条正例集合，一致。
所选查询的原始发布版本与官方 v2 图像字节尚未逐一证明相同，结果保留 conditional 条件；软件版本及格式化配方写入 manifest。

Nordland 本机 JPEG 与官方原版像素不同：抽查本机 JPEG quality 75，官方 quality 95；重新保存 JPEG 不能恢复已损失的细节。
因此另取 1,739,648,996 字节的[官方原包](https://surfdrive.surf.nl/public.php/dav/files/sbZRXzYe3l0v67W/Nordland.zip)，保留本机原图与 v1 结果。
[官方来源证据](../logs/nordland_official_protocol/provenance.json)记录包 SHA256、原图逐文件校验、真值核对和无重编码证明。
换用官方原图后，完整模型 R@1 从本机版 71.74% 变为 72.57%，192PrePool 从 70.00% 变为 70.69%。
论文对应值为 78.0% / 71.4%；官方图像恢复并未消除所有差距，不能将剩余差距归因于 JPEG 编码或修改结果以匹配论文。

Baidu 仅通过 HTTP Range 读取官方 ZIP 的 1,308,797 字节，恢复元数据，不下载整套图片。
所有 camera 均校验 CRC/SHA256；原始 XY 舍入三位能与本机 simple 图片全量唯一匹配，没有缺失、歧义或复用。
这个命名规则来自全量数据验证，尚未找到 Baidu 专用官方格式化脚本，也未证明本机图片与作者原图像素相同，因此仍为 conditional。
恢复后的 GT 含 78,446 条正边；如果改用二维 XY，会改变 160 条查询、增加 1,936 条正边，本轮未采用此近似。
来源与映射证据见 [Baidu provenance](../logs/baidu_protocol_metadata/provenance.json)及 v2 manifest。

论文来源：[数据协议 §10](https://arxiv.org/html/2409.19293v1#S10)、[Table 1](https://arxiv.org/html/2409.19293v1#S5.T1)。
作者代码的 Baidu 实际默认 XYZ 三维 10m，与论文正文概述的 25m 存在冲突，不能无说明地替换。
AmsterTime 论文 R@1 是完整模型 61.4%、192PrePool 59.4%，README 的 61.7% / 59.2% 属另一报告来源，本轮主对照用论文值。

## 运行与限制

Trainroom 最新实验：`fbafc61db1d0`；v1：`1989b7931e1a`，<http://127.0.0.1:8765>。
最新实验已关联 18 项评估的 54 个 R@1/5/10 标量，名称区分模型、数据集和协议条件；[接入核验](../logs/benchmark_author_release_v2/trainroom/verification.json)已与原结果逐项对照。
每个指标只有一个最终快照，横轴 `step=0` 表示快照编号，不表示训练进度。`export_trainroom_snapshot.py` 可将其他已完成 benchmark 的 `results.json` 导出为同样的 JSONL，再用 `trainroom.py link` 接入对应已有任务。
本轮只准备图像视图及重新评估，不重新训练或改变模型选择。
当前可用内存约 13GiB，因此此评估任务使用独立 cgroup **8GiB 硬限制、7GiB 回收阈值、禁用 swap**，保留 4GiB 系统可用内存保护。
GPU 继续使用 PyTorch 38GiB 分配上限及整卡 44GiB 停止阈值。
描述子使用磁盘 memmap，FAISS CPU IndexFlatL2 float32 分块检索；单个样本读取失败会报错，不静默丢弃。
v2 实测整卡显存峰值 **13.45GiB**、任务 cgroup 内存峰值 **7.001GiB**，最低系统可用 RAM **7.32GiB**；[资源记录](../logs/benchmark_author_release_v2/resource-budget.json)确认正常退出。
v1 覆盖其余七集的整套复评显存峰值约 13.67GiB，任务内存峰值约 7.001GiB，见 [v1 资源记录](../logs/benchmark_native_paper_v1/resource-budget.json)。
官方 Nordland 下载与解压另用 512MiB 硬限制，未使用 GPU。

新开一次完整适配和复评，使用新目录：

```bash
python3 /home/admin123/.codex/skills/local-training-terminal/scripts/trainroom.py launch \
  --cwd /home/admin123/github/VLAD-BUFF --name 'VLAD-BuFF datasets' -- \
  /usr/bin/bash /home/admin123/github/VLAD-BUFF/reproduction/benchmark.sh \
  --output-dir /home/admin123/github/VLAD-BUFF/logs/NEW_BENCHMARK
```

恢复本轮失败或中断的任务时使用同一输出目录加 `--resume`，并先确认原任务已退出。
只有 checkpoint SHA256、manifest SHA256 和协议指纹全部一致的既有结果才复用；变化时要求另开目录。
manifest 的 image inventory 包含图像 ID、实际路径、大小、mtime，明确不是图像像素内容哈希。
官方 Nordland 另外核验所有图像的真实 SHA256 和 CRC；Baidu camera 也逐文件核验 SHA256 和 CRC。

适配完成后，`logs/active_dataset_manifests` 和 `logs/active_dataset_views` 指向本轮清单/派生视图。
`source reproduction/env.sh` 使用这些路径；`eval.py --dataset_manifest_dir PATH` 可显式选择某次协议。
历史数据视图和结果目录不会被删除。

MSLS test 现已接入用户补充的公开测试元数据，独立于论文九集主表评估；此前“没有公开真值”的表述已过时。
按官方当前 im2im/all/25m 规则，参考图为 36,558，计分查询为 1,103；从旧 36,565 / 1,135 名单排除 7 张 panorama 及 32 条无正例查询。
完整名单、顺序、11,182 条正例关系均已与固定版本官方 loader 交叉核对，详见 [MSLS test 接入说明](msls-test.zh-CN.md)和[单独结果](../logs/msls_test_public_gt_v1/results.zh-CN.md)。
两模型已完成，R@1 分别为 75.97% / 75.79%，均通过官方 scorer 核验；Trainroom 实验为 `fababc85e76f`。
旧无 GT 清单、历史候选及其他九集结果保留，未直接填入新分数。

## 已做的验证

- 每个评估样本的路径、数量、唯一 ID 和正例索引有效性检查；每个划分首/中/末图像解码检查。
- Tokyo 原始 75,984 PNG 均核对为有效 IHDR、8bit RGB；派生转换的小样本测试包含尺寸、断点复用和损坏缓存修复。
- CPU 回归检查：保留顺序、RGBA→RGB、空 GT 查询进入分母、未加载 GT 时不返回零分、流式特征与内存版本一致、旧分数 provenance 不匹配时拒绝复用。
- 已完成的整套评估实际解码全部所选图片，未出现图像读取失败；这与证明所有本地图像都等同于作者发布字节是两个不同的验证范围。
- 独立终验重算全部 18 个预测 NPZ 的 R@1/5/10，与保存指标完全相同；同时核对所有 ID 顺序、checkpoint/manifest 哈希、14 项复用及 4 项新评估，见 [verification.json](../logs/benchmark_author_release_v2/verification.json)。
