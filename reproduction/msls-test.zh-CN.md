# MSLS test 公开真值接入

官方已开放测试集元数据，使用官方 `evaluate.py` 可以本地评估。
此前“没有公开真值”的说法已过时；本轮按固定版本官方代码补评估，原九集结果和原始 NPY、CSV、图像均保留。

- [本轮结果](../logs/msls_test_public_gt_v1/results.zh-CN.md)
- [任务状态](../logs/msls_test_public_gt_v1/pipeline.json)
- [完整指标和预测核验](../logs/msls_test_public_gt_v1/results.json)
- [元数据来源与 CRC/SHA256](../logs/msls_test_official_protocol/metadata-provenance.json)
- [与官方 loader 全量交叉核验](../logs/msls_test_official_protocol/verification.json)
- [官方说明](https://github.com/mapillary/mapillary_sls#-evaluate-on-the-test-set)

## 真值与计分范围

本机新增包实际名为 `datasets/msls_test/msls_meta.zip`（9,508,674 字节），其内容包含六城市的全部 60 个 CSV；同时支持官方下载名 `test_meta.zip`。
压缩包逐成员 CRC 和 CSV SHA256 已核对；只在 logs 下解压独立副本，未覆盖用户目录中的文件。
`postprocessed.csv` 提供原始 UTM 坐标；每城市独立计算 25m 正例，不跨城市匹配。

任务为 **im2im / all**，选择 `subtask_index.csv` 中 `all=True`，排除 `raw.csv` 中 panorama。
官方评分调用 loader 的 `cities='test', mode='val', posDistThr=25`，以加载新公开真值；`mode='test'` 是原先不加载坐标真值的分支。

| 城市 | 参考图 | 非全景候选查询 | 官方计分查询 | 无 25m 正例而排除 |
|---|---:|---:|---:|---:|
| Miami | 3,761 | 79 | 52 | 27 |
| Athens | 3,355 | 85 | 84 | 1 |
| Buenos Aires | 3,710 | 182 | 181 | 1 |
| Stockholm | 11,965 | 379 | 379 | 0 |
| Bengaluru | 11,731 | 325 | 322 | 3 |
| Kampala | 2,036 | 85 | 85 | 0 |
| 合计 | 36,558 | 1,135 | 1,103 | 32 |

项目原始候选 NPY 为 36,565 / 1,135；其中 7 张 Kampala 参考图属于 panorama，按当前官方规则排除。
32 条无正例查询由官方 `qIdx` 排除，Recall 分母为 **1,103**。这是 MSLS 的官方定义，与 SFSM、Baidu 保留空正例查询的协议不同，未修改其他集的分母。

独立实现的参考图顺序、全部候选查询顺序、`qIdx`、全部 **11,182** 条正例关系和无正例查询名单，与官方 loader 完全一致。
官方代码固定至 `8f950f59515bfbc69727151dcccb19e0e8a01824`，源码逐文件 SHA256 校验。
为兼容本机 NumPy 1.26，仅在单独副本中给 `pIdx` / `nonNegIdx` 的 ragged array 增加 `dtype=object`；未改上游原件或筛选规则。
源码缓存在 `logs/msls_test_official_protocol/source`；缺失时仅下载并校验这几个固定版本的小文本文件。

## 执行与核验

Trainroom 实验：`fababc85e76f`，看板 <http://127.0.0.1:8765>。
两个模型均已成功完成。完整模型 R@1/5/10 为 **75.97% / 88.30% / 91.21%**，192PrePool 为 **75.79% / 88.03% / 90.21%**。
两者预测重算的 Recall 与官方 scorer 完全一致；看板已读取全部 6 个最终指标，每项仅一个快照。
实测整卡显存峰值 **14.10GiB**、任务 cgroup 内存峰值约 **7.00GiB**，最低系统可用 RAM **7.34GiB**，正常退出；见[资源记录](../logs/msls_test_public_gt_v1/resource-budget.json)和[最终核验](../logs/msls_test_public_gt_v1/verification.json)。
复用此前两个训练完成、Whitening 后的 last checkpoint，输入 322×322、FP16 autocast，每图一个描述子；CPU float32 IndexFlatL2 精确检索。
独立 cgroup 使用 8GiB RAM 硬上限、7GiB 回收阈值、禁止 swap；PyTorch 分配器 38GiB 上限、整卡 44GiB 停止阈值，保留至少 4GiB 系统可用 RAM。

每份结果完成后检查预测 NPZ 的 ID、顺序、形状、索引和距离，重算 R@1/5/10，再与固定版本官方评分函数逐项比较。
`full/verification.json` 和 `192prepool/verification.json` 保存核验、预测 SHA256、checkpoint 和 manifest SHA256；官方格式 top-10 文本也单独保留。
两个模型均成功后才发布新的 active 清单；其中其他九集清单逐字节复制，历史清单保持原样。
本轮已发布 `logs/msls_test_public_gt_v1/combined_manifests`；除了新 MSLS test，其他 11 份既有清单（九集及两个 Pitts30k 划分）均与此前逐字节一致。

另开一次补评估时使用新输出目录：

```bash
python3 /home/admin123/.codex/skills/local-training-terminal/scripts/trainroom.py launch \
  --cwd /home/admin123/github/VLAD-BUFF --name 'VLAD-BuFF MSLS test' -- \
  /usr/bin/bash /home/admin123/github/VLAD-BUFF/reproduction/msls_test_benchmark.sh \
  --output-dir /home/admin123/github/VLAD-BUFF/logs/NEW_MSLS_TEST
```

中断后确认原任务已退出，再用原目录加 `--resume`。已有结果仅在 checkpoint、manifest 和协议指纹全部一致时复用，并重新核验预测与官方 scorer。
本评估属于附加 MSLS test，论文主表的 MSLS val 保持单独报告，不以 test 分数选择 checkpoint 或调整训练配置。
