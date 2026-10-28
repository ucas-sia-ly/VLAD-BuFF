# MSLS 复现结果与论文对照

两个模型均已完成 4 epochs、PCA Whitening 和最终 MSLS 评估；两个流水线状态均为 completed。
训练维持每批 60 个地点 × 4 张图片、seed=1。没有为满足内存限制缩小训练 batch。

## 应比较的最终结果

论文 [Table 1](https://arxiv.org/html/2409.19293v1#S5.T1) 的对应行均包含 PCA Whitening。
以下本机结果均使用 322×322 输入、18,871 张参考图、740 个查询。

| 模型 | 本机 R@1 | 论文 R@1 | 本机 R@5 | 论文 R@5 |
|---|---:|---:|---:|---:|
| VLAD-BuFF + PCAW，8192 维 | 92.43% | 92.4% | 96.08% | 95.8% |
| 192PrePool + PCAW，4096 维 | 91.62% | 91.9% | 95.81% | 95.9% |

完整模型 R@1 按论文的一位小数精度显示一致。
192PrePool 的 R@1 差约 0.28 个百分点：本机 678/740 个正确查询；论文一位小数的 91.9% 对应约 680/740 个。
每个查询影响约 0.135 个百分点。

结果原件：

- [完整模型 JSON](../logs/reproduction_20261028_guarded_full/evaluation/wpca8192_last.ckpt_MSLS_metrics.json)
- [192PrePool JSON](../logs/reproduction_20261028_guarded_full_192/evaluation/wpca4096_last.ckpt_MSLS_metrics.json)

## 训练日志为什么显示约 90.5%

Trainroom 关联的是 Lightning 的训练 CSV。
其最后一轮 MSLS 验证 R@1 为完整模型 90.54%、192PrePool 90.41%。
这是 224×224 输入、未使用后置 PCA Whitening 的结果，维度分别为 49,152 和 12,288。
最终评估则使用 322×322 输入和 Whitening，输出 8192 / 4096 维。
因此训练 CSV 中的数字不能直接与论文 Table 1 的两行最终结果比较。
当前 Trainroom 通用记录未把 evaluation 子目录 JSON 展示为独立的最终评估卡片；最终结果以以上 JSON 为准。

训练日志中的 `pitts30k_test` 与论文 Table 1 的 `Pitts250k-t` 也是不同评估划分，不能直接比较。

论文 §4 也明确区分训练 224×224 和评估 322×322。

## 剩余小差异的归因边界

已确认存在的执行差异包括：训练 DataLoader 从原默认 20 个 worker 改为 4 个；Whitening 为满足 RAM 预算使用 float64 对称求解器和磁盘分块运算；本机使用现有 DINOv2 缓存。
worker 数变化可改变增强与采样的随机序列；求解器和软件实现也可能产生数值差异。
这些是需要控制实验核实的候选因素，尚不能断言哪一项造成了 192PrePool 的 2 个查询差距。
此外，本次按 README 使用 `last.ckpt`；论文附录说明按 Pitts30k-val 保存最佳 checkpoint。本机两种模型的 Pitts30k-val 最佳轮次均为第 3 轮，最后 checkpoint 为第 4 轮。论文最终采用的具体 checkpoint 尚未确认，因此 checkpoint 选择也属于待核对项；不能根据最终测试表现反向挑选模型。
没有运行多 seed 对照，也没有在同一评估器中对照作者公开 checkpoint，因此不把差距直接归结为已证实的随机波动。

本文保留初次 MSLS 对照记录。后续已完成两模型九集评估，见[完整结果表](../logs/benchmark_author_release_v2/results.zh-CN.md)及[数据协议与来源条件](dataset-protocols.zh-CN.md)。九集均有真实分数，但 Tokyo、Baidu 保留来源条件；其余分数也未全部达到论文报告值。
