# 旧版四指标协议

0.3.0 的默认入口已改为严格的 VCC2026 counts 六指标。以下入口仅用于复核原有四指标结果，不与新协议混合比较。安装使用主 README 的 Python >=3.11 独立环境。

## Chreode GEARS 复现

```bash
gears-chreode-eval \
  --pred-path /path/to/GEARS/test_res.pkl \
  --truth-path /path/to/chreode/output/gears/data/ \
  --output-dir /path/to/evaluation/chreode
```

模块方式为 `python -m evaluate.chreode_gears`，三个参数相同。`python evaluate_gears.py` 和 `gears-eval` 已切换为新的 counts 入口。

该旧协议保留 0.2.0 的数值实现：

- PKL 读取 `pred/truth/pert_cat`，NPZ 读取 `pred/truth/conditions/gene_names`。真实扰动均值使用预测产物中的配对测试 `truth`；H5AD 提供真实 `ctrl` 对照。
- 基因集合为真实 GEARS 基因、预测基因与 Chreode 允许词表的交集。默认从预测目录的 `shared_vocab_genes.txt`、`shared_eval/shared_vocab_genes.txt` 或 Chreode 的 `artifacts/gene_vocab.parquet`、`data/phase0/gene_vocab.parquet`、`output/phase0/gene_vocab.parquet` 查找；也可用 `--gene-vocab` 指定原 `--ours-genes` 文件。
- DE20 使用真实绝对效应的原始 `np.argsort(effect)[-20:][::-1]` 顺序，原始表达 pseudobulk MSE，以及带符号效应的 SciPy Pearson，再按扰动等权平均。
- `de20_mse` 对应 Chreode `overall.shared_de_mse`；`de20_delta_r` 对应 `overall.shared_de_delta_pearson`。
- 第三项 `pearson_delta` 为所有绝对效应展开后的 Pearson；第四项 `pds` 为 L1 距离下 `1-2*sum(rank)/T²`，严格更近才计数。这两项是原下游设计的扩展，不对应 Chreode 原表的其他列。

输出为 `gears_evaluation_summary.json/csv`、逐扰动明细与共享基因列表。该入口不训练或推理，也不把归一化值变成 counts。

## 通用旧协议

```bash
perturbation-eval-legacy \
  --pred-path /path/to/predictions \
  --truth-path /path/to/datasets \
  --output-dir /path/to/evaluation/legacy
```

模块方式为 `python -m evaluate.legacy`。H5AD/NPZ 预测、H5AD 独立真值、背景分组、清单、基因交集、对照文件与原有预处理参数继续由旧入口处理；`--help` 查看完整参数。

此通用入口使用独立真值，忽略 NPZ 嵌入的 `truth`；与上述 Chreode 配对真值协议不同。四项输出仍为 `de20_mse/de20_delta_r/pearson_delta/pds`。不能用旧结果文件解释新六项指标。

Chreode 原实现核对版本：`b3cc517276dadb8295776973367d4bfb2bb4894e`。合成数据回归包含 NPZ/PKL、稀疏/稠密及 float32/float64 对照；前两项指标与原评估器的容差为 `1e-12`。这验证同输入下的计算一致性，不保证重新训练后达到文档固定分数。
