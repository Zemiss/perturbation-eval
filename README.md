# GEARS 六指标评估

直接读取已有的 GEARS `test_res.pkl` 和数据目录，计算 [Virtual Cell Challenge 2026](https://virtualcellchallenge.org/) 对应的六项**原始指标**。只需预测、真值数据、输出三个路径，不需要重新训练或手工转换 H5AD。

使用 `cell-eval2==0.18.0` 的指标函数与聚合，针对 GEARS 的 log-normalized 表达做适配。**不是比赛 counts 配置的官方成绩，不做 baseline/replicate 校准，不计算总分。**

## 安装与更新

需要 Python 3.11 或更新版本。建议使用单独的评估环境；GEARS 训练环境继续用于生成预测。

```bash
source ~/.bashrc
conda create -n perturbation-eval python=3.11 pip -y  # 只需首次创建
conda activate perturbation-eval

cd /data/xiebo/evaluate
git pull --ff-only
python -m pip install -e .
```

`source ~/.bashrc` 适用于 Bash；使用 Zsh 时换成 `source ~/.zshrc`。如果已有 `GEARS` 环境的 Python ≥3.11，也可以在该环境安装，但依赖升级可能影响训练环境。

首次克隆时使用 `git clone https://github.com/Zemiss/perturbation-eval.git /data/xiebo/evaluate`；已有仓库只需上述 `git pull`。

## 运行

```bash
gears-eval \
  --pred-path /data/xiebo/evaluate/outputs/GEARS/test_res.pkl \
  --truth-path /data/xiebo/Chreode/output/gears/data/ \
  --output-dir /data/xiebo/evaluate/results/GEARS_run2
```

同样支持 `perturbation-eval`、`python run.py`、`python evaluate_gears.py` 和安装后的 `python -m evaluate`，参数相同。默认固定使用 CPU/pdex，不需要 PyTorch、GEARS 包或 CUDA。

## 输入与计算规则

- `--pred-path`：可信的 GEARS PKL，包含细胞 × 基因的 `pred`、配对测试 `truth` 和每个细胞的 `pert_cat`；也可指定包含 `test_res.pkl` 的目录。只读取自己的可信 PKL。
- `--truth-path`：GEARS 数据根目录、Norman 目录，或明确的 H5AD。目录会查找 `norman/perturb_processed.h5ad` 或 `perturb_processed.h5ad`；发现两份时必须明确指定文件。
- 真实扰动表达使用 PKL 内的测试 `truth`；H5AD 的 `obs.condition == 'ctrl'` 和 `X` 提供真实对照。其他训练、验证或扰动细胞不加入评分。
- 基因名优先使用 H5AD 的 `var.gene_name`，否则使用 `var_names`。通常 PKL 没有基因名，因此假定两侧是同一 GEARS 数据生成的、完整且顺序相同的基因轴；维度不同时报错。若 PKL 有 `gene_names`，按名称校验并对齐。使用全部这些基因，不再读取 Chreode 词表。
- PKL 的预测、测试真值和 H5AD 的 `X` 应使用相同的 GEARS `log1p` 归一化尺度。程序不把表达值伪装成 counts，不取整，也不读取 `raw`/layers 猜测表达尺度。数值检查无法独立证明两个文件采用了相同的预处理。
- 预测负值在 log1p 空间截到 0；截断数量打印并写入 `summary.json`。真实表达负值、NaN/Inf、非法基因轴或标签会报错。
- 保留 `GENE+ctrl`、`GENE1+GENE2` 等完整扰动身份。PDS 排除整个测试面板涉及的靶基因；表达误差和 DE 排除每个扰动自身的全部靶基因。靶基因按符号匹配，大小写不同会对齐；未测量靶基因记录在结果中。
- 以 `vcc2026` 参数为基础：真实对照、余弦距离、midrank、`n-1` 排名分母、Wilcoxon、每扰动 BH-FDR 0.05。沿用原库 CPU lognorm 行为，不应用 counts 专用的 CPM 过滤。
- lognorm 表达/PDS 使用细胞表达均值，表达噪声修正使用原库的二阶矩、预测修正上限和面板预算约束；组合扰动扩展为排除全部组成基因。表达误差最终使用**分子之和／分母之和**，不平均逐扰动比值。

## 输出

只生成 `summary.csv` 和 `summary.json`，六项指标为：

| 指标 | 含义 | 方向 |
| --- | --- | --- |
| `pds_cosine` | 扰动效应余弦检索排名 | ↑ |
| `expr_mse_unbiased_capped_norm` | 经噪声修正、上限约束并按真实效应归一化的表达误差 | ↓ |
| `de_wilcoxon_direction_fidelity_yield_raw` | DE 方向准确性与覆盖 | ↑ |
| `de_wilcoxon_direction_reach_raw` | 指定方向纯度下的覆盖深度 | ↑ |
| `de_wilcoxon_sig_jaccard` | 显著 DE 基因集合 Jaccard | ↑ |
| `de_wilcoxon_lfc_nmae` | log fold-change 归一化绝对误差 | ↓ |

JSON 另含完整配置、实际路径、靶基因、预处理记录和指标状态。原库未定义的指标写为 `null`；真实效应的噪声修正分母之和非正时会报错，不生成伪造结果。旧四指标和旧协议入口已删除。

指标定义见 [cell-eval2 的 VCC2026 文档](https://github.com/ArcInstitute/cell-eval2/tree/main/docs/vcc2026_metrics)。归一化 GEARS 数据与官方 counts 的表达比较空间不同，结果只能在相同数据、预处理和配置下比较。

## 最小验证

仅保留一份测试，检查三参数命令、与原库单靶基因 lognorm 结果的一致性、组合扰动排除及关键非法输入。

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```
